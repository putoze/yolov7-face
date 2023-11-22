import argparse
import time
from pathlib import Path

import os
import copy
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import plot_one_box, show_fps, plot_kpts
from utils.torch_utils import select_device, load_classifier, time_synchronized

# self add
from post_alg import fitEllipse, headpose_alg, alg_6DRepNet, icon_alg, gaze_estimate

## 6D RepNet 
from RepNet_6D.model_6DRepNet import SixDRepNet
import RepNet_6D.utils_with_6D as utils_with_6D
import RepNet_6D.datasets as datasets
from torchvision import transforms


# YOLO Trt
import pycuda.autoinit  # This is needed for initializing CUDA driver
from utils_ten.yolo_classes import get_cls_dict
from utils_ten.visualization import BBoxVisualization
from utils_ten.yolo_with_plugins import TrtYOLO


window_name = 'YOLOV7-face'

def detect(opt):
    source, weights, view_img, save_txt, imgsz, save_txt_tidl, kpt_label = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.save_txt_tidl, opt.kpt_label
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Directories
    save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run
    (save_dir / 'labels' if (save_txt or save_txt_tidl) else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu' and not save_txt_tidl  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    
    
    if isinstance(imgsz, (list,tuple)):
        assert len(imgsz) ==2; "height and width of image has to be specified"
        imgsz[0] = check_img_size(imgsz[0], s=stride)
        imgsz[1] = check_img_size(imgsz[1], s=stride)
    else:
        imgsz = check_img_size(imgsz, s=stride)  # check img_size
    
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    # colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    colors = [[255, 0, 0],   # 紅色
    [0, 255, 0],   # 綠色
    [0, 0, 255],   # 藍色
    [255, 255, 0], # 黃色
    [255, 0, 255], # 品紅
    [0, 255, 255], # 青色
    [255, 165, 0]  # 橙色
]

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once


    # 6D_Repnet
    snapshot_path_6D = '../../weights/6DRepNet/6DRepNet_300W_LP_AFLW2000.pth'
    model_6DRepNet = SixDRepNet(backbone_name='RepVGG-B1g2',
                       backbone_file='',
                       deploy=True,
                       pretrained=False)

    saved_state_dict = torch.load(os.path.join(
        snapshot_path_6D), map_location='cpu')

    if 'model_state_dict' in saved_state_dict:
        model_6DRepNet.load_state_dict(saved_state_dict['model_state_dict'])
    else:
        model_6DRepNet.load_state_dict(saved_state_dict)
    model_6DRepNet.to(device)


    print('Loading data.')
    

    # transformations_6D
    transformations_6D = transforms.Compose([transforms.Resize(224),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    data_dir = '../datasets/AFLW2000'
    filename_list = '../datasets/AFLW2000/files.txt'
    dataset = 'AFLW2000'
    pose_dataset = datasets.getDataset(
        dataset, data_dir, filename_list, transformations_6D, train_mode = False)
    test_loader = torch.utils.data.DataLoader(
        dataset=pose_dataset,
        batch_size=32,
        num_workers=2)

    # End 6D_Repnet

    # YOLO Trt
    category_num = 9
    yolo_conf_th = 0.6
    letter_box = True
    TrtYOLO_model = '../../weights/darknet/yolov4-tiny-20231005/yolov4-tiny-custom'
    cls_dict = get_cls_dict(category_num)
    vis = BBoxVisualization(cls_dict)
    trt_yolo = TrtYOLO(TrtYOLO_model, category_num, letter_box)
    # End YOLO Trt

    # number of class
    num_cs = len(names)
    print('number of class:',num_cs)
    print(names)
    
    total = 0
    yaw_error = pitch_error = roll_error = .0
    v1_err = v2_err = v3_err = .0


    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        # print(pred[...,4].max())
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms, kpt_label=kpt_label)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            steps = 3
            coordinate = [(0,0) for kid in range(kpt_label)]

            if len(det):
                # Rescale boxes from img_size to im0 size
                scale_coords(img.shape[2:], det[:, :4], im0.shape, kpt_label=False)
                scale_coords(img.shape[2:], det[:, 6:], im0.shape, kpt_label=kpt_label, step=3)

                # Print results
                for c in det[:, 5].unique():
                    n = (reversed(det[:, 5]) == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                for det_index, (*xyxy, conf, cls) in enumerate((det[:,:6])):
                    if names[int(cls)] == 'face':
                        kpts = det[det_index, 6:]
                        for kid in range(kpt_label):
                            x_coord, y_coord = kpts[steps * kid], kpts[steps * kid + 1]
                            if not (x_coord % 640 == 0 or y_coord % 640 == 0):
                                coordinate[kid] = (int(x_coord),int(y_coord))
                        im0,yaw,pitch,roll = headpose_alg(im0,kpt_label,coordinate,coordinate[12],1)
                        #im0,yaw,pitch,roll = gaze_estimate(im0, coordinate, nose_point, pupil_left, pupil_right)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', nargs= '+', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-txt-tidl', action='store_true', help='save results to *.txt in tidl format')
    parser.add_argument('--save-bin', action='store_true', help='save base n/w outputs in raw bin format')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--kpt-label', type=int, default=5, help='number of keypoints')
    opt = parser.parse_args()
    print(opt)
    check_requirements(exclude=('tensorboard', 'pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect(opt=opt)
                strip_optimizer(opt.weights)
        else:
            detect(opt=opt)
