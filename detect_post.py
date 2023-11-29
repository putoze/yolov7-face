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
from post_alg import fitEllipse, headpose_alg, alg_6DRepNet, icon_alg, gaze_estimate, gaze_GazeML,alert #gaze_laser
from GazeML.gaze import draw_gaze
# from GazeML.gaze_laser import *

## 6D RepNet 
from face_detection import RetinaFace
from RepNet_6D.model_6DRepNet import SixDRepNet

# YOLO Trt
import pycuda.autoinit  # This is needed for initializing CUDA driver
from utils_ten.yolo_classes import get_cls_dict
from utils_ten.visualization import BBoxVisualization
from utils_ten.yolo_with_plugins import TrtYOLO


window_name = 'YOLOV7-face'

def detect(opt):
    source, weights, view_img, save_txt, imgsz, save_txt_tidl, kpt_label = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.save_txt_tidl, opt.kpt_label
    save_img = not opt.nosave  # save inference images and not source.endswith('.txt')
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
    detector = RetinaFace(gpu_id=0)

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

    # ------ self add parameters ------

    # times
    frame_cnt = 0
    frame_face = 0
    frame_6D = 0
    inference_time_nms = 0
    inference_time_6D = 0
    fps = 0

    # seatbelt
    seatbelt_cnt = 0
    max_seatbelt_cnt = 20
    # phone
    phone_cnt = 0
    max_phone_cnt = 20
    # smoke
    smoke_cnt = 0
    max_smoke_cnt = 20

    # Text show
    gap_txt_height = 35
    show_text = 1

    # drowsy
    max_drowsy_cnt = 10
    yawn_cnt = 0

    # Attentive
    yaw_boundary = 30
    pitch_boundary = 20
    roll_boundary = 10
    attentive_flag = 1
    not_attentive_cnt = 0
    max_not_attentive_cnt = 5

    # flag
    icon_flag = [0,0,0,0,0] # drowsy, not attentive, seatbelt, phone, smoke

    # landmark coordinate 
    coordinate = [(0,0) for kid in range(kpt_label)]

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        if webcam:  # batch_size >= 1
            p, s, im0, frame = path[0], '%g: ' % 0, im0s[0].copy(), dataset.count
        else:
            p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

         # local parameter 
        next_txt_height = 35
        face_max = 0
        driver_face_roi = []
        steps = 3
        driver_kpts = []
        pupil = []
        pupil_left = []
        pupil_right = []
        glasses_flag = 0
        post_flag = [0,0,0] # headpose, alert, icon

        if icon_flag[1] == 1:
            mode1_t0 = time.time()
            # t1_yolo = time_synchronized() 
            # boxes, confs, clss = trt_yolo.detect(im0, yolo_conf_th)
            # t2_yolo = time_synchronized() 
            # # Draw yolo ten
            # im0 = vis.draw_bboxes(im0, boxes, confs, clss)  

            # for bb, cf, cl in zip(boxes, confs, clss): 
            #     s += f"{n} {cls_dict[cl]}{'s' * (n > 1)}, "  # add to string
            #     if cls_dict[cl] == 'seatbelt':
            #         seatbelt_cnt += 1
            #     elif cls_dict[cl] == 'phone':
            #         phone_cnt += 1
            #     elif cls_dict[cl] == 'smoke':
            #         smoke_cnt += 1
            #     elif cls_dict[cl] == 'face':
            #         bb = [int(b) for b in bb]
            #         face_area = (bb[2] - bb[0])*(bb[3] - bb[1])
            #         if face_area > face_max :
            #             face_max = face_area
            #             driver_face_roi = bb

            faces = detector(im0)
            for box, landmarks, score in faces:
                if score < .95:
                    continue
                bb = [int(b) for b in box]
                face_area = (bb[2] - bb[0])*(bb[3] - bb[1])
                if face_area > face_max :
                    face_max = face_area
                    driver_face_roi = bb

            if len(driver_face_roi) != 0:
                # headpose 
                post_flag[0] = 1

                # find nose position
                nose_point = (int((driver_face_roi[0]+driver_face_roi[2])/2),
                        int((driver_face_roi[1]+driver_face_roi[3])/2))
                
                # 6D RepNet
                im0,yaw,pitch,roll = alg_6DRepNet(im0,driver_face_roi,model_6DRepNet,device,nose_point)

                mode1_t1 = time.time()
                frame_6D += 1
                # Print time inference
                print(f'{s}6D_Done. ({(1E3 * (mode1_t1 - mode1_t0)):.1f}ms) Inference')

                inference_time_6D += mode1_t1 - mode1_t0
            else :
                mode1_t1 = time.time()

            # update fps
            fps = 1.0 / (mode1_t1 - mode1_t0)
        
        if icon_flag[1] == 0:
            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=opt.augment)[0]
            # print(pred[...,4].max())
            t2 = time_synchronized()

            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms, kpt_label=kpt_label)
            t3 = time_synchronized()

            det = pred[0]

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            #s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            
            if len(det):
                # Rescale boxes from img_size to im0 size
                scale_coords(img.shape[2:], det[:, :4], im0.shape, kpt_label=False)
                scale_coords(img.shape[2:], det[:, 6:], im0.shape, kpt_label=kpt_label, step=3)

                # Print results
                for c in det[:, 5].unique():
                    n = (reversed(det[:, 5]) == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # find driver face
                for det_index, (*xyxy, conf, cls) in enumerate((det[:,:6])):
                    if names[int(cls)] == 'face':
                        kpts = det[det_index, 6:]
                        bb = [int(x) for x in xyxy]
                        face_area = (bb[2] - bb[0])*(bb[3] - bb[1])
                        if face_area > face_max :
                            face_max = face_area
                            driver_face_roi = bb
                            driver_kpts = kpts
                    elif names[int(cls)] == 'seatbelt' :
                        seatbelt_cnt += 1
                    elif names[int(cls)] == 'phone' :
                        phone_cnt += 1
                    elif names[int(cls)] == 'smoke' :
                        smoke_cnt += 1
                    elif names[int(cls)] == 'pupil' :
                        bb = [int(x) for x in xyxy]
                        pupil.append(bb)
                    elif names[int(cls)] == 'glasses' :
                        glasses_flag = 1

                # ====== Object Alert ======
                if frame_cnt % max_seatbelt_cnt == 0:
                    if seatbelt_cnt >= max_seatbelt_cnt * 0.5:
                        icon_flag[2] = 1
                    else:
                        icon_flag[2] = 0

                    if phone_cnt >= max_phone_cnt * 0.5:
                        icon_flag[3] = 1
                    else:
                        icon_flag[3] = 0

                    if smoke_cnt >= max_smoke_cnt * 0.5:
                        icon_flag[4] = 1
                    else:
                        icon_flag[4] = 0
                    
                    seatbelt_cnt = 0
                    phone_cnt = 0
                    smoke_cnt = 0

                if len(driver_face_roi) != 0 and len(driver_kpts) != 0:
                    post_flag = [1,1,1] # headpose, alert, icon

                    # landmark points
                    for kid in range(kpt_label):
                        x_coord, y_coord = driver_kpts[steps * kid], driver_kpts[steps * kid + 1]
                        if not (x_coord % 640 == 0 or y_coord % 640 == 0):
                            coordinate[kid] = (int(x_coord),int(y_coord))

                    # draw coordinate
                    plot_kpts(im0,coordinate)

                    # create coordinate np array
                    coordinate_np = np.array(coordinate)

                    # headpose
                    if post_flag[0]:
                        im0,yaw,pitch,roll = headpose_alg(im0,coordinate)
                    
                    # find pupil roi
                    flag_list = [1,1,1,1,1,1,1]
                    for bb in pupil:
                        x_min,y_min,x_max,y_max = bb
                        center = (int((bb[0]+bb[2])/2),int((bb[3]+bb[1])/2))
                        if center[0] < coordinate[12][0] :
                            if center[0] < coordinate[3][0] and center[0] > coordinate[0][0] \
                        and center[1] > coordinate[1][1] and center[1] < coordinate[5][1] :
                                # pupil
                                size = (int(x_max - x_min),int(y_max - y_min))
                                pupil_left = [center,size,roll]
                                # if glasses_flag:
                                #     size = (int(x_max - x_min),int(y_max - y_min))
                                #     pupil_left = [center,size,0.0]
                                # else :
                                #     pupil_imgL = im0[y_min:y_max,x_min:x_max,:]
                                #     pupil_imgL_th = fitEllipse(pupil_imgL,flag_list,bb)
                                #     if pupil_imgL_th != None:
                                #         pupil_left = pupil_imgL_th
                        else:
                            if center[0] < coordinate[9][0] and center[0] > coordinate[6][0] \
                        and center[1] > coordinate[7][1] and center[1] < coordinate[11][1] :
                                # pupil
                                size = (int(x_max - x_min),int(y_max - y_min))
                                pupil_right = [center,size,roll]
                                # if glasses_flag:
                                #     size = (int(x_max - x_min),int(y_max - y_min))
                                #     pupil_right = [center,size,0.0]
                                # else:
                                #     pupil_imgR = im0[y_min:y_max,x_min:x_max,:]
                                #     pupil_imgR_th = fitEllipse(pupil_imgR,flag_list,bb)
                                #     if pupil_imgR_th != None:
                                #         pupil_right = pupil_imgR_th


                    # gaze                    
                    if len(pupil_left) != 0:
                        gaze_left = gaze_GazeML(im0, coordinate_np[0:6], pupil_left)
                        draw_gaze(im0,pupil_left[0],gaze_left[0],length=200.0)
                    if len(pupil_right) != 0:
                        gaze_right = gaze_GazeML(im0, coordinate_np[6:12], pupil_right)
                        draw_gaze(im0,pupil_right[0],gaze_right[0],length=200.0)

                    # pupil_laser_left,eye_center_left   = gaze_laser(im0, coordinate[0:6])
                    # pupil_laser_right,eye_center_right = gaze_laser(im0, coordinate[6:12])
                    # pupils_laser = np.array([pupil_laser_left, pupil_laser_right])

                    # poi = [coordinate[0], coordinate[3]], [coordinate[6], coordinate[9]],  \
                    #     pupils_laser, [eye_center_left,eye_center_right]
                    # theta, pha, delta = calculate_3d_gaze(frame, poi)
                    
                    # im0,yaw,pitch,roll = gaze_estimate(im0, coordinate, nose_point, pupil_left, pupil_right)
                    
                    # drowsy Alert 
                    if post_flag[1] == 1:
                        im0,yawn_flag = alert(im0,coordinate_np,glasses_flag)
                        if yawn_flag:
                            yawn_cnt += 1
                        else :
                            yawn_cnt = 0


                    # drowsy
                    if yawn_cnt > max_drowsy_cnt:
                        icon_flag[0] = 1
                    else :
                        icon_flag[0] = 0

                # Write results
                for det_index, (*xyxy, conf, cls) in enumerate(reversed(det[:,:6])):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or opt.save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if opt.hide_labels else (names[c] if opt.hide_conf else f'{names[c]} {conf:.2f}')
                        if names[c] != 'pupil' :
                            plot_one_box(xyxy, im0, label=label, color=colors[c], line_thickness=opt.line_thickness)
                        if opt.save_crop:
                            save_one_box(xyxy, im0s, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

                if save_txt_tidl:  # Write to file in tidl dump format
                    for *xyxy, conf, cls in det_tidl:
                        xyxy = torch.tensor(xyxy).view(-1).tolist()
                        line = (conf, cls,  *xyxy) if opt.save_conf else (cls, *xyxy)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
            # update fps
            fps = 1.0 / (t3 - t1)

            # inference times
            if frame_face != 0:
                inference_time_nms += t3 - t1

            frame_face += 1

            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

        # Attentive
        if abs(yaw) > yaw_boundary or abs(roll) > roll_boundary or abs(pitch) > pitch_boundary:
            attentive_flag = 0
        else:
            attentive_flag = 1

        if attentive_flag == 0:
            not_attentive_cnt += 1
        else:
            not_attentive_cnt = 0

        if not_attentive_cnt > max_not_attentive_cnt :
            icon_flag[1] = 1
            yawn_cnt = 0
            icon_flag[0] = 0
        else:
            icon_flag[1] = 0
        
        # icon
        if post_flag[2]:
            im0 = icon_alg(im0,icon_flag)

        if show_text and post_flag[0]:
            pitch_str = str(round(pitch.item(), 3))
            yaw_str = str(-(round(yaw.item(), 3)))
            roll_str = str(round(roll.item(), 3))
            if icon_flag[1] == 0:
                #(img, text, org, fontFace, fontScale, color, thickness, lineType)
                cv2.putText(im0,"HEAD-POSE PNP",(0,next_txt_height), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else :
                cv2.putText(im0,"HEAD-POSE 6D",(0,next_txt_height), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            next_txt_height += gap_txt_height
            cv2.putText(im0,"roll:"+roll_str,(0,next_txt_height), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            next_txt_height += gap_txt_height
            cv2.putText(im0,"yaw:"+yaw_str,(0,next_txt_height), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            next_txt_height += gap_txt_height
            cv2.putText(im0,"pitch:"+pitch_str,(0,next_txt_height), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            next_txt_height += gap_txt_height


        # update frame_cnt
        frame_cnt += 1

        # Stream results
        if view_img:
            im0 = show_fps(im0, fps)
            cv2.imshow(window_name, im0)
            key = cv2.waitKey(1)
            if key == 27 :  # ESC key: quit program  # or frame_cnt == 500
                print("")
                print("-------------------------------")
                print("------ See You Next Time ------")
                print("-------------------------------")
                print("")
                cv2.destroyAllWindows()

                total_cal = time.time() - t0
                print('frame_cnt', frame_cnt)
                print(f'Done. ({total_cal:.3f}s)')
                print(f'Average FPS : ({frame_cnt/total_cal:.3f}frame/seconds)')
                if frame_face != 0:
                    frame_face -= 1
                    print(f'Average Inference+NMS times:({(inference_time_nms/frame_face):.3f}seconds)')
                if frame_6D != 0:
                    print(f'Average Inference+6D times:({(inference_time_6D/frame_6D):.3f}seconds)')

                return 0
            
            elif key == ord('T') or key == ord('t'):  # Toggle fullscreen
                show_text = not show_text

        # Save results (image with detections)
        if save_img:
            if dataset.mode == 'image':
                cv2.imwrite(save_path, im0)
            else:  # 'video' or 'stream'
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 10, im0.shape[1], im0.shape[0]
                        save_path += '.mp4'
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer.write(im0)

    if save_txt or save_txt_tidl or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt or save_txt_tidl else ''
        print(f"Results saved to {save_dir}{s}")

    # print(f'Done. ({time.time() - t0:.3f}s)')


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
