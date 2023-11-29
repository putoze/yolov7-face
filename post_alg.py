import cv2
import numpy as np
import math
import torch
import os
import torchvision


# alert
from scipy.spatial import distance as dist
import numpy as np
import argparse
import time
from alert.drowsiness_yawn import alarm,eye_aspect_ratio,final_ear,lip_distance

# 6DRepNet

## 6D RepNet golbal
import numpy as np
from PIL import Image
from torchvision import transforms
from numpy.lib.function_base import _quantile_unchecked
from matplotlib import pyplot as plt
import matplotlib
import RepNet_6D.utils_with_6D as utils_with_6D

## 6D RepNet local
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
matplotlib.use('TkAgg')

# GazeML
import GazeML.gaze_modelbased as GM

from GazeML.iris_localization import IrisLocalizationModel

# transformations_6D
transformations_6D = transforms.Compose([transforms.Resize(224),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# 3D model points.
model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-225.0, 170.0, -135.0),     #  eye  corner
                            (225.0, 170.0, -135.0),      # Right eye right corne
                            (-150.0, -150.0, -125.0),    #  Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner
                        ])

# 3D model points.
model_points_v2 = np.array([
    (0.0, 0.0, 0.0),  # Nose tip
    (0, -63.6, -12.5),  # Chin
    (-43.3, 32.7, -26),  #  eye,  corner
    (43.3, 32.7, -26),  # Right eye, right corner
    (-28.9, -28.9, -24.1),  #  Mouth corner
    (28.9, -28.9, -24.1)  # Right mouth corner
])

YAWN_THRESH = 25

def fitEllipse(input_img,flag_list,bb):
    target_img = None

    # Convert to grayscale if gray_flag is true
    if flag_list[0]:
        img_gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        target_img = img_gray
    
    # Normalize image
    # min_pixel_value = np.min(img_gray)
    # max_pixel_value = np.max(img_gray)
    # normalized_image = ((img_gray - min_pixel_value) / (max_pixel_value - min_pixel_value) * 255).astype(np.uint8)
    # target_img = normalized_image

    equalized_image = cv2.equalizeHist(img_gray)
    target_img = equalized_image

    kernel = np.ones((3, 3), np.uint8)

    # Thresholding if binary_flag is true
    if flag_list[1]:
        binary = cv2.bilateralFilter(target_img, 10, 15, 15)
        binary = cv2.erode(binary, kernel, iterations=3)
        # binary = cv2.threshold(binary, threshold, 255, cv2.THRESH_BINARY)[1]
        binary = cv2.adaptiveThreshold(target_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,\
                cv2.THRESH_BINARY, 11, 2)
        target_img = binary

    # Morphological operations if morphology_flag is true
    if flag_list[2]:
        morphologyDisk = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        morphology_img = cv2.morphologyEx(target_img, cv2.MORPH_CLOSE, morphologyDisk)
        target_img = morphology_img

    # Gaussian blur if Gaussblur_flag is true
    if flag_list[3]:
        Gaussblur_img = cv2.GaussianBlur(target_img, (5, 5), 0)
        target_img = Gaussblur_img

    # Sobel edge detection if Sobel_flag is true
    if flag_list[4]:
        sobelX = cv2.Sobel(target_img, cv2.CV_16S, 1, 0, ksize=3)
        sobelY = cv2.Sobel(target_img, cv2.CV_16S, 0, 1, ksize=3)
        sobelX8U = cv2.convertScaleAbs(sobelX)
        sobelY8U = cv2.convertScaleAbs(sobelY)
        Sobel_img = cv2.addWeighted(sobelX8U, 0.5, sobelY8U, 0.5, 0)
        target_img = Sobel_img

    # Canny edge detection if Canny_flag is true
    if flag_list[5]:
        canny = cv2.Canny(target_img, 30, 150)
        target_img = canny

    # Find contours if Contours_flag is true
    if flag_list[6]:
        contours, _ = cv2.findContours(target_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # cv2.drawContours(target_img, contours, 3, (0, 255, 0), 3)

    # to show image or not
    # if flag_list[1]:
    #     cv2.imshow("binary Image", cv2.resize(binary,(100,100)))
    # if flag_list[2]:
    #     cv2.imshow("morphology Image", cv2.resize(morphology_img,(100,100)))
    # if flag_list[3]:
    #     cv2.imshow("Gaussian Image", cv2.resize(Gaussblur_img,(100,100)))
    # if flag_list[4]:
    #     cv2.imshow("Sobel_img Image", cv2.resize(Sobel_img,(100,100)))
    # if flag_list[5]:
    #     cv2.imshow("canny Image", cv2.resize(canny,(100,100)))
    # if flag_list[6]:
    #     cv2.imshow("contours Image", cv2.resize(target_img,(100,100)))
    # cv2.imshow("equalized Image", cv2.resize(equalized_image,(100,100)))

    # find max area
    maxArea = 0
    max_countuor = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > maxArea:
            maxArea = area
            max_countuor = contour

    # Find best areas
    # print("max area", maxArea)

    if maxArea != 0:
        # momentsPupilThresh = cv2.moments(max_countuor)
        # center = (int(momentsPupilThresh["m10"] / momentsPupilThresh["m00"]),
        #              int(momentsPupilThresh["m01"] / momentsPupilThresh["m00"]))

        # cv2.circle(img, center, 3, (0, 0, 255), -1)
        contour_pt_array = np.array(max_countuor, dtype=np.int32)

        # Avoid to break the system
        if(contour_pt_array.shape[0] < 5):
            return None
        
        elPupilThresh = cv2.fitEllipse(contour_pt_array)
        center = (int(elPupilThresh[0][0] + bb[0]),int(elPupilThresh[0][1]) + bb[1])
        radius = (int(elPupilThresh[1][0]),int(elPupilThresh[1][1]))
        
        # print("elPupilThresh")
        # print("Center:",elPupilThresh[0])
        # print("Size:" ,elPupilThresh[1])
        # print("Angle:" ,elPupilThresh[2])

        # Color = (0, 255, 0)  # Green color
        # thickness = 2
        # center = (int(elPupilThresh[0][0]),int(elPupilThresh[0][1]))
        # cv2.ellipse(draw_img, elPupilThresh, Color, thickness)
        # cv2.circle(draw_img, center, 3, (0, 0, 255), -1)
        
        # final_img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)

        return [center,radius,elPupilThresh[2]]
    else :
        return None

def find_ellipse_point(num_points,elthresh):
    center,axes,angle = elthresh
    angle_step = 360 / num_points
    ellipse_points = []
    for i in range(num_points):
        angle_deg = angle + angle_step * i
        angle_rad = np.radians(angle_deg)
        x_point = center[0] + (axes[0] / 2) * np.cos(angle_rad)
        y_point = center[1] + (axes[1] / 2) * np.sin(angle_rad)
        ellipse_points.append([x_point, y_point])

    return np.array(ellipse_points)

def find_yolo_ellipse_point(num_points,center,axes,roll):
    angle_step = 360 / num_points
    ellipse_points = []
    for i in range(num_points):
        angle_deg = angle_step * i + roll
        angle_rad = np.radians(angle_deg)
        x_point = center[0] + (axes[0] / 2) * np.cos(angle_rad)
        y_point = center[1] + (axes[1] / 2) * np.sin(angle_rad)
        ellipse_points.append([x_point, y_point])

    return np.array(ellipse_points)

def draw_ellipse_point(img,ellipse_points,elPupilThresh):
    img_out = img
    # cv2.ellipse(img_out, elPupilThresh, (0, 255, 0), 2)
    center_point = (int(elPupilThresh[0][0]),int(elPupilThresh[0][1]))
    cv2.circle(img_out, center_point, 3, (0, 0, 255), -1)
    for pt in ellipse_points:
        point = (int(pt[0]),int(pt[1]))
        cv2.circle(img_out, point, 2, (0, 255, 0), -1)

    return img_out

def draw_yolo_ellipse_point(img,ellipse_points,center_point):
    img_out = img
    # cv2.ellipse(img_out, elPupilThresh, (0, 255, 0), 2)
    cv2.circle(img_out, center_point, 3, (0, 0, 255), -1)
    for pt in ellipse_points:
        point = (int(pt[0]),int(pt[1]))
        cv2.circle(img_out, point, 2, (0, 255, 0), -1)

    return img_out

        
def headpose_alg(im0,coordinate):

    # headpose
    size = im0.shape[:2]
    
    # Camera internals
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array(
                            [[focal_length, 0, center[0]],
                            [0, focal_length, center[1]],
                            [0, 0, 1]], dtype = "double")
                    
    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion

    image_points = np.array([coordinate[12],
                             coordinate[33],
                             coordinate[9],
                             coordinate[0],
                             coordinate[19],
                             coordinate[13]], dtype="double")

    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    rvec_matrix = cv2.Rodrigues(rotation_vector)[0]
    proj_matrix = np.hstack((rvec_matrix, translation_vector))
    eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6]

    yaw   =  eulerAngles[1]
    pitch = -eulerAngles[0]
    roll  =  eulerAngles[2]
    
    tdx = size[1] - 70
    tdy = 70

    utils_with_6D.draw_axis(im0,yaw,pitch,roll,tdx,tdy, size = 50)
    utils_with_6D.draw_gaze_6D(coordinate[12],im0,yaw,pitch,color=(0, 255, 255))
        
    return im0,yaw,pitch,roll

def alert(im0,coordinate_np,glasses_flag):
    
    leftEye = coordinate_np[0:6]
    rightEye = coordinate_np[6:12]
    distance = lip_distance(coordinate_np[13:33])
    lip = coordinate_np[13:25]

    # yawn_flag
    yawn_flag = 0

    # EAR
    ear = final_ear(leftEye,rightEye)

    # draw
    leftEyeHull = cv2.convexHull(leftEye)
    rightEyeHull = cv2.convexHull(rightEye)
    cv2.drawContours(im0, [leftEyeHull], -1, (0, 255, 255), 1)
    cv2.drawContours(im0, [rightEyeHull], -1, (0, 255, 255), 1)
    cv2.drawContours(im0, [lip], -1, (0, 255, 255), 1)

    if glasses_flag:
        cv2.putText(im0, "glasses wear", (300, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.putText(im0, "EAR: {:.2f}".format(ear), (300, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(im0, "YAWN: {:.2f}".format(distance), (300, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    if (distance > YAWN_THRESH):
        cv2.putText(im0, "Yawn Alert", (300, 120),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        yawn_flag = 1
    
    return im0, yawn_flag

def alg_6DRepNet(im0,driver_face_roi,model_6DRepNet,device,nose_point):

    x_min,y_min,x_max,y_max = driver_face_roi
    bbox_width = abs(x_max - x_min)
    bbox_height = abs(y_max - y_min)

    x_min = max(0, x_min-int(0.2*bbox_height))
    y_min = max(0, y_min-int(0.2*bbox_width))
    x_max = x_max+int(0.2*bbox_height)
    y_max = y_max+int(0.2*bbox_width)

    img = im0[y_min:y_max, x_min:x_max]
    img = Image.fromarray(img)
    img = img.convert('RGB')
    img = transformations_6D(img)

    img = torch.Tensor(img[None, :]).to(device)

    R_pred = model_6DRepNet(img)

    euler = utils_with_6D.compute_euler_angles_from_rotation_matrices(
        R_pred)*180/np.pi
    
    p_pred_deg = euler[:, 0].cpu()
    y_pred_deg = euler[:, 1].cpu()
    r_pred_deg = euler[:, 2].cpu()

    # utils_with_6D.plot_pose_cube(im0,  y_pred_deg, p_pred_deg, r_pred_deg, x_min + int(.5*(
    #     x_max-x_min)), y_min + int(.5*(y_max-y_min)), size=bbox_width)

    height, width = im0.shape[:2]
    tdx = width - 70
    tdy = 70 * 2
    utils_with_6D.draw_axis(im0,y_pred_deg,p_pred_deg,r_pred_deg,tdx,tdy, size = 50)
    utils_with_6D.draw_gaze_6D(nose_point,im0,y_pred_deg,p_pred_deg,color= (255, 255, 0))

    return im0,y_pred_deg,p_pred_deg,r_pred_deg

    # End 6DRepNet
        

def icon_alg(im0,icon_flag):

    # ICON
    icon_w = 75
    icon_h = 75
    back_size = 25

    icon_drowsiness= cv2.imread("./icon/drowsiness.png")
    icon_drowsiness_re = cv2.resize(icon_drowsiness,(icon_w,icon_h))
    icon_phone = cv2.imread("./icon/phone.png")
    icon_phone_re = cv2.resize(icon_phone,(icon_w,icon_h))
    icon_attentive = cv2.imread("./icon/attentive.png")
    icon_attentive_re = cv2.resize(icon_attentive,(icon_w,icon_h))
    icon_seatbelt = cv2.imread("./icon/seatbelt.png")
    icon_seatbelt_re = cv2.resize(icon_seatbelt,(icon_w,icon_h))
    icon_smoke = cv2.imread("./icon/smoke.png")
    icon_smoke_re = cv2.resize(icon_smoke,(icon_w,icon_h))

    # drowsiness
    icon_loc_x = 600
    next_icon_loc_x = icon_loc_x + icon_w
    if icon_flag[0] == 1:
        icon_drowsiness_re[:,:,2] = 255
        im0[back_size:icon_h+back_size, icon_loc_x:next_icon_loc_x]  = icon_drowsiness_re
    else:
        im0[back_size:icon_h+back_size, icon_loc_x:next_icon_loc_x]  = icon_drowsiness_re

    # attentive
    icon_loc_x = next_icon_loc_x + back_size
    next_icon_loc_x = icon_loc_x + icon_w
    if icon_flag[1] == 1:
        icon_attentive_re[:,:,2] = 255
        im0[back_size:icon_h+back_size, icon_loc_x:next_icon_loc_x]  = icon_attentive_re
    else:
        im0[back_size:icon_h+back_size, icon_loc_x:next_icon_loc_x]  = icon_attentive_re

    # seatbelt
    icon_loc_x = next_icon_loc_x + back_size
    next_icon_loc_x = icon_loc_x + icon_w
    if icon_flag[2] == 1:
        im0[back_size:icon_h+back_size, icon_loc_x:next_icon_loc_x]  = icon_seatbelt_re
    else:
        icon_seatbelt_re[:,:,2] = 255
        im0[back_size:icon_h+back_size, icon_loc_x:next_icon_loc_x]  = icon_seatbelt_re

    # phone
    icon_loc_x = next_icon_loc_x + back_size
    next_icon_loc_x = icon_loc_x + icon_w
    if icon_flag[3] == 1:
        icon_phone_re[:,:,2] = 255
        im0[back_size:icon_h+back_size, icon_loc_x:next_icon_loc_x]  = icon_phone_re
    else:
        im0[back_size:icon_h+back_size, icon_loc_x:next_icon_loc_x]  = icon_phone_re

    # smoke
    icon_loc_x = next_icon_loc_x + back_size
    next_icon_loc_x = icon_loc_x + icon_w
    if icon_flag[4] == 1:
        icon_smoke_re[:,:,2] = 255
        im0[back_size:icon_h+back_size, icon_loc_x:next_icon_loc_x]  = icon_smoke_re
    else:
        im0[back_size:icon_h+back_size, icon_loc_x:next_icon_loc_x]  = icon_smoke_re

    return im0



def gaze_estimate(frame, coordinate, pupil_left, pupil_right):
    """
    The gaze function gets an image and face landmarks from mediapipe framework.
    The function draws the gaze direction into the frame.
    """

    '''
    2D image points.
    relative takes mediapipe points that is normalized to [-1, 1] and returns image points
    at (x,y) format
    '''
    image_points = np.array([
        coordinate[12], # Nose tip
        coordinate[33], # Chin
        coordinate[9] , # Left eye, left corner
        coordinate[0] , # Right eye, right corner
        coordinate[19], # Left Mouth corner
        coordinate[13]  # Right mouth corner
        ], dtype="double")
    '''
    2D image points.
    relativeT takes mediapipe points that is normalized to [-1, 1] and returns image points
    at (x,y,0) format
    '''
    image_points1 = np.array([
        (coordinate[12][0],coordinate[12][1],0),  # Nose tip
        (coordinate[33][0],coordinate[33][1],0),  # Chin
        (coordinate[9] [0],coordinate[9] [1],0),  # Left eye, left corner
        (coordinate[0] [0],coordinate[0] [1],0),  # Right eye, right corner
        (coordinate[19][0],coordinate[19][1],0),  # Left Mouth corner
        (coordinate[13][0],coordinate[13][1],0)   # Right mouth corner
    ], dtype="double")

    '''
    3D model eye points
    The center of the eye ball
    '''
    Eye_ball_center_right = np.array([[-29.05], [32.7], [-39.5]])
    Eye_ball_center_left = np.array([[29.05], [32.7], [-39.5]])  # the center of the left eyeball as a vector.

    '''
    camera matrix estimation
    '''
    focal_length = frame.shape[1]
    center = (frame.shape[1] / 2, frame.shape[0] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )

    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points_v2, image_points, camera_matrix,
                                                                  dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    rvec_matrix = cv2.Rodrigues(rotation_vector)[0]
    proj_matrix = np.hstack((rvec_matrix, translation_vector))
    eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6]

    yaw   =  eulerAngles[1]
    pitch = -eulerAngles[0]
    roll  =  eulerAngles[2]

    height, width = frame.shape[:2]
    tdx = width - 70
    tdy = 70

    # Transformation between image point to world point
    _, transformation, _ = cv2.estimateAffine3D(image_points1, model_points_v2)  # image to world transformation

    utils_with_6D.draw_axis(frame,yaw,pitch,roll,tdx,tdy, size = 50)
    utils_with_6D.draw_gaze_6D(coordinate[12],frame,yaw,pitch,color=(0, 255, 255))

    if transformation is not None:  # if estimateAffine3D secsseded
        # project pupil image point into 3d world point 
        if len(pupil_left) != 0:
            left_pupil = pupil_left[4]
            pupil_world_cord_l = transformation @ np.array([[left_pupil[0], left_pupil[1], 0, 1]]).T

            # 3D gaze point (10 is arbitrary value denoting gaze distance)
            S = Eye_ball_center_left + (pupil_world_cord_l - Eye_ball_center_left) * 2

            # Project a 3D gaze direction onto the image plane.
            (eye_pupil2D, _) = cv2.projectPoints((int(S[0]), int(S[1]), int(S[2])), rotation_vector,
                                                translation_vector, camera_matrix, dist_coeffs)
            # project 3D head pose into the image plane
            (head_pose, _) = cv2.projectPoints((int(pupil_world_cord_l[0]), int(pupil_world_cord_l[1]), int(40)),
                                            rotation_vector, translation_vector, camera_matrix, dist_coeffs)
            # correct gaze for head rotation
            gaze = left_pupil + (eye_pupil2D[0][0] - left_pupil) - (head_pose[0][0] - left_pupil)

            # Draw gaze line into screen
            p1 = (int(left_pupil[0]), int(left_pupil[1]))
            p2 = (int(gaze[0]), int(gaze[1]))
            cv2.arrowedLine(frame, p1, p2, (0, 0, 255), 2)

        if len(pupil_right) != 0:
            right_pupil = pupil_right[4]
            pupil_world_cord_r = transformation @ np.array([[right_pupil[0], right_pupil[1], 0, 1]]).T

            # 3D gaze Eye_ball_center_right (10 is arbitrary value denoting gaze distance)
            S = Eye_ball_center_right + (pupil_world_cord_r - Eye_ball_center_right) * 2

            # Project a 3D gaze direction onto the image plane.
            (eye_pupil2D, _) = cv2.projectPoints((int(S[0]), int(S[1]), int(S[2])), -rotation_vector,
                                                -translation_vector, camera_matrix, dist_coeffs)
            # project 3D head pose into the image plane
            (head_pose, _) = cv2.projectPoints((int(pupil_world_cord_r[0]), int(pupil_world_cord_r[1]), int(40)),
                                            -rotation_vector,
                                            -translation_vector, camera_matrix, dist_coeffs)
            # correct gaze for head rotation
            gaze = right_pupil + (eye_pupil2D[0][0] - right_pupil) - (head_pose[0][0] - right_pupil)

            # Draw gaze line into screen
            p1 = (int(right_pupil[0]), int(right_pupil[1]))
            p2 = (int(gaze[0]), int(gaze[1]))
            cv2.arrowedLine(frame, p1, p2, (0, 0, 255), 2)

    return frame,yaw,pitch,roll

def gaze_GazeML(im0, eye, pupil, num_iris_landmark = 8):

    # define eye loc/center/length
    eye_center = ((eye[0][0] + eye[3][0])/2,(eye[0][1] + eye[3][1])/2)
    eye_length = eye[3][0] - eye[0][0]
    
    # define pupil imformation
    radius = pupil[1]
    
    iris_ldmks = find_yolo_ellipse_point(num_iris_landmark,pupil[0],radius,pupil[2])
    im0 = draw_yolo_ellipse_point(im0,iris_ldmks,pupil[0])
    gaze = GM.estimate_gaze_from_landmarks(iris_ldmks, pupil[0], eye_center, eye_length)

    gaze = gaze.reshape(1, 2)

    return gaze


def gaze_laser(frame,eye):
    gs = IrisLocalizationModel("weights/iris_landmark.tflite")
    eye_lengths = int(eye[3][0] - eye[0][0])
    eye_center = (int((eye[0][0] + eye[3][0])/2),int((eye[0][1] + eye[3][1])/2))

    iris = gs.get_mesh(frame, eye_lengths, eye_center)
    pupil, _ = gs.draw_pupil(iris, frame, thickness=1)

    return pupil, eye_center


def gaze_PFLD(img,device,gaze_pfld):
    
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()])

    height, width = img.shape[:2]

    input = cv2.resize(img, (160,112))
    input = transform(input).unsqueeze(0).to(device)
    landmarks, gaze = gaze_pfld(input)

    pre_landmark = landmarks[0]
    #print(pre_landmark.shape)
    pre_landmark = pre_landmark.cpu().detach().numpy().reshape(
        -1, 2) * [width, height]

    gaze = gaze.cpu().detach().numpy()[0]

    c_pos = pre_landmark[-1,:]

    cv2.line(img, tuple(c_pos.astype(int)), tuple(c_pos.astype(int)+(gaze*400).astype(int)), (0,255,0), 3)

    return img