!/bin/sh
echo ""
echo "Hello, choose the env you want it~"
echo [0]: yolov7-gaze
echo ----------------
echo [1]: yolov7-eye[export onnx]
echo ----------------
echo [n]: None
echo ----------------
echo -n "Press enter to start it:"

read ENV_Set

#============================================================================
if [ $ENV_Set -eq 0 ] ; then
    source activate
    conda activate yolov7-gaze

    echo ============
    echo 「Success Enter yolov7-gaze」
    echo ============ 
fi

#============================================================================

if [ $ENV_Set -eq 1 ] ; then
    source activate
    conda activate yolov7-eye

    echo ============
    echo 「Success Enter yolov7-eye」
    echo ============
fi

#============================================================================ 

echo ""
echo "Hello, choose the weight you want it~"
echo [0]: yolov7-lite-s.pt
echo ----------------
echo [1]: yolov7-lite-t.pt
echo ----------------
echo [2]: yolov7-tiny.pt
echo ----------------
echo [3]: yolov7-lite-t-Mouth+own+mirrir-34lmk-2200epochs
echo ----------------
echo [4]: yolov7-lite-t-Mouth+own+mirrir-revise-34lmk-2200epochs
echo ----------------
echo [5]: yolov7-lite-t-Mouth+own+mirrir-36lmk-revise-2200epochs
echo ----------------
echo [6]: yolov7-lite-s-Mouth+own+mirrir-34lmk-revise-2200epochs
echo ----------------
echo [7]: yolov7-tiny-Mouth+own+mirrir-34lmk-revise-2200epochs
echo ----------------
echo [8]: yolov7-lite-t+own+mouth+7cs-34lmk-revise-600epochs
echo ----------------
echo [9]: 300W
echo ----------------
echo [10]: yolov7-lite-t-Mouth+own+mirrir+6cs-34lmk-revise-2200epochs
echo ----------------
echo [11]: yolov7-lite-t-Mouth+own+mirrir+7cs-34lmk-revise-2200epochs
echo ----------------
echo [n]: None
echo -n "Press enter to start it:"

read MY_Weights

if [ $MY_Weights -eq 0 ] ; then
    Weights='yolov7-lite-s.pt'
    kpt=5
fi 
if [ $MY_Weights -eq 1 ] ; then
    Weights='yolov7-lite-t.pt'
    kpt=5
fi 
if [ $MY_Weights -eq 2 ] ; then
    Weights='yolov7-tiny.pt'
    kpt=5
fi 
if [ $MY_Weights -eq 3 ] ; then
    Weights='yolov7-lite-t-Mouth+own+mirrir-34lmk-2200epochs/best.pt' 
    kpt=34
fi 
if [ $MY_Weights -eq 4 ] ; then
    Weights='yolov7-lite-t-Mouth+own+mirrir-revise-34lmk-2200epochs/best.pt' 
    kpt=34
fi 
if [ $MY_Weights -eq 5 ] ; then
    Weights='yolov7-lite-t-Mouth+own+mirrir-36lmk-revise-2200epochs/best.pt' 
    kpt=36
fi 
if [ $MY_Weights -eq 6 ] ; then
    Weights='yolov7-lite-s-Mouth+own+mirrir-revise-34lmk-2200epochs/best.pt' 
    kpt=34
fi 
if [ $MY_Weights -eq 7 ] ; then
    Weights='yolov7-tiny-Mouth+own+mirrir-34lmk-revise-2200epochs/best.pt' 
    kpt=34
fi 
if [ $MY_Weights -eq 8 ] ; then
    Weights='yolov7-lite-t+own+mouth+7cs-34lmk-revise-600epochs/best.pt' 
    kpt=34
fi 
if [ $MY_Weights -eq 9 ] ; then
    Weights='300W/best.pt' 
    kpt=34
fi 
if [ $MY_Weights -eq 10 ] ; then
    Weights='yolov7-lite-t-Mouth+own+mirrir+6cs-34lmk-revise-2200epochs/best.pt' 
    kpt=34
fi 
if [ $MY_Weights -eq 11 ] ; then
    Weights='yolov7-lite-t+own+mouth+mirrir+7cs-34lmk-revise-2200epochs/best.pt' 
    kpt=34
fi 

echo "The weights you choose:" $Weights
echo "The kpt number you choose:" $kpt

#============================================================================ 

echo ""
echo "Hello, choose the mode you want it~"
echo ------ YOLOV7-face Demo ------
echo [0]: detect mesh_multi + save-txt
echo ----------------
echo [1]: detect mesh  + save-txt
echo ----------------
echo [2]: otocam detect
echo ----------------
echo [3]: video detect
echo ----------------
echo ============================
echo ----------------
echo [4]: otocam detect_post
echo ----------------
echo [5]: video detect_post
echo ----------------
echo [8]: otocam detect_pnp_trt
echo ----------------
echo [9]: Video detect_pnp_trt
echo ----------------
echo ============================
echo ----------------
echo [100]: export onnx
echo -n "Press enter to start it:"

read MY_mode


#============================================================================ 


if [ $MY_mode -eq 0 ] ; then
    echo ============
    echo 「detect_mesh_multi + save-txt」
    echo ============

    python detect_mesh_multi.py \
    --weight ../../weights/yolov7-face/$Weights \
    --conf-thres 0.2 \
    --iou-thres 0.5 \
    --source /media/joe/Xavierssd/widerface/WIDER_train/images/\
    --save-txt

fi

#============================================================================ 

if [ $MY_mode -eq 1 ] ; then
    echo ============
    echo 「detect_mesh save-txt」
    echo ============

    python detect_mesh.py \
    --weight ../../weights/yolov7-face/$Weights \
    --conf-thres 0.2 \
    --iou-thres 0.5 \
    --source/media/joe/Xavierssd/widerface/WIDER_train/images/ \
    --save-txt \

fi

#============================================================================  

if [ $MY_mode -eq 2 ] ; then
    echo ============
    echo 「otocam  detect」
    echo ============

    python detect.py \
    --weight ../../weights/yolov7-face/$Weights \
    --conf-thres 0.2 \
    --iou-thres 0.5 \
    --source cam.txt \
    --kpt-label $kpt \
    --project ../yolov7-face-runs/cam/

fi

#============================================================================ 


if [ $MY_mode -eq 3 ] ; then
    echo ============
    echo 「Video  detect」
    echo ============

    python3 detect.py \
    --weights ../../weights/yolov7-face/$Weights \
    --source /media/joe/Xavierssd/2023_0816_otocam_datavideo/output29.avi \
    --conf-thres 0.2 \
    --iou-thres 0.5 \
    --kpt-label $kpt \
    --project ../yolov7-face-runs/video/

fi

#============================================================================  

if [ $MY_mode -eq 4 ] ; then
    echo ============
    echo 「otocam  detect_post」
    echo ============

    python detect_post.py \
    --weight ../../weights/yolov7-face/$Weights \
    --conf-thres 0.5 \
    --iou-thres 0.5 \
    --source cam.txt \
    --kpt-label $kpt \
    --project ../yolov7-face-runs/cam/

fi

#============================================================================ 


if [ $MY_mode -eq 5 ] ; then
    echo ============
    echo 「Video  detect_post」
    echo ============

    python3 detect_post.py \
    --weights ../../weights/yolov7-face/$Weights \
    --source /media/joe/Xavierssd/2023_0816_otocam_datavideo/output29.avi \
    --conf-thres 0.2 \
    --iou-thres 0.5 \
    --kpt-label $kpt \
    --project ../yolov7-face-runs/video/

fi

#============================================================================  

if [ $MY_mode -eq 8 ] ; then
    echo ============
    echo 「otocam  detect_pnp_trt」
    echo ============

    python detect_pnp_trt.py \
    --weight ../../weights/yolov7-face/$Weights \
    --conf-thres 0.2 \
    --iou-thres 0.5 \
    --source cam.txt \
    --kpt-label $kpt \
    --project ../yolov7-face-runs/cam/

fi

#============================================================================ 


if [ $MY_mode -eq 9 ] ; then
    echo ============
    echo 「Video  detect_pnp_trt」
    echo ============

    python3 detect_pnp_trt.py \
    --weights ../../weights/yolov7-face/$Weights \
    --source /media/joe/Xavierssd/2023_0816_otocam_datavideo/output12.avi \
    --conf-thres 0.2 \
    --iou-thres 0.5 \
    --kpt-label $kpt \
    --project ../yolov7-face-runs/video/ \
    --view-img

fi

#============================================================================ 
#============================================================================ 
#============================================================================ 


if [ $MY_mode -eq 100 ] ; then
    echo ============
    echo 「export onnx」
    echo ============

    python ./models/export.py \
    --weights ../../weights/yolov7-face/$Weights 

fi


#============================================================================ End
echo [===YOLO===] ok!


