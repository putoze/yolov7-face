!/bin/sh
echo ""
echo "Hello, choose the env you want it~"
echo [0]: yolov7-gaze
echo ----------------
echo [1]: yolov3_tenrt
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
    conda activate yolov3_tenrt

    echo ============
    echo 「Success Enter yolov3_tenrt」
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
echo [3]: yolov7-lite-t-Mouth+own+mirrir-34lmk-2200epochs.pt
echo ----------------
echo [n]: None
echo -n "Press enter to start it:"

read MY_Weights

if [ $MY_Weights -eq 0 ] ; then
    Weights='yolov7-lite-s.pt'
fi 
if [ $MY_Weights -eq 1 ] ; then
    Weights='yolov7-lite-t.pt'
fi 
if [ $MY_Weights -eq 2 ] ; then
    Weights='yolov7-tiny.pt'
fi 
if [ $MY_Weights -eq 3 ] ; then
    Weights='yolov7-lite-t-Mouth+own+mirrir-34lmk-2200epochs/best.pt' 
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
echo [4]: otocam detect_6D
echo ----------------
echo [5]: video detect_6D
echo ----------------
echo [8]: otocam detect_pnp_trt
echo ----------------
echo [9]: Video detect_pnp_trt
echo ----------------
echo [12]: otocam detect_pnp_trt_6D
echo ----------------
echo [13]: Video detect_pnp_trt_6D
echo ----------------
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
    echo 「otocam  detect_6D」
    echo ============

    python detect_6D.py \
    --weight ../../weights/yolov7-face/$Weights \
    --conf-thres 0.2 \
    --iou-thres 0.5 \
    --source cam.txt \
    --kpt-label $kpt \
    --project ../yolov7-face-runs/cam/

fi

#============================================================================ 


if [ $MY_mode -eq 5 ] ; then
    echo ============
    echo 「Video  detect_6D」
    echo ============

    python3 detect_6D.py \
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
    --source /media/joe/Xavierssd/2023_0816_otocam_datavideo/output29.avi \
    --conf-thres 0.2 \
    --iou-thres 0.5 \
    --kpt-label $kpt \
    --project ../yolov7-face-runs/video/

fi

#============================================================================ End
echo [===YOLO===] ok!


