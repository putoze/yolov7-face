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
echo [3]: yolov7-light-t-80epoch-20231019.pt
echo ----------------
echo [4]: yolov7-lite-t-36lmk-80epochs-20231019.pt
echo ----------------
echo [5]: yolov7-lite-t-wilderface-36lmk-80epochs.pt
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
    Weights='yolov7-light-t-80epoch-20231019/yolov7-light-t-80epoch-20231019.pt'
fi 
if [ $MY_Weights -eq 4 ] ; then
    Weights='yolov7-lite-t-36lmk-80epochs-20231019/yolov7-lite-t-36lmk-80epochs-20231019.pt'
fi 
if [ $MY_Weights -eq 5 ] ; then
    Weights='yolov7-lite-t-wilderface-36lmk-80epochs/yolov7-lite-t-wilderface-36lmk-80epochs.pt'
fi 
echo $Weights

#============================================================================ 

echo ""
echo "Hello, choose the mode you want it~"
echo ------ Tensorrt Demo ------
echo [0]: otocam  detect
echo ----------------
echo [1]: Video  detect
echo ----------------
echo [2]: detect save-txt
echo ----------------
echo [3]: otocam detect mesh
echo ----------------
echo [4]: detect mesh_multi + save-txt
echo ----------------
echo [5]: otocam mesh_multi 
echo ----------------
echo -n "Press enter to start it:"

read MY_mode

#============================================================================ 

if [ $MY_mode -eq 0 ] ; then
    echo ============
    echo 「otocam  detect」
    echo ============

    python detect.py \
    --weight ../../weights/yolov7-face/$Weights \
    --conf-thres 0.2 \
    --iou-thres 0.5 \
    --source cam.txt \
    --kpt-label 36
    # --weight ./torch_yolov7_weight/yolov7-custom_v3/best.pt \
    
fi

#============================================================================ 
if [ $MY_mode -eq 1 ] ; then
    echo ============
    echo 「Video  detect」
    echo ============

    python3 detect.py \
    --weights ../../weights/yolov7-face/$Weights \
    --source /home/joe/Desktop/Camera_oToCAM250/2023_0816_otocam_datavideo/output29.avi \
    --conf-thres 0.2 \
    --iou-thres 0.5 \
    --kpt-label 36


fi

#============================================================================ 

if [ $MY_mode -eq 2 ] ; then
    echo ============
    echo 「otocam  detect_mesh save-txt」
    echo ============

    python detect_mesh.py \
    --weight ../../weights/yolov7-face/$Weights \
    --conf-thres 0.2 \
    --iou-thres 0.5 \
    --source/media/joe/Xavierssd/widerface/WIDER_train/images/ \
    --save-txt

fi

#============================================================================ 

if [ $MY_mode -eq 3 ] ; then
    echo ============
    echo 「otocam detect_mesh 」
    echo ============

    python detect_mesh.py \
    --weight ../../weights/yolov7-face/$Weights \
    --conf-thres 0.2 \
    --iou-thres 0.5 \
    --source cam.txt 
    # --weight ./torch_yolov7_weight/yolov7-custom_v3/best.pt \
    
fi

#============================================================================ 


if [ $MY_mode -eq 4 ] ; then
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


if [ $MY_mode -eq 5 ] ; then
    echo ============
    echo 「otocam  detect_mesh_multi 」
    echo ============

    python detect_mesh_multi.py \
    --weight ../../weights/yolov7-face/$Weights \
    --conf-thres 0.2 \
    --iou-thres 0.5 \
    --source cam.txt 

fi

#============================================================================ End
echo [===YOLO===] ok!


