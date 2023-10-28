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
echo [6]: yolov7-lite-t-Mouth-36lmk-80epochs.pt
echo ----------------
echo [7]: yolov7-lite-t-Mouth+own-36lmk-80epochs.pt
echo ----------------
echo [8]: yolov7-lite-t-Mouth+own+mirrir-36lmk-80epochs.pt
echo ----------------
echo [9]: yolov7-lite-t-Mouth+own+mirrir-36lmk-600epochs.pt
echo ----------------
echo [10]: yolov7-lite-t-Mouth+own+mirrir-36lmk-pre-600epochs.pt
echo ----------------
echo [11]: yolov7-lite-t-Mouth+own+mirrir-36lmk-2200epochs.pt
echo ----------------
echo [12]: yolov7-lite-t-Mouth+own+mirrir-34lmk-2200epochs.pt
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
if [ $MY_Weights -eq 6 ] ; then
    Weights='yolov7-lite-t-Mouth-36lmk-80epochs/yolov7-lite-t-Mouth-36lmk-80epochs.pt'
fi
if [ $MY_Weights -eq 7 ] ; then
    Weights='yolov7-lite-t-Mouth+own-36lmk-80epochs/yolov7-lite-t-Mouth+own-36lmk-80epochs.pt'
fi
if [ $MY_Weights -eq 8 ] ; then
    Weights='yolov7-lite-t-Mouth+own+mirrir-36lmk-80epochs/yolov7-lite-t-Mouth+own+mirrir-36lmk-80epochs.pt'
fi
if [ $MY_Weights -eq 9 ] ; then
    Weights='yolov7-lite-t-Mouth+own+mirrir-36lmk-600epochs/yolov7-lite-t-Mouth+own+mirrir-36lmk-600epochs.pt'
fi
if [ $MY_Weights -eq 10 ] ; then
    Weights='yolov7-lite-t-Mouth+own+mirrir-36lmk-pre-600epochs/yolov7-lite-t-Mouth+own+mirrir-36lmk-pre-600epochs.pt'
fi
if [ $MY_Weights -eq 11 ] ; then
    Weights='yolov7-lite-t-Mouth+own+mirrir-36lmk-2200epochs/best.pt'
fi
if [ $MY_Weights -eq 12 ] ; then
    Weights='yolov7-lite-t-Mouth+own+mirrir-34lmk-2200epochs/best.pt'
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
echo [6]: img detect
echo ----------------
echo [7]: otocam detect_6D
echo ----------------
echo [8]: otocam detect_pnp
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
    --kpt-label 34 \
    --project ../yolov7-face-runs/cam/

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
    --kpt-label 36 \
    --project ../yolov7-face-runs/video/

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
    --save-txt \

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

#============================================================================ 

if [ $MY_mode -eq 6 ] ; then
    echo ============
    echo 「img detect」
    echo ============

    python detect.py \
    --weight ../../weights/yolov7-face/$Weights \
    --conf-thres 0.2 \
    --iou-thres 0.5 \
    --source ./img/ \
    --kpt-label 36 \
    --project ../yolov7-face-runs/img/
    
fi

#============================================================================ 

if [ $MY_mode -eq 7 ] ; then
    echo ============
    echo 「otocam  detect_6D」
    echo ============

    python detect_6D.py \
    --weight ../../weights/yolov7-face/$Weights \
    --conf-thres 0.2 \
    --iou-thres 0.5 \
    --source cam.txt \
    --kpt-label 36 \
    --project ../yolov7-face-runs/cam/

fi

#============================================================================ 

if [ $MY_mode -eq 8 ] ; then
    echo ============
    echo 「otocam  detect_pnp」
    echo ============

    python detect_pnp.py \
    --weight ../../weights/yolov7-face/$Weights \
    --conf-thres 0.2 \
    --iou-thres 0.5 \
    --source 8 \
    --kpt-label 34 \
    --project ../yolov7-face-runs/cam/

fi


#============================================================================ End
echo [===YOLO===] ok!


