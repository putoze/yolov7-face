# yolov7-face

### New feature

* Dynamic keypoints
* WingLoss
* Efficient backbones
* EIOU and SIOU

### Demo

<p align="left">
  <img src="gif/DMS_demo.gif" width="300" heigh ="400"/>
</p>

### Result

#### yolov7-tiny-7cs-34lmk-300W-pre-600epochs

| Class      | Images | Labels | Precision | Recall | mAP@.5 | mAP@.5:.95 |
|------------|--------|--------|-----------|--------|--------|------------ |
| all        | 758    | 2894   | 0.926     | 0.899  | 0.933  | 0.715       |
| face       | 758    | 768    | 0.987     | 0.919  | 0.988  | 0.955       |
| mask       | 758    | 135    | 0.953     | 0.978  | 0.981  | 0.882       |
| glasses    | 758    | 342    | 0.969     | 0.886  | 0.950  | 0.779       |
| seatbelt   | 758    | 356    | 0.972     | 0.961  | 0.991  | 0.886       |
| phone      | 758    | 109    | 0.893     | 0.972  | 0.958  | 0.740       |
| smoke      | 758    | 58     | 0.745     | 0.707  | 0.708  | 0.276       |
| pupil      | 758    | 1126   | 0.963     | 0.869  | 0.951  | 0.490       |

lkpt, lkptv, loss : 0.003422641893848777 0.017872514203190804 0.06481204926967621
Model Summary: 224 layers, 6298926 parameters, 0 gradients, 14.0 GFLOPS

#### yolov7-lite-t-7cs-34lmk-face-pre-600epochs

| Class      | Images | Labels | Precision | Recall | mAP@.5 | mAP@.5:.95: |
|------------|--------|--------|-----------|--------|--------|------------ |
| all        | 758    | 2894   | 0.886     | 0.878  | 0.894  | 0.614       |
| face       | 758    | 768    | 0.982     | 0.943  | 0.989  | 0.890       |
| mask       | 758    | 135    | 0.909     | 0.993  | 0.980  | 0.820       |
| glasses    | 758    | 342    | 0.951     | 0.857  | 0.938  | 0.611       |
| seatbelt   | 758    | 356    | 0.972     | 0.968  | 0.990  | 0.796       |
| phone      | 758    | 109    | 0.859     | 0.954  | 0.933  | 0.635       |
| smoke      | 758    | 58     | 0.635     | 0.569  | 0.506  | 0.126       |
| pupil      | 758    | 1126   | 0.898     | 0.862  | 0.924  | 0.421       |

Model Summary: 236 layers, 292862 parameters, 0 gradients, 1.0 GFLOPS
lkpt, lkptv, loss : 0.004928789101541042 0.01692609302699566 0.07272126525640488

#### YOLOV7-face-lite-s 7cs Transfer 300W_200epochs 600epochs 

| Class      | Images | Labels | Precision | Recall | mAP@.5 | mAP@.5:.95 |
|------------|--------|--------|-----------|--------|--------|------------|
| all        | 758    | 2894   | 0.908     | 0.890  | 0.919  | 0.678      |
| face       | 758    | 768    | 0.988     | 0.929  | 0.989  | 0.939      |
| mask       | 758    | 135    | 0.939     | 0.993  | 0.987  | 0.864      |
| glasses    | 758    | 342    | 0.973     | 0.841  | 0.946  | 0.706      |
| seatbelt   | 758    | 356    | 0.972     | 0.960  | 0.990  | 0.862      |
| phone      | 758    | 109    | 0.871     | 0.972  | 0.945  | 0.699      |
| smoke      | 758    | 58     | 0.664     | 0.655  | 0.631  | 0.213      |
| pupil      | 758    | 1126   | 0.950     | 0.880  | 0.946  | 0.462      |

lkpt, lkptv, loss : 0.0038735729176551104 0.017150314524769783 0.0674903616309166
Model Summary: 276 layers, 1154910 parameters, 0 gradients, 3.4 GFLOPS


### How to use 

```
./launch_yolov7.sh
1. you can choose the enviroment by your self
2. choose the weights you want to use
3. choose the mode you want to use
```

### Dataset

- own custom dataset

### References

* [https://github.com/deepcam-cn/yolov5-face](https://github.com/deepcam-cn/yolov5-face)

* [https://github.com/WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)

* [https://github.com/TexasInstruments/edgeai-yolov5/tree/yolo-pose](https://github.com/TexasInstruments/edgeai-yolov5/tree/yolo-pose)

* [https://github.com/ppogg/YOLOv5-Lite](https://github.com/ppogg/YOLOv5-Lite)
