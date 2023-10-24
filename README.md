# yolov7-face

### New feature

* Dynamic keypoints
* WingLoss
* Efficient backbones
* EIOU and SIOU


| Method           |  Test Size | Easy  | Medium | Hard  | FLOPs (B) @640 |
| -----------------| ---------- | ----- | ------ | ----- | -------------- |
| yolov7-lite-t    | 640        | 88.7  | 85.2   | 71.5  |  0.8           | 
| yolov7-lite-s    | 640        | 92.7  | 89.9   | 78.5  |  3.0           |
| yolov7-tiny      | 640        | 94.7  | 92.6   | 82.1  |  13.2          | 
| yolov7s          | 640        | 94.8  | 93.1   | 85.2  |  16.8          | 
| yolov7           | 640        | 96.9  | 95.5   | 88.0  |  103.4         | 
| yolov7+TTA       | 640        | 97.2  | 95.8   | 87.7  |  103.4         |
| yolov7-w6        | 960        | 96.4  | 95.0   | 88.3  |  89.0          |
| yolov7-w6+TTA    | 1280       | 96.9  | 95.8   | 90.4  |  89.0          | 

#### How to use 

```
./launch_yolov7.sh
1. you can choose the enviroment by your self
2. choose the weights you want to use
3. choose the mode you want to use
```


#### Dataset

- own custom dataset

#### References

* [https://github.com/deepcam-cn/yolov5-face](https://github.com/deepcam-cn/yolov5-face)

* [https://github.com/WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)

* [https://github.com/TexasInstruments/edgeai-yolov5/tree/yolo-pose](https://github.com/TexasInstruments/edgeai-yolov5/tree/yolo-pose)

* [https://github.com/ppogg/YOLOv5-Lite](https://github.com/ppogg/YOLOv5-Lite)
