### 目标检测中的包围框回归
包围框回归公式都来自于R-CNN系列  
![bounding_box_regression](sources/bounding_box_regression.PNG)
用人话来说就是：
![bbox_regression1](sources/bbox_regression1.jpg)
![bbox_regression2](sources/bbox_regression2.jpg)
![bbox_regression3](sources/bbox_regression3.jpg)
对于YOLO系列，用YOLOv2来说，还是套用 R-CNN 的公式：
![YOLOv2_box_regression](sources/YOLOv2_box_regression2.PNG)
![YOLOv2_box_regression](sources/YOLOv2_box_regression.PNG)

### 简单介绍下NMS
NMS步骤如下：
- 1.设置一个Score的阈值，一个IOU的阈值；
- 2.对于每类对象，遍历属于该类的所有候选框，
- 3.过滤掉Score低于Score阈值的候选框；
- 4.找到剩下的候选框中最大Score对应的候选框，添加到输出列表；
- 5.进一步计算剩下的候选框与4中输出列表中每个候选框的IOU，若该IOU大于设置的IOU阈值，将该候选框过滤掉，否则加入输出列表中；
- 6.最后输出列表中的候选框即为图片中该类对象预测的所有边界框

#### 推荐/参考链接




