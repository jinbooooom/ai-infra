## [YOLOv3: An Incremental Improvement](https://pjreddie.com/media/files/papers/YOLOv3.pdf)
### YOLOv3  有哪些创新？
- 提出新的网络结构：DarkNet-53  
YOLOv3在之前Darknet-19的基础上引入了残差块，并进一步加深了网络，改进后的网络有53个卷积层，取名为Darknet-53。
- 融合FPN，多尺度预测  
YOLOv3借鉴了FPN的思想，从不同尺度提取特征。YOLOv3提取最后3层特征图，不仅在每个特征图上分别独立做预测，同时将小特征图上采样到与大的特征图相同大小，然后与大的特征图拼接做进一步预测。用维度聚类的思想聚类出9种尺度的anchor box，将9种尺度的anchor box均匀的分配给3种尺度的特征图。每一个尺度的特征图预测 S \* S \* [3 \* (4 + 1 + C)]，这个式子与 YOLOv1 很像，但 YOLOv1 只在最后一个 7  \* 7 的特征图上做预测，但 YOLOv3 在 13 \* 13、26 \* 26、52  \* 52 的特征图上作预测，每层特征图有三个 anchor box。
- 用逻辑回归替代softmax作为分类器



