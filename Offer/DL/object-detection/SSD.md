## [SSD](https://arxiv.org/abs/1512.02325)

### 简单叙述下SSD
SSD 是一个 one-stage 目标检测器，使用 VGG16 网络作为特征提取网络，但是将 VGG16的全连接层换成卷积层，并在后面添加自定义的卷积层(extras layer: conv 8_2, conv 9_2, conv 10_2, conv 11_2)，直接采用卷积层进行检测。在多个特征图上设置不同缩放比例和不同宽高比的先验框以融合多尺度特征图进行检测，低层的大尺度特征图可以用来捕捉小物体的信息，而高维的小尺度特征图能捕捉到大物体的信息，SSD融合高低维的特征，从而提高检测的准确性和定位的准确性。如下图是SSD的网络结构图。
![SSD1](sources/SSD1.PNG)

### SSD有哪些创新点？


#### 推荐/参考链接
- [SSD检测小目标](https://www.zhihu.com/search?type=content&q=ssd%E6%A3%80%E6%B5%8B%E5%B0%8F%E7%9B%AE%E6%A0%87)
