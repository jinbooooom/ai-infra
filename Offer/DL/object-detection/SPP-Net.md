
## [Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition](https://arxiv.org/abs/1406.4729)
### SPP-Net要解决的问题是什么？
spp提出的初衷是为了解决CNN对输入图片尺寸的限制。由于全连接层的存在，与之相连的最后一个卷积层的输出特征需要固定尺寸，从而要求输入图片尺寸也要固定。SPP-Net之前的做法是将图片裁剪或变形（crop/warp），如下图所示:
![spp-net](sources/spp-net1.png)
