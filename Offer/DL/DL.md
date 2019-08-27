### 激活函数对比
[NGDL91]  
![activation](sources/activation_functions.jpg)
- 1.tanh 函数几乎在所有情况下都优于sigmod函数，但有一个例外，在二分类问题上，y值是0或1，所以y尖（y头上有一个^号）值介于[0, 1]，而不是[-1, 1]，此时需要用sigmoid
- 2.sigmoid 和 tanh 函数的共同特点，当 Z->正负无穷，函数的梯度变得很小，接近于零，导致梯度下降的速度变慢。
- 3.ReLU
	- a.当 z 是负值时，导数等于0，即一旦神经元激活值进入负半区，这个神经元就不会训练，即所谓稀疏性。
	- b.ReLU函数的导数程序实现即一个 if-else语句，而sigmoid函数需要进行浮点四则运算，使用ReLU能加快计算。
	- c.**ReLU函数在z > 0 时导数均为常数，不会产生梯度弥散现象，而sigmoid与tanh函数的导数在正负饱和区的导数都接近于0，产生梯度弥散。**这是使用ReLU与Leaky ReLU的主要原因。
- 4.在实际应用中，Leaky ReLU比ReLU效果好，虽然Leaky ReLU用的不多。

### 为什么要用非线性激活函数  
![](sources/0_1.PNG)    
![](sources/0_2.PNG)  
**不能在隐藏层使用线性激活函数，唯一可用线性激活函数的是输出层**  
ReLU看起来是线性，其实是非线性，相当于一个分段函数。

## 卷积
[NGDL356, 372]
### same方式卷积中padding的好处：
- 若不使用 padding，每次作卷积图像都会缩小
- 图像边缘的像素点在输出中采用较少，意味着丢掉了图像边缘像素的许多信息，使用padding使图像边缘的像素与中间
的像素采样数一样多。
### 卷积输入输出满足的计算公式
same方式下：
输出 size = (n + 2p - f)/s + 1，结果向下取整    
n: 输入尺寸，p: padding，f: 过滤器filter size，s: stride  
其实池化也满足这个公式，池化时，相当于 p=0, s=2  
valid模式下：
上式的 p=0
![conv](sources/conv.PNG)

### 设某层有过滤器n=N，filter size = W * H * C，那么共有参数多少？
(W * H * C + 1) * N  
即卷积这种方式相比于全连接来说，参数大大减少。模型的参数不会随着输入图片的 size 变化而变化。 

### 对pooling的理解？
最大值池化：Max pooling.  
若 feature map 中的某一位值特别大，则这一位置很可能就是有用的特征（参考边缘检测过滤器输出的 feature map），保留最大值即保留特征。
- 最大池化能更好地保留纹理信息
- 平均池化（mean pooling）保留数据地整体特征，突出背景信息      
用池化提取特征有两个误差：  
- 1.领域大小受限，造成了估计值方差增大
- 2.卷积层参数误差造成了估计均值地偏移，平均池化减少了第一种误差，最大池化减少了第二种误差
池化层没有需要学习地参数，池化后 channel 与输入相同，仅改变 size  
[NGDL381]  

### 1 * 1 卷积核的理解
![1_1_kernel](sources/1_1_kernel.PNG)  
![1_1_kernel](sources/1_1_kernel1.PNG)  
![1_1_kernel](sources/1_1_kernel2.PNG)  

#### 推荐/参考链接



