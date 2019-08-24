## [Deep Residual Learning for Image Recognition](https://zhuanlan.zhihu.com/p/28413039)
随着网络的加深，出现了训练集准确率下降的现象，我们可以确定这不是由于过拟合（Overfit）造成的(过拟合的情况训练集应该准确率很高)，所以作者针对这个问题提出了一种全新的网络，叫深度残差网络，它允许网络尽可能的加深，其中引入了全新的结构。 
![resnet_block](../sources/resnet_block.PNG) 
多层的神经网络理论上可以拟合任意函数，那么可以利用一些层来拟合函数。如果深层网络的后面那些层是恒等映射，那么模型就退化为一个浅层网络。问题是直接拟合H(x)还是拟合残差函数F(x)=H(x)-x，拟合残差函数更简单，虽然理论上两者都能得到近似拟合，但是后者学习起来显然更容易。
　　作者认为，这种残差形式是由退化问题激发的。如果增加的层被构建为同等函数，那么理论上，更深的模型的训练误差不应当大于浅层模型，但是出现的退化问题表明，求解器很难去利用多层网络拟合同等函数。但是，残差的表示形式使得多层网络近似起来要容易的多，如果同等函数可被优化近似，那么多层网络的权重就会简单地逼近0来实现同等映射，即F(x)=0  。
(用人话来说就是，输入是x，希望经过几层神经网络后，输出还近似为x，这样就相当于增加了网络深度但实际上效果等价于较浅的网络。但问题是使H(x)=F(x)=x这个函数很难拟合，而F(x)=0比较容易拟合，于是定义H(x)=F(x)+x，拟合F(x)=0就好了）
如果输入是 5 ，期望经过几层网络后输出 5.1，引入残差前F'(5)=5.1，引入残差后H(5)=F(5)+5,则F(5)=0.1。  
如果输出从5.1变化到5.2那么，F'(x)的输出增加了2%，而残差结构F(x)从0.1增加到0.2，增加了100%。即残差结构对权重的调整作用更大，效果更好。残差思想就是去除相同的主题部分，突出微小的变化。
### 吴恩达深度学习课程对残差的理解
![吴恩达深度学习课程对残差的理解](../sources/resnet_NG2.PNG)
![吴恩达深度学习课程对残差的理解](../sources/resnet_NG3.PNG)
![吴恩达深度学习课程对残差的理解](../sources/resnet_NG1.PNG)

### 推荐/参考：
- [Deep Residual Learning for Image Recognition](https://zhuanlan.zhihu.com/p/28413039)
- [ResNet解析](https://blog.csdn.net/lanran2/article/details/79057994)
- [吴恩达深度学习课程对残差的理解](http://www.ai-start.com/dl2017/html/lesson4-week2.html)
- [ResNet论文笔记](https://blog.csdn.net/wspba/article/details/56019373)
- [残差网络ResNet笔记](https://www.jianshu.com/p/e58437f39f65)
- [Understand Deep Residual Networks — a simple, modular learning framework that has redefined state-of-- the-art](https://blog.waya.ai/deep-residual-learning-9610bb62c355)
- [An Overview of ResNet and its Variants](https://towardsdatascience.com/an-overview-of-resnet-and-its-variants-5281e2f56035)     
- [译文](https://www.jianshu.com/p/46d76bd56766)
- [Understanding and Implementing Architectures of ResNet and ResNeXt for state-of-the-art Image Classification: From Microsoft to Facebook [Part 1]](https://medium.com/@14prakash/understanding-and-implementing-architectures-of-resnet-and-resnext-for-state-of-the-art-image-cf51669e1624)
- [给妹纸的深度学习教学(4)——同Residual玩耍](https://zhuanlan.zhihu.com/p/28413039)
