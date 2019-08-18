[TOC]

# 申明

本资料来源于CVer公众号，只面向【CVer AI算法岗备战群】成员，仅供学习参考，不能二次转发。

进群链接：https://t.zsxq.com/FqJYZnU

注：资料内容还在完善，这还只是预览版，但从题目可以看出学习重点，希望可以帮忙各位同学了解基本情况。


# 一、自我介绍

你好，我是来自xxx大学的学生，我叫xxx，目前本科/研究生x年级。

研究生期间主要研究xxx方向。既涉及xxx，也涉及xxx技术。研一的时候。研究生期间，xxx...


# 二、深度学习/机器学习

## 2.1 逻辑回归（Logistic Regression，LR）

逻辑回归（Logistic Regression）也称为"对数几率回归"，又称为"逻辑斯谛"回归。

**知识点精简：**

- 分类，经典的二分类算法！
- 逻辑回归就是这样的一个过程：面对一个回归或者分类问题，建立代价函数，然后通过优化方法迭代求解出最优的模型参数，然后测试验证我们这个求解的模型的好坏。

- Logistic回归虽然名字里带“回归”，但是它实际上是一种分类方法，主要用于两分类问题（即输出只有两种，分别代表两个类别）

- 回归模型中，y是一个定性变量，比如y=0或1，logistic方法主要应用于研究某些事件发生的概率。

- 逻辑回归的本质——极大似然估计

- 逻辑回归的激活函数——Sigmoid

- 逻辑回归的代价函数——交叉熵

**逻辑回归的优缺点**

优点： 

1）速度快，适合二分类问题 

2）简单易于理解，直接看到各个特征的权重 

3）能容易地更新模型吸收新的数据 

缺点： 

对数据和场景的适应能力有局限性，不如决策树算法适应性那么强


**逻辑回归中最核心的概念是[Sigmoid函数](https://en.wikipedia.org/wiki/Sigmoid_function)**，Sigmoid函数可以看成逻辑回归的激活函数。

下图是逻辑回归网络：

![Logistic Regression.png](http://note.youdao.com/yws/res/72648/WEBRESOURCE1c0a6a8efd5c65f5c1f39ce0e74281e8)


对数几率函数（Sigmoid）：


```math
y = \sigma (z) = \frac{1}{1+e^{-z}}

```

![](https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Logistic-curve.svg/600px-Logistic-curve.svg.png)


通过函数S的作用，我们可以将输出的值限制在区间[0， 1]上，p(x)则可以用来表示概率p(y=1|x)，即当一个x发生时，y被分到1那一组的概率。可是，等等，我们上面说y只有两种取值，但是这里却出现了一个区间[0, 1]，这是什么鬼？？其实在真实情况下，我们最终得到的y的值是在[0, 1]这个区间上的一个数，然后我们可以选择一个阈值，通常是0.5，当y>0.5时，就将这个x归到1这一类，如果y<0.5就将x归到0这一类。但是阈值是可以调整的，比如说一个比较保守的人，可能将阈值设为0.9，也就是说有超过90%的把握，才相信这个x属于1这一类。了解一个算法，最好的办法就是自己从头实现一次。下面是逻辑回归的具体实现。


**Regression 常规步骤**

1. 寻找h函数（即预测函数）

2. 构造J函数（损失函数）

3. 想办法（迭代）使得J函数最小并求得回归参数（θ）


函数h(x)的值有特殊的含义，它表示结果取1的概率，于是可以看成类1的后验估计。因此对于输入x分类结果为类别1和类别0的概率分别为： 

P(y=1│x;θ)=hθ (x) 

P(y=0│x;θ)=1-hθ (x)


**代价函数**

**逻辑回归一般使用交叉熵作为代价函数**。关于代价函数的具体细节，请参考[代价函数](http://www.cnblogs.com/Belter/p/6653773.html)。

交叉熵是对「出乎意料」（译者注：原文使用suprise）的度量。神经元的目标是去计算函数y, 且y=y(x)。但是我们让它取而代之计算函数a, 且a=a(x)。假设我们把a当作y等于1的概率，1−a是y等于0的概率。那么，交叉熵衡量的是我们在知道y的真实值时的平均「出乎意料」程度。当输出是我们期望的值，我们的「出乎意料」程度比较低；当输出不是我们期望的，我们的「出乎意料」程度就比较高。


交叉熵代价函数如下所示：

```math
J(w)=-l(w)=-\sum_{i = 1}^n y^{(i)}ln(\phi(z^{(i)})) + (1 - y^{(i)})ln(1-\phi(z^{(i)}))

J(\phi(z),y;w)=-yln(\phi(z))-(1-y)ln(1-\phi(z))
```


注：为什么要使用交叉熵函数作为代价函数，而不是平方误差函数？请参考：[逻辑回归算法之交叉熵函数理解](https://blog.csdn.net/syyyy712/article/details/78252722)


逻辑回归伪代码


```
初始化线性函数参数为1
构造sigmoid函数
重复循环I次
	计算数据集梯度
	更新线性函数参数
确定最终的sigmoid函数
输入训练（测试）数据集
运用最终sigmoid函数求解分类

```

逻辑回归算法python算法

```python
# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import random
 
 
def text2num(string):
    """
    :param string: string
    :return: list
    """
    str_list = string.replace("\n", " ").split(" ")
    while '' in str_list:
        str_list.remove('')
    num_list = [float(i) for i in str_list]
    return num_list
 
 
def sigmoid(x):
    """
    :param x: 输入需要计算的值
    :return: 
    """
    return 1.0 / (1 + np.exp(-x))
 
 
def data_plot(data_list, weight):
    """
    :param data_list:数据点集合 
    :param weight: 参数集合
    :return: null
    """
    x_data = [list(i[0:2]) for i in data_list if i[2] == 0.0]
    y_data = [list(i[0:2]) for i in data_list if i[2] == 1.0]
    x_data = np.reshape(x_data, np.shape(x_data))
    y_data = np.reshape(y_data, np.shape(y_data))
    linear_x = np.arange(-4, 4, 1)
    linear_y = (-weight[0] - weight[1] * linear_x) / weight[2]
    print(linear_y)
    plt.figure(1)
    plt.scatter(x_data[:, 0], x_data[:, 1], c='r')
    plt.scatter(y_data[:, 0], y_data[:, 1], c='g')
    print(linear_x)
    print(linear_y.tolist()[0])
    plt.plot(linear_x, linear_y.tolist()[0])
    plt.show()
 
 
def grad_desc(data_mat, label_mat, rate, times):
    """
    :param data_mat: 数据特征
    :param label_mat: 数据标签
    :param rate: 速率
    :param times: 循环次数
    :return: 参数
    """
    data_mat = np.mat(data_mat)
    label_mat = np.mat(label_mat)
    m,n = np.shape(data_mat)
    weight = np.ones((n, 1))
    for i in range(times):
        h = sigmoid(data_mat * weight)
        error = h - label_mat
        weight = weight - rate * data_mat.transpose() * error
    return weight
 
 
def random_grad_desc(data_mat, label_mat, rate, times):
    """
    :param data_mat: 数据特征
    :param label_mat: 数据标签
    :param rate: 速率
    :param times: 循环次数
    :return: 参数
    """
    data_mat = np.mat(data_mat)
    m,n = np.shape(data_mat)
    weight = np.ones((n, 1))
    for i in range(times):
        for j in range(m):
            h = sigmoid(data_mat[j] * weight)
            error = h - label_mat[j]
            weight = weight - rate * data_mat[j].transpose() * error
    return weight
 
 
def improve_random_grad_desc(data_mat, label_mat, times):
    """
    :param data_mat: 数据特征
    :param label_mat: 数据标签
    :param rate: 速率
    :param times: 循环次数
    :return: 参数
    """
    data_mat = np.mat(data_mat)
    m,n = np.shape(data_mat)
    weight = np.ones((n, 1))
    for i in range(times):
        index_data = [i for i in range(m)]
        for j in range(m):
            rate = 0.0001 + 4 / (i + j + 1)
            index = random.sample(index_data, 1)
            h = sigmoid(data_mat[index] * weight)
            error = h - label_mat[index]
            weight = weight - rate * data_mat[index].transpose() * error
            index_data.remove(index[0])
    return weight
 
 
def main():
    file = open("/Users/chenzu/Documents/code-machine-learning/data/LR", "rb")
    file_lines = file.read().decode("UTF-8")
    data_list = text2num(file_lines)
    data_len = int(len(data_list) / 3)
    data_list = np.reshape(data_list, (data_len, 3))
    data_mat_temp = data_list[:, 0:2]
    data_mat = []
    for i in data_mat_temp:
        data_mat.append([1, i[0], i[1]])
    print(data_mat)
    label_mat = data_list[:, 2:3]
 
 
    #梯度下降求参数
    weight = improve_random_grad_desc(data_mat, label_mat, 500)
    print(weight)
    data_plot(data_list, weight)
 
 
if __name__ == '__main__':
    main()

```

逻辑回归算法python算法实验结果



参考：

《统计学习方法》 第6章  P77页

《机器学习》 西瓜书 第3章  P57页

[《Machine Learning》 吴恩达 Logistic Regression](https://d19vezwu8eufl6.cloudfront.net/ml/docs%2Fslides%2FLecture6.pdf)

[逻辑回归(logistic regression)的本质——极大似然估计](https://blog.csdn.net/zjuPeco/article/details/77165974)

[逻辑回归](https://blog.csdn.net/pakko/article/details/37878837)

[【机器学习】逻辑回归（Logistic Regression）](https://www.cnblogs.com/Belter/p/6128644.html)

[机器学习算法--逻辑回归原理介绍](https://blog.csdn.net/chibangyuxun/article/details/53148005)

[十分钟掌握经典机器学习算法-逻辑回归](https://blog.csdn.net/tangyudi/article/details/80131307)


## 2.2 支持向量机（Support Vector Machine，SVM）

**知识点精简：**

- SVM核函数
    - 多项式核函数
    - 高斯核函数
    - 字符串核函数
- SMO
- SVM损失函数


支持向量机（supporr vector machine，SVM）是一种二类分类模型，该模型是定义在特征空间上的间隔最大的线性分类器。间隔最大使它有区别于感知机；支持向量机还包括核技巧，这使它成为实质上的非线性分类器。支持向量机的学习策略就是**间隔最大化**，可形式化为一个求解凸二次规划的最小化问题。

支持向量机的学习算法是求解凸二次规划的最优化算法。

支持向量机学习方法包含构建由简至繁的模型：

- 线性可分支持向量机
- 线性支持向量机
- 非线性支持向量机（使用核函数）

当训练数据线性可分时，通过硬间隔最大化（hard margin maximization）学习一个线性的分类器，即线性可分支持向量机，又成为硬间隔支持向量机；

当训练数据近似线性可分时，通过软间隔最大化（soft margin maximization）也学习一个线性的分类器，即线性支持向量机，又称为软间隔支持向量机；

当训练数据不可分时，通过核技巧（kernel trick）及软间隔最大化，学习非线性支持向量机。


注：以上各SVM的数学推导应该熟悉：硬间隔最大化（几何间隔）---学习的对偶问题---软间隔最大化（引入松弛变量）---非线性支持向量机（核技巧）。


**SVM的主要特点**

（1）非线性映射-理论基础 

（2）最大化分类边界-方法核心 

（3）支持向量-计算结果 

（4）小样本学习方法 
（5）最终的决策函数只有少量支持向量决定，避免了“维数灾难” 

（6）少数支持向量决定最终结果—->可“剔除”大量冗余样本+算法简单+具有鲁棒性（体现在3个方面） 

（7）学习问题可表示为凸优化问题—->全局最小值 

（8）可自动通过最大化边界控制模型，但需要用户指定核函数类型和引入松弛变量 

（9）适合于小样本，优秀泛化能力（因为结构风险最小） 

（10）泛化错误率低，分类速度快，结果易解释


**SVM为什么采用间隔最大化？**

当训练数据线性可分时，存在无穷个分离超平面可以将两类数据正确分开。

感知机利用误分类最小策略，求得分离超平面，不过此时的解有无穷多个。

线性可分支持向量机利用间隔最大化求得最优分离超平面，这时，解是唯一的。另一方面，此时的分隔超平面所产生的分类结果是最鲁棒的，对未知实例的泛化能力最强。

然后应该借此阐述，几何间隔，函数间隔，及从函数间隔—>求解最小化1/2 ||w||^2 时的w和b。即线性可分支持向量机学习算法—最大间隔法的由来。


**为什么要将求解SVM的原始问题转换为其对偶问题？**


2. 对偶问题往往更易求解（当我们寻找约束存在时的最优点的时候，约束的存在虽然减小了需要搜寻的范围，但是却使问题变得更加复杂。为了使问题变得易于处理，我们的方法是把目标函数和约束全部融入一个新的函数，即拉格朗日函数，再通过这个函数来寻找最优点。）

3. 自然引入核函数，进而推广到非线性分类问题


**为什么SVM要引入核函数？**

当样本在原始空间线性不可分时，可将样本从原始空间映射到一个更高维的特征空间，使得样本在这个特征空间内线性可分。

引入映射后的对偶问题：


**SVM核函数有哪些？**

- 线性（Linear）核函数：主要用于线性可分的情形。参数少，速度快。
- 多项式核函数
- 高斯（RBF）核函数：主要用于线性不可分的情形。参数多，分类结果非常依赖于参数。
- Sigmoid核函数
- 拉普拉斯（Laplac）核函数

注：如果feature数量很大，跟样本数量差不多，建议使用LR或者Linear kernel的SVM。
如果feature数量较少，样本数量一般，建议使用Gaussian Kernel的SVM。


**SVM如何处理多分类问题？**

一般有两种做法：一种是直接法，直接在目标函数上修改，将多个分类面的参数求解合并到一个最优化问题里面。看似简单但是计算量却非常的大。

另外一种做法是间接法：对训练器进行组合。其中比较典型的有一对一，和一对多。

一对多，就是对每个类都训练出一个分类器，由svm是二分类，所以将此而分类器的两类设定为目标类为一类，其余类为另外一类。这样针对k个类可以训练出k个分类器，当有一个新的样本来的时候，用这k个分类器来测试，那个分类器的概率高，那么这个样本就属于哪一类。这种方法效果不太好，bias比较高。

svm一对一法（one-vs-one），针对任意两个类训练出一个分类器，如果有k类，一共训练出C(2,k) 个分类器，这样当有一个新的样本要来的时候，用这C(2,k) 个分类器来测试，每当被判定属于某一类的时候，该类就加一，最后票数最多的类别被认定为该样本的类。


**SVM中硬间隔和软间隔**

硬间隔分类即线性可分支持向量机，软间隔分类即线性不可分支持向量机，利用软间隔分类时是因为存在一些训练集样本不满足函数间隔（泛函间隔）大于等于1的条件，于是加入一个非负的参数 ζ （松弛变量），让得出的函数间隔加上 ζ 满足条件。于是软间隔分类法对应的拉格朗日方程对比于硬间隔分类法的方程就多了两个参数（一个ζ ，一个 β），但是当我们求出对偶问题的方程时惊奇的发现这两种情况下的方程是一致的。下面我说下自己对这个问题的理解。

我们可以先考虑软间隔分类法为什么会加入ζ 这个参数呢？硬间隔的分类法其结果容易受少数点的控制，这是很危险的，由于一定要满足函数间隔大于等于1的条件，而存在的少数离群点会让算法无法得到最优解，于是引入松弛变量，从字面就可以看出这个变量是为了缓和判定条件，所以当存在一些离群点时我们只要对应给他一个ζi，就可以在不变更最优分类超平面的情况下让这个离群点满足分类条件。

综上，我们可以看出来软间隔分类法加入ζ 参数，使得最优分类超平面不会受到离群点的影响，不会向离群点靠近或远离，相当于我们去求解排除了离群点之后，样本点已经线性可分的情况下的硬间隔分类问题，所以两者的对偶问题是一致的。


参考：

[支持向量机通俗导论（理解SVM的三层境界）](https://blog.csdn.net/v_july_v/article/details/7624837)

[数据挖掘（机器学习）面试--SVM面试常考问题](https://blog.csdn.net/szlcw1/article/details/52259668)

[机器学习实战教程（八）：支持向量机原理篇之手撕线性SVM](http://cuijiahua.com/blog/2017/11/ml_8_svm_1.html)

[支持向量机（SVM）入门理解与推导](https://blog.csdn.net/sinat_20177327/article/details/79729551)

## 2.3 如何防止过拟合和欠拟合


**过拟合（Over-Fitting）**

高方差

在训练集上误差小，但在测试集上误差大，我们将这种情况称为高方差（high variance），也叫过拟合。


**欠拟合（Under-Fitting）**

在训练集上训练效果不好（测试集上也不好），准确率不高，我们将这种情况称为高偏差（high bias），也叫欠拟合。


![](https://testerhome.com/uploads/photo/2017/ba5ebeb8-1af7-4dfa-aeba-6bb36c056aff.png!large)



**如何解决过拟合？**

- 数据增广（Data Augmentation）
- 正则化（L0正则、L1正则和L2正则），也叫限制权值Weight-decay
- Dropout
- Early Stopping
- 简化模型
- 增加噪声
- Bagging
- 贝叶斯方法


**如何解决欠拟合？**

- 添加新特征
- 添加多项式特征
- 减少正则化参数
- 增加网络复杂度
- 使用集成学习方法，如Bagging



## 2.4 L1和L2正则化（L1正则化为什么使权值稀疏）

目的：降低损失函数

机器学习中几乎都可以看到损失函数后面会添加一个额外项，常用的额外项一般有两种，一般英文称作ℓ1-norm和ℓ2-norm，中文称作L1正则化和L2正则化，或者L1范数和L2范数。


L1正则化和L2正则化可以看做是损失函数的惩罚项。所谓『惩罚』是指对损失函数中的某些参数做一些限制。对于线性回归模型，使用L1正则化的模型建叫做Lasso回归，使用L2正则化的模型叫做Ridge回归（岭回归）。下图是Python中Lasso回归的损失函数，式中加号后面一项α||w||1即为L1正则化项。

![lasso.png](https://note.youdao.com/yws/res/72659/WEBRESOURCE6dbcdbaaf05ebf3cec869f99a1a6c48e)

下图是Python中Ridge回归的损失函数，式中加号后面一项即为L2正则化项。

![ridge.png](https://note.youdao.com/yws/res/72651/WEBRESOURCE34534b3997f5d08f62474ae054ce09bc)


一般回归分析中回归w表示特征的系数，从上式可以看到正则化项是对系数做了处理（限制）。L1正则化和L2正则化的说明如下：


- L1正则化是指权值向量w中各个元素的绝对值之和，通常表示为||w||1

- L2正则化是指权值向量w中各个元素的平方和然后再求平方根（可以看到Ridge回归的L2正则化项有平方符号），通常表示为||w||2
一般都会在正则化项之前添加一个系数，Python中用α表示，一些文章也用λ表示。这个系数需要用户指定。

那添加L1和L2正则化有什么用？下面是L1正则化和L2正则化的作用，这些表述可以在很多文章中找到。

- L1正则化可以产生稀疏权值矩阵，即产生一个稀疏模型，可以用于特征选择

- L2正则化可以防止模型过拟合（overfitting）；一定程度上，L1也可以防止过拟合


**稀疏模型与特征选择**

上面提到L1正则化有助于生成一个稀疏权值矩阵，进而可以用于特征选择。为什么要生成一个稀疏矩阵？

稀疏矩阵指的是很多元素为0，只有少数元素是非零值的矩阵，即得到的线性回归模型的大部分系数都是0. 通常机器学习中特征数量很多，例如文本处理时，如果将一个词组（term）作为一个特征，那么特征数量会达到上万个（bigram）。在预测或分类时，那么多特征显然难以选择，但是如果代入这些特征得到的模型是一个稀疏模型，表示只有少数特征对这个模型有贡献，绝大部分特征是没有贡献的，或者贡献微小（因为它们前面的系数是0或者是很小的值，即使去掉对模型也没有什么影响），此时我们就可以只关注系数是非零值的特征。这就是稀疏模型与特征选择的关系。


**L1和L2正则化的直观理解**

这部分内容将解释为什么L1正则化可以产生稀疏模型（L1是怎么让系数等于零的），以及为什么L2正则化可以防止过拟合。

L1正则化和特征选择
假设有如下带L1正则化的损失函数：

![lasso2.png](https://note.youdao.com/yws/res/72657/WEBRESOURCE400344abcd53f0d15c2b4b90bfe1bbb6)


其中J0是原始的损失函数，加号后面的一项是L1正则化项，α是正则化系数。注意到L1正则化是权值的绝对值之和，J是带有绝对值符号的函数，因此J是不完全可微的。机器学习的任务就是要通过一些方法（比如梯度下降）求出损失函数的最小值。当我们在原始损失函数J0后添加L1正则化项时，相当于对J0做了一个约束。令L=α∑w|w|，则J=J0+L，此时我们的任务变成在L约束下求出J0取最小值的解。考虑二维的情况，即只有两个权值w1和w2，此时L=|w1|+|w2|对于梯度下降法，求解J0的过程可以画出等值线，同时L1正则化的函数L也可以在w1w2的二维平面上画出来。如下图：

![lasso3.png](https://note.youdao.com/yws/res/72661/WEBRESOURCE277de2586f4a66c9c1ed12feb5c8a8fd)
图1 L1正则化

图中等值线是J0的等值线，黑色方形是L函数的图形。在图中，当J0等值线与L图形首次相交的地方就是最优解。上图中J0与L在L的一个顶点处相交，这个顶点就是最优解。注意到这个顶点的值是(w1,w2)=(0,w)。可以直观想象，因为L函数有很多『突出的角』（二维情况下四个，多维情况下更多），J0与这些角接触的机率会远大于与L其它部位接触的机率，而在这些角上，会有很多权值等于0，这就是为什么L1正则化可以产生稀疏模型，进而可以用于特征选择。

而正则化前面的系数α，可以控制L图形的大小。α越小，L的图形越大（上图中的黑色方框）；α越大，L的图形就越小，可以小到黑色方框只超出原点范围一点点，这是最优点的值(w1,w2)=(0,w)中的w可以取到很小的值。

类似，假设有如下带L2正则化的损失函数： 

![ridge2.png](https://note.youdao.com/yws/res/72645/WEBRESOURCE32a75e9fec57446fabcea149fb419a48)

同样可以画出它们在二维平面上的图形，如下：

![ridge3.png](https://note.youdao.com/yws/res/72656/WEBRESOURCE71b10b753dbf692c65621cb24992313b)
图2 L2正则化

二维平面下L2正则化的函数图形是个圆，与方形相比，被磨去了棱角。因此J0与L相交时使得w1或w2等于零的机率小了许多，这就是为什么L2正则化不具有稀疏性的原因。


注：以二维平面举例，借助可视化L1和L2，可知L1正则化具有稀疏性。


**L2正则化和过拟合**

拟合过程中通常都倾向于让权值尽可能小，最后构造一个所有参数都比较小的模型。因为一般认为参数值小的模型比较简单，能适应不同的数据集，也在一定程度上避免了过拟合现象。可以设想一下对于一个线性回归方程，若参数很大，那么只要数据偏移一点点，就会对结果造成很大的影响；但如果参数足够小，数据偏移得多一点也不会对结果造成什么影响，专业一点的说法是『抗扰动能力强』。

那为什么L2正则化可以获得值很小的参数？

以线性回归中的梯度下降法为例。假设要求的参数为θ，hθ(x)是我们的假设函数，那么线性回归的代价函数如下： 

![cost function.png](https://note.youdao.com/yws/res/72652/WEBRESOURCE0a5d617e1c4e5acde0fd957b50d0560c)

那么在梯度下降法中，最终用于迭代计算参数θ的迭代式为： 

[link](https://note.youdao.com/)![cost function2.png](https://note.youdao.com/yws/res/72658/WEBRESOURCE7fa9339f7fd6ec09f46e94130b163b64)


其中α是learning rate. 上式是没有添加L2正则化项的迭代公式，如果在原始代价函数之后添加L2正则化，则迭代公式会变成下面的样子： 

![cost function3.png](https://note.youdao.com/yws/res/72660/WEBRESOURCEfb3e9fe55c2b1a2c36cc0bb559819b84)


其中λ就是正则化参数。从上式可以看到，与未添加L2正则化的迭代公式相比，每一次迭代，θj都要先乘以一个小于1的因子，从而使得θj不断减小，因此总得来看，θ是不断减小的。
最开始也提到L1正则化一定程度上也可以防止过拟合。之前做了解释，当L1的正则化系数很小时，得到的最优解会很小，可以达到和L2正则化类似的效果。


L2正则化参数

从上述公式可以看到，λ越大，θj衰减得越快。另一个理解可以参考图2，λ越大，L2圆的半径越小，最后求得代价函数最值时各参数也会变得很小。


参考：

[机器学习中正则化项L1和L2的直观理解](https://blog.csdn.net/jinping_shi/article/details/52433975)




## 2.5 CNN本质是什么？

局部卷积+Pooling


参考：

[卷积神经网络(CNN)基础介绍](https://blog.csdn.net/fengbingchun/article/details/50529500)


## 2.6 卷积操作


参考：

[Feature Extraction Using Convolution](http://ufldl.stanford.edu/tutorial/supervised/FeatureExtractionUsingConvolution/)

[convolution](https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/convolution.html)

[理解图像卷积操作的意义](https://blog.csdn.net/chaipp0607/article/details/72236892?locationNum=9&fps=1)

[关于深度学习中卷积核操作](https://www.cnblogs.com/Yu-FeiFei/p/6800519.html)


## 2.7 卷积反向传播过程






参考：

[Notes on Convolutional Neural Network](http://cogprints.org/5869/1/cnn_tutorial.pdf)

[Deep Learning论文笔记之（四）CNN卷积神经网络推导和实现](https://blog.csdn.net/zouxy09/article/details/9993371)

[反向传导算法](http://deeplearning.stanford.edu/wiki/index.php/%E5%8F%8D%E5%90%91%E4%BC%A0%E5%AF%BC%E7%AE%97%E6%B3%95)

[Deep learning：五十一(CNN的反向求导及练习)](https://www.cnblogs.com/tornadomeet/p/3468450.html)

[卷积神经网络(CNN)反向传播算法](https://www.cnblogs.com/pinard/p/6494810.html)

[卷积神经网络(CNN)反向传播算法公式详细推导](https://blog.csdn.net/walegahaha/article/details/51945421)



## 2.8 TensorFlow中的卷积操作实现

## 2.9 池化方法和操作


## 2.10 池化层怎么接收后面传过来的损失

**平均池化（Mean Pooling）**

mean pooling的前向传播就是把一个patch中的值求取平均来做pooling，那么反向传播的过程也就是把某个元素的梯度等分为n份分配给前一层，这样就保证池化前后的梯度（残差）之和保持不变，还是比较理解的，图示如下 

**最大池化（Max Pooling）**

max pooling也要满足梯度之和不变的原则，max pooling的前向传播是把patch中最大的值传递给后一层，而其他像素的值直接被舍弃掉。那么反向传播也就是把梯度直接传给前一层某一个像素，而其他像素不接受梯度，也就是为0。所以max pooling操作和mean pooling操作不同点在于需要记录下池化操作时到底哪个像素的值是最大，也就是max id，这个可以看caffe源码的pooling_layer.cpp，下面是caffe框架max pooling部分的源码


参考:
[深度学习笔记（3）——CNN中一些特殊环节的反向传播](https://blog.csdn.net/qq_21190081/article/details/72871704)




## 2.11 梯度提升决策树GDBT

下面关于GBDT的理解来自论文greedy function approximation: a gradient boosting machine

1. 损失函数的数值优化可以看成是在函数空间，而不是在参数空间。

2. 损失函数L(y,F)包含平方损失(y−F)2，绝对值损失|y−F|用于回归问题，负二项对数似然log(1+e−2yF),y∈{-1,1}用于分类。

3. 关注点是预测函数的加性扩展。

最关键的点在于损失函数的数值优化可以看成是在函数空间而不是参数空间。


GBDT对分类问题基学习器是二叉分类树，对回归问题基学习器是二叉决策树。


参考：

[简单易学的机器学习算法——梯度提升决策树GBDT](https://blog.csdn.net/google19890102/article/details/51746402/)

[GBDT原理详解](https://www.cnblogs.com/ScorpioLu/p/8296994.html)



## 2.12 XGBoost

**XGBoost全名叫（eXtreme Gradient Boosting）极端梯度提升**，经常被用在一些比赛中，其效果显著。它是大规模并行boosted tree的工具，它是目前最快最好的开源boosted tree工具包。下面我们将XGBoost的学习分为3步：

①集成思想 

②损失函数分析 

③求解。

我们知道机器学习三要素：模型、策略、算法。对于集成思想的介绍，XGBoost算法本身就是以集成思想为基础的。所以理解清楚集成学习方法对XGBoost是必要的，它能让我们更好的理解其预测函数模型。在第二部分，我们将详细分析损失函数，这就是我们将要介绍策略。第三部分，对于目标损失函数求解，也就是算法了。



参考：

[XGBoost Documentation](https://xgboost.readthedocs.io/en/latest/)

[通俗、有逻辑的写一篇说下Xgboost的原理，供讨论参考](https://blog.csdn.net/github_38414650/article/details/76061893)

[xgboost的原理没你想像的那么难](https://www.jianshu.com/p/7467e616f227)


## 2.13 随机森林（Random Forests）

随机森林属于集成学习（Ensemble Learning）中的bagging算法。在集成学习中，主要分为bagging算法和boosting算法。我们先看看这两种方法的特点和区别。

Bagging（套袋法）
bagging的算法过程如下：

从原始样本集中使用Bootstraping方法随机抽取n个训练样本，共进行k轮抽取，得到k个训练集。（k个训练集之间相互独立，元素可以有重复）
对于k个训练集，我们训练k个模型（这k个模型可以根据具体问题而定，比如决策树，knn等）
对于分类问题：由投票表决产生分类结果；对于回归问题：由k个模型预测结果的均值作为最后预测结果。（所有模型的重要性相同）
Boosting（提升法）
boosting的算法过程如下：

对于训练集中的每个样本建立权值wi，表示对每个样本的关注度。当某个样本被误分类的概率很高时，需要加大对该样本的权值。
进行迭代的过程中，每一步迭代都是一个弱分类器。我们需要用某种策略将其组合，作为最终模型。（例如AdaBoost给每个弱分类器一个权值，将其线性组合最为最终分类器。误差越小的弱分类器，权值越大）
Bagging，Boosting的主要区别
样本选择上：Bagging采用的是Bootstrap随机有放回抽样；而Boosting每一轮的训练集是不变的，改变的只是每一个样本的权重。
样本权重：Bagging使用的是均匀取样，每个样本权重相等；Boosting根据错误率调整样本权重，错误率越大的样本权重越大。
预测函数：Bagging所有的预测函数的权重相等；Boosting中误差越小的预测函数其权重越大。
并行计算：Bagging各个预测函数可以并行生成；Boosting各个预测函数必须按顺序迭代生成。
下面是将决策树与这些算法框架进行结合所得到的新的算法：

1）Bagging + 决策树 = 随机森林

2）AdaBoost + 决策树 = 提升树

3）Gradient Boosting + 决策树 = GBDT



**决策树**

常用的决策树算法有ID3，C4.5，CART三种。3种算法的模型构建思想都十分类似，只是采用了不同的指标。决策树模型的构建过程大致如下：

ID3，C4.5决策树的生成
输入：训练集D，特征集A，阈值eps 输出：决策树T

若D中所有样本属于同一类Ck，则T为单节点树，将类Ck作为该结点的类标记，返回T
若A为空集，即没有特征作为划分依据，则T为单节点树，并将D中实例数最大的类Ck作为该结点的类标记，返回T
否则，计算A中各特征对D的信息增益(ID3)/信息增益比(C4.5)，选择信息增益最大的特征Ag
若Ag的信息增益（比）小于阈值eps，则置T为单节点树，并将D中实例数最大的类Ck作为该结点的类标记，返回T
否则，依照特征Ag将D划分为若干非空子集Di，将Di中实例数最大的类作为标记，构建子节点，由结点及其子节点构成树T，返回T
对第i个子节点，以Di为训练集，以A-{Ag}为特征集，递归地调用1~5，得到子树Ti，返回Ti
CART决策树的生成
这里只简单介绍下CART与ID3和C4.5的区别。

CART树是二叉树，而ID3和C4.5可以是多叉树
CART在生成子树时，是选择一个特征一个取值作为切分点，生成两个子树
选择特征和切分点的依据是基尼指数，选择基尼指数最小的特征及切分点生成子树


**随机森林**


随机森林是一种重要的基于Bagging的集成学习方法，可以用来做分类、回归等问题。

随机森林有许多优点：

具有极高的准确率
随机性的引入，使得随机森林不容易过拟合
随机性的引入，使得随机森林有很好的抗噪声能力
能处理很高维度的数据，并且不用做特征选择
既能处理离散型数据，也能处理连续型数据，数据集无需规范化
训练速度快，可以得到变量重要性排序
容易实现并行化
随机森林的缺点：

当随机森林中的决策树个数很多时，训练时需要的空间和时间会较大
随机森林模型还有许多不好解释的地方，有点算个黑盒模型
与上面介绍的Bagging过程相似，随机森林的构建过程大致如下：

从原始训练集中使用Bootstraping方法随机有放回采样选出m个样本，共进行n_tree次采样，生成n_tree个训练集
对于n_tree个训练集，我们分别训练n_tree个决策树模型
对于单个决策树模型，假设训练样本特征的个数为n，那么每次分裂时根据信息增益/信息增益比/基尼指数选择最好的特征进行分裂
每棵树都一直这样分裂下去，直到该节点的所有训练样例都属于同一类。在决策树的分裂过程中不需要剪枝
将生成的多棵决策树组成随机森林。对于分类问题，按多棵树分类器投票决定最终分类结果；对于回归问题，由多棵树预测值的均值决定最终预测结果


参考：

[随机森林算法学习(Random Forest)](https://blog.csdn.net/qq547276542/article/details/78304454)

[[Machine Learning & Algorithm] 随机森林（Random Forest）](http://www.cnblogs.com/maybe2030/p/4585705.html)


## 2.14 AdaBoost

Adaboost算法基本原理就是将多个弱分类器（弱分类器一般选用单层决策树）进行合理的结合，使其成为一个强分类器。

Adaboost采用迭代的思想，每次迭代只训练一个弱分类器，训练好的弱分类器将参与下一次迭代的使用。也就是说，在第N次迭代中，一共就有N个弱分类器，其中N-1个是以前训练好的，其各种参数都不再改变，本次训练第N个分类器。其中弱分类器的关系是第N个弱分类器更可能分对前N-1个弱分类器没分对的数据，最终分类输出要看这N个分类器的综合效果。

参考：

[Adaboost入门教程——最通俗易懂的原理介绍（图文实例）](https://blog.csdn.net/px_528/article/details/72963977)

[AdaBoost原理详解](https://www.cnblogs.com/ScorpioLu/p/8295990.html)



## 2.15 K近邻（KNN)

1、KNN算法概述

　　kNN算法的核心思想是如果一个样本在特征空间中的k个最相邻的样本中的大多数属于某一个类别，则该样本也属于这个类别，并具有这个类别上样本的特性。该方法在确定分类决策上只依据最邻近的一个或者几个样本的类别来决定待分样本所属的类别。 
　　
2、KNN算法介绍

 　　最简单最初级的分类器是将全部的训练数据所对应的类别都记录下来，当测试对象的属性和某个训练对象的属性完全匹配时，便可以对其进行分类。但是怎么可能所有测试对象都会找到与之完全匹配的训练对象呢，其次就是存在一个测试对象同时与多个训练对象匹配，导致一个训练对象被分到了多个类的问题，基于这些问题呢，就产生了KNN。

KNN是通过测量不同特征值之间的距离进行分类。它的的思路是：如果一个样本在特征空间中的k个最相似(即特征空间中最邻近)的样本中的大多数属于某一个类别，则该样本也属于这个类别。K通常是不大于20的整数。KNN算法中，所选择的邻居都是已经正确分类的对象。该方法在定类决策上只依据最邻近的一个或者几个样本的类别来决定待分样本所属的类别。

下面通过一个简单的例子说明一下：如下图，绿色圆要被决定赋予哪个类，是红色三角形还是蓝色四方形？如果K=3，由于红色三角形所占比例为2/3，绿色圆将被赋予红色三角形那个类，如果K=5，由于蓝色四方形比例为3/5，因此绿色圆被赋予蓝色四方形类。

![](https://images0.cnblogs.com/blog2015/771535/201508/041623504236939.jpg)
　
　
　
接下来对KNN算法的思想总结一下：就是在训练集中数据和标签已知的情况下，输入测试数据，将测试数据的特征与训练集中对应的特征进行相互比较，找到训练集中与之最为相似的前K个数据，则该测试数据对应的类别就是K个数据中出现次数最多的那个分类，其算法的描述为：

1）计算测试数据与各个训练数据之间的距离；

2）按照距离的递增关系进行排序；

3）选取距离最小的K个点；

4）确定前K个点所在类别的出现频率；

5）返回前K个点中出现频率最高的类别作为测试数据的预测分类。　
　

参考：

[KNN算法原理及实现](https://www.cnblogs.com/sxron/p/5451923.html)




## 2.16 K-Means

算法思想：


```
选择K个点作为初始质心  
repeat  
    将每个点指派到最近的质心，形成K个簇  
    重新计算每个簇的质心  
until 簇不发生变化或达到最大迭代次数  
```

这里的重新计算每个簇的质心，如何计算的是根据目标函数得来的，因此在开始时我们要考虑距离度量和目标函数。

考虑欧几里得距离的数据，使用误差平方和（Sum of the Squared Error,SSE）作为聚类的目标函数，两次运行K均值产生的两个不同的簇集，我们更喜欢SSE最小的那个。



参考：

[深入理解K-Means聚类算法](https://blog.csdn.net/taoyanqi8932/article/details/53727841)



## 2.17 逻辑回归中Sigmoid的好处

1.广义模型推导所得
2.满足统计的最大熵模型
3.性质优秀，方便使用（Sigmoid函数是平滑的，而且任意阶可导，一阶二阶导数可以直接由函数值得到不用进行求导，这在实现中很实用）

参考：

[为什么逻辑回归 模型要使用 sigmoid 函数](https://blog.csdn.net/weixin_39881922/article/details/80366324)


## 2.18 PCA原理与实现

PCA（Principal Component Analysis）是一种常用的数据分析方法。PCA通过线性变换将原始数据变换为一组各维度线性无关的表示，可用于提取数据的主要特征分量，常用于高维数据的降维。网上关于PCA的文章有很多，但是大多数只描述了PCA的分析过程，而没有讲述其中的原理。这篇文章的目的是介绍PCA的基本数学原理，帮助读者了解PCA的工作机制是什么。

当然我并不打算把文章写成纯数学文章，而是希望用直观和易懂的方式叙述PCA的数学原理，所以整个文章不会引入严格的数学推导。希望读者在看完这篇文章后能更好的明白PCA的工作原理。




**参考**


[PCA的数学原理](https://www.cnblogs.com/mikewolf2002/p/3429711.html)

[PCA的数学原理(转)](https://zhuanlan.zhihu.com/p/21580949)


## 2.19 LR与SVM的异同

**相同点**

第一，LR和SVM都是分类算法。

看到这里很多人就不会认同了，因为在很大一部分人眼里，LR是回归算法。我是非常不赞同这一点的，因为我认为判断一个算法是分类还是回归算法的唯一标准就是样本label的类型，如果label是离散的，就是分类算法，如果label是连续的，就是回归算法。很明显，LR的训练数据的label是“0或者1”，当然是分类算法。其实这样不重要啦，暂且迁就我认为它是分类算法吧，再说了，SVM也可以回归用呢。

第二，如果不考虑核函数，LR和SVM都是线性分类算法，也就是说他们的分类决策面都是线性的。

这里要先说明一点，那就是LR也是可以用核函数的，至于为什么通常在SVM中运用核函数而不在LR中运用，后面讲到他们之间区别的时候会重点分析。总之，原始的LR和SVM都是线性分类器，这也是为什么通常没人问你决策树和LR什么区别，决策树和SVM什么区别，你说一个非线性分类器和一个线性分类器有什么区别？

第三，LR和SVM都是监督学习算法。

这个就不赘述什么是监督学习，什么是半监督学习，什么是非监督学习了。

第四，LR和SVM都是判别模型。

判别模型会生成一个表示P(Y|X)的判别函数（或预测模型），而生成模型先计算联合概率p(Y,X)然后通过贝叶斯公式转化为条件概率。简单来说，在计算判别模型时，不会计算联合概率，而在计算生成模型时，必须先计算联合概率。或者这样理解：生成算法尝试去找到底这个数据是怎么生成的（产生的），然后再对一个信号进行分类。基于你的生成假设，那么那个类别最有可能产生这个信号，这个信号就属于那个类别。判别模型不关心数据是怎么生成的，它只关心信号之间的差别，然后用差别来简单对给定的一个信号进行分类。常见的判别模型有：KNN、SVM、LR，常见的生成模型有：朴素贝叶斯，隐马尔可夫模型。当然，这也是为什么很少有人问你朴素贝叶斯和LR以及朴素贝叶斯和SVM有什么区别（哈哈，废话是不是太多）。


**不同点**

第一，本质上是其损失函数（loss function）不同。

注：lr的损失函数是 cross entropy loss， adaboost的损失函数是 expotional loss ,svm是hinge loss，常见的回归模型通常用 均方误差 loss。

逻辑回归的损失函数

![](http://s10.sinaimg.cn/mw690/002n6ruKgy6WWsUQfxf29)


SVM的目标函数

![](http://s4.sinaimg.cn/mw690/002n6ruKgy6WWtjCmm793)


不同的loss function代表了不同的假设前提，也就代表了不同的分类原理，也就代表了一切！！！简单来说，​逻辑回归方法基于概率理论，假设样本为1的概率可以用sigmoid函数来表示，然后通过极大似然估计的方法估计出参数的值，具体细节参考[逻辑回归](http://blog.csdn.net/pakko/article/details/37878837)。支持向量机​基于几何间隔最大化原理，认为存在最大几何间隔的分类面为最优分类面，具体细节参考[支持向量机通俗导论（理解SVM的三层境界）](http://blog.csdn.net/macyang/article/details/38782399)


第二，支持向量机只考虑局部的边界线附近的点，而逻辑回归考虑全局（远离的点对边界线的确定也起作用）。

当​你读完上面两个网址的内容，深入了解了LR和SVM的原理过后，会发现影响SVM决策面的样本点只有少数的结构支持向量，当在支持向量外添加或减少任何样本点对分类决策面没有任何影响；而在LR中，每个样本点都会影响决策面的结果。用下图进行说明：


支持向量机改变非支持向量样本并不会引起决策面的变化

![](http://s1.sinaimg.cn/mw690/002n6ruKgy6WWvMHbGgb0)


逻辑回归中改变任何样本都会引起决策面的变化

![](http://s5.sinaimg.cn/mw690/002n6ruKgy6WWw74KqM04)


理解了这一点，有可能你会问，然后呢？有什么用呢？有什么意义吗？对使用两种算法有什么帮助么？一句话回答：

因为上面的原因，得知：线性SVM不直接依赖于数据分布，分类平面不受一类点影响；LR则受所有数据点的影响，如果数据不同类别strongly unbalance，一般需要先对数据做balancing。​（引自http://www.zhihu.com/question/26768865/answer/34078149）


第三，在解决非线性问题时，支持向量机采用核函数的机制，而LR通常不采用核函数的方法。

​这个问题理解起来非常简单。分类模型的结果就是计算决策面，模型训练的过程就是决策面的计算过程。通过上面的第二点不同点可以了解，在计算决策面时，SVM算法里只有少数几个代表支持向量的样本参与了计算，也就是只有少数几个样本需要参与核计算（即kernal machine解的系数是稀疏的）。然而，LR算法里，每个样本点都必须参与决策面的计算过程，也就是说，假设我们在LR里也运用核函数的原理，那么每个样本点都必须参与核计算，这带来的计算复杂度是相当高的。所以，在具体应用时，LR很少运用核函数机制。​

第四，​线性SVM依赖数据表达的距离测度，所以需要对数据先做normalization，LR不受其影响。（引自http://www.zhihu.com/question/26768865/answer/34078149）

一个机遇概率，一个机遇距离！​

第五，SVM的损失函数就自带正则！！！（损失函数中的1/2||w||^2项），这就是为什么SVM是结构风险最小化算法的原因！！！而LR必须另外在损失函数上添加正则项！！！

以前一直不理解为什么SVM叫做结构风险最小化算法，**所谓结构风险最小化，意思就是在训练误差和模型复杂度之间寻求平衡，防止过拟合，从而达到真实误差的最小化**。未达到结构风险最小化的目的，最常用的方法就是添加正则项，后面的博客我会具体分析各种正则因子的不同，这里就不扯远了。但是，你发现没，SVM的目标函数里居然自带正则项！！！再看一下上面提到过的SVM目标函数：

SVM目标函数

![](http://s9.sinaimg.cn/mw690/002n6ruKgy6WWxdRoxy08)

有木有，那不就是L2正则项吗？

不用多说了，如果不明白看看L1正则与L2正则吧，参考http://www.mamicode.com/info-detail-517504.html


http://www.zhihu.com/question/26768865/answer/34078149
 

**快速理解LR和SVM的区别**

两种方法都是常见的分类算法，从目标函数来看，区别在于逻辑回归采用的是logistical loss，svm采用的是hinge loss。这两个损失函数的目的都是增加对分类影响较大的数据点的权重，减少与分类关系较小的数据点的权重。SVM的处理方法是只考虑support vectors，也就是和分类最相关的少数点，去学习分类器。而逻辑回归通过非线性映射，大大减小了离分类平面较远的点的权重，相对提升了与分类最相关的数据点的权重。两者的根本目的都是一样的。此外，根据需要，两个方法都可以增加不同的正则化项，如l1,l2等等。所以在很多实验中，两种算法的结果是很接近的。但是逻辑回归相对来说模型更简单，好理解，实现起来，特别是大规模线性分类时比较方便。而SVM的理解和优化相对来说复杂一些。但是SVM的理论基础更加牢固，有一套结构化风险最小化的理论基础，虽然一般使用的人不太会去关注。还有很重要的一点，SVM转化为对偶问题后，分类只需要计算与少数几个支持向量的距离，这个在进行复杂核函数计算时优势很明显，能够大大简化模型和计算量。

作者：orangeprince
链接：https://www.zhihu.com/question/21704547/answer/20293255
来源：知乎
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。



**SVM与LR的区别与联系**

联系：（1）分类（二分类） （2）可加入正则化项 

区别：（1）LR–参数模型；SVM–非参数模型？（2）目标函数：LR—logistical loss；SVM–hinge loss （3）SVM–support vectors；LR–减少较远点的权重 （4）LR–模型简单，好理解，精度低，可能局部最优；SVM–理解、优化复杂，精度高，全局最优，转化为对偶问题—>简化模型和计算 （5）LR可以做的SVM可以做（线性可分），SVM能做的LR不一定能做（线性不可分）




**总结一下**


- Linear SVM和LR都是线性分类器
- Linear SVM不直接依赖数据分布，分类平面不受一类点影响；LR则受所有数据点的影响，如果数据不同类别strongly unbalance，一般需要对数据先做balancing。
- Linear SVM依赖数据表打对距离测度，所以需要对数据先做normalization；LR不受影响
- Linear SVM依赖penalty的系数，实验中需要做validation
- Linear SVM的LR的performance都会收到outlier的影响，就敏感程度而言，无法给出明确结论。


**参考**

[LR与SVM的异同](https://www.cnblogs.com/zhizhan/p/5038747.html)

[SVM和logistic回归分别在什么情况下使用？](https://www.zhihu.com/question/21704547/answer/19059732)

[Linear SVM 和 LR 有什么异同？](https://www.zhihu.com/question/26768865/answer/34078149)


## 2.21 AlexNet

- 使用ReLU激活函数

- Dropout

- 数据增广

先给出AlexNet的一些参数和结构图： 

卷积层：5层 

全连接层：3层 

深度：8层 

参数个数：60M 

神经元个数：650k 

分类数目：1000类


参考

[AlexNet](https://dgschwend.github.io/netscope/#/preset/alexnet)


## 2.22 VGG

**《Very Deep Convolutional Networks for Large-Scale Image Recognition》**

- arXiv：https://arxiv.org/abs/1409.1556
- intro：ICLR 2015
- homepage：http://www.robots.ox.ac.uk/~vgg/research/very_deep/

[VGG](https://arxiv.org/abs/1409.1556)是Oxford的**V**isual **G**eometry **G**roup的组提出的（大家应该能看出VGG名字的由来了）。该网络是在ILSVRC 2014上的相关工作，主要工作是证明了增加网络的深度能够在一定程度上影响网络最终的性能。VGG有两种结构，分别是VGG16和VGG19，两者并没有本质上的区别，只是网络深度不一样。

VGG16相比AlexNet的一个改进是采用**连续的几个3x3的卷积核代替AlexNet中的较大卷积核（11x11，7x7，5x5）**。对于给定的感受野（与输出有关的输入图片的局部大小），**采用堆积的小卷积核是优于采用大的卷积核，因为多层非线性层可以增加网络深度来保证学习更复杂的模式，而且代价还比较小（参数更少）**。

简单来说，在VGG中，使用了3个3x3卷积核来代替7x7卷积核，使用了2个3x3卷积核来代替5*5卷积核，这样做的主要目的是在保证具有相同感知野的条件下，提升了网络的深度，在一定程度上提升了神经网络的效果。

比如，3个步长为1的3x3卷积核的一层层叠加作用可看成一个大小为7的感受野（其实就表示3个3x3连续卷积相当于一个7x7卷积），其参数总量为 3x(9xC^2) ，如果直接使用7x7卷积核，其参数总量为 49xC^2 ，这里 C 指的是输入和输出的通道数。很明显，27xC^2小于49xC^2，即减少了参数；而且3x3卷积核有利于更好地保持图像性质。

这里解释一下为什么使用2个3x3卷积核可以来代替5*5卷积核：

5x5卷积看做一个小的全连接网络在5x5区域滑动，我们可以先用一个3x3的卷积滤波器卷积，然后再用一个全连接层连接这个3x3卷积输出，这个全连接层我们也可以看做一个3x3卷积层。这样我们就可以用两个3x3卷积级联（叠加）起来代替一个 5x5卷积。

具体如下图所示：

![617848-20170902180141093-554602793.png](https://note.youdao.com/yws/res/72654/WEBRESOURCEfdcbcb399f251f9b7c82599cbcdefcc4)

至于为什么使用3个3x3卷积核可以来代替7*7卷积核，推导过程与上述类似，大家可以自行绘图理解。


下面是VGG网络的结构（VGG16和VGG19都在）：

![VGG](https://d2mxuefqeaa7sj.cloudfront.net/s_8C760A111A4204FB24FFC30E04E069BD755C4EEFD62ACBA4B54BBA2A78E13E8C_1491022251600_VGGNet.png)


- VGG16包含了16个隐藏层（13个卷积层和3个全连接层），如上图中的D列所示
- VGG19包含了19个隐藏层（16个卷积层和3个全连接层），如上图中的E列所示

VGG网络的结构非常一致，从头到尾全部使用的是3x3的卷积和2x2的max pooling。

如果你想看到更加形象化的VGG网络，可以使用[经典卷积神经网络（CNN）结构可视化工具](https://mp.weixin.qq.com/s/gktWxh1p2rR2Jz-A7rs_UQ)来查看高清无码的[VGG网络](https://dgschwend.github.io/netscope/#/preset/vgg-16)。


**VGG优点：**

- VGGNet的结构非常简洁，整个网络都使用了同样大小的卷积核尺寸（3x3）和最大池化尺寸（2x2）。

- 几个小滤波器（3x3）卷积层的组合比一个大滤波器（5x5或7x7）卷积层好：

- 验证了通过不断加深网络结构可以提升性能。

**VGG缺点**：

VGG耗费更多计算资源，并且使用了更多的参数（这里不是3x3卷积的锅），导致更多的内存占用（140M）。其中绝大多数的参数都是来自于第一个全连接层。VGG可是有3个全连接层啊！

PS：有的文章称：发现这些全连接层即使被去除，对于性能也没有什么影响，这样就显著降低了参数数量。

注：很多pretrained的方法就是使用VGG的model（主要是16和19），VGG相对其他的方法，参数空间很大，最终的model有500多m，AlexNet只有200m，GoogLeNet更少，所以train一个vgg模型通常要花费更长的时间，所幸有公开的pretrained model让我们很方便的使用。


关于感受野：

假设你一层一层地重叠了3个3x3的卷积层（层与层之间有非线性激活函数）。在这个排列下，第一个卷积层中的每个神经元都对输入数据体有一个3x3的视野。


**代码篇：VGG训练与测试**

这里推荐两个开源库，训练请参考[tensorflow-vgg](https://github.com/machrisaa/tensorflow-vgg)，快速测试请参考[VGG-in TensorFlow](https://www.cs.toronto.edu/~frossard/post/vgg16/)。

代码我就不介绍了，其实跟上述内容一致，跟着原理看code应该会很快。我快速跑了一下[VGG-in TensorFlow](https://www.cs.toronto.edu/~frossard/post/vgg16/)，代码亲测可用，效果很nice，就是model下载比较烦。

贴心的Amusi已经为你准备好了[VGG-in TensorFlow](https://www.cs.toronto.edu/~frossard/post/vgg16/)的测试代码、model和图像。需要的同学可以关注CVer微信公众号，后台回复：VGG。

天道酬勤，还有很多知识要学，想想都刺激~Fighting！

参考：

[《Very Deep Convolutional Networks for Large-Scale Image Recognition》](https://arxiv.org/abs/1409.1556)

[深度网络VGG理解](https://blog.csdn.net/wcy12341189/article/details/56281618)

[深度学习经典卷积神经网络之VGGNet](https://blog.csdn.net/marsjhao/article/details/72955935)

[VGG16 结构可视化](https://dgschwend.github.io/netscope/#/preset/vgg-16)


[tensorflow-vgg](https://github.com/machrisaa/tensorflow-vgg)

[VGG-in TensorFlow](https://www.cs.toronto.edu/~frossard/post/vgg16/)

[机器学习进阶笔记之五 | 深入理解VGG\Residual Network](https://zhuanlan.zhihu.com/p/23518167)

## 2.23 ResNet


**1.ResNet意义**

随着网络的加深，出现了训练集准确率下降的现象，我们可以确定这不是由于Overfit过拟合造成的(过拟合的情况训练集应该准确率很高)；所以作者针对这个问题提出了一种全新的网络，叫深度残差网络，它允许网络尽可能的加深，其中引入了全新的结构如图1； 
这里问大家一个问题 

残差指的是什么？ 

其中ResNet提出了两种mapping：一种是identity mapping，指的就是图1中”弯弯的曲线”，另一种residual mapping，指的就是除了”弯弯的曲线“那部分，所以最后的输出是 y=F(x)+x 
identity mapping顾名思义，就是指本身，也就是公式中的x，而residual mapping指的是“差”，也就是y−x，所以残差指的就是F(x)部分。 

为什么ResNet可以解决“随着网络加深，准确率不下降”的问题？ 


理论上，对于“随着网络加深，准确率下降”的问题，Resnet提供了两种选择方式，也就是identity mapping和residual mapping，如果网络已经到达最优，继续加深网络，residual mapping将被push为0，只剩下identity mapping，这样理论上网络一直处于最优状态了，网络的性能也就不会随着深度增加而降低了。


**2.ResNet结构**

它使用了一种连接方式叫做“shortcut connection”，顾名思义，shortcut就是“抄近道”的意思，看下图我们就能大致理解： 


参考：

[ResNet解析](https://blog.csdn.net/lanran2/article/details/79057994)

[ResNet论文笔记](https://blog.csdn.net/wspba/article/details/56019373)

[残差网络ResNet笔记](https://www.jianshu.com/p/e58437f39f65)

[Understand Deep Residual Networks — a simple, modular learning framework that has redefined state-of-the-art](https://blog.waya.ai/deep-residual-learning-9610bb62c355)


[An Overview of ResNet and its Variants](https://towardsdatascience.com/an-overview-of-resnet-and-its-variants-5281e2f56035)     
 - [译文](https://www.jianshu.com/p/46d76bd56766)

[Understanding and Implementing Architectures of ResNet and ResNeXt for state-of-the-art Image Classification: From Microsoft to Facebook [Part 1]](https://medium.com/@14prakash/understanding-and-implementing-architectures-of-resnet-and-resnext-for-state-of-the-art-image-cf51669e1624)

[给妹纸的深度学习教学(4)——同Residual玩耍](https://zhuanlan.zhihu.com/p/28413039)


## 2.24 YOLO系列

**YOLOv1**

(1) 给个一个输入图像，首先将图像划分成7 * 7的网格。

(2) 对于每个网格，每个网格预测2个bouding box（每个box包含5个预测量）以及20个类别概率，总共输出7×7×（2*5+20）=1470个tensor

(3) 根据上一步可以预测出7 * 7 * 2 = 98个目标窗口，然后根据阈值去除可能性比较低的目标窗口，再由NMS去除冗余窗口即可。

YOLOv1使用了end-to-end的回归方法，没有region proposal步骤，直接回归便完成了位置和类别的判定。种种原因使得YOLOv1在目标定位上不那么精准，直接导致YOLO的检测精度并不是很高。


**YOLOv2**

看YOLOv2到底用了多少技巧，以及这些技巧起了多少作用：

Batch Normalization

CNN在训练过程中网络每层输入的分布一直在改变, 会使训练过程难度加大，但可以通过normalize每层的输入解决这个问题。新的YOLO网络在每一个卷积层后添加batch normalization，通过这一方法，mAP获得了2%的提升。batch normalization 也有助于规范化模型，可以在舍弃dropout优化后依然不会过拟合。

High Resolution Classifier

目前的目标检测方法中，基本上都会使用ImageNet预训练过的模型（classifier）来提取特征，如果用的是AlexNet网络，那么输入图片会被resize到不足256 * 256，导致分辨率不够高，给检测带来困难。为此，新的YOLO网络把分辨率直接提升到了448 * 448，这也意味之原有的网络模型必须进行某种调整以适应新的分辨率输入。

对于YOLOv2，作者首先对分类网络（自定义的darknet）进行了fine tune，分辨率改成448 * 448，在ImageNet数据集上训练10轮（10 epochs），训练后的网络就可以适应高分辨率的输入了。然后，作者对检测网络部分（也就是后半部分）也进行fine tune。这样通过提升输入的分辨率，mAP获得了4%的提升。

Convolutional With Anchor Boxes

之前的YOLO利用全连接层的数据完成边框的预测，导致丢失较多的空间信息，定位不准。作者在这一版本中借鉴了Faster R-CNN中的anchor思想，回顾一下，anchor是RNP网络中的一个关键步骤，说的是在卷积特征图上进行滑窗操作，每一个中心可以预测9种不同大小的建议框。看到YOLOv2的这一借鉴，我只能说SSD的作者是有先见之明的。



为了引入anchor boxes来预测bounding

boxes，作者在网络中果断去掉了全连接层。剩下的具体怎么操作呢？首先，作者去掉了后面的一个池化层以确保输出的卷积特征图有更高的分辨率。然后，通过缩减网络，让图片输入分辨率为416 * 416，这一步的目的是为了让后面产生的卷积特征图宽高都为奇数，这样就可以产生一个center cell。作者观察到，大物体通常占据了图像的中间位置， 就可以只用中心的一个cell来预测这些物体的位置，否则就要用中间的4个cell来进行预测，这个技巧可稍稍提升效率。最后，YOLOv2使用了卷积层降采样（factor为32），使得输入卷积网络的416 * 416图片最终得到13 * 13的卷积特征图（416/32=13）。

加入了anchor boxes后，可以预料到的结果是召回率上升，准确率下降。我们来计算一下，假设每个cell预测9个建议框，那么总共会预测13 * 13 * 9 = 1521个boxes，而之前的网络仅仅预测7 * 7 * 2 = 98个boxes。具体数据为：没有anchor boxes，模型recall为81%，mAP为69.5%；加入anchor boxes，模型recall为88%，mAP为69.2%。这样看来，准确率只有小幅度的下降，而召回率则提升了7%，说明可以通过进一步的工作来加强准确率，的确有改进空间。



Dimension Clusters（维度聚类）

作者在使用anchor的时候遇到了两个问题，第一个是anchor boxes的宽高维度往往是精选的先验框（hand-picked priors），虽说在训练过程中网络也会学习调整boxes的宽高维度，最终得到准确的bounding boxes。但是，如果一开始就选择了更好的、更有代表性的先验boxes维度，那么网络就更容易学到准确的预测位置。

和以前的精选boxes维度不同，作者使用了K-means聚类方法类训练bounding boxes，可以自动找到更好的boxes宽高维度。传统的K-means聚类方法使用的是欧氏距离函数，也就意味着较大的boxes会比较小的boxes产生更多的error，聚类结果可能会偏离。为此，作者采用的评判标准是IOU得分（也就是boxes之间的交集除以并集），这样的话，error就和box的尺度无关了，最终的距离函数为：


参考：

[YOLOv2 论文笔记](https://blog.csdn.net/jesse_mx/article/details/53925356)

[目标检测网络之 YOLOv3](https://www.cnblogs.com/makefile/p/YOLOv3.html)



## 2.25 R-CNN系列

参考

[浅谈RCNN、SPP-net、Fast-Rcnn、Faster-Rcnn](https://blog.csdn.net/sunpeng19960715/article/details/54891652)

[From R-CNN to Faster R-CNN: The Evolution of Object Detection Technology](https://dzone.com/articles/from-r-cnn-to-faster-r-cnn-the-evolution-of-object)

[目标检测技术演化：从R-CNN到Faster R-CNN](https://zhuanlan.zhihu.com/p/40679183)

[Faster R-CNN 源码解析（Tensorflow版）](https://blog.csdn.net/u012457308/article/details/79566195)


## 2.26 FCN

一句话概括就是：FCN将传统网络后面的全连接层换成了卷积层，这样网络输出不再是类别而是 heatmap；同时为了解决因为卷积和池化对图像尺寸的影响，提出使用上采样的方式恢复。


作者的FCN主要使用了三种技术：

- 卷积化（Convolutional）


- 上采样（Upsample）

- 跳跃结构（Skip Layer）


卷积化

卷积化即是将普通的分类网络，比如VGG16，ResNet50/101等网络丢弃全连接层，换上对应的卷积层即可。

上采样

此处的上采样即是反卷积（Deconvolution）。当然关于这个名字不同框架不同，Caffe和Kera里叫Deconvolution，而tensorflow里叫conv_transpose。CS231n这门课中说，叫conv_transpose更为合适。

众所诸知，普通的池化（为什么这儿是普通的池化请看后文）会缩小图片的尺寸，比如VGG16 五次池化后图片被缩小了32倍。为了得到和原图等大的分割图，我们需要上采样/反卷积。

反卷积和卷积类似，都是相乘相加的运算。只不过后者是多对一，前者是一对多。而反卷积的前向和后向传播，只用颠倒卷积的前后向传播即可。所以无论优化还是后向传播算法都是没有问题。


跳跃结构（Skip Layers）

（这个奇怪的名字是我翻译的，好像一般叫忽略连接结构）这个结构的作用就在于优化结果，因为如果将全卷积之后的结果直接上采样得到的结果是很粗糙的，所以作者将不同池化层的结果进行上采样之后来优化输出。

上采样获得与输入一样的尺寸
文章采用的网络经过5次卷积+池化后，图像尺寸依次缩小了 2、4、8、16、32倍，对最后一层做32倍上采样，就可以得到与原图一样的大小

作者发现，仅对第5层做32倍反卷积（deconvolution），得到的结果不太精确。于是将第 4 层和第 3 层的输出也依次反卷积（图５）




参考：

[【总结】图像语义分割之FCN和CRF](https://zhuanlan.zhihu.com/p/22308032)

[图像语义分割（1）- FCN](https://blog.csdn.net/zizi7/article/details/77093447)

[全卷积网络 FCN 详解](https://www.cnblogs.com/gujianhan/p/6030639.html)



## 2.27 U-Net

本文介绍一种编码器-解码器结构。编码器逐渐减少池化层的空间维度，解码器逐步修复物体的细节和空间维度。编码器和解码器之间通常存在快捷连接，因此能帮助解码器更好地修复目标的细节。U-Net 是这种方法中最常用的结构。


fcn(fully convolutional natwork)的思想是：修改一个普通的逐层收缩的网络，用上采样(up sampling)(？？反卷积)操作代替网络后部的池化(pooling)操作。因此，这些层增加了输出的分辨率。为了使用局部的信息，在网络收缩过程（路径）中产生的高分辨率特征(high resolution features) ，被连接到了修改后网络的上采样的结果上。在此之后，一个卷积层基于这些信息综合得到更精确的结果。

与fcn(fully convolutional natwork)不同的是，我们的网络在上采样部分依然有大量的特征通道(feature channels)，这使得网络可以将环境信息向更高的分辨率层(higher resolution layers)传播。结果是，扩张路径基本对称于收缩路径。网络不存在任何全连接层(fully connected layers)，并且，只使用每个卷积的有效部分，例如，分割图(segmentation map)只包含这样一些像素点，这些像素点的完整上下文都出现在输入图像中。为了预测图像边界区域的像素点，我们采用镜像图像的方式补全缺失的环境像素。这个tiling方法在使用网络分割大图像时是非常有用的，因为如果不这么做，GPU显存会限制图像分辨率。
我们的训练数据太少，因此我们采用弹性形变的方式增加数据。这可以让模型学习得到形变不变性。这对医学图像分割是非常重要的，因为组织的形变是非常常见的情况，并且计算机可以很有效的模拟真实的形变。在[3]中指出了在无监督特征学习中，增加数据以获取不变性的重要性。


参考：


[U-net翻译](https://blog.csdn.net/natsuka/article/details/78565229)

## 2.28 DeepLab


参考：

[Semantic Segmentation --DeepLab(1,2,3)系列总结](https://blog.csdn.net/u011974639/article/details/79148719)



## 2.29 凸优化




## 2.30 Accuracy、Precision、Recall和F1 Score

在学习机器学习、深度学习，甚至做自己项目的时候，经过看到上述名词。然而因为名词容易搞混，所以经常会忘记相关的含义。

这里做一次最全最清晰的介绍，若之后再次忘记相关知识点，本文可以帮助快速回顾。

首先，列出一个清单：

- TP（true positive，真正）: 预测为正，实际为正

- FP（false positive，假正）: 预测为正，实际为负

- TN（true negative，真负）：预测为负，实际为负

- FN（false negative，假负）: 预测为负，实际为正

- ACC（accuracy，准确率）：ACC = (TP+TN)/(TP+TN+FN+FP)

- P（precision精确率、精准率、查准率P = TP/ (TP+FP)

- R（recall，召回率、查全率）： R = TP/ (TP+FN)

- TPR（true positive rate，，真正类率同召回率、查全率）：TPR = TP/ (TP+FN)

    注：Recall = TPR

- FPR（false positive rate，假正类率）：FPR =FP/ (FP+TN)

- F-Score: F-Score = (1+β^2) x (PxR) / (β^2x(P+R)) = 2xTP/(2xTP + FP + FN)

- 当β=1是，F1-score = 2xPxR/(P+R)

- P-R曲线（precision-recall，查准率-查全率曲线）

- ROC曲线（receiver operating characteristic，接收者操作特征曲线）

- AUC（area under curve）值


中文博大精深，为了不搞混，下面统一用英文全称或简称作为名词标识。



正式介绍一下前四个名词：

**True positives（TP，真正）** : 预测为正，实际为正

**True negatives（TN，真负）**：预测为负，实际为负

**False positives（FP，假正**）: 预测为正，实际为负 

**False negatives（FN，假负）**: 预测为负，实际为正


为了更好的理解，这里二元分类问题的例子：


假设，我们要对某一封邮件做出一个判定，判定这封邮件是垃圾邮件、还是这封邮件不是垃圾邮件？

如果判定是垃圾邮件，那就是做出（Positive）的判定； 

如果判定不是垃圾邮件，那就做出（Negative）的判定。

True Positive（TP）意思表示做出Positive的判定，而且判定是正确的。

因此，TP的数值表示正确的Positive判定的个数。 

同理，False Positive（TP）数值表示错误的Positive判定的个数。 

依此，True Negative（TN）数值表示正确的Negative判定个数。 

False Negative（FN）数值表示错误的Negative判定个数。


**TPR、FPR和TNR**

**TPR（true positive rate，真正类率）**

TPR = TP/(TP+FN)

真正类率TPR代表分类器预测的正类中实际正实例占所有正实例的比例。



**FPR（false positive rate，假正类率）**

FPR = FP/(FP+TN)

假正类率FPR代表分类器预测的正类中实际负实例占所有负实例的比例。


**TNR（ture negative rate，真负类率）**

TNR = TN/(FP+TN)

真负类率TNR代表分类器预测的负类中实际负实例占所有负实例的比例。


**Accuracy**

准确率（accuracy，ACC）

ACC = (TP+TN)/(TP+TN+FN+FP)


**Precision & Recall**

[Precision精确率](https://en.wikipedia.org/wiki/Precision_and_recall)：

P = TP/(TP+FP)

表示当前划分到正样本类别中，被正确分类的比例（正确正样本所占比例）。


[Recall召回率](https://en.wikipedia.org/wiki/Precision_and_recall)：

R = TP/(TP+FN)

表示当前划分到正样本类别中，真实正样本占所有正样本的比例。

**F-Score**

F-Score 是精确率Precision和召回率Recall的加权调和平均值。该值是为了综合衡量Precision和Recall而设定的。

F-Score = (1+β^2) x (PxR) / (β^2x(P+R)) = 2xTP/(2xTP + FP + FN)

当β=1时，F1-score = 2xPxR/(P+R)。这时，Precision和Recall都很重要，权重相同。

当有些情况下，我们认为Precision更重要，那就调整β的值小于1；如果我们认为Recall更加重要，那就调整β的值大于1。

一般来说，当F-Score或F1-score较高


**P-R曲线**




**ROC曲线**

横轴：负正类率(false postive rate FPR)

纵轴：真正类率(true postive rate TPR)




![ROC Curve](https://upload-images.jianshu.io/upload_images/2394427-5f11fd1e6af07393?imageMogr2/auto-orient/strip%7CimageView2/2/w/700)


**AUC值**


上面都是理论，看起来很迷糊，这里举个真实应用的实例，加强理解。


对于那些不熟悉的人，我将解释精确度和召回率，对于那些熟悉的人，我将在比较精确召回曲线时解释文献中的一些混淆。


下面从图像分类的角度举个例子：

假设现在有这样一个测试集，测试集中的图片只由大雁和飞机两种图片组成，如下图所示： 

![](https://sanchom.files.wordpress.com/2011/08/collection.png)


假设你的分类系统最终的目的是：能取出测试集中所有飞机的图片，而不是大雁的图片。

现在做如下的定义： 

True positives（TP，真正） : 飞机的图片被正确的识别成了飞机。 

True negatives（TN，真负）: 大雁的图片没有被识别出来，系统正确地认为它们是大雁。 

False positives（FP，假正）: 大雁的图片被错误地识别成了飞机。 

False negatives（FN，假负）: 飞机的图片没有被识别出来，系统错误地认为它们是大雁。






![Precision and recall](https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/Precisionrecall.svg/440px-Precisionrecall.svg.png)






**实战**



```python
'''In binary classification settings'''

######### Create simple data ##########

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
import numpy as np

iris = datasets.load_iris()
X = iris.data
y = iris.target

# Add noisy features
random_state = np.random.RandomState(0)
n_samples, n_features = X.shape
X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

# Limit to the two first classes, and split into training and test
X_train, X_test, y_train, y_test = train_test_split(X[y < 2], y[y < 2],
                                                    test_size=.5,
                                                    random_state=random_state)

# Create a simple classifier
classifier = svm.LinearSVC(random_state=random_state)
classifier.fit(X_train, y_train)
y_score = classifier.decision_function(X_test)


######## Compute the average precision score ######## 

from sklearn.metrics import average_precision_score
average_precision = average_precision_score(y_test, y_score)

print('Average precision-recall score: {0:0.2f}'.format(
      average_precision))
	  

######## Plot the Precision-Recall curve   ######
	  
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

precision, recall, _ = precision_recall_curve(y_test, y_score)

plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='b')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
          average_precision))
plt.show()
```



参考

[Accuracy, Precision, Recall & F1 Score: Interpretation of Performance Measures](http://blog.exsilio.com/all/accuracy-precision-recall-f1-score-interpretation-of-performance-measures/)

[Precision and recall](https://en.wikipedia.org/wiki/Precision_and_recall)

[average precision](https://sanchom.wordpress.com/tag/average-precision/)

[Precision-Recall](http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html)

[【YOLO学习】召回率（Recall），精确率（Precision），平均正确率（Average_precision(AP) ），交除并（Intersection-over-Union（IoU））](https://blog.csdn.net/hysteric314/article/details/54093734)

[https://blog.csdn.net/u014380165/article/details/77493978](Precision，Recall，F1score，Accuracy的理解)


[ROC、Precision、Recall、TPR、FPR理解](https://www.jianshu.com/p/be2e037900a1)

[推荐系统评测指标—准确率(Precision)、召回率(Recall)、F值(F-Measure) ](http://bookshadow.com/weblog/2014/06/10/precision-recall-f-measure/)

[机器学习之分类器性能指标之ROC曲线、AUC值](http://www.cnblogs.com/dlml/p/4403482.html)



## 2.31 LR和Boost区别


## 2.32 正则化方法



## 2.33 什么是参数范数惩罚





## 2.34 AUC和ROC



## 2.38 ROI Pooling、ROI Align和ROI Warping对比


参考：

[Mask-RCNN中的ROIAlign, ROIPooling及ROIWarp对比](https://blog.csdn.net/lanyuxuan100/article/details/71124596)


## 2.39 边框回顾（Bounding-Box Regression）

如下图所示，绿色的框表示真实值Ground Truth, 红色的框为Selective Search提取的候选区域/框Region Proposal。那么即便红色的框被分类器识别为飞机，但是由于红色的框定位不准(IoU<0.5)， 这张图也相当于没有正确的检测出飞机。

![](https://www.julyedu.com/Public/Image/Question/1525499418_635.png)


如果我们能对红色的框进行微调fine-tuning，使得经过微调后的窗口跟Ground Truth 更接近， 这样岂不是定位会更准确。 而Bounding-box regression 就是用来微调这个窗口的。


边框回归是什么？

对于窗口一般使用四维向量(x,y,w,h)(x,y,w,h) 来表示， 分别表示窗口的中心点坐标和宽高。 对于图2, 红色的框 P 代表原始的Proposal, 绿色的框 G 代表目标的 Ground Truth， 我们的目标是寻找一种关系使得输入原始的窗口 P 经过映射得到一个跟真实窗口 G 更接近的回归窗口G^。


![](https://www.julyedu.com/Public/Image/Question/1525499529_241.png)

所以，边框回归的目的即是：给定(Px,Py,Pw,Ph)寻找一种映射f， 使得f(Px,Py,Pw,Ph)=(Gx^,Gy^,Gw^,Gh^)并且(Gx^,Gy^,Gw^,Gh^)≈(Gx,Gy,Gw,Gh)


边框回归怎么做的？

那么经过何种变换才能从图2中的窗口 P 变为窗口G^呢？ 比较简单的思路就是: 平移+尺度放缩

先做平移(Δx,Δy)，Δx=Pwdx(P),Δy=Phdy(P)这是R-CNN论文的：
G^x=Pwdx(P)+Px,(1)
G^y=Phdy(P)+Py,(2)

然后再做尺度缩放(Sw,Sh), Sw=exp(dw(P)),Sh=exp(dh(P)),对应论文中：
G^w=Pwexp(dw(P)),(3)
G^h=Phexp(dh(P)),(4)

观察(1)-(4)我们发现， 边框回归学习就是dx(P),dy(P),dw(P),dh(P)这四个变换。

下一步就是设计算法那得到这四个映射。

线性回归就是给定输入的特征向量 X, 学习一组参数 W, 使得经过线性回归后的值跟真实值 Y(Ground Truth)非常接近. 即Y≈WX。 那么 Bounding-box 中我们的输入以及输出分别是什么呢？

Input:
RegionProposal→P=(Px,Py,Pw,Ph)这个是什么？ 输入就是这四个数值吗？其实真正的输入是这个窗口对应的 CNN 特征，也就是 R-CNN 中的 Pool5 feature（特征向量）。 (注：训练阶段输入还包括 Ground Truth， 也就是下边提到的t∗=(tx,ty,tw,th))

Output:
需要进行的平移变换和尺度缩放 dx(P),dy(P),dw(P),dh(P)，或者说是Δx,Δy,Sw,Sh。我们的最终输出不应该是 Ground Truth 吗？ 是的， 但是有了这四个变换我们就可以直接得到 Ground Truth。

这里还有个问题， 根据(1)~(4)我们可以知道， P 经过 dx(P),dy(P),dw(P),dh(P)得到的并不是真实值 G，而是预测值G^。的确，这四个值应该是经过 Ground Truth 和 Proposal 计算得到的真正需要的平移量(tx,ty)和尺度缩放(tw,th)。 

这也就是 R-CNN 中的(6)~(9)： 
tx=(Gx−Px)/Pw,(6)

ty=(Gy−Py)/Ph,(7)

tw=log(Gw/Pw),(8)

th=log(Gh/Ph),(9)

那么目标函数可以表示为 d∗(P)=wT∗Φ5(P)，Φ5(P)是输入 Proposal 的特征向量，w∗是要学习的参数（*表示 x,y,w,h， 也就是每一个变换对应一个目标函数） , d∗(P) 是得到的预测值。

我们要让预测值跟真实值t∗=(tx,ty,tw,th)差距最小， 得到损失函数为：
Loss=∑iN(ti∗−w^T∗ϕ5(Pi))2

函数优化目标为：

W∗=argminw∗∑iN(ti∗−w^T∗ϕ5(Pi))2+λ||w^∗||2

利用梯度下降法或者最小二乘法就可以得到 w∗。


参考：

[bounding box regression](http://caffecn.cn/?/question/160)

[边框回归(Bounding Box Regression)详解](https://blog.csdn.net/zijin0802034/article/details/77685438)

[什么是边框回归Bounding-Box regression，以及为什么要做、怎么做](https://www.julyedu.com/question/big/kp_id/26/ques_id/2139)



## 2.42 L0、L1和L2范数

最小化误差是为了让我们的模型拟合我们的训练数据，而规则化参数是防止我们的模型过分拟合我们的训练数据。

 
还有几种角度来看待规则化的。规则化符合奥卡姆剃刀(Occam's razor)原理。这名字好霸气，razor！不过它的思想很平易近人：在所有可能选择的模型中，我们应该选择能够很好地解释已知数据并且十分简单的模型。从贝叶斯估计的角度来看，规则化项对应于模型的先验概率。民间还有个说法就是，规则化是结构风险最小化策略的实现，是在经验风险上加一个正则化项(regularizer)或惩罚项(penalty term)。


参考：

[机器学习中的范数规则化之（一）L0、L1与L2范](https://blog.csdn.net/zouxy09/article/details/24971995)



## 2.43 梯度弥散和梯度爆炸


## 2.44 反卷积（deconv）/转置卷积（trans）



## 2.45 空洞卷积（dilated/Atrous conv）


## 2.46 RetinaNet（Focal loss）

《Focal Loss for Dense Object Detection》

- arXiv：https://arxiv.org/abs/1708.02002

- intro：

清华大学孔涛博士在知乎上这么写道：



目标的检测和定位中一个很困难的问题是，如何从数以万计的候选窗口中挑选包含目标物的物体。只有候选窗口足够多，才能保证模型的 Recall。

目前，目标检测框架主要有两种：

一种是 one-stage ，例如 YOLO、SSD 等，这一类方法速度很快，但识别精度没有 two-stage 的高，其中一个很重要的原因是，利用一个分类器很难既把负样本抑制掉，又把目标分类好。

另外一种目标检测框架是 two-stage ，以 Faster RCNN 为代表，这一类方法识别准确度和定位精度都很高，但存在着计算效率低，资源占用大的问题。

Focal Loss 从优化函数的角度上来解决这个问题，实验结果非常 solid，很赞的工作。

何凯明团队提出了用 Focal Loss 函数来训练。



因为，他在训练过程中发现，类别失衡是影响 one-stage 检测器准确度的主要原因。那么，如果能将“类别失衡”这个因素解决掉，one-stage 不就能达到比较高的识别精度了吗？



于是在研究中，何凯明团队采用 Focal Loss 函数来消除“类别失衡”这个主要障碍。



结果怎样呢？



为了评估该损失的有效性，该团队设计并训练了一个简单的密集目标检测器—RetinaNet。试验结果证明，当使用 Focal Loss 训练时，RetinaNet 不仅能赶上 one-stage 检测器的检测速度，而且还在准确度上超越了当前所有最先进的 two-stage 检测器。



**参考**

[如何评价Kaiming的Focal Loss for Dense Object Detection？](https://www.zhihu.com/question/63581984)

[首发 | 何恺明团队提出 Focal Loss，目标检测精度高达39.1AP，打破现有记录](https://zhuanlan.zhihu.com/p/28442066)



## 2.47 FPN 特征金字塔网络


## 2.48 ResNeXt



**参考**

[ResNeXt算法详解](https://blog.csdn.net/u014380165/article/details/71667916)

## 2.49 ResNet v1与ResNet v2的区别

## 2.50 ResNet v2的ReLU激活函数有什么不同

## 2.51 已知5个点可求解一个方程，若此时有100组点，应该怎么解方程

## 2.52 ResNet v2

![resnetv2.png](https://note.youdao.com/yws/res/72646/WEBRESOURCE397a56507b69ae3c4be6815c6f6bdc4a)


**参考**

[《Identity Mappings in Deep Residual Networks》](https://arxiv.org/abs/1603.05027)

[Feature Extractor[ResNet v2]](https://www.cnblogs.com/shouhuxianjian/p/7770658.html)

[ResNetV2：ResNet深度解析](https://blog.csdn.net/lanran2/article/details/80247515)

[ResNet v2论文笔记](https://blog.csdn.net/u014061630/article/details/80558661)

[[ResNet系] 002 ResNet-v2](https://segmentfault.com/a/1190000011228906)


## 2.53 Faster R-CNN的RPN网络


RPN结构说明： 
1) 从基础网络提取的第五卷积层特征进入RPN后分为两个分支，其中一个分支进行针对feature map（上图conv-5-3共有512个feature-map）的每一个位置预测共（9*4=36）个参数，其中9代表的是每一个位置预设的9种形状的anchor-box，4对应的是每一个anchor-box的预测值（该预测值表示的是预设anchor-box到ground-truth-box之间的变换参数），上图中指向rpn-bbox-pred层的箭头上面的数字36即是代表了上述的36个参数，所以rpn-bbox-pred层的feature-map数量是36，而每一张feature-map的形状（大小）实际上跟conv5-3一模一样的；

2) 另一分支预测该anchor-box所框定的区域属于前景和背景的概率（网上很对博客说的是，指代该点属于前景背景的概率，那样是不对的，不然怎么会有18个feature-map输出呢？否则2个就足够了），前景背景的真值给定是根据当前像素（anchor-box中心）是否在ground-truth-box内；

3) 上图RPN-data(python)运算框内所进行的操作是读取图像信息（原始宽高），groun-truth boxes的信息（bounding-box的位置，形状，类别）等，作好相应的转换，输入到下面的层当中。

4) 要注意的是RPN内部有两个loss层，一个是BBox的loss,该loss通过减小ground-truth-box与预测的anchor-box之间的差异来进行参数学习，从而使RPN网络中的权重能够学习到预测box的能力。实现细节是每一个位置的anchor-box与ground-truth里面的box进行比较，选择IOU最大的一个作为该anchor-box的真值，若没有，则将之class设为背景（概率值0，否则1），这样背景的anchor-box的损失函数中每个box乘以其class的概率后就不会对bbox的损失函数造成影响。另一个loss是class-loss,该处的loss是指代的前景背景并不是实际的框中物体类别，它的存在可以使得在最后生成roi时能快速过滤掉预测值是背景的box。也可实现bbox的预测函数不受影响，使得anchor-box能（专注于）正确的学习前景框的预测，正如前所述。所以，综合来讲，整个RPN的作用就是替代了以前的selective-search方法，因为网络内的运算都是可GPU加速的，所以一下子提升了ROI生成的速度。可以将RPN理解为一个预测前景背景，并将前景框定的一个网络，并进行单独的训练，实际上论文里面就有一个分阶段训练的训练策略，实际上就是这个原因。

5) 最后经过非极大值抑制，RPN层产生的输出是一系列的ROI-data，它通过ROI的相对映射关系，将conv5-3中的特征已经存入ROI-data中，以供后面的分类网使用。

另外两个loss层的说明： 
也许你注意到了，最后还有两个loss层，这里的class-loss指代的不再是前景背景loss，而是真正的类别loss了，这个应该就很好理解了。而bbox-loss则是因为rpn提取的只是前景背景的预测，往往很粗糙，这里其实是通过ROI-pooling后加上两层全连接实现更精细的box修正（这里其实是我猜的）。 
ROI-Pooing的作用是为了将不同大小的Roi映射（重采样）成统一的大小输入到全连接层去。

以上。




参考

[Faster-Rcnn中RPN（Region Proposal Network）的理解](https://blog.csdn.net/mllearnertj/article/details/53709766)


## 2.54 TensorFlow


## 2.55 Caffe

## 2.56 PyTorch

## 2.57 MXNet

## 2.58 CCA和PCA的区别


## 2.59 Softmax介绍

## 2.60 Pooling层原理

## 2.61 梯度下降法

## 2.62 mini-batch梯度下降法


## 2.63 随机梯度下降法

## 2.64 动量梯度下降法（momentum）



```math
v_{dW}=\beta v_{dW}+\left ( 1-\beta  \right )dW

v_{db}=\beta v_{db}+\left ( 1-\beta  \right )db

W=W-\alpha v_{dW}, b=b-\alpha \alpha v_{db}
```

超参数：α和β


在此需要引入一个叫做指数加权平均的知识点。可以将之前的dW和db都联系起来，不再是每一次梯度都是独立的情况。其中β是可以自行设置的超参数，一般情况下默认为0.9（也可以设置为其他数值）。


动量梯度下降是同理的，每一次梯度下降都会有一个之前的速度的作用，如果我这次的方向与之前相同，则会因为之前的速度继续加速；如果这次的方向与之前相反，则会由于之前存在速度的作用不会产生一个急转弯，而是尽量把路线向一条直线拉过去。


参考: [简述动量Momentum梯度下降](https://blog.csdn.net/yinruiyang94/article/details/77944338)

## 2.65 RMSprop


```math
S_{dW}=\beta S_{dW}+\left ( 1-\beta  \right )dW^{2}

S_{db}=\beta S_{db}+\left ( 1-\beta  \right )db^{2}

W=W-\alpha\frac{dW}{\sqrt{S_{dW}}}, b=b-\alpha\frac{db}{\sqrt{S_{db}}}
```
超参数：α和β

## 2.66 Adam

Adam算法结合了Momentum和RMSprop梯度下降法，是一种极其常见的学习算法，被证明能有效适用于不同神经网络，适用于广泛的结构。


```math
v_{dW}=\beta_{1} v_{dW}+\left ( 1-\beta_{1}  \right )dW

v_{db}=\beta_{1} v_{db}+\left ( 1-\beta_{1}  \right )db

S_{dW}=\beta_{2} S_{dW}+\left ( 1-\beta_{2}  \right )dW^{2}

S_{db}=\beta_{2} S_{db}+\left ( 1-\beta_{2}  \right )db^{2}


v_{dW}^{corrected}=\frac{v_{dW}}{1-\beta_{1}^{t}}

v_{db}^{corrected}=\frac{v_{db}}{1-\beta_{1}^{t}}

S_{dW}^{corrected}=\frac{S_{dW}}{1-\beta_{2}^{t}}

S_{db}^{corrected}=\frac{S_{db}}{1-\beta_{2}^{t}}

W:=W-\frac{av_{dW}^{corrected}}{\sqrt{S_{dW}^{corrected}}+\varepsilon }

```


超参数：


```math
\alpha ,\beta _{1},\beta_{2},\varepsilon 
```


## 2.67 常见的激活函数及其特点


## 2.68 ReLU及其变体

## 2.69 sigmoid

## 2.70 softmax

## 2.71 交叉熵损失函数

## 2.72 神经网络的深度和宽度作用


## 2.73 网络压缩与量化

参考：

[网络压缩-量化方法对比](https://blog.csdn.net/shuzfan/article/details/51678499)

## 2.74 Caffe新加一层需要哪些操作

## 2.75 depthwise卷积加速比推导

## 2.76 如何加快梯度下降收敛速度

## 2.77 Caffe的depthwise为什么慢，怎么解决

## 2.78 有哪些模型压缩的方法

## 2.79 MobileNetV1和MobileNetV2区别

MobileNetv1：在depthwise separable convolutions（参考Xception）方法的基础上提供了高校模型设计的两个选择：宽度因子（width multiplie）和分辨率因子（resolution multiplier）。深度可分离卷积depthwise separable convolutions（参考Xception）的本质是冗余信息更小的稀疏化表达。

下面介绍两幅Xception中 depthwise separable convolution的图示：


![image.png](http://note.youdao.com/yws/res/72649/WEBRESOURCEbc7faaa873f60130909395af32b84628)

![image.png](http://note.youdao.com/yws/res/72662/WEBRESOURCE820b306a3f2c9db8ce3c12a79e2a1065)


深度可分离卷积的过程是①用16个3×3大小的卷积核（1通道）分别与输入的16通道的数据做卷积（这里使用了16个1通道的卷积核，输入数据的每个通道用1个3×3的卷积核卷积），得到了16个通道的特征图，我们说该步操作是depthwise（逐层）的，在叠加16个特征图之前，②接着用32个1×1大小的卷积核（16通道）在这16个特征图进行卷积运算，将16个通道的信息进行融合（用1×1的卷积进行不同通道间的信息融合），我们说该步操作是pointwise（逐像素）的。这样我们可以算出整个过程使用了3×3×16+（1×1×16）×32 =656个参数。

注：上述描述与标准的卷积非常的不同，第一点在于使用非1x1卷积核时，是单channel的（可以说是1通道），即上一层输出的每个channel都有与之对应的卷积核。而标准的卷积过程，卷积核是多channel的。第二点在于使用1x1卷积核实现多channel的融合，并利用多个1x1卷积核生成多channel。表达的可能不是很清楚，但结合图示其实就容易明白了。

一般卷积核的channel也常称为深度（depth），所以叫做深度可分离，即原来为多channel组合，现在变成了单channel分离。


参考：

[深度解读谷歌MobileNet](https://blog.csdn.net/t800ghb/article/details/78879612)

[对深度可分离卷积、分组卷积、扩张卷积、转置卷积（反卷积）的理解](https://blog.csdn.net/chaolei3/article/details/79374563)

## 2.80 MobileNetv2为什么会加shotcut

## 2.81 小米MACE和腾讯NCNN框架比较

## 2.82 MACE和NCNN加速原理

## 2.83 TensorRT原理

## 2.84 MobileNet-SSD介绍

## 2.85 Adam和SGD实际效果

## 2.86 KCF算法介绍

## 2.87 非极大值抑制NMS

## 2.88 为什么降采用使用max pooling，而分类使用average pooling

## 2.89 max pooling如何反向传播

## 2.90 学习率如何调整

## 2.91 BN为什么可以防止过拟合

## 2.92 给定5个人脸关键点和5个对齐后的点，求怎么变换的？

## 2.93 SVD三个矩阵分解

## 2.94 MobileNetV2创新点

## 2.95 SVD和特征值的关系

## 2.96 LR和线性回归的关系

## 2.97 训练样本数量不均衡时，怎么处理

## 2.98 xgboot原理

## 2.99 boosting和bagging

## 2.100 EM算法

## 2.101 gbdt算法

## 2.102 Weight Normalization（WN）介绍

## 2.103 BN、LN和WN的区别

## 2.104 Layer Normalization

## 2.105 Dropout原理（在训练和测试的区别）


参考：

[理解dropout](https://blog.csdn.net/stdcoutzyx/article/details/49022443)


## 2.106 U-Net和FCN的区别

## 2.107 ROU和AUC

## 2.108 Bounding boxes 回归原理/公式

## 2.109 ResNet/Inception/Xception/DenseNet/ResNeXt/SENet


SENet

论文：《Squeeze-and-Excitation Networks》 

论文链接：https://arxiv.org/abs/1709.01507 

代码地址：https://github.com/hujie-frank/SENet


论文的动机是从特征通道之间的关系入手，希望显式地建模特征通道之间的相互依赖关系。另外，没有引入一个新的空间维度来进行特征通道间的融合，而是采用了一种全新的“特征重标定”策略。具体来说，就是通过学习的方式来自动获取到每个特征通道的重要程度，然后依照这个重要程度去增强有用的特征并抑制对当前任务用处不大的特征，通俗来讲，就是让网络利用全局信息有选择的增强有益feature通道并抑制无用feature通道，从而能实现feature通道自适应校准。 



![Schema of SE-Inception and SE-ResNet modules](http://note.youdao.com/yws/res/72653/WEBRESOURCEfc9e885b47a805d032eb91d783d44998)

参考：


[SENet学习笔记](https://blog.csdn.net/xjz18298268521/article/details/79078551)


## 2.110 深度可分离网络（Depth separable convolution）

## 2.111 如何解决异常值问题

## 2.112 SVM用什么特征 

## 2.113 决策树监枝

## 2.114 集成学习

## 2.115 随机森林

## 2.116 什么时候使用核函数

## 2.117 模型压缩方法

## 2.118 TensorFlow常用的Optimizer

## 2.119 组卷积（group convolution）

在说明分组卷积之前我们用一张图来体会一下一般的卷积操作。 

![常规卷积操作.png](http://note.youdao.com/yws/res/72655/WEBRESOURCEc81d9847cdc7fccb68af382c5d1aedea)

从上图可以看出，一般的卷积会对输入数据的整体一起做卷积操作，即输入数据：H1×W1×C1；而卷积核大小为h1×w1，通道为C1，一共有C2个，然后卷积得到的输出数据就是H2×W2×C2。这里我们假设输出和输出的分辨率是不变的。主要看这个过程是一气呵成的，这对于存储器的容量提出了更高的要求。 

但是分组卷积明显就没有那么多的参数。先用图片直观地感受一下分组卷积的过程。对于上面所说的同样的一个问题，分组卷积就如下图所示。 

![组卷积操作.png](http://note.youdao.com/yws/res/72647/WEBRESOURCE9bd3228ad1fb16ab13c1106ff0bb7a90)

可以看到，图中将输入数据分成了2组（组数为g），需要注意的是，这种分组只是在深度上进行划分，即某几个通道编为一组，这个具体的数量由（C1/g）决定。因为输出数据的改变，相应的，卷积核也需要做出同样的改变。即每组中卷积核的深度也就变成了（C1/g），而卷积核的大小是不需要改变的，此时每组的卷积核的个数就变成了（C2/g）个，而不是原来的C2了。然后用每组的卷积核同它们对应组内的输入数据卷积，得到了输出数据以后，再用concatenate的方式组合起来，最终的输出数据的通道仍旧是C2。也就是说，分组数g决定以后，那么我们将并行的运算g个相同的卷积过程，每个过程里（每组），输入数据为H1×W1×C1/g，卷积核大小为h1×w1×C1/g，一共有C2/g个，输出数据为H2×W2×C2/g。

举个例子：

Group conv本身就极大地减少了参数。比如当输入通道为256，输出通道也为256，kernel size为3×3，不做Group conv参数为256×3×3×256。实施分组卷积时，若group为8，每个group的input channel和output channel均为32，参数为8×32×3×3×32，是原来的八分之一。而Group conv最后每一组输出的feature maps应该是以concatenate的方式组合。 
Alex认为group conv的方式能够增加 filter之间的对角相关性，而且能够减少训练参数，不容易过拟合，这类似于正则的效果。

参考：

[A Tutorial on Filter Groups (Grouped Convolution)](https://blog.yani.io/filter-group-tutorial/)

[深度可分离卷积、分组卷积、扩张卷积、转置卷积（反卷积）的理解](https://blog.csdn.net/chaolei3/article/details/79374563)

## 2.120 交错组卷积（Interleaved group convolutions，IGC）

参考：

[学界 | MSRA王井东详解ICCV 2017入选论文：通用卷积神经网络交错组卷积](https://www.sohu.com/a/161110049_465975)

[视频：基于交错组卷积的高效深度神经网络](https://edu.csdn.net/course/play/8320/171433?s=1)


## 2.121 空洞/扩张卷积（Dilated/Atrous Convolution）

Dilated convolution/Atrous convolution可以叫空洞卷积或者扩张卷积。

背景：语义分割中pooling 和 up-sampling layer层。pooling会降低图像尺寸的同时增大感受野，而up-sampling操作扩大图像尺寸，这样虽然恢复了大小，但很多细节被池化操作丢失了。

需求：能不能设计一种新的操作，不通过pooling也能有较大的感受野看到更多的信息呢？

目的：替代pooling和up-sampling运算，既增大感受野又不减小图像大小。

简述：在标准的 convolution map 里注入空洞，以此来增加 reception field。相比原来的正常convolution，dilated convolution 多了一个 hyper-parameter 称之为 dilation rate 指的是kernel的间隔数量(e.g. 正常的 convolution 是 dilatation rate 1)。


空洞卷积诞生于图像分割领域，图像输入到网络中经过CNN提取特征，再经过pooling降低图像尺度的同时增大感受野。由于图像分割是pixel−wise预测输出，所以还需要通过upsampling将变小的图像恢复到原始大小。upsampling通常是通过deconv(转置卷积)完成。因此图像分割FCN有两个关键步骤：池化操作增大感受野，upsampling操作扩大图像尺寸。这儿有个问题，就是虽然图像经过upsampling操作恢复了大小，但是很多细节还是被池化操作丢失了。那么有没有办法既增大了感受野又不减小图像大小呢？Dilated conv横空出世。


![image.png](http://note.youdao.com/yws/res/72650/WEBRESOURCE4e67844a80906171dea0c57ae1d79c2c)

注意事项：

1.为什么不直接使用5x5或者7x7的卷积核？这不也增加了感受野么？

答：增大卷积核能增大感受野，但是只是线性增长，参考答案里的那个公式，(kernel-1)*layer，并不能达到空洞卷积的指数增长。


2.2-dilated要在1-dilated的基础上才能达到7的感受野（如上图a、b所示）


关于空洞卷积的另一种概括：

Dilated Convolution问题的引出，是因为down-sample之后的为了让input和output的尺寸一致。我们需要up-sample，但是up-sample会丢失信息。如果不采用pooling，就无需下采样和上采样步骤了。但是这样会导致kernel 的感受野变小，导致预测不精确。。如果采用大的kernel话，一来训练的参数变大。二来没有小的kernel叠加的正则作用，所以kernel size变大行不通。

由此Dilated Convolution是在不改变kernel size的条件下，增大感受野。


参考：

[《Multi-Scale Context Aggregation by Dilated Convolutions》](https://arxiv.org/abs/1511.07122) 

[《Rethinking Atrous Convolution for Semantic Image Segmentation》](https://arxiv.org/abs/1706.05587)

[如何理解空洞卷积（dilated convolution）？](https://www.zhihu.com/question/54149221)

[Dilated/Atrous conv 空洞卷积/多孔卷积](https://blog.csdn.net/silence2015/article/details/79748729)

[Multi-Scale Context Aggregation by Dilated Convolution 对空洞卷积（扩张卷积）、感受野的理解](https://blog.csdn.net/guvcolie/article/details/77884530?locationNum=10&fps=1)

[对深度可分离卷积、分组卷积、扩张卷积、转置卷积（反卷积）的理解](https://blog.csdn.net/chaolei3/article/details/79374563)


[tf.nn.atrous_conv2d](https://tensorflow.google.cn/api_docs/python/tf/nn/atrous_conv2d)


## 2.122 转置卷积(Transposed Convolutions/deconvlution)


转置卷积（transposed Convolutions）又名反卷积（deconvolution）或是分数步长卷积（fractially straced convolutions）。反卷积（Transposed Convolution, Fractionally Strided Convolution or Deconvolution）的概念第一次出现是Zeiler在2010年发表的论文Deconvolutional networks中。

[对深度可分离卷积、分组卷积、扩张卷积、转置卷积（反卷积）的理解](https://blog.csdn.net/chaolei3/article/details/79374563)

转置卷积和反卷积的区别

那什么是反卷积？从字面上理解就是卷积的逆过程。值得注意的反卷积虽然存在，但是在深度学习中并不常用。而转置卷积虽然又名反卷积，却不是真正意义上的反卷积。因为根据反卷积的数学含义，通过反卷积可以将通过卷积的输出信号，完全还原输入信号。而事实是，转置卷积只能还原shape大小，而不能还原value。你可以理解成，至少在数值方面上，转置卷积不能实现卷积操作的逆过程。所以说转置卷积与真正的反卷积有点相似，因为两者产生了相同的空间分辨率。但是又名反卷积（deconvolutions）的这种叫法是不合适的，因为它不符合反卷积的概念。

简单来说，转置矩阵就是一种上采样过程。

正常卷积过程如下，利用3x3的卷积核对4x4的输入进行卷积，输出结果为2x2

![卷积过程](https://github.com/vdumoulin/conv_arithmetic/blob/master/gif/no_padding_no_strides.gif?raw=true)

转置卷积过程如下，利用3x3的卷积核对"做了补0"的2x2输入进行卷积，输出结果为4x4。

![转置卷积](https://github.com/vdumoulin/conv_arithmetic/blob/master/gif/no_padding_no_strides_transposed.gif?raw=true)


上述的卷积运算和转置卷积是"尺寸"对应的，卷积的输入大小与转置卷积的输出大小一致，分别可以看成下采样和上采样操作。


参考：

[Transposed Convolution, Fractionally Strided Convolution or Deconvolution](https://buptldy.github.io/2016/10/29/2016-10-29-deconv/)

[深度学习 | 反卷积/转置卷积 的理解 transposed conv/deconv](https://blog.csdn.net/u014722627/article/details/60574260)

## 2.123 ShuffleNet v1&v2

## 2.124 Group Normalization

## 2.125 如何解决正负样本数量不均衡

## 2.126 Focal和OHEM

## 2.127 RefineDet

## 2.128 RetinaNet

## 2.129 Cascade R-CNN

## 2.130 ResNet如何解决梯度消失？

## 2.131 ResNet网络越来越深，准确率会不会提升？

## 2.132 手推SVM

## 2.133 MobileNet为什么快？有多少层？多少参数？

## 2.134 什么是Bottlenet layer？



# 三、数据结构与算法

## 3.1 快速排序（介绍和代码实现）

## 3.2 求二叉树的最大高度


## 3.3 找到链表倒数第k个结点

## 3.4 动态规划

## 3.5 打印螺旋矩阵

## 3.6 翻转链表

## 3.7 找最小字串

## 3.8 二叉排序树/二叉查找树BST



## 3.9 平衡二叉树


## 3.10 2-sum、3-sum和4-sum问题

N-sum就是从序列中找出N个数，使得N个数之和等于指定数值的问题。

**2-sum**

2-sum就是从序列中找出2个数，使得2个数之和等于0的问题，即a+b=sum。

下面的例子是规定指定数值为0：

```cpp
int sum_2()
{
	int res = 0;
	int n = data.size();
	for(int i=0; i<n-1; i++)
	{
		for(int j=i+1; j<n; j++)
		{
			if(data[i] + data[j] == 0)
			{
				res ++;
			}
		}
	}
	return res;
}

```

上述算法的由于包含了两层循环，因此时间复杂度为O(N^2)。

观察发现，上述算法时间主要花费在数据比对，为此可以考虑使用二分查找来减少数据比对时间，要想使用二分查找，首先应该对数据进行排序，在此使用归并排序或者快速排序对数组进行升序排列。排序所花时间为O(NlogN)，排序之后数据查找只需要O(logN)的时间，但是总共需要查找N次，为此改进后算法的时间复杂度为O(NlogN)。


```cpp
int cal_sum_2()
{
	int res = 0;
	for(int i=0; i<data.size(); i++)
	{
		int j = binary_search(-data[i]);
		if(j > i)	
		res++;
	}
	return res;
}

```

**3-sum**

上述2-sum的解题思路适用于3-sum及4-sum问题，如求解a+b+c=0，可将其转换为求解a+b=-c，此就为2-sum问题。


为此将2-sum，3-sum，4-sum的求解方法以及相应的优化方法实现在如下所示的sum类中。

sum.h


```cpp
#ifndef SUM_H
#define SUM_H
#include <vector>
using std::vector;
class sum
{
private:
	vector<int> data;
public:
	sum(){};
	sum(const vector<int>& a);
	~sum(){};
	int cal_sum_2() const;
	int cal_sum_3() const;
	int cal_sum_4() const;
	int cal_sum_2_update() const;
	int cal_sum_3_update() const;
	int cal_sum_3_update2() const;
	int cal_sum_4_update() const;
	void sort(int low, int high);
	void print() const;
	friend int find(const sum& s, int target); 
};
#endif

```

sum.cpp


```cpp
#include "Sum.h"
#include <iostream>
using namespace std;
sum::sum(const vector<int>& a)
{
	data = a;
}
void sum::sort(int low, int high)
{
	if(low >= high)
		return;
	int mid = (low+high)/2;
	sort(low,mid);
	sort(mid+1,high);
	vector<int> temp;
	int l = low;
	int h = mid+1;
	while(l<=mid && h <=high)
	{
		if(data[l] > data[h])
			temp.push_back(data[h++]);
		else
			temp.push_back(data[l++]);
	}
	while(l<=mid)
		temp.push_back(data[l++]);
	while(h<=high)
		temp.push_back(data[h++]);
	for(int i=low; i<=high; i++)
	{
		data[i] = temp[i-low];
	}
}
void sum::print() const
{
	for(int i=0; i<data.size(); i++)
	{
		cout<<data[i]<<" ";
	}
	cout<<endl;
}
int find(const sum& s, int target)
{
	int low = 0;
	int high = s.data.size()-1;
	while(low <= high)
	{
		int mid = (low + high)/2;
		if(s.data[mid] < target)
		{
			low = mid+1;
		}
		else if(s.data[mid] > target)
		{
			high = mid - 1;
		}
		else
		{
			return mid;
		}
	}
	return -1;
}
int sum::cal_sum_2() const
{
	int res = 0;
	for(int i=0; i<data.size(); i++)
	{
		int j = find(*this, -data[i]);
		if(j > i)	
			res++;
	}
	return res;
}
int sum::cal_sum_3() const
{
	int res = 0;
	for(int i=0; i<data.size(); i++)
	{
		for(int j=i+1; j<data.size(); j++)
		{
			for(int p=j+1;p<data.size();p++)
			{
				if(data[i] + data[j] + data[p] == 0)
					res++;
			}
		}
	}
	return res;
}
int sum::cal_sum_4() const
{
	int res = 0;
	for(int i=0; i<data.size(); i++)
	{
		for(int j=i+1; j<data.size(); j++)
		{
			for(int p=j+1; p<data.size(); p++)
			{
				for(int q=p+1; q<data.size(); q++)
				{
					if(data[i]+data[j]+data[p]+data[q] == 0)
						res++;
				}
			}
		}
	}
	return res;
}
int sum::cal_sum_2_update() const
{
	int res = 0;
	for(int i=0,j=data.size()-1; i<j; )
	{
		if(data[i] + data[j] > 0)
			j--;
		else if(data[i] + data[j] < 0)
			i++;
		else
		{
			res++;
			j--;
			i++;
		}
	}
	return res;
}
int sum::cal_sum_3_update() const
{
	int res = 0;
	for(int i=0; i<data.size(); i++)
	{
		for(int j=i+1; j<data.size(); j++)
		{
			if(find(*this, -data[i] - data[j]) > j)
				res ++;
		}
	}
	return res;
}
int sum::cal_sum_3_update2() const
{
	int res = 0;
	for(int i=0; i<data.size(); i++)
	{
		int j=i+1;
		int p=data.size()-1;
		while(j<p)
		{
			if (data[j] + data[p] < -data[i])
				j++;
			else if(data[j] + data[p] > -data[i])
				p--;
			else
			{
				res++;
				j++;
				p--;
			}
		}
	}
	return res;
}
int sum::cal_sum_4_update() const
{
	int res = 0;
	for(int i=0; i<data.size(); i++)
	{
		for(int j=i+1; j<data.size(); j++)
		{
			for(int p=j+1; p<data.size(); p++)
			{
				if(find(*this, -data[i]-data[j]-data[p])>p)
					res++;
			}
		}
	}
	return res;
}

```

test.cpp


```cpp
#include "Sum.h"
#include <iostream>
#include <fstream>
#include <vector>
using namespace std;
void main()
{
	ifstream in("1Kints.txt");
	vector<int> a;
	while(!in.eof())
	{
		int temp;
		in>>temp;
		a.push_back(temp);
	}
	sum s(a);
	s.sort(0,a.size()-1);
	s.print();
	cout<<"s.cal_sum_2() = "<<s.cal_sum_2()<<endl;
	cout<<"s.cal_sum_2_update() = "<<s.cal_sum_2_update()<<endl;
	cout<<"s.cal_sum_3() = "<<s.cal_sum_3()<<endl;
	cout<<"s.cal_sum_3_update() = "<<s.cal_sum_3_update()<<endl;
	cout<<"s.cal_sum_3_update()2 = "<<s.cal_sum_3_update2()<<endl;
	cout<<"s.cal_sum_4() = "<<s.cal_sum_4()<<endl;
	cout<<"s.cal_sum_4_update() = "<<s.cal_sum_4_update()<<endl;
}

```


牛客单选题

链接：https://www.nowcoder.com/questionTerminal/7d79ca5b122c44d59ec9ff77d0b5624d
来源：牛客网

给定一个整数sum，从有N个无序元素的数组中寻找元素a、b、c、d，使得 a+b+c+d =sum，最快的平均时间复杂度是____。答：A

A. O(N^2)

B. O(log N)

C. O(N)

D. O(N^3)

E. O(N^2LogN)

F. O(N^4)


参考：

[剖析3-sum问题(Three sum)](https://blog.csdn.net/shaya118/article/details/40755551)

[2-sum, 3-sum, 4-sum问题分析](http://shmilyaw-hotmail-com.iteye.com/blog/2085129)



## 3.11 无序数组a，b 归并排序成有序数组c

## 3.12 将256*256二维数组逆时针旋转90°

## 3.13 十大排序算法

冒泡排序

简单选择

直接插入

希尔排序

快速排序

归并排序

堆排序

基数排序

桶排序

## 3.14 二分查找

## 3.15 分治与递归：逆序对数、大数相加、大数相乘

## 3.16 动态规划：背包问题、找零钱问题和最长公共子序列（LCS）

## 3.17 BFS和DFS解决最短路径

## 3.18 DFS和BFS的区别（优缺点）

## 3.19 DFS和BFS

深度优先搜索算法（Depth-First-Search，DFS）是一种利用递归实现的搜索算法。简单来说，其搜索过程和"不撞南墙不回头"类似。

广度优先搜索算法（Breadth-First-Search，BFS）是一种利用队列实现的搜索算法。简单来说，其搜索过程和"湖面丢进一块石头激起层层涟漪"类似。

BFS 的重点在于队列，而 DFS 的重点在于递归。这是它们的本质区别。


**应用方向**


BFS 常用于找单一的最短路线，它的特点是 "搜到就是最优解"，而 DFS 用于找所有解的问题，它的空间效率高，而且找到的不一定是最优解，必须记录并完成整个搜索，故一般情况下，深搜需要非常高效的剪枝（剪枝的概念请百度）。




参考

[一文读懂 BFS 和 DFS 区别](https://www.sohu.com/a/201679198_479559)




# 四、C/C++/Python

## 4.1 判断struct的字节数

一般使用sizeof判断struct所占的字节数，那么计算规则是什么呢？

关键词：

1.变量的起始地址和变量自身的字节数

2.以最大变量字节数进行字节对齐（倍数关系）。

注：这里介绍的原则都是在没有#pragma pack宏的情况下

先举个例子：


```cpp
struct A
{
    char a[5];
    int b;
    short int c;
}struct A;
```

在上例中，要计算 sizeof(a) 是多少？

有两个原则：
1）各成员变量存放的起始地址相对于结构的起始地址的偏移量必须为该变量的类型所占用的字节数的倍数
即当 A中的a占用了5个字节后，b需要占用四个字节，此时如果b直接放在a后，则b的起始地址是5，不是sizeof(int)的整数倍，所以
需要在a后面补充3个空字节，使得b的起始地址为8. 当放完b后，总空间为5+3+4 = 12. 接着放c，此时为 12 + 2 = 14.

2）为了确保结构的大小为结构的字节边界数（即该结构中占用最大空间的类型所占用的字节数）的倍数，
所以在为最后一个成员变量申请空间后，还会根据需要自动填充空缺的字节。
这是说A中占用最大空间的类型，就是int型了，占用了4个字节，那么规定A占用的空间必须是4的整数倍。本来计算出来占用的空间为14，
不是4的整数倍，因此需要在最后补充2个字节。最终导致A占用的空间为16个字节。


再举个例子：


```cpp
struct B
{
    char *d;
    short int e;
    long long f;
    char c[1];
}b;
 
void test2() {
	printf("%d\n", sizeof(b));
}

```

对于此题，需要注意的一点是：windows系统对long long是按照8字节进行对齐的，但是Linux系统对long long则是按照4字节对齐的。

因此:
d占用4字节（因为d是指针）

e占用2字节

f占用8字节，但是其起始地址为为6，不是4的整数倍（对于Linux系统），或不是8的整数倍（对于Windows系统），因此对e之后进行字节补齐，在这里不管对于Linux还是Windows都是补充2个字节，因此 f 的起始地址是8，占用8个字节。

对于c，它占用了1个字节，起始地址是16，也是1的整数倍。 

最后，在c之后需要对整个B结构体占用的空间进行补齐，目前占用空间是16+1 = 17个字节。

对于Linux，按4字节补齐（long long 是按4字节补齐的），因此补充了3位空字节，最后占用空间是 17 + 3 = 20字节。

对于Windows系统，是按8字节补齐的，因此就补充了7个字节，最后占用的空间是24字节。


参考：

[【C++】计算struct结构体占用的长度](https://blog.csdn.net/nisxiya/article/details/22456283?utm_source=copy )






## 4.2 static作用（修饰变量和函数）

**什么是static？**

static 是C++中很常用的修饰符，它被用来控制变量的存储方式和可见性。

**为什么要引入static？**

 函数内部定义的变量，在程序执行到它的定义处时，编译器为它在栈上分配空间，大家知道，函数在栈上分配的空间在此函数执行结束时会释放掉，这样就产生了一个问题: 如果想将函数中此变量的值保存至下一次调用时，如何实现？ 最容易想到的方法是定义一个全局的变量，但定义为一个全局变量有许多缺点，最明显的缺点是破坏了此变量的访问范围（使得在此函数中定义的变量，不仅仅受此函数控制）。



**static的作用**

第一个作用是限定作用域（隐藏）；第二个作用是保持变量内容持久化；

- 函数体内static变量的作用范围为该函数体，不同于auto变量，该变量的内存只被分配一次，因此其值在下次调用时仍维持上次的值。

举个例子


```
#include<iostream>

using namespace std;

int main(){

    for(int i=0; i<10; ++i){
	    int a = 0;
		static int b = 0;
		cout<<"a: "<< a++ <<endl;
		cout<<"b(static): " << b++ <<endl;
	}
	return 0;
}
```

输出结果：


```
a: 0
b(static): 0
a: 0
b(static): 1
a: 0
b(static): 2
a: 0
b(static): 3
a: 0
b(static): 4
a: 0
b(static): 5
a: 0
b(static): 6
a: 0
b(static): 7
a: 0
b(static): 8
a: 0
b(static): 9
```


- 在模块内的static全局变量可以被模块内所有函数访问，但不能被模块外其他函数访问

- 在模型内的static函数只可被这一模块内的其他函数调用，这个函数的使用范围被限制在声明它的模块内。


- 在类中的static成员变量属于整个类所有，对类的所有对象只有一份复制。即类的所有对象访问的static成员变量是同一个，而不是对象专属的。

- 在类中的static成员函数属于整个类所有，这个函数不接收this指针，因而只能访问类的static成员变量。同类的static成员变量性质一样，类的对象访问的static成员函数是同一个，不是对象专属的。

参考

- [C++中static关键字作用总结](https://www.cnblogs.com/songdanzju/p/7422380.html)

- [C/C++中STATIC用法总结](https://www.cnblogs.com/jhmu0613/p/7131997.html)

- 《程序员面试宝典》

## 4.3 Const作用（修饰变量和函数）


## 4.4 多态

## 4.5 虚函数

## 4.6 继承

## 4.7 多线程的同步问题

## 4.8 C++的设计模式

- 工厂模式
- 策略模式
- 适配器模式
- 单例模式
- 原型模式
- 模板模式
- 建造者模式
- 外观模型
- 组合模式
- 代理模式
- 享元模式
- 桥接模式
- 装饰模式
- 备忘录模式
- 中介者模式
- 职责链模式
- 观察者模式


参考：

[C++ 常用设计模式（学习笔记）](https://www.cnblogs.com/chengjundu/p/8473564.html)

[设计模式（C++实例）](https://blog.csdn.net/phiall/article/details/52199659)

## 4.9 动态内存管理


## 4.10 智能指针

## 4.11 long long转成string

## 4.12 NULL和nullstr的区别

## 4.13 delete和delete []区别

## 4.14 C++虚函数的实现机制

## 4.15 C++基类的析构函数为什么建议是虚函数

## 4.16 STL中的vector和list的区别

## 4.17 多线程和多进程的区别

进程：系统进行资源分配和调度的基本单位。

线程：基本的CPU执行单元，也是程序执行过程中的最小单元，由线程ID、程序计数器、寄存器集合和堆栈共同组成。（有种说法是线程是进程的实体）

线程和进程的关系：

（1）一个线程只能属于一个进程，而一个进程可以有多个线程，但至少有一个线程

（2）资源分配给进程，同一个进程的所有线程共享该进程的所有资源

（3）CPU分给进程，即真正在CPU上运行的是线程

参考：[多线程与多进程](https://www.cnblogs.com/yuanchenqi/articles/6755717.html)


## 4.18 实现atoi，即将"1234"转化成1234（int类型）

## 4.19 实现atof，即将"1.234"转换成1.234（float类型）

## 4.20 结构体和联合体的区别

结构体：把不同类型的数据组合成一个整体-------自定义数据类型，结构体变量所占内存长度是各成员占的内存长度的总和。

联合体：使几个不同类型的变量共占一段内存(相互覆盖)，共同体变量所占内存长度是各最长的成员占的内存长度。

Structure 与 Union主要有以下区别:

1.struct和union都是由多个不同的数据类型成员组成, 但在任何同一时刻, union中只存放了一个被选中的成员, 而struct的所有成员都存在。在struct中，各成员都占有自己的内存空间，它们是同时存在的。一个struct变量的总长度等于所有成员长度之和。在Union中，所有成员不能同时占用它的内存空间，它们不能同时存在。Union变量的长度等于最长的成员的长度。

2.对于union的不同成员赋值, 将会对其它成员重写, 原来成员的值就不存在了, 而对于struct的不同成员赋值是互不影响的。

3.联合体的各个成员共用内存，并应该同时只能有一个成员得到这块内存的使用权（即对内存的读写），而结构体各个成员各自拥有内存，各自使用互不干涉。所以，某种意义上来说，联合体比结构体节约内存。

举个例子：


```cpp
typedef struct
{
int i;
int j;
}A;
typedef union
{
int i;
double j;
}U;
```

sizeof(A)的值是8，sizeof(U)的值也是8（不是12）。

为什么sizeof(U)不是12呢？因为union中各成员共用内存，i和j的内存是同一块。而且整体内存大小以最大内存的成员的划分。即U的内存大小是double的大小，为8了。

sizeof(A)大小为8，因为struct中i和j各自得到了一块内存，每人4个字节，加起来就是8了。
了解了联合体共用内存的概念，也就是明白了为何每次只能对其一个成员赋值了，因为如果对另一个赋值，会覆盖了上一个成员的值。



举个例子：


例如：书包；可以放置书本、笔盒、记事本等物。

联合体，仅能放入一样东西的包(限制)，其尺寸，是可放物品中，最大一件的体积。

结构体，是能放入所有物品的包，所以其尺寸，可同时容纳多样物品。

联合体，同时间只能有一个成员在内。或是说，可以用不同型态，去看同一组数据。

结构体，可以包含多个成员在一起，成员都能个别操作。

## 4.21 引用和指针

## 4.22 C++ operator new 和 new operator

## 4.23 怎么理解C++面向对象？跟Python面向对象有什么区别？

## 4.24 C++多态特性，父类和子类的区别

## 4.25 虚函数有哪些作用？多态的虚函数（父类和子类）返回值类型可以不一样吗？什么情况下，返回值类型不一样？

## 4.26 C++四种强制类型转换有哪些？每种特性是什么？有什么区别？




# 五、项目/实习



# 六、计算机视觉/图像处理知识点

## 6.1 SIFT

## 6.2 SURF


## 6.3 ORB


## 6.4 高斯滤波

## 6.5 Hough变换原理（直线和圆检测）

## 6.6 LSD

## 6.7 线性插值

## 6.8 双线性插值

## 6.9 腐蚀/膨胀

## 6.10 开运算和闭运算

## 6.11 找轮廓（findCountours）

## 6.12 Sobel

## 6.13 Canny

## 6.14 边缘检测算子

## 6.15 仿射变换

## 6.16 颜色空间介绍

- RGB
- HSI
- CMYK
- YUV

## 6.17 单应性（homography）原理


## 6.18 二维高斯滤波能否分解成一维操作

答：可以分解。

二维高斯滤波分解为两次一维高斯滤波，高斯二维公式可以推导为X轴与Y轴上的一维高斯公式。

即使用一维高斯核先对图像逐行滤波，再对中间结果逐列滤波。

参考：

[快速高斯滤波、高斯模糊、高斯平滑(二维卷积分步为一维卷积)](https://blog.csdn.net/qq_36359022/article/details/80188873)


## 6.19 HOG算法介绍

## 6.20 怎么理解图像的频率

## 6.21 双边滤波

## 6.22 图像中的低频信息和高频信息

图像频率：图像中灰度变化剧烈程度的指标

- 低频信息（低频分量）表示图像中灰度值变化缓慢的区域，对应着图像中大块平坦的区域。
- 高频信息（高频分量）表示图像中灰度值变化剧烈的区域，对应着图像的边缘（轮廓）、噪声以及细节部分。

低频分量：主要对整幅图像强度的综合度量

高频分量：主要对图像边缘和轮廓的度量


从傅里叶变换的角度，将图像从灰度分布转化为频率分布。

参考：

[理解图像中的低频分量和高频分量](https://blog.csdn.net/Chaolei3/article/details/79443520)


6.23 引导滤波

参考

[【拜小白opencv】33-平滑处理6——引导滤波/导向滤波（Guided Filter）](https://blog.csdn.net/sinat_36264666/article/details/77990790)




# 七、其它

## TCP

关键词说明

ACK：确认标志

SYN：同步标志

FIN：结束标志

### TCP与UDP的区别

UDP 与 TCP 的主要区别在于 UDP 不一定提供可靠的数据传输，它不能保证数据准确无误地到达，不过UDP在许多方面非常有效。当程序是要尽快地传输尽可能多的信息时，可以使用 UDP。TCP它是通过三次握手建立的连接，它在两个服务之间始终保持一个连接状态，目的就是为了提供可靠的数据传输。许多程序使用单独的TCP连接和单独的UDP连接，比如重要的状态信息用可靠的TCP连接发送，而主数据流通过UDP发送。

TCP与UDP区别总结：

1、TCP面向连接（如打电话要先拨号建立连接）;UDP是无连接的，即发送数据之前不需要建立连接

2、TCP提供可靠的服务。也就是说，通过TCP连接传送的数据，无差错，不丢失，不重复，且按序到达;UDP尽最大努力交付，即不保证可靠交付

3、TCP面向字节流，实际上是TCP把数据看成一连串无结构的字节流;UDP是面向报文的

UDP没有拥塞控制，因此网络出现拥塞不会使源主机的发送速率降低（对实时应用很有用，如IP电话，实时视频会议等）

4、每一条TCP连接只能是点到点的;UDP支持一对一，一对多，多对一和多对多的交互通信

5、TCP首部开销20字节;UDP的首部开销小，只有8个字节
6、TCP的逻辑通信信道是全双工的可靠信道，UDP则是不可靠信道


参考：

[TCP和UDP的优缺点及区别](https://www.cnblogs.com/xiaomayizoe/p/5258754.html)



### TCP三次握手

因为TCP是一个双向通讯协议，所以要三次握手才能建立：

第一次握手是客户端向服务端发送连接请求包（SYN=J），服务端接收到之后会给客户端发个确认标志（也就是两个包，一个是确认包ACK=J+1,另一个是连接询问请求包SYN=K)，这是第二次握手。第三次握手就是客户端会再次给服务端发送消息确认标志ACK=K+1，表示能正常接收可以开始通信。第三次握手的目的是为了防止已经失效的连接请求突然又传送到了服务端，因为网络中有可能存在延迟的问题，如果采用二次握手就会让服务端误认为client是再次发出新的连接请求，然后server一直等待client发来数据，这样就浪费了很多资源。这三次握手是在connect,bind,listen和accept函数中完成的，这几个函数创建了比较可靠的连接通道。其实断开连接的四次握手是跟连接的时候一样的，唯一多了一步就是因为双方都处在连接的时候，而且有可能在传输数据，在服务端接收到客户端的关闭连接请求后它会给客户端确认，但是由于数据还没有传送完毕，此时会进入一个TIME_WAIT状态，所以在数据传送好之后会再次给客户端发消息，这就是多出来的那一步。

过程：

第一次

第一次握手：建立连接时，客户端发送syn包（syn=j）到服务器，并进入SYN_SENT状态，等待服务器确认；SYN：同步序列编号（Synchronize Sequence Numbers）。

此时客户端状态为：SYN_SENT，服务器为LISTEN。

第二次

第二次握手：服务器收到syn包，必须确认客户的SYN（ack=j+1），同时自己也发送一个SYN包（syn=k），即SYN+ACK包，此时服务器进入SYN_RECV状态；

此时客户端状态为ESTABLISHED，服务器为SYS_RCVD

第三次

第三次握手：客户端收到服务器的SYN+ACK包，向服务器发送确认包ACK(ack=k+1），此包发送完毕，客户端和服务器进入ESTABLISHED（TCP连接成功）状态，完成三次握手

此时客户端和服务器的状态都为ESTABLISHED。

完成三次握手，客户端与服务器开始传送数据。

![](https://gss2.bdstatic.com/-fo3dSag_xI4khGkpoWK1HF6hhy/baike/c0%3Dbaike72%2C5%2C5%2C72%2C24/sign=d1c3130b070828387c00d446d9f0c264/55e736d12f2eb9386decc2e6d5628535e5dd6f25.jpg)


必要性：

考虑一次的问题，首先tcp是面向连接，一次握手肯定建立不了连接，因为客户机给服务器发出请求信息却没有得到回应，客户机是没法判定是否发送成功然后建立连接的。

再看两次，假设只有两次握手，比如图中的1，2步，当A想要建立连接时发送一个SYN，然后等待ACK，结果这个SYN因为网络问题没有及时到达B，所以A在一段时间内没收到ACK后，再发送一个SYN，这次B顺利收到，接着A也收到ACK，这时A发送的第一个SYN终于到了B，对于B来说这是一个新连接请求，然后B又为这个连接申请资源，返回ACK，然而这个SYN是个无效的请求，A收到这个SYN的ACK后也并不会理会它，而B却不知道，B会一直为这个连接维持着资源，造成资源的浪费。

两次握手的问题在于服务器端不知道一个SYN是否是无效的，而三次握手机制因为客户端会给服务器回复第二次握手，也意味着服务器会等待客户端的第三次握手，如果第三次握手迟迟不来，服务器便会认为这个SYN是无效的，释放相关资源。但这时有个问题就是客户端完成第二次握手便认为连接已建立，而第三次握手可能在传输中丢失，服务端会认为连接是无效的，这时如果Client端向Server写数据，Server端将以RST包响应，这时便感知到Server的错误。

总之，三次握手可以保证任何一次握手的失败都是可感知的，不会浪费资源


### TCP四次挥手

对于一个已经建立的连接，TCP使用改进的三次握手来释放连接（使用一个带有FIN附加标记的报文段）。TCP关闭连接的步骤如下：

第一步，当主机A的应用程序通知TCP数据已经发送完毕时，TCP向主机B发送一个带有FIN附加标记的报文段（FIN表示英文finish）。

第二步，主机B收到这个FIN报文段之后，并不立即用FIN报文段回复主机A，而是先向主机A发送一个确认序号ACK，同时通知自己相应的应用程序：对方要求关闭连接（先发送ACK的目的是为了防止在这段时间内，对方重传FIN报文段）。

第三步，主机B的应用程序告诉TCP：我要彻底的关闭连接，TCP向主机A送一个FIN报文段。

第四步，主机A收到这个FIN报文段后，向主机B发送一个ACK表示连接彻底释放。 

![四次挥手](https://gss0.bdstatic.com/-4o3dSag_xI4khGkpoWK1HF6hhy/baike/c0%3Dbaike116%2C5%2C5%2C116%2C38/sign=a72f287dd562853586edda73f1861da3/48540923dd54564eae27b6b6b3de9c82d0584ffa.jpg)

形象描述四次挥手：

假设Client端发起中断连接请求，也就是发送FIN报文。Server端接到FIN报文后，意思是说"我Client端没有数据要发给你了"，但是如果你还有数据没有发送完成，则不必急着关闭Socket，可以继续发送数据。所以你先发送ACK，"告诉Client端，你的请求我收到了，但是我还没准备好，请继续你等我的消息"。这个时候Client端就进入FIN_WAIT状态，继续等待Server端的FIN报文。当Server端确定数据已发送完成，则向Client端发送FIN报文，"告诉Client端，好了，我这边数据发完了，准备好关闭连接了"。Client端收到FIN报文后，"就知道可以关闭连接了，但是他还是不相信网络，怕Server端不知道要关闭，所以发送ACK后进入TIME_WAIT状态，如果Server端没有收到ACK则可以重传。“，Server端收到ACK后，"就知道可以断开连接了"。Client端等待了2MSL后依然没有收到回复，则证明Server端已正常关闭，那好，我Client端也可以关闭连接了。Ok，TCP连接就这样关闭了！

需四次挥手原因：由于TCP的半关闭特性，TCP连接时双全工（即数据在两个方向上能同时传递），因此，每个方向必须单独的进行关闭。这个原则就是：当一方完成它的数据发送任务后就能发送一个FIN来终止这个方向上的连接。当一端收到一个FIN后，它必须通知应用层另一端已经终止了那个方向的数据传送。即收到一个FIN意味着在这一方向上没有数据流动了。


假设客户机A向服务器B请求释放TCP连接，则：

第一次挥手：主机A向主机B发送FIN包；A告诉B，我（A）发送给你（B）的数据大小是N，我发送完毕，请求断开A->B的连接。

第二次挥手：主机B收到了A发送的FIN包，并向主机A发送ACK包；B回答A，是的，我总共收到了你发给我N大小的数据，A->B的连接关闭。

第三次挥手：主机B向主机A发送FIN包；B告诉A，我（B）发送给你（A）的数据大小是M，我发送完毕，请求断开B->A的连接。

第四次挥手：主机A收到了B发送的FIN包，并向主机B发送ACK包；A回答B，是的，我收到了你发送给我的M大小的数据，B->A的连接关闭。


这里再系统性的介绍四次握手

当客户端和服务器通过三次握手建立了TCP连接以后，当数据传送完毕，肯定是要断开TCP连接的啊。那对于TCP的断开连接，这里就有了神秘的“四次挥手”。

1. 第一次挥手：主机1（可以使客户端，也可以是服务器端），设置Sequence Number和Acknowledgment Number，向主机2发送一个FIN报文段；此时，主机1进入FIN_WAIT_1状态；这表示主机1没有数据要发送给主机2了；

2. 第二次挥手：主机2收到了主机1发送的FIN报文段，向主机1回一个ACK报文段，Acknowledgment Number为Sequence Number加1；主机1进入FIN_WAIT_2状态；主机2告诉主机1，我“同意”你的关闭请求；

3. 第三次挥手：主机2向主机1发送FIN报文段，请求关闭连接，同时主机2进入LAST_ACK状态；

4. 第四次挥手：主机1收到主机2发送的FIN报文段，向主机2发送ACK报文段，然后主机1进入TIME_WAIT状态；主机2收到主机1的ACK报文段以后，就关闭连接；此时，主机1等待2MSL后依然没有收到回复，则证明Server端已正常关闭，那好，主机1也可以关闭连接了。



**为什么要四次挥手？**

那四次分手又是为何呢？TCP协议是一种面向连接的、可靠的、基于字节流的运输层通信协议。TCP是全双工模式，这就意味着，当主机1发出FIN报文段时，只是表示主机1已经没有数据要发送了，主机1告诉主机2，它的数据已经全部发送完毕了；但是，这个时候主机1还是可以接受来自主机2的数据；当主机2返回ACK报文段时，表示它已经知道主机1没有数据发送了，但是主机2还是可以发送数据到主机1的；当主机2也发送了FIN报文段时，这个时候就表示主机2也没有数据要发送了，就会告诉主机1，我也没有数据要发送了，之后彼此就会愉快的中断这次TCP连接。如果要正确的理解四次分手的原理，就需要了解四次分手过程中的状态变化。

- FIN_WAIT_1: 这个状态要好好解释一下，其实FIN_WAIT_1和FIN_WAIT_2状态的真正含义都是表示等待对方的FIN报文。而这两种状态的区别是：FIN_WAIT_1状态实际上是当SOCKET在ESTABLISHED状态时，它想主动关闭连接，向对方发送了FIN报文，此时该SOCKET即进入到FIN_WAIT_1状态。而当对方回应ACK报文后，则进入到FIN_WAIT_2状态，当然在实际的正常情况下，无论对方何种情况下，都应该马上回应ACK报文，所以FIN_WAIT_1状态一般是比较难见到的，而FIN_WAIT_2状态还有时常常可以用netstat看到。（主动方）

- FIN_WAIT_2：上面已经详细解释了这种状态，实际上FIN_WAIT_2状态下的SOCKET，表示半连接，也即有一方要求close连接，但另外还告诉对方，我暂时还有点数据需要传送给你(ACK信息)，稍后再关闭连接。（主动方）

- CLOSE_WAIT：这种状态的含义其实是表示在等待关闭。怎么理解呢？当对方close一个SOCKET后发送FIN报文给自己，你系统毫无疑问地会回应一个ACK报文给对方，此时则进入到CLOSE_WAIT状态。接下来呢，实际上你真正需要考虑的事情是察看你是否还有数据发送给对方，如果没有的话，那么你也就可以 close这个SOCKET，发送FIN报文给对方，也即关闭连接。所以你在CLOSE_WAIT状态下，需要完成的事情是等待你去关闭连接。（被动方）

- LAST_ACK: 这个状态还是比较容易好理解的，它是被动关闭一方在发送FIN报文后，最后等待对方的ACK报文。当收到ACK报文后，也即可以进入到CLOSED可用状态了。（被动方）

- TIME_WAIT: 表示收到了对方的FIN报文，并发送出了ACK报文，就等2MSL后即可回到CLOSED可用状态了。如果FINWAIT1状态下，收到了对方同时带FIN标志和ACK标志的报文时，可以直接进入到TIME_WAIT状态，而无须经过FIN_WAIT_2状态。（主动方）

- CLOSED: 表示连接中断。


参考

[三次握手](https://baike.baidu.com/item/%E4%B8%89%E6%AC%A1%E6%8F%A1%E6%89%8B/5111559)

[tcp三次握手及其必要性](https://blog.csdn.net/u013344815/article/details/72134950)

[c++面试题（网络通信篇）](https://blog.csdn.net/zhouchunyue/article/details/79271908)

[tcp建立连接为什么需要三次握手](https://www.jianshu.com/p/e7f45779008a)

[TCP相关面试题(转)](https://www.cnblogs.com/huajiezh/p/7492416.html)


### TCP连接的可靠性

TCP通过以下方式提供数据传输的可靠性：

（1）TCP在传输数据之前，都会把要传输的数据分割成TCP认为最合适的报文段大小。在TCP三次我握手的前两次握手中（也就是两个SYN报文段中），通过一个“协商”的方式来告知对方自己期待收到的最大报文段长度（MSS），结果使用通信双发较小的MSS最为最终的MSS。在SYN=1的报文段中，会在报文段的选项部分来指定MSS大小（相当于告知对方自己所能接收的最大报文段长度）。在后续通信双发发送应用层数据之前，如果发送数据超过MSS，会对数据进行分段。

（2）使用了超时重传机制。当发送一个TCP报文段后，发送发就会针对该发送的段启动一个定时器。如果在定时器规定时间内没有收到对该报文段的确认，发送方就认为发送的报文段丢失了要重新发送。

（3）确认机制。当通信双发的某一端收到另一个端发来的一个报文段时，就会返回对该报文段的确认报文。

（4）首部校验和。在TCP报文段首部中有16位的校验和字段，该字段用于校验整个TCP报文段（包括首部和数据部分）。IP数据报的首部校验和只对IP首部进行校验。TCP详细的校验过程如下，发送TCP报文段前求一个值放在校验位，接收端接受到数据后再求一个值，如果两次求值形同则说明传输过程中没有出错；如果两次求值不同，说明传输过程中发生错误，无条件丢弃该报文段引发超时重传。

（5）使用滑动窗口流量控制协议。

（6）由于在TCP发送端可能对数据分段，那么在接收端会对接收到的数据重新排序。


参考：[腾讯面试TCP连接相关问题](https://blog.csdn.net/bian_qing_quan11/article/details/74999463)


### TCP的ICMP

ICMP是（Internet Control Message Protocol）Internet控制报文协议。它是TCP/IP协议族的一个子协议，用于在IP主机、路由器之间传递控制消息。控制消息是指网络通不通、主机是否可达、路由是否可用等网络本身的消息。这些控制消息虽然并不传输用户数据，但是对于用户数据的传递起着重要的作用。

ICMP协议是一种面向无连接的协议，用于传输出错报告控制信息。它是一个非常重要的协议，它对于网络安全具有极其重要的意义。

**它是TCP/IP协议族的一个子协议，属于网络层协议，主要用于在主机与路由器之间传递控制信息**，包括报告错误、交换受限控制和状态信息等。当遇到IP数据无法访问目标、IP路由器无法按当前的传输速率转发数据包等情况时，会自动发送ICMP消息。ICMP报文在IP帧结构的首部协议类型字段（Protocol 8bit)的值=1.



## ISO的7层网络模型

口诀：应表会传网数物

- 应用层：处理网络应用
- 表示层：数据表示
- 会话层：互连主机通信
- 传输层：端到端连接
- 网络层：寻址和最短路径
- 数字链路层：接入介质
- 物理层：二进制传输


## UDP


## TCP和UDP

一般面试官都会问TCP和UDP的区别，这个很好回答啊，TCP面向连接，可靠，基于字节流，而UDP不面向连接，不可靠，基于数据报。对于连接而言呢，其实真正的就不存在，TCP面向连接只不过三次握手在客户端和服务端之间初始化好了序列号。只要满足TCP的四元组+序列号，那客户端和服务端之间发送的消息就有效，可以正常接收。虽然说TCP可靠，但是可靠的背后却是lol无尽之刃的复杂和痛苦，滑动窗口，拥塞避免，四个超时定时器，还有什么慢启动啊，快恢复，快重传啊这里推荐大家看看（图解TCP/IP,这个简单容易，TCP卷123，大量的文字描述真是烦），所以什么都是相对呢，可靠性的实现也让TCP变的复杂，在网络的状况很差的时候，TCP的优势会变成。基于字节流什么意思呢？一句话就可以说明白，对于读写没有相对应的次数。UDP基于数据报就是每对应一个发，就要对应一个收。而TCP无所谓啊，现在应该懂了吧。对于UDP而言，不面向连接，不可靠，没有三次握手，我给你发送数据之前，不需要知道你在不在，不要你的同意，我只管把数据发送出去至于你收到不收到，从来和我没有半毛钱的关系。

对于可靠不可靠而言，没有绝对的说法，TCP可靠仅仅是在传输层实现了可靠，我也可以让UDP可靠啊，那么就要向上封装，在应该层实现可靠性。因此很多公司都不是直接用TCP和UDP，都是经过封装，满足业务的需要而已。说到这里的话，那就在提一下心跳包，在linux下有keep-alive系统自带的，但是默认时间很长，如果让想使用话可以setsockopt设置，我也可以在应用层实现一个简单心跳包，上次自己多开了一个线程来处理，还是包头解决。


上面解释完这个之后面试官可能问，那什么时候用TCP，什么时候用UDP呢？就是问应用场景，所以简历上的知识点自己应该提前做好准备应用场景，知识就是要用在显示场景中，废话真多。不管用TCP和UDP，应用只要看需求，对于TCP更加注重的是可靠性，而不是实时性，如果我发送的数据很重要一点也不能出错，有延迟无所谓的话，那就TCP啊。UDP更加注重是速度快，也就是实时性，对于可靠性要求不那么高，所以像斗鱼，熊猫这些在线直播网站应该在UDP基础是封装了其他协议，比如视频实时传输协议。而且UDP的支持多播，那就很符合这些直播网站了，有时候看直播视频卡顿，人飘逸那可能就是丢包了，但是你也只能往下看。

参考: [关于面试中的TCP和UDP怎么用自己的话给面试官说](https://blog.csdn.net/lotluck/article/details/52688851)


## DNS

DNS（Domain Name System，域名系统），万维网上作为域名和IP地址相互映射的一个分布式数据库，能够使用户更方便的访问互联网，而不用去记住能够被机器直接读取的IP数串。**通过域名，最终得到该域名对应的IP地址的过程叫做域名解析（或主机名解析）。DNS协议运行在UDP协议之上，使用端口号53**。在RFC文档中RFC 2181对DNS有规范说明，RFC 2136对DNS的动态更新进行说明，RFC 2308对DNS查询的反向缓存进行说明。


## DOS

Dos攻击在众多网络攻击技术中是一种简单有效并且具有很大危害性的攻击方法。它通过各种手段消耗网络带宽和系统资源，或者攻击系统缺陷，使正常系统的正常服务陷于瘫痪状态，不能对正常用户进行服务，从而实现拒绝正常用户访问服务。

DDOS攻击是基于DOS攻击的一种特殊形式。攻击者将多台受控制的计算机联合起来向目标计算机发起DOS攻击，它是一种大规模协作的攻击方式，主要瞄准比较大的商业站点，具有较大的破坏性。


如何防止DOS攻击？

1. 确保服务器的系统文件是最新的版本，并及时更新系统补丁。 
2. 关闭不必要的服务。  
3. 限制同时打开的SYN半连接数目。  
4. 缩短SYN半连接的time out 时间。  
5. 正确设置防火墙  禁止对主机的非开放服务的访问  限制特定IP地址的访问  启用防火墙的防DDoS的属性  严格限制对外开放的服务器的向外访问  运行端口映射程序祸端口扫描程序，要认真检查特权端口和非特权端口。 
6. 认真检查网络设备和主机/服务器系统的日志。只要日志出现漏洞或是时间变更，那这台机器就可   能遭到了攻击。 
7. 限制在防火墙外与网络文件共享。这样会给黑客截取系统文件的机会，主机的信息暴露给黑客，无疑是给了对方入侵的机会。