## 评估指标

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



