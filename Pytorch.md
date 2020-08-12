# Pytorch

## pytorch中的广播机制

广播机制：张量参数可以自动扩展为相同大小

广播机制需要满足两个条件：

- 每个张量至少有一个维度
- 满足右对齐

右对齐实例：troch.rand(2,1,1)+torch.rand(3) 该例满足右对齐（除了加法，减，乘，除也满足）

右对齐判别方式，维度小的往左侧补1知道同维，然后依次判断，**所有的数（各维度）相等或者其中一个为1**，如果所有数均满足，则满足右对齐。结果为2\*1\*3（维度相等取相等，维度含1取另一个）

## Tensor的比较运算

torch.eq(input, other, out=None) #按成员进行等式操作，相同返回True（返回的是同尺寸向量，里面元素是True和False）

torch.equal(tensor1, tensor2) #如果tensor1和tensor2有**相同的size和elements**，则为true（单个值，True或者False）

## Tensor张量裁剪

a.clamp(2, 5) #a为tensor张量，2和5是范围上下界，小于2的会被修改为2，大于5的会被修改为5，其余保留原本的数值。

## 有监督学习、无监督学习、半监督学习

- 样本X，标签Y

所谓有监督学习就是指数据集中既有X又有Y（典型的例子有LDA线性判别分析、SVM）；所谓无监督学习就是指数据集中只有X没有Y（典型的例子为聚类）

所谓**半监督学习**有几种情况：

1. 弱标签：标签不准确、标签可能错误
2. 伪标签：比如标签是通过聚类获取到的
3. 一部分有标签，一部分没标签

## FC层（全连接层）

层中包含的皆为线性运算。

## 损失函数

对于回归问题，$l_1$、$l_2$损失函数；对于分类问题，softmax、交叉熵计算损失。

## 过拟合与欠拟合

### 概念

![image-20200812151919337](C:\Users\Ricky\AppData\Roaming\Typora\typora-user-images\image-20200812151919337.png) 

这部分可以结合西瓜书中所提到的偏差-方差分析来思考。

模型的不稳定也代表了模型可能存在过拟合或欠拟合。

高偏差一般是欠拟合（模型选择可能存在问题）。

### 应对方法：

![image-20200812152254260](C:\Users\Ricky\AppData\Roaming\Typora\typora-user-images\image-20200812152254260.png)

## 正则化问题

![image-20200812152837924](C:\Users\Ricky\AppData\Roaming\Typora\typora-user-images\image-20200812152837924.png)






