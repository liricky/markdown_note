# FSSD: Feature Fusion Single Shot Multibox Detector

## Abstract

SSD是最好的物体检测算法之一，具有高精度和快速性。但是，SSD的特征金字塔检测方法使得难以融合不同尺度上的特征。在本文中，我们提出了FSSD，它是一种具有新颖轻巧的特征融合模块的增强型SSD，在速度下降不大的情况下，可以大大提高SSD的性能。在特征融合模块中，来自不同层的具有不同尺度的特征被串联在一起，然后是一些下采样块以生成新的特征金字塔，该特征金字塔将被馈送到多盒检测器以预测最终的检测结果。在Pascal VOC 2007测试中，我们的网络可以使用单个Nvidia 1080Ti GPU以65.8 FPS（每秒帧）的速度实现82.7 mAP（平均每秒精度），输入尺寸为300×300。此外，我们在COCO方面的结果也比传统SSD拥有更大的优势。我们的FSSD在准确性和速度方面均胜过许多最新的目标检测算法。



# 背景

## SSD

[参考1](https://blog.csdn.net/zj15939317693/article/details/80596870)

[参考2](https://www.cnblogs.com/pacino12134/p/10353959.html)

目标检测主流的算法主要分为两个类型：

1. tow-stage

   R-CNN系列算法，其主要思路是先通过启发式方法（**selective search**）或者CNN网络（**RPN，RegionProposal Network**）产生一些列稀疏的候选框，然后对这些候选框进行分类和回归。two-stage方法的优势是准确度高。

2. one-stage

   如YOLO和SSD，主要思路是均匀的在图片的不同位置进行**密集抽样**，抽样时可以采用不同尺度和长宽比，然后利用CNN提取特征后直接进行分类和回归，整个过程只需要一部，所以其优势是速度快。

   均匀的密集采样的一个重要缺点是**训练比较困难**，这主要是因为正样本与负样本极其不平衡，导致模型准确度稍低。

不同算法的性能如图：

![img](https://img2018.cnblogs.com/blog/1476416/201902/1476416-20190206172739867-1605776180.png)

