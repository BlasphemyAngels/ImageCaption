## 图像描述

最近老师让我写一个图像描述的项目，于是我就结合谷歌很早发表的[NIC](https://www.computer.org/csdl/trans/tp/2017/04/07505636-abs.html)和今年CVPR的两篇论文:[Skeleton Key: Image Captioning by Skeleton-Attribute Decomposition](https://arxiv.org/abs/1704.06972)和[SCA-CNN: Spatial and Channel-wise Attention in Convolutional Networksfor Image Captioning](https://arxiv.org/abs/1611.05594)的内容做了这个项目，目前这个项目还在阿里云上运行，下面给出运行少量数据(几百张)得到的结果。

## 实验环境

* `python3`
* `opencv-python`
* `tensorlayer`
* `vgg网络`

## 结果

![Image1](https://github.com/BlasphemyAngels/MarkDownPhotos/blob/master/1000268201_693b08cb0e.jpg?raw=true
)

a dog wearing a red is is with in the of a dog .

![image2](https://github.com/BlasphemyAngels/MarkDownPhotos/blob/master/1002674143_1b742ab4b8.jpg?raw=true
)
  
a dog dog is a blue of . 

可以看出给出的描述看起来很荒谬而且不完整，但是起码有些语法是正确的，所以等完全跑完再看看效果吧(-_-)。



