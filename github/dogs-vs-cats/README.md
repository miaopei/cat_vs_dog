# dogs-vs-cats
利用深度学习识别图片中的猫狗

利用pre-trained神经网络(VGG16 & ResNet50)对图像进行特征提取，再将提取得到的特征向量作为输入来训练一个新的全连接神经网络，以处理“猫狗”分类问题，实现约99.1%的正确率，LogLoss约为0.053。

### 项目概述

“猫狗大战”是[Kaggle](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition)的一个竞赛项目，从下图的排行榜可以看到，目前最好的模型可以实现约0.033的LogLoss。其实，在包括更加复杂的ImageNet问题上，基于CNN的图像识别算法可以远远超过人类的表现，这也是这个项目所要达到的目标。

<img width="500" height="300" src="https://github.com/TIFOSI528/dogs-vs-cats/raw/master/raw/1.png"/>

项目将主要包含数据探索、可视化、算法和技术的调研选取、基准模型的选择、数据预处理、模型的构建与优化、模型的评价与验证等步骤，并最终获得能够精准识别猫狗的神经网络模型。

### 数据的探索
在该项目中，输入数据为jpeg格式的图片，kaggle上提供的训练数据集包含了25,000张图片，其中猫和狗各12,500张，每一张图片对应一种类别，kaggle还提供了12,500张没有标记的图片用于测试。

<img width="350" height="150" src="https://github.com/TIFOSI528/dogs-vs-cats/raw/master/raw/3.png"/>

### 探索性可视化
对 train/文件夹中，属于猫和狗的部分图片分别进行可视化，如下所示:

<img width="450" height="700" src="https://github.com/TIFOSI528/dogs-vs-cats/raw/master/raw/4.png"/>

### 数据预处理

通过对图片中的色彩-像素比进行IQR分析，可以发现很多分辨率低、无关的图片，我们需要把不合格的图片删除，下面是其中一些不合格的图片：

<img width="900" height="550" src="https://github.com/TIFOSI528/dogs-vs-cats/raw/master/raw/5.png"/>

此外，通过对图片数据的探索，我们可以知道，图片中猫狗的拍摄角度不尽相同，而且猫狗占整张图片的比例也有所差别。为了让模型尽量不受这些因素的干扰，增强模型的泛化能力，需要对原始图片进行一些随机操作，比如旋转、剪切变换、缩放、水平翻转等。

Keras提供的图片生成器ImageDataGenerator可以很方便地对图片进行提升。简单地对train/cat.0.jpg做一些旋转、剪切变换、缩放等随机操作，可以得到以下结果：

    from keras.preprocessing.image import ImageDataGenerator
	raw_data_gen = ImageDataGenerator(
        	rotation_range=30,
        	shear_range=0.2,
        	zoom_range=0.2,
        	horizontal_flip=True,
        	fill_mode='nearest'
        	)
<img width="500" height="600" src="https://github.com/TIFOSI528/dogs-vs-cats/raw/master/raw/6.png"/>

### 模型可视化

<img width="600" height="500" src="https://github.com/TIFOSI528/dogs-vs-cats/raw/master/raw/7.png"/>

### 评价指标
1、在模型的训练过程中，可使用准确率作为评估指标：因为kaggle提供的用于训练的图片分类是已知的，使用准确率评估模型相对直观且计算简单，便于模型更新迭代；

2、而在使用测试集评估模型时，由于测试集的图片分类是未知的，我们将把预测结果提交到kaggle，kaggle官方将使用LogLoss对模型进行评分，以下是针对二分类问题的 LogLoss定义：

<img width="300" height="60" src="https://github.com/TIFOSI528/dogs-vs-cats/raw/master/raw/2.png"/>   
LogLoss越小，模型的预测越准确。

由于深度神经网络模型对图像分类的预测结果为概率，若要计算准确率，则需要先把概率转化成类别，这需要手动设置一个阈值，如果对一个样本的预测概率高于这个预测，就把这个样本放进一个类别里面，低于这个阈值，放进另一个类别里面。所以这个阈值很大程度上影响了准确率的计算，使用LogLoss可以避免把预测概率转换成类别，实现对模型的精确评估。

### 项目中调用的库

项目是基于python 2.7, keras 2.0.4, tensorflow-gpu-1.1.0完成的，主要用到了以下库：

import os:  os.listdir用于读取文件夹中的文件名，返回一个list

from keras.preprocessing.image import load_img: 用于读取图片

import matplotlib.pyplot as plt: 用于可视化图片

from keras.preprocessing.image import img_to_array, load_img: img_to_array 可将图片转化为array

from graphviz import Digraph: 用于画模型图

from keras.applications.vgg16 import VGG16: 用于导入VGG16预训练模型

from keras.applications.resnet50 import ResNet50: 用于导入ResNet50预训练模型

from keras.applications.imagenet_utils import preprocess_input: 用于导入预处理函数

from keras.models import Model: 用于构建函数式模型

from keras.layers import *: 用于添加layer

from keras.optimizers import SGD, Adadelta: 用于导入模型的优化方法

from keras.preprocessing.image import ImageDataGenerator: 图片生成器，可用于提升数据，构造batch数据

import h5py: 用于保存模型的训练参数


### Reference
[1]崔天依. 计算机视觉技术及其在自动化中的应用[J]. 电脑知识与技术, 2016 (3).

[2] “猫狗大战”数据集，kaggle，https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition

[3] Donahue J, Jia Y, Vinyals O, et al. DeCAF: A Deep Convolutional Activation Feature for Generic Visual Recognition[C]//Icml. 2014, 32: 647-655.

[4] Simonyan K, Zisserman A. Very Deep Convolutional Networks for Large-Scale Image Recognition[J]. Computer Science, 2014.

[5] He K, Zhang X, Ren S, et al. Deep residual learning for image recognition[C]. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016: 770-778.

[6] Szegedy C, Vanhoucke V, Ioffe S, et al. Rethinking the inception architecture for computer vision[C]. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016: 2818-2826.

[7] 郑泽宇, 顾思宇等. Tensorflow 实战google深度学习框架[B].2017.

[9] Chollet, François, Keras, GitHub, https://github.com/fchollet/keras.

[9] Martín Abadi, Ashish Agarwal, Paul Barham, et al. TensorFlow: Large-scale machine learning on heterogeneous systems, 2015. Software available from tensorflow.org.

[10] Ruder S. An overview of gradient descent optimization algorithms[J]. arXiv preprint arXiv:1609.04747, 2016.

[11] Zeiler M D. ADADELTA: an adaptive learning rate method[J]. arXiv preprint arXiv:1212.5701, 2012.

[12] Xie S, Girshick R, Dollár P, et al. Aggregated residual transformations for deep neural networks[J]. arXiv preprint arXiv:1611.05431, 2016.
