# 猫狗大战


## 描述

使用深度学习方法识别一张图片是猫还是狗。

* 输入：一张彩色图片
* 输出：是猫还是狗

## 数据

此数据集可以从 kaggle 上下载。[Dogs vs. Cats Redux: Kernels Edition](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data)

## 实验环境

```
-- python 3.5.4
-- cuda 8.0
-- tensorflow-gpu (1.4.0)
-- Keras (2.0.9)
-- Ubuntu 16.04.2 LTS
-- gpu gtx 1080ti
-- cpu  12  Intel(R) Core(TM) i7-6850K CPU @ 3.60GHz
```

详细实验环境保存在导出的 conda 配置文件 `dlnd-tf-lab.yml`

## 训练时长

#### 优化之前
ResNet50 

部分参数如下：

samples_per_epoch=2048
nb_epoch=50
nb_val_samples=1024

平均每个1个小时左右

#### 优化之后
ResNet50、InceptionV3、Xception、VGG16、VGG19
导出特征向量耗时平均每个20~30分钟

总共探索调整，花费时长大概在20小时

## 项目目录

```
.
├── CapstoneReport.md                   [报告markdown]
├── CapstoneReport.pdf                  [报告pdf]
├── PetOrNot.jpeg
├── README.bak.md
├── README.md                           [readme]
├── dlnd-tf-lab.yml                     [实验环境配置文件]
├── dogvscat.png
├── error_test                          [测试集里一些效果不好的文件]
├── error_train                         [训练集里的一些不好的文件]
├── media                               [md文件的图片]
├── proposal.md                         [开题报告md]
├── proposal.pdf                        [开题报告pdf]
├── submission_file                     [提交kaggle的结果文件]
├── tensorboard_log                     [训练的日志文件，供tensorboard可视化]
├── train_model.py                      [Resnet50模型训练脚本]
├── 机器学习纳米学位毕业项目--猫狗大战+优化版本.html
├── 机器学习纳米学位毕业项目--猫狗大战+优化版.ipynb
├── 机器学习纳米学位毕业项目--猫狗大战.html
└── 机器学习纳米学位毕业项目--猫狗大战.ipynb
```



## 提交检查

* [x] 报告文件
* [x] 数据预处理代码（jupyter notebook）
* [x] 模型训练代码（jupyter notebook）
* [x] notebook 导出的 html 文件
* [x] 包含使用的库，机器硬件，机器操作系统，训练时间等数据的 README 文档（建议使用 Markdown ）


