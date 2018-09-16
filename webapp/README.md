# cat_vs_dog
项目使用 Keras 和 Flask 搭建部署一个简单易用的深度学习图像网页应用，可以通过网页导入一张彩色猫或者狗的图片预测是猫或者狗的概率。

项目目录结构：

```python
.
├── README.md
├── ResNet50_image_predict.ipynb
├── app.py
├── environmert.yml
├── static
│   ├── css
│   │   └── main.css
│   └── js
│       └── main.js
├── templates
│   ├── base.html
│   └── index.html
├── models
│   └── ResNet50_catdog_model.h5
├── uploads
│   ├── test01.jpg
│   └── test02.jpg
└── webapp_image_predict.ipynb
```

## 环境搭建

```shell
$ conda env create -f environmert.yml
```

## 运行

```shell
$ python app.py
```

这时候用浏览器打开 <http://localhost:5088/> 就可以进行网页导入图片预测图片是狗的概率了。