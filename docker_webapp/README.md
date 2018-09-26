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

## Docker 使用

构建 docker image


```shell
$ cd webapp
$ docker build -t cat_vs_dog_webapp:1.0.0 .
```

生成容器

```shell
#$ docker run -it --rm cat_vs_dog_webapp:1.0.0
# 通过-e来设定任意的环境变量，甚至覆盖已经存在的环境变量，或者是在Dockerfile中通过ENV设定的环境变量。
#$ docker run -e MODEL_PATH=models/your_model.h5 -p 5000:5000 
# -v, --volume=[]   给容器挂载存储卷，挂载到容器的某个目录
# $ docker run -e MODEL_PATH=/mnt/models/your_model.h5  -v volume-name:/app/models -p 5000:5000 keras_flask
$ docker run -p 5000:5000 cat_vs_dog_webapp:1.0.0
```

到此就可以在浏览器中输入 [http://localhost:5000](http://localhost:5000) 就可以使用网页对导入的猫狗图片做预测了。

### note

RUN在Dockerfile构建镜像的过程(Build)中运行，最终被commit的到镜像。

ENTRYPOINT和CMD在容器运行(run、start)时运行。