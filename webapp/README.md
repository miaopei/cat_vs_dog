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

docker webapp构建过程


```shell
$ cd webapp
$ docker build -t webapp:1.0.0 .
$ docker run -d -p 5000:5000 webapp:1.0.0
$ docker tar webapp:1.0.0 miaowmiaow/webapp:1.0.0
$ docker login
$ docker push miaowmiaow/webapp:1.0.0
```

Dockerfile 内容如下

```dockerfile
FROM ubuntu:16.04

RUN rm /bin/sh && ln -s /bin/bash /bin/sh

#ENTRYPOINT [ "/bin/bash", "-c" ]

MAINTAINER MiaoPei <miaopei163@163.com.com>

# Install basic dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        wget \
        libopencv-dev \
        libsnappy-dev \
        python-dev \
        python-pip \
        tzdata \
        vim

# Install anaconda for python 3.6
RUN wget --quiet https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh && \
    echo "export PATH=/opt/conda/bin:$PATH" >> ~/.bashrc

# Set timezone
RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime

# Set locale
ENV LANG C.UTF-8 LC_ALL=C.UTF-8

WORKDIR app

COPY . .

#ADD environmert.yml .

RUN /opt/conda/bin/conda env create -f environmert.yml

RUN ["/bin/bash", "-c", "source /opt/conda/bin/activate webapp"]

ENTRYPOINT [ "/bin/bash", "-c", "source /opt/conda/bin/activate webapp && python app.py"]
```

如果想快速复现实验结果可以进行如下操作：

```shell
$ docker pull miaowmiaow/webapp:1.1.0
$ docker run -p 5000:5000 miaowmiaow/webapp:1.1.0
```

到此就可以在浏览器中输入 [http://localhost:5000](http://localhost:5000) 就可以使用网页对导入的猫狗图片做预测了。

### note

RUN在Dockerfile构建镜像的过程(Build)中运行，最终被commit的到镜像。

ENTRYPOINT和CMD在容器运行(run、start)时运行。