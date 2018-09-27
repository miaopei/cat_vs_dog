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
