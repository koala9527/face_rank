FROM ubuntu

# Install Python.
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y software-properties-common  wget \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y python3.6 \
    && apt-get install -y python3-pip  \
    && apt install -y python3.6-dev \
    &&  wget https://bootstrap.pypa.io/get-pip.py \
    &&  python3.6 get-pip.py \
    &&  update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1 \
    &&  update-alternatives --config python3 \
    && pip3 install --upgrade pip \
    && pip3 install face_recognition_models -i https://pypi.python.org/simple/ \
    && pip3 install scikit-build -i https://pypi.python.org/simple/ \
    && pip3 install cmake \
    && pip3 install dlib -i https://pypi.python.org/simple/ \
    && pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple face_recognition \
    && pip3 install tensorflow==1.4.0 -i https://pypi.douban.com/simple \
    && pip3 install Werkzeug \
    && pip3 install Flask  \
    && pip3 install Flask-RESTful \
    && pip3 install numpy \
    && pip3 install Keras==2.2.4 \
    && pip3 install gevent \
    && pip3 install pillow \
    && pip3 install h5py==2.10.0 

ADD . /app
WORKDIR /app



ENTRYPOINT [ "python3" , "app.py"]

# docker build -t face_rank . 
# docker run --name face_rank -d -p 5000:5000 face_rank