FROM ubuntu:18.04

# Install Python.
RUN \
    apt-get update && \
    apt-get install -y python python-dev python-pip python-virtualenv && \
    pip install --upgrade pip && \
    pip install --upgrade setuptools && \
    rm -rf /var/lib/apt/lists/*

WORKDIR app

# reqs from file, to speed up dev iteration
RUN pip install Werkzeug Flask numpy Keras gevent pillow h5py tensorflow>2.2

COPY . .

RUN pip install -r requirements.txt

ENTRYPOINT [ "python" , "app.py"]

#pip install -i https://pypi.tuna.tsinghua.edu.cn/simple face_recognition
#pip install dlib -i https://pypi.python.org/simple/
#keras 2.2.4