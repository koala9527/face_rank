from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import json
import requests
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
import keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import pickle
import face_recognition as fr
from flask_restful import Api, Resource, reqparse, marshal_with, fields

# Define a flask app
app = Flask(__name__)
app.config.update(RESTFUL_JSON=dict(ensure_ascii=False))
api = Api(app)
# Model saved with Keras model.save()
MODEL_PATH = 'models/face_rank_model.h5'

# Load your trained model
# model=make_network()
# model.load_weights (MODEL_PATH)
model=load_model(MODEL_PATH)
model.summary()
model._make_predict_function()          # Necessary
print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
# from keras.applications.resnet50 import ResNet50
# model = ResNet50(weights='imagenet') #zhushidiaole
# print('Model loaded. Check http://127.0.0.1:5000/') #zhushidiaole


def model_predict(img_path, model):

    # Preprocessing the image
    image = fr.load_image_file(img_path)
    encs = fr.face_encodings(image)
    # if len(encs) != 1:
    #     print("Find %d faces in %s" % (len(encs), img_path))
    #     continue

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    # x = preprocess_input(x, mode='caffe')  wozhushidiaole

    preds = model.predict(np.array(encs))
    print(type(preds))
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

class Rankapi(Resource):
    def post(self, message=None):
        if request.method == 'POST':
            res = request.get_data()
            res = json.loads(res)
            if 'face_url' in res:
                ret = download_img(res['face_url'])
                if not ret:
                    return_data = {"code":400,"msg":"图片下载失败"}
                else:
                    basepath = os.path.dirname(__file__)
                    img_name = res['face_url'].split('/').pop()
                    file_path=  os.path.join(basepath, 'uploads', img_name)
                    try:
                        preds = model_predict(file_path, model)
                    except:
                        return {"code":403,"msg":"未识别到人脸"}
                    print(t=round(preds[0][0]*2,3))
                    print('t',t)
                    return_data = {"code":200,"data":{"score":str(t)},"msg":"预测成功"}
                    return return_data
            else:
                return_data = {"code":401,"msg":"图片参数缺失"}
        else:
            return_data = {"code":402,"msg":"请求方法有误，请使用POST方法"}
        return return_data, 200

api.add_resource(Rankapi, '/rankapi')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        try:
            preds = model_predict(file_path, model)
            t=round(preds[0][0]*2,3)
            print('t',t)
            # print(type(t))
            # result=t.tolist()
            # print(result)
            # print(type(result))
            result=str(t)
            print(result)
            result1 = str(result)
        except:
            result1="未识别到人脸"
        # preds=model_predict("C://l/c1.jpg",model)
        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        # pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        # result = str(pred_class[0][0][1])               # Convert to string

        return result1
    return None

def download_img(img_url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.105 Safari/537.36'
    }
    r = requests.get(img_url, headers=headers, stream=True)

    if r.status_code == 200:
        # 截取图片文件名
        basepath = os.path.dirname(__file__)
        img_name = img_url.split('/').pop()
        file_path = os.path.join(
            basepath, 'uploads', img_name)
        with open(file_path, 'wb') as f:
            f.write(r.content)
        return True

if __name__ == '__main__':
    # app.run(port=5002, debug=True)

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()

