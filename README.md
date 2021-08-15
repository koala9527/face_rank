## 来源
来源：https://github.com/lwpyh/face_mark
教程地址：https://blog.csdn.net/lwpyh/article/details/87902245  
需要：
python3.5及以上，keras2.2.4,face_recognition1.0.0,glob2 0.6  
face_mark文件夹下是后台颜值检测算法的整套代码。  
prepare.py:数据预处理代码  
train.py:训练神经网络代码  
predict.py:测试代码  
keras-flask-deploy-webapp文件夹下是具体如何实现前端网页demo的代码  
app.py:实现交互的核心部分  
注意将生成的h5文件放到model文件夹下  

## 注意
安装依赖比较麻烦，需要多尝试，搜索报错信息  
#pip install -i https://pypi.tuna.tsinghua.edu.cn/simple face_recognition  
#pip install dlib -i https://pypi.python.org/simple/  
keras>=2.2  

## fix
dockerfile已修复
增加未识别到人脸的报错返回
增加api接口
![效果图片](https://github.com/koala9527/face_rank/blob/main/%E5%BE%AE%E4%BF%A1%E6%88%AA%E5%9B%BE_20210808194904.png)
## todo
增加男女性别识别

