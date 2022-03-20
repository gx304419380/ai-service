### 0x0 简介

DJL全称deep java library，官网[https://djl.ai](https://djl.ai) ，是一个可以让java程序员快速集成深度学习的框架，你无需对神经网络算法有多么深入的了解就可以快速搭建一个ai服务器。


本项目基于DJL+springboot开发，OCR推理引擎使用paddle+pytorch，对象检测推理引擎使用onnx+pytorch；
OCR深度学习模型采用百度paddle的ocr模型，支持快速识别和精确识别两种类型；对象检测模型使用yolov5。


项目已发布至github：https://github.com/gx304419380/ai-service

**注意：若使用gpu运算，你的电脑需要安装cuda环境，具体方法请自行百度**

### 0x1 项目介绍

项目共有两个模块：ocr和yolo，其中ocr用于文字识别（Optical Character Recognition），yolo模块用于图片对象检测，具体如下图所示：
![image.png](https://upload-images.jianshu.io/upload_images/13277366-7d6d8f12e4e6c957.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

项目启动后可以进入swagger进行接口使用 http://localhost:8080/swagger-ui/：
![image.png](https://upload-images.jianshu.io/upload_images/13277366-c2dc2ec3a8437690.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

ocr相关功能测试图如下：
![ocr.png](https://upload-images.jianshu.io/upload_images/13277366-090e53216f35db04.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

yolo对象检测相关功能测试图如下：
![微信截图_20220319151847.png](https://upload-images.jianshu.io/upload_images/13277366-c5fa962cdd70bd9c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

