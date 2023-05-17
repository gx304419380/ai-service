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


---

###2023-05-17 更新：
> 增加对YOLOv8的支持，djl升级到0.21.0，该版本对应的pytorch为
> 
> 需要注意的是：这个版本中使用onnx模型可能会报错：
> 
> ``` model_load_utils.h:57 onnxruntime::model_load_utils::ValidateOpsetForDomain ONNX Runtime only *guarantees* support for models stamped with official released onnx opset versions. Opset 17 is under development and support for this is limited. The operator schemas and or other functionality may change before next ONNX release and in this case ONNX Runtime will not guarantee backward compatibility. Current official support for domain ai.onnx is till opset 16. ```
> 
> 如果想继续使用onnx模型，有两种方案，一种是回退djl到0.19.0版本；
> 
> 第二种是导出onnx时设置参数opset为16，代码如下：
>
>```shell
>from ultralytics import YOLO
>model = YOLO("pathToYourModel/best.pt")
>model.export(format='onnx', opset=16)
>```
>
> 自测时发现onnx使用gpu会报错，暂时未找到原因，如果使用gpu执行yolo模型的话，建议采用torchscript格式的模型，导出模型代码如下：
> 
> ```shell
> model = YOLO("pathToYourModel/best.pt")
> //导出为gpu的模型
> model.export(device=0)
> //导出为cpu的模型
> model.export(device='cpu')
>```
> 
> 
> 
> 
> 

### 附cuda和djl版本

> nvidia-smi
```shell
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 516.94       Driver Version: 516.94       CUDA Version: 11.7     |
|-------------------------------+----------------------+----------------------+
| GPU  Name            TCC/WDDM | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA T600 Lap... WDDM  | 00000000:01:00.0 Off |                  N/A |
| N/A   56C    P8    N/A /  N/A |      0MiB /  4096MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

> nvcc -V
```shell
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Wed_Jun__8_16:59:34_Pacific_Daylight_Time_2022
Cuda compilation tools, release 11.7, V11.7.99
Build cuda_11.7.r11.7/compiler.31442593_0
```
