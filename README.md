## 简介

原有模型是编码器解码器架构，其编码器基于ResNet提取了512D的cnn特征，为了使用该cnn特征，故修改pre_img.py以及models/model.py，得到编码器的512D的cnn特征。
仓库名HGA-STR-STVQA是因为我要在STVQA场景文本视觉问答任务下应用该技术。



## 环境部署

可以参考原文[HGA-STR](https://github.com/luyang-NWPU/HGA-STR)进行部署，如果不成功，可以按照如下命令部署：

`python=3.9.12`

`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

`pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple six lmdb`



## 使用

cd /your_path/HGA-STR
python pre_img.py



## 结果

对于一张图片，可以得到其512D的cnn特征。
