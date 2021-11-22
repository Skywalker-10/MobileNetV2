代码运行说明：
代码目录里只有两个文件，一个为train.py，一个为eval.py。
运行环境为mindarts的notebook，将代码复制到notebook中，在notebook上挂载并行文件系统，并行文件系统中为RP2K数据集，挂载完成后，训练集的路径为/data/RP2K/train，测试集路径为/data/RP2K/test。运行train.py前需执行 
！pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.3.0/Hub/any/mindspore_hub-1.3.0-py3-none-any.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
命令安装MindSpore Hub
运行train.py进行训练，一共训练60轮，产生60个ckpt文件
运行eval.py进行评估，加载最后一个ckpt文件，在测试集上进行测试。


代码说明:
模型基于MindSpore Hub提供的预训练模型，在ImageNet2012训练的MobileNetV2预训练模型，在预训练模型的基础上添加一个新的分类层对RP2K数据集进行训练。
