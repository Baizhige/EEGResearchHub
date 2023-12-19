# 一个简单的EEG分类任务
这个GitHub仓库是一个通用的EEG神经网络训练框架。部分代码来自作者即将发布的私有仓库`EEG-IRT`。🔥🔥

## 快速开始
0. 通过`$ conda create --name <你的环境名> --file <requirements.txt>`创建环境。在当前版本中，CUDA已经嵌入到环境中。使用此命令创建环境时，无需额外安装或修改您的CUDA环境。
1. 编辑`config.ini`以修改训练参数。
2. 运行`main.py`。

## `architectures` 目录

`architectures`目录包括五个使用`PyTorch`实现的深度神经网络模型。具体描述如下：

### 简介
这个项目中的模型主要用于运动想象分类任务。它们基于`PyTorch`框架实现，具有优秀的可扩展性和易用性。EEGSym的作者提供了基于`tensorflow`的模型代码，可直接从原文中提供的链接下载。

### 文件列表
- `model_standard_EEGNet.py`：包含EEGNet模型的实现代码。[研究论文](https://arxiv.org/abs/1611.08024)
- `model_standard_InceptionEEG.py`：包含InceptionEEG模型的实现代码。[研究论文](https://ieeexplore.ieee.org/document/9311146)
- `model_standard_Deep.py`：包含DeepConvNet模型的实现代码。[研究论文](https://arxiv.org/abs/1703.05051) [代码参考](https://github.com/braindecode/braindecode/tree/master/braindecode/models)
- `model_standard_ShallowFBCSPNet.py`：包含ShallowConvNet模型的实现代码。[研究论文](https://arxiv.org/abs/1703.05051) [代码参考](https://github.com/braindecode/braindecode/tree/master/braindecode/models)
- `model_standard_EEGSym.py`：包含EEGSym模型的实现代码。[研究论文](https://ieeexplore.ieee.org/document/9807323)
- `utils`：包含DeepConvNet和ShallowConvNet模型所需的依赖项。

### 使用方法
要使用这些模型，只需将它们集成到您的框架中。每个模型文件中都有模型输入和输出的演示。输入形状为：`[batch_size, n, num_channels, num_sampling]`。`n`通常为1。如果有其他操作，例如两个试验作为一个样本，您可以将`n`设置为2。

## `utils` 目录
- `data_loader.py`：一个实现PyTorch特定的EEG数据加载功能的类。
- `parse_config.py`：一个实现读取训练参数的功能的类。
- `test.py`：包含模型的测试代码的函数。

## `record` 目录
存储训练结果

## `demo_data` 目录
存储示例数据集。它目前包括BCICIV 2a数据集。`DL_BCICIV2a_c2RL_640_no_filter_pro`文件夹存储了按5折交叉验证划分的数据索引，这些索引也符合跨用户研究的标准。例如，MI_train0、MI_eval0和MI_test0分别代表第0折的训练集、验证集和测试集的数据索引。

## 版权信息
该项目的代码遵循[MIT许可](LICENSE)。
