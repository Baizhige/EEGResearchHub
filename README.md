# 项目名称

这个项目是一个EEG分类模型的最小训练框架

## architectures 文件夹

`architectures` 文件夹内包含了 5 个使用 `PyTorch` 实现的深度神经网络模型。详细介绍如下：

### 简介

本项目中的这几个模型均主要用于运动想象(Motor Imagery)分类任务。它们的实现基于 `PyTorch` 框架，具有较好的可扩展性和易用性。EEGSym作者提供了基于`tensorflow`的模型代码，可点击下面原文中直接下载。

### 文件列表

- `model_standard_EEGNet.py`: 包含了EEGNet模型的实现代码。[论文](https://arxiv.org/abs/1611.08024) 
- `model_standard_InceptionEEG.py`: 包含了InceptionEEG模型的实现代码。[论文](https://ieeexplore.ieee.org/document/9311146) 
- `model_standard_Deep.py`: 包含了DeepConvNet模型的实现代码。[论文](https://arxiv.org/abs/1703.05051) [代码参考](https://github.com/braindecode/braindecode/tree/master/braindecode/models)
- `model_standard_ShallowFBCSPNet.py`: 包含了ShallowConvNet模型的实现代码。[论文](https://arxiv.org/abs/1703.05051) [代码参考](https://github.com/braindecode/braindecode/tree/master/braindecode/models)
- `model_standard_EEGSym.py`: 包含了EEGSym模型的实现代码。[论文](https://ieeexplore.ieee.org/document/9807323) 
- `utils`: 包含了DeepConvNet模型和ShallowConvNet模型的一些依赖。

### 使用方法

将模型放入到你的框架里面即可，模型的输入输出在每个模型文件中有demo，输入的shape是：`[batch_size, n, num_channels, num_sampling]`， `n`一般是1，如果有其他操作，比如两个trial作为一个样本，你可以设置为2。

## utils 文件夹
- `data_loader.py`: 一个类，实现pytorch专用的EEG数据加载功能。
- `parse_config.py`: 一个类，实现读取训练参数的功能。
- `test.py`: 一个函数，包含了模型的测试代码。

### 使用方法

1. 准备数据：将源域和目标域的数据分别保存在不同的文件夹中，并确保每个文件夹内的图像都属于同一个类别。
2. 确定超参数：在 `main.py` 中设置网络超参数，如迭代次数、学习率，模型保存路径等, 然后运行，训练模型。
3. 训练完成后：运行 `evaluate.py` 测试模型。


## record 文件夹
存放训练结果

## demo_data 文件夹
存放示例数据集

## 版权信息

本项目代码遵循 [MIT LICENSE](LICENSE) 许可证。

