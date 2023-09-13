# EEG Data Augmentation Project

## 简介

本项目主要用于EEG（脑电图）数据增强，包括了多种EEG数据增强技术如SVG。该项目结构主要由两个目录组成：`config` 和 `tools`。
## 目录结构
config/
XXX_channel_pos.npy
XXX_channel_name.npy
...
tools/
SVG.py
...
### `tools` 目录
该目录存放了数据增强的代码文件，包括SVG算法实现。代码文件都附带有详细的注释，以帮助理解各种算法的实现。

**更多信息：** 查看[作者的论文](https://ieeexplore.ieee.org/abstract/document/10248038)。

### `config` 目录

该目录主要存放与EEG脑电帽电极位置相关的配置信息。

- `XXX_channel_pos`：存放所有电极的位置信息。
- `XXX_channel_name`：存放所有电极的名称。
- 数据存放的顺序与数据集的通道顺序一致。

**支持的数据集：** 包括但不限于 PhysioNet Motor Imagery，Meng，BCIIV2A，kaggle Motor Imagery。

**自定义支持：** 如果你需要在其他的数据集中使用，你需要自己制作一份包含电极信息的 `.npy` 文件。

**参考：** 如果你需要引用，请引用一下这篇论文：
`@ARTICLE{10248038,
  author={Qin, Chengxuan and Yang, Rui and Huang, Mengjie and Liu, Weibo and Wang, Zidong},
  journal={IEEE Transactions on Neural Systems and Rehabilitation Engineering}, 
  title={Spatial Variation Generation Algorithm for Motor Imagery Data Augmentation: Increasing the Density of Sample Vicinity}, 
  year={2023},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TNSRE.2023.3314679}}`



