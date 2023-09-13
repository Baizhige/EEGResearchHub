# EEG Data Augmentation Project

## Introduction

This project primarily focuses on EEG (Electroencephalogram) data augmentation and incorporates various EEG data enhancement techniques like SVG. The project mainly consists of two directories: `config` and `tools`.

## Directory Structure
config/
XXX_channel_pos.npy
XXX_channel_name.npy
...
tools/
SVG.py
...
### `tools` Directory

This directory contains code files related to data augmentation, including the implementation of the SVG algorithm. All code files come with detailed comments to help understand the implementation of various algorithms.

**For more information:** Check out [the author's paper](https://ieeexplore.ieee.org/abstract/document/10248038).

### `config` Directory

This directory primarily contains configuration information related to the EEG electrode cap positions.

- `XXX_channel_pos`: Contains positional information of all electrodes.
- `XXX_channel_name`: Contains names of all electrodes.
- The data is stored in an order consistent with the channel order of the dataset.

**Supported Datasets:** This includes but is not limited to PhysioNet Motor Imagery, Meng, BCIIV2A, Kaggle Motor Imagery datasets.

**Custom Support:** If you need to use this on other datasets, you will need to create your own `.npy` file containing the electrode information.

**Reference**: If you wish to cite an article for reference, kindly consider citing the following publication:

`@ARTICLE{10248038, author={Qin, Chengxuan and Yang, Rui and Huang, Mengjie and Liu, Weibo and Wang, Zidong}, journal={IEEE Transactions on Neural Systems and Rehabilitation Engineering},  title={Spatial Variation Generation Algorithm for Motor Imagery Data Augmentation: Increasing the Density of Sample Vicinity},  year={2023}, volume={}, number={}, pages={1-1}, doi={10.1109/TNSRE.2023.3314679}}`
