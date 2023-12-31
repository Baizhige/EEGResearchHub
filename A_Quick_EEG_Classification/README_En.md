# A Simple EEG Classification Task
This GitHub repository is a general-purpose EEG neural network training framework. Some of the code is derived and reorganized from the author's upcoming private repository ` EEG-IRT` .🔥 🔥 

Quick Start: 

0. Create environment by  ` $ conda create --name <your env name> --file <requirements.txt>` . In the current version, CUDA has been embedded into the environment. When creating the environment with this command, there's no need for an external installation or modification of your CUDA environment
1. Edit ` config.ini` to modify training parameters.
2. Run ` main.py` .
## `architectures` Directory

The `architectures` directory includes five deep neural network models implemented using `PyTorch`. Detailed descriptions are as follows:

### Introduction

The models in this project are mainly used for Motor Imagery classification tasks. They are implemented based on the `PyTorch` framework, offering excellent scalability and ease of use. The author of EEGSym provides the model code based on `tensorflow`, which can be directly downloaded from the links provided in the original text.

### File List

- `model_standard_EEGNet.py`: Contains the implementation code for the EEGNet model. [Research Paper](https://arxiv.org/abs/1611.08024)
- `model_standard_InceptionEEG.py`: Contains the implementation code for the InceptionEEG model. [Research Paper](https://ieeexplore.ieee.org/document/9311146)
- `model_standard_Deep.py`: Contains the implementation code for the DeepConvNet model. [Research Paper](https://arxiv.org/abs/1703.05051) [Code Reference](https://github.com/braindecode/braindecode/tree/master/braindecode/models)
- `model_standard_ShallowFBCSPNet.py`: Contains the implementation code for the ShallowConvNet model. [Research Paper](https://arxiv.org/abs/1703.05051) [Code Reference](https://github.com/braindecode/braindecode/tree/master/braindecode/models)
- `model_standard_EEGSym.py`: Contains the implementation code for the EEGSym model. [Research Paper](https://ieeexplore.ieee.org/document/9807323)
- `utils`: Contains dependencies for the DeepConvNet and ShallowConvNet models.

### Usage

To use the models, simply incorporate them into your framework. A demo for the model input and output is available in each model file. The input shape is: `[batch_size, n, num_channels, num_sampling]`. `n` is typically 1. If there are other operations, such as two trials as one sample, you can set `n` to 2.

## `utils` Directory
- `data_loader.py`: A class that implements a PyTorch-specific EEG data loading function.
- `parse_config.py`: A class that implements a function for reading training parameters.
- `test.py`: A function containing the model's test code.

## `record` Directory
Stores training results

## `demo_data` Directory
Stores example datasets. It currently consists of BCICIV 2a dataset. The `DL_BCICIV2a_c2RL_640_no_filter_pro` folder stores data indices divided by 5-fold cross-validation, these indices also abide by the criteria of cross-subject studies. For instance, MI_train0, MI_eval0, and MI_test0 represent the data indices for the training set, validation set, and test set of the 0th fold, respectively. 

## Copyright Information

The code of this project follows the [MIT LICENSE](LICENSE).
