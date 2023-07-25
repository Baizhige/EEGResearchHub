# Project Name

This project is a minimalist training framework for EEG (Electroencephalography) classification models.
Simple use: 
Run ` main.py` .
` config.ini` records training parameters.
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

### Usage

1. Data preparation: Save the source and target domain data in different folders, ensuring that all images in each folder belong to the same category.
2. Determine hyperparameters: Set network hyperparameters in `main.py`, such as the number of iterations, learning rate, model save path, etc., then run it to train the model.
3. After training is complete: Run `evaluate.py` to test the model.

## `record` Directory
Stores training results

## `demo_data` Directory
Stores example datasets

## Copyright Information

The code of this project follows the [MIT LICENSE](LICENSE).
