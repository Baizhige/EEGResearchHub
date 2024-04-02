# Revision: Chengxuan Qin <Chengxuan.Qin@outlook.com>
# Description:
# Vernon J Lawhern developed "EEGNet" in tensorflow, as shown in "https://github.com/vlawhern/arl-eegmodels".
# aliasvishnu developed "EEGNet" in pytroch, as shown in "https://github.com/aliasvishnu/EEGNet.
# I simply implement in pytorch the input and output shapes to maintain uniformity with other networks.
# The class is called "EEGNet".

import torch.nn as nn
import torch

class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout, kernel_size):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=(1, kernel_size), padding=0, groups=nin, bias=False)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class conv2d_weight_constraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(conv2d_weight_constraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(conv2d_weight_constraint, self).forward(x)

class EEGNet(nn.Module):
    """
    EEGNet: A Deep Convolutional Neural Network for EEG-based Motor Imagery Classification.

    Args:
    - kernel_size (int): Size of the convolutional kernel.
    - F1 (int): Number of filters in the first convolutional layer.
    - D (int): Depth factor for the second convolutional layer.
    - F2 (int): Number of filters in the second convolutional layer.
    - num_channel (int): Number of EEG channels.
    - num_class (int): Number of output classes.
    - len_window (int): Length of the input EEG window.

    Returns:
    - class_output (torch.Tensor): Predicted class scores.
    """

    def __init__(self, kernel_size=80, F1=8, D=2, F2=16, num_channel=64, num_class=2, len_window=480):
        super(EEGNet, self).__init__()
        # Feature extractor with two convolutional layers
        self.kernel_size = kernel_size
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.num_channel = num_channel
        self.num_classes = num_class
        self.len_window = len_window
        self.feature = nn.Sequential()
        # (N, 1, 64, 256)
        self.feature.add_module('f_conv1', nn.Conv2d(1, self.F1, (1, self.kernel_size), padding=0, bias=False))
        self.feature.add_module('f_padding1',
                                nn.ZeroPad2d((int(self.kernel_size / 2) - 1, int(self.kernel_size / 2), 0, 0)))
        self.feature.add_module('f_batchnorm1', nn.BatchNorm2d(self.F1, False))
        # (N, F1, 64, 256)
        # self.feature.add_module('f_conv2', nn.Conv2d(self.F1, self.F1 * self.D, (self.num_channel, 1), groups=8))
        # DepthwiseConv2D
        self.feature.add_module('f_depthwiseconv2d', conv2d_weight_constraint(in_channels=self.F1,out_channels=self.F1 * self.D,kernel_size=(self.num_channel, 1), max_norm=1, bias=False,groups=self.F1,padding=0))
        self.feature.add_module('f_batchnorm2', nn.BatchNorm2d(self.F1 * self.D, False))
        self.feature.add_module('f_ELU2', nn.ELU())
        # (N, F1 * D, 1, 256)
        self.feature.add_module('f_Pooling3', nn.AvgPool2d(kernel_size=(1, 4)))
        self.feature.add_module('f_dropout3', nn.Dropout(p=0.25))
        # (N, F1 * D, 1, 256/4)
        self.feature.add_module('f_conv4',
                                depthwise_separable_conv(self.F1 * self.D, self.F2, int(self.kernel_size / 4)))
        self.feature.add_module('f_padding4',
                                nn.ZeroPad2d((int(self.kernel_size / 8) - 1, int(self.kernel_size / 8), 0, 0)))
        self.feature.add_module('f_batchnorm4', nn.BatchNorm2d(self.F2, False))
        self.feature.add_module('f_ELU4', nn.ELU())
        self.feature.add_module('f_Pooling4', nn.AvgPool2d(kernel_size=(1, 8)))
        self.feature.add_module('f_dropout4', nn.Dropout(p=0.25))
        # (N, F2, 1, 256/32)

        # Classifier with three fully connected layers
        __hidden_feature__ = self.feature(torch.rand(1, 1, self.num_channel, self.len_window))
        self.__hidden_len__ = __hidden_feature__.shape[1] * __hidden_feature__.shape[2] * __hidden_feature__.shape[3]
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(self.__hidden_len__, 128))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(128))
        self.class_classifier.add_module('c_fc2', nn.Linear(128, 64))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(64))
        self.class_classifier.add_module('c_fc3', nn.Linear(64, self.num_classes))

    def forward(self, input_data):
        """
        Forward pass of the EEGNet model.

        Args:
        - input_data (torch.Tensor): Input EEG data of shape (batch_size, num_channel, len_window).

        Returns:
        - class_output (torch.Tensor): Predicted class scores.
        """
        input_data = input_data.type(torch.cuda.FloatTensor)
        feature = self.feature(input_data)
        feature = feature.view(-1, self.__hidden_len__)
        class_output = self.class_classifier(feature)
        return class_output


if __name__ == "__main__":
    # Test the EEGNet model with random input
    from thop import profile

    net = EEGNet(num_channel=64, num_class=2, len_window=480).cuda()
    test = torch.rand(32, 1, 64, 480)
    flops, params = profile(net, inputs=(test,), )
    print("Parameters:{0}, Flops:{1}".format(params, flops))
    output = net(test)
    print(output.size())
