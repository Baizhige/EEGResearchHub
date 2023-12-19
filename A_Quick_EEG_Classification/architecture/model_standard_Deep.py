# Revision: Chengxuan Qin <Chengxuan.Qin@outlook.com>
# Description:
# Robin Schirrmeister <robintibor@gmail.com> developed "Deep4Net".
# I simply modified the input and output shapes to maintain uniformity with other networks.
# The updated class is now called "DeepNetIRT".

import numpy as np
from torch import nn
from torch.nn import init
from torch.nn.functional import elu
from collections import OrderedDict
from .utils.model_standard_deep4_modules import Expression, AvgPool2dWithConv, Ensure4d
from .utils.model_standard_deep4_functions import identity, transpose_time_to_spat, squeeze_final_output
from .utils.model_standard_deep4_util import np_to_th
import torch

class DeepNetIRT(nn.Sequential):
    """Deep ConvNet model from Schirrmeister et al 2017.
    The code was re-refactored by Chengxuan.Qin@outlook.com
    Model described in [Schirrmeister2017]_.
    Parameters
    ----------
    num_channel : int
     Number of EEG input channels.
    num_class: int
        Number of classes to predict (number of output filters of last layer).
    len_window: int | None
        Only used to determine the length of the last convolutional kernel if
        final_conv_length is "auto".
    final_conv_length: int | str
        Length of the final convolution layer.
        If set to "auto", input_window_samples must not be None.
    n_filters_time: int
        Number of temporal filters.
    n_filters_spat: int
        Number of spatial filters.
    filter_time_length: int
        Length of the temporal filter in layer 1.
    pool_time_length: int
        Length of temporal pooling filter.
    pool_time_stride: int
        Length of stride between temporal pooling filters.
    n_filters_2: int
        Number of temporal filters in layer 2.
    filter_length_2: int
        Length of the temporal filter in layer 2.
    n_filters_3: int
        Number of temporal filters in layer 3.
    filter_length_3: int
        Length of the temporal filter in layer 3.
    n_filters_4: int
        Number of temporal filters in layer 4.
    filter_length_4: int
        Length of the temporal filter in layer 4.
    first_conv_nonlin: callable
        Non-linear activation function to be used after convolution in layer 1.
    first_pool_mode: str
        Pooling mode in layer 1. "max" or "mean".
    first_pool_nonlin: callable
        Non-linear activation function to be used after pooling in layer 1.
    later_conv_nonlin: callable
        Non-linear activation function to be used after convolution in later layers.
    later_pool_mode: str
        Pooling mode in later layers. "max" or "mean".
    later_pool_nonlin: callable
        Non-linear activation function to be used after pooling in later layers.
    drop_prob: float
        Dropout probability.
    split_first_layer: bool
        Split first layer into temporal and spatial layers (True) or just use temporal (False).
        There would be no non-linearity between the split layers.
    batch_norm: bool
        Whether to use batch normalisation.
    batch_norm_alpha: float
        Momentum for BatchNorm2d.
    stride_before_pool: bool
        Stride before pooling.
    References
    ----------
    .. [Schirrmeister2017] Schirrmeister, R. T., Springenberg, J. T., Fiederer,
       L. D. J., Glasstetter, M., Eggensperger, K., Tangermann, M., Hutter, F.
       & Ball, T. (2017).
       Deep learning with convolutional neural networks for EEG decoding and
       visualization.
       Human Brain Mapping , Aug. 2017.
       Online: http://dx.doi.org/10.1002/hbm.23730
    """

    def __init__(
            self,
            num_channel,
            num_class,
            len_window,
            final_conv_length='auto',
            n_filters_time=25,
            n_filters_spat=25,
            filter_time_length=10,
            pool_time_length=3,
            pool_time_stride=3,
            n_filters_2=50,
            filter_length_2=10,
            n_filters_3=100,
            filter_length_3=10,
            n_filters_4=200,
            filter_length_4=10,
            first_conv_nonlin=elu,
            first_pool_mode="max",
            first_pool_nonlin=identity,
            later_conv_nonlin=elu,
            later_pool_mode="max",
            later_pool_nonlin=identity,
            drop_prob=0.5,
            split_first_layer=True,
            batch_norm=True,
            batch_norm_alpha=0.1,
            stride_before_pool=False
    ):
        super().__init__()
        self.in_chans = num_channel
        self.n_classes = num_class
        self.input_window_samples = len_window
        self.final_conv_length = final_conv_length
        self.n_filters_time = n_filters_time
        self.n_filters_spat = n_filters_spat
        self.filter_time_length = filter_time_length
        self.pool_time_length = pool_time_length
        self.pool_time_stride = pool_time_stride
        self.n_filters_2 = n_filters_2
        self.filter_length_2 = filter_length_2
        self.n_filters_3 = n_filters_3
        self.filter_length_3 = filter_length_3
        self.n_filters_4 = n_filters_4
        self.filter_length_4 = filter_length_4
        self.first_conv_nonlin = first_conv_nonlin
        self.first_pool_mode = first_pool_mode
        self.first_pool_nonlin = first_pool_nonlin
        self.later_conv_nonlin = later_conv_nonlin
        self.later_pool_mode = later_pool_mode
        self.later_pool_nonlin = later_pool_nonlin
        self.drop_prob = drop_prob
        self.split_first_layer = split_first_layer
        self.batch_norm = batch_norm
        self.batch_norm_alpha = batch_norm_alpha
        self.stride_before_pool = stride_before_pool

        self._build_feature_extractor()
        self._build_feature_classifier()
        self._initialize_weights()

    def _build_feature_extractor(self):
        # Feature extractor construction
        pool_class_dict = dict(max=nn.MaxPool2d, mean=AvgPool2dWithConv)
        first_pool_class = pool_class_dict[self.first_pool_mode]
        later_pool_class = pool_class_dict[self.later_pool_mode]

        feature_layers = []

        feature_layers.append(("ensuredims", Ensure4d()))
        feature_layers.append(("dimshuffle", Expression(transpose_time_to_spat)))

        if self.split_first_layer:
            feature_layers.append(
                ("conv_time",
                 nn.Conv2d(
                     1, self.n_filters_time, (self.filter_time_length, 1), stride=1
                 ))
            )
            feature_layers.append(
                ("conv_spat",
                 nn.Conv2d(
                     self.n_filters_time, self.n_filters_spat,
                     (1, self.in_chans), stride=(self._get_conv_stride(), 1),
                     bias=not self.batch_norm
                 ))
            )
            n_filters_conv = self.n_filters_spat
        else:
            feature_layers.append(
                ("conv_time",
                 nn.Conv2d(
                     self.in_chans, self.n_filters_time,
                     (self.filter_time_length, 1), stride=(self._get_conv_stride(), 1),
                     bias=not self.batch_norm
                 ))
            )
            n_filters_conv = self.n_filters_time

        if self.batch_norm:
            feature_layers.append(
                ("bnorm",
                 nn.BatchNorm2d(n_filters_conv, momentum=self.batch_norm_alpha, affine=True, eps=1e-5))
            )

        feature_layers.append(("conv_nonlin", Expression(self.first_conv_nonlin)))
        feature_layers.append(
            ("pool",
             first_pool_class(kernel_size=(self.pool_time_length, 1), stride=(self._get_pool_stride(), 1)))
        )
        feature_layers.append(("pool_nonlin", Expression(self.first_pool_nonlin)))

        self._add_conv_pool_block(feature_layers, n_filters_conv, self.n_filters_2, self.filter_length_2, 2)
        self._add_conv_pool_block(feature_layers, self.n_filters_2, self.n_filters_3, self.filter_length_3, 3)
        self._add_conv_pool_block(feature_layers, self.n_filters_3, self.n_filters_4, self.filter_length_4, 4)

        self.feature_extractor = nn.Sequential(OrderedDict(feature_layers))

    def _build_feature_classifier(self):
        # Feature classifier construction
        classifier_layers = []

        if self.final_conv_length == "auto":
            temp_model = nn.Sequential(OrderedDict(self.feature_extractor.named_children()))
            out = temp_model(np_to_th(np.ones((1, self.in_chans, self.input_window_samples, 1), dtype=np.float32)))
            n_out_time = out.cpu().data.numpy().shape[2]
            self.final_conv_length = n_out_time

        classifier_layers.append(
            ("conv_classifier",
             nn.Conv2d(self.n_filters_4, self.n_classes, (self.final_conv_length, 1), bias=True))
        )
        classifier_layers.append(("squeeze", Expression(squeeze_final_output)))

        self.classifier = nn.Sequential(OrderedDict(classifier_layers))

    def _initialize_weights(self):
        # Initialize weights for feature extractor
        for name, module in self.feature_extractor.named_modules():
            if isinstance(module, nn.Conv2d):
                init.xavier_uniform_(module.weight, gain=1)
                if module.bias is not None:
                    init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                init.constant_(module.weight, 1)
                init.constant_(module.bias, 0)

        # Initialize weights for feature classifier
        for name, module in self.classifier.named_modules():
            if isinstance(module, nn.Conv2d):
                init.xavier_uniform_(module.weight, gain=1)
                if module.bias is not None:
                    init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                init.constant_(module.weight, 1)
                init.constant_(module.bias, 0)

    def _get_conv_stride(self):
        return self.pool_time_stride if self.stride_before_pool else 1

    def _get_pool_stride(self):
        return 1 if self.stride_before_pool else self.pool_time_stride

    def _add_conv_pool_block(self, feature_layers, n_filters_before, n_filters, filter_length, block_nr):
        suffix = f"_{block_nr}"
        feature_layers.append(
            ("drop" + suffix, nn.Dropout(p=self.drop_prob))
        )
        feature_layers.append(
            ("conv" + suffix,
             nn.Conv2d(
                 n_filters_before, n_filters, (filter_length, 1), stride=(self._get_conv_stride(), 1),
                 bias=not self.batch_norm
             ))
        )
        if self.batch_norm:
            feature_layers.append(
                ("bnorm" + suffix,
                 nn.BatchNorm2d(n_filters, momentum=self.batch_norm_alpha, affine=True, eps=1e-5))
            )
        feature_layers.append(("nonlin" + suffix, Expression(self.later_conv_nonlin)))

        pool_class = dict(max=nn.MaxPool2d, mean=AvgPool2dWithConv)[self.later_pool_mode]
        feature_layers.append(
            ("pool" + suffix,
             pool_class(kernel_size=(self.pool_time_length, 1), stride=(self._get_pool_stride(), 1)))
        )
        feature_layers.append(("pool_nonlin" + suffix, Expression(self.later_pool_nonlin)))

    def forward(self, input_data):
        # Define the forward pass
        input_data = input_data.type(torch.cuda.FloatTensor)
        input_data = input_data.permute(0, 2, 3, 1)
        features = self.feature_extractor(input_data)
        output = self.classifier(features)
        return output




if __name__ == "__main__":
    from thop import profile

    net = DeepNetIRT(num_channel=32, num_class=2, len_window=480).cuda()
    test = torch.rand(1, 1, 32, 480).cuda()
    flops, params = profile(net, inputs=(test,), )
    print("Parameters:{0}, Flops:{1}".format(params, flops))
    out = net(test)
    print(out.size())
