# Revision: Chengxuan Qin <Chengxuan.Qin@outlook.com>
# Description:
# Robin Schirrmeister developed "ShallowFBCSPNet".
# I simply modified the input and output shapes to maintain uniformity with other networks.
# The updated class is now called "ShallowFBCSPNetIRT".


import numpy as np
from torch import nn
from torch.nn import init
import torch
from .utils.model_standard_deep4_util import np_to_th
from .utils.model_standard_deep4_modules import Expression, Ensure4d
from .utils.model_standard_deep4_functions import (
    safe_log, square, transpose_time_to_spat, squeeze_final_output
)

class ShallowFBCSPNetIRT(nn.Module):
    """
    The code was re-refactored by Chengxuan.Qin@outlook.com
    Shallow ConvNet model from Schirrmeister et al 2017.
    Model described in [Schirrmeister2017]_.
    Parameters
    ----------
    num_channel : int
        Number of EEG input channels.
    num_class: int
        Number of classes to predict (number of output filters of last layer).
    input_window_samples: int | None
        Only used to determine the length of the last convolutional kernel if
        final_conv_length is "auto".
    n_filters_time: int
        Number of temporal filters.
    filter_time_length: int
        Length of the temporal filter.
    n_filters_spat: int
        Number of spatial filters.
    pool_time_length: int
        Length of temporal pooling filter.
    pool_time_stride: int
        Length of stride between temporal pooling filters.
    final_conv_length: int | str
        Length of the final convolution layer.
        If set to "auto", input_window_samples must not be None.
    conv_nonlin: callable
        Non-linear function to be used after convolution layers.
    pool_mode: str
        Method to use on pooling layers. "max" or "mean".
    pool_nonlin: callable
        Non-linear function to be used after pooling layers.
    split_first_layer: bool
        Split first layer into temporal and spatial layers (True) or just use temporal (False).
        There would be no non-linearity between the split layers.
    batch_norm: bool
        Whether to use batch normalisation.
    batch_norm_alpha: float
        Momentum for BatchNorm2d.
    drop_prob: float
        Dropout probability.
    References
    ----------
    .. [Schirrmeister2017] Schirrmeister, R. T., Springenberg, J. T., Fiederer,
       L. D. J., Glasstetter, M., Eggensperger, K., Tangermann, M., Hutter, F.
       & Ball, T. (2017).
       Deep learning with convolutional neural networks for EEG decoding and
       visualization.
       Human Brain Mapping , Aug. 2017.
       Online: https://dx.doi.org/10.1002/hbm.23730
    """
    def __init__(
            self,
            num_channel,
            num_class,
            len_window=None,
            n_filters_time=40,
            filter_time_length=25,
            n_filters_spat=40,
            pool_time_length=75,
            pool_time_stride=15,
            final_conv_length='auto',
            conv_nonlin=square,
            pool_mode="mean",
            pool_nonlin=safe_log,
            split_first_layer=True,
            batch_norm=True,
            batch_norm_alpha=0.1,
            drop_prob=0.5,
        ):
        super(ShallowFBCSPNetIRT, self).__init__()
        self.in_chans = num_channel
        self.n_classes = num_class
        self.input_window_samples = len_window
        self.n_filters_time = n_filters_time
        self.filter_time_length = filter_time_length
        self.n_filters_spat = n_filters_spat
        self.pool_time_length = pool_time_length
        self.pool_time_stride = pool_time_stride
        self.final_conv_length = final_conv_length
        self.conv_nonlin = conv_nonlin
        self.pool_mode = pool_mode
        self.pool_nonlin = pool_nonlin
        self.split_first_layer = split_first_layer
        self.batch_norm = batch_norm
        self.batch_norm_alpha = batch_norm_alpha
        self.drop_prob = drop_prob

        self._build_feature_extractor()          # build_feature_extractor
        self._build_feature_classifier()          # build_feature_classifier
        self._initialize_weights()     # initialize the weights of networks

    def _build_feature_extractor(self):
        self.ensuredims = Ensure4d()
        pool_class = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[self.pool_mode]

        # Temporal and Spatial Convolution
        if self.split_first_layer:
            self.conv_time = nn.Conv2d(
                1,
                self.n_filters_time,
                (self.filter_time_length, 1),
                stride=1,
            )
            self.conv_spat = nn.Conv2d(
                self.n_filters_time,
                self.n_filters_spat,
                (1, self.in_chans),
                stride=1,
                bias=not self.batch_norm,
            )
            n_filters_conv = self.n_filters_spat
        else:
            self.conv_time = nn.Conv2d(
                self.in_chans,
                self.n_filters_time,
                (self.filter_time_length, 1),
                stride=1,
                bias=not self.batch_norm,
            )
            n_filters_conv = self.n_filters_time

        # Batchnorm Layer
        if self.batch_norm:
            self.bnorm = nn.BatchNorm2d(
                n_filters_conv, momentum=self.batch_norm_alpha, affine=True
            )

        # Non-linear Expression
        self.conv_nonlin_exp = Expression(self.conv_nonlin)

        # Pooling Layer
        self.pool = pool_class(
            kernel_size=(self.pool_time_length, 1),
            stride=(self.pool_time_stride, 1),
        )

        # Non-linear Expression (after pooling layer)
        self.pool_nonlin_exp = Expression(self.pool_nonlin)

        # Dropout Layer
        self.drop = nn.Dropout(p=self.drop_prob)



        # Feature Extractor
        feature_layers = [
            self.ensuredims,
            Expression(transpose_time_to_spat) if self.split_first_layer else nn.Identity(),
            self.conv_time,
            self.conv_spat if self.split_first_layer else nn.Identity(),
            self.bnorm if self.batch_norm else nn.Identity(),
            self.conv_nonlin_exp,
            self.pool,
            self.pool_nonlin_exp,
            self.drop
        ]
        self.n_filters_conv = n_filters_conv
        self.feature_extractor = nn.Sequential(*feature_layers)


    def _build_feature_classifier(self):
        # Feature Classifier
        if self.final_conv_length == "auto":
            out = self.feature_extractor(
                np_to_th(
                    np.ones(
                        (1, self.in_chans, self.input_window_samples, 1),
                        dtype=np.float32,
                    )
                )
            )
            n_out_time = out.cpu().data.numpy().shape[2]
            self.final_conv_length = n_out_time
        self.conv_classifier = nn.Conv2d(
            self.n_filters_conv,
            self.n_classes,
            (self.final_conv_length, 1),
            bias=True
        )
        self.classifier = nn.Sequential(
            self.conv_classifier,
            Expression(squeeze_final_output)
        )

    def _initialize_weights(self):
        init.xavier_uniform_(self.conv_time.weight, gain=1)
        if self.split_first_layer or not self.batch_norm:
            if self.conv_time.bias is not None:
                init.zeros_(self.conv_time.bias)

        if self.split_first_layer:
            init.xavier_uniform_(self.conv_spat.weight, gain=1)
            if not self.batch_norm:
                init.zeros_(self.conv_spat.bias)

        if self.batch_norm:
            init.ones_(self.bnorm.weight)
            init.zeros_(self.bnorm.bias)

        init.xavier_uniform_(self.conv_classifier.weight, gain=1)
        init.zeros_(self.conv_classifier.bias)
    def forward(self, input_data):
        input_data = input_data.type(torch.cuda.FloatTensor)
        input_data = input_data.permute(0, 2, 3, 1)
        features = self.feature_extractor(input_data)
        output = self.classifier(features)
        return output

if __name__ == "__main__":
    from thop import profile

    net = ShallowFBCSPNetIRT(num_channel=32, num_class=2, len_window=480).cuda()
    test = torch.rand(1, 1, 32, 480).cuda()
    flops, params = profile(net, inputs=(test,), )
    print("Parameters:{0}, Flops:{1}".format(params, flops))
    out = net(test)
    print(out.size())
