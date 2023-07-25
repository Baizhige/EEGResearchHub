# Revision: Chengxuan Qin <Chengxuan.Qin@outlook.com>
# Description:
# S. Pérez-Velasco et.al developed "EEGInception" in tensorflow, as shown in https://github.com/Serpeve/EEGSym
# I simply implement in pytorch the input and output shapes to maintain uniformity with other networks.
# The class is called "EEGSym".
import torch.nn as nn
import torch

class Symmetric_layer(nn.Module):
    def __init__(self, right_idx, left_idx):
        super(Symmetric_layer, self).__init__()
        self.right_idx = right_idx
        self.left_idx = left_idx

    def forward(self, x):
        '''
        :param x: B N C T
        :return: B N C T 2
        '''
        s_right = x[:, :, self.right_idx, :].unsqueeze(4)
        s_left = x[:, :, self.left_idx, :].unsqueeze(4)
        return torch.cat((s_right, s_left), dim=4)


class EEGSym_inception_block(nn.Module):
    def __init__(self, inChannel, hiddenChannel, outChannel, kernel_size, padding, dropoutRate=0.3):
        super(EEGSym_inception_block, self).__init__()
        self.inChannel = inChannel
        self.hiddenChannel = hiddenChannel
        self.outChannel = outChannel

        self.t_conv1 = nn.Sequential(
            nn.Conv3d(inChannel, hiddenChannel, (1, kernel_size[0], 1), stride=(1, 1, 1),
                      padding=(0, padding[0], 0)),
            nn.BatchNorm3d(hiddenChannel),
            nn.ELU(),
            nn.Dropout(p=dropoutRate)
        )


        self.t_conv2 = nn.Sequential(
            nn.Conv3d(inChannel, hiddenChannel, (1, kernel_size[1], 1), stride=(1, 1, 1),
                      padding=(0, padding[1], 0)),
            nn.BatchNorm3d(hiddenChannel),
            nn.ELU(),
            nn.Dropout(p=dropoutRate)
        )
        self.t_conv3 = nn.Sequential(
            nn.Conv3d(inChannel, hiddenChannel, (1, kernel_size[2], 1), stride=(1, 1, 1),
                      padding=(0, padding[2], 0)),
            nn.BatchNorm3d(hiddenChannel),
            nn.ELU(),
            nn.Dropout(p=dropoutRate)
        )
        self.pooling = nn.AvgPool3d(kernel_size=(1, 2, 1))

        self.group_conv = nn.Sequential(
            nn.Conv3d(hiddenChannel * 3, outChannel, (kernel_size[3], 1, 1), stride=(1, 1, 1),
                      padding=(padding[3], 0, 0), groups=hiddenChannel * 3),
            nn.BatchNorm3d(outChannel),
            nn.ELU(),
            nn.Dropout(p=dropoutRate)
        )
        self.conv_res1 = nn.Conv3d(inChannel, hiddenChannel * 3, (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        self.bn_res1 = nn.BatchNorm3d(hiddenChannel * 3)

        if hiddenChannel * 3 != outChannel:
            self.conv_res2 = nn.Conv3d(hiddenChannel * 3, outChannel, (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
            self.bn_res2 = nn.BatchNorm3d(outChannel)

    def forward(self, x):
        brand1 = self.t_conv1(x)
        brand2 = self.t_conv2(x)
        brand3 = self.t_conv3(x)
        res1 = self.bn_res1(self.conv_res1(x))
        out_1 = torch.add(torch.cat((brand1, brand2, brand3), dim=1), res1)  # torch.add会自动广播
        out_2 = self.pooling(out_1)
        out_3 = self.group_conv(out_2)
        if self.hiddenChannel * 3 != self.outChannel:
            res2 = self.bn_res2(self.conv_res2(out_2))
        else:
            res2 = out_2
        out = torch.add(out_3, res2)
        return out


class EEGSym_residual_block(nn.Module):
    def __init__(self, inChannel, hiddenChannel, outChannel, kernel_size, padding, dropoutRate=0.3):
        super(EEGSym_residual_block, self).__init__()
        self.hiddenChannel = hiddenChannel
        self.outChannel = outChannel

        self.t_conv1 = nn.Sequential(
            nn.Conv3d(inChannel, hiddenChannel, (1, kernel_size[0], 1), stride=(1, 1, 1),
                      padding=(0, padding[0], 0)),
            nn.BatchNorm3d(hiddenChannel),
            nn.ELU(),
            nn.Dropout(p=dropoutRate)
        )

        self.t_conv2 = nn.Sequential(
            nn.Conv3d(inChannel, hiddenChannel, (1, kernel_size[1], 1), stride=(1, 1, 1),
                      padding=(0, padding[1], 0)),
            nn.BatchNorm3d(hiddenChannel),
            nn.ELU(),
            nn.Dropout(p=dropoutRate)
        )
        self.t_conv3 = nn.Sequential(
            nn.Conv3d(hiddenChannel, outChannel, (kernel_size[2], 1, 1), stride=(1, 1, 1),
                      padding=(padding[2], 0, 0)),
            nn.BatchNorm3d(outChannel),
            nn.ELU(),
            nn.Dropout(p=dropoutRate)
        )



        self.pooling = nn.AvgPool3d(kernel_size=(1, 2, 1))
        if hiddenChannel != outChannel:
            self.conv_res = nn.Conv3d(hiddenChannel, outChannel, (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
            self.bn_res = nn.BatchNorm3d(outChannel)

    def forward(self, x):
        out_1 = self.t_conv1(x) + self.t_conv2(x)
        out_2 = self.pooling(out_1)
        out_3 = self.t_conv3(out_2)
        if self.hiddenChannel != self.outChannel:
            res1 = self.bn_res(self.conv_res(out_2))
        else:
            res1 = out_2
        return torch.add(out_3, res1)


class EEGSym_residual_mini_block(nn.Module):
    def __init__(self, inChannel, outChannel, kernel_size, padding, dropoutRate=0.3):
        super(EEGSym_residual_mini_block, self).__init__()
        self.inChannel = inChannel
        self.outChannel = outChannel
        self.t_conv1 = nn.Sequential(
            nn.Conv3d(inChannel, outChannel, (1, kernel_size[0], 1), stride=(1, 1, 1),
                      padding=(0, padding[0], 0)),
            nn.BatchNorm3d(outChannel),
            nn.ELU(),
            nn.Dropout(p=dropoutRate)
        )

        if inChannel != outChannel:
            self.conv_res = nn.Conv3d(inChannel, outChannel, (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
            self.bn_res = nn.BatchNorm3d(outChannel)

    def forward(self, x):
        if self.inChannel != self.outChannel:
            res = self.bn_res(self.conv_res(x))
        else:
            res = x
        out = self.t_conv1(x)
        return torch.add(out, res)


class EEGSym_Channel_Merging_block(nn.Module):
    def __init__(self, inChannel, hiddenChannel, outChannel, numChannel, dropoutRate=0.3):
        super(EEGSym_Channel_Merging_block, self).__init__()
        self.inChannel = inChannel
        self.hiddenChannel = hiddenChannel
        self.outChannel = outChannel
        self.num_channel = numChannel
        self.res_block1 = EEGSym_residual_mini_block(inChannel, hiddenChannel, [5], [2])
        self.res_block2 = EEGSym_residual_mini_block(inChannel, hiddenChannel, [5], [2])
        self.channel_merging_block = nn.Sequential(
            nn.Conv3d(hiddenChannel, outChannel, (numChannel, 1, 2), stride=(1, 1, 1),
                      padding=(0, 0, 0), groups=9),
            nn.BatchNorm3d(outChannel),
            nn.ELU(),
            nn.Dropout(p=dropoutRate)
        )



    def forward(self, x):
        output = self.res_block1(x)
        output = self.res_block2(output)
        output = self.channel_merging_block(output)
        return output


class EEGSym_Temporal_Merging_block(nn.Module):
    def __init__(self, inChannel, hiddenChannel, outChannel, NumTemperal, dropoutRate=0.3):
        super(EEGSym_Temporal_Merging_block, self).__init__()
        self.inChannel = inChannel
        self.hiddenChannel = hiddenChannel
        self.outChannel = outChannel
        self.NumTemperal = NumTemperal
        self.res_block1 = EEGSym_residual_mini_block(inChannel, hiddenChannel, [NumTemperal], [0])
        self.g_conv1 = nn.Sequential(
            nn.Conv3d(hiddenChannel, outChannel, (1, NumTemperal, 1), stride=(1, 1, 1),
                      padding=(0, 0, 0), groups=hiddenChannel),
            nn.BatchNorm3d(outChannel),
            nn.ELU(),
            nn.Dropout(p=dropoutRate)
        )


    def forward(self, x):
        output = self.res_block1(x)
        output = self.g_conv1(output)
        return output




class EEGSym(nn.Module):
    def __init__(self, right_idx=None, left_idx=None, feature_map_size=None, num_classes=None):
        '''
        :param right_idx: channel index for electrodes located in right hemisphere
        :param left_idx:  channel index for electrodes located in left hemisphere
        '''
        super(EEGSym, self).__init__()
        # Parameters----------------------------------------------
        if feature_map_size == None:
            self.feature_map_size = 36
        else:
            self.feature_map_size = feature_map_size
        if num_classes == None:
            self.num_classes = 4
        else:
            self.num_classes = num_classes
        if right_idx is None or left_idx is None:
            self.sym_layer = Symmetric_layer([1, 2, 3, 4, 5], [4, 5, 6, 7, 8])
            self.num_channels = 5
        else:
            self.sym_layer = Symmetric_layer(right_idx, left_idx)
            self.num_channels=len(right_idx)
        # Convolution Block----------------------------------------------
        self.Block1 = nn.Sequential(
            EEGSym_inception_block(1, 24, 72, [81, 41, 21, self.num_channels], [40, 20, 10, 0]),
            EEGSym_inception_block(72, 24, 72, [21, 11, 5, self.num_channels], [10, 5, 2, 0])
        )
        self.Block2 = nn.Sequential(
            EEGSym_residual_block(72, 36, 36, [1, 21, self.num_channels], [0, 10, 0]),
            EEGSym_residual_block(36, 36, 36, [1, 11, self.num_channels], [0, 5, 0]),
            EEGSym_residual_block(36, 18, 18, [1, 5, self.num_channels], [0, 2, 0])
        )
        self.Block3 = nn.Sequential(
            EEGSym_residual_mini_block(18, 18, [5], [2]),
            nn.AvgPool3d(kernel_size=(1, 2, 1))
        )
        self.Block4 = EEGSym_Channel_Merging_block(18, 18, 18, self.num_channels).cuda()
        self.Block5 = EEGSym_Temporal_Merging_block(18, 18, 36, 7).cuda()
        self.Block6 = nn.Sequential(
            EEGSym_residual_mini_block(36, 36, [1], [0]),
            EEGSym_residual_mini_block(36, 36, [1], [0]),
            EEGSym_residual_mini_block(36, 36, [1], [0]),
            EEGSym_residual_mini_block(36, 36, [1], [0]),
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(self.feature_map_size, 32),
            nn.BatchNorm1d(32),
            nn.Dropout(p=0.25),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.Dropout(p=0.25),
            nn.Linear(16, self.num_classes)
        )

    def forward(self, input_data):
        input_data = input_data.type(torch.cuda.FloatTensor)
        output = self.sym_layer(input_data)
        output = self.Block1(output)
        # print(output.size())
        output = self.Block2(output)
        # print(output.size())
        output = self.Block3(output)
        # print(output.size())
        output = self.Block4(output)
        # print(output.size())
        output = self.Block5(output)
        # print(output.size())
        output = self.Block6(output)
        # print(output.size())

        output = self.fc_layer(output.view(-1, self.feature_map_size))
        return output


if __name__ == "__main__":
    import numpy as np
    import os
    from thop import profile

    PhysioNet_right_idx=np.load(os.path.join("..","config","kaggleMI_right_idx.npy"))-1
    PhysioNet_left_idx = np.load(os.path.join("..","config", "kaggleMI_left_idx.npy"))-1

    net = EEGSym(right_idx=PhysioNet_right_idx,left_idx=PhysioNet_left_idx).cuda()

    test = torch.rand(32, 1, 32, 480).cuda()
    flops, params = profile(net, inputs=(test,), )
    print("Parameters:{0}, Flops:{1}".format(params,flops))
    out = net(test)
    print(out.size())
