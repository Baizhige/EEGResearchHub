import torch
import numpy as np
import os

class SVG(object):
    """
    SVG类 用于EEG数据的增强。
    参数:
    cuda (bool): 是否使用CUDA加速。默认为False。
    right_idx (list): 右侧电极索引。默认为None。
    left_idx (list): 左侧电极索引。默认为None。
    sigma_rotate_theta (float): Theta方向旋转的标准差。默认为0.314。
    sigma_rotate_phi (float): Phi方向旋转的标准差。默认为0.1。
    sigma_stretch_theta (float): Theta方向拉伸的标准差。默认为0.1。
    sigma_stretch_phi (float): Phi方向拉伸的标准差。默认为0.05。
    sigma_pos (float): 位置扭曲的标准差。默认为0.05。
    probability_flip (float): 翻转概率。默认为0.5。
    cap (str): EEG帽的类型（例如'PhysioNetMI'）。默认为'physionet'。

    示例:
        # 创建SVG对象
    svg = SVG(cuda=True, sigma_rotate_theta=0.3, sigma_rotate_phi=0.1, cap='PhysioNetMI')

    # 输入数据
    input_data = torch.rand([64,480]).cuda()  # 假设这是EEG数据，64为通道个数，480为序列长度。

    # 进行自动模式下的变换
    output_data, is_flip = svg.transform(input_data, mode='Auto')

    # 进行旋转操作
    output_data, is_flip = svg.transform(input_data, mode='rotation')

    # 进行缩放操作
    output_data, is_flip = svg.transform(input_data, mode='scaling')

    # 进行扭曲操作
    output_data, is_flip = svg.transform(input_data, mode='distortion')

    # 进行左右翻转
    output_data, is_flip = svg.transform(input_data, mode='flipping')

    """
    def __init__(self,
                 cuda=False, # bool类型，是否使用GPU加速
                 right_idx=None, # 右侧电极通道的索引
                 left_idx=None, # 左侧电极通道的索引
                 sigma_rotate_theta=0.314, # theta旋转角度的标准差
                 sigma_rotate_phi=0.1,  # phi旋转角度的标准差
                 sigma_stretch_theta=0.1, # theta拉伸角度的标准差
                 sigma_stretch_phi=0.05, # phi拉伸角度的标准差
                 sigma_pos=0.05, # 随机电极标准差
                 probability_flip=0.5, # 左右翻转的概率
                 cap='PhysioNetMI'): # 脑电帽型号（physionet意味着数据集physionet所使用的脑电帽，详见参数config）
        # 初始化设备，如果cuda为True，使用GPU
        self.device = torch.device("cuda:0" if cuda else "cpu")
        # 设置RBF 高斯核的指数， 取值为负数
        self.att_k = -3
        self.right_idx = right_idx
        self.left_idx = left_idx
        self.sigma_rotate_theta = sigma_rotate_theta
        self.sigma_rotate_phi = sigma_rotate_phi
        self.sigma_stretch_theta = sigma_stretch_theta
        self.sigma_stretch_phi = sigma_stretch_phi
        self.sigma_pos = sigma_pos
        self.probability_flip = probability_flip
        self.pi2 = torch.tensor(np.pi / 2).type(torch.FloatTensor).to(self.device)
        self.parent_dir = '..'

        self.channel_name = np.load(os.path.join(self.parent_dir,"config", f"{cap}_channel_name.npy"))
        self.num_channels = self.channel_name.shape[0]
        # electrode_pos_raw或者source_signal_pos_raw代表电极的经纬度坐标，第一行theta 第二行phi。
        self.electrode_pos_raw = torch.tensor(np.load(os.path.join(self.parent_dir,"config", f"{cap}_channel_pos.npy"))).type(
            torch.FloatTensor).to(self.device)
        self.source_signal_pos_raw = torch.tensor(np.load(os.path.join(self.parent_dir,"config", f"{cap}_channel_pos.npy"))).type(
            torch.FloatTensor).to(self.device)
        self.weight_matrix_raw = self.get_weight_matrix(self.electrode_pos_raw, self.source_signal_pos_raw)
        self.weight_matrix_raw_inv = torch.inverse(self.weight_matrix_raw)
        self.electrode_pos_flipping, self.source_signal_pos_flipping = self.flipping()
        self.weight_matrix_flipping = self.get_weight_matrix(self.electrode_pos_flipping, self.source_signal_pos_flipping)
        self.transform_matrix_flipping = torch.mm(self.weight_matrix_flipping, self.weight_matrix_raw_inv)


    # 计算权重矩阵
    def get_weight_matrix(self, electrode_pos, source_signal_pos):
        theta_e = electrode_pos[0, :].view(self.num_channels, 1)
        theta_s = source_signal_pos[0, :].view(self.num_channels, 1)
        phi_e = electrode_pos[1, :].view(self.num_channels, 1)
        phi_s = source_signal_pos[1, :].view(self.num_channels, 1)
        geo_adj = torch.arccos((torch.sin(phi_e) @ torch.sin(phi_s).view(1, self.num_channels)) + (torch.cos(phi_e) @ torch.cos(phi_s).view(1, self.num_channels)) * torch.cos(theta_e - theta_s.view(1, self.num_channels)))
        geo_adj = torch.where(torch.isnan(geo_adj), torch.full_like(geo_adj, 0), geo_adj)  # 这是pytorch的bug, arcos(1)有概率变成nan
        weight_matrix = torch.exp(self.att_k * geo_adj)
        return weight_matrix

    # 获取变换矩阵
    def get_transform_matrix(self, electrode_pos, source_signal_pos):
        transform_matrix_aug = torch.mm(self.get_weight_matrix(electrode_pos, source_signal_pos), self.weight_matrix_raw_inv)
        transform_matrix_aug = torch.div(transform_matrix_aug,torch.transpose(torch.sum(transform_matrix_aug,dim=1).expand(self.num_channels, self.num_channels),0,1))
        return transform_matrix_aug

    # 进行旋转操作
    def rotation(self, angle="both"):
        electrode_pos_aug = self.electrode_pos_raw.clone().detach()
        source_signal_pos_aug = self.source_signal_pos_raw.clone().detach()
        if angle == "both":
            rand_theta = torch.normal(0, self.sigma_rotate_theta, (1, 1)).to(self.device)
            electrode_pos_aug[0, :] = torch.add(electrode_pos_aug[0, :], rand_theta)
            rand_phi = torch.normal(0, self.sigma_rotate_phi, (1, 1)).to(self.device)
            electrode_pos_aug[1, :] = torch.add(electrode_pos_aug[1, :], rand_phi)
        elif angle == "theta":
            rand_theta = torch.normal(0, self.sigma_rotate_theta, (1, 1)).to(self.device)
            electrode_pos_aug[0, :] = torch.add(electrode_pos_aug[0, :], rand_theta)
        elif angle == "phi":
            rand_phi = torch.normal(0, self.sigma_rotate_phi, (1, 1)).to(self.device)
            electrode_pos_aug[1, :] = torch.add(electrode_pos_aug[1, :], rand_phi)
        else:
            raise "Warning!"
        return electrode_pos_aug, source_signal_pos_aug

    # 进行缩放操作
    def scaling(self, angle="both"):
        electrode_pos_aug = self.electrode_pos_raw.clone().detach()
        source_signal_pos_aug = self.source_signal_pos_raw.clone().detach()
        if angle == "both":
            rand_theta = torch.normal(1, self.sigma_stretch_theta, (1, 1)).to(self.device)
            electrode_pos_aug[0, :] = torch.mul(electrode_pos_aug[0, :], rand_theta)
            rand_phi = torch.normal(1, self.sigma_stretch_phi, (1, 1)).to(self.device)
            electrode_pos_aug[1, :] = torch.mul(electrode_pos_aug[1, :], rand_phi)
        elif angle == "theta":
            rand_theta = torch.normal(1, self.sigma_stretch_theta, (1, 1)).to(self.device)
            electrode_pos_aug[0, :] = torch.mul(electrode_pos_aug[0, :], rand_theta)
        elif angle == "phi":
            rand_phi = torch.normal(1, self.sigma_stretch_phi, (1, 1)).to(self.device)
            rand_phi = torch.clamp(rand_phi, min=-0.8, max=+1.2)  # 限制在0.2倍以内
            electrode_pos_aug[1, :] = torch.mul(electrode_pos_aug[1, :] - self.pi2, rand_phi) + self.pi2
        else:
            raise "Warning!"
        return electrode_pos_aug, source_signal_pos_aug

    # 执行左右翻转
    def flipping(self):
        electrode_pos_aug = self.electrode_pos_raw.clone().detach()
        source_signal_pos_aug = self.source_signal_pos_raw.clone().detach()
        source_signal_pos_aug[0, :] = -1*source_signal_pos_aug[0, :]
        return electrode_pos_aug, source_signal_pos_aug

    # 执行扭曲操作
    def distortion(self):
        electrode_pos_aug = self.electrode_pos_raw.clone().detach()
        source_signal_pos_aug = self.source_signal_pos_raw.clone().detach()
        rand_pos = torch.normal(0, self.sigma_pos, (2, self.num_channels)).to(self.device)
        rand_pos = torch.clamp(rand_pos, min=-0.08, max=+0.08)  # 限制在5 deg以内
        electrode_pos_aug = electrode_pos_aug + rand_pos
        return electrode_pos_aug, source_signal_pos_aug

    def transform(self, input_data, mode='Auto'):
        if mode=='Auto':
            if np.random.rand() < self.probability_flip:
                transform_matrix_auto = self.transform_matrix_flipping
                is_flip=True
            else:
                transform_matrix_auto = torch.eye(self.num_channels).to(self.device)
                is_flip=False
            choose_mode = np.random.randint(0,10)
            if choose_mode<3:
                electrode_pos_aug, source_signal_pos_aug = self.rotation(angle="theta")
                transform_matrix_auto = torch.mm(transform_matrix_auto, self.get_transform_matrix(electrode_pos_aug, source_signal_pos_aug))
            elif choose_mode < 6:
                electrode_pos_aug, source_signal_pos_aug = self.distortion()
                transform_matrix_auto = torch.mm(transform_matrix_auto, self.get_transform_matrix(electrode_pos_aug, source_signal_pos_aug))
            elif choose_mode < 9:
                electrode_pos_aug, source_signal_pos_aug = self.scaling()
                transform_matrix_auto = torch.mm(transform_matrix_auto, self.get_transform_matrix(electrode_pos_aug, source_signal_pos_aug))
            else:
                pass
            out = torch.matmul(transform_matrix_auto, input_data)
            return out, is_flip
        elif mode=='rotation':
            electrode_pos_aug, source_signal_pos_aug = self.rotation(angle="theta")
            out = torch.mm(self.get_transform_matrix(electrode_pos_aug, source_signal_pos_aug), input_data)
            return out, False
        elif mode == 'distortion':
            electrode_pos_aug, source_signal_pos_aug = self.distortion()
            out = torch.mm(self.get_transform_matrix(electrode_pos_aug, source_signal_pos_aug), input_data)
            return out, False
        elif mode == 'scaling':
            electrode_pos_aug, source_signal_pos_aug = self.scaling(angle="phi")
            out = torch.mm(self.get_transform_matrix(electrode_pos_aug, source_signal_pos_aug), input_data)
            return out, False
        elif mode == 'flipping':
            out = torch.mm(self.transform_matrix_flipping, input_data)
            return out, True
        else:
            raise 'Warning! No specified mode for transform'

if __name__ == "__main__":

    # 创建SVG对象
    svg = SVG(cuda=True, sigma_rotate_theta=0.3, sigma_rotate_phi=0.1, cap='PhysioNetMI')

    # 输入数据
    input_data = torch.rand([64,480]).cuda()  # 假设这是EEG数据

    # 进行自动模式下的变换
    output_data, is_flip = svg.transform(input_data, mode='Auto')
    print(output_data.size())
    print(is_flip)
    # 进行旋转操作
    output_data, is_flip = svg.transform(input_data, mode='rotation')
    print(output_data.size())
    print(is_flip)
    # 进行缩放操作
    output_data, is_flip = svg.transform(input_data, mode='scaling')
    print(output_data.size())
    print(is_flip)
    # 进行扭曲操作
    output_data, is_flip = svg.transform(input_data, mode='distortion')
    print(output_data.size())
    print(is_flip)
    # 进行左右翻转
    output_data, is_flip = svg.transform(input_data, mode='flipping')

    print(output_data.size())
    print(is_flip)
