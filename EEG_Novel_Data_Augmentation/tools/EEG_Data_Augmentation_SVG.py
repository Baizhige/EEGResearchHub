import torch
import numpy as np
import os

class SVG(object):
    """
SVG class for EEG data augmentation.
Parameters:
    cuda (bool): Whether to use CUDA for acceleration. Default is False.
    right_idx (list): Indexes for right-side electrodes. Default is None.
    left_idx (list): Indexes for left-side electrodes. Default is None.
    sigma_rotate_theta (float): Standard deviation for theta rotation. Default is 0.314.
    sigma_rotate_phi (float): Standard deviation for phi rotation. Default is 0.1.
    sigma_stretch_theta (float): Standard deviation for theta stretching. Default is 0.1.
    sigma_stretch_phi (float): Standard deviation for phi stretching. Default is 0.05.
    sigma_pos (float): Standard deviation for position distortion. Default is 0.05.
    probability_flip (float): Probability of flipping. Default is 0.5.
    cap (str): Type of EEG cap (e.g., 'PhysioNetMI'). Default is 'physionet'.

Examples:
    # Create SVG object
    svg = SVG(cuda=True, sigma_rotate_theta=0.3, sigma_rotate_phi=0.1, cap='PhysioNetMI')

    # Input data
    input_data = torch.rand([64,480]).cuda()  # Assume this is EEG data with 64 channels and 480 time points.

    # Perform transformation in automatic mode
    output_data, is_flip = svg.transform(input_data, mode='Auto')

    # Perform rotation
    output_data, is_flip = svg.transform(input_data, mode='rotation')

    # Perform scaling
    output_data, is_flip = svg.transform(input_data, mode='scaling')

    # Perform distortion
    output_data, is_flip = svg.transform(input_data, mode='distortion')

    # Perform left-right flipping
    output_data, is_flip = svg.transform(input_data, mode='flipping')

    """
    def __init__(self,
                 cuda=False,  # Boolean type, whether to use GPU acceleration
                 right_idx=None,  # Index of the electrode channel on the right side
                 left_idx=None,  # Index of the electrode channel on the left side
                 sigma_rotate_theta=0.314,  # Standard deviation of the theta rotation angle
                 sigma_rotate_phi=0.1,  # Standard deviation of the phi rotation angle
                 sigma_stretch_theta=0.1,  # Standard deviation of the theta stretching angle
                 sigma_stretch_phi=0.05,  # Standard deviation of the phi stretching angle
                 sigma_pos=0.05,  # Standard deviation for random electrode placement
                 probability_flip=0.5,  # Probability of left-right flipping
                 cap='PhysioNetMI' # EEG cap model (PhysioNet means the EEG cap used in the PhysioNet dataset, see parameter config for details)
                 ):
        # Initialize the device; use GPU if cuda is True
        self.device = torch.device("cuda:0" if cuda else "cpu")
        # Set the exponent for the RBF Gaussian kernel, the value is negative
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


    # Compute Weight Matrix
    def get_weight_matrix(self, electrode_pos, source_signal_pos):
        theta_e = electrode_pos[0, :].view(self.num_channels, 1)
        theta_s = source_signal_pos[0, :].view(self.num_channels, 1)
        phi_e = electrode_pos[1, :].view(self.num_channels, 1)
        phi_s = source_signal_pos[1, :].view(self.num_channels, 1)
        geo_adj = torch.arccos((torch.sin(phi_e) @ torch.sin(phi_s).view(1, self.num_channels)) + (torch.cos(phi_e) @ torch.cos(phi_s).view(1, self.num_channels)) * torch.cos(theta_e - theta_s.view(1, self.num_channels)))
        geo_adj = torch.where(torch.isnan(geo_adj), torch.full_like(geo_adj, 0), geo_adj)  # 这是pytorch的bug, arcos(1)有概率变成nan
        weight_matrix = torch.exp(self.att_k * geo_adj)
        return weight_matrix

    # Compute Transform Matrix
    def get_transform_matrix(self, electrode_pos, source_signal_pos):
        transform_matrix_aug = torch.mm(self.get_weight_matrix(electrode_pos, source_signal_pos), self.weight_matrix_raw_inv)
        transform_matrix_aug = torch.div(transform_matrix_aug,torch.transpose(torch.sum(transform_matrix_aug,dim=1).expand(self.num_channels, self.num_channels),0,1))
        return transform_matrix_aug

    # Implement Rotation
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

    # Implement Scaling
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

    # Implement Flipping
    def flipping(self):
        electrode_pos_aug = self.electrode_pos_raw.clone().detach()
        source_signal_pos_aug = self.source_signal_pos_raw.clone().detach()
        source_signal_pos_aug[0, :] = -1*source_signal_pos_aug[0, :]
        return electrode_pos_aug, source_signal_pos_aug

    # Implement Distortion
    def distortion(self):
        electrode_pos_aug = self.electrode_pos_raw.clone().detach()
        source_signal_pos_aug = self.source_signal_pos_raw.clone().detach()
        rand_pos = torch.normal(0, self.sigma_pos, (2, self.num_channels)).to(self.device)
        rand_pos = torch.clamp(rand_pos, min=-0.08, max=+0.08)  # 限制在5 deg以内
        electrode_pos_aug = electrode_pos_aug + rand_pos
        return electrode_pos_aug, source_signal_pos_aug

    # Implement Data Augmentation Transform
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
    # Create SVG object
    svg = SVG(cuda=True, sigma_rotate_theta=0.3, sigma_rotate_phi=0.1, cap='PhysioNetMI')
    # Input data
    input_data = torch.rand([64, 480]).cuda()  # Assuming this is EEG data
    # Perform transformation in automatic mode
    output_data, is_flip = svg.transform(input_data, mode='Auto')
    # Perform rotation operation
    output_data, is_flip = svg.transform(input_data, mode='rotation')
    # Perform scaling operation
    output_data, is_flip = svg.transform(input_data, mode='scaling')
    # Perform distortion operation
    output_data, is_flip = svg.transform(input_data, mode='distortion')
    # Perform left-right flipping
    output_data, is_flip = svg.transform(input_data, mode='flipping')

