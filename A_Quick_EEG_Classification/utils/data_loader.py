import torch.utils.data as data
import numpy as np
import os
import torch


class EEGDataSet(data.Dataset):
    def __init__(self, data_root, data_list, transform=None, start=0, len_data=None,
                 num_channel=None, max_num_samples=None, cuda=False):
        """
        A custom PyTorch-based dataset for loading EEG data.
        Author: Chengxuan.Qin@outlook.com
        Args:
        - data_root (str): Root directory of the EEG data.
        - data_list (str): File containing a list of EEG data file path with labels and subjects.
        - transform (callable, optional): A function/transform to apply to the data.
        - start (int, optional): The starting index of the EEG data in time dimension.
        - len_data (int, optional): Length of the EEG data to be used. If None, it will
                                       be inferred from the first EEG data sample.
        - num_channel (int, optional): Number of channels in the EEG data. If None, it will
                                       be inferred from the first EEG data sample.
        - max_num_samples (int, optional): Maximum number of samples to load from the data_list.
        - cuda (bool, optional): Whether to use CUDA (GPU) for tensor operations.
        """
        self.device = torch.device("cuda:0" if cuda else "cpu")
        self.root = data_root
        self.start = start
        self.transform = transform

        # Read the data_list file and extract metadata from the first EEG data sample
        with open(os.path.join(data_root, data_list), 'r') as f:
            data_list = f.readlines()
        __test_data__ = np.load(os.path.join(self.root, data_list[0].split()[0]), allow_pickle=True)

        # Set len_data and num_channel if not provided
        self.len_data = len_data if len_data is not None else __test_data__.shape[0]
        self.num_channel = num_channel if num_channel is not None else __test_data__.shape[1]

        # Initialize lists to store EEG data paths, labels, and subjects
        self.eeg_paths = []
        self.eeg_labels = []
        self.eeg_subjects = []

        # Determine the number of data samples to load
        self.n_data = min(len(data_list), max_num_samples) if max_num_samples is not None else len(data_list)

        # Parse data_list and populate the lists
        for data in data_list[:self.n_data]:
            t_list = data.split()
            self.eeg_paths.append(t_list[0])
            self.eeg_labels.append(int(t_list[2]))
            self.eeg_subjects.append(int(t_list[1]))

    def __getitem__(self, item):
        # Get the paths, labels, and subjects for the given item
        eeg_paths, labels, subjects = self.eeg_paths[item], self.eeg_labels[item], self.eeg_subjects[item]
        labels = int(labels)

        # Load EEG data from the file
        eegs = np.load(os.path.join(self.root, eeg_paths), allow_pickle=True)
        eegs = np.float64(eegs)

        # Extract the specified portion of the EEG data and format it for PyTorch
        start_p = self.start
        eegs = eegs[start_p:start_p + self.len_data, :self.num_channel]
        eegs = np.expand_dims(eegs.T, axis=0)
        eegs = torch.tensor(eegs).type(torch.FloatTensor).to(self.device)

        if self.transform:
            eegs = self.transform(eegs)

        return eegs, subjects, labels

    def __len__(self):
        return self.n_data
