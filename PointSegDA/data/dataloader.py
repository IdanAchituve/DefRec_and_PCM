from torch.utils.data import Dataset
import glob
import numpy as np
from utils.pc_utils import (jitter_pointcloud, random_rotate_one_axis)
import os

class datareader(Dataset):
    def __init__(self, dataroot, dataset, partition='train', domain='target'):
        self.partition = partition
        self.domain = domain
        self.dataset = dataset

        self.data = []
        folders = os.path.join(dataroot, dataset, partition, '*.npy')
        data_files = glob.glob(folders)
        for file in data_files:
            self.data.append(np.load(file))

    def __getitem__(self, item):
        pointcloud = self.data[item].astype(np.float32)[:, :3]
        label = self.data[item].astype(np.long)[:, 3] - 1  # labels 1-8

        # apply data rotation and augmentation on train samples
        if self.partition == 'train':
            pointcloud = jitter_pointcloud(random_rotate_one_axis(pointcloud, "z"))

        return (pointcloud, label)

    def __len__(self):
        return len(self.data)