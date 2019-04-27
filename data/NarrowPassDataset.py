from torch.utils.data import Dataset
import numpy as np
import pandas as pd


class NarrowPassDataset(Dataset):
    def __init__(self, csv_file, sample_dim, condition_dim, occ_dim, train=True, ratio_test=0.8):
        self.sample_dim = sample_dim
        self.con_dim = condition_dim
        self.occ_dim = occ_dim
        self.dataset = np.array(pd.read_csv(csv_file, header=None), dtype='float32')
        numTrain = int(ratio_test * len(self.dataset))
        if (train):
            self.dataset = self.dataset[:numTrain, :]
        else:
            self.dataset = self.dataset[numTrain:, :]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx, :self.sample_dim], \
                self.dataset[idx, self.sample_dim:(self.sample_dim + self.con_dim)], \
                self.dataset[idx, -self.occ_dim:]


class NarrowPassCnnDataset(Dataset):
    def __init__(self, csv_file, sample_dim, condition_dim, gridSize, train=True, ratio_test=0.8):
        self.sample_dim = sample_dim
        self.con_dim = condition_dim
        self.gridSize = gridSize
        self.occ_dim = gridSize**2
        self.dataset = np.array(pd.read_csv(csv_file, header=None), dtype='float32')
        numTrain = int(ratio_test * len(self.dataset))
        if (train):
            self.dataset = self.dataset[:numTrain, :]
        else:
            self.dataset = self.dataset[numTrain:, :]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx, :self.sample_dim], \
               self.dataset[idx, self.sample_dim:(self.sample_dim + self.con_dim)], \
               self.dataset[idx, (self.con_dim - self.sample_dim):(self.sample_dim + self.con_dim)], \
               self.dataset[idx, -self.occ_dim:].reshape((1, self.gridSize, self.gridSize))
