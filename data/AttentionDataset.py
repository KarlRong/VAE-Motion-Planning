from torch.utils.data import Dataset
import numpy as np
import pandas as pd


class AttentionDataset(Dataset):
    def __init__(self, np_file_data, sample_dim, condition_dim, gridSize, train=True, ratio_test=0.8):
        self.sample_dim = sample_dim
        self.con_dim = condition_dim
        self.gridSize = gridSize
        self.occ_dim = gridSize**2
        data = np.load(np_file_data)
        self.sampleData = data['sampleData']
        self.gapData = data['gapData']
        self.occData = data['occ'].astype('float32')
        numTrain = int(ratio_test * len(self.sampleData))
        if (train):
            self.sampleData = self.sampleData[:numTrain, :]
            self.gapData = self.gapData[:numTrain, :]
            self.occData = self.occData[:numTrain, :]
        else:
            self.sampleData = self.sampleData[numTrain:, :]
            self.gapData = self.gapData[numTrain:, :]
            self.occData = self.occData[numTrain:, :]

    def __len__(self):
        return len(self.sampleData)

    def __getitem__(self, idx):
        return self.sampleData[idx, :], \
                self.gapData[idx, :], \
                self.gapData[idx, -self.sample_dim*2:], \
                self.occData[idx].reshape((1, self.gridSize, self.gridSize))

