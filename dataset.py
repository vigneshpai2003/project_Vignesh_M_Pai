import os
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split

from config import *
from preprocess import preprocess_all


class DashCamDataset(TensorDataset):
    def __init__(self):
        if not os.path.exists(f'processed/{DATA_CONFIG}'):
            print("Processed data not found. Preprocessing data...")
            preprocess_all()

        X = torch.from_numpy(np.load(f'processed/{DATA_CONFIG}/samples.npy'))
        y = torch.from_numpy(np.load(f'processed/{DATA_CONFIG}/labels.npy'))

        # Normalize X and y
        X = (X - X.mean(dim=0)) / X.std(dim=0)

        self.speed_mean = y.mean()
        self.speed_std = y.std()

        y = (y - self.speed_mean) / self.speed_std

        super().__init__(X, y)

    def get_speed(self, y):
        return self.speed_mean + self.speed_std * y

    def get_test_train_split(self):
        split_path = f'processed/{DATA_CONFIG}/train_test_split.pt'

        if not os.path.exists(split_path):
            print("Train/test split not found. Splitting data...")
            train_dataset, test_dataset = random_split(self, [TRAIN_FACTOR, TEST_FACTOR])
            torch.save({
                'train': train_dataset,
                'test': test_dataset
            }, split_path)
            return train_dataset, test_dataset

        split_data = torch.load(split_path, weights_only=False)
        return split_data['train'], split_data['test']


class DashCamDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
