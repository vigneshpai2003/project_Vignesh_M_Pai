import os
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split

from config import *
from preprocess import preprocess_all


class DashCamDataset(TensorDataset):
    data_dir = f'processed/{DATA_CONFIG}/'
    samples_path = data_dir + 'samples.npy'
    labels_path = data_dir + 'labels.npy'
    split_path = data_dir + SPLIT_CONFIG + '.pt'

    def __init__(self):
        X, y = self.load_from_file()

        X = (X - X.mean(dim=0)) / X.std(dim=0)

        self.speed_mean = y.mean().item()
        self.speed_std = y.std().item()

        y = (y - self.speed_mean) / self.speed_std

        super().__init__(X, y)

    @classmethod
    def cutoff_speed(cls, X, y):
        """
        Cut off the speed values below SPEED_CUT_OFF.
        """
        new_X = []
        new_y = []

        for x, speed in zip(X, y):
            if speed >= SPEED_CUT_OFF:
                new_X.append(x)
                new_y.append(speed)

        new_X = np.expand_dims(np.concatenate(new_X, axis=0), axis=1)
        new_y = np.array(new_y)

        return new_X, new_y

    @classmethod
    def load_from_file(cls):
        """
        Load the data from the processed files.
        If the files do not exist, preprocess the data.
        """
        if not os.path.exists(cls.data_dir):
            print("Processed data not found. Preprocessing data...")
            print("This may take a while...")
            preprocess_all()
        else:
            print(f"Loading processed data from {cls.data_dir}...")

        X, y = cls.cutoff_speed(np.load(cls.samples_path), np.load(cls.labels_path))
        X = torch.from_numpy(X)
        y = torch.from_numpy(y)

        return X, y

    def get_speed(self, y):
        return self.speed_mean + self.speed_std * y

    def get_speed_kmph(self, y):
        return self.get_speed(y) * 3.6

    def get_test_train_split(self):
        """
        Load the train/test split from the split file.
        If the split file does not exist, split the data and save it to the split file.
        """
        if not os.path.exists(self.split_path):
            print("Train/test split not found. Splitting data...")
            train_dataset, test_dataset = random_split(self, [TRAIN_FACTOR, TEST_FACTOR])
            torch.save({
                'train': train_dataset,
                'test': test_dataset
            }, self.split_path)
            return train_dataset, test_dataset

        print(f"Loading train/test split from {self.split_path}...")
        split_data = torch.load(self.split_path, weights_only=False)
        return split_data['train'], split_data['test']


class DashCamDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        kwargs['batch_size'] = kwargs.get('batch_size', BATCH_SIZE)
        super().__init__(*args, **kwargs)
