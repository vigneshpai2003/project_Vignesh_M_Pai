import os
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from config import DATA_CONFIG
from preprocess import preprocess_all


class DashCamDataset(TensorDataset):
    def __init__(self):
        if not os.path.exists(f'processed/{DATA_CONFIG}.npz'):
            print("Processed data not found. Preprocessing data...")
            preprocess_all()

        data = np.load(f'processed/{DATA_CONFIG}.npz')

        super().__init__(torch.from_numpy(data['X']), torch.from_numpy(data['y']))


class DashCamDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
