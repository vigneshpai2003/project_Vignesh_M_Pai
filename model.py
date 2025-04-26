import hashlib
import torch
import torch.nn as nn

from config import *


class Speed3DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),

            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),

            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 8, 8)),

            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(64 * 1 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.network(x).squeeze(1)

    @property
    def hash(self):
        """
        Generate a hash for the model architecture.
        """
        arch_string = str(self)
        arch_bytes = arch_string.encode('utf-8')
        arch_hash = hashlib.md5(arch_bytes).hexdigest()
        return arch_hash

    @property
    def checkpoint_dir(self):
        return f"checkpoints/{self.hash}/{DATA_CONFIG}_{SPLIT_CONFIG}/"

    def checkpoint_path(self, epoch):
        return self.checkpoint_dir + f"{epoch}.pt"
