import torch
import torch.nn as nn


class Speed3DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),     # (B, 16, 2, 64, 64)
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),            # (B, 16, 2, 32, 32)

            nn.Conv3d(16, 32, kernel_size=3, padding=1),    # (B, 32, 2, 32, 32)
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),            # (B, 32, 1, 16, 16)

            nn.Flatten(),                                   # (B, 32 * 1 * 16 * 16)
            nn.Dropout(0.3),
            nn.Linear(32 * 1 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.network(x).squeeze(1)
