from torch.utils.data import random_split

from config import *
from dataset import DashCamDataLoader, DashCamDataset


dataset = DashCamDataset()

train_dataset, test_dataset = random_split(dataset, [TRAIN_FACTOR, TEST_FACTOR])

train_loader = DashCamDataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
