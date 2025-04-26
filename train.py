import os
from glob import glob
import torch
from torch.utils.data import random_split
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from config import *
from dataset import DashCamDataLoader, DashCamDataset
from model import Speed3DCNN

loss_function = torch.nn.SmoothL1Loss()


def calc_test_loss(model, loader, N):
    """
    Calculate the test loss for the model on the given data loader.
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        progress_bar = tqdm(loader, desc="Testing")
        for x, y in progress_bar:
            y_pred = model(x)
            loss = loss_function(y_pred, y)
            total_loss += loss.item()
    return total_loss / N


def load_checkpoint(model, optimizer):
    """
    Load the latest checkpoint for the model and optimizer.
    """
    epoch_files = glob(model.checkpoint_path("*"))

    if epoch_files:
        latest_epoch_file = max(epoch_files, key=lambda x: int(os.path.basename(x).split(".")[0]))

        print(f"Loading model from checkpoint {latest_epoch_file} ...")

        checkpoint = torch.load(latest_epoch_file, weights_only=False)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        print(f"Epoch {checkpoint['epoch']}: Train Loss = {checkpoint['train_loss']:.4f}, Test Loss = {checkpoint['test_loss']:.4f}")
        return checkpoint['epoch'] + 1

    print(f"No checkpoint found.")
    os.makedirs(model.checkpoint_dir, exist_ok=True)
    return 1


def train():
    dataset = DashCamDataset()

    train_dataset, test_dataset = dataset.get_test_train_split()

    train_loader = DashCamDataLoader(train_dataset, shuffle=True)
    test_loader = DashCamDataLoader(test_dataset, shuffle=False)

    model = Speed3DCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    epoch_start = load_checkpoint(model, optimizer)

    for epoch in range(epoch_start, epoch_start + TOTAL_EPOCHS):
        model.train()

        train_loss = 0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{epoch_start + TOTAL_EPOCHS - 1}"):
            optimizer.zero_grad()
            y_pred = model(x)
            loss = loss_function(y_pred, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_dataset)
        test_loss = calc_test_loss(model, test_loader, len(test_dataset))

        print(f"Train Loss = {train_loss:.4f}, Test Loss = {test_loss:.4f}")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'test_loss': test_loss,
        }, model.checkpoint_path(epoch))

        print(f"Checkpoint saved at {model.checkpoint_path(epoch)}.")


if __name__ == "__main__":
    print(SUMMARY)
    train()
