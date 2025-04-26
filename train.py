import os
import torch
from torch.utils.data import random_split
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from config import *
from dataset import DashCamDataLoader, DashCamDataset
from model import Speed3DCNN

loss_function = torch.nn.MSELoss()


def calc_test_loss(model, loader, N):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        progress_bar = tqdm(loader, desc="Testing")
        for x, y in progress_bar:
            y_pred = model(x)
            loss = loss_function(y_pred, y)
            total_loss += loss.item()
    return total_loss / N


def train():
    dataset = DashCamDataset()

    train_dataset, test_dataset = dataset.get_test_train_split()

    train_loader = DashCamDataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DashCamDataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = Speed3DCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    epoch_start = 1
    if os.path.exists("checkpoints/main.pt"):
        print("Loading model from checkpoint...")
        input("Press Enter to continue...")
        checkpoint = torch.load("checkpoints/main.pt", weights_only=False)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_start = checkpoint['epoch'] + 1

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

    os.makedirs("checkpoints", exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'test_loss': test_loss,
    }, "checkpoints/main.pt")


if __name__ == "__main__":
    train()
