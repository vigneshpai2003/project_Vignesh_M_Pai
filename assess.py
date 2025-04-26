import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from tqdm import tqdm

from config import *
from dataset import DashCamDataset, DashCamDataLoader
from model import Speed3DCNN


def plot_checkpoint(checkpoint_path, model, dataset, test_loader, N):
    """
    Plot the RMSE of the model predictions against the true speed for a given checkpoint.
    """
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for x, y in test_loader:
            y_true.append(y.numpy())
            y_pred.append(model(x).numpy())

    y_true = np.concatenate(y_true).flatten()
    y_pred = np.concatenate(y_pred).flatten()

    y_true = dataset.get_speed_kmph(y_true)
    y_pred = dataset.get_speed_kmph(y_pred)

    error = (y_pred - y_true)

    # Bin settings
    bin_width = 2
    bins = np.arange(0, np.max(y_true) + bin_width, bin_width)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Digitize true speeds into bins
    bin_indices = np.digitize(y_true, bins)

    # Store RMSE per bin
    rmse_per_bin = []
    count_per_bin = []

    for i in range(1, len(bins)):
        idx = bin_indices == i
        if np.sum(idx) > 0:
            rmse = np.sqrt(np.mean(error[idx]**2))
            rmse_per_bin.append(rmse)
        else:
            rmse_per_bin.append(np.nan)  # or 0 if you prefer
        count_per_bin.append(np.sum(idx))

    plt.plot(bin_centers, rmse_per_bin, marker='o', alpha=min(1, checkpoint['epoch'] / N)**2, c='blue')


def assess_checkpoint(checkpoint_path):
    """
    Assess the model's performance on the test dataset using a specific checkpoint or checkpoint folder.
    """
    dataset = DashCamDataset()
    test_dataset = dataset.get_test_train_split()[1]
    test_loader = DashCamDataLoader(test_dataset, shuffle=False)

    model = Speed3DCNN()

    if checkpoint_path.endswith('.pt'):
        if not os.path.exists(checkpoint_path):
            print("Checkpoint not found. Please train the model first.")
            return

        checkpoints = [checkpoint_path]
    else:
        if not os.path.exists(checkpoint_path + "1.pt"):
            print("Checkpoint not found. Please train the model first.")
            return

        checkpoints = [checkpoint_path + fname for fname in os.listdir(checkpoint_path) if fname.endswith('.pt')]

    plt.figure(figsize=(8, 6))

    for checkpoint in tqdm(checkpoints, desc="Loading checkpoints"):
        plot_checkpoint(checkpoint, model, dataset, test_loader, len(checkpoints))

    plt.xlabel("True Speed (km/h)")
    plt.ylabel("RMSE (km/h)")
    plt.title("Prediction RMSE vs True Speed")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python assess.py <checkpoint_path>")
        sys.exit(1)

    checkpoint_path = sys.argv[1]
    assess_checkpoint(checkpoint_path)
