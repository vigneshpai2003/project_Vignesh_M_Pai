import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from config import *
from dataset import DashCamDataset, DashCamDataLoader
from model import Speed3DCNN

def plot_result():
    dataset = DashCamDataset()
    test_dataset = dataset.get_test_train_split()[1]
    test_loader = DashCamDataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = Speed3DCNN()

    if not os.path.exists("checkpoints/main.pt"):
        print("Checkpoint not found. Please train the model first.")
        return
    
    checkpoint = torch.load("checkpoints/main.pt", weights_only=False)   
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    true_vals = []
    preds = []

    with torch.no_grad():
        for x, y in test_loader:
            y_pred = model(x)
            true_vals.append(y.numpy())
            preds.append(y_pred.numpy())

    true_vals = np.concatenate(true_vals).flatten()
    preds = np.concatenate(preds).flatten()
    
    true_vals = dataset.get_speed(true_vals)
    preds = dataset.get_speed(preds)

    plt.figure(figsize=(8, 5))
    plt.scatter(true_vals, preds - true_vals, alpha=0.5, s=10)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("True Speed (y)")
    plt.ylabel("Prediction Error (y_pred - y)")
    plt.title("Prediction Error vs True Speed")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_result()