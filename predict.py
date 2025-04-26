import os
import sys
import torch

from config import *
from preprocess import preprocess_segment
from model import Speed3DCNN
from dataset import DashCamDataLoader, DashCamDataset


def predict_segment(segment):
    """
    Predict the speed of a dashcam video using a pre-trained model.

    Args:
        segment (str): Path to the data folder.

    Returns:
        float: Predicted speed in km/h.
    """
    # Suppress print statements from the dataset loading
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    dataset = DashCamDataset()
    sys.stdout = original_stdout

    # Load the model
    model = Speed3DCNN()
    checkpoint = torch.load('checkpoints/final.pt', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Preprocess the video
    clips = torch.from_numpy(preprocess_segment(segment)[0])
    clips = (clips - clips.mean(dim=0)) / clips.std(dim=0)

    # Create a DataLoader for the video data
    clips = DashCamDataLoader(clips, batch_size=1, shuffle=False)

    # Make predictions
    speeds = []

    with torch.no_grad():
        for x in clips:
            speed = dataset.get_speed_kmph(model(x)).item()
            speeds.extend([speed] * CLIP_LENGTH)

    return speeds


def predict(segment_paths):
    """
    Predict the speed of dashcam videos in a folder.

    Args:
        segment_paths (str): Path to the data folder.

    Returns:
        list: List of predicted speeds in km/h.
    """
    speeds = []
    for segment in segment_paths:
        speeds.append(predict_segment(segment))
    return speeds


if __name__ == "__main__":
    predict(['data/1/', 'data/2/', 'data/3/', 'data/4/', 'data/5/', 'data/6/', 'data/7/', 'data/8/', 'data/9/', 'data/10/'])
