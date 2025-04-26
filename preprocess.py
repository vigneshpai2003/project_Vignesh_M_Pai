import os
import cv2
import numpy as np
from tqdm import tqdm

from config import *


def load_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(frame)
    cap.release()
    return np.array(frames, dtype=np.float32) / 255.0


def create_clips(frames, speeds):
    X, y = [], []

    for i in range(0, len(frames) - CLIP_LENGTH, CLIP_STRIDE):
        frames[i], frames[i + CLIP_STEP],
        clip = frames[i:i+CLIP_LENGTH:CLIP_STEP]
        clip_speeds = speeds[i:i+CLIP_LENGTH:CLIP_STEP]

        if len(clip) == CLIP_COUNT:
            X.append(clip)
            y.append(np.mean(clip_speeds))

    return np.expand_dims(np.array(X), axis=1), np.array(y, dtype=np.float32)


def preprocess_segment(segment):
    frames = load_frames(segment + 'video.hevc')
    frame_times = np.load(segment + 'global_pose/frame_times')[:len(frames)]
    speeds = np.load(segment + 'processed_log/CAN/speed/value')
    speed_times = np.load(segment + 'processed_log/CAN/speed/t')

    assert len(frame_times) == len(frames), "Frame times and frames length mismatch"
    assert len(speed_times) == len(speeds), "Speed times and speeds length mismatch"

    for i, t in enumerate(frame_times):
        if t > speed_times[0]:
            start = i
            break

    for i, t in enumerate(frame_times):
        if t > speed_times[-1]:
            end = i
            break
    else:
        end = len(speed_times)

    frames = frames[start:end]
    frame_times = frame_times[start:end]
    frame_speeds = []

    for t, frame in zip(frame_times, frames):
        i = np.searchsorted(speed_times, t)

        t0, t1 = speed_times[i - 1], speed_times[i]
        s0, s1 = speeds[i - 1], speeds[i]
        alpha = (t - t0) / (t1 - t0)

        frame_speeds.append(s0 + alpha * (s1 - s0))

    return create_clips(frames, speeds)


def preprocess_route(route):
    all_X, all_y = [], []

    for segment in tqdm(os.listdir(route), desc=f"Processing route {route}", leave=False):
        X, y = preprocess_segment(route + segment + '/')
        all_X.append(X)
        all_y.append(y)

    all_X = np.concatenate(all_X, axis=0)
    all_y = np.concatenate(all_y, axis=0)

    return all_X, all_y


def preprocess_chunk(chunk):
    all_X, all_y = [], []

    for route in tqdm(os.listdir(chunk), desc=f"Processing chunk {chunk}"):
        if not os.path.isdir(chunk + route):
            continue
        X, y = preprocess_route(chunk + route + '/')
        all_X.append(X)
        all_y.append(y)

    all_X = np.concatenate(all_X, axis=0)
    all_y = np.concatenate(all_y, axis=0)
    
    return all_X, all_y


def preprocess_all():
    all_X, all_y = [], []
    
    for chunk in ['./Chunk_1/']:
        X, y = preprocess_chunk(chunk)
        all_X.append(X)
        all_y.append(y)
    
    all_X = np.concatenate(all_X, axis=0)
    all_y = np.concatenate(all_y, axis=0)

    os.makedirs(f'processed/{DATA_CONFIG}', exist_ok=True)
    np.save(f'processed/{DATA_CONFIG}/samples.npy', all_X)
    np.save(f'processed/{DATA_CONFIG}/labels.npy', all_y)


if __name__ == '__main__':
    preprocess_all()
