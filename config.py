"""
This file contains the configuration for the dataset generation and training parameters.
"""

# Dataset configuration

# frame size

FRAME_HEIGHT = 64
FRAME_WIDTH = 64

# number of frames in a clip
CLIP_COUNT = 2
# time step between frames in a clip
CLIP_STEP = 8
# number of frames the clip represents
CLIP_LENGTH = CLIP_COUNT * CLIP_STEP
# number of frames to skip between clips
CLIP_STRIDE = CLIP_LENGTH // 2
# config string for the dataset
DATA_CONFIG = f'{CLIP_COUNT}x{FRAME_HEIGHT}x{FRAME_WIDTH}_step={CLIP_STEP}_stride={CLIP_STRIDE}'

# Training configuration

# minimum speed to consider a clip
SPEED_CUT_OFF_KMPH = 0
SPEED_CUT_OFF = SPEED_CUT_OFF_KMPH / 3.6

TRAIN_FACTOR = 0.8
TEST_FACTOR = 1 - TRAIN_FACTOR

SPLIT_CONFIG = f'split={round(TRAIN_FACTOR * 100)}%_cutoff={SPEED_CUT_OFF_KMPH}'

# parameters to modify on the fly
BATCH_SIZE = 32
LEARNING_RATE = 0.000001
TOTAL_EPOCHS = 10

SUMMARY = f"""
Dataset configuration:
    Frame size: {FRAME_HEIGHT}x{FRAME_WIDTH}
    Clip count: {CLIP_COUNT}
    Clip step: {CLIP_STEP}
    Clip length: {CLIP_LENGTH}
    Clip stride: {CLIP_STRIDE}

Training configuration:
    Speed cut off: {SPEED_CUT_OFF_KMPH} km/h
    Train test split: {round(TRAIN_FACTOR * 100)}% train, {round(TEST_FACTOR * 100)}% test

Flyable parameters:
    Batch size: {BATCH_SIZE}
    Learning rate: {LEARNING_RATE}
    Total epochs: {TOTAL_EPOCHS}
"""
