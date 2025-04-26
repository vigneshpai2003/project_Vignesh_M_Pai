### Dataset configuration

# frame size
FRAME_HEIGHT = 64
FRAME_WIDTH = 64

# number of frames in a clip
CLIP_COUNT = 2
# time step between frames in a clip
CLIP_STEP = 4
# number of frames the clip represents
CLIP_LENGTH = CLIP_COUNT * CLIP_STEP
# number of frames to skip between clips
CLIP_STRIDE = CLIP_LENGTH
# config string for the dataset
DATA_CONFIG = f'{FRAME_HEIGHT}_{FRAME_WIDTH}_{CLIP_COUNT}_{CLIP_STEP}_{CLIP_LENGTH}_{CLIP_STRIDE}'

### Training configuration

TRAIN_FACTOR = 0.8
TEST_FACTOR = 1 - TRAIN_FACTOR
BATCH_SIZE = 32
LEARNING_RATE = 0.001
TOTAL_EPOCHS = 3