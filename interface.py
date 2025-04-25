# replace MyCustomModel with the name of your model
from model import MyCustomModel as TheModel

# change my_descriptively_named_train_function to
# the function inside train.py that runs the training loop.
from train import my_descriptively_named_train_function as the_trainer

# change cryptic_inf_f to the function inside predict.py that
# can be called to generate inference on a single image/batch.
from predict import cryptic_inf_f as the_predictor

from dataset import DashCamDataset as TheDataset

from dataset import DashCamDataLoader as the_dataloader
