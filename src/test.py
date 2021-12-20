from detectron2.engine import DefaultPredictor
import os
import pickle
from utils import *
from configs import CKPT_SAVE_PATH

with open(CKPT_SAVE_PATH, 'rb') as f:
    # Obtain the configuration file
    cfg = pickle.load(f)