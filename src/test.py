from detectron2.engine import DefaultPredictor
import os
import pickle
from utils import *
from configs import CKPT_SAVE_PATH, TEST_DATA_PATH

'''with open(CKPT_SAVE_PATH, 'rb') as f:
    # Obtain the configuration file
    cfg = pickle.load(f)

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # Point to the complete trained model
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7     # Only display detected objects with confidence level greater than 70%
predictor = DefaultPredictor(cfg)  # Initialiser the object predictor'''
predictor = None

on_image_batch(TEST_DATA_PATH, predictor)
