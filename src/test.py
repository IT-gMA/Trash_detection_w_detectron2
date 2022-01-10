from detectron2.engine import DefaultPredictor
import pickle
from utils import *
from configs import CKPT_SAVE_PATH, INFERENCE_IMG_PATH, TEST_IMG_PATH, ANN_FILE_NAME, TEST_DATASET_NAME, \
    INF_MODEL_PATH, MIN_CONFIDENCE
from detectron2.data.datasets import register_coco_instances
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

with open(CKPT_SAVE_PATH, 'rb') as f:
    # Obtain the configuration file
    cfg = pickle.load(f)


register_coco_instances(TEST_DATASET_NAME, {}, TEST_IMG_PATH + "/" + ANN_FILE_NAME, TEST_IMG_PATH)
dataset_custom = DatasetCatalog.get(TEST_DATASET_NAME)
dataset_custom_metadata = MetadataCatalog.get(TEST_DATASET_NAME)

cfg.MODEL.WEIGHTS = os.path.join(INF_MODEL_PATH)  # Point to the complete trained model
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = MIN_CONFIDENCE    # Only display detected objects with confidence level greater than 70%
predictor = DefaultPredictor(cfg)  # Initialise the object predictor

image_inference(INFERENCE_IMG_PATH, predictor, dataset_custom_metadata)
