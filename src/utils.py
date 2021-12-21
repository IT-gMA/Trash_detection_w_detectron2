from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.config import get_cfg
from detectron2 import model_zoo
import random
import cv2
import matplotlib.pyplot as plt
from configs import TRAIN_DATASET_NAME, VALIDATION_DATASET_NAME, NUM_CLASSES, OUTPUT_DIR, MODEL_CONFIG_FILE, DEVICE_NAME


def draw_samples(dataset_name, sample_dir, n=1):
    dataset_custom = DatasetCatalog.get(dataset_name)
    dataset_custom_metadata = MetadataCatalog.get(dataset_name)

    for sample in random.sample(dataset_custom, n):
        img = cv2.imread(sample["file_name"])
        visualiser = Visualizer(img[:, :, ::-1], metadata=dataset_custom_metadata, scale=0.5)
        out = visualiser.draw_dataset_dict(sample)

        img_name = sample["file_name"].replace(sample_dir, "")
        cv2.imshow(img_name, out.get_image()[:, :, ::-1])
        cv2.waitKey(0)


def get_train_cfg(train_dataset_name, valid_dataset_name):
    cfg = get_cfg()

    cfg.merge_from_file(model_zoo.get_config_file(MODEL_CONFIG_FILE))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(MODEL_CONFIG_FILE)
    cfg.DATASETS.TRAIN = (train_dataset_name,)
    cfg.DATASETS.TEST = (valid_dataset_name,)

    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 6
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 300
    cfg.SOLVER.STEPS = []
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES
    cfg.MODEL.DEVICE_NAME = DEVICE_NAME
    cfg.OUTPUT_DIR = OUTPUT_DIR

    return cfg

