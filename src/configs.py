import torch


MODEL_CONFIG_FILE = "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"

# Dataset resources
DATASET_PATH = "../images"
TRAIN_IMG_PATH = "../images/train"
VALIDATION_IMG_PATH = "../images/valid"
TEST_IMG_PATH = "../images/test"
# Name of the annotation file in json format for COCO dataset, given the config file's name is the same across train, valid and test
ANN_FILE_NAME = "_annotations.coco.json"
RESIZE_FACTOR = 0.2    # The percentage for resize the original image down to
MIN_DIM_THRESH = 720

VISUALISE_SAMPLES = False

DEVICE_NAME = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')       # Device torch will run on
DEVICE_NAME = f"{DEVICE_NAME}"

# Custom classes of our dataset
CLASSES = [
    'Trash', 'aluminium wrap', 'cardboard', 'cigarette', 'foil', 'general waste', 'glass',
    'metal', 'negative', 'nylon bag', 'paper', 'plastic', 'plastic bottle', 'plastic cup',
    'plastic film', 'plastic straw', 'styrofoam', 'styrofoam cup'
]
NUM_CLASSES = 18

OUTPUT_DIR = "outputs"
TRAIN_DATASET_NAME = "trash_train"
VALIDATION_DATASET_NAME = "trash_valid"
TEST_DATASET_NAME = "trash_test"
CKPT_SAVE_PATH = "OD_cfg.pickle"
INFERENCE_IMG_PATH = "../test_data"
INFERENCE_VIDEO_PATH = "../.mp4"

