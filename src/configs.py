import torch


MODEL_CONFIG_FILE = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"

# Dataset resources
DATASET_PATH = "../images"
TRAIN_IMG_PATH = "../images/train"
VALIDATION_IMG_PATH = "../images/valid"
TEST_IMG_PATH = "../images/test"
ANN_FILE_NAME = "_annotations.coco.json"    # Name of the annotation file in json format for COCO dataset
RESIZE_FACTOR = 0.5     # The percentage for resize the original image down to

DEVICE_NAME = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')       # Device torch will run on
DEVICE_NAME = f"{DEVICE_NAME}"

# Custom classes of our dataset
CLASSES = [
    'Trash', 'Negative', 'aluminium wrap', 'cardboard', 'cigarette', 'foil', 'general waste', 'glass', 'metal',
    'nylon bag', 'organic waste', 'paper', 'plastic', 'plastc bag', 'plastic bottle', 'plastic cup', 'plastic straw',
    'styrofoam', 'styrofoam cup'
]
NUM_CLASSES = 18

OUTPUT_DIR = "outputs"
TRAIN_DATASET_NAME = "trash_train"
VALIDATION_DATASET_NAME = "trash_valid"
TEST_DATASET_NAME = "trash_test"
CKPT_SAVE_PATH = "OD_cfg.pickle"

