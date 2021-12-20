import torch


MODEL_CONFIG_FILE = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"

# Dataset resources
DATASET_PATH = "../images"
TRAIN_IMG_PATH = "../images/train"
VALIDATION_IMG_PATH = "../images/valid"
TEST_IMG_PATH = "../images/test"
ANN_FILE_NAME = "_annotations.coco.json"    # Name of the annotation file in json format for COCO dataset
RESIZE_FACTOR = 0.5     # The percentage for resize the original image down to

TORCH_DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')       # Device torch will run on
DEVICE_NAME = 'None'
if TORCH_DEVICE == 'cuda':
    print("Running on cuda")
    DEVICE_NAME = 'cuda'
else:
    print("Running on cpu")
    DEVICE_NAME = 'cpu'


# Custom classes of our dataset
CLASSES = [
    'Trash', 'Negative', 'aluminium wrap', 'cardboard', 'cigarette', 'foil', 'general waste', 'glass', 'metal',
    'nylon bag', 'organic waste', 'paper', 'plastic', 'plastc bag', 'plastic bottle', 'plastic cup', 'plastic straw',
    'styrofoam', 'styrofoam cup'
]
NUM_CLASSES = 18

