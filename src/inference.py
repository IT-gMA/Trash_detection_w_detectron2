from detectron2.engine import DefaultPredictor
import pickle
from utils import *
from configs import CKPT_SAVE_PATH, INFERENCE_IMG_PATH, TEST_IMG_PATH, ANN_FILE_NAME, TEST_DATASET_NAME, \
    INF_MODEL_PATH, MIN_CONFIDENCE, INF_RESULT_SAVE_DIR, INFERENCE_VIDEO_PATH
from detectron2.data.datasets import register_coco_instances
import os
import argparse

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', type=str, default='false',
                        help='save the resulting images from inference to a directory')
    parser.add_argument('--show', type=str, default='false', help='display the labeled images')
    parser.add_argument('--mode', type=str, default='img', help='display the labeled images')
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


def is_show():
    show_images = False
    opt = parse_opt()
    if opt.show:
        arg = opt.show
        if arg != 'false':
            print("Show labeled images")
            show_images = True
        else:
            print("Display images disabled")
    return show_images


def is_save():
    save_images = False
    opt = parse_opt()
    if opt.save:
        arg = opt.save
        if arg != 'false':
            print("Save inferenced images at {}".format(INF_RESULT_SAVE_DIR))
            save_images = True
        else:
            print("No save")
    return save_images


def inf_mode():
    inference_mode = 0
    opt = parse_opt()
    if opt.mode:
        arg = opt.mode
        if arg == "img" or arg == "image" or arg == "IMG" or arg == "images" or arg == "IMAGES":
            # Still image inference
            print("Inference on images")
        elif arg == "video" or arg == "VIDEO" or arg == "vid" or arg == "VID":
            # Plain video inference
            inference_mode = 1
            print("Video inference")
        elif arg == "live" or arg == "LIVE" or arg == "live vid" or arg == "LIVE VID" or arg == "live_video":
            # Live video inference
            inference_mode = 2
            print("Live video inference")
        else:
            inference_mode = 3
    return inference_mode


def main():
    with open(CKPT_SAVE_PATH, 'rb') as f:
        # Obtain the configuration file
        cfg = pickle.load(f)

    register_coco_instances(TEST_DATASET_NAME, {}, TEST_IMG_PATH + "/" + ANN_FILE_NAME, TEST_IMG_PATH)
    dataset_custom = DatasetCatalog.get(TEST_DATASET_NAME)
    dataset_custom_metadata = MetadataCatalog.get(TEST_DATASET_NAME)

    cfg.MODEL.WEIGHTS = os.path.join(INF_MODEL_PATH)  # Point to the complete trained model
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = MIN_CONFIDENCE  # Only display detected objects with confidence level greater than 70%
    predictor = DefaultPredictor(cfg)  # Initialise the object predictor

    if inf_mode() == 0:  # Image mode is selected for inference
        show_img = is_show()
        save_img = is_save()
        image_inference(INFERENCE_IMG_PATH, predictor, dataset_custom_metadata, show_img, save_img)
    elif inf_mode() == 1:  # video is selected for inference
        video_inference(INFERENCE_VIDEO_PATH, predictor, dataset_custom_metadata)
    elif inf_mode() == 2:  # live video inference
        video_inference(None, predictor, dataset_custom_metadata)
    else:
        print("Invalid argument")


if __name__ == '__main__':
    main()
