import os
import numpy as np
import json
from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from configs import CLASSES, NUM_CLASSES, TRAIN_IMG_PATH, VALIDATION_IMG_PATH, DATASET_PATH, ANN_FILE_NAME, TRAIN_DATASET_NAME, VALIDATION_DATASET_NAME
from detectron2.data.datasets import register_coco_instances
import glob
import cv2
import random


def get_garbage_dicts(img_directory):
    dataset_dicts = []
    json_file = os.path.join(img_directory, ANN_FILE_NAME)
    with open(json_file) as f:
        anns = json.load(f)
        imgs_anns = anns["images"]
        bboxes_anns = anns["annotations"]

    for idx, image in enumerate(imgs_anns):
        record = {"file_name": os.path.join(img_directory, image["file_name"]), "image_id": image["id"],
                  "height": image["height"],
                  "width": image["width"]}

        objs = []
        for idx, annotation in enumerate(bboxes_anns):
            img_id = annotation["image_id"]
            if img_id == record["image_id"]:
                obj = {
                    "bbox": annotation["bbox"],
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "segmentation": annotation["segmentation"],
                    "category_id": annotation["category_id"],
                    "area": annotation["area"]
                }
                objs.append(obj)

        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts


def register():
    for d in ["train", "valid"]:
        DatasetCatalog.register("trash_" + d, lambda d=d: get_garbage_dicts(DATASET_PATH + "/" + d))
        MetadataCatalog.get("trash_" + d).set(thing_classes=CLASSES)


garbage_metadata = MetadataCatalog.get("trash_train")
#print(garbage_metadata)


'''if __name__ == '__main__':
    get_garbage_dicts()'''
