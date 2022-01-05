import os
import numpy as np
import json
import torch
from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog, DatasetCatalog, detection_utils
from configs import CLASSES, DATASET_PATH, ANN_FILE_NAME, TEST_DATASET_NAME, RESIZE_FACTOR, TEST_IMG_PATH, MIN_DIM_THRESH
from utils import draw_samples
from detectron2.data import transforms as T
import copy
import cv2


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
                bbox = annotation["bbox"]
                bbox = np.array(bbox)
                bbox = RESIZE_FACTOR * bbox
                bbox = np.array(bbox, dtype=int).tolist()
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


def visualize_sample(image, target):
    box = target['bbox'][0]
    label = CLASSES[target['category_id'][0]]
    cv2.rectangle(
        image,
        (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
        (0, 255, 0), 2
    )
    cv2.putText(
        image, label, (int(box[0]), int(box[1] - 5)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
    )
    # print("Image size is {}".format(image.shape))
    cv2.imshow('Image', image)
    cv2.waitKey(0)


def register():
    for d in ["train", "valid", "test"]:
        DatasetCatalog.register("trash_" + d, lambda d=d: get_garbage_dicts(DATASET_PATH + "/" + d))
        MetadataCatalog.get("trash_" + d).set(thing_classes=CLASSES)


#garbage_metadata = MetadataCatalog.get("trash_train")
#print(garbage_metadata)


def custom_mapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)
    image = detection_utils.read_image(dataset_dict["file_name"], format="BGR")

    new_height = int(dataset_dict["height"] * RESIZE_FACTOR)
    new_width = int(dataset_dict["width"] * RESIZE_FACTOR)
    new_dimension = (new_height, new_width)
    if new_dimension < (MIN_DIM_THRESH, MIN_DIM_THRESH):
        new_dimension = (MIN_DIM_THRESH, MIN_DIM_THRESH)
        new_height = MIN_DIM_THRESH
        new_width = MIN_DIM_THRESH

    dataset_dict["height"] = new_height
    dataset_dict["width"] = new_width

    transform_list = [T.Resize(new_dimension),      # Scale down the image a bit
                      T.RandomBrightness(0.8, 1.1),
                      T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
                      T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
                      T.RandomContrast(intensity_min=0.8, intensity_max=1.1),
                      T.RandomLighting(scale=0.6),
                      T.RandomRotation(sample_style="range", angle=[-30, 5]),
                      T.RandomCrop(crop_type="relative", crop_size=(0.6, 0.5))
                      ]
    image, transforms = T.apply_transform_gens(transform_list, image)
    dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

    annos = [
        detection_utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in dataset_dict.pop("annotations")
        if obj.get("iscrowd", 0) == 0
    ]
    instances = detection_utils.annotations_to_instances(annos, image.shape[:2])
    dataset_dict["instances"] = detection_utils.filter_empty_instances(instances)
    return dataset_dict


