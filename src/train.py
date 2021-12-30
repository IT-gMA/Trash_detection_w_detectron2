from detectron2.engine import DefaultTrainer
from configs import CKPT_SAVE_PATH, TRAIN_IMG_PATH, VALIDATION_IMG_PATH, ANN_FILE_NAME, \
    TRAIN_DATASET_NAME, VALIDATION_DATASET_NAME, DEVICE_NAME, VISUALISE_SAMPLES, TEST_DATASET_NAME, TEST_IMG_PATH, \
    OUTPUT_DIR
from detectron2.data.datasets import register_coco_instances
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from utils import draw_samples, get_train_cfg
import os
from dataset import custom_mapper
import pickle
import torch
from eval import evaluate_model

# Register the COCO format train and validation dataset
register_coco_instances(TRAIN_DATASET_NAME, {}, TRAIN_IMG_PATH + "/" + ANN_FILE_NAME, TRAIN_IMG_PATH)
register_coco_instances(VALIDATION_DATASET_NAME, {}, VALIDATION_IMG_PATH + "/" + ANN_FILE_NAME, VALIDATION_IMG_PATH)

if VISUALISE_SAMPLES:
    num_visualised_samples = 5
    draw_samples(dataset_name=TRAIN_DATASET_NAME, sample_dir=TRAIN_IMG_PATH, n=num_visualised_samples)


class AugTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=custom_mapper)


def main():
    print("Running on {}".format(DEVICE_NAME))
    if DEVICE_NAME == 'cuda':
        torch.cuda.empty_cache()

    cfg = get_train_cfg(train_dataset_name=TRAIN_DATASET_NAME, valid_dataset_name=VALIDATION_DATASET_NAME)

    # Save this cfg for testing or inferencing later
    with open(CKPT_SAVE_PATH, 'wb') as f:
        pickle.dump(cfg, f, protocol=pickle.HIGHEST_PROTOCOL)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    trainer = AugTrainer(cfg)
    trainer.resume_or_load(resume=True)
    trainer.train()

    # Start evaluation
    evaluate_model(after_train=True, cfg=cfg, predictor=trainer, eval_dataset_name=VALIDATION_DATASET_NAME)


if __name__ == '__main__':
    main()
