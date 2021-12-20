from detectron2.engine import DefaultTrainer
from configs import CKPT_SAVE_PATH, TRAIN_IMG_PATH, VALIDATION_IMG_PATH, ANN_FILE_NAME, \
    TRAIN_DATASET_NAME, VALIDATION_DATASET_NAME, DEVICE_NAME
from detectron2.data.datasets import register_coco_instances
from utils import *
import os
import pickle

register_coco_instances(TRAIN_DATASET_NAME, {}, TRAIN_IMG_PATH + "/" + ANN_FILE_NAME, TRAIN_IMG_PATH)
register_coco_instances(VALIDATION_DATASET_NAME, {}, VALIDATION_IMG_PATH + "/" + ANN_FILE_NAME, VALIDATION_IMG_PATH)


# draw_samples(dataset_name=TRAIN_DATASET_NAME, sample_dir=TRAIN_IMG_PATH, n=4)

def main():
    print("Running on {}".format(DEVICE_NAME))
    cfg = get_train_cfg(train_dataset_name=TRAIN_DATASET_NAME, valid_dataset_name=VALIDATION_DATASET_NAME)

    # Save this cfg for testing or inferencing later
    with open(CKPT_SAVE_PATH, 'wb') as f:
        pickle.dump(cfg, f, protocol=pickle.HIGHEST_PROTOCOL)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)  # Since we're not continuing the training from any checkpoint
    trainer.train()


if __name__ == '__main__':
    main()
