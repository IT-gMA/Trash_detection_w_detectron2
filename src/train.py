import argparse
from detectron2.engine import DefaultTrainer
from configs import CKPT_SAVE_PATH, TRAIN_IMG_PATH, VALIDATION_IMG_PATH, ANN_FILE_NAME, \
    TRAIN_DATASET_NAME, VALIDATION_DATASET_NAME, DEVICE_NAME, VISUALISE_SAMPLES, TEST_DATASET_NAME, TEST_IMG_PATH, \
    OUTPUT_DIR, EVAL_OUTPUT_DIR, NUM_GPUS_RES
from detectron2.data.datasets import register_coco_instances
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.engine import launch
from utils import draw_samples, get_train_cfg
import os
from dataset import custom_mapper, custom_mapper_valididation
import pickle
import torch
import logging
from collections import OrderedDict
from eval import evaluate_model
#Evaluation with AP metric
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, DatasetEvaluators

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

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name, mapper=custom_mapper_valididation)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return COCOEvaluator(dataset_name, output_dir=output_folder)


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default='false', help='resume training progress from a checkpoint .pth')
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


def is_resumed():
    resume_training = False
    opt = parse_opt()
    if opt.resume:
        ckpt_path = opt.resume
        if ckpt_path != 'false':
            print("Resume training from {}".format(ckpt_path))
            resume_training = True
        else:
            print("Start from scratch: ")
    return resume_training


def main():
    # Train model from scratch or from previous checkpoint (.pth) saved in OUTPUT_DIR so no need to specify ckpt path
    # in run arguments
    resume_training = is_resumed()

    print("Running on {}".format(DEVICE_NAME))
    if DEVICE_NAME == 'cuda':
        torch.cuda.empty_cache()

    cfg = get_train_cfg(train_dataset_name=TRAIN_DATASET_NAME, valid_dataset_name=VALIDATION_DATASET_NAME)

    # Save this cfg for testing or inferencing later
    with open(CKPT_SAVE_PATH, 'wb') as f:
        pickle.dump(cfg, f, protocol=pickle.HIGHEST_PROTOCOL)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    trainer = AugTrainer(cfg)
    trainer.resume_or_load(resume=resume_training)
    trainer.train()

    # Start evaluation
    eval_loader = AugTrainer.build_test_loader(cfg, VALIDATION_DATASET_NAME)
    evaluator = AugTrainer.build_evaluator(cfg, VALIDATION_DATASET_NAME, EVAL_OUTPUT_DIR)
    AugTrainer.test(cfg, trainer.model, evaluator)


if __name__ == '__main__':
    launch(main_func=main(), num_gpus_per_machine=NUM_GPUS_RES)
    #main()
