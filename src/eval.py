from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
import pickle
from configs import TEST_DATASET_NAME, TEST_IMG_PATH, ANN_FILE_NAME, CKPT_SAVE_PATH, OUTPUT_DIR


def evaluate_model(after_train=False, cfg=None, predictor=None, eval_dataset_name=None):
    if not after_train:
        register_coco_instances(TEST_DATASET_NAME, {}, TEST_IMG_PATH + "/" + ANN_FILE_NAME, TEST_IMG_PATH)
        with open(CKPT_SAVE_PATH, 'rb') as f:
            # Obtain the configuration file
            cfg = pickle.load(f)

        predictor = DefaultPredictor(cfg)
        eval_dataset_name = TEST_DATASET_NAME

    evaluator = COCOEvaluator(eval_dataset_name, cfg, False, output_dir=OUTPUT_DIR)
    eval_loader = build_detection_test_loader(cfg, eval_dataset_name)
    inference_on_dataset(predictor.model, eval_loader, evaluator)


if __name__ == '__main__':
    evaluate_model()
