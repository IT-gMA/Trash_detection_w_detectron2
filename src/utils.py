from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.config import get_cfg
from detectron2 import model_zoo
import random
import cv2
import glob
from configs import NUM_CLASSES, OUTPUT_DIR, MODEL_CONFIG_FILE, DEVICE_NAME


def draw_samples(dataset_name, sample_dir, n=1):
    dataset_custom = DatasetCatalog.get(dataset_name)
    dataset_custom_metadata = MetadataCatalog.get(dataset_name)

    for sample in random.sample(dataset_custom, n):
        img = cv2.imread(sample["file_name"])
        visualiser = Visualizer(img[:, :, ::-1], metadata=dataset_custom_metadata, scale=0.5)
        out = visualiser.draw_dataset_dict(sample)

        img_name = sample["file_name"].replace(sample_dir, "")
        out_img = out.get_image()[:, :, ::-1]
        print("dataset size {}, sample = {}".format(out_img.shape, sample))
        cv2.imshow(img_name, out.get_image()[:, :, ::-1])
        cv2.waitKey(0)


def get_train_cfg(train_dataset_name, valid_dataset_name):
    cfg = get_cfg()

    cfg.merge_from_file(model_zoo.get_config_file(MODEL_CONFIG_FILE))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(MODEL_CONFIG_FILE)
    cfg.DATASETS.TRAIN = (train_dataset_name,)
    cfg.DATASETS.TEST = (valid_dataset_name,)

    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.000025
    cfg.SOLVER.MAX_ITER = 10000
    cfg.SOLVER.STEPS = []
    #cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES
    cfg.MODEL.DEVICE_NAME = DEVICE_NAME
    cfg.OUTPUT_DIR = OUTPUT_DIR

    return cfg


def image_inference(img_dir, predictor, meta_data):
    test_images = glob.glob(f"{img_dir}/*")
    for img_path in test_images:
        img = cv2.imread(img_path)
        output = predictor(img)
        visualiser = Visualizer(img[:, :, ::-1], metadata=meta_data, scale=0.5, instance_mode=ColorMode.SEGMENTATION)
        labeled_img = visualiser.draw_instance_predictions(output["instances"].to("cpu"))

        cv2.imshow("Result", labeled_img.get_image()[:, :, ::-1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def video_inference(video_path, predictor):
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        raise Exception(f"Error opening video at {video_path}")

    success, frame = video_capture.read()       # Try to play/read the first frame of the video
    while success:  # While the video's frame is successfully read
        output = predictor(frame)
        visualiser = Visualizer(frame[:, :, ::-1], metadata={}, instance_mode=ColorMode.SEGMENTATION)
        output = visualiser.draw_instance_predictions(output["instances"].to("cpu"))

        cv2.imshow("Video", output.get_image()[:, :, ::-1])

        # Configuring key press for quitting video playback
        key = cv2.waitKey(1) & 0xFF
        if key == ord("esc") or key == ord("q"):
            break

        success, frame = video_capture.read()   # Keep playing the next frame of the video




