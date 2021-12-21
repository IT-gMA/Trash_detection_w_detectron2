from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2 import model_zoo
from configs import DEVICE_NAME, MODEL_CONFIG_FILE
import cv2


class Detector:
    def __init__(self):
        self.cfg = get_cfg()  # config object

        # Load model's configurations and the respective pretrained model
        self.cfg.merge_from_file(
            model_zoo.get_config_file(MODEL_CONFIG_FILE))  # Tell the model zoo where the model resides
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(MODEL_CONFIG_FILE)

        self.cfg.MODEL.ROI_HEADS.SCORE_HEADS_TEST = 0.7
        self.cfg.MODEL.DEVICE_NAME = DEVICE_NAME

        self.predictor = DefaultPredictor(self.cfg)  # Initialise the object predictor

    def img_pred(self, img_path):
        img = cv2.imread(img_path)
        preds = self.predictor(img)

        visualiser = Visualizer(img[:, :, ::-1], metadata=MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]),
                                instance_mode=ColorMode.IMAGE_BW)

        # Draw the predictions (bboxes) on the image using the visualiser constructed above
        output = visualiser.draw_instance_predictions(preds["instances"].to("cpu"))

        cv2.imshow("Result", output.get_image()[:, :, ::-1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
