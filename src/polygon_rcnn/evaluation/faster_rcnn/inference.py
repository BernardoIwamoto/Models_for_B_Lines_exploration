from pathlib import Path

import cv2
import numpy as np

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from detectron2.data import DatasetCatalog

from src.polygon_rcnn.register_dataset import register_blines


MODEL_PATH = "output_faster_rcnn/model_best.pth"

TEST_DATASET = "blines_val"

OUTPUT_DIR = "results_faster_rcnn"

SCORE_THRESHOLD = 0.50

DEVICE = "mps"      # use "cpu" se necessário


def build_predictor():

    cfg = get_cfg()

    cfg.MODEL.DEVICE = DEVICE

    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
        )
    )

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

    cfg.MODEL.WEIGHTS = MODEL_PATH

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SCORE_THRESHOLD

    return DefaultPredictor(cfg)


def main():

    register_blines()

    predictor = build_predictor()

    dataset = DatasetCatalog.get(TEST_DATASET)

    output = Path(OUTPUT_DIR)
    output.mkdir(exist_ok=True)

    for sample in dataset:

        image = cv2.imread(sample["file_name"])

        prediction = predictor(image)

        instances = prediction["instances"].to("cpu")

        vis = image.copy()

        # Ground Truth (verde)

        for ann in sample["annotations"]:

            polygon = np.array(
                ann["segmentation"][0],
                dtype=np.int32,
            ).reshape(-1, 2)

            cv2.polylines(
                vis,
                [polygon],
                True,
                (0, 255, 0),
                2,
            )

        # Predições -- bounding box (vermelho), sem máscara (Faster R-CNN não tem mask head)

        boxes = instances.pred_boxes.tensor.numpy()

        scores = instances.scores.numpy()

        for box, score in zip(boxes, scores):

            x1, y1, x2, y2 = box.astype(int)

            cv2.rectangle(
                vis,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                2,
            )

            cv2.putText(
                vis,
                f"{score:.3f}",
                (x1, max(20, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

        out_name = Path(sample["file_name"]).name

        cv2.imwrite(
            str(output / out_name),
            vis,
        )

        print(f"Saved {out_name}")


if __name__ == "__main__":
    main()
