from pathlib import Path

import cv2
import numpy as np
import torch

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from detectron2.data import DatasetCatalog

from src.polygon_rcnn.register_dataset import register_blines


MODEL_PATH = "output_polygon_head/model_best.pth"

TEST_DATASET = "blines_val"

OUTPUT_DIR = "results_polygon_head"

SCORE_THRESHOLD = 0.50

NUM_KEYPOINTS = 4

# Must match train_polygon_head.py's KEYPOINT_POOLER_RESOLUTION.
KEYPOINT_POOLER_RESOLUTION = 14

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def build_predictor():

    cfg = get_cfg()

    cfg.MODEL.DEVICE = DEVICE

    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"
        )
    )

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

    cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = NUM_KEYPOINTS

    cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION = KEYPOINT_POOLER_RESOLUTION

    cfg.TEST.KEYPOINT_OKS_SIGMAS = [0.05] * NUM_KEYPOINTS

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

        # Predições -- poligono conectando os 4 vertices previstos (vermelho)

        keypoints = instances.pred_keypoints.numpy()  # (N, 4, 3): x, y, heatmap score

        scores = instances.scores.numpy()

        for kpts, score in zip(keypoints, scores):

            polygon = kpts[:, :2].astype(np.int32)

            cv2.polylines(
                vis,
                [polygon],
                True,
                (0, 0, 255),
                2,
            )

            x1, y1 = polygon[0]

            cv2.putText(
                vis,
                f"{score:.3f}",
                (int(x1), max(20, int(y1) - 5)),
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
