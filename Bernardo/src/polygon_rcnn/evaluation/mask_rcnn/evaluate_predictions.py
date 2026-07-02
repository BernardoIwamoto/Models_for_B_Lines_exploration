from pathlib import Path
import csv

import cv2
import numpy as np

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from detectron2.data import DatasetCatalog

from src.polygon_rcnn.register_dataset import register_blines

from src.polygon_rcnn.evaluation.metrics.segmentation import (
    iou,
    dice,
    precision,
    recall,
    f1,
)

from src.polygon_rcnn.evaluation.metrics.geometry import area

import torch


MODEL_PATH = "output_maskrcnn/model_final.pth"

DATASET = "blines_val"

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

SCORE_THRESHOLD = 0.5

OUTPUT = Path(__file__).parent / "outputs"
OUTPUT.mkdir(exist_ok=True)

CSV_FILE = OUTPUT / "metrics.csv"

def build_predictor():

    cfg = get_cfg()

    cfg.MODEL.DEVICE = DEVICE

    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
    )

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

    cfg.MODEL.WEIGHTS = MODEL_PATH

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SCORE_THRESHOLD

    return DefaultPredictor(cfg)

def polygon_to_mask(annotation, shape):

    mask = np.zeros(shape[:2], dtype=np.uint8)

    polygon = np.array(
        annotation["segmentation"][0],
        dtype=np.int32,
    ).reshape(-1, 2)

    cv2.fillPoly(mask, [polygon], 1)

    return mask.astype(bool)

def main():

    print("Registering dataset...")
    register_blines()

    print("Building predictor...")
    predictor = build_predictor()

    print("Loading dataset...")
    dataset = DatasetCatalog.get(DATASET)

    print(f"Dataset size: {len(dataset)}")

    with open(CSV_FILE, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "image",
            "score",
            "IoU",
            "Dice",
            "Precision",
            "Recall",
            "F1",
            "PredArea",
            "GtArea",
            "TP",
            "FP",
        ])

        for i, sample in enumerate(dataset):

            matched = set()

            image = cv2.imread(sample["file_name"])

            outputs = predictor(image)

            instances = outputs["instances"].to("cpu")

            if len(instances) == 0:
                continue

            gt_masks = [
                polygon_to_mask(a, image.shape)
                for a in sample["annotations"]
            ]

            pred_masks = instances.pred_masks.numpy()

            scores = instances.scores.numpy()

            for pred_mask, score in zip(pred_masks, scores):

                best = -1
                best_iou = 0

                for idx, gt in enumerate(gt_masks):

                    if idx in matched:
                        continue

                    current = iou(pred_mask, gt)

                    if current > best_iou:
                        best_iou = current
                        best = idx

                tp = 0
                fp = 1

                if best != -1 and best_iou >= 0.5:

                    matched.add(best)

                    tp = 1
                    fp = 0

                gt = gt_masks[best] if best != -1 else np.zeros_like(pred_mask)

                writer.writerow([
                    Path(sample["file_name"]).name,
                    float(score),
                    best_iou,
                    dice(pred_mask, gt),
                    precision(pred_mask, gt),
                    recall(pred_mask, gt),
                    f1(pred_mask, gt),
                    area(pred_mask),
                    area(gt),
                    tp,
                    fp,
                ])

    print("Finished.")

if __name__ == "__main__":
    main()