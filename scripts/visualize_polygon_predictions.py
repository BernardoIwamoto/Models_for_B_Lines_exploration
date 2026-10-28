"""Render validation ground truth and Polygon Head predictions for a checkpoint."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog
from detectron2.engine import DefaultPredictor

from src.polygon_rcnn.polygon_vertex_head import PolygonVertexHead  # noqa: F401
from src.polygon_rcnn.register_dataset import register_blines


def build_predictor(checkpoint: Path, score_threshold: float) -> DefaultPredictor:
    os.environ["POLYGON_REPRESENTATION"] = "structured"
    cfg = get_cfg()
    cfg.MODEL.DEVICE = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
    )
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_threshold
    cfg.MODEL.ROI_KEYPOINT_HEAD.NAME = "PolygonVertexHead"
    cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 4
    cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION = 14
    cfg.MODEL.ROI_KEYPOINT_HEAD.CONV_DIMS = (256, 256, 256, 256)
    cfg.TEST.KEYPOINT_OKS_SIGMAS = [0.05] * 4
    cfg.MODEL.WEIGHTS = str(checkpoint)
    return DefaultPredictor(cfg)


def draw_polygon(image: np.ndarray, points: np.ndarray, color: tuple[int, int, int], label: str | None = None) -> None:
    polygon = np.rint(points).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(image, [polygon], isClosed=True, color=color, thickness=2)
    if label:
        x, y = polygon[0, 0]
        cv2.putText(
            image, label, (int(x), max(18, int(y) - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Salva imagens de validação com GT (verde) e predições (vermelho)."
    )
    parser.add_argument("--weights", required=True, type=Path, help="Checkpoint Polygon Structured.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--score-threshold", type=float, default=0.50)
    args = parser.parse_args()
    if not args.weights.is_file():
        parser.error(f"Checkpoint não encontrado: {args.weights}")

    register_blines()
    predictor = build_predictor(args.weights, args.score_threshold)
    dataset = DatasetCatalog.get("blines_val")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detections = 0

    for sample in dataset:
        image = cv2.imread(sample["file_name"])
        if image is None:
            raise ValueError(f"Não foi possível abrir {sample['file_name']}")
        for annotation in sample["annotations"]:
            gt = np.asarray(annotation["segmentation"][0], dtype=np.float32).reshape(-1, 2)
            draw_polygon(image, gt, (0, 200, 0), "GT")

        instances = predictor(image)["instances"].to("cpu")
        points = instances.pred_keypoints.numpy()
        scores = instances.scores.numpy()
        for vertices, score in zip(points, scores):
            draw_polygon(image, vertices[:, :2], (0, 0, 255), f"Pred {score:.2f}")
            detections += 1

        destination = args.output_dir / Path(sample["file_name"]).name
        if not cv2.imwrite(str(destination), image):
            raise OSError(f"Não foi possível salvar {destination}")

    print(f"{len(dataset)} imagens salvas em {args.output_dir}; {detections} predições desenhadas.")


if __name__ == "__main__":
    main()
