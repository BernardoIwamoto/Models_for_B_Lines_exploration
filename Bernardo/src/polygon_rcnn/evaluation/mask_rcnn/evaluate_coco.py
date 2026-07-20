from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.data import build_detection_test_loader
from detectron2 import model_zoo

from src.polygon_rcnn.register_dataset import register_blines

import torch


MODEL_PATH = "output_maskrcnn/model_final.pth"

DATASET = "blines_val"

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


register_blines()

cfg = get_cfg()

cfg.MODEL.DEVICE = DEVICE

cfg.merge_from_file(
    model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )
)

cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

cfg.MODEL.WEIGHTS = MODEL_PATH

cfg.DATASETS.TEST = (DATASET,)

predictor = DefaultPredictor(cfg)

evaluator = COCOEvaluator(
    DATASET,
    output_dir="output_maskrcnn/coco_eval"
)

loader = build_detection_test_loader(
    cfg,
    DATASET,
)

results = DefaultTrainer.test(
    cfg,
    predictor.model,
    evaluators=[evaluator],
)

print(type(evaluator))
print(dir(evaluator))

import json
import numpy as np

with open(
    "output_maskrcnn/coco_eval/results.json",
    "w",
) as f:
    json.dump(results, f, indent=4)

coco_eval = evaluator._coco_eval["segm"]

np.save(
    "output_maskrcnn/coco_eval/precision.npy",
    coco_eval.eval["precision"],
)

np.save(
    "output_maskrcnn/coco_eval/recall.npy",
    coco_eval.eval["recall"],
)

np.save(
    "output_maskrcnn/coco_eval/scores.npy",
    coco_eval.eval["scores"],
)

print(results)

print(results)