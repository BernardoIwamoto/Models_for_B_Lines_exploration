from pathlib import Path
import json

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.data import build_detection_test_loader
from detectron2 import model_zoo

from src.polygon_rcnn.register_dataset import register_blines

import torch


MODEL_PATH = "output_faster_rcnn/model_final.pth"

DATASET = "blines_val"

OUTPUT_DIR = Path("output_faster_rcnn/coco_eval")

TASK = "bbox"

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
        "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    )
)

cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

cfg.MODEL.WEIGHTS = MODEL_PATH

cfg.DATASETS.TEST = (DATASET,)

predictor = DefaultPredictor(cfg)

evaluator = COCOEvaluator(
    DATASET,
    output_dir=str(OUTPUT_DIR),
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

with open(OUTPUT_DIR / "results.json", "w") as f:
    json.dump(results, f, indent=4)

# Same rationale as mask_rcnn/evaluate_coco.py: rebuild COCOeval from the files
# COCOEvaluator already wrote to disk instead of reaching into its (now absent)
# internal state, keeping every model on the exact same evaluation code path.
coco_gt = COCO(str(OUTPUT_DIR / "blines_val_coco_format.json"))
coco_dt = coco_gt.loadRes(str(OUTPUT_DIR / "coco_instances_results.json"))

coco_eval = COCOeval(coco_gt, coco_dt, iouType=TASK)

coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

np.save(OUTPUT_DIR / "precision.npy", coco_eval.eval["precision"])
np.save(OUTPUT_DIR / "recall.npy", coco_eval.eval["recall"])
np.save(OUTPUT_DIR / "scores.npy", coco_eval.eval["scores"])
np.save(OUTPUT_DIR / "iou_thresholds.npy", coco_eval.params.iouThrs)
np.save(OUTPUT_DIR / "recall_thresholds.npy", coco_eval.params.recThrs)

print(results)
