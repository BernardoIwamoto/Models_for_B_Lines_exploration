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
from src.polygon_rcnn.polygon_vertex_head import PolygonVertexHead  # noqa: F401

import torch


MODEL_PATH = "output_polygon_head/model_best.pth"

DATASET = "blines_val"

OUTPUT_DIR = Path("output_polygon_head/coco_eval")

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


register_blines()

cfg = get_cfg()

cfg.MODEL.DEVICE = DEVICE

cfg.merge_from_file(
    model_zoo.get_config_file(
        "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"
    )
)

cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

# Must match train_polygon_head.py exactly, or the checkpoint's shapes won't line
# up with the model this config builds.
cfg.MODEL.ROI_KEYPOINT_HEAD.NAME = "PolygonVertexHead"

cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = NUM_KEYPOINTS

cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION = KEYPOINT_POOLER_RESOLUTION

cfg.MODEL.ROI_KEYPOINT_HEAD.CONV_DIMS = (256, 256, 256, 256)

cfg.TEST.KEYPOINT_OKS_SIGMAS = [0.05] * NUM_KEYPOINTS

cfg.MODEL.WEIGHTS = MODEL_PATH

cfg.DATASETS.TEST = (DATASET,)

predictor = DefaultPredictor(cfg)

evaluator = COCOEvaluator(
    DATASET,
    output_dir=str(OUTPUT_DIR),
    # Must be passed explicitly -- cfg.TEST.KEYPOINT_OKS_SIGMAS alone is only read
    # from a (deprecated) CfgNode passed as `tasks`, never picked up otherwise. Without
    # it this crashes evaluating its own auto-detected "keypoints" task (COCO's 17
    # default sigmas vs our 4 keypoints), before we even get to the polygon conversion
    # below.
    kpt_oks_sigmas=cfg.TEST.KEYPOINT_OKS_SIGMAS,
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

# Our dataset dicts carry both "segmentation" (raw polygon) and "keypoints" (same 4
# vertices, canonically ordered) per instance, so the ground-truth json COCOEvaluator
# just wrote already has "segmentation" -- only the *predictions* file needs
# converting, from Detectron2's native "keypoints" task format to a polygon
# "segmentation", so we can evaluate with the exact same segm-task COCOeval code path
# as Mask R-CNN/Faster R-CNN/YOLO (directly comparable AP numbers, no metric
# reimplementation).
with open(OUTPUT_DIR / "coco_instances_results.json") as f:
    predictions = json.load(f)

for pred in predictions:
    keypoints = np.array(pred.pop("keypoints")).reshape(-1, 3)
    pred["segmentation"] = [keypoints[:, :2].flatten().tolist()]

polygon_predictions_path = OUTPUT_DIR / "coco_instances_results_polygon.json"

with open(polygon_predictions_path, "w") as f:
    json.dump(predictions, f)

coco_gt = COCO(str(OUTPUT_DIR / "blines_val_coco_format.json"))
coco_dt = coco_gt.loadRes(str(polygon_predictions_path))

coco_eval = COCOeval(coco_gt, coco_dt, iouType="segm")

coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

np.save(OUTPUT_DIR / "precision.npy", coco_eval.eval["precision"])
np.save(OUTPUT_DIR / "recall.npy", coco_eval.eval["recall"])
np.save(OUTPUT_DIR / "scores.npy", coco_eval.eval["scores"])
np.save(OUTPUT_DIR / "iou_thresholds.npy", coco_eval.params.iouThrs)
np.save(OUTPUT_DIR / "recall_thresholds.npy", coco_eval.params.recThrs)

# "segm_polygon" to keep it clearly distinct from Detectron2's native "bbox"/
# "keypoints" entries already in `results`: this is a polygon rasterized through
# COCOeval's own polygon-IoU handling, not a genuine per-pixel mask AP, even though
# it's the number directly comparable to Mask R-CNN's own "segm" AP.
results["segm_polygon"] = {
    "AP": float(coco_eval.stats[0]) * 100,
    "AP50": float(coco_eval.stats[1]) * 100,
    "AP75": float(coco_eval.stats[2]) * 100,
    "APs": float(coco_eval.stats[3]) * 100,
    "APm": float(coco_eval.stats[4]) * 100,
    "APl": float(coco_eval.stats[5]) * 100,
}

with open(OUTPUT_DIR / "results.json", "w") as f:
    json.dump(results, f, indent=4)

print(results)
