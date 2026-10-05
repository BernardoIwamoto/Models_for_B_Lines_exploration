from pathlib import Path
import json
import os

import numpy as np
from ultralytics import YOLO
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from detectron2.data import DatasetCatalog

from src.polygon_rcnn.register_dataset import register_blines
from src.polygon_rcnn.evaluation.common.metrics_report import save_coco_metrics_report


MODEL_PATH = os.environ.get("MODEL_PATH", "runs/detect/output_yolo_detect/train/weights/best.pt")

GT_JSON = os.environ.get("GT_JSON", "output_faster_rcnn/coco_eval/blines_val_coco_format.json")

OUTPUT_DIR = Path(os.environ.get("EVAL_OUTPUT_DIR", "output_yolo_detect/coco_eval"))

CONF_THRESHOLD = 0.001


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)

    register_blines()

    dataset_dicts = DatasetCatalog.get("blines_val")

    model = YOLO(MODEL_PATH)

    predictions = []

    for record in dataset_dicts:

        result = model.predict(
            record["file_name"],
            conf=CONF_THRESHOLD,
            verbose=False,
        )[0]

        boxes = result.boxes.xywh.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()

        for (cx, cy, w, h), score, cls in zip(boxes, scores, classes):

            predictions.append({
                "image_id": record["image_id"],
                "category_id": int(cls),
                "bbox": [
                    float(cx - w / 2),
                    float(cy - h / 2),
                    float(w),
                    float(h),
                ],
                "score": float(score),
            })

    predictions_path = OUTPUT_DIR / "coco_instances_results.json"

    with open(predictions_path, "w") as f:
        json.dump(predictions, f)

    coco_gt = COCO(GT_JSON)

    coco_dt = coco_gt.loadRes(str(predictions_path))

    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    results = {
        "bbox": {
            "AP": float(coco_eval.stats[0]) * 100,
            "AP50": float(coco_eval.stats[1]) * 100,
            "AP75": float(coco_eval.stats[2]) * 100,
            "APs": float(coco_eval.stats[3]) * 100,
            "APm": float(coco_eval.stats[4]) * 100,
            "APl": float(coco_eval.stats[5]) * 100,
        }
    }

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=4)

    save_coco_metrics_report(coco_eval, OUTPUT_DIR, results)

    print(results)


if __name__ == "__main__":
    main()
