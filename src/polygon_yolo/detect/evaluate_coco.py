from pathlib import Path
import json

import numpy as np
from ultralytics import YOLO
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from detectron2.data import DatasetCatalog

from src.polygon_rcnn.register_dataset import register_blines


MODEL_PATH = "runs/detect/output_yolo_detect/train/weights/best.pt"

# Ground-truth COCO json produced by evaluate_coco.py in the Faster R-CNN pipeline,
# built from the same blines_val dataset dicts (same image order => same image_id,
# same category_id=0 for "bline"). Run that script first.
GT_JSON = "output_faster_rcnn/coco_eval/blines_val_coco_format.json"

OUTPUT_DIR = Path("output_yolo_detect/coco_eval")

# Low threshold so pycocotools sees the full score spectrum needed to build the
# precision-recall curve, exactly like Detectron2's own evaluation does.
CONF_THRESHOLD = 0.001


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

    np.save(OUTPUT_DIR / "precision.npy", coco_eval.eval["precision"])
    np.save(OUTPUT_DIR / "recall.npy", coco_eval.eval["recall"])
    np.save(OUTPUT_DIR / "scores.npy", coco_eval.eval["scores"])
    np.save(OUTPUT_DIR / "iou_thresholds.npy", coco_eval.params.iouThrs)
    np.save(OUTPUT_DIR / "recall_thresholds.npy", coco_eval.params.recThrs)

    print(results)


if __name__ == "__main__":
    main()
