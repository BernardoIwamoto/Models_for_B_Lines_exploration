"""Image-cluster bootstrap confidence intervals and paired COCO AP differences."""

from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
import random
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


METRIC_NAMES = ("AP", "AP50", "AP75")


def _evaluate_sample(gt_data, predictions, draws, task):
    """Reindex sampled images so duplicate draws are independent bootstrap clusters."""
    sampled_gt = {"images": [], "annotations": [], "categories": copy.deepcopy(gt_data["categories"])}
    if "info" in gt_data:
        sampled_gt["info"] = copy.deepcopy(gt_data["info"])
    detections = []
    by_gt_image = {}
    by_pred_image = {}
    for ann in gt_data["annotations"]:
        by_gt_image.setdefault(ann["image_id"], []).append(ann)
    for pred in predictions:
        by_pred_image.setdefault(pred["image_id"], []).append(pred)
    image_by_id = {image["id"]: image for image in gt_data["images"]}
    next_annotation_id = 1
    for draw_id, original_id in enumerate(draws, start=1):
        image = copy.deepcopy(image_by_id[original_id])
        image["id"] = draw_id
        sampled_gt["images"].append(image)
        for ann in by_gt_image.get(original_id, []):
            duplicate = copy.deepcopy(ann)
            duplicate["id"] = next_annotation_id
            next_annotation_id += 1
            duplicate["image_id"] = draw_id
            sampled_gt["annotations"].append(duplicate)
        for pred in by_pred_image.get(original_id, []):
            duplicate = copy.deepcopy(pred)
            duplicate["image_id"] = draw_id
            detections.append(duplicate)
    coco_gt = COCO()
    coco_gt.dataset = sampled_gt
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt.createIndex()
        if detections:
            coco_dt = coco_gt.loadRes(detections)
        else:
            coco_dt = COCO()
            coco_dt.dataset = {"images": sampled_gt["images"], "categories": sampled_gt["categories"], "annotations": []}
            coco_dt.createIndex()
        evaluator = COCOeval(coco_gt, coco_dt, iouType=task)
        evaluator.params.imgIds = list(range(1, len(draws) + 1))
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return np.asarray(evaluator.stats[:3], dtype=float)


def _interval(values):
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if not len(finite):
        return {"n": 0, "mean": None, "median": None, "ci95": [None, None]}
    lo, hi = np.percentile(finite, [2.5, 97.5])
    return {"n": int(len(finite)), "mean": float(np.mean(finite)),
            "median": float(np.median(finite)), "ci95": [float(lo), float(hi)]}


def bootstrap(gt_path, named_predictions, output, *, task="segm", replicates=1000, seed=0):
    if task not in {"bbox", "segm"}:
        raise ValueError("task must be 'bbox' or 'segm'")
    gt_data = json.loads(Path(gt_path).read_text())
    ids = [image["id"] for image in gt_data["images"]]
    if not ids:
        raise ValueError("Ground-truth COCO file contains no images")
    prediction_data = {
        name: json.loads(Path(path).read_text()) for name, path in named_predictions.items()
    }
    rng = random.Random(seed)
    draws = [[rng.choice(ids) for _ in ids] for _ in range(replicates)]
    arrays = {name: [] for name in prediction_data}
    for draw in draws:
        for name, predictions in prediction_data.items():
            arrays[name].append(_evaluate_sample(gt_data, predictions, draw, task))
    output_data = {
        "task": task,
        "sampling_unit": "validation image, sampled with replacement; same draws for every method",
        "replicates": replicates,
        "seed": seed,
        "interval": "percentile 95% bootstrap CI; conditional on supplied fixed predictions/checkpoints",
        "methods": {},
        "paired_differences_vs_first_method": {},
    }
    names = list(prediction_data)
    for name in names:
        values = np.stack(arrays[name])
        output_data["methods"][name] = {
            metric: _interval(values[:, i]) for i, metric in enumerate(METRIC_NAMES)
        }
    if names:
        baseline = names[0]
        for name in names[1:]:
            difference = np.stack(arrays[name]) - np.stack(arrays[baseline])
            output_data["paired_differences_vs_first_method"][f"{name} - {baseline}"] = {
                metric: _interval(difference[:, i]) for i, metric in enumerate(METRIC_NAMES)
            }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(output_data, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(output_data, indent=2, allow_nan=False))
    return output_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", required=True)
    parser.add_argument("--prediction", action="append", required=True,
                        help="NAME=COCO_predictions.json; first method is the paired-comparison reference")
    parser.add_argument("--task", choices=("bbox", "segm"), default="segm")
    parser.add_argument("--output", required=True)
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    named = {}
    for item in args.prediction:
        name, sep, path = item.partition("=")
        if not sep or not name or name in named:
            parser.error("Each --prediction must be a unique NAME=PATH")
        named[name] = path
    bootstrap(args.gt, named, args.output, task=args.task,
              replicates=args.replicates, seed=args.seed)


if __name__ == "__main__":
    main()
