"""Write comparable validation metrics and plots from a completed COCOeval pass."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.polygon_rcnn.evaluation.common.coco_threshold_plots import (
    plot_coco_threshold_analysis,
)


def _evaluation_images(coco_eval, iou_threshold):
    iou_index = int(np.argmin(np.abs(coco_eval.params.iouThrs - iou_threshold)))
    area_range = list(coco_eval.params.areaRng[0])
    max_dets = int(coco_eval.params.maxDets[-1])
    images = []
    for item in coco_eval.evalImgs:
        if item is None:
            continue
        if list(item["aRng"]) != area_range or int(item["maxDet"]) != max_dets:
            continue
        scores = np.asarray(item["dtScores"], dtype=float)
        matches = np.asarray(item["dtMatches"], dtype=float)[iou_index]
        ignored = np.asarray(item["dtIgnore"], dtype=bool)[iou_index]
        gt_ignored = np.asarray(item["gtIgnore"], dtype=bool)
        images.append({
            "image_id": int(item["image_id"]),
            "scores": scores,
            "matches": matches,
            "ignored": ignored,
            "gt_count": int((~gt_ignored).sum()),
        })
    return images


def _counts_at_threshold(image_evals, threshold):
    tp = fp = fn = 0
    image_presence = {}
    thresholds = float(threshold)

    for item in image_evals:
        active = (item["scores"] >= thresholds) & ~item["ignored"]
        matched = item["matches"] > 0
        image_tp = int((active & matched).sum())
        image_fp = int((active & ~matched).sum())
        image_fn = item["gt_count"] - image_tp
        tp += image_tp
        fp += image_fp
        fn += image_fn
        old = image_presence.get(item["image_id"])
        if old is None:
            image_presence[item["image_id"]] = {
                "gt_positive": item["gt_count"] > 0,
                "prediction_positive": bool(active.any()),
            }
        else:
            old["gt_positive"] = old["gt_positive"] or item["gt_count"] > 0
            old["prediction_positive"] = old["prediction_positive"] or bool(active.any())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    presence = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    for item in image_presence.values():
        gt_positive = item["gt_positive"]
        pred_positive = item["prediction_positive"]
        if gt_positive and pred_positive:
            presence["TP"] += 1
        elif not gt_positive and not pred_positive:
            presence["TN"] += 1
        elif not gt_positive and pred_positive:
            presence["FP"] += 1
        else:
            presence["FN"] += 1

    image_precision = presence["TP"] / max(presence["TP"] + presence["FP"], 1)
    image_recall = presence["TP"] / max(presence["TP"] + presence["FN"], 1)
    image_f1 = (
        2 * image_precision * image_recall / (image_precision + image_recall)
        if image_precision + image_recall else 0.0
    )
    image_count = len(image_presence)
    presence.update({
        "n_images": image_count,
        "accuracy": (presence["TP"] + presence["TN"]) / image_count if image_count else 0.0,
        "precision": image_precision,
        "recall": image_recall,
        "F1": image_f1,
    })

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "F1": f1,
        "image_presence": presence,
    }


def _best_f1_sweep(coco_eval, iou_threshold=0.5):
    image_evals = _evaluation_images(coco_eval, iou_threshold)
    scores = [
        item["scores"][~item["ignored"]]
        for item in image_evals
        if item["scores"].size
    ]
    candidates = np.unique(np.concatenate(scores)) if scores else np.asarray([1.0])
    best = None
    curve = []
    for threshold in candidates:
        counts = _counts_at_threshold(image_evals, float(threshold))
        row = {
            "score_threshold": float(threshold),
            "TP": counts["TP"],
            "FP": counts["FP"],
            "FN": counts["FN"],
            "precision": counts["precision"],
            "recall": counts["recall"],
            "F1": counts["F1"],
        }
        curve.append(row)
        key = (row["F1"], row["precision"], row["score_threshold"])
        if best is None or key > best[0]:
            best = (key, float(threshold), counts)

    _, threshold, counts = best
    return threshold, counts, curve


def _plot_exact_curves(curve, output_dir, task, iou_threshold):
    output_dir = Path(output_dir)
    rows = sorted(curve, key=lambda row: row["score_threshold"])
    scores = [row["score_threshold"] for row in rows]
    precision = [row["precision"] for row in rows]
    recall = [row["recall"] for row in rows]
    f1 = [row["F1"] for row in rows]

    plt.figure(figsize=(7, 5))
    plt.plot(scores, precision, label="Precision")
    plt.plot(scores, recall, label="Recall")
    plt.xlabel("Confidence threshold")
    plt.ylabel("Metric")
    plt.ylim(0, 1.05)
    plt.title(f"{task} precision/recall at IoU={iou_threshold:.2f}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "detection_precision_recall_vs_confidence.png")
    plt.close()

    best_index = int(np.argmax(f1)) if f1 else 0
    plt.figure(figsize=(7, 5))
    plt.plot(scores, f1, color="tab:green")
    if scores:
        plt.axvline(scores[best_index], linestyle="--", color="gray")
        plt.scatter([scores[best_index]], [f1[best_index]], color="red", zorder=5)
    plt.xlabel("Confidence threshold")
    plt.ylabel("F1")
    plt.ylim(0, 1.05)
    plt.title(f"Detection F1 at IoU={iou_threshold:.2f}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "detection_f1_vs_confidence.png")
    plt.close()


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    return value


def save_coco_metrics_report(coco_eval, output_dir, coco_results=None, iou_threshold=0.5):
    """Save COCO curves, exact operating-point metrics, and image-presence accuracy.

    The confidence threshold is selected to maximize micro detection F1 on the
    validation split at the requested IoU. Accuracy is separately defined at image
    level (B-line present/absent), where true negatives have a meaningful definition.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "precision.npy", coco_eval.eval["precision"])
    np.save(output_dir / "recall.npy", coco_eval.eval["recall"])
    np.save(output_dir / "scores.npy", coco_eval.eval["scores"])
    np.save(output_dir / "iou_thresholds.npy", coco_eval.params.iouThrs)
    np.save(output_dir / "recall_thresholds.npy", coco_eval.params.recThrs)

    curve_analysis = plot_coco_threshold_analysis(output_dir, output_dir, prefix="coco")
    threshold, counts, sweep = _best_f1_sweep(coco_eval, iou_threshold)
    _plot_exact_curves(sweep, output_dir, coco_eval.params.iouType, iou_threshold)

    report = {
        "task": str(coco_eval.params.iouType),
        "selection_split": "validation",
        "operating_point": {
            "match_iou_threshold": float(iou_threshold),
            "score_threshold_selected_by_max_validation_F1": threshold,
            "detection": {key: counts[key] for key in ("TP", "FP", "FN", "precision", "recall", "F1")},
            "image_presence": counts["image_presence"],
        },
        "coco_curve_analysis": curve_analysis,
        "COCO": coco_results or {},
    }
    report_path = output_dir / "metrics_summary.json"
    report_path.write_text(
        json.dumps(_json_safe(report), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report
