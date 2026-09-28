"""One-to-one detection matching followed by per-instance geometry analysis.

Matching is greedy, descending confidence, bbox IoU >= 0.50. This same rule is
used for all models, independent of whether their output is a box, mask, or polygon.
Geometry scores are conditional on this detection match and are not detection AP.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import cv2

from src.polygon_rcnn.evaluation.metrics.geometry import box_iou, polygon_to_mask
from src.polygon_rcnn.evaluation.metrics.polygon import (
    angular_difference,
    is_self_intersecting,
    orientation,
    polygon_area,
    vertex_error,
)
from src.polygon_rcnn.evaluation.metrics.statistics import summarize


def _polygon_from_segmentation(segmentation):
    if isinstance(segmentation, list) and segmentation and isinstance(segmentation[0], list):
        arr = np.asarray(segmentation[0], dtype=float)
        if arr.size >= 6 and arr.size % 2 == 0:
            return arr.reshape(-1, 2)
    return None


def _segmentation_mask(segmentation, shape):
    polygon = _polygon_from_segmentation(segmentation)
    if polygon is not None:
        return polygon_to_mask(polygon.flatten(), shape)
    if isinstance(segmentation, dict):
        from pycocotools import mask as mask_util
        rle = dict(segmentation)
        if isinstance(rle.get("counts"), str):
            rle["counts"] = rle["counts"].encode("ascii")
        return mask_util.decode(rle).astype(bool)
    return np.zeros(shape[:2], dtype=bool)


def _mask_orientation(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) < 3:
        return float("nan")
    points = np.column_stack([xs, ys]).astype(float)
    centered = points - points.mean(axis=0)
    vals, vecs = np.linalg.eigh(centered.T @ centered)
    v = vecs[:, int(np.argmax(vals))]
    return float(np.arctan2(v[1], v[0]) % np.pi)


def _axis_extents(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) < 3:
        return float("nan"), float("nan")
    points = np.column_stack([xs, ys]).astype(float)
    centered = points - points.mean(axis=0)
    _, vectors = np.linalg.eigh(centered.T @ centered)
    axis = vectors[:, -1]
    normal = np.asarray([-axis[1], axis[0]])
    long_extent = np.ptp(centered @ axis)
    short_extent = np.ptp(centered @ normal)
    return float(long_extent), float(short_extent)


def _prediction_shape(pred, image_shape):
    polygon = _polygon_from_segmentation(pred.get("segmentation"))
    if polygon is not None:
        return polygon_to_mask(polygon.flatten(), image_shape), polygon
    if "segmentation" in pred:
        return _segmentation_mask(pred["segmentation"], image_shape), None
    # Give bbox-only detectors their actual rectangular support so their shape
    # overlap can be compared consistently with polygon and mask outputs.
    x, y, width, height = map(float, pred["bbox"])
    rectangle = np.asarray([
        [x, y], [x + width, y], [x + width, y + height], [x, y + height]
    ])
    return polygon_to_mask(rectangle.flatten(), image_shape), rectangle


def _draw_polygon(image, polygon, color):
    if polygon is not None and len(polygon) >= 3:
        points = np.round(polygon).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(image, [points], isClosed=True, color=color, thickness=2)


def _save_example(image_path, output_path, gt_ann=None, pred=None):
    image = cv2.imread(str(image_path))
    if image is None:
        return False
    if gt_ann is not None:
        gt_polygon = _polygon_from_segmentation(gt_ann.get("segmentation"))
        if gt_polygon is not None:
            _draw_polygon(image, gt_polygon, (0, 220, 0))
        else:
            x, y, w, h = map(int, gt_ann["bbox"])
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 220, 0), 2)
    if pred is not None:
        pred_polygon = _polygon_from_segmentation(pred.get("segmentation"))
        if pred_polygon is not None:
            _draw_polygon(image, pred_polygon, (0, 0, 255))
        else:
            x, y, w, h = map(int, pred["bbox"])
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(output_path), image))


def analyze(gt_json, predictions_json, output_dir, *, score_threshold=0.001, match_iou=0.5):
    gt = json.loads(Path(gt_json).read_text())
    preds = json.loads(Path(predictions_json).read_text())
    images = {im["id"]: im for im in gt["images"]}
    gts_by_image = defaultdict(list)
    preds_by_image = defaultdict(list)
    for ann in gt["annotations"]:
        gts_by_image[ann["image_id"]].append(ann)
    for pred in preds:
        if float(pred.get("score", 0)) >= score_threshold:
            preds_by_image[pred["image_id"]].append(pred)

    rows = []
    counts = Counter()
    examples = defaultdict(list)
    polygon_invalid_predictions = 0
    polygon_prediction_count = 0
    image_shapes = {iid: (im["height"], im["width"]) for iid, im in images.items()}
    for image_id in sorted(images):
        gts = gts_by_image[image_id]
        image_preds = sorted(preds_by_image[image_id], key=lambda p: -float(p["score"]))
        used_gt = set()
        matched_pred_indices = set()
        for pi, pred in enumerate(image_preds):
            pred_poly_all = _polygon_from_segmentation(pred.get("segmentation"))
            if pred_poly_all is not None:
                polygon_prediction_count += 1
                if (pred_poly_all.shape != (4, 2)
                        or is_self_intersecting(pred_poly_all)
                        or not np.isfinite(pred_poly_all).all()
                        or polygon_area(pred_poly_all) <= 1e-6):
                    polygon_invalid_predictions += 1
            choices = [
                (idx, box_iou(pred["bbox"], ann["bbox"]))
                for idx, ann in enumerate(gts) if idx not in used_gt
            ]
            best_idx, best_iou = max(choices, key=lambda item: item[1], default=(-1, 0.0))
            if best_idx < 0 or best_iou < match_iou:
                continue
            used_gt.add(best_idx)
            matched_pred_indices.add(pi)
            gt_ann = gts[best_idx]
            shape = image_shapes[image_id]
            gt_mask = _segmentation_mask(gt_ann.get("segmentation", []), shape)
            pred_mask, pred_poly = _prediction_shape(pred, shape)
            inter = np.logical_and(gt_mask, pred_mask).sum()
            union = np.logical_or(gt_mask, pred_mask).sum()
            poly_iou = float(inter / union) if union else 0.0
            gt_poly = _polygon_from_segmentation(gt_ann.get("segmentation", []))
            gt_area = int(gt_mask.sum())
            pred_area = int(pred_mask.sum())
            rel_area_error = abs(pred_area - gt_area) / max(gt_area, 1)
            gt_angle = orientation(gt_poly) if gt_poly is not None and len(gt_poly) == 4 else _mask_orientation(gt_mask)
            pred_angle = (
                orientation(pred_poly) if pred_poly is not None and len(pred_poly) == 4
                else _mask_orientation(pred_mask)
            )
            orient_error = (
                float(np.degrees(angular_difference(gt_angle, pred_angle)))
                if np.isfinite(gt_angle) and np.isfinite(pred_angle) else float("nan")
            )
            vertex_dist = (
                vertex_error(pred_poly, gt_poly)
                if pred_poly is not None and gt_poly is not None
                and pred_poly.shape == (4, 2) and gt_poly.shape == (4, 2)
                else float("nan")
            )
            invalid = (
                bool(pred_poly.shape != (4, 2)
                     or is_self_intersecting(pred_poly)
                     or not np.isfinite(pred_poly).all()
                     or polygon_area(pred_poly) <= 1e-6)
                if pred_poly is not None else False
            )
            gt_length, gt_width = _axis_extents(gt_mask)
            pred_length, pred_width = _axis_extents(pred_mask)
            length_error = abs(pred_length - gt_length) / max(gt_length, 1e-6)
            width_error = abs(pred_width - gt_width) / max(gt_width, 1e-6)
            geometry_correct = poly_iou >= 0.5
            label = "correct_detection_correct_geometry" if geometry_correct else "correct_detection_wrong_geometry"
            counts[label] += 1
            if len(examples[label]) < 3:
                examples[label].append((images[image_id], gt_ann, pred))
            if not geometry_correct and rel_area_error > 0.5:
                counts["wrong_extent_flag"] += 1
            if np.isfinite(orient_error) and orient_error > 15.0:
                counts["wrong_orientation_flag"] += 1
            if invalid:
                counts["invalid_polygon"] += 1
            rows.append({
                "image_id": image_id,
                "image_file": images[image_id].get("file_name", ""),
                "score": float(pred["score"]),
                "bbox_iou": float(best_iou),
                "polygon_or_mask_iou": poly_iou,
                "area_error": rel_area_error,
                "orientation_error_deg": orient_error,
                "length_error": length_error,
                "width_error": width_error,
                "vertex_error_px": vertex_dist,
                "invalid_polygon": invalid,
                "error_category": label,
            })
        for idx, ann in enumerate(gts):
            if idx not in used_gt:
                counts["missed_bline"] += 1
                if len(examples["missed_bline"]) < 3:
                    examples["missed_bline"].append((images[image_id], ann, None))
                rows.append({
                    "image_id": image_id,
                    "image_file": images[image_id].get("file_name", ""),
                    "score": "",
                    "bbox_iou": "",
                    "polygon_or_mask_iou": "",
                    "area_error": "",
                    "orientation_error_deg": "",
                    "length_error": "",
                    "width_error": "",
                    "vertex_error_px": "",
                    "invalid_polygon": "",
                    "error_category": "missed_bline",
                })
        for pi, pred in enumerate(image_preds):
            if pi not in matched_pred_indices:
                counts["false_positive"] += 1
                if len(examples["false_positive"]) < 3:
                    examples["false_positive"].append((images[image_id], None, pred))
                rows.append({
                    "image_id": image_id,
                    "image_file": images[image_id].get("file_name", ""),
                    "score": float(pred["score"]),
                    "bbox_iou": "",
                    "polygon_or_mask_iou": "",
                    "area_error": "",
                    "orientation_error_deg": "",
                    "length_error": "",
                    "width_error": "",
                    "vertex_error_px": "",
                    "invalid_polygon": "",
                    "error_category": "false_positive",
                })

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    for category, items in examples.items():
        for index, (image, gt_ann, pred) in enumerate(items, start=1):
            source = Path(image.get("file_name", ""))
            if not source.is_absolute():
                source = Path.cwd() / source
            _save_example(source, out / "examples" / category / f"{index:02d}_{source.name}",
                          gt_ann=gt_ann, pred=pred)
    with (out / "per_instance.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    matched = [r for r in rows if r["error_category"].startswith("correct_detection_")]
    tp = counts["correct_detection_correct_geometry"] + counts["correct_detection_wrong_geometry"]
    fp, fn = counts["false_positive"], counts["missed_bline"]
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    def stats_for(key):
        values = [float(r[key]) for r in matched if np.isfinite(float(r[key]))]
        return summarize(values) if values else {"n": 0}

    summary = {
        "matching": {"rule": "greedy score-descending bbox IoU", "threshold": match_iou,
                     "score_threshold": score_threshold},
        "detection": {"TP": tp, "FP": fp, "FN": fn, "precision": precision,
                      "recall": recall, "F1": 2 * precision * recall / max(precision + recall, 1e-12)},
        "geometry_on_matched_detections": {
            key: stats_for(key)
            for key in ("bbox_iou", "polygon_or_mask_iou", "area_error",
                        "orientation_error_deg", "length_error", "width_error",
                        "vertex_error_px")
        },
        "invalid_polygon_rate": {
            "count": polygon_invalid_predictions,
            "total_polygon_predictions": polygon_prediction_count,
            "rate": (polygon_invalid_predictions / polygon_prediction_count
                     if polygon_prediction_count else None),
            "definition": "self-intersection, non-finite coordinate, or near-zero shoelace area",
        },
        "error_counts": dict(counts),
        "n_images": len(images),
        "n_ground_truth": sum(map(len, gts_by_image.values())),
        "n_predictions_above_threshold": sum(map(len, preds_by_image.values())),
        "note": "Geometry is conditional on bbox matching; mask outputs use their binary masks and bbox-only outputs use their rectangular extent.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2, allow_nan=False))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--score-threshold", type=float, default=0.001)
    parser.add_argument("--match-iou", type=float, default=0.5)
    args = parser.parse_args()
    analyze(args.gt, args.predictions, args.output,
            score_threshold=args.score_threshold, match_iou=args.match_iou)


if __name__ == "__main__":
    main()
