import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from src.polygon_rcnn.evaluation.metrics.geometry import box_iou
from src.polygon_rcnn.evaluation.metrics.statistics import summarize
from src.polygon_rcnn.evaluation.metrics.polygon import orientation


MATCH_IOU_THRESHOLD = 0.5


def match_predictions_to_gt(gt_by_image, preds_by_image, match_iou_threshold=MATCH_IOU_THRESHOLD):
    """Greedy IoU matching, one GT per prediction, highest-score predictions first --
    a distinct box-matching geometry protocol (xywh). Do not conflate it with the
    mask-IoU matching used by evaluate_predictions.py.
    """

    matches = []

    for image_id, gts in gt_by_image.items():

        preds = sorted(preds_by_image.get(image_id, []), key=lambda p: -p["score"])

        matched_gt = set()

        for pred in preds:

            best_idx, best_iou = -1, 0.0

            for idx, gt in enumerate(gts):

                if idx in matched_gt:
                    continue

                current = box_iou(pred["bbox"], gt["bbox"])

                if current > best_iou:
                    best_iou, best_idx = current, idx

            if best_idx != -1 and best_iou >= match_iou_threshold:

                matched_gt.add(best_idx)

                matches.append((pred, gts[best_idx], best_iou))

    return matches


def analyze(gt_json, predictions_json, output_dir, score_threshold=0.5):
    """Quantifies predicted-vs-ground-truth box bias for one model: area/width/
    height ratio and center error, per matched pair, plus how area bias varies
    with object size and confidence. Formalizes the ad hoc check that first
    surfaced the ~12-16% box inflation during the Polygon Head experiments, as
    reusable, reproducible code instead of a one-off script.
    """

    output_dir = Path(output_dir)

    with open(gt_json) as f:
        gt = json.load(f)

    with open(predictions_json) as f:
        predictions = json.load(f)

    gt_by_image = {}
    for ann in gt["annotations"]:
        gt_by_image.setdefault(ann["image_id"], []).append(ann)

    preds_by_image = {}
    for pred in predictions:
        if pred["score"] >= score_threshold:
            preds_by_image.setdefault(pred["image_id"], []).append(pred)

    matches = match_predictions_to_gt(gt_by_image, preds_by_image)

    if not matches:
        raise RuntimeError(
            f"No matched pairs at score>={score_threshold} -- check the paths, or "
            "lower --score-threshold."
        )

    output_dir.mkdir(parents=True, exist_ok=False)

    rows = []

    for pred, gt_ann, matched_iou in matches:

        px, py, pw, ph = pred["bbox"]
        gx, gy, gw, gh = gt_ann["bbox"]

        pred_center = (px + pw / 2, py + ph / 2)
        gt_center = (gx + gw / 2, gy + gh / 2)

        center_error = float(np.hypot(pred_center[0] - gt_center[0], pred_center[1] - gt_center[1]))

        gt_area = gw * gh

        segmentation = gt_ann.get("segmentation", [])
        if isinstance(segmentation, list) and segmentation and segmentation[0]:
            angle = float(np.degrees(orientation(segmentation[0])) % 180.0)
        else:
            angle = float("nan")

        rows.append({
            "image_id": pred["image_id"],
            "score": pred["score"],
            "iou": matched_iou,
            "area_ratio": (pw * ph) / gt_area if gt_area > 0 else float("nan"),
            "width_ratio": pw / gw if gw > 0 else float("nan"),
            "height_ratio": ph / gh if gh > 0 else float("nan"),
            "center_error": center_error,
            "gt_area": gt_area,
            "orientation_deg": angle,
        })

    with open(output_dir / "bbox_bias.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        key: summarize([r[key] for r in rows])
        for key in ("iou", "area_ratio", "width_ratio", "height_ratio", "center_error")
    }
    summary["n_matched"] = len(rows)
    summary["n_gt"] = sum(len(v) for v in gt_by_image.values())
    summary["matched_fraction_of_gt"] = len(rows) / max(summary["n_gt"], 1)
    summary["conditional_on_score_threshold"] = score_threshold

    image_rows = []
    for image_id in sorted({r["image_id"] for r in rows}):
        image_matches = [r for r in rows if r["image_id"] == image_id]
        image_rows.append({
            "image_id": image_id,
            "n_matched": len(image_matches),
            "median_area_ratio": float(np.median([r["area_ratio"] for r in image_matches])),
            "median_width_ratio": float(np.median([r["width_ratio"] for r in image_matches])),
            "median_height_ratio": float(np.median([r["height_ratio"] for r in image_matches])),
            "median_center_error": float(np.median([r["center_error"] for r in image_matches])),
        })
    with open(output_dir / "bbox_bias_by_image.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(image_rows[0].keys()))
        writer.writeheader()
        writer.writerows(image_rows)

    orientation_groups = {}
    for lower in range(0, 180, 30):
        selected = [r["area_ratio"] for r in rows
                    if np.isfinite(r["orientation_deg"]) and lower <= r["orientation_deg"] < lower + 30]
        orientation_groups[f"{lower:03d}-{lower + 30:03d}_deg"] = (
            summarize(selected) if selected else {"n": 0}
        )
    summary["area_ratio_by_orientation"] = orientation_groups

    for field, edges, label in (
        ("gt_area", np.quantile([r["gt_area"] for r in rows], [0, .25, .5, .75, 1]), "gt_area"),
        ("score", np.quantile([r["score"] for r in rows], [0, .25, .5, .75, 1]), "score"),
    ):
        groups = {}
        for index in range(4):
            lo, hi = float(edges[index]), float(edges[index + 1])
            selected = [r["area_ratio"] for r in rows
                        if (lo <= r[field] <= hi if index == 3 else lo <= r[field] < hi)]
            groups[f"q{index + 1}_{lo:.4g}_{hi:.4g}"] = summarize(selected) if selected else {"n": 0}
        summary[f"area_ratio_by_{label}_quartile"] = groups

    with open(output_dir / "bbox_bias_summary.json", "w") as f:
        json.dump(summary, f, indent=4)

    for key, label in [
        ("area_ratio", "Predicted / GT box area"),
        ("width_ratio", "Predicted / GT box width"),
        ("height_ratio", "Predicted / GT box height"),
        ("center_error", "Center error (px)"),
    ]:
        values = [r[key] for r in rows]

        plt.figure(figsize=(6, 4))
        plt.hist(values, bins=20)
        if key.endswith("_ratio"):
            plt.axvline(1.0, color="red", linestyle="--", label="no bias (ratio=1)")
            plt.legend()
        plt.xlabel(label)
        plt.ylabel("Count")
        plt.title(label)
        plt.tight_layout()
        plt.savefig(output_dir / f"hist_{key}.png")
        plt.close()

    # Split the area bias by GT size and by confidence, to tell apart "bigger
    # objects are more inflated" from "it's just the low-confidence noisy tail" --
    # both are plausible a priori, only the data can say which (if either) holds.
    gt_areas = np.array([r["gt_area"] for r in rows])
    area_ratios = np.array([r["area_ratio"] for r in rows])
    scores = np.array([r["score"] for r in rows])

    plt.figure(figsize=(6, 5))
    plt.scatter(gt_areas, area_ratios, alpha=0.6, s=15)
    plt.axhline(1.0, color="red", linestyle="--")
    plt.xlabel("Ground-truth box area (px^2)")
    plt.ylabel("Predicted / GT area ratio")
    plt.title("Area bias vs. object size")
    plt.tight_layout()
    plt.savefig(output_dir / "area_ratio_vs_gt_size.png")
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.scatter(scores, area_ratios, alpha=0.6, s=15)
    plt.axhline(1.0, color="red", linestyle="--")
    plt.xlabel("Detection confidence")
    plt.ylabel("Predicted / GT area ratio")
    plt.title("Area bias vs. confidence")
    plt.tight_layout()
    plt.savefig(output_dir / "area_ratio_vs_score.png")
    plt.close()

    angle = np.array([r["orientation_deg"] for r in rows])
    plt.figure(figsize=(6, 5))
    valid = np.isfinite(angle)
    plt.scatter(angle[valid], area_ratios[valid], alpha=0.6, s=15)
    plt.axhline(1.0, color="red", linestyle="--")
    plt.xlabel("Ground-truth long-axis orientation (degrees, [0, 180))")
    plt.ylabel("Predicted / GT box area ratio")
    plt.title("Area bias vs. B-line orientation")
    plt.tight_layout()
    plt.savefig(output_dir / "area_ratio_vs_orientation.png")
    plt.close()

    print(json.dumps(summary, indent=4))

    return summary


if __name__ == "__main__":
    # python -m src.polygon_rcnn.evaluation.common.bbox_bias_analysis \
    #     <gt_json> <predictions_json> <output_dir> [score_threshold]
    gt_json = sys.argv[1]
    predictions_json = sys.argv[2]
    output_dir = sys.argv[3]
    score_threshold = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5

    analyze(gt_json, predictions_json, output_dir, score_threshold)
