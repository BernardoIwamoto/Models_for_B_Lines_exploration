"""Post-hoc confidence-threshold sweep for saved polygon-head COCO predictions.

This does not load model weights or run inference. It reuses saved validation
predictions and COCOeval's polygon matching to report detection precision,
recall, F1, and image-presence metrics at fixed confidence cutoffs.
"""

from __future__ import annotations

import argparse
import csv
import json
from decimal import Decimal
from pathlib import Path

import matplotlib
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.polygon_rcnn.evaluation.common.metrics_report import (
    _counts_at_threshold,
    _evaluation_images,
)


METRICS = (
    "precision",
    "recall",
    "F1",
    "image_presence_accuracy",
    "image_presence_precision",
    "image_presence_recall",
    "image_presence_F1",
)


def make_thresholds(start: str, stop: str, step: str) -> list[float]:
    start_d, stop_d, step_d = Decimal(start), Decimal(stop), Decimal(step)
    if step_d <= 0 or stop_d < start_d:
        raise ValueError("Threshold step must be positive and stop >= start.")
    values = []
    current = start_d
    while current <= stop_d:
        values.append(float(current))
        current += step_d
    if not values or Decimal(str(values[-1])) != stop_d:
        raise ValueError("The requested range must land exactly on the stop threshold.")
    return values


def evaluate_run(eval_dir: Path, seed: int, thresholds: list[float]) -> list[dict]:
    gt_path = eval_dir / "blines_val_coco_format.json"
    predictions_path = eval_dir / "coco_instances_results_polygon.json"
    if not gt_path.is_file() or not predictions_path.is_file():
        missing = [str(path) for path in (gt_path, predictions_path) if not path.is_file()]
        raise FileNotFoundError("Missing saved evaluation inputs: " + ", ".join(missing))

    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    coco_gt = COCO(str(gt_path))
    if not predictions:
        raise ValueError(f"No saved polygon predictions in {predictions_path}")
    coco_dt = coco_gt.loadRes(predictions)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="segm")
    coco_eval.evaluate()
    coco_eval.accumulate()
    image_evals = _evaluation_images(coco_eval, 0.5)

    rows = []
    for threshold in thresholds:
        counts = _counts_at_threshold(image_evals, threshold)
        presence = counts["image_presence"]
        rows.append({
            "seed": seed,
            "confidence_threshold": threshold,
            "predictions_above_threshold": sum(
                float(pred["score"]) >= threshold for pred in predictions
            ),
            "TP": counts["TP"],
            "FP": counts["FP"],
            "FN": counts["FN"],
            "precision": counts["precision"],
            "recall": counts["recall"],
            "F1": counts["F1"],
            "image_presence_accuracy": presence["accuracy"],
            "image_presence_precision": presence["precision"],
            "image_presence_recall": presence["recall"],
            "image_presence_F1": presence["F1"],
        })
    return rows


def summarize(rows: list[dict], thresholds: list[float]) -> list[dict]:
    summary = []
    baseline = {
        row["seed"]: row
        for row in rows
        if row["confidence_threshold"] == thresholds[0]
    }
    for threshold in thresholds:
        selected = [row for row in rows if row["confidence_threshold"] == threshold]
        result = {"confidence_threshold": threshold, "n_seeds": len(selected)}
        for metric in METRICS:
            values = np.asarray([row[metric] for row in selected], dtype=float)
            result[f"{metric}_mean"] = float(values.mean())
            result[f"{metric}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            deltas = np.asarray(
                [row[metric] - baseline[row["seed"]][metric] for row in selected],
                dtype=float,
            )
            result[f"{metric}_delta_vs_{thresholds[0]:.2f}_mean"] = float(deltas.mean())
        result["predictions_above_threshold_mean"] = float(
            np.mean([row["predictions_above_threshold"] for row in selected])
        )
        summary.append(result)
    return summary


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("No threshold results to write.")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(path: Path, rows: list[dict]) -> None:
    thresholds = [row["confidence_threshold"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for metric, label in (("precision", "Precisão"), ("recall", "Recall"), ("F1", "F1")):
        means = np.asarray([row[f"{metric}_mean"] for row in rows])
        sds = np.asarray([row[f"{metric}_sd"] for row in rows])
        axes[0].plot(thresholds, means, marker="o", label=label)
        axes[0].fill_between(thresholds, means - sds, means + sds, alpha=0.12)
    axes[0].set_title("Métricas por detecção (IoU=0,50)")
    axes[0].set_xlabel("Limiar de confiança")
    axes[0].set_ylabel("Métrica")
    axes[0].set_ylim(0, 1.02)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    for metric, label in (
        ("image_presence_precision", "Precisão"),
        ("image_presence_recall", "Recall"),
        ("image_presence_F1", "F1"),
    ):
        means = np.asarray([row[f"{metric}_mean"] for row in rows])
        axes[1].plot(thresholds, means, marker="o", label=label)
    axes[1].set_title("Presença de B-lines por imagem")
    axes[1].set_xlabel("Limiar de confiança")
    axes[1].set_ylabel("Métrica média entre seeds")
    axes[1].set_ylim(0, 1.02)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", required=True, type=Path,
                        help="Directory containing <run-prefix>_sN/eval/")
    parser.add_argument("--run-prefix", default="polygon_struct_iou_area")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--start", default="0.50")
    parser.add_argument("--stop", default="0.85")
    parser.add_argument("--step", default="0.05")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    runs_dir = args.runs_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    thresholds = make_thresholds(args.start, args.stop, args.step)
    input_dirs = [
        (seed, runs_dir / f"{args.run_prefix}_s{seed}" / "eval")
        for seed in args.seeds
    ]
    # Check all inputs before producing output, so a missing seed cannot leave a
    # misleading partial sweep behind.
    for seed, eval_dir in input_dirs:
        for name in ("blines_val_coco_format.json", "coco_instances_results_polygon.json"):
            if not (eval_dir / name).is_file():
                raise FileNotFoundError(f"Seed {seed} is missing {eval_dir / name}")

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed, eval_dir in input_dirs:
        print(f"Evaluating saved predictions for seed {seed}: {eval_dir}")
        rows.extend(evaluate_run(eval_dir, seed, thresholds))

    summary = summarize(rows, thresholds)
    write_csv(output_dir / "threshold_sweep_per_seed.csv", rows)
    write_csv(output_dir / "threshold_sweep_summary.csv", summary)
    (output_dir / "threshold_sweep.json").write_text(
        json.dumps({
            "evaluation": "post-hoc on saved validation predictions; no model inference",
            "matching": "COCO polygon segmentation IoU=0.50",
            "thresholds": thresholds,
            "per_seed": rows,
            "summary": summary,
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    plot_summary(output_dir / "threshold_sweep.png", summary)
    print(f"Threshold sweep complete: {output_dir}")


if __name__ == "__main__":
    main()
