"""Aggregate every completed per-run validation metric across seeds."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


SUITES = {
    "baselines": {
        "mask_bbox": "mask_bbox_s{}",
        "faster_bbox": "faster_bbox_s{}",
        "polygon_legacy": "polygon_legacy_s{}",
        "yolo11n_bbox": "yolo11n_bbox_s{}",
    },
    "polygon_ablations": {
        "polygon_center_vertex": "polygon_center_s{}",
        "polygon_struct_vertex": "polygon_struct_s{}",
        "polygon_struct_iou": "polygon_struct_iou_s{}",
        "polygon_struct_iou_area": "polygon_struct_iou_area_s{}",
    },
}


def _flatten_metrics(report):
    values = {}
    for task, metrics in report.get("COCO", {}).items():
        if isinstance(metrics, dict):
            for metric in ("AP", "AP50", "AP75"):
                value = metrics.get(metric)
                if isinstance(value, (int, float)) and np.isfinite(value):
                    values[f"COCO/{task}/{metric}"] = float(value)

    operating = report.get("operating_point", {})
    detection = operating.get("detection", {})
    for metric in ("precision", "recall", "F1"):
        value = detection.get(metric)
        if isinstance(value, (int, float)) and np.isfinite(value):
            values[f"detection/{metric}"] = float(value)

    value = operating.get("score_threshold_selected_by_max_validation_F1")
    if isinstance(value, (int, float)) and np.isfinite(value):
        values["detection/score_threshold"] = float(value)

    presence = operating.get("image_presence", {})
    for metric in ("accuracy", "precision", "recall", "F1"):
        value = presence.get(metric)
        if isinstance(value, (int, float)) and np.isfinite(value):
            values[f"image_presence/{metric}"] = float(value)
    return values


def aggregate(runs_dir, suite, seeds=(0, 1, 2)):
    runs_dir = Path(runs_dir)
    grouped = defaultdict(list)
    individual = []
    for method, pattern in SUITES[suite].items():
        for seed in seeds:
            run_id = pattern.format(seed)
            report_path = runs_dir / run_id / "eval" / "metrics_summary.json"
            if not report_path.is_file():
                raise FileNotFoundError(
                    f"Missing metrics for {run_id}: expected {report_path}"
                )
            report = json.loads(report_path.read_text(encoding="utf-8"))
            values = _flatten_metrics(report)
            grouped[method].append(values)
            individual.append({"method": method, "seed": seed, "metrics": values})

    summary = {}
    for method, runs in grouped.items():
        metric_names = sorted(set.intersection(*(set(run) for run in runs)))
        summary[method] = {}
        for metric in metric_names:
            values = np.asarray([run[metric] for run in runs], dtype=float)
            summary[method][metric] = {
                "n": int(values.size),
                "mean": float(values.mean()),
                "std_sample": float(values.std(ddof=1)) if values.size > 1 else None,
                "values_by_seed": values.tolist(),
            }

    output = {
        "suite": suite,
        "split": "validation",
        "seeds": list(seeds),
        "accuracy_definition": (
            "Binary image-level accuracy for presence/absence of at least one B-line "
            "at each run's validation-selected confidence threshold."
        ),
        "f1_threshold_note": (
            "Each run selects its own confidence threshold to maximize object-level "
            "F1 on validation; these are validation estimates, not independent test results."
        ),
        "individual_runs": individual,
        "mean_and_sample_std_by_method": summary,
    }
    output_path = runs_dir / "aggregated_metrics.json"
    output_path.write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, indent=2, allow_nan=False))
    print(f"Aggregate salvo em: {output_path}")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", required=True)
    parser.add_argument("--suite", choices=tuple(SUITES), required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = parser.parse_args()
    aggregate(args.runs_dir, args.suite, args.seeds)


if __name__ == "__main__":
    main()
