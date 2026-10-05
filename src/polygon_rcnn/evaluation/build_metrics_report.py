"""Create readable comparison tables and figures from completed experiment suites."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.polygon_rcnn.evaluation.aggregate_validation_metrics import (
    SUITES,
    aggregate,
)


SUITE_LABELS = {
    "baselines": "Baselines",
    "polygon_ablations": "Polygon ablations",
}
METHOD_LABELS = {
    "mask_bbox": "Mask R-CNN",
    "faster_bbox": "Faster R-CNN",
    "polygon_legacy": "Polygon legacy",
    "yolo11n_bbox": "YOLOv11n",
    "polygon_center_vertex": "Center + vertex loss",
    "polygon_struct_vertex": "Structured + vertex loss",
    "polygon_struct_iou": "Structured + IoU loss",
    "polygon_struct_iou_area": "Structured + IoU + area loss",
}
DETECTION_TASKS = {
    "mask_bbox": "segm",
    "faster_bbox": "bbox",
    "polygon_legacy": "segm",
    "yolo11n_bbox": "bbox",
    "polygon_center_vertex": "segm",
    "polygon_struct_vertex": "segm",
    "polygon_struct_iou": "segm",
    "polygon_struct_iou_area": "segm",
}
RATE_PREFIXES = ("detection/", "image_presence/")
RATE_METRICS = {
    "detection/precision", "detection/recall", "detection/F1",
    "image_presence/accuracy", "image_presence/precision",
    "image_presence/recall", "image_presence/F1",
}


def _read_suite(runs_dir: Path, suite: str) -> dict:
    path = runs_dir / f"aggregated_{suite}_metrics.json"
    # Accept the baseline aggregate produced by the earlier runner version.
    if not path.is_file() and suite == "baselines":
        legacy_path = runs_dir / "aggregated_metrics.json"
        if legacy_path.is_file():
            path = legacy_path
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("suite") != suite:
            raise ValueError(f"{path} describes suite={data.get('suite')!r}, expected {suite!r}")
        return data
    return aggregate(runs_dir, suite)


def _scale(metric: str, value: float) -> tuple[float, str]:
    if metric in RATE_METRICS:
        return value * 100.0, "%"
    if metric.startswith("COCO/"):
        return value, "%"
    if metric == "detection/score_threshold":
        return value, "confidence"
    return value, "count"


def _fmt(summary: dict, metric: str) -> str:
    item = summary.get(metric)
    if item is None:
        return "—"
    mean, unit = _scale(metric, float(item["mean"]))
    std = item.get("std_sample")
    if std is None:
        text = f"{mean:.2f}"
    else:
        std_value, _ = _scale(metric, float(std))
        text = f"{mean:.2f} ± {std_value:.2f}"
    return f"{text}{'%' if unit == '%' else ''}"


def _markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def _build_markdown(suites: dict[str, dict]) -> str:
    lines = [
        "# Validation metrics report",
        "",
        "Values are mean ± sample standard deviation across seeds. COCO AP values and rate metrics are shown as percentages.",
        "Object-level precision, recall, and F1 use each model's task at IoU 0.50. Image-presence metrics classify whether each image contains at least one B-line.",
        "The confidence threshold is selected separately for each run to maximize F1 on validation; these are not independent test estimates.",
        "",
    ]
    for suite, data in suites.items():
        summary = data["mean_and_sample_std_by_method"]
        methods = list(summary)
        lines.extend([f"## {SUITE_LABELS[suite]}", ""])

        coco_tasks = sorted({
            key.split("/")[1]
            for stats in summary.values()
            for key in stats
            if key.startswith("COCO/") and len(key.split("/")) == 3
        })
        for task in coco_tasks:
            lines.extend([f"### COCO {task} AP (%)", ""])
            rows = []
            for method in methods:
                stats = summary[method]
                if f"COCO/{task}/AP" not in stats:
                    continue
                rows.append([
                    METHOD_LABELS.get(method, method),
                    *[_fmt(stats, f"COCO/{task}/{metric}") for metric in ("AP", "AP50", "AP75")],
                ])
            lines.extend(_markdown_table(["Modelo", "AP", "AP50", "AP75"], rows))
            lines.append("")

        detection_groups = {}
        for method in methods:
            task = DETECTION_TASKS.get(method, "unknown")
            detection_groups.setdefault(task, []).append(method)
        for task, task_methods in detection_groups.items():
            lines.extend([f"### Detecção por instância ({task}, IoU 0,50; %) ", ""])
            rows = []
            for method in task_methods:
                stats = summary[method]
                rows.append([
                    METHOD_LABELS.get(method, method),
                    *[_fmt(stats, f"detection/{metric}") for metric in ("precision", "recall", "F1")],
                    _fmt(stats, "detection/score_threshold"),
                ])
            lines.extend(_markdown_table(["Modelo", "Precision", "Recall", "F1", "Limiar"], rows))
            lines.append("")

        lines.extend(["### Presença de B-line por imagem (%)", ""])
        rows = []
        for method in methods:
            stats = summary[method]
            rows.append([
                METHOD_LABELS.get(method, method),
                *[_fmt(stats, f"image_presence/{metric}") for metric in ("accuracy", "precision", "recall", "F1")],
            ])
        lines.extend(_markdown_table(["Modelo", "Accuracy", "Precision", "Recall", "F1"], rows))
        lines.append("")
    return "\n".join(lines)


def _write_csvs(out_dir: Path, suites: dict[str, dict]) -> None:
    summary_rows = []
    seed_rows = []
    for suite, data in suites.items():
        summary = data["mean_and_sample_std_by_method"]
        for method, metrics in summary.items():
            for metric, item in metrics.items():
                mean, unit = _scale(metric, float(item["mean"]))
                std = item.get("std_sample")
                std_value = _scale(metric, float(std))[0] if std is not None else ""
                task = DETECTION_TASKS.get(method, "") if metric.startswith("detection/") else ""
                summary_rows.append({
                    "suite": suite,
                    "method": method,
                    "metric": metric,
                    "task": task,
                    "mean": mean,
                    "std_sample": std_value,
                    "unit": unit,
                    "values_by_seed": json.dumps(
                        [_scale(metric, float(value))[0] for value in item.get("values_by_seed", [])]
                    ),
                })
        for run in data.get("individual_runs", []):
            method = run["method"]
            for metric, value in run["metrics"].items():
                scaled, unit = _scale(metric, float(value))
                seed_rows.append({
                    "suite": suite,
                    "method": method,
                    "seed": run["seed"],
                    "metric": metric,
                    "task": DETECTION_TASKS.get(method, "") if metric.startswith("detection/") else "",
                    "value": scaled,
                    "unit": unit,
                })

    summary_fields = ["suite", "method", "metric", "task", "mean", "std_sample", "unit", "values_by_seed"]
    with (out_dir / "metrics_mean_std.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    seed_fields = ["suite", "method", "seed", "metric", "task", "value", "unit"]
    with (out_dir / "metrics_per_seed.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=seed_fields)
        writer.writeheader()
        writer.writerows(seed_rows)


def _bar_plot(out_path: Path, suites: dict[str, dict], metric_keys: list[str], title: str, group: str) -> None:
    entries = []
    for suite, data in suites.items():
        for method, metrics in data["mean_and_sample_std_by_method"].items():
            if group == "bbox" and DETECTION_TASKS.get(method) != "bbox":
                continue
            if group == "segm" and DETECTION_TASKS.get(method) != "segm":
                continue
            if not any(key in metrics for key in metric_keys):
                continue
            entries.append((suite, method, metrics))
    if not entries:
        return

    fig, axes = plt.subplots(1, len(metric_keys), figsize=(max(7, 4.6 * len(metric_keys)), 5), squeeze=False)
    axes = axes[0]
    labels = [METHOD_LABELS.get(method, method) for _, method, _ in entries]
    colors = ["#2878B5" if suite == "baselines" else "#E07A24" for suite, _, _ in entries]
    x = np.arange(len(entries))
    for axis, metric in zip(axes, metric_keys):
        means, stds = [], []
        for _, _, metrics in entries:
            if metric in metrics:
                mean, _ = _scale(metric, float(metrics[metric]["mean"]))
                std = metrics[metric].get("std_sample")
                std = _scale(metric, float(std))[0] if std is not None else 0.0
            else:
                mean, std = np.nan, 0.0
            means.append(mean)
            stds.append(std)
        axis.bar(x, means, yerr=stds, capsize=3, color=colors, edgecolor="white")
        axis.set_title(metric.split("/")[-1])
        axis.set_ylabel("%")
        axis.set_ylim(0, 105)
        axis.grid(axis="y", alpha=0.25)
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=35, ha="right")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_plots(out_dir: Path, suites: dict[str, dict]) -> list[str]:
    generated = []
    image_keys = [f"image_presence/{name}" for name in ("accuracy", "precision", "recall", "F1")]
    path = out_dir / "image_presence_metrics.png"
    _bar_plot(path, suites, image_keys, "B-line presence per image", "all")
    if path.exists():
        generated.append(path.name)

    for group, label in (("bbox", "bbox"), ("segm", "segmentation")):
        keys = [f"detection/{name}" for name in ("precision", "recall", "F1")]
        path = out_dir / f"detection_metrics_{label}.png"
        _bar_plot(path, suites, keys, f"Instance detection metrics: {label}", group)
        if path.exists():
            generated.append(path.name)

    coco_tasks = sorted({
        key.split("/")[1]
        for data in suites.values()
        for stats in data["mean_and_sample_std_by_method"].values()
        for key in stats
        if key.startswith("COCO/") and len(key.split("/")) == 3
    })
    for task in coco_tasks:
        keys = [f"COCO/{task}/{name}" for name in ("AP", "AP50", "AP75")]
        path = out_dir / f"coco_ap_{task}.png"
        _bar_plot(path, suites, keys, f"COCO {task} AP", "all")
        if path.exists():
            generated.append(path.name)
    return generated


def build_report(runs_dir: str | Path, out_dir: str | Path | None = None) -> Path:
    runs_dir = Path(runs_dir).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve() if out_dir else runs_dir / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    suites = {suite: _read_suite(runs_dir, suite) for suite in SUITES}
    markdown = _build_markdown(suites)
    (out_dir / "metrics_comparison.md").write_text(markdown + "\n", encoding="utf-8")
    _write_csvs(out_dir, suites)
    plots = _write_plots(out_dir, suites)
    index = {
        "runs_dir": str(runs_dir),
        "suites": list(suites),
        "files": ["metrics_comparison.md", "metrics_mean_std.csv", "metrics_per_seed.csv", *plots],
        "metric_scale_note": "COCO AP and rate metrics are expressed in percent in CSV/Markdown; confidence thresholds remain 0-1.",
    }
    (out_dir / "report_files.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Relatório legível salvo em: {out_dir}")
    for filename in index["files"]:
        print(f"  {filename}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", required=True, help="Parent folder containing baseline and ablation runs")
    parser.add_argument("--output-dir", help="Defaults to RUNS_DIR/reports")
    args = parser.parse_args()
    build_report(args.runs_dir, args.output_dir)


if __name__ == "__main__":
    main()
