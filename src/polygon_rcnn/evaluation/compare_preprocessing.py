"""Compare raw input and preprocessing variants by paired validation seed."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.polygon_rcnn.evaluation.aggregate_validation_metrics import _flatten_metrics


VARIANTS = ("clahe", "nlm", "unsharp", "canny", "nlm_clahe")
CONTROL_PATTERN = "polygon_struct_iou_area_s{}"
VARIANT_LABELS = {
    "control": "Original",
    "clahe": "CLAHE",
    "nlm": "NLM",
    "unsharp": "Unsharp",
    "canny": "Canny",
    "nlm_clahe": "NLM → CLAHE",
}


def _load(root: Path, pattern: str, seeds: list[int]):
    result = {}
    for seed in seeds:
        run_id = pattern.format(seed)
        path = root / run_id / "eval" / "metrics_summary.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing validation metrics for {run_id}: {path}")
        result[seed] = _flatten_metrics(json.loads(path.read_text(encoding="utf-8")))
    return result


def _mean_std(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else None


def _write_group_plot(out: Path, rows, metrics, title, filename):
    metric_names = [metric for metric in metrics if any(row["metric"] == metric for row in rows)]
    if not metric_names:
        return
    variant_order = ("control", *VARIANTS)
    lookup = {(row["variant"], row["metric"]): row for row in rows}
    positions = np.arange(len(metric_names))
    width = 0.82 / len(variant_order)
    fig, ax = plt.subplots(figsize=(max(9, 2.1 * len(metric_names)), 5.5))
    for index, variant in enumerate(variant_order):
        values = [lookup[(variant, metric)] for metric in metric_names]
        means = np.asarray([row["mean"] for row in values], dtype=float)
        stds = np.asarray([row["std_sample"] or 0.0 for row in values], dtype=float)
        # COCO AP is already expressed in percentage points; other rates are fractions.
        scales = np.asarray([1.0 if metric.startswith("COCO/") else 100.0 for metric in metric_names])
        ax.bar(
            positions + (index - (len(variant_order) - 1) / 2) * width,
            means * scales,
            width,
            yerr=stds * scales,
            capsize=2,
            label=VARIANT_LABELS[variant],
            color="#777777" if variant == "control" else None,
        )
    labels = [metric.rsplit("/", 1)[-1] for metric in metric_names]
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Métrica (%)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def compare(reference_dir: Path, runs_dir: Path, output_dir: Path, seeds=(0, 1, 2)):
    reference_dir = reference_dir.expanduser().resolve()
    runs_dir = runs_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(seeds)
    per_variant = {"control": _load(reference_dir, CONTROL_PATTERN, seeds)}
    for variant in VARIANTS:
        per_variant[variant] = _load(
            runs_dir,
            f"polygon_struct_iou_area_{variant}_s{{}}",
            seeds,
        )

    common_metrics = set.intersection(*(
        set(per_variant[variant][seed])
        for variant in per_variant
        for seed in seeds
    ))
    rows = []
    for variant, seed_metrics in per_variant.items():
        for metric in sorted(common_metrics):
            values = [seed_metrics[seed][metric] for seed in seeds]
            mean, std = _mean_std(values)
            if variant == "control":
                deltas = [0.0] * len(seeds)
                delta_mean, delta_std = 0.0, 0.0 if len(seeds) > 1 else None
            else:
                deltas = [
                    seed_metrics[seed][metric] - per_variant["control"][seed][metric]
                    for seed in seeds
                ]
                delta_mean, delta_std = _mean_std(deltas)
            rows.append({
                "variant": variant,
                "metric": metric,
                "mean": mean,
                "std_sample": std,
                "paired_delta_vs_original_mean": delta_mean,
                "paired_delta_vs_original_std_sample": delta_std,
                "values_by_seed": values,
                "paired_delta_by_seed": deltas,
            })

    report = {
        "control_runs_dir": str(reference_dir),
        "preprocessing_runs_dir": str(runs_dir),
        "split": "validation",
        "seeds": seeds,
        "comparison": "same seeds; deltas are variant minus original-input control",
        "variants": list(per_variant),
        "metrics": rows,
        "notes": [
            "All variants use the same structured polygon head, IoU + area loss, training budget, data split, and seeds.",
        "Canny is applied to the raw image by the documented binary/blur/edge/contour sequence; the referenced paper pseudocode starts from a segmented image, so this is an adapted edge-input ablation.",
            "Metrics are validation results; confidence thresholds are selected separately per run on validation.",
        ],
    }
    (output_dir / "preprocessing_comparison.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    flat_rows = []
    for row in rows:
        flat_rows.append({key: row[key] for key in (
            "variant", "metric", "mean", "std_sample",
            "paired_delta_vs_original_mean", "paired_delta_vs_original_std_sample",
        )})
    with (output_dir / "preprocessing_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)

    def fmt(value, std):
        return f"{value:.4f}" if std is None else f"{value:.4f} ± {std:.4f}"

    lines = [
        "# Comparação de pré-processamentos",
        "",
        f"Controle: `{reference_dir}`",
        f"Experimentos: `{runs_dir}`",
        "",
        "Média ± desvio-padrão amostral entre três seeds. Δ é a diferença pareada para a entrada original.",
        "AP está em pontos percentuais; demais taxas estão entre 0 e 1. Resultados de validação.",
        "",
        "| Pré-processamento | Métrica | Média ± DP | Δ pareado vs. original |",
        "|---|---|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {VARIANT_LABELS[row['variant']]} | {row['metric']} | "
            f"{fmt(row['mean'], row['std_sample'])} | "
            f"{fmt(row['paired_delta_vs_original_mean'], row['paired_delta_vs_original_std_sample'])} |"
        )
    lines += [
        "",
        "Canny usa uma adaptação à imagem original: limiarização de Otsu, dilatação, blur Gaussiano, Canny, dilatação e extração de contornos.",
        "O pseudocódigo de referência parte de uma imagem já segmentada; portanto, essa variante testa entrada de bordas e não substitui a segmentação treinada.",
        "A combinação definida antes dos resultados é NLM seguido por CLAHE: a hipótese é reduzir speckle antes de aumentar contraste local.",
    ]
    (output_dir / "preprocessing_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    _write_group_plot(
        output_dir, rows,
        ["COCO/segm_polygon/AP", "COCO/segm_polygon/AP50", "COCO/segm_polygon/AP75"],
        "AP de polígonos por pré-processamento (média ± DP)",
        "preprocessing_polygon_ap.png",
    )
    _write_group_plot(
        output_dir, rows,
        ["detection/precision", "detection/recall", "detection/F1"],
        "Precision, recall e F1 de instâncias",
        "preprocessing_detection_metrics.png",
    )
    _write_group_plot(
        output_dir, rows,
        ["image_presence/accuracy", "image_presence/precision", "image_presence/recall", "image_presence/F1"],
        "Métricas de presença de B-lines por imagem",
        "preprocessing_image_presence_metrics.png",
    )
    _write_group_plot(
        output_dir, rows,
        [
            "COCO/segm_polygon/AP", "COCO/segm_polygon/AP50", "COCO/segm_polygon/AP75",
            "detection/precision", "detection/recall", "detection/F1",
            "image_presence/accuracy", "image_presence/precision",
            "image_presence/recall", "image_presence/F1",
        ],
        "Visão geral de todas as métricas por pré-processamento",
        "preprocessing_all_metrics.png",
    )
    print(f"Comparação consolidada salva em: {output_dir}")
    for name in (
        "preprocessing_comparison.md", "preprocessing_comparison.csv",
        "preprocessing_comparison.json", "preprocessing_polygon_ap.png",
        "preprocessing_detection_metrics.png", "preprocessing_image_presence_metrics.png",
        "preprocessing_all_metrics.png",
    ):
        if (output_dir / name).exists():
            print(f"  {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-runs-dir", required=True)
    parser.add_argument("--runs-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = parser.parse_args()
    compare(Path(args.reference_runs_dir), Path(args.runs_dir), Path(args.output_dir), args.seeds)


if __name__ == "__main__":
    main()
