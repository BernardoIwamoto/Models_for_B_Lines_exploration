"""Aggregate the same validation AP metrics across independent seeds."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def aggregate(named_results, task, output):
    grouped = defaultdict(list)
    individual = []
    for name, result_path in named_results:
        data = json.loads(Path(result_path).read_text())
        metrics = data.get(task)
        if not isinstance(metrics, dict):
            raise KeyError(f"{result_path} does not contain task '{task}'")
        values = {metric: float(metrics[metric]) for metric in ("AP", "AP50", "AP75")
                  if metric in metrics}
        if len(values) != 3 or not np.isfinite(list(values.values())).all():
            raise ValueError(f"{result_path} must contain finite AP, AP50, and AP75")
        grouped[name].append(values)
        individual.append({"method": name, "path": str(result_path), "metrics": values})

    summary = {}
    for name, rows in grouped.items():
        summary[name] = {}
        for metric in ("AP", "AP50", "AP75"):
            values = np.asarray([row[metric] for row in rows], dtype=float)
            summary[name][metric] = {
                "n": int(len(values)),
                "mean": float(values.mean()),
                "std_sample": float(values.std(ddof=1)) if len(values) > 1 else None,
                "values": values.tolist(),
            }
    rendered = {"task": task, "individual_runs": individual, "mean_and_std_by_method": summary}
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(rendered, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(rendered, indent=2, allow_nan=False))
    return rendered


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("bbox", "segm", "segm_polygon"), required=True)
    parser.add_argument("--result", action="append", required=True,
                        help="METHOD=results.json; repeat the method name for its seeds")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    parsed = []
    for item in args.result:
        name, sep, path = item.partition("=")
        if not sep or not name or not path:
            parser.error("Each --result must be METHOD=results.json")
        parsed.append((name, path))
    aggregate(parsed, args.task, args.output)


if __name__ == "__main__":
    main()
