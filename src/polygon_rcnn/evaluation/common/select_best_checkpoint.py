import json
import shutil
from pathlib import Path
import argparse
import math
import re


def select_best_checkpoint(output_dir, metric="segm/AP", destination=None):
    """Copies the checkpoint with the highest validation `metric` to model_best.pth.

    Requires cfg.SOLVER.CHECKPOINT_PERIOD to be set (train_mask_rcnn.py/
    train_faster_rcnn.py match it to TEST.EVAL_PERIOD), so a checkpoint exists at
    every iteration metrics.json has an evaluation row for. Without periodic
    checkpointing, only model_final.pth exists and this has nothing to pick from.
    """

    output_dir = Path(output_dir)

    rows = []
    with open(output_dir / "metrics.json") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    scored = [
        (r["iteration"], r[metric]) for r in rows
        if metric in r and isinstance(r[metric], (int, float)) and math.isfinite(r[metric])
    ]

    if not scored:
        raise RuntimeError(f"No rows with '{metric}' found in {output_dir}/metrics.json")

    best_iteration, best_value = max(scored, key=lambda x: x[1])

    # Detectron2 stores the completed iteration in metrics.json (0-based) but the
    # periodic checkpointer names the corresponding weights using iteration + 1.
    # Prefer that exact checkpoint, while retaining compatibility with old runs.
    candidates = [
        output_dir / f"model_{best_iteration + 1:07d}.pth",
        output_dir / f"model_{best_iteration:07d}.pth",
    ]
    if not any(p.exists() for p in candidates) and best_iteration + 1 == max(
        r.get("iteration", -1) for r in rows
    ):
        candidates.insert(0, output_dir / "model_final.pth")
    checkpoint = next((p for p in candidates if p.exists()), None)
    if checkpoint is None:
        expected = ", ".join(p.name for p in candidates)
        raise FileNotFoundError(
            f"No checkpoint for best {metric}={best_value} at iteration "
            f"{best_iteration}; checked: {expected}"
        )

    slug = re.sub(r"[^A-Za-z0-9]+", "_", metric).strip("_")
    destination = Path(destination) if destination else output_dir / f"model_best_{slug}.pth"
    if destination.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing selected checkpoint: {destination}. "
            "Choose a new destination to preserve the prior run."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(checkpoint, destination)
    selection = {
        "metric": metric,
        "value": best_value,
        "metrics_iteration": best_iteration,
        "checkpoint": str(checkpoint),
        "selected_copy": str(destination),
    }
    metadata_path = destination.with_suffix(destination.suffix + ".selection.json")
    with metadata_path.open("x", encoding="utf-8") as handle:
        json.dump(selection, handle, indent=2)
        handle.write("\n")
    print(
        f"Best {metric}={best_value:.2f} at metrics iteration {best_iteration}: "
        f"{checkpoint.name} -> {destination}"
    )
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", nargs="?", default="output_maskrcnn")
    parser.add_argument("metric", nargs="?", default="segm/AP")
    parser.add_argument("--destination")
    args = parser.parse_args()
    select_best_checkpoint(args.output_dir, args.metric, args.destination)
