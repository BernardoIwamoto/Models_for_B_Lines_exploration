import json
import shutil
import sys
from pathlib import Path


def select_best_checkpoint(output_dir, metric="segm/AP"):
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

    scored = [(r["iteration"], r[metric]) for r in rows if metric in r]

    if not scored:
        raise RuntimeError(f"No rows with '{metric}' found in {output_dir}/metrics.json")

    best_iteration, best_value = max(scored, key=lambda x: x[1])

    checkpoint = output_dir / f"model_{best_iteration:07d}.pth"

    if not checkpoint.exists():
        # The last evaluated iteration has no periodic checkpoint of its own --
        # model_final.pth already is that checkpoint.
        checkpoint = output_dir / "model_final.pth"

    shutil.copy(checkpoint, output_dir / "model_best.pth")

    print(f"Best {metric}={best_value:.2f} at iteration {best_iteration} -> {checkpoint.name}")
    print(f"Copied to {output_dir / 'model_best.pth'}")

    # Each periodic checkpoint is a full model (hundreds of MB); with CHECKPOINT_PERIOD
    # matching EVAL_PERIOD that's ~20 of them per run. Keep only what's needed going
    # forward: model_best.pth (used from here on) and model_final.pth (for reference).
    removed = 0
    for stale in output_dir.glob("model_*.pth"):
        if stale.name not in {"model_best.pth", "model_final.pth"}:
            stale.unlink()
            removed += 1

    if removed:
        print(f"Removed {removed} intermediate checkpoint(s) to free up disk space.")

    return checkpoint


if __name__ == "__main__":
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "output_maskrcnn"
    metric = sys.argv[2] if len(sys.argv) > 2 else "segm/AP"

    select_best_checkpoint(output_dir, metric)
