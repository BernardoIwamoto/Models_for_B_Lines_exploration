"""Preview or remove numbered training snapshots for fully evaluated runs.

Only model_NNNNNNN.pth snapshots are eligible. The selected best checkpoint,
model_final.pth, metrics, logs, and all incomplete runs are preserved.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


VARIANTS = ("clahe", "nlm", "unsharp", "canny", "nlm_clahe")
CHECKPOINT = re.compile(r"model_\d+\.pth$")


def cleanup(runs_dir: Path, *, delete: bool = False):
    runs_dir = runs_dir.expanduser().resolve()
    if not runs_dir.is_dir():
        raise FileNotFoundError(runs_dir)
    total_bytes = 0
    run_count = 0
    snapshot_count = 0
    for variant in VARIANTS:
        for seed in range(3):
            run_dir = runs_dir / f"polygon_struct_iou_area_{variant}_s{seed}"
            metrics = run_dir / "eval" / "metrics_summary.json"
            best = run_dir / "model_best_segm_polygon_AP.pth"
            if not metrics.is_file() or not best.is_file():
                continue
            snapshots = sorted(
                path for path in run_dir.iterdir()
                if path.is_file() and CHECKPOINT.fullmatch(path.name)
            )
            if not snapshots:
                continue
            run_count += 1
            run_bytes = sum(path.stat().st_size for path in snapshots)
            total_bytes += run_bytes
            snapshot_count += len(snapshots)
            print(f"{run_dir.name}: {len(snapshots)} numbered snapshots ({run_bytes / (1024 ** 3):.2f} GiB)")
            if delete:
                for path in snapshots:
                    path.unlink()
    gib = total_bytes / (1024 ** 3)
    action = "Removed" if delete else "Would remove"
    print(f"{action} {snapshot_count} snapshots from {run_count} fully evaluated runs ({gib:.2f} GiB).")
    if not delete:
        print("Preview only. Add --delete to remove these numbered snapshots.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", required=True)
    parser.add_argument("--delete", action="store_true", help="Actually delete eligible numbered snapshots")
    args = parser.parse_args()
    cleanup(Path(args.runs_dir), delete=args.delete)


if __name__ == "__main__":
    main()
