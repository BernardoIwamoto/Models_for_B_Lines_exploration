"""Immutable run directories and manifests for remote research experiments."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from src.polygon_rcnn.annotation import audit_dataset


def prepare_run(model, seed, config, selection_metric, *, dataset_root="data"):
    experiment_id = os.environ.get("EXPERIMENT_ID", "").strip()
    if not experiment_id or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", experiment_id):
        raise ValueError(
            "Set a unique EXPERIMENT_ID (letters, digits, dot, dash, underscore) "
            "before training. Existing run paths are never reused."
        )
    audit = audit_dataset(dataset_root)
    if audit["errors"]:
        raise ValueError(
            f"Dataset audit failed ({len(audit['errors'])} issue(s)); "
            "run `python -m src.polygon_rcnn.validate_dataset --root data` "
            "and fix annotations before training."
        )
    run_root = Path(os.environ.get("RUNS_DIR", "runs"))
    run_dir = run_root / experiment_id
    run_dir.mkdir(parents=True, exist_ok=False)
    label_hashes = {}
    for split in ("train", "val"):
        for path in sorted((Path(dataset_root) / split / "labels").glob("*.txt")):
            label_hashes[str(path.relative_to(dataset_root))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    packages = {}
    for name in ("torch", "torchvision", "detectron2", "ultralytics", "numpy", "pycocotools"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    source_hashes = {}
    for folder in (Path("src/polygon_rcnn"), Path("src/polygon_yolo/detect")):
        if folder.exists():
            for path in sorted(folder.rglob("*.py")):
                source_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "experiment_id": experiment_id,
        "model": model,
        "representation": os.environ.get("POLYGON_REPRESENTATION", "n/a"),
        "loss": os.environ.get("POLYGON_LOSS", "n/a"),
        "backbone": "ResNet-50-FPN" if model != "yolo11n_bbox" else "YOLO11n",
        "pretrained_weights": os.environ.get("PRETRAINED_WEIGHTS", "COCO model-zoo / configured weights"),
        "dataset_root": str(dataset_root),
        "dataset_audit": audit["splits"],
        "label_sha256": label_hashes,
        "train_split": "train",
        "validation_split": "val",
        "augmentation": os.environ.get("AUGMENTATION_POLICY", "framework defaults; see config"),
        "seed": int(seed),
        "batch_size": config.get("batch_size"),
        "learning_rate": config.get("learning_rate"),
        "iterations": config.get("max_iter", config.get("epochs")),
        "checkpoint_selection_metric": selection_metric,
        "config": config,
        "packages": packages,
        "git_commit": commit,
        "source_files_sha256": source_hashes,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (run_dir / "experiment.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return str(run_dir)


def save_detectron_config(run_dir, cfg):
    """Write the final Detectron2 config after OUTPUT_DIR has been assigned."""
    path = Path(run_dir) / "config.yaml"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(cfg.dump())


def timed_stage(run_dir, stage):
    """Context manager that records wall time for a training/evaluation stage."""
    from contextlib import contextmanager

    @contextmanager
    def _measure():
        started = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - started
            path = Path(run_dir) / "timing.json"
            data = json.loads(path.read_text()) if path.exists() else {}
            data[stage] = elapsed
            path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    return _measure()
