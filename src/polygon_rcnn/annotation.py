"""Shared parsing and validation for the dataset's four-vertex YOLO polygons."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def parse_yolo_polygon_line(line: str, *, source: str = "<annotation>"):
    """Parse one class + 8 normalized coordinates row without silently truncating."""
    fields = line.split()
    if not fields:
        return None
    if len(fields) != 9:
        raise ValueError(
            f"{source}: expected class + 8 coordinates (4 vertices), got {len(fields)} values"
        )
    try:
        values = np.asarray([float(value) for value in fields], dtype=np.float64)
    except ValueError as exc:
        raise ValueError(f"{source}: annotation contains a non-numeric value") from exc
    if not np.isfinite(values).all():
        raise ValueError(f"{source}: annotation contains NaN or infinity")
    class_id = int(values[0])
    if values[0] != class_id or class_id != 0:
        raise ValueError(f"{source}: expected the single supported class id 0")
    coords = values[1:].reshape(4, 2)
    if ((coords < 0.0) | (coords > 1.0)).any():
        raise ValueError(f"{source}: normalized coordinates must be in [0, 1]")
    return class_id, coords


def audit_dataset(root: str | Path):
    """Return a deterministic audit of image/label pairing and annotation rows."""
    root = Path(root)
    report = {"root": str(root), "splits": {}, "errors": []}
    image_suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    for split in ("train", "val", "test"):
        image_dir = root / split / "images"
        label_dir = root / split / "labels"
        images = sorted(
            p for p in image_dir.glob("*") if p.is_file() and p.suffix.lower() in image_suffixes
        ) if image_dir.exists() else []
        labels = sorted(label_dir.glob("*.txt")) if label_dir.exists() else []
        label_by_stem = {p.stem: p for p in labels}
        split_report = {
            "images": len(images), "label_files": len(labels), "instances": 0,
            "images_without_label_file": [], "orphan_label_files": [],
            "empty_label_files": 0, "malformed_rows": [],
        }
        image_stems = {p.stem for p in images}
        split_report["orphan_label_files"] = sorted(
            p.name for p in labels if p.stem not in image_stems
        )
        for image in images:
            label = label_by_stem.get(image.stem)
            if label is None:
                split_report["images_without_label_file"].append(image.name)
                continue
            rows = [row for row in label.read_text().splitlines() if row.strip()]
            if not rows:
                split_report["empty_label_files"] += 1
            for line_number, row in enumerate(rows, 1):
                try:
                    parse_yolo_polygon_line(row, source=f"{label}:{line_number}")
                    split_report["instances"] += 1
                except ValueError as exc:
                    split_report["malformed_rows"].append(str(exc))
        report["splits"][split] = split_report
        expected_images = {"train": 256, "val": 75, "test": 70}[split]
        if len(images) != expected_images:
            report["errors"].append(
                f"{split}: expected {expected_images} images for the registered dataset version, found {len(images)}"
            )
        # The published test split is blind and intentionally ships no labels.
        keys_to_check = ("orphan_label_files", "malformed_rows")
        if split != "test":
            keys_to_check = ("images_without_label_file",) + keys_to_check
        for key in keys_to_check:
            for problem in split_report[key]:
                report["errors"].append(f"{split}: {problem}")
        if split != "test" and split_report["empty_label_files"]:
            report["errors"].append(
                f"{split}: {split_report['empty_label_files']} empty label files need review"
            )
        if split == "val" and split_report["instances"] != 115:
            report["errors"].append(
                f"val: expected 115 valid four-vertex instances, found {split_report['instances']}"
            )
    return report
