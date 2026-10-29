"""Create an isolated preprocessing variant while preserving labels and splits."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
VARIANTS = ("clahe", "nlm", "unsharp", "canny", "nlm_clahe")


def _clahe_luminance(image, clip_limit=2.0, tile_grid_size=8):
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(tile_grid_size, tile_grid_size),
    )
    if image.ndim == 2:
        return clahe.apply(image)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _nlm(image, strength=5):
    if image.ndim == 2:
        return cv2.fastNlMeansDenoising(image, None, strength, 7, 21)
    return cv2.fastNlMeansDenoisingColored(
        image, None, strength, strength, 7, 21
    )


def _unsharp(image, amount=0.5, sigma=1.0):
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)


def _canny_pipeline(image):
    """Apply the supplied binary -> dilate -> blur -> Canny -> dilate sequence."""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), dtype=np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)
    blurred = cv2.GaussianBlur(binary, (5, 5), sigmaX=1.0)
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150)
    edges = cv2.dilate(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(
        edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    contour_image = np.zeros_like(edges)
    cv2.drawContours(contour_image, contours, contourIdx=-1, color=255, thickness=1)
    # The detector expects three channels. Replicate the contour map without adding
    # image information or changing the ground-truth polygon coordinates.
    return cv2.cvtColor(contour_image, cv2.COLOR_GRAY2BGR)


def transform_image(image, variant):
    if variant == "clahe":
        return _clahe_luminance(image)
    if variant == "nlm":
        return _nlm(image)
    if variant == "unsharp":
        return _unsharp(image)
    if variant == "canny":
        return _canny_pipeline(image)
    if variant == "nlm_clahe":
        # Denoise first so local contrast enhancement is less likely to amplify
        # speckle noise; this is the combined variant selected a priori.
        return _clahe_luminance(_nlm(image))
    raise ValueError(f"Unknown preprocessing variant: {variant}")


def prepare_dataset(source: Path, destination: Path, variant: str):
    source = source.expanduser().resolve()
    destination = destination.expanduser().resolve()
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of: {', '.join(VARIANTS)}")
    if not source.is_dir():
        raise FileNotFoundError(f"Dataset source not found: {source}")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset: {destination}")
    if destination == source or source in destination.parents:
        raise ValueError("Destination must be separate from the source dataset")

    destination.mkdir(parents=True)
    counts = {}
    try:
        for split in ("train", "val", "test"):
            source_split = source / split
            output_split = destination / split
            source_images = source_split / "images"
            output_images = output_split / "images"
            output_images.mkdir(parents=True)
            image_paths = sorted(
                p for p in source_images.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
            )
            for image_path in image_paths:
                output_path = output_images / image_path.name
                if split == "test":
                    # The blind test split is preserved but not transformed or used.
                    shutil.copy2(image_path, output_path)
                    continue
                # Match the BGR, three-channel image format used by the
                # Detectron2 dataset loader, including for grayscale/alpha files.
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f"Could not read image: {image_path}")
                transformed = transform_image(image, variant)
                if not cv2.imwrite(str(output_path), transformed):
                    raise OSError(f"Could not write image: {output_path}")

            counts[split] = len(image_paths)
            labels = source_split / "labels"
            if labels.is_dir():
                shutil.copytree(labels, output_split / "labels")

        parameters = {
            "clahe": {"clipLimit": 2.0, "tileGridSize": [8, 8], "channel": "L in CIELAB"},
            "nlm": {"method": "fastNlMeansDenoisingColored / fastNlMeansDenoising", "h": 5, "hColor": 5, "templateWindowSize": 7, "searchWindowSize": 21},
            "unsharp": {"amount": 0.5, "gaussian_sigma": 1.0},
            "canny": {"threshold": "Otsu binary", "initial_dilation_kernel": [3, 3], "gaussian_kernel": [5, 5], "gaussian_sigma": 1.0, "canny_thresholds": [50, 150], "edge_dilation_kernel": [3, 3], "contours": "RETR_LIST, CHAIN_APPROX_SIMPLE, redrawn at 1-pixel thickness", "output": "replicated 3-channel contour map"},
            "nlm_clahe": {"order": ["NLM", "CLAHE"], "NLM_h": 5, "CLAHE_clipLimit": 2.0, "CLAHE_tileGridSize": [8, 8]},
        }
        metadata = {
            "variant": variant,
            "source_dataset": str(source),
            "preprocessing_parameters": parameters[variant],
            "train_val_images_processed": True,
            "test_images_processed": False,
            "labels_copied_without_changes": True,
            "image_counts": counts,
        }
        (destination / "preprocessing.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise

    print(json.dumps({"output": str(destination), **metadata}, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="data", help="Original YOLO polygon dataset")
    parser.add_argument("--output", required=True, help="Output directory for the processed dataset")
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    args = parser.parse_args()
    prepare_dataset(Path(args.source), Path(args.output), args.variant)


if __name__ == "__main__":
    main()
