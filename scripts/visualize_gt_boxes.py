"""Save train/val images with ground-truth bounding boxes derived from polygons."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from src.polygon_rcnn.annotation import parse_yolo_polygon_line


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def visualize_split(data_root: Path, output_root: Path, split: str) -> tuple[int, int]:
    image_dir = data_root / split / "images"
    label_dir = data_root / split / "labels"
    output_dir = output_root / split
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    instance_count = 0

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Não foi possível abrir a imagem: {image_path}")
        height, width = image.shape[:2]
        label_path = label_dir / f"{image_path.stem}.txt"

        if label_path.exists():
            for line_number, line in enumerate(label_path.read_text().splitlines(), 1):
                parsed = parse_yolo_polygon_line(
                    line, source=f"{label_path}:{line_number}"
                )
                if parsed is None:
                    continue
                _, normalized_vertices = parsed
                vertices = normalized_vertices.copy()
                vertices[:, 0] *= width
                vertices[:, 1] *= height

                x0 = max(0, min(width - 1, int(vertices[:, 0].min())))
                y0 = max(0, min(height - 1, int(vertices[:, 1].min())))
                x1 = max(0, min(width - 1, int(vertices[:, 0].max())))
                y1 = max(0, min(height - 1, int(vertices[:, 1].max())))
                cv2.rectangle(image, (x0, y0), (x1, y1), (0, 0, 255), 2)
                instance_count += 1

        destination = output_dir / image_path.name
        if not cv2.imwrite(str(destination), image):
            raise OSError(f"Não foi possível salvar a imagem: {destination}")

    return len(image_paths), instance_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gera cópias das imagens de treino/val com caixas ground truth."
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("anotacao"))
    args = parser.parse_args()

    for split in ("train", "val"):
        image_count, instance_count = visualize_split(
            args.data_root, args.output_dir, split
        )
        print(
            f"{split}: {image_count} imagens salvas, "
            f"{instance_count} caixas ground truth em {args.output_dir / split}"
        )


if __name__ == "__main__":
    main()
