import os
from pathlib import Path

from src.polygon_rcnn.annotation import parse_yolo_polygon_line


DATA_ROOT = Path("data")

OUTPUT_ROOT = Path("data/yolo_detect")

SPLITS = ["train", "val", "test"]


def convert_split(split):

    image_dir = DATA_ROOT / split / "images"
    label_dir = DATA_ROOT / split / "labels"

    out_image_dir = OUTPUT_ROOT / split / "images"
    out_label_dir = OUTPUT_ROOT / split / "labels"

    out_image_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    for image_path in sorted(image_dir.glob("*")):

        link_path = out_image_dir / image_path.name

        if not link_path.exists():
            target = os.path.relpath(image_path.resolve(), start=link_path.parent)
            link_path.symlink_to(target)
        elif not link_path.is_symlink() or link_path.resolve() != image_path.resolve():
            raise FileExistsError(f"Refusing to replace non-matching dataset image link: {link_path}")

        label_path = label_dir / f"{image_path.stem}.txt"

        out_label_path = out_label_dir / f"{image_path.stem}.txt"

        lines = []

        if label_path.exists():

            with open(label_path) as f:

                for line_number, line in enumerate(f, 1):

                    line = line.strip()

                    if not line:
                        continue

                    parsed = parse_yolo_polygon_line(
                        line, source=f"{label_path}:{line_number}"
                    )
                    if parsed is None:
                        continue
                    cls, coords = parsed

                    xmin, ymin = coords.min(axis=0)
                    xmax, ymax = coords.max(axis=0)

                    cx = (xmin + xmax) / 2
                    cy = (ymin + ymax) / 2
                    w = xmax - xmin
                    h = ymax - ymin

                    lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        converted = "\n".join(lines)
        if out_label_path.exists():
            if out_label_path.read_text() != converted:
                raise FileExistsError(
                    f"Refusing to overwrite changed converted labels: {out_label_path}"
                )
        else:
            out_label_path.write_text(converted)


for split in SPLITS:
    convert_split(split)

data_yaml = f"""\
path: {OUTPUT_ROOT.resolve()}
train: train/images
val: val/images
test: test/images

names:
  0: bline
"""

yaml_path = OUTPUT_ROOT / "data.yaml"
if yaml_path.exists():
    if yaml_path.read_text() != data_yaml:
        raise FileExistsError(f"Refusing to overwrite changed dataset config: {yaml_path}")
else:
    yaml_path.write_text(data_yaml)

print(f"YOLO detection dataset written to {OUTPUT_ROOT}")
