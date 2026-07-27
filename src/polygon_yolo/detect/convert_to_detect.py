import os
from pathlib import Path

import numpy as np


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

        label_path = label_dir / f"{image_path.stem}.txt"

        out_label_path = out_label_dir / f"{image_path.stem}.txt"

        lines = []

        if label_path.exists():

            with open(label_path) as f:

                for line in f:

                    line = line.strip()

                    if not line:
                        continue

                    values = list(map(float, line.split()))

                    cls = int(values[0])

                    coords = np.array(values[1:]).reshape(-1, 2)

                    xmin, ymin = coords.min(axis=0)
                    xmax, ymax = coords.max(axis=0)

                    cx = (xmin + xmax) / 2
                    cy = (ymin + ymax) / 2
                    w = xmax - xmin
                    h = ymax - ymin

                    lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        out_label_path.write_text("\n".join(lines))


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

(OUTPUT_ROOT / "data.yaml").write_text(data_yaml)

print(f"YOLO detection dataset written to {OUTPUT_ROOT}")
