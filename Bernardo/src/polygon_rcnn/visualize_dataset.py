from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


DATASET_ROOT = Path("data/train")


def load_label(label_path, w, h):
    polygons = []

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        values = list(map(float, line.strip().split()))

        cls = int(values[0])

        coords = np.array(values[1:]).reshape(-1, 2)

        if len(coords) > 4:
            coords = coords[:4]

        coords[:, 0] *= w
        coords[:, 1] *= h

        polygons.append((cls, coords.astype(np.int32)))

    return polygons


image_dir = DATASET_ROOT / "images"
label_dir = DATASET_ROOT / "labels"

image_path = sorted(image_dir.glob("*"))[0]

label_path = label_dir / f"{image_path.stem}.txt"

img = cv2.imread(str(image_path))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

h, w = img.shape[:2]

polygons = load_label(label_path, w, h)

for cls, poly in polygons:

    cv2.polylines(
        img,
        [poly],
        isClosed=True,
        color=(255, 0, 0),
        thickness=3,
    )

plt.figure(figsize=(8, 8))
plt.imshow(img)
plt.axis("off")
plt.show()