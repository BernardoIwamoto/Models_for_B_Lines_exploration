from pathlib import Path

import cv2
import numpy as np


def yolo_polygon_to_detectron(image_dir, label_dir):
    dataset_dicts = []

    image_paths = sorted(image_dir.glob("*"))

    for idx, image_path in enumerate(image_paths):

        img = cv2.imread(str(image_path))

        h, w = img.shape[:2]

        record = {
            "file_name": str(image_path),
            "image_id": idx,
            "height": h,
            "width": w,
        }

        label_path = label_dir / f"{image_path.stem}.txt"

        objects = []

        if label_path.exists():

            with open(label_path) as f:

                for line in f:

                    values = list(map(float, line.strip().split()))

                    cls = int(values[0])

                    coords = np.array(values[1:]).reshape(-1, 2)

                    if len(coords) > 4:
                        coords = coords[:4]

                    coords[:, 0] *= w
                    coords[:, 1] *= h

                    segmentation = coords.flatten().tolist()

                    xmin = float(coords[:, 0].min())
                    xmax = float(coords[:, 0].max())

                    ymin = float(coords[:, 1].min())
                    ymax = float(coords[:, 1].max())

                    objects.append(
                        {
                            "bbox": [xmin, ymin, xmax, ymax],
                            "bbox_mode": 0,
                            "category_id": cls,
                            "segmentation": [segmentation],
                            "iscrowd": 0,
                        }
                    )

        record["annotations"] = objects

        dataset_dicts.append(record)

    return dataset_dicts