from pathlib import Path

import cv2
import numpy as np


def _canonical_vertex_order(coords):
    """Reorders 4 polygon vertices so index 0 is always the topmost point, followed
    by a consistent winding order.

    Needed for keypoint-style vertex regression (the Polygon Head): each output slot
    must consistently correspond to the same geometric corner across instances, but
    the raw annotations have no guaranteed starting vertex or winding direction.
    """

    cx, cy = coords.mean(axis=0)

    angles = np.arctan2(coords[:, 1] - cy, coords[:, 0] - cx)

    coords = coords[np.argsort(angles)]

    start = int(np.argmin(coords[:, 1]))

    return np.roll(coords, -start, axis=0)


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

                    # Extra field, ignored by any model with KEYPOINT_ON=False (Mask/
                    # Faster R-CNN, YOLO) -- only the Polygon Head consumes this.
                    ordered = _canonical_vertex_order(coords)

                    keypoints = np.concatenate(
                        [ordered, np.full((4, 1), 2.0)],  # visibility=2 (labeled, visible)
                        axis=1,
                    ).flatten().tolist()

                    objects.append(
                        {
                            "bbox": [xmin, ymin, xmax, ymax],
                            "bbox_mode": 0,
                            "category_id": cls,
                            "segmentation": [segmentation],
                            "keypoints": keypoints,
                            "iscrowd": 0,
                        }
                    )

        record["annotations"] = objects

        dataset_dicts.append(record)

    return dataset_dicts