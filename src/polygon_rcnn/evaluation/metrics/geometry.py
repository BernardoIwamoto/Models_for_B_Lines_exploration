import cv2
import numpy as np


def area(mask):

    return mask.sum()


def polygon_to_mask(polygon_xy, shape):
    """Rasterizes a flat (x1,y1,...,xn,yn) polygon into a boolean mask of `shape`
    (h, w, ...). Moved here from evaluate_predictions.py so both the existing
    per-instance mask analysis and the newer polygon-geometry metrics (matching.py,
    polygon.py) rasterize the exact same way instead of two versions drifting apart.
    """

    mask = np.zeros(shape[:2], dtype=np.uint8)

    polygon = np.array(polygon_xy, dtype=np.int32).reshape(-1, 2)

    cv2.fillPoly(mask, [polygon], 1)

    return mask.astype(bool)


def box_iou(box1, box2):
    """IoU between two axis-aligned boxes in COCO xywh format. Analytic, not
    rasterized -- boxes don't need polygon_to_mask's pixel grid, and this is the
    inner loop of bbox_bias_analysis.py's matching, called many times.
    """

    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    ix1, iy1 = max(x1, x2), max(y1, y2)
    ix2, iy2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)

    inter_w, inter_h = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = inter_w * inter_h

    union = w1 * h1 + w2 * h2 - inter

    return inter / union if union > 0 else 0.0