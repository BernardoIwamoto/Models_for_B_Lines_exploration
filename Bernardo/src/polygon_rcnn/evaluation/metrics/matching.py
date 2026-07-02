import numpy as np

from .segmentation import iou


def best_match(pred_mask, gt_masks):

    if len(gt_masks) == 0:
        return -1, 0

    scores = [iou(pred_mask, gt) for gt in gt_masks]

    idx = np.argmax(scores)

    return idx, scores[idx]