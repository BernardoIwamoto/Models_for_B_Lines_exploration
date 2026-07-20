import numpy as np


def iou(mask1, mask2):

    inter = np.logical_and(mask1, mask2).sum()

    union = np.logical_or(mask1, mask2).sum()

    if union == 0:
        return 0.0

    return inter / union


def dice(mask1, mask2):

    inter = np.logical_and(mask1, mask2).sum()

    total = mask1.sum() + mask2.sum()

    if total == 0:
        return 0.0

    return 2 * inter / total


def precision(mask1, mask2):

    tp = np.logical_and(mask1, mask2).sum()

    fp = np.logical_and(mask1, np.logical_not(mask2)).sum()

    if tp + fp == 0:
        return 0

    return tp / (tp + fp)


def recall(mask1, mask2):

    tp = np.logical_and(mask1, mask2).sum()

    fn = np.logical_and(np.logical_not(mask1), mask2).sum()

    if tp + fn == 0:
        return 0

    return tp / (tp + fn)


def f1(mask1, mask2):

    p = precision(mask1, mask2)

    r = recall(mask1, mask2)

    if p + r == 0:
        return 0

    return 2 * p * r / (p + r)