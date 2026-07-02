import numpy as np


def summarize(values):

    return {

        "mean": float(np.mean(values)),

        "std": float(np.std(values)),

        "median": float(np.median(values)),

        "min": float(np.min(values)),

        "max": float(np.max(values))
    }   