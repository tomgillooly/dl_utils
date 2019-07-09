import numpy as np

def iou(predicted, gt):
    total = 0

    for cls in np.unique(gt):
        intersection = np.logical_and(predicted == cls, gt == cls).sum()
        union = np.logical_or(predicted == cls, gt == cls).sum()

        total += intersection.float() / union

    return total / len(np.unique(gt))