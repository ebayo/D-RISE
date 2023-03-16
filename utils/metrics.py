# Similarity metric for algorithm and auxillary metrics
import numpy as np


def iou(a, b):
    # intersection box points
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    iarea = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    aarea = (a[2] - a[0] + 1) * (a[3] - a[1] + 1)
    barea = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)
    return iarea / float(aarea + barea - iarea)


def similarity_metric(dt, dj, obj=True):
    # dt and dj are expected as:
    # d = [x1,y1, x2,y2,  obj, pi, ..., pc]
    # x1,y1, x2,y2 --> bounding box
    # obj --> objectness score (1 for ground truth)
    # pi, ..., pc --> "probability for each of the c classes
    if obj:
        ioutj = iou(dt[:4], dj[:4])
        predtj = np.dot(dt[5:], dj[5:])
        objtj = dt[4] * dj[4]
    else:
        ioutj = iou(dt[:4], dj[:4])
        predtj = np.dot(dt[4:], dj[4:])
        objtj = 1

    return ioutj * predtj * objtj


def similarity_metric_simplified(dt, dj, obj=True):
    # dt and dj are expected as:
    # d = [x1,y1, x2,y2,  obj, conf, cls]
    # x1,y1, x2,y2 --> bounding box
    # obj --> objectness score (1 for ground truth)
    # conf --> confidence score for class cls, 1 if ground truth
    # cls --> class asigned to the bounding boxx
    if dt[-1] != dj[-1]:
        # Not the same class, similarity is 0
        return 0

    ioutj = iou(dt[:4], dj[:4])
    predtj = dt[-2] * dj[-2]

    if not obj:
        return ioutj * predtj

    objtj = dt[4] * dj[4]
    return ioutj * predtj * objtj



