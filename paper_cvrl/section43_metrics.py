"""Confusion-matrix metrics shared by the Section 4.3 audit and figures."""

from __future__ import annotations

import numpy as np


def per_class_iou(confusion_matrix):
    """Return IoU per class; absent classes are NaN and excluded from mIoU."""
    confusion = np.asarray(confusion_matrix, dtype=np.float64)
    if confusion.ndim != 2 or confusion.shape[0] != confusion.shape[1]:
        raise ValueError("confusion_matrix must be a square 2-D array")
    true_positive = np.diag(confusion)
    union = confusion.sum(axis=1) + confusion.sum(axis=0) - true_positive
    return np.divide(
        true_positive, union,
        out=np.full_like(true_positive, np.nan), where=union > 0,
    )


def mean_iou(confusion_matrix):
    values = per_class_iou(confusion_matrix)
    return float(np.nanmean(values)) if np.isfinite(values).any() else 0.0


def metric_miou(metric_block):
    """Read mIoU from old or new audit JSON without changing its test rows."""
    if "mIoU" in metric_block:
        return float(metric_block["mIoU"])
    return mean_iou(metric_block["confusion_matrix"])


def metric_class_iou(metric_block, class_id):
    if metric_block["per_class"][str(class_id)].get("IoU") is not None:
        return float(metric_block["per_class"][str(class_id)]["IoU"])
    class_ids = list(metric_block["per_class"])
    return float(per_class_iou(metric_block["confusion_matrix"])[
        class_ids.index(str(class_id))
    ])
