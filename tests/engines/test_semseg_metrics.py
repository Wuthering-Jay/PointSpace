import numpy as np
import pytest

from pointspace.utils.misc import semantic_segmentation_metrics


def test_semantic_segmentation_metrics_from_iut():
    metrics = semantic_segmentation_metrics(
        intersection=np.asarray([3, 4]),
        union=np.asarray([6, 7]),
        target=np.asarray([5, 5]),
    )

    assert metrics["iou_class"] == pytest.approx([0.5, 4 / 7])
    assert metrics["precision_class"] == pytest.approx([3 / 4, 4 / 6])
    assert metrics["recall_class"] == pytest.approx([3 / 5, 4 / 5])
    assert metrics["mIoU"] == pytest.approx((0.5 + 4 / 7) / 2)
    assert metrics["allAcc"] == pytest.approx(0.7)
    assert metrics["fwIoU"] == pytest.approx((0.5 + 4 / 7) / 2)
    assert metrics["kappa"] == pytest.approx(0.4)


def test_semantic_segmentation_metrics_rejects_bad_shape():
    with pytest.raises(ValueError, match="same shape"):
        semantic_segmentation_metrics([1, 2], [2], [2, 3])
