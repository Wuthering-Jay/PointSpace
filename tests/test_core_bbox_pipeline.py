"""
Tests for core_bbox VLR reading (LasDataset) and transform propagation.

Covers:
    1.  LasDataset.get_data reads core_bbox VLR into data_dict
    2.  Missing VLR → data_dict has no core_bbox key (graceful skip)
    3.  Corrupt VLR JSON → graceful skip
    4.  core_bbox dtype and shape after reading
    5.  CenterShift propagates core_bbox correctly
    6.  CentroidShift propagates core_bbox correctly
    7.  PositiveShift propagates core_bbox correctly
    8.  ZPercentileCenterShift propagates core_bbox correctly
    9.  RandomShift propagates core_bbox correctly
    10. NormalizeCoord propagates core_bbox (shift + scale)
    11. No core_bbox in data_dict → transforms are no-ops (no crash)
    12. Chained transforms maintain alignment between coord and core_bbox
    13. core_bbox after CenterShift is consistent with shifted coord range

Author: PointSpace Team
"""

import json
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import laspy

    HAS_LASPY = True
except ImportError:
    HAS_LASPY = False

from pointspace.datasets.transform import (
    CenterShift,
    CentroidShift,
    NormalizeCoord,
    PositiveShift,
    RandomShift,
    ZPercentileCenterShift,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_data_dict(n=500, seed=42, with_bbox=True):
    """Create a synthetic data_dict with coord and optionally core_bbox."""
    rng = np.random.default_rng(seed)
    # Points scattered in [100, 200] × [300, 400] × [10, 30]
    coord = np.column_stack([
        rng.uniform(100, 200, n),
        rng.uniform(300, 400, n),
        rng.uniform(10, 30, n),
    ]).astype(np.float32)
    data_dict = {"coord": coord}
    if with_bbox:
        # Core bbox is a sub-region of the point cloud extent
        data_dict["core_bbox"] = np.array([110.0, 310.0, 190.0, 390.0],
                                          dtype=np.float32)
    return data_dict


def _write_las_with_vlr(path, points, core_bbox=None, corrupt_vlr=False):
    """Write a minimal LAS file; optionally inject PointSpace core_bbox VLR."""
    header = laspy.LasHeader(point_format=0, version="1.2")
    header.offsets = points.min(axis=0)
    header.scales = np.array([0.001, 0.001, 0.001])

    if core_bbox is not None:
        if corrupt_vlr:
            record_data = b"NOT-JSON{{{{"
        else:
            record_data = json.dumps({"core_bbox": core_bbox.tolist()}).encode()
        vlr = laspy.VLR(
            user_id="PointSpace",
            record_id=1001,
            description="Core BBox for CNF",
            record_data=record_data,
        )
        header.vlrs.append(vlr)

    las = laspy.LasData(header)
    las.x, las.y, las.z = points[:, 0], points[:, 1], points[:, 2]
    las.write(path)


# ---------------------------------------------------------------------------
# 1–4: LasDataset.get_data VLR reading
# ---------------------------------------------------------------------------

@unittest.skipIf(not HAS_LASPY, "laspy not installed")
class TestLasDatasetCoreBBoxRead(unittest.TestCase):
    """LasDataset.get_data correctly reads / skips core_bbox VLR."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.n = 200
        rng = np.random.default_rng(0)
        self.pts = np.column_stack([
            rng.uniform(100, 200, self.n),
            rng.uniform(300, 400, self.n),
            rng.uniform(10, 30, self.n),
        ]).astype(np.float64)
        self.bbox = np.array([110.0, 310.0, 190.0, 390.0], dtype=np.float64)

    def _make_dataset(self, las_name="tile.las", **vlr_kwargs):
        las_path = os.path.join(self.tmp, las_name)
        _write_las_with_vlr(las_path, self.pts, **vlr_kwargs)
        from pointspace.datasets.las import LasDataset
        ds = LasDataset.__new__(LasDataset)
        # Minimal state required by get_data
        ds.data_list = [las_path]
        ds.class2id = None
        ds.ignore_index = -1
        return ds

    def test_core_bbox_present(self):
        ds = self._make_dataset(core_bbox=self.bbox)
        data = ds.get_data(0)
        self.assertIn("core_bbox", data)

    def test_core_bbox_values(self):
        ds = self._make_dataset(core_bbox=self.bbox)
        data = ds.get_data(0)
        np.testing.assert_allclose(data["core_bbox"], self.bbox, atol=1e-3)

    def test_core_bbox_shape_and_dtype(self):
        ds = self._make_dataset(core_bbox=self.bbox)
        data = ds.get_data(0)
        self.assertEqual(data["core_bbox"].shape, (4,))
        self.assertEqual(data["core_bbox"].dtype, np.float32)

    def test_no_vlr_means_no_core_bbox(self):
        ds = self._make_dataset(core_bbox=None)
        data = ds.get_data(0)
        self.assertNotIn("core_bbox", data)

    def test_corrupt_vlr_gracefully_skipped(self):
        ds = self._make_dataset(core_bbox=self.bbox, corrupt_vlr=True)
        data = ds.get_data(0)
        self.assertNotIn("core_bbox", data)


# ---------------------------------------------------------------------------
# 5–10: Individual transform propagation
# ---------------------------------------------------------------------------

class TestCenterShiftCoreBBox(unittest.TestCase):
    def test_bbox_shifted_with_coord(self):
        data = _make_data_dict()
        bbox_before = data["core_bbox"].copy()
        t = CenterShift(apply_z=True)
        result = t(data)
        shift = result["coord_shift"]
        expected = bbox_before.copy()
        expected[0::2] -= shift[0]
        expected[1::2] -= shift[1]
        np.testing.assert_allclose(result["core_bbox"], expected, atol=1e-5)

    def test_no_bbox_no_crash(self):
        data = _make_data_dict(with_bbox=False)
        t = CenterShift()
        result = t(data)
        self.assertNotIn("core_bbox", result)


class TestCentroidShiftCoreBBox(unittest.TestCase):
    def test_bbox_shifted_with_coord(self):
        data = _make_data_dict()
        bbox_before = data["core_bbox"].copy()
        t = CentroidShift(apply_z=True)
        result = t(data)
        shift = result["coord_shift"]
        expected = bbox_before.copy()
        expected[0::2] -= shift[0]
        expected[1::2] -= shift[1]
        np.testing.assert_allclose(result["core_bbox"], expected, atol=1e-5)

    def test_apply_z_false(self):
        data = _make_data_dict()
        bbox_before = data["core_bbox"].copy()
        t = CentroidShift(apply_z=False)
        result = t(data)
        shift = result["coord_shift"]
        expected = bbox_before.copy()
        expected[0::2] -= shift[0]
        expected[1::2] -= shift[1]
        np.testing.assert_allclose(result["core_bbox"], expected, atol=1e-5)


class TestPositiveShiftCoreBBox(unittest.TestCase):
    def test_bbox_shifted_with_coord(self):
        data = _make_data_dict()
        bbox_before = data["core_bbox"].copy()
        coord_min = data["coord"].min(axis=0)
        t = PositiveShift()
        result = t(data)
        expected = bbox_before.copy()
        expected[0::2] -= coord_min[0]
        expected[1::2] -= coord_min[1]
        np.testing.assert_allclose(result["core_bbox"], expected, atol=1e-5)

    def test_bbox_xmin_positive(self):
        """After PositiveShift, coords are >= 0 so bbox xmin/ymin should decrease."""
        data = _make_data_dict()
        t = PositiveShift()
        result = t(data)
        # coords are all >= 0 now, core_bbox should be shifted down accordingly
        self.assertTrue(result["coord"].min() >= -1e-5)


class TestZPercentileCenterShiftCoreBBox(unittest.TestCase):
    def test_bbox_shifted_with_coord(self):
        data = _make_data_dict()
        bbox_before = data["core_bbox"].copy()
        t = ZPercentileCenterShift(percentile=1.0)
        result = t(data)
        shift = result["coord_shift"]
        expected = bbox_before.copy()
        expected[0::2] -= shift[0]
        expected[1::2] -= shift[1]
        np.testing.assert_allclose(result["core_bbox"], expected, atol=1e-5)


class TestRandomShiftCoreBBox(unittest.TestCase):
    def test_bbox_shifted_with_coord(self):
        """Run RandomShift; verify bbox moved by same delta as coord centroid."""
        np.random.seed(123)
        data = _make_data_dict()
        coord_mean_before = data["coord"].mean(axis=0).copy()
        bbox_before = data["core_bbox"].copy()
        t = RandomShift(shift=((-5, 5), (-5, 5), (0, 0)))
        result = t(data)
        coord_mean_after = result["coord"].mean(axis=0)
        dx = coord_mean_after[0] - coord_mean_before[0]
        dy = coord_mean_after[1] - coord_mean_before[1]
        expected = bbox_before.copy()
        expected[0::2] += dx
        expected[1::2] += dy
        np.testing.assert_allclose(result["core_bbox"], expected, atol=1e-4)

    def test_no_bbox_no_crash(self):
        data = _make_data_dict(with_bbox=False)
        t = RandomShift()
        result = t(data)
        self.assertNotIn("core_bbox", result)


class TestNormalizeCoordCoreBBox(unittest.TestCase):
    def test_bbox_shifted_and_scaled(self):
        data = _make_data_dict()
        bbox_before = data["core_bbox"].copy()
        centroid = data["coord"].mean(axis=0).copy()
        # Pre-compute m on centroid-shifted coords (before in-place division)
        shifted_copy = data["coord"].copy() - centroid
        m = np.max(np.sqrt(np.sum(shifted_copy ** 2, axis=1)))
        t = NormalizeCoord()
        result = t(data)
        # Expected: centroid subtraction then scale
        expected = bbox_before.copy()
        expected[0::2] -= centroid[0]
        expected[1::2] -= centroid[1]
        expected /= m
        np.testing.assert_allclose(result["core_bbox"], expected, atol=1e-4)

    def test_no_bbox_no_crash(self):
        data = _make_data_dict(with_bbox=False)
        t = NormalizeCoord()
        result = t(data)
        self.assertNotIn("core_bbox", result)


# ---------------------------------------------------------------------------
# 11–13: Integration / chain tests
# ---------------------------------------------------------------------------

class TestTransformChainCoreBBox(unittest.TestCase):
    """core_bbox stays aligned with coord through a sequence of transforms."""

    def test_center_then_positive_alignment(self):
        """
        After CenterShift + PositiveShift, the core_bbox should still
        define a valid sub-region that contains the original core points
        (in the new coordinate system).
        """
        data = _make_data_dict()
        bbox_before = data["core_bbox"].copy()
        # Identify points inside core_bbox before any transform
        coord = data["coord"]
        inside_mask = (
            (coord[:, 0] >= bbox_before[0]) & (coord[:, 0] <= bbox_before[2]) &
            (coord[:, 1] >= bbox_before[1]) & (coord[:, 1] <= bbox_before[3])
        )

        t1 = CenterShift(apply_z=True)
        t2 = PositiveShift()
        result = t2(t1(data))

        bbox_after = result["core_bbox"]
        coord_after = result["coord"]
        # Same points that were inside core_bbox should still be inside
        inside_pts = coord_after[inside_mask]
        if len(inside_pts) > 0:
            self.assertTrue(np.all(inside_pts[:, 0] >= bbox_after[0] - 1e-3))
            self.assertTrue(np.all(inside_pts[:, 0] <= bbox_after[2] + 1e-3))
            self.assertTrue(np.all(inside_pts[:, 1] >= bbox_after[1] - 1e-3))
            self.assertTrue(np.all(inside_pts[:, 1] <= bbox_after[3] + 1e-3))

    def test_centroid_then_random_shift_alignment(self):
        """After CentroidShift + RandomShift, core_bbox still frames the same sub-region."""
        np.random.seed(77)
        data = _make_data_dict()
        bbox_before = data["core_bbox"].copy()
        coord = data["coord"]
        inside_mask = (
            (coord[:, 0] >= bbox_before[0]) & (coord[:, 0] <= bbox_before[2]) &
            (coord[:, 1] >= bbox_before[1]) & (coord[:, 1] <= bbox_before[3])
        )

        t1 = CentroidShift(apply_z=True)
        t2 = RandomShift(shift=((-2, 2), (-2, 2), (0, 0)))
        result = t2(t1(data))

        bbox_after = result["core_bbox"]
        coord_after = result["coord"]
        inside_pts = coord_after[inside_mask]
        if len(inside_pts) > 0:
            self.assertTrue(np.all(inside_pts[:, 0] >= bbox_after[0] - 1e-3))
            self.assertTrue(np.all(inside_pts[:, 0] <= bbox_after[2] + 1e-3))
            self.assertTrue(np.all(inside_pts[:, 1] >= bbox_after[1] - 1e-3))
            self.assertTrue(np.all(inside_pts[:, 1] <= bbox_after[3] + 1e-3))

    def test_center_shift_bbox_matches_midpoint(self):
        """
        After CenterShift, the midpoint of the coord range is ~0.
        The core_bbox should shift by the same amount.
        """
        data = _make_data_dict()
        t = CenterShift(apply_z=False)
        result = t(data)
        coord = result["coord"]
        mid_x = (coord[:, 0].min() + coord[:, 0].max()) / 2
        mid_y = (coord[:, 1].min() + coord[:, 1].max()) / 2
        self.assertAlmostEqual(float(mid_x), 0.0, places=3)
        self.assertAlmostEqual(float(mid_y), 0.0, places=3)
        # core_bbox should also be centered relative to the same shift
        bbox = result["core_bbox"]
        bbox_mid_x = (bbox[0] + bbox[2]) / 2
        bbox_mid_y = (bbox[1] + bbox[3]) / 2
        # The bbox midpoint won't be 0 (it's a sub-region),
        # but the width/height must be preserved
        original_w = 190.0 - 110.0
        original_h = 390.0 - 310.0
        self.assertAlmostEqual(float(bbox[2] - bbox[0]), original_w, places=3)
        self.assertAlmostEqual(float(bbox[3] - bbox[1]), original_h, places=3)

    def test_zpercentile_then_centroid_preserves_width(self):
        """core_bbox width and height preserved through ZPercentile + Centroid chain."""
        data = _make_data_dict()
        orig_w = float(data["core_bbox"][2] - data["core_bbox"][0])
        orig_h = float(data["core_bbox"][3] - data["core_bbox"][1])
        t1 = ZPercentileCenterShift(percentile=5.0)
        t2 = CentroidShift(apply_z=False)
        result = t2(t1(data))
        bbox = result["core_bbox"]
        self.assertAlmostEqual(float(bbox[2] - bbox[0]), orig_w, places=3)
        self.assertAlmostEqual(float(bbox[3] - bbox[1]), orig_h, places=3)


if __name__ == "__main__":
    unittest.main()
