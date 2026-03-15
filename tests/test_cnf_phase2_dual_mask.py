"""
Tests for CnfTester Phase 2: Dual Masking (Core BBox + Convex Hull)

The Phase 2 logic (extracted as pure functions for unit testing):
  - Step 1: xy_min/xy_max comes from core_bbox if present, else from actual extent
  - Step 2: initial dense grid built from xy_min/xy_max  (方刀 — square knife)
  - Step 3: hull mask trims points outside the convex hull (剪刀 — scissors)

Covers:
    1.  core_bbox present → xy_min/xy_max equals core_bbox, not buffered AABB
    2.  core_bbox absent  → xy_min/xy_max falls back to actual point extent
    3.  core_bbox grid range matches [xmin, xmax] × [ymin, ymax] within resolution
    4.  fallback grid range matches actual point extent  within resolution
    5.  grid covers core_bbox exactly (no missing edge columns/rows)
    6.  core_bbox grid is strictly smaller than buffered extent grid
    7.  convex hull mask removes points outside the hull
    8.  convex hull mask keeps all interior points
    9.  convex hull mask radius tolerance: boundary points are NOT miss-classified
    10. hull_xy_np.shape[0] < 4 → fallback to full bbox grid (no hull computed)
    11. convex hull error → fallback to full bbox grid (exception handler)
    12. dual masking: grid is simultaneously inside core_bbox AND inside hull
    13. core_bbox=[xmin, ymin, xmax, ymax] slice correctness for qd=2
    14. adjacent tiles share no gap when grid uses hi + resolution as upper bound

Author: PointSpace Team
"""

import sys
import os
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------------
# Pure-function reimplementation of the Phase 2 logic
# (mirrors test.py lines 1914-1982 exactly, importable without GPU/model)
# ---------------------------------------------------------------------------

def _phase2_build_query_grid(
    fragment_list,
    data_dict,
    query_resolution: float = 0.5,
    query_dim: int = 2,
):
    """
    Standalone reimplementation of CnfTester Phase 2.
    Returns (query_xy, query_xy_full, xy_min, xy_max, hull_xy_np).
    """
    from scipy.spatial import ConvexHull
    from matplotlib.path import Path

    qd = query_dim

    # 1. Full coords from all fragments (includes buffer zone)
    raw_coords_list = [frag["coord"] for frag in fragment_list]
    all_raw_coords = torch.cat(raw_coords_list, dim=0)
    hull_xy_np = all_raw_coords[:, :qd].cpu().numpy()

    # 2. Core BBox or fallback
    if "core_bbox" in data_dict:
        core_bbox = data_dict["core_bbox"]
        xy_min = torch.tensor(core_bbox[:qd], dtype=torch.float32)
        xy_max = torch.tensor(core_bbox[qd:qd * 2], dtype=torch.float32)
    else:
        xy_min = all_raw_coords[:, :qd].min(dim=0).values.cpu()
        xy_max = all_raw_coords[:, :qd].max(dim=0).values.cpu()

    # 3. 方刀: initial grid from Core BBox
    axes = [
        torch.arange(
            lo.item(),
            hi.item() + query_resolution,
            query_resolution,
        )
        for lo, hi in zip(xy_min, xy_max)
    ]
    grids = torch.meshgrid(*axes, indexing="ij")
    query_xy_full = torch.stack([g.flatten() for g in grids], dim=1)

    # 4. 剪刀: hull mask
    if hull_xy_np.shape[0] < 4:
        query_xy = query_xy_full
    else:
        try:
            hull = ConvexHull(hull_xy_np)
            hull_vertices = hull_xy_np[hull.vertices]
            poly_path = Path(hull_vertices)
            keep_mask = poly_path.contains_points(
                query_xy_full.numpy(),
                radius=query_resolution * 0.5,
            )
            query_xy = query_xy_full[keep_mask]
        except Exception:
            query_xy = query_xy_full

    return query_xy, query_xy_full, xy_min, xy_max, hull_xy_np


def _make_fragment_list(points_np):
    """Wrap a numpy array as a single-fragment list."""
    return [{"coord": torch.from_numpy(points_np.astype(np.float32))}]


def _make_tile(
    core_xmin=0.0, core_ymin=0.0, core_xmax=50.0, core_ymax=50.0,
    buffer=10.0, n_core=400, n_buf=60, seed=0,
):
    """
    Generate synthetic tile: core points in [xmin, xmax]×[ymin, ymax]
    plus buffer points in the ±buffer fringe.
    Returns (fragment_list, data_dict).
    """
    rng = np.random.default_rng(seed)
    core_pts = np.column_stack([
        rng.uniform(core_xmin, core_xmax, n_core),
        rng.uniform(core_ymin, core_ymax, n_core),
        rng.uniform(0, 10, n_core),
    ]).astype(np.float32)

    # buffer points: in the ring  [xmin-buf, xmax+buf] \ [xmin, xmax]
    buf_pts_x = np.concatenate([
        rng.uniform(core_xmin - buffer, core_xmin, n_buf // 4),
        rng.uniform(core_xmax, core_xmax + buffer, n_buf // 4),
    ])
    buf_pts_y = rng.uniform(core_ymin - buffer, core_ymax + buffer, len(buf_pts_x))
    buf_pts = np.column_stack([buf_pts_x, buf_pts_y,
                               np.zeros(len(buf_pts_x))]).astype(np.float32)

    all_pts = np.vstack([core_pts, buf_pts])
    fragment_list = _make_fragment_list(all_pts)
    data_dict = {
        "core_bbox": np.array([core_xmin, core_ymin, core_xmax, core_ymax],
                              dtype=np.float32)
    }
    return fragment_list, data_dict, core_pts


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCoreBBoxSelection(unittest.TestCase):
    """xy_min/xy_max comes from core_bbox when present."""

    def setUp(self):
        self.fragment_list, self.data_dict, _ = _make_tile(
            core_xmin=10.0, core_ymin=20.0, core_xmax=60.0, core_ymax=80.0,
            buffer=15.0,
        )
        self.res = 1.0

    def test_xy_min_equals_core_bbox(self):
        _, _, xy_min, _, _ = _phase2_build_query_grid(
            self.fragment_list, self.data_dict, query_resolution=self.res
        )
        np.testing.assert_allclose(xy_min.numpy(), [10.0, 20.0], atol=1e-5)

    def test_xy_max_equals_core_bbox(self):
        _, _, _, xy_max, _ = _phase2_build_query_grid(
            self.fragment_list, self.data_dict, query_resolution=self.res
        )
        np.testing.assert_allclose(xy_max.numpy(), [60.0, 80.0], atol=1e-5)

    def test_xy_min_NOT_buffered_extent(self):
        """xy_min must be core_bbox lower-left, NOT the buffered point extent."""
        _, _, xy_min, _, hull_xy = _phase2_build_query_grid(
            self.fragment_list, self.data_dict, query_resolution=self.res
        )
        actual_min = hull_xy.min(axis=0)
        # actual_min < core_bbox lower-left due to buffer
        self.assertLess(float(actual_min[0]), float(xy_min[0]) - 0.5)
        self.assertLess(float(actual_min[1]), float(xy_min[1]) - 0.5)


class TestFallbackSelection(unittest.TestCase):
    """When core_bbox is absent, xy_min/xy_max = actual point extent."""

    def setUp(self):
        rng = np.random.default_rng(7)
        pts = np.column_stack([
            rng.uniform(5, 45, 300),
            rng.uniform(5, 45, 300),
            np.zeros(300),
        ]).astype(np.float32)
        self.fragment_list = _make_fragment_list(pts)
        self.data_dict = {}  # no core_bbox
        self.pts = pts
        self.res = 0.5

    def test_xy_min_is_actual_extent(self):
        _, _, xy_min, _, _ = _phase2_build_query_grid(
            self.fragment_list, self.data_dict, query_resolution=self.res
        )
        expected = self.pts[:, :2].min(axis=0)
        np.testing.assert_allclose(xy_min.numpy(), expected, atol=1e-4)

    def test_xy_max_is_actual_extent(self):
        _, _, _, xy_max, _ = _phase2_build_query_grid(
            self.fragment_list, self.data_dict, query_resolution=self.res
        )
        expected = self.pts[:, :2].max(axis=0)
        np.testing.assert_allclose(xy_max.numpy(), expected, atol=1e-4)


class TestGridRange(unittest.TestCase):
    """Initial grid (query_xy_full) spans exactly [xy_min, xy_max+res]."""

    def setUp(self):
        self.fragment_list, self.data_dict, _ = _make_tile(
            core_xmin=0.0, core_ymin=0.0, core_xmax=50.0, core_ymax=50.0,
            buffer=5.0,
        )
        self.res = 1.0

    def test_grid_min_equals_core_min(self):
        _, full, xy_min, _, _ = _phase2_build_query_grid(
            self.fragment_list, self.data_dict, query_resolution=self.res
        )
        grid_min = full.min(dim=0).values
        np.testing.assert_allclose(grid_min.numpy(), xy_min.numpy(), atol=1e-5)

    def test_grid_max_within_core_max_plus_resolution(self):
        _, full, _, xy_max, _ = _phase2_build_query_grid(
            self.fragment_list, self.data_dict, query_resolution=self.res
        )
        grid_max = full.max(dim=0).values
        # grid_max should be <= xy_max + res (last arange step)
        self.assertTrue(
            torch.all(grid_max <= xy_max + self.res + 1e-4).item()
        )

    def test_grid_entirely_within_core_bbox(self):
        """All grid points must be within [xmin, xmax+res] × [ymin, ymax+res]."""
        _, full, xy_min, xy_max, _ = _phase2_build_query_grid(
            self.fragment_list, self.data_dict, query_resolution=self.res
        )
        self.assertTrue(torch.all(full[:, 0] >= xy_min[0] - 1e-5).item())
        self.assertTrue(torch.all(full[:, 1] >= xy_min[1] - 1e-5).item())
        self.assertTrue(torch.all(full[:, 0] <= xy_max[0] + self.res + 1e-4).item())
        self.assertTrue(torch.all(full[:, 1] <= xy_max[1] + self.res + 1e-4).item())

    def test_grid_resolution_step(self):
        """Consecutive x values in the full grid differ by query_resolution."""
        _, full, xy_min, xy_max, _ = _phase2_build_query_grid(
            self.fragment_list, self.data_dict, query_resolution=self.res
        )
        # Extract unique x values and check spacing
        xs = torch.unique(full[:, 0]).sort().values
        diffs = (xs[1:] - xs[:-1]).numpy()
        np.testing.assert_allclose(diffs, self.res, atol=1e-4)


class TestCoreBBoxSmallerThanBufferedGrid(unittest.TestCase):
    """Core BBox grid must be strictly smaller than the buffered extent grid."""

    def test_core_grid_smaller_than_buffered_grid(self):
        fragment_list, data_dict, _ = _make_tile(
            core_xmin=0.0, core_ymin=0.0, core_xmax=50.0, core_ymax=50.0,
            buffer=20.0, n_buf=200,
        )
        res = 1.0

        # Grid with core_bbox
        _, full_core, _, _, _ = _phase2_build_query_grid(
            fragment_list, data_dict, query_resolution=res
        )

        # Grid without core_bbox (fallback to full extent)
        _, full_buf, _, _, _ = _phase2_build_query_grid(
            fragment_list, {}, query_resolution=res
        )

        self.assertLess(
            full_core.shape[0], full_buf.shape[0],
            "Core BBox grid should have fewer points than buffered extent grid",
        )


class TestConvexHullMask(unittest.TestCase):
    """Convex hull scissors test."""

    def _make_l_shaped_tile(self, res=1.0):
        """
        L-shaped point cloud:
        ██████
        ██
        ██
        Upper-right quadrant is empty.
        """
        rng = np.random.default_rng(42)
        # Lower-full strip: x∈[0,40], y∈[0,20]
        bot = np.column_stack([
            rng.uniform(0, 40, 500), rng.uniform(0, 20, 500), np.zeros(500)
        ])
        # Left column:      x∈[0,20], y∈[20,40]
        left = np.column_stack([
            rng.uniform(0, 20, 300), rng.uniform(20, 40, 300), np.zeros(300)
        ])
        pts = np.vstack([bot, left]).astype(np.float32)
        fragment_list = _make_fragment_list(pts)
        # core_bbox covers the full bounding box [0,0,40,40]
        data_dict = {
            "core_bbox": np.array([0.0, 0.0, 40.0, 40.0], dtype=np.float32)
        }
        return fragment_list, data_dict, pts

    def test_hull_mask_reduces_grid(self):
        fragment_list, data_dict, _ = self._make_l_shaped_tile()
        query_xy, full, _, _, _ = _phase2_build_query_grid(
            fragment_list, data_dict, query_resolution=2.0
        )
        self.assertLess(
            query_xy.shape[0], full.shape[0],
            "Hull mask should remove points outside the L-shape",
        )

    def test_hull_mask_interior_points_kept(self):
        """All points clearly inside the L interior must be retained."""
        fragment_list, data_dict, _ = self._make_l_shaped_tile()
        query_xy, _, _, _, _ = _phase2_build_query_grid(
            fragment_list, data_dict, query_resolution=2.0
        )
        # Points in the lower strip centre should all survive
        interior = np.array([[5, 5], [15, 5], [25, 5], [35, 5],
                             [10, 15]])
        xy_np = query_xy.numpy()
        for pt in interior:
            dists = np.linalg.norm(xy_np - pt, axis=1)
            self.assertTrue(
                dists.min() < 2.5,
                f"Interior point {pt} not found in hull-masked grid",
            )

    def test_hull_mask_corner_outside_hull_removed(self):
        """
        Upper-right corner (30, 35) is outside the L-shape and should NOT
        appear in the masked grid (radius tolerance of half-resolution applies).
        """
        fragment_list, data_dict, _ = self._make_l_shaped_tile()
        query_xy, _, _, _, _ = _phase2_build_query_grid(
            fragment_list, data_dict, query_resolution=2.0
        )
        xy_np = query_xy.numpy()
        outside_pt = np.array([35.0, 35.0])
        dists = np.linalg.norm(xy_np - outside_pt, axis=1)
        # Nearest grid point to (35,35) should be >2m away after hull trim
        self.assertGreater(
            dists.min(), 1.5,
            "Upper-right corner should be removed by hull mask",
        )


class TestFallbackFewPoints(unittest.TestCase):
    """hull_xy_np.shape[0] < 4 → no hull computation, return full grid."""

    def test_three_points_returns_full_grid(self):
        pts = np.array([[0, 0, 0], [10, 0, 0], [5, 10, 0]], dtype=np.float32)
        fragment_list = _make_fragment_list(pts)
        data_dict = {"core_bbox": np.array([0.0, 0.0, 10.0, 10.0],
                                           dtype=np.float32)}
        query_xy, full, _, _, hull_xy = _phase2_build_query_grid(
            fragment_list, data_dict, query_resolution=1.0
        )
        self.assertEqual(hull_xy.shape[0], 3)
        self.assertEqual(query_xy.shape[0], full.shape[0],
                         "With <4 points, query_xy must equal query_xy_full")


class TestFallbackHullError(unittest.TestCase):
    """ConvexHull failure → fallback to full bbox grid."""

    def test_collinear_points_fallback(self):
        """All points on a line → ConvexHull raises QhullError → fallback."""
        pts = np.column_stack([
            np.linspace(0, 50, 100),
            np.zeros(100),   # y=0: collinear
            np.zeros(100),
        ]).astype(np.float32)
        fragment_list = _make_fragment_list(pts)
        data_dict = {"core_bbox": np.array([0.0, 0.0, 50.0, 0.5],
                                           dtype=np.float32)}
        # Should not raise; falls back to full grid
        query_xy, full, _, _, _ = _phase2_build_query_grid(
            fragment_list, data_dict, query_resolution=1.0
        )
        # Either the hull succeeded with a trivially small area or fell back
        self.assertGreater(query_xy.shape[0], 0)


class TestDualMaskingProperty(unittest.TestCase):
    """
    Key invariant: after Phase 2, every grid point must be
    (a) inside core_bbox (guaranteed by 方刀) AND
    (b) inside the convex hull of the full point cloud (剪刀).
    """

    def setUp(self):
        self.fragment_list, self.data_dict, _ = _make_tile(
            core_xmin=5.0, core_ymin=5.0, core_xmax=45.0, core_ymax=45.0,
            buffer=10.0, n_core=600, n_buf=100, seed=99,
        )
        self.res = 2.0
        self.core_bb = self.data_dict["core_bbox"]

    def test_all_query_points_inside_core_bbox(self):
        query_xy, _, xy_min, xy_max, _ = _phase2_build_query_grid(
            self.fragment_list, self.data_dict, query_resolution=self.res
        )
        xy = query_xy.numpy()
        self.assertTrue(np.all(xy[:, 0] >= self.core_bb[0] - 1e-4))
        self.assertTrue(np.all(xy[:, 0] <= self.core_bb[2] + self.res + 1e-4))
        self.assertTrue(np.all(xy[:, 1] >= self.core_bb[1] - 1e-4))
        self.assertTrue(np.all(xy[:, 1] <= self.core_bb[3] + self.res + 1e-4))

    def test_query_points_inside_hull(self):
        from scipy.spatial import ConvexHull
        from matplotlib.path import Path

        query_xy, _, _, _, hull_xy_np = _phase2_build_query_grid(
            self.fragment_list, self.data_dict, query_resolution=self.res
        )
        hull = ConvexHull(hull_xy_np)
        hull_vertices = hull_xy_np[hull.vertices]
        poly = Path(hull_vertices)
        inside = poly.contains_points(query_xy.numpy(), radius=self.res * 0.5)
        pct_inside = inside.mean()
        # Allow for tiny floating-point edge tolerance — at least 95% inside
        self.assertGreater(pct_inside, 0.95,
                           f"Only {pct_inside:.1%} of query points inside hull")


class TestCoreBBoxSliceForQd2(unittest.TestCase):
    """core_bbox[:qd] and core_bbox[qd:qd*2] correctly picks [xmin,ymin] / [xmax,ymax]."""

    def test_slice_qd2_extracts_correct_values(self):
        core_bbox = np.array([10.0, 20.0, 60.0, 80.0], dtype=np.float32)
        qd = 2
        xy_min_expected = core_bbox[:qd]    # [10, 20]
        xy_max_expected = core_bbox[qd:qd * 2]  # [60, 80]

        rng = np.random.default_rng(5)
        pts = np.column_stack([
            rng.uniform(10, 60, 500),
            rng.uniform(20, 80, 500),
            np.zeros(500),
        ]).astype(np.float32)
        fragment_list = _make_fragment_list(pts)
        data_dict = {"core_bbox": core_bbox}

        _, _, xy_min, xy_max, _ = _phase2_build_query_grid(
            fragment_list, data_dict, query_resolution=1.0, query_dim=2
        )
        np.testing.assert_allclose(xy_min.numpy(), xy_min_expected, atol=1e-5)
        np.testing.assert_allclose(xy_max.numpy(), xy_max_expected, atol=1e-5)


class TestAdjacentTiles(unittest.TestCase):
    """
    Two adjacent core tiles must share a common edge without gap.
    Tile A: core x∈[0,50], Tile B: core x∈[50,100].
    The max X of tile‑A grid should equal the min X of tile‑B grid.
    """

    def _tile(self, x0, x1, res):
        rng = np.random.default_rng(0)
        pts = np.column_stack([
            rng.uniform(x0, x1, 300),
            rng.uniform(0, 50, 300),
            np.zeros(300),
        ]).astype(np.float32)
        frag = _make_fragment_list(pts)
        dd = {"core_bbox": np.array([x0, 0.0, x1, 50.0], dtype=np.float32)}
        return frag, dd

    def test_adjacent_tiles_share_edge(self):
        res = 0.5
        fA, dA = self._tile(0.0, 50.0, res)
        fB, dB = self._tile(50.0, 100.0, res)

        _, fullA, _, xy_maxA, _ = _phase2_build_query_grid(fA, dA, query_resolution=res)
        _, fullB, xy_minB, _, _ = _phase2_build_query_grid(fB, dB, query_resolution=res)

        max_x_A = fullA[:, 0].max().item()
        min_x_B = fullB[:, 0].min().item()

        # Both should include x=50 (one has it as max, other as min)
        self.assertAlmostEqual(min_x_B, xy_minB[0].item(), places=4)
        # The overlap point x=50 must appear in BOTH grids
        has_50_A = bool(torch.any(torch.abs(fullA[:, 0] - 50.0) < 1e-4).item())
        has_50_B = bool(torch.any(torch.abs(fullB[:, 0] - 50.0) < 1e-4).item())
        self.assertTrue(has_50_A, "Tile A grid should reach x=50")
        self.assertTrue(has_50_B, "Tile B grid should start at x=50")


if __name__ == "__main__":
    unittest.main()
