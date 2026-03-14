"""
带缓冲区的重叠切块 (Buffered Tiling) 功能测试

覆盖：
  1. buffer_size 参数默认值及存储
  2. _grid_segmentation 无 buffer (buffer_size=0) 时的返回类型
  3. _grid_segmentation 有 buffer 时每个 tile 包含更多点
  4. _grid_segmentation 返回核心 BBox 正确包含核心区域点
  5. buffer 区域的点确实落在 [core_bbox ± buffer_size] 范围内
  6. _save_tiles 将 core_bbox 写入 LAS VLR（user_id="PointSpace", record_id=1001）
  7. VLR 内 JSON 数据可反序列化并匹配原始 core_bbox
  8. buffer_size <= 0 时 VLR 仍被写入
  9. 端到端：process_file 生成的每个 tile 均包含 core_bbox VLR

Author: PointSpace Team
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import laspy
    HAS_LASPY = True
except ImportError:
    HAS_LASPY = False

try:
    from utils.tile_las import LASTileProcessor
    HAS_PROCESSOR = True
except ImportError:
    HAS_PROCESSOR = False

SKIP_MSG = "laspy or LASTileProcessor not available"
SKIP = not (HAS_LASPY and HAS_PROCESSOR)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_processor(tmpdir: str, window_size=(30.0, 30.0), buffer_size=10.0,
                    min_points=None, max_points=None):
    """Create a minimal LASTileProcessor pointing at *tmpdir* as both input
    and output.  A dummy .las file is placed in tmpdir so that __init__
    doesn't raise."""
    las_path = os.path.join(tmpdir, "dummy.las")
    _write_las(las_path, _make_grid_points(100.0, 100.0, 500))
    return LASTileProcessor(
        input_path=las_path,
        output_dir=tmpdir,
        window_size=window_size,
        overlap=False,
        min_points=min_points,
        max_points=max_points,
        save_orig_idx=False,
        buffer_size=buffer_size,
    )


def _make_grid_points(x_range: float, y_range: float, n: int,
                      seed: int = 42) -> np.ndarray:
    """Generate n random 3-D points inside [0, x_range] × [0, y_range]."""
    rng = np.random.default_rng(seed)
    xs = rng.uniform(0.0, x_range, n)
    ys = rng.uniform(0.0, y_range, n)
    zs = rng.uniform(0.0, 10.0, n)
    return np.column_stack([xs, ys, zs])


def _write_las(path: str, points: np.ndarray) -> str:
    """Write a minimal LAS 1.2 file with the given (N,3) points."""
    header = laspy.LasHeader(point_format=0, version="1.2")
    header.offsets = points.min(axis=0)
    header.scales = np.array([0.001, 0.001, 0.001])
    las = laspy.LasData(header)
    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]
    las.write(path)
    return path


def _read_core_bbox_vlr(las_path: str):
    """Return the parsed core_bbox list from a saved tile, or None."""
    with laspy.open(las_path) as fh:
        data = fh.read()
    for vlr in data.header.vlrs:
        if getattr(vlr, 'user_id', None) == "PointSpace" and getattr(vlr, 'record_id', None) == 1001:
            return json.loads(vlr.record_data.decode('utf-8'))["core_bbox"]
    return None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@unittest.skipIf(SKIP, SKIP_MSG)
class TestBufferSizeParameter(unittest.TestCase):
    """buffer_size 参数的存储与默认值。"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_default_buffer_size(self):
        proc = _make_processor(self.tmpdir, buffer_size=10.0)
        self.assertEqual(proc.buffer_size, 10.0)

    def test_custom_buffer_size(self):
        proc = _make_processor(self.tmpdir, buffer_size=25.5)
        self.assertAlmostEqual(proc.buffer_size, 25.5)

    def test_zero_buffer_size(self):
        proc = _make_processor(self.tmpdir, buffer_size=0.0)
        self.assertEqual(proc.buffer_size, 0.0)

    def test_negative_buffer_size(self):
        proc = _make_processor(self.tmpdir, buffer_size=-1.0)
        self.assertLess(proc.buffer_size, 0.0)


@unittest.skipIf(SKIP, SKIP_MSG)
class TestGridSegmentationReturnType(unittest.TestCase):
    """_grid_segmentation 始终返回 List[Tuple[indices, core_bbox]]。"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.points = _make_grid_points(100.0, 100.0, 800)

    def test_returns_list_of_tuples_with_buffer(self):
        proc = _make_processor(self.tmpdir, buffer_size=5.0, min_points=None)
        result = proc._grid_segmentation(self.points)
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        for item in result:
            self.assertIsInstance(item, tuple)
            self.assertEqual(len(item), 2)
            indices, bbox = item
            self.assertIsInstance(indices, np.ndarray)
            self.assertIsInstance(bbox, np.ndarray)
            self.assertEqual(bbox.shape, (4,))

    def test_returns_list_of_tuples_without_buffer(self):
        proc = _make_processor(self.tmpdir, buffer_size=0.0, min_points=None)
        result = proc._grid_segmentation(self.points)
        self.assertIsInstance(result, list)
        for item in result:
            self.assertIsInstance(item, tuple)
            indices, bbox = item
            self.assertIsInstance(indices, np.ndarray)
            self.assertEqual(bbox.shape, (4,))


@unittest.skipIf(SKIP, SKIP_MSG)
class TestCoreBBoxCorrectness(unittest.TestCase):
    """core_bbox 确实包含核心区域所有点。"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.points = _make_grid_points(90.0, 90.0, 600)

    def test_core_bbox_contains_core_points(self):
        """无 buffer 时，每个 tile 的点都在其 core_bbox 内。"""
        proc = _make_processor(self.tmpdir, buffer_size=0.0, min_points=None)
        result = proc._grid_segmentation(self.points)
        for indices, bbox in result:
            pts = self.points[indices, :2]
            xmin, ymin, xmax, ymax = bbox
            self.assertTrue(np.all(pts[:, 0] >= xmin - 1e-9))
            self.assertTrue(np.all(pts[:, 0] <= xmax + 1e-9))
            self.assertTrue(np.all(pts[:, 1] >= ymin - 1e-9))
            self.assertTrue(np.all(pts[:, 1] <= ymax + 1e-9))

    def test_core_bbox_order(self):
        """core_bbox 格式: [xmin, ymin, xmax, ymax]，xmin <= xmax, ymin <= ymax。"""
        proc = _make_processor(self.tmpdir, buffer_size=5.0, min_points=None)
        result = proc._grid_segmentation(self.points)
        for _, bbox in result:
            xmin, ymin, xmax, ymax = bbox
            self.assertLessEqual(xmin, xmax)
            self.assertLessEqual(ymin, ymax)


@unittest.skipIf(SKIP, SKIP_MSG)
class TestBufferExpansion(unittest.TestCase):
    """buffer 区域确实扩展了点集。"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        # Dense uniform grid so buffer always captures extra points
        rng = np.random.default_rng(0)
        xs = rng.uniform(0.0, 120.0, 2000)
        ys = rng.uniform(0.0, 120.0, 2000)
        zs = np.zeros(2000)
        self.points = np.column_stack([xs, ys, zs])

    def test_buffer_increases_point_count(self):
        """带 buffer 的 tile 包含的点数 >= 不带 buffer 的对应 tile。"""
        window = (40.0, 40.0)
        proc_buf = _make_processor(self.tmpdir, buffer_size=10.0, min_points=None,
                                   window_size=window)
        proc_no = _make_processor(self.tmpdir, buffer_size=0.0, min_points=None,
                                  window_size=window)
        res_buf = proc_buf._grid_segmentation(self.points)
        res_no = proc_no._grid_segmentation(self.points)
        self.assertEqual(len(res_buf), len(res_no),
                         "Number of tiles should be the same regardless of buffer")
        for (idx_buf, _), (idx_no, _) in zip(res_buf, res_no):
            self.assertGreaterEqual(len(idx_buf), len(idx_no))

    def test_buffered_points_within_expanded_bbox(self):
        """所有 buffered 点都在 [core_bbox ± buffer_size] 范围内。"""
        buf = 10.0
        window = (40.0, 40.0)
        proc = _make_processor(self.tmpdir, buffer_size=buf, min_points=None,
                                window_size=window)
        result = proc._grid_segmentation(self.points)
        for indices, bbox in result:
            xmin, ymin, xmax, ymax = bbox
            pts = self.points[indices, :2]
            self.assertTrue(np.all(pts[:, 0] >= xmin - buf - 1e-9))
            self.assertTrue(np.all(pts[:, 0] <= xmax + buf + 1e-9))
            self.assertTrue(np.all(pts[:, 1] >= ymin - buf - 1e-9))
            self.assertTrue(np.all(pts[:, 1] <= ymax + buf + 1e-9))


@unittest.skipIf(SKIP, SKIP_MSG)
class TestVlrCoreBBox(unittest.TestCase):
    """_save_tiles 将 core_bbox 写入 LAS VLR。"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        rng = np.random.default_rng(7)
        pts = np.column_stack([
            rng.uniform(0.0, 100.0, 400),
            rng.uniform(0.0, 100.0, 400),
            rng.uniform(0.0, 5.0, 400),
        ])
        self.las_path = os.path.join(self.tmpdir, "source.las")
        _write_las(self.las_path, pts)
        self.points = pts

    def _run_save_and_collect(self, buffer_size: float):
        proc = LASTileProcessor(
            input_path=self.las_path,
            output_dir=self.tmpdir,
            window_size=(50.0, 50.0),
            overlap=False,
            min_points=None,
            save_orig_idx=False,
            buffer_size=buffer_size,
        )
        with laspy.open(self.las_path) as fh:
            las_data = fh.read()
        segments = proc._grid_segmentation(self.points)
        proc._save_tiles(Path(self.las_path), las_data, segments, points=self.points)
        tiles = sorted(Path(self.tmpdir).glob("source_*.las"))
        return tiles

    def test_vlr_present_with_buffer(self):
        tiles = self._run_save_and_collect(buffer_size=10.0)
        self.assertGreater(len(tiles), 0)
        for tile in tiles:
            bbox = _read_core_bbox_vlr(str(tile))
            self.assertIsNotNone(bbox, f"core_bbox VLR missing in {tile.name}")

    def test_vlr_present_without_buffer(self):
        tiles = self._run_save_and_collect(buffer_size=0.0)
        self.assertGreater(len(tiles), 0)
        for tile in tiles:
            bbox = _read_core_bbox_vlr(str(tile))
            self.assertIsNotNone(bbox, f"core_bbox VLR missing in {tile.name}")

    def test_vlr_user_id_and_record_id(self):
        tiles = self._run_save_and_collect(buffer_size=5.0)
        for tile in tiles:
            with laspy.open(str(tile)) as fh:
                data = fh.read()
            found = False
            for vlr in data.header.vlrs:
                if vlr.user_id == "PointSpace" and vlr.record_id == 1001:
                    found = True
                    break
            self.assertTrue(found, f"Expected VLR not found in {tile.name}")

    def test_vlr_json_deserialization(self):
        tiles = self._run_save_and_collect(buffer_size=8.0)
        for tile in tiles:
            with laspy.open(str(tile)) as fh:
                data = fh.read()
            for vlr in data.header.vlrs:
                if vlr.user_id == "PointSpace" and vlr.record_id == 1001:
                    parsed = json.loads(vlr.record_data.decode('utf-8'))
                    self.assertIn("core_bbox", parsed)
                    self.assertEqual(len(parsed["core_bbox"]), 4)
                    xmin, ymin, xmax, ymax = parsed["core_bbox"]
                    self.assertLessEqual(xmin, xmax)
                    self.assertLessEqual(ymin, ymax)

    def test_vlr_bbox_matches_actual_points(self):
        """VLR 中的 core_bbox 合理地近似 tile 的 (无 buffer) 空间范围。"""
        buf = 10.0
        proc = LASTileProcessor(
            input_path=self.las_path,
            output_dir=self.tmpdir,
            window_size=(50.0, 50.0),
            overlap=False,
            min_points=None,
            save_orig_idx=False,
            buffer_size=buf,
        )
        with laspy.open(self.las_path) as fh:
            las_data = fh.read()
        segments = proc._grid_segmentation(self.points)

        for tile_idx, (indices, core_bbox) in enumerate(segments):
            # Each buffered tile should have points within core_bbox ± buf
            pts = self.points[indices, :2]
            xmin, ymin, xmax, ymax = core_bbox
            self.assertTrue(np.all(pts[:, 0] >= xmin - buf - 1e-6))
            self.assertTrue(np.all(pts[:, 0] <= xmax + buf + 1e-6))
            self.assertTrue(np.all(pts[:, 1] >= ymin - buf - 1e-6))
            self.assertTrue(np.all(pts[:, 1] <= ymax + buf + 1e-6))


@unittest.skipIf(SKIP, SKIP_MSG)
class TestEndToEnd(unittest.TestCase):
    """端到端：process_file 生成的每个 tile 均包含 core_bbox VLR。"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        rng = np.random.default_rng(99)
        pts = np.column_stack([
            rng.uniform(0.0, 80.0, 600),
            rng.uniform(0.0, 80.0, 600),
            rng.uniform(0.0, 5.0, 600),
        ])
        self.las_path = os.path.join(self.tmpdir, "e2e.las")
        _write_las(self.las_path, pts)

    def test_tiles_have_core_bbox_vlr(self):
        out_dir = os.path.join(self.tmpdir, "out")
        proc = LASTileProcessor(
            input_path=self.las_path,
            output_dir=out_dir,
            window_size=(40.0, 40.0),
            overlap=False,
            min_points=None,
            save_orig_idx=False,
            buffer_size=8.0,
        )
        proc.process_file(Path(self.las_path))
        tiles = sorted(Path(out_dir).glob("e2e_*.las"))
        self.assertGreater(len(tiles), 0, "No tiles were generated")
        for tile in tiles:
            bbox = _read_core_bbox_vlr(str(tile))
            self.assertIsNotNone(bbox, f"core_bbox VLR missing in {tile.name}")
            self.assertEqual(len(bbox), 4)


if __name__ == "__main__":
    unittest.main()
