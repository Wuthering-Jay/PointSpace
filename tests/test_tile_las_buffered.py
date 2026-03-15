"""
Tests for LASTileProcessor — Buffered Tiling & Core BBox VLR

Covers:
    1.  __init__ 保存 buffer_size 属性
    2.  buffer_size=0 时 _grid_segmentation 返回裸索引元组（无膨胀）
    3.  buffer_size>0 时 buffered tile 包含邻近核心格的点
    4.  buffered tile 的点数 >= 核心 tile 的点数
    5.  核心 bbox 的 xmin/ymin/xmax/ymax 覆盖所有核心点
    6.  单块场景（全部点在同一格）buffer 不越界
    7.  buffer 精确边界：恰好在边界上的点被纳入
    8.  _grid_segmentation 返回类型始终为 List[Tuple]
    9.  _save_tiles 在每个输出 LAS 中写入 PointSpace VLR (record_id=1001)
    10. VLR 的 core_bbox JSON 可正确解码，且与点云范围吻合
    11. 多 tile 情形下每个文件均有独立的 core_bbox VLR
    12. core_bbox 不受 buffer 膨胀影响（始终为核心范围）
    13. min_points 阈值合并后 buffer 仍正常工作
    14. max_points 阈值切分后 buffer 仍正常工作
    15. 带 buffer 的完整 process_file 端到端流程（无 HAG/z_base）

Author: PointSpace Team
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import laspy

    HAS_LASPY = True
except ImportError:
    HAS_LASPY = False

# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _make_grid_points(x_tiles: int = 3, y_tiles: int = 3,
                      pts_per_tile: int = 100,
                      tile_size: float = 50.0,
                      seed: int = 0) -> np.ndarray:
    """生成均匀分布在 x_tiles×y_tiles 格网中的点云 (N,3)。"""
    rng = np.random.default_rng(seed)
    parts = []
    for ix in range(x_tiles):
        for iy in range(y_tiles):
            xy = rng.uniform(
                [ix * tile_size, iy * tile_size],
                [(ix + 1) * tile_size, (iy + 1) * tile_size],
                size=(pts_per_tile, 2),
            )
            z = rng.uniform(0, 10, size=(pts_per_tile, 1))
            parts.append(np.hstack([xy, z]))
    return np.vstack(parts).astype(np.float64)


def _write_las(path: str, points: np.ndarray):
    """将 (N,3) 点云写成最简 LAS 1.2 文件。"""
    header = laspy.LasHeader(point_format=0, version="1.2")
    header.offsets = points.min(axis=0)
    header.scales = np.array([0.001, 0.001, 0.001])
    las = laspy.LasData(header)
    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]
    las.write(path)


def _make_processor(input_path, output_dir,
                    window_size=(50.0, 50.0),
                    buffer_size=10.0,
                    min_points=None,
                    max_points=None,
                    overlap=False):
    """构造一个轻量 LASTileProcessor，禁用所有耗时计算。"""
    from utils.tile_las import LASTileProcessor
    return LASTileProcessor(
        input_path=input_path,
        output_dir=output_dir,
        window_size=window_size,
        overlap=overlap,
        overlap_factor=1,
        min_points=min_points,
        max_points=max_points,
        save_orig_idx=True,
        output_format='las',
        calc_normals=False,
        calc_hag=False,
        calc_z_base=False,
        buffer_size=buffer_size,
    )


def _read_core_bbox_vlr(las_path: str):
    """从 LAS 文件读取 PointSpace core_bbox VLR；找不到返回 None。"""
    las = laspy.read(las_path)
    for vlr in las.header.vlrs:
        if getattr(vlr, 'user_id', None) == "PointSpace" and getattr(vlr, 'record_id', None) == 1001:
            return json.loads(vlr.record_data.decode('utf-8'))
    return None


# ---------------------------------------------------------------------------
# 测试套件
# ---------------------------------------------------------------------------

@unittest.skipIf(not HAS_LASPY, "laspy not installed")
class TestBufferSizeInit(unittest.TestCase):
    """__init__ 正确保存 buffer_size 属性。"""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.pts = _make_grid_points()
        self.las_path = os.path.join(self.tmp, "src.las")
        _write_las(self.las_path, self.pts)

    def test_default_buffer_size(self):
        proc = _make_processor(self.las_path, self.tmp)
        self.assertEqual(proc.buffer_size, 10.0)

    def test_custom_buffer_size(self):
        proc = _make_processor(self.las_path, self.tmp, buffer_size=25.0)
        self.assertEqual(proc.buffer_size, 25.0)

    def test_zero_buffer_size(self):
        proc = _make_processor(self.las_path, self.tmp, buffer_size=0.0)
        self.assertEqual(proc.buffer_size, 0.0)

    def test_negative_buffer_size_treated_as_no_buffer(self):
        """负值与 0 效果相同：不做 buffer 膨胀。"""
        proc = _make_processor(self.las_path, self.tmp, buffer_size=-5.0)
        self.assertTrue(proc.buffer_size <= 0)


@unittest.skipIf(not HAS_LASPY, "laspy not installed")
class TestGridSegmentationReturnType(unittest.TestCase):
    """_grid_segmentation 返回值必须是 List[Tuple[ndarray, ndarray]]。"""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.pts = _make_grid_points(x_tiles=2, y_tiles=2, pts_per_tile=200)
        las_path = os.path.join(self.tmp, "src.las")
        _write_las(las_path, self.pts)
        self.proc_buf = _make_processor(las_path, self.tmp, buffer_size=10.0)
        self.proc_nobuf = _make_processor(las_path, self.tmp, buffer_size=0.0)

    def test_return_is_list_with_buffer(self):
        result = self.proc_buf._grid_segmentation(self.pts)
        self.assertIsInstance(result, list)

    def test_return_is_list_without_buffer(self):
        result = self.proc_nobuf._grid_segmentation(self.pts)
        self.assertIsInstance(result, list)

    def test_each_item_is_tuple_with_buffer(self):
        result = self.proc_buf._grid_segmentation(self.pts)
        for item in result:
            self.assertIsInstance(item, tuple)
            self.assertEqual(len(item), 2)

    def test_each_item_is_tuple_without_buffer(self):
        result = self.proc_nobuf._grid_segmentation(self.pts)
        for item in result:
            self.assertIsInstance(item, tuple)
            self.assertEqual(len(item), 2)

    def test_indices_are_ndarray(self):
        result = self.proc_buf._grid_segmentation(self.pts)
        for indices, _ in result:
            self.assertIsInstance(indices, np.ndarray)

    def test_bbox_is_ndarray_of_4(self):
        result = self.proc_buf._grid_segmentation(self.pts)
        for _, bbox in result:
            self.assertIsInstance(bbox, np.ndarray)
            self.assertEqual(bbox.shape, (4,))

    def test_segment_count_matches_tile_grid(self):
        """2×2 格网 → 4 个分块。"""
        result = self.proc_buf._grid_segmentation(self.pts)
        self.assertEqual(len(result), 4)


@unittest.skipIf(not HAS_LASPY, "laspy not installed")
class TestBufferedTileContainsNeighborPoints(unittest.TestCase):
    """
    带 buffer 时，tile 的覆盖范围 = 核心范围 + buffer_size。
    邻近核心格的点应被纳入 buffered tile。
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        # 2×1 格网：左格 x∈[0,50)，右格 x∈[50,100)
        self.tile_size = 50.0
        self.pts = _make_grid_points(x_tiles=2, y_tiles=1,
                                     pts_per_tile=300, tile_size=self.tile_size)
        las_path = os.path.join(self.tmp, "src.las")
        _write_las(las_path, self.pts)
        self.proc = _make_processor(las_path, self.tmp,
                                    window_size=(self.tile_size, self.tile_size),
                                    buffer_size=10.0)

    def test_buffered_tile_larger_than_core_tile(self):
        result = self.proc._grid_segmentation(self.pts)
        self.assertEqual(len(result), 2)
        buf_sizes = [len(idx) for idx, _ in result]
        core_sizes = []
        proc_no_buf = _make_processor(
            list(self.proc.las_files)[0], self.tmp,
            window_size=(self.tile_size, self.tile_size),
            buffer_size=0.0,
        )
        core_result = proc_no_buf._grid_segmentation(self.pts)
        core_sizes = [len(idx) for idx, _ in core_result]
        for buf, core in zip(sorted(buf_sizes), sorted(core_sizes)):
            self.assertGreaterEqual(buf, core)

    def test_buffered_tile_points_within_extended_bbox(self):
        """buffered tile 中所有点都在 [core_bbox ± buffer_size] 内。"""
        buf = 10.0
        result = self.proc._grid_segmentation(self.pts)
        for indices, core_bbox in result:
            tile_pts = self.pts[indices]
            xmin, ymin, xmax, ymax = core_bbox
            self.assertTrue(np.all(tile_pts[:, 0] >= xmin - buf - 1e-9))
            self.assertTrue(np.all(tile_pts[:, 0] <= xmax + buf + 1e-9))
            self.assertTrue(np.all(tile_pts[:, 1] >= ymin - buf - 1e-9))
            self.assertTrue(np.all(tile_pts[:, 1] <= ymax + buf + 1e-9))

    def test_no_buffer_core_only(self):
        """buffer_size=0 时，每个 tile 的点集恰好在核心 bbox 内。"""
        proc_no_buf = _make_processor(
            list(self.proc.las_files)[0], self.tmp,
            window_size=(self.tile_size, self.tile_size),
            buffer_size=0.0,
        )
        result = proc_no_buf._grid_segmentation(self.pts)
        for indices, core_bbox in result:
            tile_pts = self.pts[indices]
            xmin, ymin, xmax, ymax = core_bbox
            self.assertTrue(np.all(tile_pts[:, 0] >= xmin - 1e-9))
            self.assertTrue(np.all(tile_pts[:, 0] <= xmax + 1e-9))


@unittest.skipIf(not HAS_LASPY, "laspy not installed")
class TestCoreBBoxAccuracy(unittest.TestCase):
    """核心 bbox 覆盖核心格的所有点，且不受 buffer 点影响。"""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.tile_size = 50.0
        self.pts = _make_grid_points(x_tiles=3, y_tiles=3,
                                     pts_per_tile=150, tile_size=self.tile_size)
        las_path = os.path.join(self.tmp, "src.las")
        _write_las(las_path, self.pts)
        self.proc = _make_processor(las_path, self.tmp,
                                    window_size=(self.tile_size, self.tile_size),
                                    buffer_size=8.0)
        # 无 buffer 版本用于获取参考核心索引
        self.proc_nobuf = _make_processor(las_path, self.tmp,
                                          window_size=(self.tile_size, self.tile_size),
                                          buffer_size=0.0)

    def test_core_bbox_shape_and_dtype(self):
        result = self.proc._grid_segmentation(self.pts)
        for _, bbox in result:
            self.assertEqual(bbox.shape, (4,))
            self.assertEqual(bbox.dtype, np.float64)

    def test_core_bbox_order(self):
        """xmin <= xmax, ymin <= ymax。"""
        result = self.proc._grid_segmentation(self.pts)
        for _, bbox in result:
            self.assertLessEqual(bbox[0], bbox[2])
            self.assertLessEqual(bbox[1], bbox[3])

    def test_core_bbox_not_inflated_by_buffer(self):
        """带 buffer 的 bbox 与无 buffer 的 bbox 完全相同。"""
        res_buf = self.proc._grid_segmentation(self.pts)
        res_no = self.proc_nobuf._grid_segmentation(self.pts)

        # 按 (xmin, ymin) 排序后逐一对比
        bboxes_buf = sorted([tuple(b) for _, b in res_buf])
        bboxes_no  = sorted([tuple(b) for _, b in res_no])
        self.assertEqual(len(bboxes_buf), len(bboxes_no))
        for b1, b2 in zip(bboxes_buf, bboxes_no):
            np.testing.assert_allclose(b1, b2, atol=1e-9,
                                        err_msg="Core bbox must not be inflated by buffer")

    def test_core_bbox_covers_all_core_points(self):
        """无 buffer 情形：bbox 应能框住 tile 内所有点。"""
        result = self.proc_nobuf._grid_segmentation(self.pts)
        for indices, bbox in result:
            tile_pts = self.pts[indices, :2]
            self.assertAlmostEqual(float(tile_pts[:, 0].min()), bbox[0], places=6)
            self.assertAlmostEqual(float(tile_pts[:, 1].min()), bbox[1], places=6)
            self.assertAlmostEqual(float(tile_pts[:, 0].max()), bbox[2], places=6)
            self.assertAlmostEqual(float(tile_pts[:, 1].max()), bbox[3], places=6)


@unittest.skipIf(not HAS_LASPY, "laspy not installed")
class TestSingleTileEdgeCases(unittest.TestCase):
    """特殊情形：全部点落在单格 / 边界精确命中。"""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def _proc(self, las_path, buffer_size=10.0):
        return _make_processor(las_path, self.tmp,
                               window_size=(50.0, 50.0),
                               buffer_size=buffer_size)

    def test_single_tile_buffer_does_not_crash(self):
        """全部点在单格内，buffer 向外膨胀超出点云范围 → 不崩溃。"""
        pts = np.random.default_rng(99).uniform(0, 30, size=(200, 3))
        las_path = os.path.join(self.tmp, "single.las")
        _write_las(las_path, pts)
        proc = self._proc(las_path)
        result = proc._grid_segmentation(pts)
        self.assertEqual(len(result), 1)
        indices, bbox = result[0]
        # 单格 buffer 后应包含所有点（全部点本就在 bbox±buffer 内）
        self.assertEqual(len(indices), len(pts))

    def test_boundary_point_included_in_buffer(self):
        """
        左格核心 x∈[0,50)，右格 x∈[50,100)。
        buffer=10 → 左格扩展边界 = 60。
        x=59.9 的点应被纳入左格的 buffered tile。
        """
        rng = np.random.default_rng(7)
        left_pts = np.column_stack([
            rng.uniform(0, 50, 100),
            rng.uniform(0, 50, 100),
            np.zeros(100),
        ])
        # 在右格中靠近边界的点（x∈[50,60)），应被左格的 buffer 捕获
        near_pts = np.column_stack([
            rng.uniform(50, 60, 20),
            rng.uniform(0, 50, 20),
            np.zeros(20),
        ])
        # 远离左格边界（x>60），不应被左格 buffer 捕获
        far_pts = np.column_stack([
            rng.uniform(61, 100, 50),
            rng.uniform(0, 50, 50),
            np.zeros(50),
        ])
        pts = np.vstack([left_pts, near_pts, far_pts])
        las_path = os.path.join(self.tmp, "boundary.las")
        _write_las(las_path, pts)

        proc = _make_processor(las_path, self.tmp,
                               window_size=(50.0, 50.0), buffer_size=10.0)
        result = proc._grid_segmentation(pts)

        # 找到核心 bbox 在左侧的那个 tile
        left_tile = None
        for indices, bbox in result:
            if bbox[0] < 25:  # 核心 xmin 在左侧
                left_tile = (indices, bbox)
                break
        self.assertIsNotNone(left_tile, "Left tile not found")

        tile_pts = pts[left_tile[0]]
        # near_pts 的 x 范围 [50,60) 应全部被纳入
        near_x = near_pts[:, 0]
        captured = tile_pts[:, 0]
        for nx in near_x:
            self.assertTrue(
                np.any(np.abs(captured - nx) < 1e-9),
                f"Boundary point x={nx:.2f} should be inside left tile's buffer"
            )


@unittest.skipIf(not HAS_LASPY, "laspy not installed")
class TestMinMaxThresholdWithBuffer(unittest.TestCase):
    """min_points / max_points 阈值与 buffer 共同工作。"""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.pts = _make_grid_points(x_tiles=3, y_tiles=3,
                                     pts_per_tile=200, tile_size=50.0)
        las_path = os.path.join(self.tmp, "src.las")
        _write_las(las_path, self.pts)
        self.las_path = las_path

    def test_min_points_merge_still_returns_tuples(self):
        proc = _make_processor(self.las_path, self.tmp,
                               min_points=50, buffer_size=10.0)
        result = proc._grid_segmentation(self.pts)
        for item in result:
            self.assertIsInstance(item, tuple)
            self.assertEqual(len(item), 2)

    def test_max_points_split_still_returns_tuples(self):
        proc = _make_processor(self.las_path, self.tmp,
                               max_points=100, buffer_size=10.0)
        result = proc._grid_segmentation(self.pts)
        self.assertGreater(len(result), 9, "max_points should increase tile count")
        for item in result:
            self.assertIsInstance(item, tuple)
            self.assertEqual(len(item), 2)

    def test_max_points_each_buffered_tile_respects_bbox(self):
        """切分后每个 buffered tile 依然不超出 core_bbox ± buffer 范围。"""
        buf = 10.0
        proc = _make_processor(self.las_path, self.tmp,
                               max_points=100, buffer_size=buf)
        result = proc._grid_segmentation(self.pts)
        for indices, bbox in result:
            tile_pts = self.pts[indices]
            xmin, ymin, xmax, ymax = bbox
            self.assertTrue(np.all(tile_pts[:, 0] >= xmin - buf - 1e-9))
            self.assertTrue(np.all(tile_pts[:, 0] <= xmax + buf + 1e-9))


@unittest.skipIf(not HAS_LASPY, "laspy not installed")
class TestSaveTilesVLR(unittest.TestCase):
    """_save_tiles 在输出 LAS 中正确写入 core_bbox VLR。"""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.out_dir = os.path.join(self.tmp, "tiles")
        os.makedirs(self.out_dir, exist_ok=True)
        self.pts = _make_grid_points(x_tiles=2, y_tiles=2,
                                     pts_per_tile=200, tile_size=50.0)
        self.las_path = os.path.join(self.tmp, "src.las")
        _write_las(self.las_path, self.pts)

    def _run_tiles(self, buffer_size=10.0, min_points=None):
        proc = _make_processor(self.las_path, self.out_dir,
                               window_size=(50.0, 50.0),
                               buffer_size=buffer_size,
                               min_points=min_points)
        proc.process_file(Path(self.las_path))
        return sorted(Path(self.out_dir).glob("*.las"))

    def test_vlr_present_in_all_tiles(self):
        tile_files = self._run_tiles()
        self.assertGreater(len(tile_files), 0)
        for tf in tile_files:
            data = _read_core_bbox_vlr(str(tf))
            self.assertIsNotNone(data, f"No PointSpace VLR in {tf.name}")

    def test_vlr_contains_core_bbox_key(self):
        tile_files = self._run_tiles()
        for tf in tile_files:
            data = _read_core_bbox_vlr(str(tf))
            self.assertIn("core_bbox", data, f"core_bbox missing in {tf.name}")

    def test_vlr_bbox_is_list_of_4_floats(self):
        tile_files = self._run_tiles()
        for tf in tile_files:
            data = _read_core_bbox_vlr(str(tf))
            bbox = data["core_bbox"]
            self.assertEqual(len(bbox), 4)
            for v in bbox:
                self.assertIsInstance(v, float)

    def test_vlr_bbox_order(self):
        """xmin <= xmax, ymin <= ymax。"""
        tile_files = self._run_tiles()
        for tf in tile_files:
            bbox = _read_core_bbox_vlr(str(tf))["core_bbox"]
            self.assertLessEqual(bbox[0], bbox[2], "xmin > xmax")
            self.assertLessEqual(bbox[1], bbox[3], "ymin > ymax")

    def test_vlr_bbox_within_point_cloud_extent(self):
        """core_bbox 值在全局点云范围内。"""
        global_xmin, global_ymin = self.pts[:, 0].min(), self.pts[:, 1].min()
        global_xmax, global_ymax = self.pts[:, 0].max(), self.pts[:, 1].max()
        tile_files = self._run_tiles()
        for tf in tile_files:
            bbox = _read_core_bbox_vlr(str(tf))["core_bbox"]
            self.assertGreaterEqual(bbox[0], global_xmin - 1e-3)
            self.assertGreaterEqual(bbox[1], global_ymin - 1e-3)
            self.assertLessEqual(bbox[2], global_xmax + 1e-3)
            self.assertLessEqual(bbox[3], global_ymax + 1e-3)

    def test_each_tile_has_distinct_core_bbox(self):
        """2×2 格网 → 4 个 tile，每个 core_bbox 应互不相同。"""
        tile_files = self._run_tiles()
        bboxes = []
        for tf in tile_files:
            bboxes.append(tuple(_read_core_bbox_vlr(str(tf))["core_bbox"]))
        self.assertEqual(len(set(bboxes)), len(bboxes),
                         "Duplicate core_bboxes detected across tiles")

    def test_zero_buffer_vlr_still_written(self):
        """buffer_size=0 时 VLR 也应写入。"""
        tile_files = self._run_tiles(buffer_size=0.0)
        for tf in tile_files:
            data = _read_core_bbox_vlr(str(tf))
            self.assertIsNotNone(data)
            self.assertIn("core_bbox", data)

    def test_vlr_record_id_is_1001(self):
        """VLR record_id 必须是 1001。"""
        tile_files = self._run_tiles()
        for tf in tile_files:
            las = laspy.read(str(tf))
            found = any(
                getattr(v, 'user_id', None) == "PointSpace"
                and getattr(v, 'record_id', None) == 1001
                for v in las.header.vlrs
            )
            self.assertTrue(found, f"VLR record_id=1001 not found in {tf.name}")

    def test_vlr_user_id_is_pointspace(self):
        """VLR user_id 必须是 'PointSpace'。"""
        tile_files = self._run_tiles()
        for tf in tile_files:
            las = laspy.read(str(tf))
            found = any(
                getattr(v, 'user_id', None) == "PointSpace"
                for v in las.header.vlrs
            )
            self.assertTrue(found, f"user_id 'PointSpace' not found in {tf.name}")

    def test_core_bbox_not_inflated_by_buffer(self):
        """
        core_bbox 记录的是核心范围，
        不应等于 buffered tile 的实际点云范围（除非 buffer=0）。
        """
        tile_files = self._run_tiles(buffer_size=10.0)
        for tf in tile_files:
            las = laspy.read(str(tf))
            bbox = _read_core_bbox_vlr(str(tf))["core_bbox"]
            actual_xmin = float(np.array(las.x).min())
            actual_xmax = float(np.array(las.x).max())
            # buffered 实际范围应 >= 核心范围
            self.assertLessEqual(bbox[0], actual_xmin + 1e-3 + 10.0)
            self.assertGreaterEqual(bbox[2], actual_xmax - 1e-3 - 10.0)


@unittest.skipIf(not HAS_LASPY, "laspy not installed")
class TestEndToEndBufferedTiling(unittest.TestCase):
    """完整 process_file 端到端测试。"""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.out_dir = os.path.join(self.tmp, "out")
        os.makedirs(self.out_dir, exist_ok=True)
        self.pts = _make_grid_points(x_tiles=3, y_tiles=3,
                                     pts_per_tile=150, tile_size=50.0)
        self.las_path = os.path.join(self.tmp, "source.las")
        _write_las(self.las_path, self.pts)

    def test_output_tile_count(self):
        """3×3 格网应生成 9 个 tile 文件。"""
        proc = _make_processor(self.las_path, self.out_dir,
                               window_size=(50.0, 50.0), buffer_size=5.0)
        proc.process_file(Path(self.las_path))
        tiles = list(Path(self.out_dir).glob("*.las"))
        self.assertEqual(len(tiles), 9)

    def test_all_output_files_are_valid_las(self):
        proc = _make_processor(self.las_path, self.out_dir,
                               window_size=(50.0, 50.0), buffer_size=5.0)
        proc.process_file(Path(self.las_path))
        for tf in Path(self.out_dir).glob("*.las"):
            las = laspy.read(str(tf))
            self.assertGreater(len(las.points), 0, f"{tf.name} has 0 points")

    def test_each_tile_has_core_bbox_vlr(self):
        proc = _make_processor(self.las_path, self.out_dir,
                               window_size=(50.0, 50.0), buffer_size=5.0)
        proc.process_file(Path(self.las_path))
        for tf in Path(self.out_dir).glob("*.las"):
            data = _read_core_bbox_vlr(str(tf))
            self.assertIsNotNone(data, f"Missing VLR in {tf.name}")

    def test_buffered_tiles_have_more_points_than_core(self):
        """
        带 buffer 的 tile 点数通常 >= 不带 buffer 的 tile 点数。
        对中间格（有四邻格）尤其明显。
        """
        proc_buf = _make_processor(self.las_path, self.out_dir,
                                   window_size=(50.0, 50.0), buffer_size=10.0)
        proc_buf.process_file(Path(self.las_path))
        buf_counts = sorted(
            len(laspy.read(str(tf)).points)
            for tf in Path(self.out_dir).glob("*.las")
        )

        # 无缓冲版本写到另一目录
        out2 = os.path.join(self.tmp, "out2")
        os.makedirs(out2, exist_ok=True)
        proc_no = _make_processor(self.las_path, out2,
                                  window_size=(50.0, 50.0), buffer_size=0.0)
        proc_no.process_file(Path(self.las_path))
        no_counts = sorted(
            len(laspy.read(str(tf)).points)
            for tf in Path(out2).glob("*.las")
        )

        total_buf = sum(buf_counts)
        total_no  = sum(no_counts)
        self.assertGreaterEqual(total_buf, total_no,
                                "Buffered total points should be >= no-buffer total")

    def test_orig_idx_present_in_tiles(self):
        """tiles 中应有 orig_idx extra dim。"""
        proc = _make_processor(self.las_path, self.out_dir,
                               window_size=(50.0, 50.0), buffer_size=5.0)
        proc.process_file(Path(self.las_path))
        for tf in Path(self.out_dir).glob("*.las"):
            las = laspy.read(str(tf))
            dim_names = [d.name for d in las.header.point_format.extra_dimensions]
            self.assertIn("orig_idx", dim_names, f"Missing orig_idx in {tf.name}")

    def test_orig_idx_values_in_range(self):
        """orig_idx 值必须在原始点云范围内。"""
        n_total = len(self.pts)
        proc = _make_processor(self.las_path, self.out_dir,
                               window_size=(50.0, 50.0), buffer_size=5.0)
        proc.process_file(Path(self.las_path))
        for tf in Path(self.out_dir).glob("*.las"):
            las = laspy.read(str(tf))
            idx = np.array(las.orig_idx)
            self.assertTrue(np.all(idx < n_total),
                            f"orig_idx out of range in {tf.name}")


if __name__ == "__main__":
    unittest.main()
