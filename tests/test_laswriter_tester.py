"""
LASWriter × Tester 联合测试

模拟 Tester 中 general_writer (LASWriter) 的完整调用链，覆盖：
  1. 有源文件模式 + pred_sem only（Tester 实际调用方式: coord=None, source_dir 提供坐标）
  2. 有源文件模式 + benchmark_writer 先变换 pred 后传给 general_writer
  3. 无源文件 + 有 coord 模式（直接创建 LAS）
  4. 无源文件 + coord=None → 应抛 ValueError
  5. build_writer() 构建 → write() 端到端流程
  6. 多样本连续写入（模拟 Tester for loop）
  7. Benchmark Writer + General Writer 同时启用

Author: PointSpace Team
"""

import os
import sys
import tempfile
import unittest
import warnings

import numpy as np

try:
    import laspy

    HAS_LASPY = True
except ImportError:
    HAS_LASPY = False

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pointspace.writers import build_writer, WRITERS
from pointspace.writers.las_writer import LASWriter
from pointspace.writers.benchmark import create_benchmark_writer


# ===========================================================================
#  辅助工具
# ===========================================================================


class _FakeDataset:
    """模拟数据集对象。"""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def _create_source_las(source_dir: str, name: str, n_points: int = 200):
    """在 source_dir 下创建一个 .las 源文件，返回 (文件路径, 点数)。"""
    header = laspy.LasHeader(point_format=2, version="1.2")
    np.random.seed(42)
    coords = np.random.rand(n_points, 3) * 100
    header.offsets = coords.min(axis=0)
    header.scales = np.array([0.001, 0.001, 0.001])
    las = laspy.LasData(header)
    las.x = coords[:, 0]
    las.y = coords[:, 1]
    las.z = coords[:, 2]
    path = os.path.join(source_dir, f"{name}.las")
    las.write(path)
    return path, n_points


# ===========================================================================
#  1. 有源文件模式 + pred_sem only (Tester 实际调用方式)
# ===========================================================================


@unittest.skipIf(not HAS_LASPY, "laspy not installed")
class TestLASWriterSourceMode_PredSemOnly(unittest.TestCase):
    """
    Tester 中的调用：general_writer.write(data_name, pred_sem=pred)
    无 coord 参数 —— 依赖 source_dir 中的源文件获取坐标和点数。
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.source_dir = os.path.join(self.tmpdir, "source")
        self.save_dir = os.path.join(self.tmpdir, "output")
        os.makedirs(self.source_dir)
        self.n_points = 200
        _create_source_las(self.source_dir, "scene_001", self.n_points)
        self.writer = LASWriter(
            save_dir=self.save_dir, source_dir=self.source_dir
        )

    def test_write_without_coord_succeeds(self):
        """核心场景：Tester 不传 coord，LASWriter 从源文件获取。"""
        pred = np.random.randint(0, 20, size=self.n_points)
        out_path = self.writer.write("scene_001", pred_sem=pred)
        self.assertTrue(os.path.isfile(out_path))

    def test_output_has_correct_classification(self):
        pred = np.arange(self.n_points) % 15
        out_path = self.writer.write("scene_001", pred_sem=pred)
        las = laspy.read(out_path)
        np.testing.assert_array_equal(
            np.array(las.classification), pred.astype(np.uint8)
        )

    def test_output_preserves_point_count(self):
        pred = np.zeros(self.n_points, dtype=int)
        out_path = self.writer.write("scene_001", pred_sem=pred)
        las = laspy.read(out_path)
        self.assertEqual(len(las.points), self.n_points)

    def test_output_preserves_coordinates(self):
        """源文件的坐标不应被破坏。"""
        source_las = laspy.read(os.path.join(self.source_dir, "scene_001.las"))
        original_x = np.array(source_las.x)

        pred = np.zeros(self.n_points, dtype=int)
        out_path = self.writer.write("scene_001", pred_sem=pred)
        out_las = laspy.read(out_path)
        np.testing.assert_allclose(np.array(out_las.x), original_x, atol=0.01)


# ===========================================================================
#  2. 有源文件 + Benchmark Writer 变换后传给 General Writer
# ===========================================================================


@unittest.skipIf(not HAS_LASPY, "laspy not installed")
class TestBenchmarkPluGeneral(unittest.TestCase):
    """
    模拟 Tester 中 benchmark_writer + general_writer 同时启用的场景：
    pred 先经 benchmark_writer.pred_for_eval() 变换，再传给 general_writer。
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.source_dir = os.path.join(self.tmpdir, "source")
        self.save_dir = os.path.join(self.tmpdir, "output")
        os.makedirs(self.source_dir)
        self.n_points = 150
        _create_source_las(self.source_dir, "scene_pp", self.n_points)

    def test_scannetpp_topk_then_las_write(self):
        """ScanNet++ top-3 → pred_for_eval → top-1 → LASWriter。"""
        bw = create_benchmark_writer("ScanNetPPDataset", self.tmpdir)
        gw = LASWriter(save_dir=self.save_dir, source_dir=self.source_dir)

        # 模拟 topk=3 预测（类别号 0-30，确保在 LAS classification 字段范围内）
        # 注意：当数据集类别数 > 31 时，LAS point_format=2 的 classification
        # 字段会溢出，需要使用 point_format>=6 (LAS 1.4) 或 extra bytes。
        pred_topk = np.random.randint(0, 30, size=(self.n_points, 3))

        # benchmark writer 写入提交文件
        bw.setup()
        bw.write("scene_pp", pred_topk)

        # pred_for_eval 转 top-1
        pred_eval = bw.pred_for_eval(pred_topk)
        self.assertEqual(pred_eval.shape, (self.n_points,))

        # general writer 写入 LAS
        out_path = gw.write("scene_pp", pred_sem=pred_eval)
        self.assertTrue(os.path.isfile(out_path))
        las = laspy.read(out_path)
        np.testing.assert_array_equal(
            np.array(las.classification), pred_eval.astype(np.uint8)
        )

    def test_scannet_identity_then_las_write(self):
        """ScanNet (pred_for_eval=identity) → LASWriter。"""
        _create_source_las(self.source_dir, "scene_sn", self.n_points)
        ds = _FakeDataset(class2id=np.arange(20))
        bw = create_benchmark_writer("ScanNetDataset", self.tmpdir, ds)
        gw = LASWriter(save_dir=self.save_dir, source_dir=self.source_dir)

        pred = np.random.randint(0, 20, size=self.n_points)
        bw.setup()
        bw.write("scene_sn", pred)
        pred_eval = bw.pred_for_eval(pred)
        np.testing.assert_array_equal(pred_eval, pred)

        out_path = gw.write("scene_sn", pred_sem=pred_eval)
        self.assertTrue(os.path.isfile(out_path))


# ===========================================================================
#  3. 无源文件 + 有 coord 模式
# ===========================================================================


@unittest.skipIf(not HAS_LASPY, "laspy not installed")
class TestLASWriterNoSource_WithCoord(unittest.TestCase):
    """Tester 中如果未来传入 coord，无源文件模式依然工作正常。"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.n_points = 100

    def test_write_with_coord_and_pred(self):
        writer = LASWriter(save_dir=self.tmpdir)
        coord = np.random.rand(self.n_points, 3) * 50
        pred = np.random.randint(0, 10, size=self.n_points)
        out_path = writer.write("test_scene", coord=coord, pred_sem=pred)
        self.assertTrue(os.path.isfile(out_path))
        las = laspy.read(out_path)
        self.assertEqual(len(las.points), self.n_points)
        np.testing.assert_array_equal(
            np.array(las.classification), pred.astype(np.uint8)
        )


# ===========================================================================
#  4. 无源文件 + coord=None → 应抛 ValueError
# ===========================================================================


@unittest.skipIf(not HAS_LASPY, "laspy not installed")
class TestLASWriterNoSource_NoCoord(unittest.TestCase):
    """没有源文件也没有 coord，write() 应抛出 ValueError。"""

    def test_raises_value_error(self):
        writer = LASWriter(save_dir=tempfile.mkdtemp())
        pred = np.array([0, 1, 2])
        with self.assertRaises(ValueError) as ctx:
            writer.write("test", pred_sem=pred)
        self.assertIn("coord", str(ctx.exception))

    def test_source_dir_but_missing_file_and_no_coord(self):
        """source_dir 存在但无匹配文件，且 coord=None。"""
        tmpdir = tempfile.mkdtemp()
        source_dir = os.path.join(tmpdir, "empty_source")
        os.makedirs(source_dir)
        writer = LASWriter(save_dir=tmpdir, source_dir=source_dir)
        with self.assertRaises(ValueError):
            writer.write("nonexistent_scene", pred_sem=np.array([0]))

    def test_classification_overflow_with_large_labels(self):
        """
        LAS point_format=2 的 classification 字段仅支持 0-31，
        超过此范围（如 100 类数据集）会触发 OverflowError。
        此测试记录了该已知局限性。
        """
        tmpdir = tempfile.mkdtemp()
        source_dir = os.path.join(tmpdir, "source")
        os.makedirs(source_dir)
        n = 50
        _create_source_las(source_dir, "scene_big", n)
        writer = LASWriter(save_dir=tmpdir, source_dir=source_dir)
        pred = np.array([99] * n)  # 超出 point_format=2 范围
        with self.assertRaises(OverflowError):
            writer.write("scene_big", pred_sem=pred)


# ===========================================================================
#  5. build_writer() → write() 端到端（模拟 Tester 配置路径）
# ===========================================================================


@unittest.skipIf(not HAS_LASPY, "laspy not installed")
class TestBuildWriterEndToEnd(unittest.TestCase):
    """模拟从 cfg.writer 配置到 write() 的完整路径。"""

    def test_build_and_write_source_mode(self):
        tmpdir = tempfile.mkdtemp()
        source_dir = os.path.join(tmpdir, "raw")
        save_dir = os.path.join(tmpdir, "output")
        os.makedirs(source_dir)

        n = 80
        _create_source_las(source_dir, "scan_001", n)

        # 模拟 cfg.writer
        writer_cfg = dict(
            type="LASWriter",
            save_dir=save_dir,
            source_dir=source_dir,
        )
        writer = build_writer(writer_cfg)
        self.assertIsInstance(writer, LASWriter)

        pred = np.random.randint(0, 5, size=n)
        out_path = writer.write("scan_001", pred_sem=pred)
        self.assertTrue(os.path.isfile(out_path))
        self.assertTrue(out_path.endswith(".las"))

    def test_build_compressed(self):
        tmpdir = tempfile.mkdtemp()
        source_dir = os.path.join(tmpdir, "raw")
        save_dir = os.path.join(tmpdir, "output")
        os.makedirs(source_dir)

        n = 50
        _create_source_las(source_dir, "s1", n)

        writer_cfg = dict(
            type="LASWriter",
            save_dir=save_dir,
            source_dir=source_dir,
            compressed=True,
        )
        writer = build_writer(writer_cfg)
        pred = np.zeros(n, dtype=int)
        out_path = writer.write("s1", pred_sem=pred)
        self.assertTrue(out_path.endswith(".laz"))
        self.assertTrue(os.path.isfile(out_path))


# ===========================================================================
#  6. 多样本连续写入（模拟 Tester for loop）
# ===========================================================================


@unittest.skipIf(not HAS_LASPY, "laspy not installed")
class TestMultiSampleLoop(unittest.TestCase):
    """模拟 Tester 对多个样本依次调用 write() 的循环。"""

    def test_sequential_writes(self):
        tmpdir = tempfile.mkdtemp()
        source_dir = os.path.join(tmpdir, "source")
        save_dir = os.path.join(tmpdir, "output")
        os.makedirs(source_dir)

        scene_names = ["room_01", "room_02", "room_03", "hallway_01"]
        n_points_list = [100, 200, 150, 300]

        for name, n in zip(scene_names, n_points_list):
            _create_source_las(source_dir, name, n)

        writer = LASWriter(save_dir=save_dir, source_dir=source_dir)

        for name, n in zip(scene_names, n_points_list):
            pred = np.random.randint(0, 13, size=n)
            out_path = writer.write(name, pred_sem=pred)
            self.assertTrue(os.path.isfile(out_path))
            las = laspy.read(out_path)
            self.assertEqual(len(las.points), n)
            np.testing.assert_array_equal(
                np.array(las.classification), pred.astype(np.uint8)
            )

        # 验证输出了正确数量的文件
        output_files = [f for f in os.listdir(save_dir) if f.endswith(".las")]
        self.assertEqual(len(output_files), len(scene_names))


# ===========================================================================
#  7. 完整 Tester 模拟 (benchmark + general writer 同时工作)
# ===========================================================================


@unittest.skipIf(not HAS_LASPY, "laspy not installed")
class TestFullTesterSimulation(unittest.TestCase):
    """
    完整模拟 Tester.test() 中 Writer 相关的逻辑，
    不依赖 GPU/模型/数据集加载器。
    """

    def test_semseg_tester_flow(self):
        """模拟 SemSegTester 的完整 Writer 流程。"""
        tmpdir = tempfile.mkdtemp()
        source_dir = os.path.join(tmpdir, "source")
        save_path = os.path.join(tmpdir, "result")
        os.makedirs(source_dir)
        os.makedirs(save_path)

        # --- 准备数据 ---
        n_points = 300
        num_classes = 20
        scene_names = ["scene_0001", "scene_0002"]
        for name in scene_names:
            _create_source_las(source_dir, name, n_points)

        # --- 1. 创建 benchmark writer ---
        ds = _FakeDataset(class2id=np.arange(num_classes))
        benchmark_writer = create_benchmark_writer(
            dataset_type="ScanNetDataset",
            save_dir=save_path,
            dataset=ds,
        )
        self.assertIsNotNone(benchmark_writer)
        benchmark_writer.setup()

        # --- 2. 创建 general writer (等价于 build_writer) ---
        general_writer = LASWriter(
            save_dir=os.path.join(save_path, "las_output"),
            source_dir=source_dir,
        )

        # --- 3. 推理循环 ---
        for name in scene_names:
            # 模拟模型 argmax 结果
            pred = np.random.randint(0, num_classes, size=n_points)

            # benchmark: 写提交 + eval 变换
            benchmark_writer.write(name, pred)
            pred_eval = benchmark_writer.pred_for_eval(pred)
            np.testing.assert_array_equal(pred_eval, pred)  # ScanNet: identity

            # general: 写 LAS
            out_path = general_writer.write(name, pred_sem=pred_eval)
            self.assertTrue(os.path.isfile(out_path))

        # --- 4. finalize ---
        benchmark_writer.finalize()

        # --- 验证所有输出文件 ---
        # benchmark 提交文件
        for name in scene_names:
            txt_path = os.path.join(save_path, "submit", f"{name}.txt")
            self.assertTrue(os.path.isfile(txt_path))

        # LAS 输出
        las_dir = os.path.join(save_path, "las_output")
        for name in scene_names:
            las_path = os.path.join(las_dir, f"{name}.las")
            self.assertTrue(os.path.isfile(las_path))
            las = laspy.read(las_path)
            self.assertEqual(len(las.points), n_points)

    def test_scannetpp_tester_flow_with_topk(self):
        """模拟 ScanNet++ topk=3 场景下两种 Writer 协同工作。"""
        tmpdir = tempfile.mkdtemp()
        source_dir = os.path.join(tmpdir, "source")
        save_path = os.path.join(tmpdir, "result")
        os.makedirs(source_dir)
        os.makedirs(save_path)

        n_points = 200
        _create_source_las(source_dir, "scene_pp", n_points)

        # benchmark writer (ScanNet++)
        benchmark_writer = create_benchmark_writer(
            dataset_type="ScanNetPPDataset",
            save_dir=save_path,
        )
        benchmark_writer.setup()
        self.assertEqual(benchmark_writer.topk, 3)

        # general writer
        general_writer = LASWriter(
            save_dir=os.path.join(save_path, "las_output"),
            source_dir=source_dir,
        )

        # 模拟 topk=3 预测（类别号保持在 classification 字段范围内）
        pred_topk = np.random.randint(0, 30, size=(n_points, 3))

        # benchmark write (top-3 csv)
        benchmark_writer.write("scene_pp", pred_topk)

        # pred_for_eval → top-1
        pred_eval = benchmark_writer.pred_for_eval(pred_topk)
        self.assertEqual(pred_eval.shape, (n_points,))

        # general writer 写 LAS (1D label)
        out_path = general_writer.write("scene_pp", pred_sem=pred_eval)
        las = laspy.read(out_path)
        self.assertEqual(len(las.points), n_points)
        np.testing.assert_array_equal(
            np.array(las.classification), pred_eval.astype(np.uint8)
        )

    def test_no_general_writer_only_benchmark(self):
        """general_writer=None 时仅 benchmark_writer 工作，不出错。"""
        tmpdir = tempfile.mkdtemp()
        save_path = os.path.join(tmpdir, "result")
        os.makedirs(save_path)

        n = 100
        ds = _FakeDataset(class2id=np.arange(20))
        bw = create_benchmark_writer("ScanNetDataset", save_path, ds)
        bw.setup()
        general_writer = None  # 模拟 cfg.writer = None

        pred = np.random.randint(0, 20, size=n)
        bw.write("scene_001", pred)
        pred_eval = bw.pred_for_eval(pred)

        if general_writer is not None:
            general_writer.write("scene_001", pred_sem=pred_eval)
        # 不应抛异常

        # benchmark 文件存在
        self.assertTrue(
            os.path.isfile(os.path.join(save_path, "submit", "scene_001.txt"))
        )

    def test_no_benchmark_writer_only_general(self):
        """未知数据集时 benchmark_writer=None，仅 general_writer 工作。"""
        tmpdir = tempfile.mkdtemp()
        source_dir = os.path.join(tmpdir, "source")
        save_dir = os.path.join(tmpdir, "output")
        os.makedirs(source_dir)

        n = 100
        _create_source_las(source_dir, "custom_scene", n)

        benchmark_writer = create_benchmark_writer("CustomDataset", tmpdir)
        self.assertIsNone(benchmark_writer)

        general_writer = LASWriter(save_dir=save_dir, source_dir=source_dir)
        pred = np.random.randint(0, 10, size=n)

        # Tester 逻辑
        if benchmark_writer is not None:
            benchmark_writer.write("custom_scene", pred)
            pred = benchmark_writer.pred_for_eval(pred)

        out_path = general_writer.write("custom_scene", pred_sem=pred)
        self.assertTrue(os.path.isfile(out_path))


# ===========================================================================
#  8. CNF pred_coord + source_dir（借用头文件）
# ===========================================================================


@unittest.skipIf(not HAS_LASPY, "laspy not installed")
class TestCNFPredCoordSourceHeader(unittest.TestCase):
    """
    验证 LASWriter 在收到 pred_coord 时能从 source_dir 借用
    CRS/scale/offset，而点数及坐标全部来自 pred_coord。
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.source_dir = os.path.join(self.tmpdir, "source")
        self.save_dir = os.path.join(self.tmpdir, "output")
        os.makedirs(self.source_dir)

        # 源文件：100 个点，特定 scale/offset 模拟真实 CRS
        src_header = laspy.LasHeader(point_format=2, version="1.2")
        src_header.scales = np.array([0.01, 0.01, 0.01])
        src_header.offsets = np.array([500000.0, 4000000.0, 1000.0])
        src_las = laspy.LasData(src_header)
        np.random.seed(7)
        src_xyz = np.random.rand(100, 3)
        src_las.x = src_xyz[:, 0] + src_header.offsets[0]
        src_las.y = src_xyz[:, 1] + src_header.offsets[1]
        src_las.z = src_xyz[:, 2] + src_header.offsets[2]
        self.src_path = os.path.join(self.source_dir, "tile_001.las")
        src_las.write(self.src_path)

        self.src_scales = src_header.scales.copy()
        self.src_offsets = src_header.offsets.copy()

        # CNF 预测网格：500 个点（点数与源文件不同）
        np.random.seed(13)
        self.pred_coord = np.random.rand(500, 3)
        self.pred_coord[:, 0] += self.src_offsets[0]
        self.pred_coord[:, 1] += self.src_offsets[1]
        self.pred_coord[:, 2] += self.src_offsets[2]

        self.writer = LASWriter(save_dir=self.save_dir, source_dir=self.source_dir)

    def test_pred_coord_creates_new_point_count(self):
        """输出 LAS 的点数等于 pred_coord 行数，而非源文件点数。"""
        out_path = self.writer.write("tile_001", pred_coord=self.pred_coord)
        las = laspy.read(out_path)
        self.assertEqual(len(las.points), len(self.pred_coord))

    def test_pred_coord_inherits_source_scale_offset(self):
        """输出 LAS 继承源文件的 scale 和 offset。"""
        out_path = self.writer.write("tile_001", pred_coord=self.pred_coord)
        las = laspy.read(out_path)
        np.testing.assert_allclose(
            np.array(las.header.scales), self.src_scales, rtol=1e-9
        )
        np.testing.assert_allclose(
            np.array(las.header.offsets), self.src_offsets, rtol=1e-6
        )

    def test_pred_coord_xyz_preserved(self):
        """输出 LAS 的 XYZ 坐标与 pred_coord 一致（scale 精度内）。"""
        out_path = self.writer.write("tile_001", pred_coord=self.pred_coord)
        las = laspy.read(out_path)
        # scale = 0.01 → atol=0.01
        np.testing.assert_allclose(
            np.array(las.x), self.pred_coord[:, 0], atol=0.011
        )
        np.testing.assert_allclose(
            np.array(las.y), self.pred_coord[:, 1], atol=0.011
        )
        np.testing.assert_allclose(
            np.array(las.z), self.pred_coord[:, 2], atol=0.011
        )

    def test_pred_coord_no_source_falls_back_to_create_new(self):
        """source_dir 中无匹配文件时，退回到 _create_new（自动 scale/offset）。"""
        writer = LASWriter(save_dir=self.save_dir, source_dir=self.source_dir)
        with warnings.catch_warnings(record=True):
            out_path = writer.write("tile_nonexistent", pred_coord=self.pred_coord)
        las = laspy.read(out_path)
        self.assertEqual(len(las.points), len(self.pred_coord))

    def test_pred_coord_without_source_dir(self):
        """未设置 source_dir 时，pred_coord 路径仍正常创建文件。"""
        writer = LASWriter(save_dir=self.save_dir)
        out_path = writer.write("tile_nosrc", pred_coord=self.pred_coord)
        las = laspy.read(out_path)
        self.assertEqual(len(las.points), len(self.pred_coord))


# ===========================================================================
#  9. classification 参数
# ===========================================================================


@unittest.skipIf(not HAS_LASPY, "laspy not installed")
class TestLASWriterClassificationParam(unittest.TestCase):
    """验证 classification 构造参数的行为。"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.source_dir = os.path.join(self.tmpdir, "source")
        self.save_dir = os.path.join(self.tmpdir, "output")
        os.makedirs(self.source_dir)
        self.n = 80
        _create_source_las(self.source_dir, "scene_cls", self.n)

        np.random.seed(99)
        self.pred_coord = np.random.rand(200, 3) * 100.0

    def test_classification_set_on_pred_coord(self):
        """pred_coord 输出时，每个点的 classification == 构造参数值。"""
        writer = LASWriter(
            save_dir=self.save_dir, source_dir=self.source_dir, classification=2
        )
        out_path = writer.write("scene_cls", pred_coord=self.pred_coord)
        las = laspy.read(out_path)
        expected = np.full(len(self.pred_coord), 2, dtype=np.uint8)
        np.testing.assert_array_equal(np.array(las.classification), expected)

    def test_classification_set_on_standard_write_no_pred_sem(self):
        """无 pred_sem 时，标准 write() 也应用默认 classification。"""
        writer = LASWriter(
            save_dir=self.save_dir, source_dir=self.source_dir, classification=5
        )
        out_path = writer.write("scene_cls")
        las = laspy.read(out_path)
        expected = np.full(self.n, 5, dtype=np.uint8)
        np.testing.assert_array_equal(np.array(las.classification), expected)

    def test_pred_sem_overrides_classification_param(self):
        """显式传入 pred_sem 时，优先使用 pred_sem，忽略 classification 参数。"""
        writer = LASWriter(
            save_dir=self.save_dir, source_dir=self.source_dir, classification=2
        )
        override = np.full(self.n, 9, dtype=np.uint8)
        out_path = writer.write("scene_cls", pred_sem=override)
        las = laspy.read(out_path)
        np.testing.assert_array_equal(np.array(las.classification), override)

    def test_classification_none_no_field_written(self):
        """classification=None（默认）且无 pred_sem 时，classification 字段保持源文件默认值（0）。"""
        writer = LASWriter(
            save_dir=self.save_dir, source_dir=self.source_dir, classification=None
        )
        out_path = writer.write("scene_cls")
        las = laspy.read(out_path)
        # 源文件创建时未设置 classification → 全部为 0
        expected = np.zeros(self.n, dtype=np.uint8)
        np.testing.assert_array_equal(np.array(las.classification), expected)


if __name__ == "__main__":
    unittest.main()
