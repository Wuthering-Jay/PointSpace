"""
Tests for the Writer system (WRITERS registry).

覆盖范围:
    1. 注册表与构建器 (Registry & Builder)
    2. LASWriter - 无源文件模式（从零创建）
    3. LASWriter - 有源文件模式（基于原始文件追加字段）
    4. LASWriter - 源文件缺失回退 (fallback)
    5. LASWriter - 多任务字段写入（语义 + 实例 + 自定义维度）
    6. LASWriter - 输入校验 (点数不匹配)
    7. 占位 Writer (PLY, PCD) 抛出 NotImplementedError
    8. BaseWriter 抽象约束

Author: PointSpace Team
"""

import os
import tempfile
import unittest
import warnings

import numpy as np

# 延迟导入，跳过缺少 laspy 的环境
try:
    import laspy
    from laspy import ExtraBytesParams

    HAS_LASPY = True
except ImportError:
    HAS_LASPY = False

# ----- 导入被测模块 -----
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pointspace.writers import WRITERS, build_writer, BaseWriter
from pointspace.writers.las_writer import LASWriter
from pointspace.writers.ply_writer import PLYWriter
from pointspace.writers.pcd_writer import PCDWriter


# ===========================================================================
#  辅助函数
# ===========================================================================


def _make_coords(n=1000):
    """生成 (n, 3) 的随机点坐标。"""
    np.random.seed(42)
    return np.random.rand(n, 3) * 100


def _make_source_las(path, n=1000):
    """
    创建一个带有自定义字段和 RGB 的源 LAS 文件用于测试。
    """
    header = laspy.LasHeader(point_format=2, version="1.2")
    header.offsets = np.array([0.0, 0.0, 0.0])
    header.scales = np.array([0.001, 0.001, 0.001])
    las = laspy.LasData(header)
    coords = _make_coords(n)
    las.x = coords[:, 0]
    las.y = coords[:, 1]
    las.z = coords[:, 2]
    las.red = np.random.randint(0, 65535, n, dtype=np.uint16)
    las.green = np.random.randint(0, 65535, n, dtype=np.uint16)
    las.blue = np.random.randint(0, 65535, n, dtype=np.uint16)
    las.classification = np.zeros(n, dtype=np.uint8)
    las.write(path)
    return coords


# ===========================================================================
#  测试: 注册表与构建器
# ===========================================================================


class TestWriterRegistry(unittest.TestCase):
    """测试 WRITERS 注册表和 build_writer 构建函数。"""

    def test_registered_writers(self):
        """所有 Writer 类都已注册到 WRITERS。"""
        for name in ("LASWriter", "PLYWriter", "PCDWriter"):
            self.assertIn(name, WRITERS._module_dict, f"{name} 未注册")

    @unittest.skipUnless(HAS_LASPY, "laspy not installed")
    def test_build_las_writer(self):
        """通过 build_writer 构建 LASWriter 实例。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = build_writer(dict(type="LASWriter", save_dir=tmpdir))
            self.assertIsInstance(writer, LASWriter)

    def test_build_unknown_writer(self):
        """构建不存在的 Writer 应抛出 KeyError。"""
        with self.assertRaises(KeyError):
            build_writer(dict(type="FooBarWriter", save_dir="/tmp"))


# ===========================================================================
#  测试: LASWriter
# ===========================================================================


@unittest.skipUnless(HAS_LASPY, "laspy not installed")
class TestLASWriterCreateMode(unittest.TestCase):
    """测试 LASWriter 无源文件模式（从零创建）。"""

    def test_write_coord_only(self):
        """仅写入坐标，不附加任何预测。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = LASWriter(save_dir=tmpdir)
            coord = _make_coords(500)
            out_path = writer.write("test_scene", coord)

            self.assertTrue(os.path.isfile(out_path))
            self.assertTrue(out_path.endswith(".las"))

            # 读回验证
            las = laspy.read(out_path)
            self.assertEqual(len(las.points), 500)
            np.testing.assert_allclose(
                np.stack([las.x, las.y, las.z], axis=-1), coord, atol=0.002
            )

    def test_write_with_color(self):
        """传入 uint8 颜色，验证 RGB 缩放到 uint16。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = LASWriter(save_dir=tmpdir)
            coord = _make_coords(100)
            color = np.random.randint(0, 256, (100, 3), dtype=np.uint8)
            writer.write("color_scene", coord, color=color)

            las = laspy.read(os.path.join(tmpdir, "color_scene.las"))
            # 颜色应该已缩放: val * 257
            np.testing.assert_array_equal(
                np.array(las.red), color[:, 0].astype(np.uint16) * 257
            )

    def test_write_compressed_laz(self):
        """测试 .laz 压缩输出。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = LASWriter(save_dir=tmpdir, compressed=True)
            coord = _make_coords(200)
            out_path = writer.write("laz_scene", coord)
            self.assertTrue(out_path.endswith(".laz"))
            self.assertTrue(os.path.isfile(out_path))


@unittest.skipUnless(HAS_LASPY, "laspy not installed")
class TestLASWriterSemSeg(unittest.TestCase):
    """测试 LASWriter 语义分割字段写入。"""

    def test_write_pred_sem(self):
        """写入 pred_sem 应反映在 classification 字段中。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = LASWriter(save_dir=tmpdir)
            n = 300
            coord = _make_coords(n)
            labels = np.random.randint(0, 20, n)
            writer.write("sem_scene", coord, pred_sem=labels)

            las = laspy.read(os.path.join(tmpdir, "sem_scene.las"))
            np.testing.assert_array_equal(
                np.array(las.classification), labels.astype(np.uint8)
            )


@unittest.skipUnless(HAS_LASPY, "laspy not installed")
class TestLASWriterInsSeg(unittest.TestCase):
    """测试 LASWriter 实例分割字段写入。"""

    def test_write_pred_ins(self):
        """写入 pred_ins 应在 extra bytes 中出现 instance_id 字段。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = LASWriter(save_dir=tmpdir)
            n = 400
            coord = _make_coords(n)
            instance_ids = np.random.randint(0, 50, n)
            writer.write("ins_scene", coord, pred_ins=instance_ids)

            las = laspy.read(os.path.join(tmpdir, "ins_scene.las"))
            self.assertIn("instance_id", list(las.point_format.dimension_names))
            np.testing.assert_array_equal(
                np.array(las.instance_id), instance_ids.astype(np.int32)
            )


@unittest.skipUnless(HAS_LASPY, "laspy not installed")
class TestLASWriterMultiTask(unittest.TestCase):
    """测试 LASWriter 同时写入多个任务的结果。"""

    def test_sem_and_ins_together(self):
        """同时写入语义分割和实例分割。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = LASWriter(save_dir=tmpdir)
            n = 500
            coord = _make_coords(n)
            pred_sem = np.random.randint(0, 10, n)
            pred_ins = np.random.randint(0, 100, n)
            writer.write("multi_scene", coord, pred_sem=pred_sem, pred_ins=pred_ins)

            las = laspy.read(os.path.join(tmpdir, "multi_scene.las"))
            np.testing.assert_array_equal(
                np.array(las.classification), pred_sem.astype(np.uint8)
            )
            np.testing.assert_array_equal(
                np.array(las.instance_id), pred_ins.astype(np.int32)
            )

    def test_extra_dims(self):
        """通过 extra_dims 写入任意自定义维度。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = LASWriter(save_dir=tmpdir)
            n = 200
            coord = _make_coords(n)
            confidence = np.random.rand(n).astype(np.float32)
            writer.write(
                "extra_scene",
                coord,
                extra_dims={"confidence": (confidence, np.float32)},
            )

            las = laspy.read(os.path.join(tmpdir, "extra_scene.las"))
            self.assertIn("confidence", list(las.point_format.dimension_names))
            np.testing.assert_allclose(
                np.array(las.confidence), confidence, atol=1e-6
            )


@unittest.skipUnless(HAS_LASPY, "laspy not installed")
class TestLASWriterSourceMode(unittest.TestCase):
    """测试 LASWriter 有源文件模式。"""

    def test_load_source_and_append(self):
        """从源文件加载后追加语义分割字段，原始 RGB 应保留。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            source_dir = os.path.join(tmpdir, "source")
            save_dir = os.path.join(tmpdir, "output")
            os.makedirs(source_dir)

            n = 600
            source_coords = _make_source_las(
                os.path.join(source_dir, "scene_src.las"), n=n
            )
            source_las = laspy.read(os.path.join(source_dir, "scene_src.las"))
            original_red = np.array(source_las.red).copy()

            writer = LASWriter(save_dir=save_dir, source_dir=source_dir)
            pred_sem = np.random.randint(0, 15, n)
            writer.write("scene_src", source_coords, pred_sem=pred_sem)

            result = laspy.read(os.path.join(save_dir, "scene_src.las"))
            # 语义标签正确
            np.testing.assert_array_equal(
                np.array(result.classification), pred_sem.astype(np.uint8)
            )
            # 原始 RGB 保留
            np.testing.assert_array_equal(np.array(result.red), original_red)

    def test_source_missing_fallback(self):
        """源目录下找不到文件时应发出警告并从零创建。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            source_dir = os.path.join(tmpdir, "empty_source")
            save_dir = os.path.join(tmpdir, "output")
            os.makedirs(source_dir)

            writer = LASWriter(save_dir=save_dir, source_dir=source_dir)
            coord = _make_coords(100)

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                writer.write("nonexistent", coord)
                # 应该收到 RuntimeWarning
                self.assertTrue(
                    any(issubclass(x.category, RuntimeWarning) for x in w),
                    "应发出 RuntimeWarning 警告",
                )

            # 文件仍然应该被成功创建（fallback 模式）
            self.assertTrue(
                os.path.isfile(os.path.join(save_dir, "nonexistent.las"))
            )


@unittest.skipUnless(HAS_LASPY, "laspy not installed")
class TestLASWriterValidation(unittest.TestCase):
    """测试 LASWriter 输入校验。"""

    def test_pred_sem_length_mismatch(self):
        """pred_sem 长度与点数不匹配应抛出 ValueError。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = LASWriter(save_dir=tmpdir)
            coord = _make_coords(100)
            pred_sem = np.zeros(50)  # 故意不匹配
            with self.assertRaises(ValueError):
                writer.write("bad_scene", coord, pred_sem=pred_sem)

    def test_pred_ins_length_mismatch(self):
        """pred_ins 长度与点数不匹配应抛出 ValueError。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = LASWriter(save_dir=tmpdir)
            coord = _make_coords(100)
            pred_ins = np.zeros(200)  # 故意不匹配
            with self.assertRaises(ValueError):
                writer.write("bad_ins", coord, pred_ins=pred_ins)

    def test_extra_dims_length_mismatch(self):
        """extra_dims 中数据长度与点数不匹配应抛出 ValueError。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = LASWriter(save_dir=tmpdir)
            coord = _make_coords(100)
            bad_data = np.zeros(50)
            with self.assertRaises(ValueError):
                writer.write(
                    "bad_extra", coord, extra_dims={"bad": (bad_data, np.float32)}
                )


# ===========================================================================
#  测试: 占位 Writer
# ===========================================================================


class TestPlaceholderWriters(unittest.TestCase):
    """测试 PLYWriter 和 PCDWriter 占位符。"""

    def test_ply_writer_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = PLYWriter(save_dir=tmpdir)
            with self.assertRaises(NotImplementedError):
                writer.write("scene", _make_coords(10))

    def test_pcd_writer_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = PCDWriter(save_dir=tmpdir)
            with self.assertRaises(NotImplementedError):
                writer.write("scene", _make_coords(10))


# ===========================================================================
#  测试: BaseWriter 抽象约束
# ===========================================================================


class TestBaseWriterAbstract(unittest.TestCase):
    """验证 BaseWriter 不能直接实例化。"""

    def test_cannot_instantiate(self):
        """直接实例化 BaseWriter 应失败。"""
        with self.assertRaises(TypeError):
            BaseWriter(save_dir="/tmp")

    def test_subclass_must_implement_write(self):
        """子类不实现 write() 时，实例化应失败。"""

        class IncompleteWriter(BaseWriter):
            pass

        with self.assertRaises(TypeError):
            IncompleteWriter(save_dir="/tmp")

    def test_valid_subclass(self):
        """正确实现 write() 的子类可以实例化。"""

        class GoodWriter(BaseWriter):
            def write(self, data_name, coord, **kwargs):
                return "ok"

        with tempfile.TemporaryDirectory() as tmpdir:
            w = GoodWriter(save_dir=tmpdir)
            self.assertEqual(w.write("test", np.zeros((1, 3))), "ok")


if __name__ == "__main__":
    unittest.main()
