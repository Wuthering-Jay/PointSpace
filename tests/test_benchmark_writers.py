"""
Tests for the Benchmark Writer system and Tester integration.

覆盖范围:
    1. 工厂函数 create_benchmark_writer() - 已知/未知数据集类型
    2. BaseBenchmarkWriter - 抽象约束、默认行为
    3. ScanNetBenchmarkWriter - setup/write/pred_for_eval
    4. ScanNetPPBenchmarkWriter - topk=3, write, pred_for_eval
    5. SemanticKITTIBenchmarkWriter - learning_map_inv 映射, 目录结构
    6. NuScenesBenchmarkWriter - submission.json, 二进制 bin 写入
    7. S3DISBenchmarkWriter - 跳过逐样本写入, finalize 保存 .pth
    8. Tester 集成 - benchmark_writer 生命周期调用链 (mock)

Author: PointSpace Team
"""

import json
import os
import sys
import tempfile
import unittest

import numpy as np

# 延迟导入 torch（某些 CI 环境可能无 GPU）
try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# ----- 导入被测模块 -----
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pointspace.writers.benchmark import (
    create_benchmark_writer,
    BaseBenchmarkWriter,
    ScanNetBenchmarkWriter,
    ScanNetPPBenchmarkWriter,
    SemanticKITTIBenchmarkWriter,
    NuScenesBenchmarkWriter,
    S3DISBenchmarkWriter,
)


# ===========================================================================
#  辅助工具
# ===========================================================================


class _FakeDataset:
    """模拟测试数据集对象，提供各 Writer 所需的属性。"""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def _make_pred(n=500, num_classes=20):
    """生成随机的 argmax 预测 (N,)。"""
    np.random.seed(42)
    return np.random.randint(0, num_classes, size=n)


def _make_pred_topk(n=500, num_classes=100, k=3):
    """生成随机的 topk 预测 (N, k)。"""
    np.random.seed(42)
    return np.random.randint(0, num_classes, size=(n, k))


# ===========================================================================
#  1. 工厂函数
# ===========================================================================


class TestCreateBenchmarkWriter(unittest.TestCase):
    """测试 create_benchmark_writer() 工厂函数。"""

    def test_scannet_returns_writer(self):
        ds = _FakeDataset(class2id=np.arange(20))
        w = create_benchmark_writer("ScanNetDataset", "/tmp/test", ds)
        self.assertIsInstance(w, ScanNetBenchmarkWriter)

    def test_scannet200_returns_same_writer(self):
        ds = _FakeDataset(class2id=np.arange(200))
        w = create_benchmark_writer("ScanNet200Dataset", "/tmp/test", ds)
        self.assertIsInstance(w, ScanNetBenchmarkWriter)

    def test_scannetpp_returns_writer(self):
        w = create_benchmark_writer("ScanNetPPDataset", "/tmp/test")
        self.assertIsInstance(w, ScanNetPPBenchmarkWriter)

    def test_semantic_kitti_returns_writer(self):
        ds = _FakeDataset(learning_map_inv={0: 0, 1: 10, 2: 11})
        w = create_benchmark_writer("SemanticKITTIDataset", "/tmp/test", ds)
        self.assertIsInstance(w, SemanticKITTIBenchmarkWriter)

    def test_nuscenes_returns_writer(self):
        w = create_benchmark_writer("NuScenesDataset", "/tmp/test")
        self.assertIsInstance(w, NuScenesBenchmarkWriter)

    def test_s3dis_returns_writer(self):
        ds = _FakeDataset(split="Area_1")
        w = create_benchmark_writer("S3DISDataset", "/tmp/test", ds)
        self.assertIsInstance(w, S3DISBenchmarkWriter)

    def test_unknown_dataset_returns_none(self):
        w = create_benchmark_writer("UnknownDataset", "/tmp/test")
        self.assertIsNone(w)

    def test_empty_string_returns_none(self):
        w = create_benchmark_writer("", "/tmp/test")
        self.assertIsNone(w)


# ===========================================================================
#  2. BaseBenchmarkWriter 抽象约束
# ===========================================================================


class TestBaseBenchmarkWriter(unittest.TestCase):
    """测试 BaseBenchmarkWriter 抽象基类。"""

    def test_cannot_instantiate_directly(self):
        """抽象基类不能直接实例化。"""
        with self.assertRaises(TypeError):
            BaseBenchmarkWriter("/tmp/test")

    def test_default_topk_is_one(self):
        self.assertEqual(BaseBenchmarkWriter.topk, 1)

    def test_default_pred_for_eval_is_identity(self):
        """子类未重写 pred_for_eval 时应原样返回。"""

        class _DummyWriter(BaseBenchmarkWriter):
            def write(self, data_name, pred, **kwargs):
                pass

        w = _DummyWriter("/tmp/test")
        pred = np.array([1, 2, 3])
        result = w.pred_for_eval(pred)
        np.testing.assert_array_equal(result, pred)

    def test_default_finalize_is_noop(self):
        """子类未重写 finalize 时不应抛异常。"""

        class _DummyWriter(BaseBenchmarkWriter):
            def write(self, data_name, pred, **kwargs):
                pass

        w = _DummyWriter("/tmp/test")
        w.finalize()  # should not raise


# ===========================================================================
#  3. ScanNetBenchmarkWriter
# ===========================================================================


class TestScanNetBenchmarkWriter(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        # class2id: training label -> submission label
        self.class2id = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                  11, 12, 14, 16, 24, 28, 33, 34, 36, 39])
        ds = _FakeDataset(class2id=self.class2id)
        self.writer = ScanNetBenchmarkWriter(self.tmpdir, ds)

    def test_topk_is_one(self):
        self.assertEqual(self.writer.topk, 1)

    def test_setup_creates_submit_dir(self):
        self.writer.setup()
        self.assertTrue(os.path.isdir(os.path.join(self.tmpdir, "submit")))

    def test_write_creates_txt_file(self):
        self.writer.setup()
        pred = _make_pred(n=100, num_classes=20)
        self.writer.write("scene0001_00", pred)
        out_path = os.path.join(self.tmpdir, "submit", "scene0001_00.txt")
        self.assertTrue(os.path.isfile(out_path))

    def test_write_content_uses_class2id(self):
        self.writer.setup()
        pred = np.array([0, 1, 2, 19])  # training labels
        self.writer.write("scene_test", pred)
        out_path = os.path.join(self.tmpdir, "submit", "scene_test.txt")
        result = np.loadtxt(out_path, dtype=int)
        expected = self.class2id[pred]
        np.testing.assert_array_equal(result, expected)

    def test_pred_for_eval_returns_same(self):
        pred = _make_pred(n=50)
        result = self.writer.pred_for_eval(pred)
        np.testing.assert_array_equal(result, pred)


# ===========================================================================
#  4. ScanNetPPBenchmarkWriter
# ===========================================================================


class TestScanNetPPBenchmarkWriter(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.writer = ScanNetPPBenchmarkWriter(self.tmpdir)

    def test_topk_is_three(self):
        self.assertEqual(self.writer.topk, 3)

    def test_setup_creates_submit_dir(self):
        self.writer.setup()
        self.assertTrue(os.path.isdir(os.path.join(self.tmpdir, "submit")))

    def test_write_creates_txt_file(self):
        self.writer.setup()
        pred = _make_pred_topk(n=100, num_classes=100, k=3)
        self.writer.write("scene_pp", pred)
        out_path = os.path.join(self.tmpdir, "submit", "scene_pp.txt")
        self.assertTrue(os.path.isfile(out_path))

    def test_write_content_comma_separated(self):
        self.writer.setup()
        pred = np.array([[10, 20, 30], [5, 15, 25]], dtype=np.int32)
        self.writer.write("test_scene", pred)
        out_path = os.path.join(self.tmpdir, "submit", "test_scene.txt")
        with open(out_path, "r") as f:
            lines = f.read().strip().split("\n")
        self.assertEqual(len(lines), 2)
        # 验证逗号分隔格式
        vals_0 = [int(x) for x in lines[0].split(",")]
        self.assertEqual(vals_0, [10, 20, 30])

    def test_pred_for_eval_returns_top1(self):
        """pred_for_eval 应返回 top-1 列。"""
        pred = np.array([[10, 20, 30], [5, 15, 25]])
        result = self.writer.pred_for_eval(pred)
        np.testing.assert_array_equal(result, np.array([10, 5]))

    def test_pred_for_eval_shape(self):
        pred = _make_pred_topk(n=100)
        result = self.writer.pred_for_eval(pred)
        self.assertEqual(result.shape, (100,))


# ===========================================================================
#  5. SemanticKITTIBenchmarkWriter
# ===========================================================================


class TestSemanticKITTIBenchmarkWriter(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.learning_map_inv = {0: 0, 1: 10, 2: 11, 3: 15, 4: 18, 5: 20}
        ds = _FakeDataset(learning_map_inv=self.learning_map_inv)
        self.writer = SemanticKITTIBenchmarkWriter(self.tmpdir, ds)

    def test_topk_is_one(self):
        self.assertEqual(self.writer.topk, 1)

    def test_setup_creates_submit_dir(self):
        self.writer.setup()
        self.assertTrue(os.path.isdir(os.path.join(self.tmpdir, "submit")))

    def test_write_creates_label_file(self):
        self.writer.setup()
        pred = np.array([0, 1, 2, 3, 4, 5])
        self.writer.write("00_000123", pred)
        out_path = os.path.join(
            self.tmpdir, "submit", "sequences", "00", "predictions", "000123.label"
        )
        self.assertTrue(os.path.isfile(out_path))

    def test_write_applies_learning_map_inv(self):
        self.writer.setup()
        pred = np.array([0, 1, 2, 3, 4, 5])
        self.writer.write("08_000000", pred)
        out_path = os.path.join(
            self.tmpdir, "submit", "sequences", "08", "predictions", "000000.label"
        )
        result = np.fromfile(out_path, dtype=np.uint32)
        expected = np.array([0, 10, 11, 15, 18, 20], dtype=np.uint32)
        np.testing.assert_array_equal(result, expected)

    def test_write_binary_format_uint32(self):
        self.writer.setup()
        pred = np.array([1, 2], dtype=np.int64)
        self.writer.write("03_000001", pred)
        out_path = os.path.join(
            self.tmpdir, "submit", "sequences", "03", "predictions", "000001.label"
        )
        result = np.fromfile(out_path, dtype=np.uint32)
        self.assertEqual(result.dtype, np.uint32)

    def test_multiple_sequences(self):
        """验证不同 sequence 的目录结构正确。"""
        self.writer.setup()
        pred = np.array([0])
        self.writer.write("00_000000", pred)
        self.writer.write("08_000100", pred)
        self.writer.write("11_000050", pred)
        for seq in ["00", "08", "11"]:
            self.assertTrue(
                os.path.isdir(
                    os.path.join(self.tmpdir, "submit", "sequences", seq, "predictions")
                )
            )


# ===========================================================================
#  6. NuScenesBenchmarkWriter
# ===========================================================================


class TestNuScenesBenchmarkWriter(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.writer = NuScenesBenchmarkWriter(self.tmpdir)

    def test_topk_is_one(self):
        self.assertEqual(self.writer.topk, 1)

    def test_setup_creates_directories(self):
        self.writer.setup()
        self.assertTrue(
            os.path.isdir(os.path.join(self.tmpdir, "submit", "lidarseg", "test"))
        )
        self.assertTrue(
            os.path.isdir(os.path.join(self.tmpdir, "submit", "test"))
        )

    def test_setup_writes_submission_json(self):
        self.writer.setup()
        json_path = os.path.join(self.tmpdir, "submit", "test", "submission.json")
        self.assertTrue(os.path.isfile(json_path))
        with open(json_path, "r") as f:
            data = json.load(f)
        self.assertIn("meta", data)
        self.assertTrue(data["meta"]["use_lidar"])
        self.assertFalse(data["meta"]["use_camera"])

    def test_write_creates_bin_file(self):
        self.writer.setup()
        pred = _make_pred(n=200, num_classes=16)
        self.writer.write("sample_token_001", pred)
        out_path = os.path.join(
            self.tmpdir, "submit", "lidarseg", "test", "sample_token_001_lidarseg.bin"
        )
        self.assertTrue(os.path.isfile(out_path))

    def test_write_content_has_offset_plus_one(self):
        """NuScenes 标签需要 +1 偏移。"""
        self.writer.setup()
        pred = np.array([0, 5, 15], dtype=np.int64)
        self.writer.write("tok", pred)
        out_path = os.path.join(
            self.tmpdir, "submit", "lidarseg", "test", "tok_lidarseg.bin"
        )
        result = np.fromfile(out_path, dtype=np.uint8)
        np.testing.assert_array_equal(result, np.array([1, 6, 16], dtype=np.uint8))

    def test_write_dtype_uint8(self):
        self.writer.setup()
        pred = np.array([0, 254])
        self.writer.write("tok2", pred)
        out_path = os.path.join(
            self.tmpdir, "submit", "lidarseg", "test", "tok2_lidarseg.bin"
        )
        result = np.fromfile(out_path, dtype=np.uint8)
        self.assertEqual(result.dtype, np.uint8)


# ===========================================================================
#  7. S3DISBenchmarkWriter
# ===========================================================================


class TestS3DISBenchmarkWriter(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        ds = _FakeDataset(split="Area_5")
        self.writer = S3DISBenchmarkWriter(self.tmpdir, ds)

    def test_topk_is_one(self):
        self.assertEqual(self.writer.topk, 1)

    def test_setup_does_not_create_submit(self):
        """S3DIS 不需要 submit 目录。"""
        self.writer.setup()
        self.assertFalse(
            os.path.isdir(os.path.join(self.tmpdir, "submit"))
        )

    def test_write_is_noop(self):
        """S3DIS 没有逐样本提交文件。"""
        pred = _make_pred(n=100)
        # 不应抛异常，也不应创建文件
        self.writer.write("room_001", pred)
        # submit 目录不存在
        files = os.listdir(self.tmpdir)
        self.assertEqual(len(files), 0)

    @unittest.skipIf(not HAS_TORCH, "torch not available")
    def test_finalize_saves_pth(self):
        intersection = np.array([10.0, 20.0, 30.0])
        union = np.array([100.0, 200.0, 300.0])
        target = np.array([50.0, 100.0, 150.0])
        self.writer.finalize(intersection=intersection, union=union, target=target)
        pth_path = os.path.join(self.tmpdir, "Area_5.pth")
        self.assertTrue(os.path.isfile(pth_path))

    @unittest.skipIf(not HAS_TORCH, "torch not available")
    def test_finalize_pth_content(self):
        intersection = np.array([1.0, 2.0])
        union = np.array([10.0, 20.0])
        target = np.array([5.0, 10.0])
        self.writer.finalize(intersection=intersection, union=union, target=target)
        pth_path = os.path.join(self.tmpdir, "Area_5.pth")
        data = torch.load(pth_path, weights_only=False)
        np.testing.assert_array_equal(data["intersection"], intersection)
        np.testing.assert_array_equal(data["union"], union)
        np.testing.assert_array_equal(data["target"], target)

    @unittest.skipIf(not HAS_TORCH, "torch not available")
    def test_finalize_with_missing_kwargs_does_nothing(self):
        """缺少必要参数时 finalize 不应创建文件。"""
        self.writer.finalize(intersection=np.array([1.0]))
        pth_path = os.path.join(self.tmpdir, "Area_5.pth")
        self.assertFalse(os.path.isfile(pth_path))

    def test_split_attribute(self):
        self.assertEqual(self.writer.split, "Area_5")


# ===========================================================================
#  8. Tester 集成（mock 测试）
# ===========================================================================


class TestTesterIntegration(unittest.TestCase):
    """
    测试 benchmark_writer 在 Tester 中的使用模式。
    不依赖 GPU/模型/数据集，仅验证 Writer 生命周期调用链正确。
    """

    def test_full_lifecycle_scannet(self):
        """模拟 ScanNet 数据集的完整 Writer 生命周期。"""
        tmpdir = tempfile.mkdtemp()
        ds = _FakeDataset(class2id=np.arange(20))
        writer = create_benchmark_writer("ScanNetDataset", tmpdir, ds)
        self.assertIsNotNone(writer)

        # 1. setup
        writer.setup()
        self.assertTrue(os.path.isdir(os.path.join(tmpdir, "submit")))

        # 2. write multiple samples
        for i in range(3):
            pred = _make_pred(n=100 + i * 50)
            writer.write(f"scene_{i:04d}", pred)

        # 3. pred_for_eval (identity for ScanNet)
        pred = _make_pred(n=50)
        eval_pred = writer.pred_for_eval(pred)
        np.testing.assert_array_equal(eval_pred, pred)

        # 4. finalize (no-op for ScanNet)
        writer.finalize()

    def test_full_lifecycle_scannetpp(self):
        """模拟 ScanNet++ 数据集的 top-3 工作流。"""
        tmpdir = tempfile.mkdtemp()
        writer = create_benchmark_writer("ScanNetPPDataset", tmpdir)
        self.assertIsNotNone(writer)
        self.assertEqual(writer.topk, 3)

        writer.setup()
        pred = _make_pred_topk(n=200, k=3)
        writer.write("scene_pp_001", pred)

        # pred_for_eval 应将 (N, 3) -> (N,)
        eval_pred = writer.pred_for_eval(pred)
        self.assertEqual(eval_pred.ndim, 1)
        self.assertEqual(eval_pred.shape[0], 200)
        np.testing.assert_array_equal(eval_pred, pred[:, 0])

    def test_full_lifecycle_semantic_kitti(self):
        """模拟 SemanticKITTI 二进制 .label 写入。"""
        tmpdir = tempfile.mkdtemp()
        mapping = {i: i * 10 for i in range(6)}
        ds = _FakeDataset(learning_map_inv=mapping)
        writer = create_benchmark_writer("SemanticKITTIDataset", tmpdir, ds)
        self.assertIsNotNone(writer)

        writer.setup()
        pred = np.array([0, 1, 2, 3])
        writer.write("08_000050", pred)

        out_path = os.path.join(
            tmpdir, "submit", "sequences", "08", "predictions", "000050.label"
        )
        result = np.fromfile(out_path, dtype=np.uint32)
        np.testing.assert_array_equal(result, np.array([0, 10, 20, 30], dtype=np.uint32))

    def test_full_lifecycle_nuscenes(self):
        """模拟 NuScenes 二进制 .bin 写入 + submission.json。"""
        tmpdir = tempfile.mkdtemp()
        writer = create_benchmark_writer("NuScenesDataset", tmpdir)
        self.assertIsNotNone(writer)

        writer.setup()
        json_path = os.path.join(tmpdir, "submit", "test", "submission.json")
        self.assertTrue(os.path.isfile(json_path))

        pred = np.array([0, 1, 2, 3])
        writer.write("sample_tok", pred)
        bin_path = os.path.join(
            tmpdir, "submit", "lidarseg", "test", "sample_tok_lidarseg.bin"
        )
        result = np.fromfile(bin_path, dtype=np.uint8)
        np.testing.assert_array_equal(result, np.array([1, 2, 3, 4], dtype=np.uint8))

    @unittest.skipIf(not HAS_TORCH, "torch not available")
    def test_full_lifecycle_s3dis(self):
        """模拟 S3DIS 6-fold 交叉验证工作流。"""
        tmpdir = tempfile.mkdtemp()
        ds = _FakeDataset(split="Area_1")
        writer = create_benchmark_writer("S3DISDataset", tmpdir, ds)
        self.assertIsNotNone(writer)

        # setup/write 都是空操作
        writer.setup()
        writer.write("room_test", np.array([0, 1, 2]))

        # finalize 保存 .pth
        intersection = np.array([10.0, 20.0])
        union = np.array([100.0, 200.0])
        target = np.array([50.0, 100.0])
        writer.finalize(intersection=intersection, union=union, target=target)

        pth_path = os.path.join(tmpdir, "Area_1.pth")
        self.assertTrue(os.path.isfile(pth_path))
        data = torch.load(pth_path, weights_only=False)
        np.testing.assert_array_equal(data["intersection"], intersection)

    def test_unknown_dataset_returns_none(self):
        """未知数据集类型不应影响推理流程。"""
        writer = create_benchmark_writer("CustomDataset", "/tmp/test")
        self.assertIsNone(writer)
        # Tester 中的 `if benchmark_writer is not None:` 分支不会执行

    def test_topk_property_controls_decode(self):
        """验证 topk 属性可用于控制预测解码方式。"""
        for dtype, expected_topk in [
            ("ScanNetDataset", 1),
            ("ScanNet200Dataset", 1),
            ("ScanNetPPDataset", 3),
            ("SemanticKITTIDataset", 1),
            ("NuScenesDataset", 1),
            ("S3DISDataset", 1),
        ]:
            ds = _FakeDataset(
                class2id=np.arange(20),
                learning_map_inv={0: 0},
                split="Area_1",
            )
            w = create_benchmark_writer(dtype, "/tmp/test", ds)
            self.assertEqual(w.topk, expected_topk, f"{dtype}: topk should be {expected_topk}")

    def test_benchmark_writer_without_dataset(self):
        """部分 Writer 可以在 dataset=None 时创建（优雅降级）。"""
        # ScanNetPP 和 NuScenes 不需要 dataset 属性
        w1 = create_benchmark_writer("ScanNetPPDataset", "/tmp/test")
        self.assertIsNotNone(w1)
        w2 = create_benchmark_writer("NuScenesDataset", "/tmp/test")
        self.assertIsNotNone(w2)

    def test_scannet_writer_without_class2id_graceful(self):
        """ScanNetBenchmarkWriter 无 class2id 时 write 不崩溃。"""
        tmpdir = tempfile.mkdtemp()
        w = ScanNetBenchmarkWriter(tmpdir, dataset=None)
        w.setup()
        # write 应跳过（因为 class2id 为 None）
        pred = _make_pred(n=10)
        w.write("test_scene", pred)  # 不应抛异常
        # 不应生成文件
        submit_dir = os.path.join(tmpdir, "submit")
        if os.path.isdir(submit_dir):
            self.assertEqual(len(os.listdir(submit_dir)), 0)


if __name__ == "__main__":
    unittest.main()
