"""
Benchmark Writer 构建器

提供工厂函数 create_benchmark_writer()，根据数据集类型自动创建对应的
Benchmark Writer 实例。

Usage:
    from pointspace.writers.benchmark import create_benchmark_writer

    writer = create_benchmark_writer(
        dataset_type="ScanNetDataset",
        save_dir="exp/result/",
        dataset=test_dataset,
    )
    if writer is not None:
        writer.setup()
        writer.write(data_name, pred)
"""

from .scannet_writer import ScanNetBenchmarkWriter
from .scannetpp_writer import ScanNetPPBenchmarkWriter
from .semantic_kitti_writer import SemanticKITTIBenchmarkWriter
from .nuscenes_writer import NuScenesBenchmarkWriter
from .s3dis_writer import S3DISBenchmarkWriter


# 数据集类型 -> Benchmark Writer 类的映射
_DATASET_TO_WRITER = {
    "ScanNetDataset": ScanNetBenchmarkWriter,
    "ScanNet200Dataset": ScanNetBenchmarkWriter,
    "ScanNetPPDataset": ScanNetPPBenchmarkWriter,
    "SemanticKITTIDataset": SemanticKITTIBenchmarkWriter,
    "NuScenesDataset": NuScenesBenchmarkWriter,
    "S3DISDataset": S3DISBenchmarkWriter,
}


def create_benchmark_writer(dataset_type: str, save_dir: str, dataset=None):
    """
    根据数据集类型自动创建对应的 Benchmark Writer。

    Args:
        dataset_type (str): 数据集类型名称（如 "ScanNetDataset"）。
        save_dir (str): 提交文件保存的根目录。
        dataset: 测试数据集对象（可选），Writer 会从中提取所需的映射表等属性。

    Returns:
        BaseBenchmarkWriter | None: 对应的 Writer 实例。
            未知数据集类型时返回 None（不影响推理流程）。
    """
    writer_cls = _DATASET_TO_WRITER.get(dataset_type)
    if writer_cls is None:
        return None
    return writer_cls(save_dir=save_dir, dataset=dataset)
