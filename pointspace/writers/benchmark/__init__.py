"""
Benchmark Writers

各数据集竞赛/评测平台提交格式的写入器集合。
通过 create_benchmark_writer() 工厂函数根据数据集类型自动创建。

已支持的数据集:
    - ScanNet / ScanNet200
    - ScanNet++
    - SemanticKITTI
    - NuScenes
    - S3DIS

Usage:
    from pointspace.writers.benchmark import create_benchmark_writer

    writer = create_benchmark_writer("ScanNetDataset", save_dir, dataset)
"""

from .builder import create_benchmark_writer
from .base_benchmark_writer import BaseBenchmarkWriter
from .scannet_writer import ScanNetBenchmarkWriter
from .scannetpp_writer import ScanNetPPBenchmarkWriter
from .semantic_kitti_writer import SemanticKITTIBenchmarkWriter
from .nuscenes_writer import NuScenesBenchmarkWriter
from .s3dis_writer import S3DISBenchmarkWriter

__all__ = [
    "create_benchmark_writer",
    "BaseBenchmarkWriter",
    "ScanNetBenchmarkWriter",
    "ScanNetPPBenchmarkWriter",
    "SemanticKITTIBenchmarkWriter",
    "NuScenesBenchmarkWriter",
    "S3DISBenchmarkWriter",
]
