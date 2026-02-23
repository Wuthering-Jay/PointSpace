"""
Point Cloud Writers

提供统一的点云结果写入接口，通过 WRITERS 注册表实现格式与任务的解耦。
遵循开闭原则 (OCP)：新增格式仅需编写新子类并注册，无需修改已有代码。

已注册的 Writer:
    - LASWriter:  LAS/LAZ 格式（完整实现）
    - PLYWriter:  PLY 格式（占位符）
    - PCDWriter:  PCD 格式（占位符）

Usage:
    from pointspace.writers import build_writer, WRITERS

    writer = build_writer(dict(type="LASWriter", save_dir="output/", source_dir="data/raw/"))
    writer.write("scene_001", coord, pred_sem=pred_labels)
"""

from .builder import WRITERS, build_writer
from .base_writer import BaseWriter
from .las_writer import LASWriter
from .ply_writer import PLYWriter
from .pcd_writer import PCDWriter
from .benchmark import create_benchmark_writer, BaseBenchmarkWriter

__all__ = [
    "WRITERS",
    "build_writer",
    "BaseWriter",
    "LASWriter",
    "PLYWriter",
    "PCDWriter",
    "create_benchmark_writer",
    "BaseBenchmarkWriter",
]
