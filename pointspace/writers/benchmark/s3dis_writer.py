"""
S3DIS Benchmark Writer

处理 S3DIS 数据集的评测结果保存。
S3DIS 没有标准的在线提交格式，但 6-fold 交叉验证需要保存各个 Area 的指标
以便后续合并计算。

Author: PointSpace Team
"""

import os
import numpy as np
import torch

from .base_benchmark_writer import BaseBenchmarkWriter


class S3DISBenchmarkWriter(BaseBenchmarkWriter):
    """
    S3DIS 评测结果写入器。

    与其他数据集不同，S3DIS 不需要逐样本写入提交文件，
    而是在 finalize() 中保存整个 split 的交叉验证指标。

    保存格式: save_dir/{split}.pth
    内容: dict(intersection=..., union=..., target=...)

    Args:
        save_dir (str): 结果保存根目录。
        dataset: 测试数据集对象，需包含 split 属性。
    """

    def __init__(self, save_dir: str, dataset=None):
        super().__init__(save_dir, dataset)
        self.split = getattr(dataset, "split", "unknown")

    def setup(self):
        """S3DIS 不需要创建 submit 目录。"""
        pass

    def write(self, data_name: str, pred: np.ndarray, **kwargs):
        """S3DIS 不需要逐样本写入提交文件。"""
        pass

    def finalize(self, **kwargs):
        """
        保存 6-fold 交叉验证的中间结果。

        Args:
            **kwargs: 需包含 intersection, union, target (np.ndarray)。
        """
        intersection = kwargs.get("intersection")
        union = kwargs.get("union")
        target = kwargs.get("target")
        if intersection is not None and union is not None and target is not None:
            torch.save(
                dict(intersection=intersection, union=union, target=target),
                os.path.join(self.save_dir, f"{self.split}.pth"),
            )
