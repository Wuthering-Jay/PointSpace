"""
ScanNet / ScanNet200 Benchmark Writer

处理 ScanNet 和 ScanNet200 数据集的提交文件格式。
提交格式：每个场景一个 .txt 文件，每行一个整数标签（使用 class2id 映射）。

Author: PointSpace Team
"""

import os
import numpy as np

from .base_benchmark_writer import BaseBenchmarkWriter


class ScanNetBenchmarkWriter(BaseBenchmarkWriter):
    """
    ScanNet / ScanNet200 竞赛提交格式写入器。

    提交格式: save_dir/submit/{data_name}.txt
    内容: 每行一个整数，为 class2id[pred] 的映射结果。

    Args:
        save_dir (str): 提交文件根目录。
        dataset: 测试数据集对象，需包含 class2id 属性。
    """

    def __init__(self, save_dir: str, dataset=None):
        super().__init__(save_dir, dataset)
        self.class2id = getattr(dataset, "class2id", None)

    def write(self, data_name: str, pred: np.ndarray, **kwargs):
        if self.class2id is not None:
            np.savetxt(
                os.path.join(self.save_dir, "submit", "{}.txt".format(data_name)),
                self.class2id[pred].reshape([-1, 1]),
                fmt="%d",
            )
