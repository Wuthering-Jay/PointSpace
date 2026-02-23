"""
ScanNet++ Benchmark Writer

处理 ScanNet++ 数据集的提交文件格式。
ScanNet++ 提交 top-3 预测，评测使用 top-1。

Author: PointSpace Team
"""

import os
import numpy as np

from .base_benchmark_writer import BaseBenchmarkWriter


class ScanNetPPBenchmarkWriter(BaseBenchmarkWriter):
    """
    ScanNet++ 竞赛提交格式写入器。

    特殊之处:
        - topk = 3：模型输出取 top-3 类别
        - 提交格式：逗号分隔的 int32
        - pred_for_eval：评测时仅使用 top-1（pred[:, 0]）

    提交格式: save_dir/submit/{data_name}.txt
    内容: 每行 3 个逗号分隔的整数，为 top-3 预测类别。

    Args:
        save_dir (str): 提交文件根目录。
        dataset: 测试数据集对象。
    """

    topk: int = 3

    def write(self, data_name: str, pred: np.ndarray, **kwargs):
        np.savetxt(
            os.path.join(self.save_dir, "submit", "{}.txt".format(data_name)),
            pred.astype(np.int32),
            delimiter=",",
            fmt="%d",
        )

    def pred_for_eval(self, pred: np.ndarray) -> np.ndarray:
        """ScanNet++ 评测使用 top-1 预测。"""
        return pred[:, 0]
