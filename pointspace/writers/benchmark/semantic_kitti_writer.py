"""
SemanticKITTI Benchmark Writer

处理 SemanticKITTI 数据集的提交文件格式。
提交格式：二进制 .label 文件，使用 learning_map_inv 映射回原始标签空间。

Author: PointSpace Team
"""

import os
import numpy as np

from .base_benchmark_writer import BaseBenchmarkWriter


class SemanticKITTIBenchmarkWriter(BaseBenchmarkWriter):
    """
    SemanticKITTI 竞赛提交格式写入器。

    提交格式: save_dir/submit/sequences/{seq}/predictions/{frame}.label
    内容: 二进制 uint32 数组，标签经 learning_map_inv 逆映射。

    data_name 格式: "{sequence}_{frame}"，如 "00_000000"。

    Args:
        save_dir (str): 提交文件根目录。
        dataset: 测试数据集对象，需包含 learning_map_inv 属性。
    """

    def __init__(self, save_dir: str, dataset=None):
        super().__init__(save_dir, dataset)
        self.learning_map_inv = getattr(dataset, "learning_map_inv", None)

    def write(self, data_name: str, pred: np.ndarray, **kwargs):
        # data_name 格式: "00_000000" -> sequence="00", frame="000000"
        sequence_name, frame_name = data_name.split("_")
        os.makedirs(
            os.path.join(
                self.save_dir, "submit", "sequences", sequence_name, "predictions"
            ),
            exist_ok=True,
        )
        submit = pred.astype(np.uint32)
        if self.learning_map_inv is not None:
            submit = np.vectorize(
                self.learning_map_inv.__getitem__
            )(submit).astype(np.uint32)
        submit.tofile(
            os.path.join(
                self.save_dir,
                "submit",
                "sequences",
                sequence_name,
                "predictions",
                f"{frame_name}.label",
            )
        )
