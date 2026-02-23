"""
NuScenes Benchmark Writer

处理 NuScenes 数据集的提交文件格式。
提交格式：二进制 .bin 文件 + submission.json 元信息。

Author: PointSpace Team
"""

import json
import os
import numpy as np

from .base_benchmark_writer import BaseBenchmarkWriter


class NuScenesBenchmarkWriter(BaseBenchmarkWriter):
    """
    NuScenes 竞赛提交格式写入器。

    setup() 会创建 lidarseg/test 目录并写入 submission.json。

    提交格式: save_dir/submit/lidarseg/test/{data_name}_lidarseg.bin
    内容: 二进制 uint8 数组，值为 pred + 1（NuScenes label offset）。

    Args:
        save_dir (str): 提交文件根目录。
        dataset: 测试数据集对象。
    """

    def setup(self):
        os.makedirs(
            os.path.join(self.save_dir, "submit", "lidarseg", "test"), exist_ok=True
        )
        os.makedirs(
            os.path.join(self.save_dir, "submit", "test"), exist_ok=True
        )
        submission = dict(
            meta=dict(
                use_camera=False,
                use_lidar=True,
                use_radar=False,
                use_map=False,
                use_external=False,
            )
        )
        submission_path = os.path.join(
            self.save_dir, "submit", "test", "submission.json"
        )
        with open(submission_path, "w") as f:
            json.dump(submission, f, indent=4)

    def write(self, data_name: str, pred: np.ndarray, **kwargs):
        np.array(pred + 1).astype(np.uint8).tofile(
            os.path.join(
                self.save_dir,
                "submit",
                "lidarseg",
                "test",
                "{}_lidarseg.bin".format(data_name),
            )
        )
