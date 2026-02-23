"""
Benchmark Writer 基类

定义了所有 Benchmark 提交文件 Writer 的抽象接口。
Benchmark Writer 专门处理各个数据集竞赛/评测平台要求的提交文件格式，
与通用 Writer（LAS, PLY, PCD 等用于实际生产输出）相互独立。

设计思路:
    - 每个数据集的提交格式由对应的子类实现
    - 通过 topk 属性控制预测解码方式（argmax vs topk）
    - 通过 pred_for_eval() 支持提交格式与评测格式不同的情况（如 ScanNet++）
    - 通过 finalize() 支持评测循环结束后的收尾操作（如 S3DIS 的 .pth 保存）

Author: PointSpace Team
"""

import os
import logging
from abc import ABC, abstractmethod

import numpy as np

logger = logging.getLogger(__name__)


class BaseBenchmarkWriter(ABC):
    """
    Benchmark 提交文件写入器的抽象基类。

    子类需实现 write() 方法，可选重写 setup() / pred_for_eval() / finalize()。

    Attributes:
        topk (int): 预测解码时取 top-k 类别。默认 1（argmax），
                     ScanNet++ 需要设为 3。
    """

    topk: int = 1

    def __init__(self, save_dir: str, dataset=None):
        """
        Args:
            save_dir (str): 提交文件的根保存目录（通常是 cfg.save_path/result）。
            dataset: 测试数据集对象，子类可从中提取必要的映射表等属性。
        """
        self.save_dir = save_dir
        self.dataset = dataset

    def setup(self):
        """
        创建提交所需的目录结构。在评测循环开始前、仅在主进程上调用。
        默认实现创建 save_dir/submit 目录。
        """
        os.makedirs(os.path.join(self.save_dir, "submit"), exist_ok=True)

    @abstractmethod
    def write(self, data_name: str, pred: np.ndarray, **kwargs):
        """
        写入单个样本的提交文件。

        Args:
            data_name (str): 数据/场景名称。
            pred (np.ndarray): 预测结果数组。
            **kwargs: 子类可能需要的额外参数。
        """
        raise NotImplementedError

    def pred_for_eval(self, pred: np.ndarray) -> np.ndarray:
        """
        将提交用的 pred 变换为评测用的 pred。

        大多数数据集提交格式与评测格式一致，默认直接返回原 pred。
        ScanNet++ 提交的是 top-3 预测，而评测使用 top-1。

        Args:
            pred (np.ndarray): 提交用的预测结果。

        Returns:
            np.ndarray: 评测用的预测结果。
        """
        return pred

    def finalize(self, **kwargs):
        """
        评测循环结束后的收尾工作。仅在主进程上调用。

        默认为空操作。S3DIS 需要在此保存交叉验证的中间结果。

        Args:
            **kwargs: 可能包含 intersection, union, target 等评测指标。
        """
        pass
