"""
Base Writer

定义了所有 Writer 的抽象基类 BaseWriter。
所有格式特定的 Writer（LAS, PLY, PCD 等）都应继承此类并实现 write() 方法。

设计原则:
    - 开闭原则 (OCP)：对扩展开放、对修改关闭。新增格式仅需编写新子类并注册。
    - 通过 **kwargs 接收不定长的预测结果，为未来任务（全景分割、目标检测、回归等）
      预留好数据通道，无需修改基类接口。

Author: PointSpace Team
"""

import os
import logging
from abc import ABC, abstractmethod

import numpy as np

logger = logging.getLogger(__name__)


class BaseWriter(ABC):
    """
    点云结果写入器的抽象基类。

    所有子类必须实现 write() 方法。

    Args:
        save_dir (str): 输出文件的保存目录。不存在时自动创建。
    """

    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    @abstractmethod
    def write(self, data_name: str, coord: np.ndarray = None, **kwargs) -> str:
        """
        将推理结果写入文件。

        Args:
            data_name (str): 数据/场景名称，用于生成输出文件名。
            coord (np.ndarray | None): 点坐标，形状 (N, 3)。
                某些 Writer（如 LASWriter 有源文件模式）可从源文件获取坐标，此时允许为 None。
            **kwargs: 不定长的预测结果字段，由具体 Writer 子类自行解析。
                已规划的 key 约定（子类按需读取）：
                    - pred_sem    (np.ndarray): 语义分割标签, shape (N,), dtype int
                    - pred_ins    (np.ndarray): 实例分割 ID, shape (N,), dtype int
                    - pred_panoptic (np.ndarray): 全景分割标签（预留）
                    - pred_bbox   (np.ndarray): 3D 检测框（预留）
                    - pred_reg    (np.ndarray): 回归值（预留）
                    - color       (np.ndarray): RGB 颜色, shape (N, 3)（可选）
                    - extra_dims  (dict): 其他需要写入的自定义维度（可选）

        Returns:
            str: 实际写入文件的完整路径。
        """
        raise NotImplementedError
