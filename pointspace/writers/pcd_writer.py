"""
PCD Writer (占位符)

PCD 格式点云写入器的占位实现。

PCD (Point Cloud Data) 是 PCL (Point Cloud Library) 定义的原生点云格式，
也被 Open3D 等库广泛支持。

未来实现要点:
    - 支持 ASCII 和 Binary 两种编码模式
    - 支持灵活的 FIELDS 定义（x, y, z, rgb, normal_x, label 等）
    - 语义分割结果可写入 'label' 字段
    - 实例分割结果可写入 'instance_id' 字段
    - 推荐使用 open3d 或手动写入 PCD 格式头 + 二进制数据

依赖（未来）: open3d 或 pypcd4

Author: PointSpace Team
"""

import numpy as np
from .builder import WRITERS
from .base_writer import BaseWriter


@WRITERS.register_module()
class PCDWriter(BaseWriter):
    """
    PCD 格式点云写入器。

    当前为占位实现，调用 write() 将抛出 NotImplementedError。

    Args:
        save_dir (str): 输出文件保存目录。
        binary (bool): 是否使用二进制模式（默认 True，性能更优）。
    """

    def __init__(self, save_dir: str, binary: bool = True):
        super().__init__(save_dir)
        self.binary = binary

    def write(self, data_name: str, coord: np.ndarray, **kwargs) -> str:
        """
        将推理结果写入 PCD 文件。

        Args:
            data_name (str): 数据名称。
            coord (np.ndarray): 点坐标 (N, 3)。
            **kwargs: 推理结果字段（pred_sem, pred_ins, color 等）。

        Raises:
            NotImplementedError: 当前版本尚未实现。
        """
        # TODO: 实现 PCD 格式写入
        #   1. 构造 PCD 文件头（VERSION, FIELDS, SIZE, TYPE, COUNT, WIDTH, HEIGHT, POINTS, DATA）
        #   2. pred_sem -> 'label' FIELD (I4)
        #   3. pred_ins -> 'instance_id' FIELD (I4)
        #   4. color -> 将 RGB 打包为 float32 或分为三个 U1 字段
        #   5. 写入 ASCII 或 Binary 格式数据
        raise NotImplementedError(
            "PCDWriter 尚未实现，请在未来版本中补充 PCD 格式写入逻辑。"
        )
