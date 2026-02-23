"""
PLY Writer (占位符)

PLY 格式点云写入器的占位实现。

PLY (Polygon File Format / Stanford Triangle Format) 是一种灵活的多边形和点云文件格式，
广泛用于 3D 扫描和计算机图形学领域。

未来实现要点:
    - 支持 ASCII 和 Binary 两种编码模式
    - 支持任意 property 的灵活写入（坐标、颜色、法线、自定义标量等）
    - 语义分割结果可写入 'label' 或 'scalar_classification' property
    - 实例分割结果可写入 'scalar_instance_id' property
    - 推荐使用 plyfile 或 open3d 库实现

依赖（未来）: plyfile 或 open3d

Author: PointSpace Team
"""

import numpy as np
from .builder import WRITERS
from .base_writer import BaseWriter


@WRITERS.register_module()
class PLYWriter(BaseWriter):
    """
    PLY 格式点云写入器。

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
        将推理结果写入 PLY 文件。

        Args:
            data_name (str): 数据名称。
            coord (np.ndarray): 点坐标 (N, 3)。
            **kwargs: 推理结果字段（pred_sem, pred_ins, color 等）。

        Raises:
            NotImplementedError: 当前版本尚未实现。
        """
        # TODO: 实现 PLY 格式写入
        #   1. 根据 coord 和 kwargs 中的字段构造 PLY vertex 元素
        #   2. pred_sem -> 'label' property (int)
        #   3. pred_ins -> 'instance_id' property (int)
        #   4. color -> 'red', 'green', 'blue' properties (uint8)
        #   5. 使用 plyfile.PlyData 或 open3d 写入
        raise NotImplementedError(
            "PLYWriter 尚未实现，请在未来版本中补充 PLY 格式写入逻辑。"
        )
