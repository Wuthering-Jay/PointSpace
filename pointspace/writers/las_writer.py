"""
LAS / LAZ Writer

完整的 LAS/LAZ 格式点云写入器，支持：
  - 从原始 LAS/LAZ 文件复制头信息和已有点维度（保留 Scale, Offset, CRS,
    GPS time, Intensity, RGB, 回波等所有元数据及自定义字段）
  - 无原始文件时从零创建 LAS 文件
  - 语义分割结果写入 classification 字段
  - 实例分割结果通过 ExtraBytesParams 添加 instance_id 自定义维度
  - 为未来任务（全景分割、目标检测、回归）预留扩展位

依赖: laspy >= 2.0

Author: PointSpace Team
"""

import os
import glob
import logging
import warnings

import numpy as np

try:
    import laspy
    from laspy import ExtraBytesParams
except ImportError:
    laspy = None

from .builder import WRITERS
from .base_writer import BaseWriter

logger = logging.getLogger(__name__)


@WRITERS.register_module()
class LASWriter(BaseWriter):
    """
    将推理结果保存为 LAS/LAZ 格式。

    工作模式:
        1. **有源文件模式** (source_dir 不为 None)：
           读取 source_dir 下同名的 .las/.laz 文件，原封不动保留其头信息
           （Scale, Offset, CRS, VLRs, EVLRs）和所有已有点维度
           （GPS time, Intensity, RGB, 回波, 自定义字段等），
           仅覆写/追加推理结果字段。

        2. **无源文件模式** (source_dir 为 None)：
           从零创建 LAS 文件，使用默认 point_format=2（含 RGB），
           point 坐标由传入的 coord 填充。

    Args:
        save_dir (str): 输出文件保存目录。
        source_dir (str | None): 原始 LAS/LAZ 文件所在目录。
            为 None 时退化为从零创建模式。
        compressed (bool): 是否输出为 .laz 压缩格式，默认 False（输出 .las）。
    """

    # 输出扩展名映射
    _EXT_MAP = {False: ".las", True: ".laz"}

    def __init__(self, save_dir: str, source_dir: str = None, compressed: bool = False):
        super().__init__(save_dir)
        if laspy is None:
            raise ImportError(
                "LASWriter 需要 laspy 库，请通过 `pip install laspy[lazrs]` 安装。"
            )
        self.source_dir = source_dir
        self.compressed = compressed
        self._ext = self._EXT_MAP[compressed]

    # ------------------------------------------------------------------
    #  核心写入接口
    # ------------------------------------------------------------------

    def write(self, data_name: str, coord: np.ndarray = None, **kwargs) -> str:
        """
        将点云坐标及推理结果写入 LAS/LAZ 文件。

        Args:
            data_name (str): 场景/数据名称（不含扩展名），
                会被用于查找源文件和生成输出文件名。
            coord (np.ndarray | None): 点坐标 (N, 3)，float64。
                在有源文件模式 (source_dir) 下允许为 None，此时坐标
                和点数从源文件获取。无源文件且 coord 为 None 时抛异常。
            **kwargs: 推理结果字段，支持:
                pred_sem (np.ndarray): 语义分割标签 (N,)
                pred_ins (np.ndarray): 实例分割 ID (N,)
                pred_panoptic (np.ndarray): 全景分割标签（预留）
                pred_bbox (np.ndarray): 3D 检测框（预留）
                pred_reg (np.ndarray): 回归值（预留）
                color (np.ndarray): RGB 颜色 (N, 3)（仅无源文件模式使用）
                extra_dims (dict): 额外自定义维度 {name: (np.ndarray, dtype_str)}

        Returns:
            str: 写入的文件完整路径。

        Raises:
            ValueError: 当 coord 与预测结果的点数不匹配时，
                或无源文件且 coord 为 None 时。
        """
        out_path = os.path.join(self.save_dir, f"{data_name}{self._ext}")

        # ---------- 获取 LAS 对象（有源 / 无源两种路径） ----------
        las = self._load_or_create(data_name, coord, **kwargs)
        n_points = len(las.points)

        # ---------- 解析 kwargs 并写入对应字段 ----------
        self._apply_predictions(las, n_points, **kwargs)

        # ---------- 写入磁盘 ----------
        las.write(out_path)
        logger.info(f"LASWriter: 已保存 {n_points} 个点 -> {out_path}")
        return out_path

    # ------------------------------------------------------------------
    #  私有方法: 加载源文件 / 从零创建
    # ------------------------------------------------------------------

    def _load_or_create(
        self, data_name: str, coord: np.ndarray = None, **kwargs
    ) -> "laspy.LasData":
        """
        尝试从 source_dir 加载同名 LAS/LAZ 文件；失败则从零创建。

        Args:
            data_name: 场景名称。
            coord: 点坐标，可为 None（仅在有源文件时允许）。

        Returns:
            laspy.LasData

        Raises:
            ValueError: 当 coord 为 None 且无法从源文件获取坐标时。
        """
        if self.source_dir is not None:
            las = self._try_load_source(data_name)
            if las is not None:
                return las
            # 加载失败，回退到从零创建并发出警告
            warnings.warn(
                f"LASWriter: 在 source_dir='{self.source_dir}' 下"
                f"未找到与 '{data_name}' 匹配的 LAS/LAZ 文件，"
                f"将从零创建 LAS 文件。",
                RuntimeWarning,
            )

        if coord is None:
            raise ValueError(
                f"LASWriter: coord 为 None 且没有可用的源文件。"
                f"请在配置中指定 source_dir 并确保存在同名 LAS/LAZ 文件，"
                f"或在调用 write() 时传入 coord 参数。"
            )
        return self._create_new(coord, **kwargs)

    def _try_load_source(self, data_name: str):
        """
        在 source_dir 下查找同名 .las 或 .laz 文件并读取。

        Returns:
            laspy.LasData | None
        """
        for ext in (".las", ".laz"):
            candidate = os.path.join(self.source_dir, f"{data_name}{ext}")
            if os.path.isfile(candidate):
                try:
                    las = laspy.read(candidate)
                    logger.info(f"LASWriter: 已加载源文件 {candidate}")
                    return las
                except Exception as e:
                    warnings.warn(
                        f"LASWriter: 读取源文件 '{candidate}' 时出错: {e}",
                        RuntimeWarning,
                    )
        return None

    def _create_new(self, coord: np.ndarray, **kwargs) -> "laspy.LasData":
        """
        从零创建 LAS 文件（无源文件模式）。

        使用 point_format=2（含 RGB 字段），file_version="1.2"。
        """
        # point_format=2 支持 RGB 字段；file_version=1.2 兼容性较好
        header = laspy.LasHeader(point_format=2, version="1.2")

        # 根据坐标范围自动设置 scale 和 offset
        coord = np.asarray(coord, dtype=np.float64)
        mins = coord.min(axis=0)
        maxs = coord.max(axis=0)
        header.offsets = mins
        # scale: 毫米精度（0.001），如果范围很小则使用更精细的值
        header.scales = np.array([0.001, 0.001, 0.001])

        las = laspy.LasData(header)
        las.x = coord[:, 0]
        las.y = coord[:, 1]
        las.z = coord[:, 2]

        # 如果传入了颜色信息，写入 RGB
        color = kwargs.get("color", None)
        if color is not None:
            color = np.asarray(color)
            # LAS RGB 字段为 uint16（0-65535），若传入 uint8 则需缩放
            if color.max() <= 255:
                color = (color.astype(np.uint16) * 257)  # 0-255 -> 0-65535
            las.red = color[:, 0]
            las.green = color[:, 1]
            las.blue = color[:, 2]

        return las

    # ------------------------------------------------------------------
    #  私有方法: 将推理结果写入 LAS 字段
    # ------------------------------------------------------------------

    def _apply_predictions(self, las: "laspy.LasData", n_points: int, **kwargs):
        """
        解析 kwargs 中的预测结果并写入对应的 LAS 点维度。

        遵循开闭原则: 新增任务只需在此处添加 elif 分支或新增私有方法，
        无需修改 write() 的公开接口。
        """

        # ========== 语义分割 (Semantic Segmentation) ==========
        pred_sem = kwargs.get("pred_sem", None)
        if pred_sem is not None:
            pred_sem = np.asarray(pred_sem)
            if pred_sem.shape[0] != n_points:
                raise ValueError(
                    f"pred_sem 长度 ({pred_sem.shape[0]}) 与点数 ({n_points}) 不匹配"
                )
            # LAS 标准 classification 字段为 uint8
            las.classification = pred_sem.astype(np.uint8)
            logger.debug(
                f"  -> classification 字段已写入 "
                f"(unique labels: {np.unique(pred_sem).tolist()})"
            )

        # ========== 实例分割 (Instance Segmentation) ==========
        pred_ins = kwargs.get("pred_ins", None)
        if pred_ins is not None:
            pred_ins = np.asarray(pred_ins)
            if pred_ins.shape[0] != n_points:
                raise ValueError(
                    f"pred_ins 长度 ({pred_ins.shape[0]}) 与点数 ({n_points}) 不匹配"
                )
            self._add_extra_dim(las, "instance_id", pred_ins.astype(np.int32), np.int32)
            logger.debug(
                f"  -> instance_id 字段已写入 "
                f"(unique ids: {len(np.unique(pred_ins))})"
            )

        # ========== 全景分割 (Panoptic Segmentation) ==========
        # TODO: 全景分割通常需要同时编码语义类别和实例 ID，
        #       可以拆分为 classification + panoptic_id 两个字段，
        #       或使用单个 uint32 编码 (semantic_id * offset + instance_id)。
        pred_panoptic = kwargs.get("pred_panoptic", None)
        if pred_panoptic is not None:
            pass  # 预留: 未来实现全景分割结果写入

        # ========== 3D 目标检测 (3D Object Detection - Bounding Boxes) ==========
        # TODO: 检测结果通常不是逐点的，而是一组 bounding box。
        #       可以考虑：
        #       1. 将 bbox 信息写入 LAS 的 VLR (Variable Length Record)
        #       2. 将每个点对应的 bbox ID 作为额外维度写入
        #       3. 同时生成一个伴随的 JSON/CSV 文件存储 bbox 参数
        pred_bbox = kwargs.get("pred_bbox", None)
        if pred_bbox is not None:
            pass  # 预留: 未来实现 3D 检测框结果写入

        # ========== 回归任务 (Regression) ==========
        # TODO: 逐点回归值可以作为额外的 float64 维度写入 LAS，
        #       例如 height_pred, curvature 等。
        pred_reg = kwargs.get("pred_reg", None)
        if pred_reg is not None:
            pass  # 预留: 未来实现回归结果写入

        # ========== 通用自定义维度 (Extra Dimensions) ==========
        # 允许用户直接指定 {字段名: (数据数组, dtype)} 的字典来写入任意维度
        extra_dims = kwargs.get("extra_dims", None)
        if extra_dims is not None:
            for dim_name, (dim_data, dim_dtype) in extra_dims.items():
                dim_data = np.asarray(dim_data)
                if dim_data.shape[0] != n_points:
                    raise ValueError(
                        f"extra_dims['{dim_name}'] 长度 ({dim_data.shape[0]}) "
                        f"与点数 ({n_points}) 不匹配"
                    )
                self._add_extra_dim(las, dim_name, dim_data, dim_dtype)
                logger.debug(f"  -> 自定义维度 '{dim_name}' 已写入")

    # ------------------------------------------------------------------
    #  工具方法
    # ------------------------------------------------------------------

    @staticmethod
    def _add_extra_dim(
        las: "laspy.LasData", name: str, data: np.ndarray, dtype
    ):
        """
        安全地向 LAS 文件添加或覆写一个额外维度（Extra Bytes）。

        如果该维度已存在（例如源文件中已有），则直接覆写值；
        否则通过 ExtraBytesParams 新建。

        Args:
            las: laspy.LasData 对象
            name: 维度名称
            data: 数据数组
            dtype: numpy dtype
        """
        # 检查维度是否已存在于点记录中
        existing_dims = list(las.point_format.dimension_names)
        if name in existing_dims:
            # 已存在，直接赋值覆写
            setattr(las, name, data)
        else:
            # 不存在，通过 ExtraBytesParams 添加
            las.add_extra_dim(ExtraBytesParams(name=name, type=dtype))
            setattr(las, name, data)
