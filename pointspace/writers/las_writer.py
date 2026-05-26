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

           对于 **pred_coord（外部提供的稠密坐标输出）**，上述模式有所不同：
           仅从源文件借用 scale、offset 和坐标系 VLR（GeoKey / WKT），
           输出始终为标准 LAS 1.2 / point_format=0，以保证最大兼容性。

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

    def __init__(
        self,
        save_dir: str,
        source_dir: str = None,
        compressed: bool = False,
        classification: int = None,
    ):
        super().__init__(save_dir)
        if laspy is None:
            raise ImportError(
                "LASWriter 需要 laspy 库，请通过 `pip install laspy[lazrs]` 安装。"
            )
        self.source_dir = source_dir
        self.compressed = compressed
        self.classification = classification
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
                pred_coord (np.ndarray): 外部提供的预测坐标 (Q, 3)。当提供时
                    忽略 coord 和源文件，直接从 pred_coord 创建新 LAS。
                pred_sem (np.ndarray): 语义分割标签 (N,)
                pred_ins (np.ndarray): 实例分割 ID (N,)
                pred_panoptic (np.ndarray): 全景分割标签（预留）
                pred_bbox (np.ndarray): 3D 检测框（预留）
                pred_reg (np.ndarray): 回归值（预留）
                slope (np.ndarray): 派生坡度属性 (Q,)
                curvature (np.ndarray): 派生曲率属性 (Q,)
                color (np.ndarray): RGB 颜色 (N, 3)（仅无源文件模式使用）
                extra_dims (dict): 额外自定义维度 {name: (np.ndarray, dtype_str)}

        Returns:
            str: 写入的文件完整路径。

        Raises:
            ValueError: 当 coord 与预测结果的点数不匹配时，
                或无源文件且 coord 为 None 时。
        """
        out_path = os.path.join(self.save_dir, f"{data_name}{self._ext}")

        # ---------- pred_coord shortcut (dense coordinate output) ----------
        pred_coord = kwargs.pop("pred_coord", None)
        if pred_coord is not None:
            pred_coord = np.asarray(pred_coord, dtype=np.float64)
            # Borrow CRS / scale / offset from source file when source_dir is set
            src_las = (
                self._try_load_source(data_name)
                if self.source_dir is not None
                else None
            )
            if src_las is not None:
                las = self._create_from_source_header(src_las, pred_coord, **kwargs)
            else:
                las = self._create_new(pred_coord, **kwargs)
            n_points = len(las.points)
            self._apply_predictions(las, n_points, **kwargs)
            las.write(out_path)
            logger.info(f"LASWriter: 已保存 {n_points} 个点 (pred_coord) -> {out_path}")
            return out_path

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

    def _create_from_source_header(
        self, src_las: "laspy.LasData", coord: np.ndarray, **kwargs
    ) -> "laspy.LasData":
        """
        创建标准新 LAS 1.2 / point_format=0 文件，仅从源文件借用：
        - scale / offset（精度/基准）
        - 坐标系 VLR（GeoKeyDirectory、GeoDoubleParams、GeoAsciiParams、WKT OGC CRS）

        其他源文件头信息一律不复制，以保证最大的软件兼容性。
        """
        src_hdr = src_las.header

        # 始终使用 point_format=0 + version=1.2，确保软件兼容性
        new_header = laspy.LasHeader(point_format=0, version="1.2")

        # 1. 借用 scale & offset（坐标精度 / 平移基准）
        new_header.scales = np.asarray(src_hdr.scales, dtype=np.float64)
        new_header.offsets = np.asarray(src_hdr.offsets, dtype=np.float64)

        # 2. 仅复制 CRS 相关的 VLR（按 user_id / record_id 过滤）
        #    GeoTIFF keys:  user_id="LASF_Projection", record_id in (34735,34736,34737)
        #    OGC WKT:       user_id="LASF_Projection", record_id=2112
        CRS_RECORD_IDS = {34735, 34736, 34737, 2112}
        crs_vlrs = [
            v for v in src_hdr.vlrs
            if getattr(v, "user_id", "").strip().upper() == "LASF_PROJECTION"
            and getattr(v, "record_id", -1) in CRS_RECORD_IDS
        ]
        if crs_vlrs:
            new_header.vlrs = crs_vlrs

        las = laspy.LasData(new_header)
        coord = np.asarray(coord, dtype=np.float64)
        las.x = coord[:, 0]
        las.y = coord[:, 1]
        las.z = coord[:, 2]

        return las

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
        elif self.classification is not None:
            # 未传入 pred_sem 时，用 writer 配置的默认类别值填充所有点
            las.classification = np.full(
                n_points, self.classification, dtype=np.uint8
            )
            logger.debug(
                f"  -> classification 字段已写入 (default: {self.classification})"
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
        pred_reg = kwargs.get("pred_reg", None)
        if pred_reg is not None:
            pred_reg = np.asarray(pred_reg, dtype=np.float64)
            if pred_reg.ndim == 1:
                # Single-target: write as one extra dimension
                if pred_reg.shape[0] != n_points:
                    raise ValueError(
                        f"pred_reg 长度 ({pred_reg.shape[0]}) 与点数 ({n_points}) 不匹配"
                    )
                self._add_extra_dim(las, "reg_pred", pred_reg, np.float64)
                logger.debug("  -> reg_pred 字段已写入 (scalar)")
            elif pred_reg.ndim == 2:
                # Multi-target: write each column as reg_pred_0, reg_pred_1, …
                if pred_reg.shape[0] != n_points:
                    raise ValueError(
                        f"pred_reg 行数 ({pred_reg.shape[0]}) 与点数 ({n_points}) 不匹配"
                    )
                for d in range(pred_reg.shape[1]):
                    dim_name = f"reg_pred_{d}"
                    self._add_extra_dim(las, dim_name, pred_reg[:, d], np.float64)
                logger.debug(
                    f"  -> reg_pred_0..{pred_reg.shape[1]-1} 字段已写入 "
                    f"(multi-target, D={pred_reg.shape[1]})"
                )
            else:
                raise ValueError(
                    f"pred_reg 维度不合法: ndim={pred_reg.ndim}, 期望 1 或 2"
                )

        # ========== 派生地形属性 (Slope / Curvature) ==========
        slope = kwargs.get("slope", None)
        if slope is not None:
            slope = np.asarray(slope, dtype=np.float64)
            if slope.shape[0] != n_points:
                raise ValueError(
                    f"slope 长度 ({slope.shape[0]}) 与点数 ({n_points}) 不匹配"
                )
            self._add_extra_dim(las, "slope", slope, np.float64)
            logger.debug("  -> slope 字段已写入")

        curvature = kwargs.get("curvature", None)
        if curvature is not None:
            curvature = np.asarray(curvature, dtype=np.float64)
            if curvature.shape[0] != n_points:
                raise ValueError(
                    f"curvature 长度 ({curvature.shape[0]}) 与点数 ({n_points}) 不匹配"
                )
            self._add_extra_dim(las, "curvature", curvature, np.float64)
            logger.debug("  -> curvature 字段已写入")

        # ========== 超点分割 (Superpoint Partition) ==========
        oracle_pred = kwargs.get("oracle_pred", None)
        if oracle_pred is not None:
            oracle_pred = np.asarray(oracle_pred, dtype=np.int32)
            if oracle_pred.shape[0] != n_points:
                raise ValueError(
                    f"oracle_pred 长度 ({oracle_pred.shape[0]}) 与点数 ({n_points}) 不匹配"
                )
            self._add_extra_dim(las, "oracle_pred", oracle_pred, np.int32)
            logger.debug(
                f"  -> oracle_pred 字段已写入 "
                f"(unique labels: {np.unique(oracle_pred).tolist()})"
            )

        # ========== 超点分割 (Superpoint Partition) ==========
        # 支持多级超点标签，字段名格式：superpoint_level_1, superpoint_level_2, ...
        # 用于 EZ-SP 等可学习超点分割方法的输出
        for key, value in kwargs.items():
            if key.startswith("superpoint_level_"):
                sp_data = np.asarray(value, dtype=np.int32)
                if sp_data.shape[0] != n_points:
                    raise ValueError(
                        f"{key} 长度 ({sp_data.shape[0]}) 与点数 ({n_points}) 不匹配"
                    )
                self._add_extra_dim(las, key, sp_data, np.int32)
                logger.debug(
                    f"  -> {key} 字段已写入 "
                    f"(unique superpoints: {len(np.unique(sp_data))})"
                )

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
