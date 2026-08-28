"""
LAS/LAZ and GeoTIFF Tile Processor

将坐标对齐的 LAS/LAZ 点云和 GeoTIFF 正射影像一起分块，支持：
- 点云与影像通过空间范围自动匹配，不要求文件同名或数量相同
- 复用 tile_las.py 的滑动窗口、最小点数合并和最大点数递归切分逻辑
- GeoTIFF 按窗口读取，不会将整幅大影像载入内存
- 多幅相交影像自动拼接，无影像覆盖区域填充为黑色（像素值 0）
- 使用 Safetensors 保存逐点像素坐标、影像覆盖有效标记和连续可观测度
"""

import copy
import json
import logging
import multiprocessing
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import laspy
import numpy as np
import rasterio
from safetensors import safe_open
from safetensors.numpy import load_file as load_safetensors
from safetensors.numpy import save_file as save_safetensors
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import reproject
from rasterio.windows import Window, from_bounds
from scipy.ndimage import maximum_filter
from tqdm import tqdm


# Try to import pointspace logger, fallback to standard logging
try:
    from pointspace.utils.logger import get_root_logger
except ImportError:
    def get_root_logger():
        """Fallback logger when pointspace is not available"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        return logging.getLogger(__name__)


class LASImageTileProcessor:
    """
    LAS/LAZ 点云与 GeoTIFF 影像联合分块处理器

    Args:
        pointcloud_path: 输入 LAS/LAZ 文件或目录
        image_path: 输入 GeoTIFF 文件或目录
        output_dir: 输出目录，内部创建 pointcloud、image、correspondence
        output_format: 点云输出格式，'las' 或 'laz'
        window_size: 分块窗口大小 (x_size, y_size)，单位与坐标系一致
        overlap: 是否启用重叠分块
        overlap_factor: 重叠因子，生成 overlap_factor^2 组偏移网格
        min_points: 最小点数阈值，小于此值的块合并到邻近块
        max_points: 最大点数阈值，超过此值的块递归切分
        save_orig_idx: 是否在点云中保存源文件内的原始点索引
        surface_only_valid: 是否仅将正射视角下的可见表面点标记为 valid=1
        surface_cell_size: 表面 DSM 栅格尺寸；None 或 'auto' 表示根据点云
            平均点间距和影像分辨率自动估算
        surface_radius: 点的水平遮挡作用半径；None 或 'auto' 表示采用一个
            surface_cell_size，设为 0 则不向相邻栅格扩展
        surface_z_tolerance: 点与局部最高表面的允许高差
        echo_decay: 连续可观测度随回波序号增加的指数衰减系数。第一回波
            权重为 1，第 r 回波权重为 exp(-echo_decay * (r - 1))

    Notes:
        输入影像必须具有相同的 CRS、波段数、数据类型、分辨率和无旋转
        仿射变换。LAS 缺少 CRS 时会假定其坐标系与影像一致并记录警告；
        LAS 和影像都具有 CRS 且不一致时会报错。
    """

    def __init__(
        self,
        pointcloud_path: Union[str, Path],
        image_path: Union[str, Path],
        output_dir: Union[str, Path],
        output_format: str = 'las',
        window_size: Tuple[float, float] = (50.0, 50.0),
        overlap: bool = True,
        overlap_factor: int = 1,
        min_points: int = 1000,
        max_points: Optional[int] = None,
        save_orig_idx: bool = True,
        surface_only_valid: bool = False,
        surface_cell_size: Optional[Union[float, str]] = None,
        surface_radius: Optional[Union[float, str]] = None,
        surface_z_tolerance: float = 0.15,
        echo_decay: float = 0.7,
    ):
        self.pointcloud_path = Path(pointcloud_path)
        self.image_path = Path(image_path)
        self.output_dir = Path(output_dir)
        self.output_format = output_format.lower().lstrip('.')
        self.window_size = tuple(float(value) for value in window_size)
        self.overlap = overlap
        self.overlap_factor = overlap_factor if overlap else 1
        self.min_points = min_points
        self.max_points = max_points
        self.save_orig_idx = save_orig_idx
        self.surface_only_valid = surface_only_valid
        self.surface_cell_size = surface_cell_size
        self.surface_radius = surface_radius
        self.surface_z_tolerance = float(surface_z_tolerance)
        self.echo_decay = float(echo_decay)

        self.logger = get_root_logger()
        self._validate_parameters()

        self.pointcloud_output_dir = self.output_dir / 'pointcloud'
        self.image_output_dir = self.output_dir / 'image'
        self.correspondence_output_dir = self.output_dir / 'correspondence'
        for directory in (
            self.pointcloud_output_dir,
            self.image_output_dir,
            self.correspondence_output_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        self.las_files = self._find_files(
            self.pointcloud_path, ('.las', '.laz'), 'LAS/LAZ'
        )
        self.image_files = self._find_files(
            self.image_path, ('.tif', '.tiff'), 'GeoTIFF'
        )
        self.raster_catalog = self._build_raster_catalog()
        self.reference_raster = self.raster_catalog[0]

    def _validate_parameters(self):
        """验证输入参数。"""
        if self.output_format not in ('las', 'laz'):
            raise ValueError("output_format must be 'las' or 'laz'")
        if len(self.window_size) != 2 or any(value <= 0 for value in self.window_size):
            raise ValueError('window_size must contain two positive values')
        if not isinstance(self.overlap_factor, int) or self.overlap_factor < 1:
            raise ValueError('overlap_factor must be a positive integer')
        if self.min_points is not None and self.min_points < 1:
            raise ValueError('min_points must be None or a positive integer')
        if self.max_points is not None and self.max_points < 1:
            raise ValueError('max_points must be None or a positive integer')
        if (
            self.min_points is not None
            and self.max_points is not None
            and self.min_points > self.max_points
        ):
            raise ValueError('min_points cannot be greater than max_points')
        self._validate_auto_or_nonnegative(
            self.surface_cell_size, 'surface_cell_size', allow_zero=False
        )
        self._validate_auto_or_nonnegative(
            self.surface_radius, 'surface_radius', allow_zero=True
        )
        if self.surface_z_tolerance < 0:
            raise ValueError('surface_z_tolerance must be non-negative')
        if self.echo_decay < 0:
            raise ValueError('echo_decay must be non-negative')

    @staticmethod
    def _validate_auto_or_nonnegative(value, name: str, allow_zero: bool):
        """验证自动参数为 None、'auto' 或有效数值。"""
        if value is None or (isinstance(value, str) and value.lower() == 'auto'):
            return
        if isinstance(value, str):
            raise ValueError(f"{name} must be None, 'auto', or a number")
        numeric_value = float(value)
        if numeric_value < 0 or (numeric_value == 0 and not allow_zero):
            qualifier = 'non-negative' if allow_zero else 'positive'
            raise ValueError(f'{name} must be {qualifier}')

    @staticmethod
    def _find_files(
        input_path: Path,
        suffixes: Tuple[str, ...],
        description: str,
    ) -> List[Path]:
        """查找输入文件或输入目录下指定扩展名的文件。"""
        if input_path.is_file():
            if input_path.suffix.lower() not in suffixes:
                raise ValueError(f'Unsupported {description} format: {input_path.suffix}')
            return [input_path]

        if input_path.is_dir():
            files = sorted(
                path for path in input_path.iterdir()
                if path.is_file() and path.suffix.lower() in suffixes
            )
            if not files:
                raise ValueError(f'No {description} files found in: {input_path}')
            return files

        raise ValueError(f'Invalid path: {input_path}')

    def _build_raster_catalog(self) -> List[Dict]:
        """只读取影像元数据并建立空间范围目录。"""
        catalog = []
        for image_file in self.image_files:
            with rasterio.open(image_file) as dataset:
                if dataset.crs is None:
                    raise ValueError(f'GeoTIFF has no CRS: {image_file}')
                if not np.isclose(dataset.transform.b, 0.0) or not np.isclose(
                    dataset.transform.d, 0.0
                ):
                    raise ValueError(
                        f'Rotated GeoTIFF is not supported: {image_file}'
                    )
                catalog.append({
                    'path': image_file,
                    'bounds': tuple(dataset.bounds),
                    'crs': dataset.crs,
                    'transform': dataset.transform,
                    'resolution': (abs(dataset.transform.a), abs(dataset.transform.e)),
                    'count': dataset.count,
                    'dtype': dataset.dtypes[0],
                    'nodata': dataset.nodata,
                    'profile': dataset.profile.copy(),
                    'mask_all_valid': all(
                        any(flag.name == 'all_valid' for flag in flags)
                        for flags in dataset.mask_flag_enums
                    ),
                })

        reference = catalog[0]
        for raster in catalog[1:]:
            if raster['crs'] != reference['crs']:
                raise ValueError(
                    f'GeoTIFF CRS mismatch: {raster["path"]} has {raster["crs"]}, '
                    f'expected {reference["crs"]}'
                )
            if raster['count'] != reference['count']:
                raise ValueError(f'GeoTIFF band count mismatch: {raster["path"]}')
            if raster['dtype'] != reference['dtype']:
                raise ValueError(f'GeoTIFF dtype mismatch: {raster["path"]}')
            if not np.allclose(raster['resolution'], reference['resolution']):
                raise ValueError(f'GeoTIFF resolution mismatch: {raster["path"]}')

        return catalog

    def process_all_files(self, n_workers: int = None):
        """处理全部 LAS/LAZ 文件。"""
        if n_workers is None:
            n_workers = max(1, multiprocessing.cpu_count() - 1)

        start_time = time.time()
        self.logger.info('LAS/Image Tile Processor started')
        self.logger.info(f'  Point cloud input: {self.pointcloud_path}')
        self.logger.info(f'  Image input: {self.image_path}')
        self.logger.info(f'  Output: {self.output_dir}')
        self.logger.info(f'  Point cloud files: {len(self.las_files)}')
        self.logger.info(f'  Image files: {len(self.image_files)}')
        self.logger.info(f'  Window size: {self.window_size}')
        self.logger.info(
            f'  Overlap: {self.overlap} (factor={self.overlap_factor})'
        )
        self.logger.info(
            f'  Points range: {self.min_points} ~ '
            f'{self.max_points or "unlimited"}'
        )
        if self.surface_only_valid:
            cell_size_text = (
                'auto' if self._is_auto(self.surface_cell_size)
                else self.surface_cell_size
            )
            radius_text = (
                'auto' if self._is_auto(self.surface_radius)
                else self.surface_radius
            )
            self.logger.info(
                '  Surface-only valid: enabled '
                f'(cell_size={cell_size_text}, '
                f'radius={radius_text}, '
                f'z_tolerance={self.surface_z_tolerance}, '
                f'echo_decay={self.echo_decay})'
            )

        total_tiles = 0
        for file_idx, las_file in enumerate(self.las_files, 1):
            total_tiles += self.process_file(
                las_file,
                n_workers=n_workers,
                file_idx=file_idx,
                total_files=len(self.las_files),
            )

        elapsed = time.time() - start_time
        self.logger.info(
            f'Processing completed: {total_tiles} tiles in {elapsed:.2f}s'
        )
        return total_tiles

    def process_file(
        self,
        las_file: Path,
        n_workers: int = None,
        file_idx: int = 1,
        total_files: int = 1,
    ) -> int:
        """处理单个 LAS/LAZ 文件。"""
        self.logger.info(f'[{file_idx}/{total_files}] Processing {las_file.name}')
        file_start = time.time()

        with laspy.open(las_file) as reader:
            las_data = reader.read()

        las_crs = las_data.header.parse_crs()
        raster_crs = self.reference_raster['crs']
        if las_crs is None:
            self.logger.warning(
                f'  {las_file.name} has no CRS; assuming raster CRS {raster_crs}'
            )
        elif rasterio.crs.CRS.from_user_input(las_crs) != raster_crs:
            raise ValueError(
                f'CRS mismatch for {las_file}: LAS={las_crs}, raster={raster_crs}'
            )

        points = np.column_stack((las_data.x, las_data.y, las_data.z))
        self.logger.info(f'  Read {len(points):,} points')

        surface_visible = None
        surface_observability = None
        if self.surface_only_valid:
            surface_start = time.time()
            surface_visible, surface_observability = self._compute_surface_state(
                points
            )
            if 'return_number' in las_data.point_format.dimension_names:
                return_number = np.asarray(las_data.return_number, dtype=np.float32)
                echo_depth = np.maximum(return_number - 1.0, 0.0)
                surface_observability *= np.exp(-self.echo_decay * echo_depth)
            else:
                self.logger.warning(
                    f'  {las_file.name} has no return_number; echo weighting skipped'
                )
            visible_count = int(np.count_nonzero(surface_visible))
            self.logger.info(
                f'  Surface visibility: {visible_count:,}/{len(points):,} points '
                f'({visible_count / len(points):.1%}) in '
                f'{time.time() - surface_start:.2f}s'
            )

        segments, stats_list = self._segment_point_cloud(points)
        if self.overlap and len(stats_list) > 1:
            stats_str = '+'.join(str(value) for value in stats_list)
            self.logger.info(f'  Generated {len(segments)} tiles ({stats_str})')
        else:
            self.logger.info(f'  Generated {len(segments)} tiles')

        saved_count = self._save_tiles(
            las_file,
            las_data,
            points,
            segments,
            surface_visible,
            surface_observability,
        )
        self.logger.info(
            f'  Completed {saved_count} tiles in {time.time() - file_start:.2f}s'
        )
        return saved_count

    @staticmethod
    def _is_auto(value) -> bool:
        """判断参数是否要求自动估算。"""
        return value is None or (isinstance(value, str) and value.lower() == 'auto')

    def _resolve_surface_parameters(
        self,
        points: np.ndarray,
    ) -> Tuple[float, float, int, int]:
        """解析表面栅格尺寸与作用半径，并限制 DSM 的最大单元数。"""
        xy_min = np.min(points[:, :2], axis=0)
        xy_max = np.max(points[:, :2], axis=0)
        extent = np.maximum(xy_max - xy_min, 0.0)
        raster_resolution = max(self.reference_raster['resolution'])

        if self._is_auto(self.surface_cell_size):
            estimated_spacing = self._estimate_surface_spacing(points, xy_min, extent)
            cell_size = max(raster_resolution, estimated_spacing)
        else:
            cell_size = float(self.surface_cell_size)

        # 防止异常稀疏或超大范围产生过大的致密 DSM。约 25M float32
        # 单元占 100 MB，连同 maximum_filter 的输出保持在合理内存范围内。
        area = float(extent[0] * extent[1])
        max_grid_cells = 25000000
        width = max(1, int(np.floor(extent[0] / cell_size)) + 1)
        height = max(1, int(np.floor(extent[1] / cell_size)) + 1)
        if width * height > max_grid_cells:
            minimum_cell_size = np.sqrt(area / max_grid_cells)
            cell_size = max(cell_size, minimum_cell_size)
            width = max(1, int(np.floor(extent[0] / cell_size)) + 1)
            height = max(1, int(np.floor(extent[1] / cell_size)) + 1)
            self.logger.warning(
                f'  Surface DSM was limited to {max_grid_cells:,} cells; '
                f'effective cell size increased to {cell_size:.3f}'
            )

        if self._is_auto(self.surface_radius):
            radius = cell_size
        else:
            radius = float(self.surface_radius)

        return cell_size, radius, width, height

    @staticmethod
    def _estimate_surface_spacing(
        points: np.ndarray,
        xy_min: np.ndarray,
        extent: np.ndarray,
    ) -> float:
        """
        通过粗网格内的局部点密度快速估算 XY 点间距。

        只对非空粗格计算 ``sqrt(cell_area / point_count)`` 并取中位数，
        因此水域等内部空洞不会扩大估计值，不规则边界产生的少量低密度
        粗格也不会主导结果。该估算仅用于设置正射表面 DSM 的大致尺度。
        """
        if len(points) <= 1 or np.any(extent <= 0):
            return 0.0

        # 每个方向最多 128 格，并根据点数降低小点云的网格数量；计算过程
        # 只需要一组长度为 N 的整数编号，不引入 KD-tree。
        target_bins = int(np.clip(np.sqrt(len(points) / 64.0), 8, 128))
        bin_width = extent[0] / target_bins
        bin_height = extent[1] / target_bins
        if bin_width <= 0 or bin_height <= 0:
            return 0.0

        columns = np.floor(
            (points[:, 0] - xy_min[0]) / bin_width
        ).astype(np.int64)
        rows = np.floor(
            (points[:, 1] - xy_min[1]) / bin_height
        ).astype(np.int64)
        np.clip(columns, 0, target_bins - 1, out=columns)
        np.clip(rows, 0, target_bins - 1, out=rows)
        counts = np.bincount(
            rows * target_bins + columns,
            minlength=target_bins * target_bins,
        )
        occupied_counts = counts[counts > 0]
        if len(occupied_counts) == 0:
            return 0.0

        cell_area = bin_width * bin_height
        local_spacing = np.sqrt(cell_area / occupied_counts.astype(np.float64))
        return float(np.median(local_spacing))

    def _compute_surface_visibility(self, points: np.ndarray) -> np.ndarray:
        """兼容旧调用，只返回正射表面硬可见标记。"""
        visible, _ = self._compute_surface_state(points)
        return visible

    def _compute_surface_state(
        self,
        points: np.ndarray,
        log_parameters: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用正射 Z-buffer 计算硬可见标记和连续表面可观测度。

        每个 DSM 单元先记录其中的最高 Z，然后用二维最大滤波将最高表面
        扩展到 surface_radius 邻域。点高度距邻域最高表面不超过
        surface_z_tolerance 时视为正射可见。连续可观测度采用高差的高斯
        衰减；它只描述表面接近程度，影像覆盖会在写 correspondence 时
        继续作为硬门控。
        """
        if len(points) == 0:
            return np.empty(0, dtype=bool), np.empty(0, dtype=np.float32)

        cell_size, radius, width, height = self._resolve_surface_parameters(points)
        xy_min = np.min(points[:, :2], axis=0)
        columns = np.floor((points[:, 0] - xy_min[0]) / cell_size).astype(np.int64)
        rows = np.floor((points[:, 1] - xy_min[1]) / cell_size).astype(np.int64)
        np.clip(columns, 0, width - 1, out=columns)
        np.clip(rows, 0, height - 1, out=rows)
        flat_indices = rows * width + columns

        dsm = np.full(width * height, -np.inf, dtype=np.float32)
        np.maximum.at(dsm, flat_indices, points[:, 2].astype(np.float32))
        dsm = dsm.reshape(height, width)

        radius_cells = int(np.ceil(radius / cell_size))
        if radius_cells > 0:
            filter_size = radius_cells * 2 + 1
            surface_dsm = maximum_filter(
                dsm,
                size=(filter_size, filter_size),
                mode='constant',
                cval=-np.inf,
            )
        else:
            surface_dsm = dsm

        surface_z = surface_dsm[rows, columns]
        height_gap = np.maximum(
            surface_z.astype(np.float64) - points[:, 2],
            0.0,
        )
        visible = height_gap <= self.surface_z_tolerance
        if self.surface_z_tolerance > 0:
            normalized_gap = height_gap / self.surface_z_tolerance
            observability = np.exp(-0.5 * normalized_gap ** 2)
        else:
            observability = (height_gap <= 1e-9).astype(np.float64)
        observability = np.clip(observability, 0.0, 1.0).astype(np.float32)
        if log_parameters:
            self.logger.info(
                f'  Surface DSM: cell={cell_size:.3f}, radius={radius:.3f}, '
                f'grid={width}x{height}'
            )
        return visible, observability

    def _segment_point_cloud(
        self,
        points: np.ndarray,
    ) -> Tuple[List[np.ndarray], List[int]]:
        """将点云按一组或多组偏移网格分块。"""
        x_size, y_size = self.window_size
        steps = np.linspace(0, 1, self.overlap_factor, endpoint=False)
        offset_configs = [
            (step_x * x_size, step_y * y_size)
            for step_x in steps
            for step_y in steps
        ]

        all_segments = []
        stats_list = []
        for offset_x, offset_y in offset_configs:
            segments = self._grid_segmentation(points, offset_x, offset_y)
            stats_list.append(len(segments))
            all_segments.extend(segments)

        return all_segments, stats_list

    def _grid_segmentation(
        self,
        points: np.ndarray,
        offset_x: float = 0.0,
        offset_y: float = 0.0,
    ) -> List[np.ndarray]:
        """使用与 tile_las.py 一致的 XY 网格逻辑分块。"""
        x_size, y_size = self.window_size
        min_x, min_y = np.min(points[:, :2], axis=0)
        origin_x, origin_y = min_x - offset_x, min_y - offset_y

        x_bins = np.floor((points[:, 0] - origin_x) / x_size).astype(np.int64)
        y_bins = np.floor((points[:, 1] - origin_y) / y_size).astype(np.int64)

        # 使用结构化二元键，避免 tile_las.py 中固定乘数带来的潜在碰撞。
        window_keys = np.empty(len(points), dtype=[('x', np.int64), ('y', np.int64)])
        window_keys['x'] = x_bins
        window_keys['y'] = y_bins
        sort_idx = np.argsort(window_keys, order=('x', 'y'))
        sorted_x = x_bins[sort_idx]
        sorted_y = y_bins[sort_idx]
        split_indices = np.flatnonzero(
            (np.diff(sorted_x) != 0) | (np.diff(sorted_y) != 0)
        ) + 1
        segments = list(np.split(sort_idx, split_indices))

        if self.min_points is not None:
            segments = self._apply_min_threshold(points, segments)
        if self.max_points is not None:
            segments = self._apply_max_threshold(points, segments)

        return segments

    def _apply_min_threshold(
        self,
        points: np.ndarray,
        segments: List[np.ndarray],
    ) -> List[np.ndarray]:
        """将小于 min_points 的块合并到最近的有效块。"""
        if len(segments) <= 1:
            return segments

        centroids = np.array([
            np.mean(points[segment, :2], axis=0) for segment in segments
        ])
        small_indices = [
            index for index, segment in enumerate(segments)
            if len(segment) < self.min_points
        ]
        valid_indices = [
            index for index in range(len(segments)) if index not in small_indices
        ]
        if not small_indices or not valid_indices:
            return segments

        valid_centroids = centroids[valid_indices]
        small_indices.sort(key=lambda index: len(segments[index]))
        for small_index in small_indices:
            distances = np.sum(
                (valid_centroids - centroids[small_index]) ** 2, axis=1
            )
            nearest_index = valid_indices[int(np.argmin(distances))]
            segments[nearest_index] = np.concatenate(
                (segments[nearest_index], segments[small_index])
            )
            segments[small_index] = np.empty(0, dtype=np.int64)

        return [segment for segment in segments if len(segment) > 0]

    def _apply_max_threshold(
        self,
        points: np.ndarray,
        segments: List[np.ndarray],
    ) -> List[np.ndarray]:
        """对超过 max_points 的块沿较长方向递归二分。"""
        result = []

        def split_segment(segment: np.ndarray):
            if len(segment) <= self.max_points:
                result.append(segment)
                return

            segment_points = points[segment, :2]
            split_dimension = int(np.argmax(np.ptp(segment_points, axis=0)))
            order = np.argsort(segment_points[:, split_dimension])
            middle = len(order) // 2
            split_segment(segment[order[:middle]])
            split_segment(segment[order[middle:]])

        for segment in segments:
            split_segment(segment)
        return result

    def _save_tiles(
        self,
        las_file: Path,
        las_data: laspy.LasData,
        points: np.ndarray,
        segments: List[np.ndarray],
        surface_visible: Optional[np.ndarray] = None,
        surface_observability: Optional[np.ndarray] = None,
    ) -> int:
        """保存点云、影像和点像素对应关系。"""
        saved_count = 0
        for tile_index, indices in enumerate(
            tqdm(segments, desc='  Saving joint tiles', leave=False)
        ):
            if len(indices) == 0:
                continue

            tile_name = f'{las_file.stem}_{tile_index:04d}'
            tile_points = points[indices]
            pointcloud_path = (
                self.pointcloud_output_dir
                / f'{tile_name}.{self.output_format}'
            )
            image_path = self.image_output_dir / f'{tile_name}.tif'
            correspondence_path = (
                self.correspondence_output_dir / f'{tile_name}.safetensors'
            )

            self._write_pointcloud_tile(
                las_data, indices, tile_points, pointcloud_path
            )
            transform, width, height, coverage = self._write_image_tile(
                tile_points[:, :2], image_path
            )
            self._write_correspondence(
                tile_points[:, :2],
                transform,
                width,
                height,
                coverage,
                correspondence_path,
                surface_visible=(
                    surface_visible[indices]
                    if surface_visible is not None
                    else None
                ),
                surface_observability=(
                    surface_observability[indices]
                    if surface_observability is not None
                    else None
                ),
            )
            saved_count += 1

        return saved_count

    def _write_pointcloud_tile(
        self,
        las_data: laspy.LasData,
        indices: np.ndarray,
        tile_points: np.ndarray,
        output_path: Path,
    ):
        """保留源点云全部维度并写入单个 LAS/LAZ tile。"""
        source_header = las_data.header
        version = source_header.version
        if version.major == 1 and version.minor < 2:
            version = laspy.header.Version(1, 2)

        header = laspy.LasHeader(
            point_format=source_header.point_format.id,
            version=version,
        )
        header.scales = source_header.scales
        header.offsets = source_header.offsets
        header.system_identifier = source_header.system_identifier
        header.generating_software = source_header.generating_software

        # 复制 VLR（Extra Bytes VLR 由 add_extra_dim 重新生成）。
        for vlr in source_header.vlrs:
            if vlr.user_id != 'LASF_Spec' or vlr.record_id != 4:
                header.vlrs.append(copy.deepcopy(vlr))

        core_bbox = [
            float(np.min(tile_points[:, 0])),
            float(np.min(tile_points[:, 1])),
            float(np.max(tile_points[:, 0])),
            float(np.max(tile_points[:, 1])),
        ]
        header.vlrs.append(laspy.VLR(
            user_id='PointSpace',
            record_id=1001,
            description='Core BBox',
            record_data=json.dumps({'core_bbox': core_bbox}).encode('utf-8'),
        ))

        for extra_dimension in source_header.point_format.extra_dimensions:
            header.add_extra_dim(laspy.ExtraBytesParams(
                name=extra_dimension.name,
                type=extra_dimension.dtype,
                description=extra_dimension.description,
                offsets=extra_dimension.offsets,
                scales=extra_dimension.scales,
            ))

        existing_names = {
            dimension.name for dimension in header.point_format.extra_dimensions
        }
        if self.save_orig_idx and 'orig_idx' not in existing_names:
            header.add_extra_dim(laspy.ExtraBytesParams(
                name='orig_idx',
                type=np.uint64,
                description='Original point index',
            ))

        new_las = laspy.LasData(header)
        source_points = las_data.points[indices]
        for dimension_name in source_points.array.dtype.names:
            if dimension_name == 'orig_idx':
                continue
            if dimension_name in new_las.points.array.dtype.names:
                new_las.points[dimension_name] = source_points[dimension_name]

        if self.save_orig_idx:
            new_las.orig_idx = indices.astype(np.uint64)
        new_las.update_header()
        new_las.write(output_path)

    def _aligned_output_grid(
        self,
        xy: np.ndarray,
    ) -> Tuple[Affine, int, int, Tuple[float, float, float, float]]:
        """将点云范围向外扩展到参考影像的完整像素边界。"""
        reference_transform = self.reference_raster['transform']
        inverse = ~reference_transform
        xmin, ymin = np.min(xy, axis=0)
        xmax, ymax = np.max(xy, axis=0)

        corner_columns = []
        corner_rows = []
        for x_value, y_value in (
            (xmin, ymin), (xmin, ymax), (xmax, ymin), (xmax, ymax)
        ):
            column, row = inverse * (x_value, y_value)
            corner_columns.append(column)
            corner_rows.append(row)

        column_start = int(np.floor(min(corner_columns)))
        column_stop = int(np.floor(max(corner_columns))) + 1
        row_start = int(np.floor(min(corner_rows)))
        row_stop = int(np.floor(max(corner_rows))) + 1
        width = max(1, column_stop - column_start)
        height = max(1, row_stop - row_start)
        transform = reference_transform * Affine.translation(column_start, row_start)

        left, top = transform * (0, 0)
        right, bottom = transform * (width, height)
        bounds = (
            min(left, right),
            min(bottom, top),
            max(left, right),
            max(bottom, top),
        )
        return transform, width, height, bounds

    @staticmethod
    def _bounds_intersect(
        first: Tuple[float, float, float, float],
        second: Tuple[float, float, float, float],
    ) -> bool:
        """判断两个 (left, bottom, right, top) 范围是否有正面积交集。"""
        return (
            max(first[0], second[0]) < min(first[2], second[2])
            and max(first[1], second[1]) < min(first[3], second[3])
        )

    def _write_image_tile(
        self,
        xy: np.ndarray,
        output_path: Path,
    ) -> Tuple[Affine, int, int, np.ndarray]:
        """按窗口读取相交影像并拼接；未覆盖区域保持为黑色。"""
        transform, width, height, output_bounds = self._aligned_output_grid(xy)
        reference = self.reference_raster
        image = np.zeros(
            (reference['count'], height, width),
            dtype=np.dtype(reference['dtype']),
        )
        coverage = np.zeros((height, width), dtype=bool)

        intersecting_rasters = [
            raster for raster in self.raster_catalog
            if self._bounds_intersect(output_bounds, raster['bounds'])
        ]
        for raster in intersecting_rasters:
            with rasterio.open(raster['path']) as source:
                intersection = (
                    max(output_bounds[0], source.bounds.left),
                    max(output_bounds[1], source.bounds.bottom),
                    min(output_bounds[2], source.bounds.right),
                    min(output_bounds[3], source.bounds.top),
                )
                source_window = from_bounds(
                    *intersection, transform=source.transform
                ).round_offsets().round_lengths()
                source_window = source_window.intersection(
                    Window(0, 0, source.width, source.height)
                )
                source_data = source.read(window=source_window)
                source_mask = source.dataset_mask(window=source_window)
                source_transform = source.window_transform(source_window)

                warped_data = np.zeros_like(image)
                warped_mask = np.zeros((height, width), dtype=np.uint8)
                for band_index in range(reference['count']):
                    reproject(
                        source=source_data[band_index],
                        destination=warped_data[band_index],
                        src_transform=source_transform,
                        src_crs=source.crs,
                        dst_transform=transform,
                        dst_crs=reference['crs'],
                        resampling=Resampling.nearest,
                        init_dest_nodata=True,
                        dst_nodata=0,
                    )
                reproject(
                    source=source_mask,
                    destination=warped_mask,
                    src_transform=source_transform,
                    src_crs=source.crs,
                    dst_transform=transform,
                    dst_crs=reference['crs'],
                    resampling=Resampling.nearest,
                    init_dest_nodata=True,
                    dst_nodata=0,
                )

                valid = (warped_mask > 0) & ~coverage
                image[:, valid] = warped_data[:, valid]
                coverage[valid] = True

        profile = reference['profile'].copy()
        profile.update({
            'driver': 'GTiff',
            'width': width,
            'height': height,
            'transform': transform,
            'crs': reference['crs'],
            'count': reference['count'],
            'dtype': reference['dtype'],
            'nodata': None,
            'compress': 'deflate',
            'predictor': 2 if np.issubdtype(image.dtype, np.integer) else 3,
            'tiled': width >= 16 and height >= 16,
        })
        profile.pop('blockxsize', None)
        profile.pop('blockysize', None)
        profile.pop('interleave', None)
        with rasterio.open(output_path, 'w', **profile) as destination:
            destination.write(image)

        return transform, width, height, coverage

    def _write_correspondence(
        self,
        xy: np.ndarray,
        transform: Affine,
        width: int,
        height: int,
        coverage: np.ndarray,
        output_path: Path,
        surface_visible: Optional[np.ndarray] = None,
        surface_observability: Optional[np.ndarray] = None,
    ):
        """保存逐点像素坐标、硬有效标记和连续可观测度。"""
        inverse = ~transform
        columns_float = inverse.a * xy[:, 0] + inverse.b * xy[:, 1] + inverse.c
        rows_float = inverse.d * xy[:, 0] + inverse.e * xy[:, 1] + inverse.f
        columns = np.floor(columns_float).astype(np.int64)
        rows = np.floor(rows_float).astype(np.int64)

        inside = (
            (columns >= 0) & (columns < width)
            & (rows >= 0) & (rows < height)
        )
        covered = np.zeros(len(xy), dtype=bool)
        covered[inside] = coverage[rows[inside], columns[inside]]
        valid = covered.astype(np.uint8)
        if surface_visible is not None:
            if len(surface_visible) != len(xy):
                raise ValueError('surface_visible and xy must have the same length')
            valid &= surface_visible.astype(np.uint8)
        if surface_observability is None:
            observability = covered.astype(np.float32)
        else:
            if len(surface_observability) != len(xy):
                raise ValueError(
                    'surface_observability and xy must have the same length'
                )
            observability = np.clip(
                np.asarray(surface_observability, dtype=np.float32),
                0.0,
                1.0,
            )
            observability *= covered.astype(np.float32)
        pixel_coord = np.empty((len(xy), 2), dtype=np.int32)
        pixel_coord[:, 0] = rows
        pixel_coord[:, 1] = columns
        temporary_path = output_path.with_name(
            f'.{output_path.name}.{os.getpid()}.tmp'
        )
        try:
            save_safetensors(
                {
                    'pixel_coord': pixel_coord,
                    'valid': valid.astype(bool, copy=False),
                    'observability': observability.astype(np.float16),
                },
                temporary_path,
                metadata={
                    'schema': 'pointspace_image_mapping_v2',
                    'coordinate_order': 'row_col',
                    'observability': (
                        'coverage_x_surface_gaussian_x_echo_decay'
                        if surface_observability is not None
                        else 'coverage_only'
                    ),
                    'surface_z_tolerance': str(self.surface_z_tolerance),
                    'echo_weighting': (
                        'exp_decay_by_return_number'
                        if surface_observability is not None
                        else 'none'
                    ),
                    'echo_decay': str(self.echo_decay),
                },
            )
            os.replace(temporary_path, output_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    def _point_image_coverage(self, xy: np.ndarray) -> np.ndarray:
        """按原始正射影像范围与 mask 判断每个点是否具有像素覆盖。"""
        covered = np.zeros(len(xy), dtype=bool)
        if len(xy) == 0:
            return covered
        xmin, ymin = np.min(xy, axis=0)
        xmax, ymax = np.max(xy, axis=0)
        for raster in self.raster_catalog:
            left, bottom, right, top = raster['bounds']
            if right <= xmin or left > xmax or top <= ymin or bottom > ymax:
                continue
            inside = (
                (xy[:, 0] >= left) & (xy[:, 0] < right)
                & (xy[:, 1] >= bottom) & (xy[:, 1] < top)
                & ~covered
            )
            indices = np.flatnonzero(inside)
            if len(indices) == 0:
                continue
            if raster['mask_all_valid']:
                covered[indices] = True
                continue
            with rasterio.open(raster['path']) as dataset:
                samples = np.ma.asarray(list(dataset.sample(xy[indices], masked=True)))
            sample_mask = np.ma.getmaskarray(samples)
            if sample_mask.ndim == 1:
                sample_valid = ~sample_mask
            else:
                sample_valid = ~np.any(sample_mask, axis=1)
            covered[indices[sample_valid]] = True
        return covered

    def update_correspondence_observability(
        self,
        correspondence_path: Union[str, Path],
        backup_dir: Optional[Union[str, Path]] = None,
        raise_on_error: bool = True,
    ) -> List[Dict]:
        """
        用同名 LAS/LAZ、原始正射覆盖、DSM 和回波序号原地更新映射。

        更新只改写 ``valid``、``observability`` 及相关元数据，保留已有
        ``pixel_coord``、``patch_index`` 和未来扩展 tensor。若给出
        ``backup_dir``，每个源文件会在首次改写前复制一份可恢复备份。
        """
        correspondence_files = self._find_files(
            Path(correspondence_path), ('.safetensors',), 'correspondence'
        )
        correspondence_by_stem = {path.stem: path for path in correspondence_files}
        if len(correspondence_by_stem) != len(correspondence_files):
            raise ValueError('Duplicate correspondence stems are not supported')

        backup_path = Path(backup_dir) if backup_dir is not None else None
        if backup_path is not None:
            source_path = Path(correspondence_path).resolve()
            if backup_path.resolve() == source_path:
                raise ValueError('backup_dir must differ from correspondence_path')
            backup_path.mkdir(parents=True, exist_ok=True)

        results = []
        for las_file in tqdm(self.las_files, desc='Update observability'):
            mapping_path = correspondence_by_stem.get(las_file.stem)
            if mapping_path is None:
                message = f'No same-stem correspondence for {las_file.name}'
                if raise_on_error:
                    raise FileNotFoundError(message)
                results.append({'file': las_file.name, 'status': 'skipped', 'reason': message})
                continue
            try:
                with laspy.open(las_file) as reader:
                    las_data = reader.read()
                points = np.column_stack((las_data.x, las_data.y, las_data.z))
                tensors = load_safetensors(mapping_path)
                with safe_open(mapping_path, framework='numpy') as source_file:
                    metadata = dict(source_file.metadata() or {})

                required = {'pixel_coord', 'patch_index', 'valid'}
                missing = required.difference(tensors)
                if missing:
                    raise ValueError(f'missing tensors: {sorted(missing)}')
                if len(tensors['valid']) != len(points):
                    raise ValueError(
                        f'point/mapping length mismatch: {len(points)} != '
                        f'{len(tensors["valid"])}'
                    )

                covered = self._point_image_coverage(points[:, :2])
                surface_visible, observability = self._compute_surface_state(
                    points, log_parameters=False
                )
                if 'return_number' in las_data.point_format.dimension_names:
                    return_number = np.asarray(
                        las_data.return_number, dtype=np.float32
                    )
                    echo_depth = np.maximum(return_number - 1.0, 0.0)
                    observability *= np.exp(-self.echo_decay * echo_depth)
                pixel_coord = np.asarray(tensors['pixel_coord'])
                try:
                    original_width, original_height = json.loads(
                        metadata['original_size']
                    )
                    grid_width, grid_height = json.loads(
                        metadata['feature_grid_size']
                    )
                    patch_size = int(metadata['patch_size'])
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                    raise ValueError(
                        'mapping metadata must contain original_size, '
                        'feature_grid_size and patch_size'
                    ) from error
                pixel_rows = pixel_coord[:, 0]
                pixel_columns = pixel_coord[:, 1]
                patch_rows = np.floor_divide(pixel_rows, patch_size)
                patch_columns = np.floor_divide(pixel_columns, patch_size)
                patch_inside = (
                    (pixel_columns >= 0) & (pixel_columns < original_width)
                    & (pixel_rows >= 0) & (pixel_rows < original_height)
                    & (patch_columns >= 0) & (patch_columns < grid_width)
                    & (patch_rows >= 0) & (patch_rows < grid_height)
                )
                valid = covered & surface_visible & patch_inside
                observability *= covered.astype(np.float32)
                observability *= patch_inside.astype(np.float32)
                patch_index = patch_rows * grid_width + patch_columns
                patch_index = patch_index.astype(np.int32, copy=False)
                patch_index[~valid] = -1

                output_tensors = {
                    name: np.ascontiguousarray(value)
                    for name, value in tensors.items()
                }
                output_tensors['valid'] = np.ascontiguousarray(valid)
                output_tensors['patch_index'] = np.ascontiguousarray(patch_index)
                output_tensors['observability'] = np.ascontiguousarray(
                    np.clip(observability, 0.0, 1.0), dtype=np.float16
                )
                metadata.update({
                    'schema': 'pointspace_image_mapping_v3',
                    'observability': 'coverage_x_surface_gaussian_x_echo_decay',
                    'surface_z_tolerance': str(self.surface_z_tolerance),
                    'echo_weighting': 'exp_decay_by_return_number',
                    'echo_decay': str(self.echo_decay),
                })

                if backup_path is not None:
                    backup_file = backup_path / mapping_path.name
                    if not backup_file.exists():
                        shutil.copy2(mapping_path, backup_file)
                temporary_path = mapping_path.with_name(
                    f'.{mapping_path.name}.{os.getpid()}.tmp'
                )
                try:
                    save_safetensors(output_tensors, temporary_path, metadata=metadata)
                    os.replace(temporary_path, mapping_path)
                finally:
                    if temporary_path.exists():
                        temporary_path.unlink()
                results.append({
                    'file': mapping_path.name,
                    'status': 'updated',
                    'valid': int(np.count_nonzero(valid)),
                    'points': len(points),
                })
            except Exception as error:
                if raise_on_error:
                    raise
                results.append({
                    'file': mapping_path.name,
                    'status': 'failed',
                    'reason': str(error),
                })
        return results


def update_correspondence_observability(
    pointcloud_path: Union[str, Path],
    image_path: Union[str, Path],
    correspondence_path: Union[str, Path],
    backup_dir: Optional[Union[str, Path]] = None,
    surface_cell_size: Optional[Union[float, str]] = None,
    surface_radius: Optional[Union[float, str]] = None,
    surface_z_tolerance: float = 0.15,
    echo_decay: float = 0.7,
    raise_on_error: bool = True,
) -> List[Dict]:
    """更新已有 correspondence 的硬有效标记与连续可观测度。"""
    correspondence_path = Path(correspondence_path)
    output_dir = (
        correspondence_path.parent
        if correspondence_path.is_file()
        else correspondence_path.parent
    )
    processor = LASImageTileProcessor(
        pointcloud_path=pointcloud_path,
        image_path=image_path,
        output_dir=output_dir,
        surface_only_valid=True,
        surface_cell_size=surface_cell_size,
        surface_radius=surface_radius,
        surface_z_tolerance=surface_z_tolerance,
        echo_decay=echo_decay,
    )
    return processor.update_correspondence_observability(
        correspondence_path=correspondence_path,
        backup_dir=backup_dir,
        raise_on_error=raise_on_error,
    )


def tile_las_image(
    pointcloud_path: Union[str, Path],
    image_path: Union[str, Path],
    output_dir: Union[str, Path],
    output_format: str = 'las',
    window_size: Tuple[float, float] = (50.0, 50.0),
    overlap: bool = True,
    overlap_factor: int = 1,
    min_points: int = 1000,
    max_points: Optional[int] = None,
    save_orig_idx: bool = True,
    surface_only_valid: bool = False,
    surface_cell_size: Optional[Union[float, str]] = None,
    surface_radius: Optional[Union[float, str]] = None,
    surface_z_tolerance: float = 0.15,
    echo_decay: float = 0.7,
):
    """创建处理器并联合分块全部点云和 GeoTIFF。"""
    processor = LASImageTileProcessor(
        pointcloud_path=pointcloud_path,
        image_path=image_path,
        output_dir=output_dir,
        output_format=output_format,
        window_size=window_size,
        overlap=overlap,
        overlap_factor=overlap_factor,
        min_points=min_points,
        max_points=max_points,
        save_orig_idx=save_orig_idx,
        surface_only_valid=surface_only_valid,
        surface_cell_size=surface_cell_size,
        surface_radius=surface_radius,
        surface_z_tolerance=surface_z_tolerance,
        echo_decay=echo_decay,
    )
    return processor.process_all_files()


if __name__ == '__main__':
    tile_las_image(
        pointcloud_path=r'E:\data\湖北\4个点',
        image_path=r'E:\data\湖北\正射影像',
        output_dir=r'E:\data\湖北\joint_tiles',
        output_format='las',
        window_size=(200.0, 200.0),
        overlap=True,
        overlap_factor=1,
        min_points=5000,
        max_points=None,
        save_orig_idx=True,

        surface_only_valid=True,
        surface_cell_size='auto',
        surface_radius='auto',
        surface_z_tolerance=0.15,
        echo_decay=0.7,
    )
