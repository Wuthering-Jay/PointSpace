"""
LAS/LAZ and georeferenced image tile processor

将坐标对齐的 LAS/LAZ 点云和 GeoTIFF 正射影像一起分块，支持：
- 点云与影像通过空间范围自动匹配，不要求文件同名或数量相同
- 复用 tile_las.py 的滑动窗口、最小点数合并和最大点数递归切分逻辑
- GeoTIFF 按窗口读取，不会将整幅大影像载入内存
- 支持 GeoTIFF 以及 JPG/JPEG + JGW/JPGW/WLD + PRJ 地理影像
- 可递归扫描所有子目录，输出仍扁平保存且自动规避同名覆盖
- 可在所有几何、表面可见性与分块计算前按 classification 过滤点
- 多幅相交影像自动拼接，无影像覆盖区域填充为黑色（像素值 0）
- 使用 Safetensors 保存逐点像素坐标、影像覆盖有效标记和连续可观测度
"""

import copy
import hashlib
import json
import logging
import multiprocessing
import os
import re
import shutil
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import laspy
import numpy as np
import rasterio
from safetensors import safe_open
from safetensors.numpy import load_file as load_safetensors
from safetensors.numpy import save_file as save_safetensors
from rasterio.enums import Resampling

try:
    from .las_io import write_standard_las
except ImportError:
    from las_io import write_standard_las
from rasterio.transform import Affine, array_bounds
from rasterio.warp import reproject
from rasterio.windows import Window, from_bounds
from tqdm import tqdm

try:
    from pointspace.utils.surface_visibility import compute_surface_visibility
except ModuleNotFoundError as error:
    if error.name != 'pointspace':
        raise
    # Support direct/IDE execution where only ``utils`` is put on sys.path.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pointspace.utils.surface_visibility import compute_surface_visibility


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


def _normalize_optional_class_filters(
    classes_to_keep: Optional[Sequence[int]],
    classes_to_remove: Optional[Sequence[int]],
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Normalize optional, mutually exclusive LAS classification filters."""

    def normalize(values, name):
        if values is None:
            return ()
        normalized = tuple(dict.fromkeys(int(value) for value in values))
        if any(value < 0 or value > 255 for value in normalized):
            raise ValueError(f'{name} values must be in [0, 255]')
        return normalized

    keep = normalize(classes_to_keep, 'classes_to_keep')
    remove = normalize(classes_to_remove, 'classes_to_remove')
    if keep and remove:
        raise ValueError(
            'classes_to_keep and classes_to_remove cannot both be non-empty'
        )
    return keep, remove


class LASImageTileProcessor:
    """
    LAS/LAZ 点云与 GeoTIFF 影像联合分块处理器

    Args:
        pointcloud_path: 输入 LAS/LAZ 文件或目录
        image_path: 输入 GeoTIFF 文件或目录
        output_dir: 输出目录，内部创建 pointcloud、image、correspondence
        output_format: 点云输出格式，'las' 或 'laz'
        image_output_format: 影像输出格式，'jpg' 或 'tif'。
        jpeg_quality: JPG 输出质量，范围 1~100。
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
        surface_oversample: 表面栅格相对 nominal cell size 的超采样倍数
        surface_num_offsets: Z-buffer 偏移数量，支持 1 或 4
        surface_adaptive_radius: 是否依据局部第一回波密度调整 splat 半径
        surface_max_radius_cells: 自适应 splat 的最大栅格半径
        surface_visibility_threshold: 连续可观测度二值化为 valid 的阈值
        echo_decay: 连续可观测度随回波序号增加的指数衰减系数。第一回波
            权重为 1，第 r 回波权重为 exp(-echo_decay * (r - 1))
        surface_slope_compensation: 是否对 splat 邻域中的高度传播加入最大
            坡度补偿，避免屋脊高度横向传播后误删斜屋面。
        surface_max_slope_degrees: 坡度补偿允许的最大表面坡角。
        first_return_bonus: 一次/第一回波的有界正向先验，按
            q <- q + bonus * (1 - q) 提升可观测度。
        recursive: 为 True 时递归扫描点云和影像输入目录的所有层级。
            输出仍集中在同一组 pointcloud/image/correspondence 目录中。
        require_full_image_coverage: 是否仅保留 XY 包围盒被一幅或多幅
            影像范围联合完整覆盖的 tile。部分覆盖的 LAS/LAZ 仍会
            进入分块，只跳过其中覆盖不完整的 tile。
        classes_to_keep: 分块前仅保留的 LAS classification；None 或空序列
            表示不使用保留筛选。
        classes_to_remove: 分块前移除的 LAS classification。不能与
            classes_to_keep 同时非空。
        skip_processed: 是否跳过已有完整输出的源点云，用于中断后续跑。

    Notes:
        输入影像必须具有相同的 CRS、波段数、数据类型、分辨率和无旋转
        仿射变换。LAS 缺少 CRS 时会假定其坐标系与影像一致并记录警告；
        LAS 和影像都具有 CRS 且不一致时会报错。
        JPG/JPEG 必须配套世界文件和同名 PRJ。
    """

    def __init__(
        self,
        pointcloud_path: Union[str, Path],
        image_path: Union[str, Path],
        output_dir: Union[str, Path],
        output_format: str = 'las',
        image_output_format: str = 'jpg',
        jpeg_quality: int = 95,
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
        surface_oversample: int = 2,
        surface_num_offsets: int = 4,
        surface_adaptive_radius: bool = True,
        surface_max_radius_cells: int = 2,
        surface_visibility_threshold: float = 0.25,
        echo_decay: float = 0.7,
        surface_slope_compensation: bool = True,
        surface_max_slope_degrees: float = 60.0,
        first_return_bonus: float = 0.15,
        recursive: bool = False,
        require_full_image_coverage: bool = True,
        classes_to_keep: Optional[Sequence[int]] = None,
        classes_to_remove: Optional[Sequence[int]] = None,
        skip_processed: bool = False,
    ):
        self.pointcloud_path = Path(pointcloud_path)
        self.image_path = Path(image_path)
        self.output_dir = Path(output_dir)
        self.output_format = output_format.lower().lstrip('.')
        self.image_output_format = image_output_format.lower().lstrip('.')
        if self.image_output_format == 'jpeg':
            self.image_output_format = 'jpg'
        self.jpeg_quality = int(jpeg_quality)
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
        self.surface_oversample = int(surface_oversample)
        self.surface_num_offsets = int(surface_num_offsets)
        self.surface_adaptive_radius = bool(surface_adaptive_radius)
        self.surface_max_radius_cells = int(surface_max_radius_cells)
        self.surface_visibility_threshold = float(
            surface_visibility_threshold
        )
        self.echo_decay = float(echo_decay)
        self.surface_slope_compensation = bool(surface_slope_compensation)
        self.surface_max_slope_degrees = float(surface_max_slope_degrees)
        self.first_return_bonus = float(first_return_bonus)
        self.recursive = bool(recursive)
        self.require_full_image_coverage = bool(require_full_image_coverage)
        self.skip_processed = bool(skip_processed)
        self.classes_to_keep, self.classes_to_remove = (
            _normalize_optional_class_filters(
                classes_to_keep, classes_to_remove
            )
        )
        self.skipped_no_image_files = 0
        self.skipped_incomplete_image_tiles = 0
        self.skipped_empty_after_filter_files = 0
        self.filtered_points = 0
        self.skipped_processed_files = 0
        self._warned_missing_pyproj = False

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
        self.completion_dir = self.output_dir / '.completed'
        self.completion_dir.mkdir(parents=True, exist_ok=True)

        self.las_files = self._find_files(
            self.pointcloud_path, ('.las', '.laz'), 'LAS/LAZ', self.recursive
        )
        self.image_files = self._find_files(
            self.image_path,
            ('.tif', '.tiff', '.jpg', '.jpeg'),
            'georeferenced image',
            self.recursive,
        )
        self.output_stems = self._build_output_stems()
        self.raster_catalog = self._build_raster_catalog()
        self.raster_bounds = np.asarray(
            [raster['bounds'] for raster in self.raster_catalog], dtype=np.float64
        )
        self.reference_raster = self.raster_catalog[0]
        if self.image_output_format == 'jpg':
            if self.reference_raster['dtype'] != 'uint8':
                raise ValueError('JPG output requires uint8 input imagery')
            if self.reference_raster['count'] not in (1, 3):
                raise ValueError('JPG output requires one or three image bands')

    def _validate_parameters(self):
        """验证输入参数。"""
        if self.output_format not in ('las', 'laz'):
            raise ValueError("output_format must be 'las' or 'laz'")
        if self.image_output_format not in ('jpg', 'tif'):
            raise ValueError("image_output_format must be 'jpg' or 'tif'")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError('jpeg_quality must be in [1, 100]')
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
        if self.surface_oversample < 1:
            raise ValueError('surface_oversample must be a positive integer')
        if self.surface_num_offsets not in (1, 4):
            raise ValueError('surface_num_offsets must be 1 or 4')
        if self.surface_max_radius_cells < 0:
            raise ValueError(
                'surface_max_radius_cells must be a non-negative integer'
            )
        if not 0 <= self.surface_visibility_threshold <= 1:
            raise ValueError('surface_visibility_threshold must be in [0, 1]')
        if self.echo_decay < 0:
            raise ValueError('echo_decay must be non-negative')
        if not 0 <= self.surface_max_slope_degrees < 90:
            raise ValueError('surface_max_slope_degrees must be in [0, 90)')
        if not 0 <= self.first_return_bonus < 1:
            raise ValueError('first_return_bonus must be in [0, 1)')

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
        recursive: bool = False,
    ) -> List[Path]:
        """查找输入文件或输入目录下指定扩展名的文件。"""
        if input_path.is_file():
            if input_path.suffix.lower() not in suffixes:
                raise ValueError(f'Unsupported {description} format: {input_path.suffix}')
            return [input_path]

        if input_path.is_dir():
            iterator = input_path.rglob('*') if recursive else input_path.iterdir()
            files = sorted(
                path for path in iterator
                if path.is_file() and path.suffix.lower() in suffixes
            )
            if not files:
                raise ValueError(f'No {description} files found in: {input_path}')
            return files

        raise ValueError(f'Invalid path: {input_path}')

    def _build_output_stems(self) -> Dict[Path, str]:
        """Create flat, deterministic output names without overwriting duplicates."""
        stem_counts = {}
        for path in self.las_files:
            stem_counts[path.stem] = stem_counts.get(path.stem, 0) + 1

        result = {}
        used = set()
        input_root = self.pointcloud_path if self.pointcloud_path.is_dir() else None
        for path in self.las_files:
            candidate = path.stem
            if stem_counts[path.stem] > 1 and input_root is not None:
                relative = path.relative_to(input_root).with_suffix('')
                candidate = '__'.join(relative.parts)
            candidate = ''.join(
                '_' if character in '<>:"/\\|?*' else character
                for character in candidate
            )
            if candidate in used:
                digest = hashlib.sha1(str(path.resolve()).encode('utf-8')).hexdigest()[:8]
                candidate = f'{candidate}__{digest}'
            used.add(candidate)
            result[path] = candidate
        return result

    @staticmethod
    def _find_sidecar(image_file: Path, suffixes: Tuple[str, ...]) -> Optional[Path]:
        """Find a same-stem sidecar while accepting conventional suffix variants."""
        candidates = {suffix.lower() for suffix in suffixes}
        # Normal datasets use lower- or upper-case conventional extensions;
        # direct probes avoid repeatedly scanning a directory containing many
        # thousands of orthophotos.
        for suffix in suffixes:
            for variant in (suffix.lower(), suffix.upper()):
                candidate = image_file.with_suffix(variant)
                if candidate.is_file():
                    return candidate
        for path in image_file.parent.iterdir():
            if (
                path.is_file()
                and path.stem.lower() == image_file.stem.lower()
                and path.suffix.lower() in candidates
            ):
                return path
        return None

    @classmethod
    def _jpeg_georeference(cls, image_file: Path) -> Tuple[Affine, rasterio.crs.CRS]:
        """Read the affine transform and CRS from JPEG sidecar files."""
        world_file = cls._find_sidecar(image_file, ('.jgw', '.jpgw', '.jpegw', '.wld'))
        projection_file = cls._find_sidecar(image_file, ('.prj',))
        if world_file is None or projection_file is None:
            missing = []
            if world_file is None:
                missing.append('JGW/JPGW/WLD')
            if projection_file is None:
                missing.append('PRJ')
            raise ValueError(
                f'JPEG georeference sidecar missing ({", ".join(missing)}): '
                f'{image_file}'
            )

        values = [
            float(value.strip())
            for value in world_file.read_text(encoding='utf-8-sig').splitlines()
            if value.strip()
        ]
        if len(values) != 6:
            raise ValueError(f'World file must contain six numbers: {world_file}')
        # World files describe the centre of the upper-left pixel. Rasterio/GDAL
        # affine transforms describe its upper-left corner.
        pixel_centre_transform = Affine(
            values[0], values[2], values[4], values[1], values[3], values[5]
        )
        transform = pixel_centre_transform * Affine.translation(-0.5, -0.5)
        try:
            crs = rasterio.crs.CRS.from_wkt(
                projection_file.read_text(encoding='utf-8-sig')
            )
        except Exception as error:
            raise ValueError(f'Invalid PRJ file: {projection_file}') from error
        return transform, crs

    @staticmethod
    def _declared_epsg(crs) -> Optional[int]:
        """Return an EPSG code, including a root WKT authority declaration."""
        if crs is None:
            return None
        for resolver in ('to_epsg', 'to_authority'):
            method = getattr(crs, resolver, None)
            if method is None:
                continue
            try:
                value = method()
            except Exception:
                continue
            if resolver == 'to_epsg' and value is not None:
                return int(value)
            if (
                resolver == 'to_authority'
                and value is not None
                and str(value[0]).upper() == 'EPSG'
            ):
                return int(value[1])

        try:
            wkt = crs.to_wkt().strip()
        except Exception:
            return None
        # Match only the authority/ID of the outer CRS object at the WKT tail,
        # not nested datum, ellipsoid or unit authority codes.
        match = re.search(
            r'(?:AUTHORITY|ID)\[\s*"EPSG"\s*,\s*"?(\d+)"?\s*\]\s*\]\s*$',
            wkt,
            re.IGNORECASE,
        )
        return int(match.group(1)) if match else None

    @classmethod
    def _crs_equivalent(cls, first, second) -> bool:
        """Compare CRS meaning rather than requiring identical WKT spelling."""
        first = rasterio.crs.CRS.from_user_input(first)
        second = rasterio.crs.CRS.from_user_input(second)
        first_epsg = cls._declared_epsg(first)
        second_epsg = cls._declared_epsg(second)
        if first_epsg is not None and second_epsg is not None:
            return first_epsg == second_epsg
        if first == second:
            return True
        # Some legacy WKT1 producers use non-canonical object names while the
        # actual projection, datum transform and units remain identical.
        return first.to_dict() == second.to_dict()

    def _build_raster_catalog(self) -> List[Dict]:
        """只读取影像元数据并建立空间范围目录。"""
        catalog = []
        for image_file in self.image_files:
            with rasterio.open(image_file) as dataset:
                if image_file.suffix.lower() in ('.jpg', '.jpeg'):
                    transform, crs = self._jpeg_georeference(image_file)
                else:
                    transform, crs = dataset.transform, dataset.crs
                if crs is None:
                    raise ValueError(f'Georeferenced image has no CRS: {image_file}')
                if not np.isclose(transform.b, 0.0) or not np.isclose(
                    transform.d, 0.0
                ):
                    raise ValueError(
                        f'Rotated georeferenced image is not supported: {image_file}'
                    )
                bounds = array_bounds(dataset.height, dataset.width, transform)
                catalog.append({
                    'path': image_file,
                    'bounds': tuple(bounds),
                    'crs': crs,
                    'transform': transform,
                    'resolution': (abs(transform.a), abs(transform.e)),
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
            if not self._crs_equivalent(raster['crs'], reference['crs']):
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
        self.skipped_no_image_files = 0
        self.skipped_incomplete_image_tiles = 0
        self.skipped_empty_after_filter_files = 0
        self.filtered_points = 0
        self.skipped_processed_files = 0
        self.logger.info('LAS/Image Tile Processor started')
        self.logger.info(f'  Point cloud input: {self.pointcloud_path}')
        self.logger.info(f'  Image input: {self.image_path}')
        self.logger.info(f'  Output: {self.output_dir}')
        self.logger.info(f'  Point cloud files: {len(self.las_files)}')
        self.logger.info(f'  Image files: {len(self.image_files)}')
        self.logger.info(f'  Recursive scan: {self.recursive}')
        self.logger.info(f'  Skip processed sources: {self.skip_processed}')
        self.logger.info(
            '  Require full image coverage: '
            f'{self.require_full_image_coverage}'
        )
        self.logger.info(
            f'  Image output: {self.image_output_format}'
            + (
                f' (quality={self.jpeg_quality})'
                if self.image_output_format == 'jpg'
                else ''
            )
        )
        if self.classes_to_keep:
            self.logger.info(
                f'  Keep classifications before tiling: '
                f'{list(self.classes_to_keep)}'
            )
        elif self.classes_to_remove:
            self.logger.info(
                f'  Remove classifications before tiling: '
                f'{list(self.classes_to_remove)}'
            )
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
                f'oversample={self.surface_oversample}, '
                f'offsets={self.surface_num_offsets}, '
                f'adaptive_radius={self.surface_adaptive_radius}, '
                f'max_radius_cells={self.surface_max_radius_cells}, '
                f'visibility_threshold={self.surface_visibility_threshold}, '
                f'echo_decay={self.echo_decay}, '
                f'slope_compensation={self.surface_slope_compensation}, '
                f'max_slope={self.surface_max_slope_degrees}, '
                f'first_return_bonus={self.first_return_bonus})'
            )

        self._initialize_completion_tracking(
            migrate_legacy=self.skip_processed
        )

        total_tiles = 0
        for file_idx, las_file in enumerate(self.las_files, 1):
            if self.skip_processed:
                processed_tiles, completion_kind = self._processed_tile_count(
                    las_file
                )
                if processed_tiles is not None:
                    self.skipped_processed_files += 1
                    total_tiles += processed_tiles
                    self.logger.info(
                        f'[{file_idx}/{len(self.las_files)}] Skipping processed '
                        f'{las_file.name} ({processed_tiles} tiles, '
                        f'{completion_kind})'
                    )
                    continue

            saved_tiles = self.process_file(
                las_file,
                n_workers=n_workers,
                file_idx=file_idx,
                total_files=len(self.las_files),
            )
            total_tiles += saved_tiles
            self._write_completion_marker(las_file, saved_tiles)

        elapsed = time.time() - start_time
        self.logger.info(
            f'Processing completed: {total_tiles} tiles, '
            f'{self.skipped_incomplete_image_tiles} tiles skipped without full '
            f'image coverage, {self.skipped_no_image_files} files skipped '
            'without any image intersection, '
            f'{self.skipped_processed_files} files skipped as processed, '
            f'{self.skipped_empty_after_filter_files} files empty after filtering, '
            f'{self.filtered_points:,} points filtered '
            f'in {elapsed:.2f}s'
        )
        return total_tiles

    def _output_tile_indices(self, directory: Path, stem: str, suffix: str):
        pattern = re.compile(
            rf'^{re.escape(stem)}_(\d{{4}}){re.escape(suffix)}$', re.IGNORECASE
        )
        return {
            int(match.group(1))
            for path in directory.glob(f'{stem}_*{suffix}')
            if (match := pattern.match(path.name)) is not None
        }

    def _initialize_completion_tracking(self, migrate_legacy: bool):
        """Enable durable markers and optionally register pre-marker outputs."""
        schema_path = self.completion_dir / '_schema.json'
        if schema_path.is_file():
            return

        migrated = 0
        if migrate_legacy:
            for las_file in self.las_files:
                stem = self.output_stems[las_file]
                point_indices = self._output_tile_indices(
                    self.pointcloud_output_dir, stem, f'.{self.output_format}'
                )
                image_indices = self._output_tile_indices(
                    self.image_output_dir, stem, f'.{self.image_output_format}'
                )
                correspondence_indices = self._output_tile_indices(
                    self.correspondence_output_dir, stem, '.safetensors'
                )
                if (
                    point_indices
                    and point_indices == image_indices == correspondence_indices
                ):
                    self._write_completion_marker(las_file, len(point_indices))
                    migrated += 1
            if migrated:
                self.logger.warning(
                    f'Registered {migrated} sources with matching legacy output '
                    'triplets as completed. Future interrupted sources require an '
                    'atomic completion marker and will not be skipped partially.'
                )

        temporary_path = schema_path.with_suffix('.json.tmp')
        temporary_path.write_text(
            json.dumps({'version': 1}), encoding='utf-8'
        )
        os.replace(temporary_path, schema_path)

    def _processed_tile_count(self, las_file: Path):
        """Return completed tile count, accepting legacy output triplets."""
        stem = self.output_stems[las_file]
        point_indices = self._output_tile_indices(
            self.pointcloud_output_dir, stem, f'.{self.output_format}'
        )
        image_indices = self._output_tile_indices(
            self.image_output_dir, stem, f'.{self.image_output_format}'
        )
        correspondence_indices = self._output_tile_indices(
            self.correspondence_output_dir, stem, '.safetensors'
        )
        outputs_match = point_indices == image_indices == correspondence_indices

        marker_path = self.completion_dir / f'{stem}.json'
        if marker_path.is_file():
            try:
                marker = json.loads(marker_path.read_text(encoding='utf-8'))
                tile_count = int(marker['tile_count'])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                self.logger.warning(
                    f'Invalid completion marker, source will be reprocessed: '
                    f'{marker_path}'
                )
            else:
                if outputs_match and len(point_indices) == tile_count:
                    return tile_count, 'completion marker'
                self.logger.warning(
                    f'Incomplete outputs despite completion marker; source will '
                    f'be reprocessed: {las_file.name}'
                )
                return None, None

        schema_path = self.completion_dir / '_schema.json'
        if not schema_path.is_file() and outputs_match and point_indices:
            self.logger.warning(
                f'No completion marker for {las_file.name}; treating its '
                f'{len(point_indices)} matching legacy output triplets as complete'
            )
            return len(point_indices), 'legacy output inference'
        return None, None

    def _write_completion_marker(self, las_file: Path, tile_count: int):
        stem = self.output_stems[las_file]
        marker_path = self.completion_dir / f'{stem}.json'
        temporary_path = marker_path.with_suffix('.json.tmp')
        temporary_path.write_text(
            json.dumps({
                'source': str(las_file.resolve()),
                'tile_count': int(tile_count),
            }),
            encoding='utf-8',
        )
        os.replace(temporary_path, marker_path)

    def _filter_las_classes(
        self,
        las_data: laspy.LasData,
    ) -> Tuple[Optional[np.ndarray], int]:
        """Filter classes in-place before any geometry-derived computation.

        The optional returned array maps filtered point indices back to the
        source file's original indices. Existing ``orig_idx`` values are
        preserved when the input has already passed through another tiling
        stage.
        """
        dimension_names = set(las_data.point_format.dimension_names)
        existing_orig_idx = None
        if self.save_orig_idx and 'orig_idx' in dimension_names:
            existing_orig_idx = np.asarray(
                las_data.orig_idx, dtype=np.uint64
            ).copy()

        if not self.classes_to_keep and not self.classes_to_remove:
            return existing_orig_idx, 0

        lookup = np.zeros(256, dtype=bool)
        if self.classes_to_keep:
            lookup[list(self.classes_to_keep)] = True
        else:
            lookup[:] = True
            lookup[list(self.classes_to_remove)] = False
        mask = lookup[np.asarray(las_data.classification, dtype=np.uint8)]
        original_count = len(mask)

        if self.save_orig_idx:
            if existing_orig_idx is None:
                original_indices = np.flatnonzero(mask).astype(
                    np.uint64, copy=False
                )
            else:
                original_indices = existing_orig_idx[mask]
        else:
            original_indices = None

        las_data.points = las_data.points[mask]
        las_data.update_header()
        return original_indices, original_count - int(np.count_nonzero(mask))

    @staticmethod
    def _source_point_count(las_data: laspy.LasData) -> int:
        """Return the pre-filter source size, preserving metadata when re-tiling."""
        for vlr in las_data.header.vlrs:
            if vlr.user_id == 'PointSpace' and vlr.record_id == 1002:
                try:
                    value = json.loads(vlr.record_data.decode('utf-8'))[
                        'source_point_count'
                    ]
                    return int(value)
                except (KeyError, TypeError, ValueError, UnicodeDecodeError):
                    break
        return len(las_data.points)

    @staticmethod
    def _crs_from_las_vlrs(header):
        """Parse common LAS CRS VLRs through rasterio without importing pyproj."""
        for vlr in header.vlrs:
            if vlr.user_id == 'LASF_Projection' and vlr.record_id == 2112:
                wkt = getattr(vlr, 'string', None)
                if not wkt:
                    record_data = getattr(vlr, 'record_data', b'')
                    if isinstance(record_data, bytes):
                        wkt = record_data.rstrip(b'\0').decode(
                            'utf-8', errors='replace'
                        )
                if wkt:
                    return rasterio.crs.CRS.from_wkt(wkt)

        # GeoTIFF ProjectedCSTypeGeoKey (3072) or GeographicTypeGeoKey (2048).
        for key_id in (3072, 2048):
            for vlr in header.vlrs:
                for key in getattr(vlr, 'geo_keys', ()):
                    if (
                        getattr(key, 'id', None) == key_id
                        and getattr(key, 'tiff_tag_location', None) == 0
                    ):
                        epsg = int(key.value_offset)
                        if 0 < epsg < 32767:
                            return rasterio.crs.CRS.from_epsg(epsg)
        return None

    def _parse_las_crs(self, header, las_file: Path):
        try:
            return header.parse_crs()
        except ModuleNotFoundError as error:
            if error.name != 'pyproj':
                raise
            crs = self._crs_from_las_vlrs(header)
            if crs is None:
                raise RuntimeError(
                    f'Cannot parse CRS for {las_file} without pyproj. Install it '
                    'with `conda install pyproj` or `pip install pyproj`.'
                ) from error
            if not self._warned_missing_pyproj:
                self.logger.warning(
                    'pyproj is unavailable; using rasterio to parse LAS CRS VLRs. '
                    'Install pyproj for full LAS CRS support.'
                )
                self._warned_missing_pyproj = True
            return crs

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
            header = reader.header
            las_bounds = (
                float(header.mins[0]),
                float(header.mins[1]),
                float(header.maxs[0]),
                float(header.maxs[1]),
            )
            if (
                self.require_full_image_coverage
                and not self._intersecting_rasters(las_bounds)
            ):
                self.skipped_no_image_files += 1
                self.logger.info(
                    f'  Skipped {las_file.name}: no imagery intersects the '
                    'LAS/LAZ XY bounds'
                )
                return 0
            las_data = reader.read()

        source_point_count = self._source_point_count(las_data)
        original_indices, filtered_count = self._filter_las_classes(las_data)
        self.filtered_points += filtered_count
        if filtered_count:
            self.logger.info(
                f'  Class filter removed {filtered_count:,} points; '
                f'{len(las_data.points):,} remain'
            )
        if len(las_data.points) == 0:
            self.skipped_empty_after_filter_files += 1
            self.logger.info(
                f'  Skipped {las_file.name}: no points remain after class filtering'
            )
            return 0

        las_crs = self._parse_las_crs(las_data.header, las_file)
        raster_crs = self.reference_raster['crs']
        if las_crs is None:
            self.logger.warning(
                f'  {las_file.name} has no CRS; assuming raster CRS {raster_crs}'
            )
        elif not self._crs_equivalent(las_crs, raster_crs):
            raise ValueError(
                f'CRS mismatch for {las_file}: LAS={las_crs}, raster={raster_crs}'
            )

        points = np.column_stack((las_data.x, las_data.y, las_data.z))
        self.logger.info(f'  Read {len(points):,} points')

        surface_visible = None
        surface_observability = None
        if self.surface_only_valid:
            surface_start = time.time()
            return_number = (
                np.asarray(las_data.return_number, dtype=np.float32)
                if 'return_number' in las_data.point_format.dimension_names
                else None
            )
            surface_visible, surface_observability = self._compute_surface_state(
                points, return_number=return_number
            )
            if return_number is None:
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

        skipped_tiles_before = self.skipped_incomplete_image_tiles
        saved_count = self._save_tiles(
            las_file,
            las_data,
            points,
            segments,
            original_indices,
            source_point_count,
            surface_visible,
            surface_observability,
        )
        skipped_tiles = (
            self.skipped_incomplete_image_tiles - skipped_tiles_before
        )
        self.logger.info(
            f'  Completed {saved_count} tiles, {skipped_tiles} skipped without '
            f'full image coverage in {time.time() - file_start:.2f}s'
        )
        return saved_count

    @staticmethod
    def _is_auto(value) -> bool:
        """判断参数是否要求自动估算。"""
        return value is None or (isinstance(value, str) and value.lower() == 'auto')

    def _compute_surface_visibility(self, points: np.ndarray) -> np.ndarray:
        """兼容旧调用，只返回正射表面硬可见标记。"""
        visible, _ = self._compute_surface_state(points)
        return visible

    def _compute_surface_state(
        self,
        points: np.ndarray,
        return_number: Optional[np.ndarray] = None,
        log_parameters: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用自适应多偏移 splat Z-buffer 计算硬标记和连续可观测度。

        几何分数在多个 half-cell 偏移结果上平均，splat 半径依据局部
        第一回波密度调整；再乘以回波序号先验并通过统一阈值得到硬标记。
        """
        result = compute_surface_visibility(
            points=points,
            return_number=return_number,
            surface_cell_size=self.surface_cell_size,
            surface_radius=self.surface_radius,
            surface_z_tolerance=self.surface_z_tolerance,
            reference_resolution=max(self.reference_raster['resolution']),
            surface_oversample=self.surface_oversample,
            surface_num_offsets=self.surface_num_offsets,
            surface_adaptive_radius=self.surface_adaptive_radius,
            surface_max_radius_cells=self.surface_max_radius_cells,
            surface_visibility_threshold=self.surface_visibility_threshold,
            echo_decay=self.echo_decay,
            surface_slope_compensation=self.surface_slope_compensation,
            surface_max_slope_degrees=self.surface_max_slope_degrees,
            first_return_bonus=self.first_return_bonus,
        )
        if log_parameters:
            self.logger.info(
                f'  Adaptive surface Z-buffer: cell={result.cell_size:.3f}, '
                f'base_radius={result.base_radius:.3f}, '
                f'grid={result.width}x{result.height}, '
                f'offsets={self.surface_num_offsets}, '
                f'threshold={self.surface_visibility_threshold:.3f}'
            )
        return result.visible, result.observability

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
        original_indices: Optional[np.ndarray] = None,
        source_point_count: Optional[int] = None,
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

            tile_points = points[indices]
            tile_bounds = (
                float(np.min(tile_points[:, 0])),
                float(np.min(tile_points[:, 1])),
                float(np.max(tile_points[:, 0])),
                float(np.max(tile_points[:, 1])),
            )
            if (
                self.require_full_image_coverage
                and not self._bounds_fully_covered(tile_bounds)
            ):
                self.skipped_incomplete_image_tiles += 1
                continue

            tile_name = f'{self.output_stems[las_file]}_{tile_index:04d}'
            pointcloud_path = (
                self.pointcloud_output_dir
                / f'{tile_name}.{self.output_format}'
            )
            image_path = (
                self.image_output_dir
                / f'{tile_name}.{self.image_output_format}'
            )
            correspondence_path = (
                self.correspondence_output_dir / f'{tile_name}.safetensors'
            )

            self._write_pointcloud_tile(
                las_data,
                indices,
                tile_points,
                pointcloud_path,
                original_indices,
                source_point_count,
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
        original_indices: Optional[np.ndarray] = None,
        source_point_count: Optional[int] = None,
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
            is_extra_bytes = vlr.user_id == 'LASF_Spec' and vlr.record_id == 4
            is_source_count = vlr.user_id == 'PointSpace' and vlr.record_id == 1002
            if not is_extra_bytes and not is_source_count:
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
        if source_point_count is not None:
            header.vlrs.append(laspy.VLR(
                user_id='PointSpace',
                record_id=1002,
                description='Source Point Count',
                record_data=json.dumps({
                    'source_point_count': int(source_point_count),
                }).encode('utf-8'),
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
            if dimension_name == 'orig_idx' and self.save_orig_idx:
                continue
            if dimension_name in new_las.points.array.dtype.names:
                new_las.points[dimension_name] = source_points[dimension_name]

        if self.save_orig_idx:
            if original_indices is None:
                new_las.orig_idx = indices.astype(np.uint64)
            else:
                new_las.orig_idx = original_indices[indices]
        new_las.update_header()
        write_standard_las(new_las, output_path)

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

    def _intersecting_rasters(
        self,
        bounds: Tuple[float, float, float, float],
    ) -> List[Dict]:
        """Vectorized catalog query, avoiding one Python comparison per image."""
        left, bottom, right, top = bounds
        raster_bounds = self.raster_bounds
        selected = (
            (raster_bounds[:, 2] > left)
            & (raster_bounds[:, 0] < right)
            & (raster_bounds[:, 3] > bottom)
            & (raster_bounds[:, 1] < top)
        )
        return [self.raster_catalog[index] for index in np.flatnonzero(selected)]

    def _bounds_fully_covered(
        self,
        bounds: Tuple[float, float, float, float],
    ) -> bool:
        """Check whether the union of raster rectangles covers all bounds.

        A sweep over raster x edges detects both uncovered borders and internal
        gaps. Input rasters are axis-aligned by construction, so no geometry
        dependency or pixel loading is required.
        """
        left, bottom, right, top = bounds
        if right < left or top < bottom:
            return False
        candidates = self._intersecting_rasters(bounds)
        if not candidates:
            return False

        tolerance = max(self.reference_raster['resolution']) * 1e-6
        rectangles = []
        x_edges = [left, right]
        for raster in candidates:
            raster_left, raster_bottom, raster_right, raster_top = raster['bounds']
            clipped = (
                max(left, raster_left),
                max(bottom, raster_bottom),
                min(right, raster_right),
                min(top, raster_top),
            )
            if clipped[2] + tolerance < clipped[0] or clipped[3] + tolerance < clipped[1]:
                continue
            rectangles.append(clipped)
            x_edges.extend((clipped[0], clipped[2]))
        if not rectangles:
            return False

        x_edges = sorted(set(x_edges))
        # Degenerate LAS bounds are checked at their single x coordinate.
        slabs = [
            (start, stop)
            for start, stop in zip(x_edges[:-1], x_edges[1:])
            if stop - start > tolerance
        ]
        if not slabs:
            slabs = [(left, right)]

        for start, stop in slabs:
            x_value = (start + stop) * 0.5
            intervals = sorted(
                (rectangle[1], rectangle[3])
                for rectangle in rectangles
                if rectangle[0] - tolerance <= x_value <= rectangle[2] + tolerance
            )
            if not intervals or intervals[0][0] > bottom + tolerance:
                return False
            covered_top = intervals[0][1]
            for interval_bottom, interval_top in intervals[1:]:
                if interval_bottom > covered_top + tolerance:
                    break
                covered_top = max(covered_top, interval_top)
                if covered_top >= top - tolerance:
                    break
            if covered_top < top - tolerance:
                return False
        return True

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

        intersecting_rasters = self._intersecting_rasters(output_bounds)
        for raster in intersecting_rasters:
            with rasterio.open(raster['path']) as source:
                intersection = (
                    max(output_bounds[0], raster['bounds'][0]),
                    max(output_bounds[1], raster['bounds'][1]),
                    min(output_bounds[2], raster['bounds'][2]),
                    min(output_bounds[3], raster['bounds'][3]),
                )
                source_window = from_bounds(
                    *intersection, transform=raster['transform']
                ).round_offsets().round_lengths()
                source_window = source_window.intersection(
                    Window(0, 0, source.width, source.height)
                )
                source_data = source.read(window=source_window)
                source_mask = source.dataset_mask(window=source_window)
                source_transform = raster['transform'] * Affine.translation(
                    source_window.col_off, source_window.row_off
                )

                warped_data = np.zeros_like(image)
                warped_mask = np.zeros((height, width), dtype=np.uint8)
                for band_index in range(reference['count']):
                    reproject(
                        source=source_data[band_index],
                        destination=warped_data[band_index],
                        src_transform=source_transform,
                        src_crs=raster['crs'],
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
                    src_crs=raster['crs'],
                    dst_transform=transform,
                    dst_crs=reference['crs'],
                    resampling=Resampling.nearest,
                    init_dest_nodata=True,
                    dst_nodata=0,
                )

                valid = (warped_mask > 0) & ~coverage
                image[:, valid] = warped_data[:, valid]
                coverage[valid] = True

        if self.image_output_format == 'jpg':
            # Georeferencing is intentionally omitted: correspondence stores
            # the final point-to-pixel mapping and DINO pairs files by stem.
            profile = {
                'driver': 'JPEG',
                'width': width,
                'height': height,
                'count': reference['count'],
                'dtype': 'uint8',
                'quality': self.jpeg_quality,
            }
        else:
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
            # JPEG input profiles can expose creation options that are invalid
            # for a Deflate-compressed GeoTIFF.
            profile.pop('photometric', None)
            profile.pop('quality', None)
        if self.image_output_format == 'jpg':
            with warnings.catch_warnings():
                warnings.simplefilter(
                    'ignore', rasterio.errors.NotGeoreferencedWarning
                )
                with rasterio.open(output_path, 'w', **profile) as destination:
                    destination.write(image)
        else:
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
                    'schema': 'pointspace_image_mapping_v4',
                    'coordinate_order': 'row_col',
                    'observability': (
                        'coverage_x_slope_compensated_surface_x_return_prior'
                        if surface_observability is not None
                        else 'coverage_only'
                    ),
                    'surface_z_tolerance': str(self.surface_z_tolerance),
                    'surface_method': (
                        'slope_compensated_adaptive_multi_offset_splat_zbuffer'
                        if self.surface_slope_compensation
                        else 'adaptive_multi_offset_splat_zbuffer'
                    ),
                    'surface_oversample': str(self.surface_oversample),
                    'surface_num_offsets': str(self.surface_num_offsets),
                    'surface_adaptive_radius': str(
                        self.surface_adaptive_radius
                    ).lower(),
                    'surface_max_radius_cells': str(
                        self.surface_max_radius_cells
                    ),
                    'surface_visibility_threshold': str(
                        self.surface_visibility_threshold
                    ),
                    'echo_weighting': (
                        'exp_decay_by_return_number'
                        if surface_observability is not None
                        else 'none'
                    ),
                    'echo_decay': str(self.echo_decay),
                    'surface_slope_compensation': str(
                        self.surface_slope_compensation
                    ).lower(),
                    'surface_max_slope_degrees': str(
                        self.surface_max_slope_degrees
                    ),
                    'first_return_bonus': str(self.first_return_bonus),
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
        for raster in self._intersecting_rasters((xmin, ymin, xmax, ymax)):
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
            Path(correspondence_path),
            ('.safetensors',),
            'correspondence',
            self.recursive,
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
                return_number = (
                    np.asarray(las_data.return_number, dtype=np.float32)
                    if 'return_number' in las_data.point_format.dimension_names
                    else None
                )
                surface_visible, observability = self._compute_surface_state(
                    points,
                    return_number=return_number,
                    log_parameters=False,
                )
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
                    'schema': 'pointspace_image_mapping_v4',
                    'observability': (
                        'coverage_x_slope_compensated_surface_x_return_prior'
                    ),
                    'surface_z_tolerance': str(self.surface_z_tolerance),
                    'surface_method': (
                        'slope_compensated_adaptive_multi_offset_splat_zbuffer'
                        if self.surface_slope_compensation
                        else 'adaptive_multi_offset_splat_zbuffer'
                    ),
                    'surface_oversample': str(self.surface_oversample),
                    'surface_num_offsets': str(self.surface_num_offsets),
                    'surface_adaptive_radius': str(
                        self.surface_adaptive_radius
                    ).lower(),
                    'surface_max_radius_cells': str(
                        self.surface_max_radius_cells
                    ),
                    'surface_visibility_threshold': str(
                        self.surface_visibility_threshold
                    ),
                    'echo_weighting': 'exp_decay_by_return_number',
                    'echo_decay': str(self.echo_decay),
                    'surface_slope_compensation': str(
                        self.surface_slope_compensation
                    ).lower(),
                    'surface_max_slope_degrees': str(
                        self.surface_max_slope_degrees
                    ),
                    'first_return_bonus': str(self.first_return_bonus),
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
    surface_oversample: int = 2,
    surface_num_offsets: int = 4,
    surface_adaptive_radius: bool = True,
    surface_max_radius_cells: int = 2,
    surface_visibility_threshold: float = 0.25,
    echo_decay: float = 0.7,
    surface_slope_compensation: bool = True,
    surface_max_slope_degrees: float = 60.0,
    first_return_bonus: float = 0.15,
    raise_on_error: bool = True,
    recursive: bool = False,
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
        surface_oversample=surface_oversample,
        surface_num_offsets=surface_num_offsets,
        surface_adaptive_radius=surface_adaptive_radius,
        surface_max_radius_cells=surface_max_radius_cells,
        surface_visibility_threshold=surface_visibility_threshold,
        echo_decay=echo_decay,
        surface_slope_compensation=surface_slope_compensation,
        surface_max_slope_degrees=surface_max_slope_degrees,
        first_return_bonus=first_return_bonus,
        recursive=recursive,
        image_output_format='tif',
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
    image_output_format: str = 'jpg',
    jpeg_quality: int = 95,
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
    surface_oversample: int = 2,
    surface_num_offsets: int = 4,
    surface_adaptive_radius: bool = True,
    surface_max_radius_cells: int = 2,
    surface_visibility_threshold: float = 0.25,
    echo_decay: float = 0.7,
    surface_slope_compensation: bool = True,
    surface_max_slope_degrees: float = 60.0,
    first_return_bonus: float = 0.15,
    recursive: bool = False,
    require_full_image_coverage: bool = True,
    classes_to_keep: Optional[Sequence[int]] = None,
    classes_to_remove: Optional[Sequence[int]] = None,
    skip_processed: bool = False,
):
    """创建处理器并联合分块全部点云和地理影像。"""
    processor = LASImageTileProcessor(
        pointcloud_path=pointcloud_path,
        image_path=image_path,
        output_dir=output_dir,
        output_format=output_format,
        image_output_format=image_output_format,
        jpeg_quality=jpeg_quality,
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
        surface_oversample=surface_oversample,
        surface_num_offsets=surface_num_offsets,
        surface_adaptive_radius=surface_adaptive_radius,
        surface_max_radius_cells=surface_max_radius_cells,
        surface_visibility_threshold=surface_visibility_threshold,
        echo_decay=echo_decay,
        surface_slope_compensation=surface_slope_compensation,
        surface_max_slope_degrees=surface_max_slope_degrees,
        first_return_bonus=first_return_bonus,
        recursive=recursive,
        require_full_image_coverage=require_full_image_coverage,
        classes_to_keep=classes_to_keep,
        classes_to_remove=classes_to_remove,
        skip_processed=skip_processed,
    )
    return processor.process_all_files()


if __name__ == '__main__':
    tile_las_image(
        pointcloud_path=r'G:\data\swiss\swiss_cvrl_70km2',
        image_path=r'G:\data\swiss\swiss_cvrl_70km2',
        output_dir=r'G:\data\swiss\joint_tiles_100',

        window_size=(100.0, 100.0),
        overlap=True,
        overlap_factor=1,
        min_points=5000,
        max_points=None,
        save_orig_idx=True,

        output_format='las',
        image_output_format='jpg',
        jpeg_quality=95,

        recursive=True,    # True 递归扫描所有子目录，输出仍扁平保存。
        classes_to_keep=None,   # 类别过滤最先执行，然后才计算表面可见性、分块和影像覆盖。
        classes_to_remove=None,
        require_full_image_coverage=True,   # 部分覆盖的 LAZ 仍分块，只保留影像联合范围完整覆盖的 tile。
        skip_processed=False,  # 中断后续跑时跳过已有完整输出的源点云。

        surface_only_valid=True,
        surface_cell_size='auto',
        surface_radius='auto',
        surface_z_tolerance=0.10,
        surface_oversample=2,
        surface_num_offsets=4,
        surface_adaptive_radius=True,
        surface_max_radius_cells=2,
        surface_visibility_threshold=0.40,
        echo_decay=0.9,
        surface_slope_compensation=True,
        surface_max_slope_degrees=60.0,
        first_return_bonus=0.15,
    )
