"""
LAS/LAZ Tile Processor (Simple Version)

将大型 LAS/LAZ 点云文件按滑动窗口分割为多个小块，支持：
- 滑动窗口分割（可配置重叠）
- 保留所有原始属性（GPS time, intensity, RGB 等）
- 保留原始点索引（orig_idx）
- 支持 LAS 1.0 ~ 1.4 多种格式
- 保留坐标系和头文件信息（VLRs）
- HAG (Height Above Ground) 计算：基于 IDW 插值
- Z_base 计算：提取宏观物理地面基准面，专为深度学习（隐式地形重建）设计
"""

import numpy as np
import laspy
import time
import multiprocessing
from pathlib import Path
from typing import Union, List, Tuple, Optional
from tqdm import tqdm
from sklearn.neighbors import KDTree, NearestNeighbors
from scipy.spatial import cKDTree
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import logging
import open3d as o3d
import torch

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

# Try to import pointseg for superpoint segmentation
try:
    from pointseg.functions import segment_point
    POINTSEG_AVAILABLE = True
except ImportError:
    POINTSEG_AVAILABLE = False


def _compute_sp_for_single_tile(tile_path, sp_params):
    """
    为单个 tile 计算超点（模块级函数，支持多进程）

    Args:
        tile_path: tile 文件路径
        sp_params: 超点计算参数字典

    Returns:
        结果字典
    """
    try:
        # 读取 tile
        with laspy.open(tile_path) as fh:
            las = fh.read()

        # 获取坐标
        points = np.vstack((las.x, las.y, las.z)).transpose()
        N = len(points)

        # 应用 Z 轴敏感度调整
        z_sensitivity = sp_params['z_sensitivity']
        if z_sensitivity != 1.0:
            xyz_scaled = points.copy()
            xyz_scaled[:, 2] *= z_sensitivity
        else:
            xyz_scaled = points

        # 计算法向量
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz_scaled)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=sp_params['normal_radius'], max_nn=30
            )
        )
        pcd.orient_normals_towards_camera_location([0., 0., 0.])
        normals = np.asarray(pcd.normals).astype(np.float32)

        # 构建 KNN 拓扑图
        edge_k = sp_params['edge_k']
        nbrs = NearestNeighbors(
            n_neighbors=edge_k + 1,
            algorithm='kd_tree',
            n_jobs=1
        ).fit(xyz_scaled)
        _, indices = nbrs.kneighbors(xyz_scaled)

        source_nodes = np.repeat(np.arange(N), edge_k)
        target_nodes = indices[:, 1:].flatten()
        edges = np.vstack((source_nodes, target_nodes)).T

        # 执行 C++ 超点分割
        vertices_tensor = torch.tensor(xyz_scaled, dtype=torch.float32)
        normals_tensor = torch.tensor(normals, dtype=torch.float32)
        edges_tensor = torch.tensor(edges, dtype=torch.int64)

        sp_idx_tensor = segment_point(
            vertices=vertices_tensor,
            normals=normals_tensor,
            edges=edges_tensor,
            kThresh=sp_params['kThresh'],
            segMinVerts=sp_params['segMinVerts'],
            segMaxVerts=sp_params['segMaxVerts']
        )

        sp_idx = sp_idx_tensor.numpy().astype(np.int32)

        # 更新文件
        las.superpoint = sp_idx
        las.write(tile_path)

        return {
            'status': 'success',
            'filename': tile_path.name,
            'num_points': N,
            'num_superpoints': len(np.unique(sp_idx))
        }
    except Exception as e:
        import traceback
        return {
            'status': 'error',
            'filename': tile_path.name if hasattr(tile_path, 'name') else str(tile_path),
            'error': str(e),
            'traceback': traceback.format_exc()
        }


class LASTileProcessor:
    """
    LAS/LAZ 点云分块处理器

    Args:
        input_path: 输入 LAS/LAZ 文件或目录
        output_dir: 输出目录
        window_size: 分块窗口大小 (x_size, y_size)，单位：米
        overlap: 是否启用重叠分块
        overlap_factor: 重叠因子，生成 overlap_factor^2 组网格
        min_points: 最小点数阈值，小于此值的块会被合并到邻近块
        max_points: 最大点数阈值，超过此值的块会被递归切分
        save_orig_idx: 是否保存原始点索引
        output_format: 输出格式 'las' 或 'laz'

        calc_hag: 是否计算 HAG (Height Above Ground)
        hag_ground_class: 地面点的分类 ID（默认 2，ASPRS 标准）
        hag_on_source: True=在原始点云上计算HAG（避免边界效应），
                       False=在分割后的每个tile上计算（节省内存）
        hag_k_neighbors: IDW 插值使用的邻近地面点数量（默认 12）
        hag_power: IDW 插值的幂次（默认 2，即反距离平方）

        calc_z_base: 是否利用 CSF 计算深度学习基准面 (Z_base)
        z_base_on_source: 是否在源点云上全局计算基准面（强烈建议 True，消除块间断层）

        calc_superpoint: 是否计算超点分割 (Superpoint Segmentation)
        superpoint_on_source: 是否在原始点云上计算超点
                             - True: 在原始点云上计算好超点，然后 tile（避免边界效应）
                             - False: 先完成所有 tile，然后批量并行计算所有 tile 的超点（更高效）
        sp_kThresh: 分割阈值，越大块越大
        sp_segMinVerts: 最小超点点数
        sp_segMaxVerts: 最大超点点数（-1 表示不限制）
        sp_edge_k: KNN 边数
        sp_normal_radius: 法向量搜索半径
        sp_z_sensitivity: Z 轴敏感度系数（>1 增强高度差异，<1 弱化）
        sp_num_workers: 超点并行计算进程数（仅在 superpoint_on_source=False 时生效）
    """
    
    def __init__(
        self,
        input_path: Union[str, Path],
        output_dir: Union[str, Path] = None,
        window_size: Tuple[float, float] = (50.0, 50.0),
        overlap: bool = True,
        overlap_factor: int = 1,
        min_points: int = 1000,
        max_points: Optional[int] = None,
        save_orig_idx: bool = True,
        output_format: str = 'las',
        buffer_size: float = 10.0,

        calc_normals: bool = False,
        normal_on_source: bool = True,
        normal_k_neighbors: int = 12,
        normal_class=None,

        calc_hag: bool = False,
        hag_ground_class: int = 2,
        hag_on_source: bool = True,
        hag_k_neighbors: int = 12,
        hag_power: float = 2.0,

        calc_z_base: bool = False,
        z_base_on_source: bool = True,
        z_base_denoise_radius: float = 2.0,
        z_base_denoise_elev_diff: float = 2.0,
        z_base_ptd_radius: float = 15.0,
        z_base_ptd_slope: float = 15.0,
        z_base_ptd_height: float = 0.25,
        z_base_ptd_slope_norm: bool = True,

        calc_superpoint: bool = False,
        superpoint_on_source: bool = False,
        sp_kThresh: float = 0.01,
        sp_segMinVerts: int = 5,
        sp_segMaxVerts: int = 8192,
        sp_edge_k: int = 10,
        sp_normal_radius: float = 2.0,
        sp_z_sensitivity: float = 2.5,
        sp_num_workers: int = 4,
    ):
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir) if output_dir else self.input_path.parent / 'tiles'
        self.window_size = window_size
        self.overlap = overlap
        self.overlap_factor = overlap_factor if overlap else 1
        self.min_points = min_points
        self.max_points = max_points
        self.save_orig_idx = save_orig_idx
        self.output_format = output_format.lower()

        # 法向量参数
        self.calc_normals = calc_normals
        self.normals_on_source = normal_on_source
        self.normal_k_neighbors = normal_k_neighbors
        # 若非 None，仅对指定类别计算法向量，其他类别赋 [0,0,1]
        self.normal_class = (
            list(normal_class) if isinstance(normal_class, (list, tuple))
            else ([normal_class] if normal_class is not None else None)
        )
        
        # HAG 参数
        self.calc_hag = calc_hag
        self.hag_ground_class = hag_ground_class
        self.hag_on_source = hag_on_source
        self.hag_k_neighbors = hag_k_neighbors
        self.hag_power = hag_power
        
        # Z_base 参数
        self.calc_z_base = calc_z_base
        self.z_base_on_source = z_base_on_source
        self.z_base_denoise_radius = z_base_denoise_radius
        self.z_base_denoise_elev_diff = z_base_denoise_elev_diff
        self.z_base_ptd_radius = z_base_ptd_radius
        self.z_base_ptd_slope = z_base_ptd_slope
        self.z_base_ptd_height = z_base_ptd_height
        self.z_base_ptd_slope_norm = z_base_ptd_slope_norm

        # Superpoint 参数
        self.calc_superpoint = calc_superpoint
        self.superpoint_on_source = superpoint_on_source
        self.sp_kThresh = sp_kThresh
        self.sp_segMinVerts = sp_segMinVerts
        self.sp_segMaxVerts = sp_segMaxVerts
        self.sp_edge_k = sp_edge_k
        self.sp_normal_radius = sp_normal_radius
        self.sp_z_sensitivity = sp_z_sensitivity
        self.sp_num_workers = sp_num_workers

        # 验证 pointseg 库可用性
        if self.calc_superpoint and not POINTSEG_AVAILABLE:
            raise ImportError("pointseg library is required for superpoint calculation. "
                              "Please install it from libs/pointseg.")

        self.buffer_size = buffer_size

        self.logger = get_root_logger()
        
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)
            
        self.las_files = self._find_las_files()

    def _find_las_files(self) -> List[Path]:
        """查找输入路径下的所有 LAS/LAZ 文件"""
        if self.input_path.is_file():
            if self.input_path.suffix.lower() in ['.las', '.laz']:
                return [self.input_path]
            else:
                raise ValueError(f"Unsupported file format: {self.input_path.suffix}")
        elif self.input_path.is_dir():
            files = list(self.input_path.glob('*.las')) + list(self.input_path.glob('*.laz'))
            return sorted(files)
        else:
            raise ValueError(f"Invalid path: {self.input_path}")

    def process_all_files(self, n_workers: int = None):
        """处理所有 LAS/LAZ 文件"""
        if n_workers is None:
            n_workers = max(1, multiprocessing.cpu_count() - 1)

        start_time = time.time()

        self.logger.info(f"LAS Tile Processor started")
        self.logger.info(f"  Input: {self.input_path}")
        self.logger.info(f"  Output: {self.output_dir}")
        self.logger.info(f"  Files: {len(self.las_files)}")
        self.logger.info(f"  Window size: {self.window_size}")
        self.logger.info(f"  Overlap: {self.overlap} (factor={self.overlap_factor})")
        self.logger.info(f"  Points range: {self.min_points} ~ {self.max_points or 'unlimited'}")
        if self.calc_hag:
            mode = 'source' if self.hag_on_source else 'tile'
            self.logger.info(f"  HAG: enabled (ground_class={self.hag_ground_class}, k={self.hag_k_neighbors}, mode={mode})")
        if self.calc_z_base:
            mode = 'source' if self.z_base_on_source else 'tile'
            self.logger.info(f"  Z_base: enabled (denoise_radius={self.z_base_denoise_radius}m, ptd_radius={self.z_base_ptd_radius}m, mode={mode})")
        if self.calc_superpoint:
            mode = 'source' if self.superpoint_on_source else 'tile-post'
            self.logger.info(f"  Superpoint: enabled (kThresh={self.sp_kThresh}, maxVerts={self.sp_segMaxVerts}, z_sens={self.sp_z_sensitivity}, mode={mode})")

        # 第一阶段：处理所有文件（tile + source 模式的特征计算）
        for idx, las_file in enumerate(self.las_files, 1):
            try:
                self.process_file(las_file, n_workers=n_workers, file_idx=idx, total_files=len(self.las_files))
            except Exception as e:
                self.logger.error(f"Failed to process {las_file.name}: {e}")
                import traceback
                traceback.print_exc()

        # 第二阶段：如果启用 superpoint 且为 tile-post 模式，批量计算所有 tile 的 superpoint
        if self.calc_superpoint and not self.superpoint_on_source:
            self.logger.info("")
            self.logger.info("="*60)
            self.logger.info("Starting batch superpoint computation for all tiles...")
            self.logger.info("="*60)
            self._compute_superpoints_for_all_tiles()

        elapsed = time.time() - start_time
        self.logger.info(f"Processing completed in {elapsed:.2f}s")

    def process_file(self, las_file: Path, n_workers: int = None, file_idx: int = 1, total_files: int = 1):
        """处理单个 LAS/LAZ 文件"""
        self.logger.info(f"[{file_idx}/{total_files}] Processing {las_file.name}")
        file_start = time.time()

        # 1. 读取数据
        with laspy.open(las_file) as fh:
            las_data = fh.read()
        
        num_points = len(las_data.points)
        self.logger.info(f"  Read {num_points:,} points")
            
        # 获取坐标
        points = np.vstack((las_data.x, las_data.y, las_data.z)).transpose()
        
        # 2.1 在原始点云上计算 HAG（如果启用且选择 source 模式）
        source_hag = None
        if self.calc_hag and self.hag_on_source:
            self.logger.info(f"  Computing HAG on source point cloud...")
            hag_start = time.time()
            classification = np.array(las_data.classification)
            source_hag = self._compute_hag(points, classification)
            self.logger.info(f"  HAG computed in {time.time() - hag_start:.2f}s")

        # 2.2 在原始点云上计算 Z_base（如果启用且选择 source 模式）
        source_z_base = None
        if self.calc_z_base and self.z_base_on_source:
            self.logger.info(f"  Computing Z_base on source point cloud...")
            z_base_start = time.time()
            source_z_base = self._compute_z_base(points)
            self.logger.info(f"  Z_base computed in {time.time() - z_base_start:.2f}s")

        # 2.3 在原始点云上计算法向量（如果启用）
        source_normals = None
        if self.calc_normals and self.normals_on_source:
            self.logger.info(f"  Computing normals on source point cloud...")
            normals_start = time.time()
            src_cls = np.array(las_data.classification) if self.normal_class is not None else None
            source_normals = self._compute_normals(points, classification=src_cls)
            self.logger.info(f"  Normals computed in {time.time() - normals_start:.2f}s")

        # 2.4 在原始点云上计算超点分割（如果启用且选择 source 模式）
        source_superpoint = None
        if self.calc_superpoint and self.superpoint_on_source:
            self.logger.info(f"  Computing superpoint on source point cloud...")
            sp_start = time.time()
            source_superpoint = self._compute_superpoint(points)
            n_sp = len(np.unique(source_superpoint))
            self.logger.info(f"  Superpoint computed in {time.time() - sp_start:.2f}s, {n_sp} segments")

        
        # 3. 滑动窗口切块 (获取索引列表)
        segments_indices, stats_list = self._segment_point_cloud(points, n_workers=n_workers)
        
        total_segs = len(segments_indices)
        if self.overlap and len(stats_list) > 1:
            stats_str = "+".join([str(s) for s in stats_list])
            self.logger.info(f"  Generated {total_segs} tiles ({stats_str})")
        else:
            self.logger.info(f"  Generated {total_segs} tiles")
        
        # 4. 保存分块
        self._save_tiles(las_file, las_data, segments_indices, points, source_hag, source_z_base, source_normals, source_superpoint)
        
        elapsed = time.time() - file_start
        self.logger.info(f"  Completed in {elapsed:.2f}s")

    def _segment_point_cloud(self, points: np.ndarray, n_workers: int = 4) -> Tuple[List[np.ndarray], List[int]]:
        """
        将点云按窗口分割
        
        Returns:
            (segments_indices, stats_list): 分块索引列表和每组的数量统计
        """
        x_size, y_size = self.window_size
        
        # 生成偏移量配置
        steps = np.linspace(0, 1, self.overlap_factor, endpoint=False)
        offset_configs = []
        for sx in steps:
            for sy in steps:
                offset_configs.append((sx * x_size, sy * y_size))
        
        # 如果只有一种配置（无重叠），直接运行
        if len(offset_configs) == 1:
            segments = self._grid_segmentation(points, offset_x=0, offset_y=0)
            return segments, [len(segments)]

        # 多线程并行处理所有偏移配置
        all_segments = []
        stats_list = [0] * len(offset_configs)
        
        max_overlap_workers = min(len(offset_configs), n_workers)
        
        with ThreadPoolExecutor(max_workers=max_overlap_workers) as executor:
            future_to_idx = {
                executor.submit(
                    self._grid_segmentation, 
                    points, 
                    config[0],
                    config[1]
                ): i for i, config in enumerate(offset_configs)
            }
            
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    segs = future.result()
                    stats_list[idx] = len(segs)
                    all_segments.extend(segs)
                except Exception as e:
                    self.logger.error(f"Segment generation failed for config {idx}: {e}")
        
        return all_segments, stats_list
        
    def _grid_segmentation(self, points: np.ndarray, offset_x: float = 0, offset_y: float = 0):
        """
        基于网格的分割，支持带 Buffer 的重叠切块。

        Args:
            points: 点云坐标 (N, 3)
            offset_x: X 方向偏移
            offset_y: Y 方向偏移

        Returns:
            List[Tuple[np.ndarray, np.ndarray]]:
                [(含 buffer 的点索引, 核心边界 [xmin, ymin, xmax, ymax]), ...]
        """
        x_size, y_size = self.window_size
        min_x, min_y = np.min(points[:, 0]), np.min(points[:, 1])
        origin_x, origin_y = min_x - offset_x, min_y - offset_y

        x_bins = ((points[:, 0] - origin_x) / x_size).astype(np.int64)
        y_bins = ((points[:, 1] - origin_y) / y_size).astype(np.int64)
        window_ids = x_bins * 1000000 + y_bins

        sort_idx = np.argsort(window_ids)
        sorted_window_ids = window_ids[sort_idx]
        _, split_indices = np.unique(sorted_window_ids, return_index=True)
        segments = np.split(sort_idx, split_indices[1:])

        if self.min_points is not None:
            segments = self._apply_min_threshold(points, segments)
        if self.max_points is not None:
            segments = self._apply_max_threshold(points, segments)

        core_bboxes = []
        for seg in segments:
            c_xmin, c_ymin = points[seg, :2].min(axis=0)
            c_xmax, c_ymax = points[seg, :2].max(axis=0)
            core_bboxes.append(np.array([c_xmin, c_ymin, c_xmax, c_ymax], dtype=np.float64))

        if self.buffer_size <= 0:
            return list(zip(segments, core_bboxes))

        self.logger.info("    Applying buffer using fast binary search...")
        sort_x_idx = np.argsort(points[:, 0])
        sorted_points = points[sort_x_idx]
        sorted_x, sorted_y = sorted_points[:, 0], sorted_points[:, 1]

        buffered_results = []
        for bbox in core_bboxes:
            b_xmin, b_xmax = bbox[0] - self.buffer_size, bbox[2] + self.buffer_size
            b_ymin, b_ymax = bbox[1] - self.buffer_size, bbox[3] + self.buffer_size

            left_idx = np.searchsorted(sorted_x, b_xmin, side='left')
            right_idx = np.searchsorted(sorted_x, b_xmax, side='right')

            cand_y = sorted_y[left_idx:right_idx]
            mask_y = (cand_y >= b_ymin) & (cand_y <= b_ymax)

            final_indices = sort_x_idx[left_idx:right_idx][mask_y]
            buffered_results.append((final_indices, bbox))

        return buffered_results
    
    def _apply_max_threshold(self, points: np.ndarray, segments: List[np.ndarray]) -> List[np.ndarray]:
        """对超过 max_points 的块进行递归切分"""
        large_segment_indices = [i for i, segment in enumerate(segments) if len(segment) > self.max_points]
        
        if not large_segment_indices:
            return segments
        
        result_segments = [segment for i, segment in enumerate(segments) if i not in large_segment_indices]
        large_segments = [segments[i] for i in large_segment_indices]
        
        def process_segment(segment):
            if len(segment) <= self.max_points:
                return [segment]
            
            segment_points = points[segment]
            ranges = np.ptp(segment_points[:, :2], axis=0)
            split_dim = np.argmax(ranges[:2])
            sorted_indices = np.argsort(segment_points[:, split_dim])
            
            mid = len(sorted_indices) // 2
            left_half = segment[sorted_indices[:mid]]
            right_half = segment[sorted_indices[mid:]]
            
            result = []
            result.extend(process_segment(left_half))
            result.extend(process_segment(right_half))
            return result
        
        for segment in large_segments:
            result_segments.extend(process_segment(segment))
        
        return result_segments

    def _apply_min_threshold(self, points: np.ndarray, segments: List[np.ndarray]) -> List[np.ndarray]:
        """将小于 min_points 的块合并到邻近块"""
        if len(segments) <= 1:
            return segments
        
        centroids = np.array([np.mean(points[segment][:, :2], axis=0) for segment in segments])
        small_segments = [i for i, segment in enumerate(segments) if len(segment) < self.min_points]
        
        if not small_segments:
            return segments
        
        valid_indices = [i for i in range(len(segments)) if i not in small_segments]
        if not valid_indices:
            return segments
        
        valid_centroids = centroids[valid_indices]
        kdtree = KDTree(valid_centroids)
        
        small_segments.sort(key=lambda i: len(segments[i]))
        
        for small_idx in small_segments:
            if small_idx >= len(segments):
                continue
            
            _, nearest_idx = kdtree.query([centroids[small_idx]], k=1)
            nearest_idx = valid_indices[nearest_idx[0][0]]
            
            if nearest_idx != small_idx and nearest_idx < len(segments):
                segments[nearest_idx] = np.concatenate([segments[nearest_idx], segments[small_idx]])
                segments[small_idx] = np.array([], dtype=int)
        
        return [segment for segment in segments if len(segment) > 0]

    def _compute_normals(self, points: np.ndarray,
                         classification: np.ndarray = None) -> np.ndarray:
        """
        计算点云法向量
        使用 Open3D 极速 C++ 引擎，支持 K 近邻平滑
        强制所有法向量 Z 分量为正 (朝向天空)

        Args:
            points: 点云坐标 (N, 3)
            classification: 逐点分类标签 (N,)；仅当 ``normal_class`` 非 None
                时使用。指定类别的点参与 KNN 计算，其他类别法向量默认为 [0, 0, 1]。
        """
        # ------------------------------------------------------------------
        # 确定参与计算的点集
        # ------------------------------------------------------------------
        if self.normal_class is not None and classification is not None:
            mask = np.isin(classification, self.normal_class)
            sub_points = points[mask]
        else:
            mask = None
            sub_points = points

        # 所有点默认法向量为 [0, 0, 1]（朝天）
        normals = np.zeros((len(points), 3), dtype=np.float32)
        normals[:, 2] = 1.0

        if len(sub_points) < 3:
            return normals

        # ------------------------------------------------------------------
        # Open3D KNN 法向估算（仅在子集点上构建 KD-Tree）
        # ------------------------------------------------------------------
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(sub_points)

        # 使用 KNN 估算法向。K值越大，法向越平滑，越抗噪
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn=self.normal_k_neighbors)
        )

        sub_normals = np.asarray(pcd.normals).astype(np.float32)

        # -----------------------------------------------------
        # 🔥 极其关键的鲁棒性保证：法向一致性约束
        # 对于地形，真实的法向必然是指向天空的（Z > 0）
        # 将所有 Z < 0 的法向翻转，彻底解决法向乱翻的问题！
        # -----------------------------------------------------
        flip_mask = sub_normals[:, 2] < 0
        sub_normals[flip_mask] *= -1.0

        if mask is not None:
            normals[mask] = sub_normals
        else:
            normals = sub_normals

        return normals

    def _compute_hag(self, points: np.ndarray, classification: np.ndarray) -> np.ndarray:
        """
        计算 HAG (Height Above Ground)
        
        使用 IDW (Inverse Distance Weighting) 插值方法：
        1. 地面点的 HAG 直接设为 0
        2. 非地面点基于 XY 平面上最邻近的 k 个地面点进行 IDW 插值
        
        Args:
            points: 点云坐标 (N, 3)
            classification: 点分类标签 (N,)
            
        Returns:
            HAG 值数组 (N,)，单位与输入 Z 坐标相同
        """
        n_points = len(points)
        hag = np.zeros(n_points, dtype=np.float32)
        
        # 识别地面点和非地面点
        ground_mask = classification == self.hag_ground_class
        non_ground_mask = ~ground_mask
        
        ground_indices = np.where(ground_mask)[0]
        non_ground_indices = np.where(non_ground_mask)[0]
        
        n_ground = len(ground_indices)
        n_non_ground = len(non_ground_indices)
        
        self.logger.info(f"    Ground points: {n_ground:,}, Non-ground: {n_non_ground:,}")
        
        # 地面点 HAG = 0（已经初始化为 0）
        
        # 如果没有地面点或非地面点，直接返回
        if n_ground < 3:
            self.logger.warning(f"    Too few ground points ({n_ground}), using Z-min as reference")
            z_min = points[:, 2].min()
            hag = (points[:, 2] - z_min).astype(np.float32)
            hag[ground_mask] = 0.0
            return hag
        
        if n_non_ground == 0:
            return hag
        
        # 坐标中心化，避免大坐标的浮点精度问题
        xy_mean = points[:, :2].mean(axis=0)
        
        # 提取地面点的 XY 坐标（中心化）和 Z 坐标
        ground_xy = points[ground_indices, :2] - xy_mean
        ground_z = points[ground_indices, 2]
        
        # 提取非地面点的 XY 坐标（中心化）和 Z 坐标
        non_ground_xy = points[non_ground_indices, :2] - xy_mean
        non_ground_z = points[non_ground_indices, 2]
        
        # 构建地面点的 cKDTree（基于 XY 平面）
        tree = cKDTree(ground_xy)
        
        # 确定查询的邻居数量（不超过地面点总数）
        k = min(self.hag_k_neighbors, n_ground)
        
        # 批量查询所有非地面点的 k 个最近地面点
        distances, indices = tree.query(non_ground_xy, k=k)
        
        # 处理 k=1 的情况（distances 和 indices 可能是 1D）
        if k == 1:
            distances = distances.reshape(-1, 1)
            indices = indices.reshape(-1, 1)
        
        # IDW 插值计算地面高程
        # 权重 = 1 / distance^power
        # 避免除零：将距离为 0 的点权重设为极大值
        with np.errstate(divide='ignore', invalid='ignore'):
            weights = 1.0 / np.power(distances, self.hag_power)
            # 处理距离为 0 的情况（点恰好在地面点位置）
            zero_dist_mask = distances == 0
            if np.any(zero_dist_mask):
                # 如果有距离为 0 的邻居，直接使用该邻居的 Z 值
                weights[zero_dist_mask] = 1e10
                # 对于有零距离的行，将其他非零距离的权重设为 0
                rows_with_zero = zero_dist_mask.any(axis=1)
                if np.any(rows_with_zero):
                    # 创建掩码：属于有零距离的行，但本身不是零距离
                    mask_to_zero = rows_with_zero[:, np.newaxis] & ~zero_dist_mask
                    weights[mask_to_zero] = 0
        
        # 获取邻近地面点的 Z 值
        neighbor_z = ground_z[indices]  # (n_non_ground, k)
        
        # 加权平均计算插值地面高程
        weight_sum = weights.sum(axis=1, keepdims=True)
        interpolated_ground_z = (weights * neighbor_z).sum(axis=1) / weight_sum.squeeze()
        
        # 计算 HAG = 点的 Z 坐标 - 插值地面高程
        hag[non_ground_indices] = (non_ground_z - interpolated_ground_z).astype(np.float32)
        
        # 注意：保留负值，地下噪点会有 HAG < 0，便于识别和过滤
        
        return hag
    
    def _compute_z_base(self, points: np.ndarray, source_las_path: Path = None) -> np.ndarray:
        """
        利用 WhiteboxTools (去噪 + PTD) 计算全局地形底座 (Z_base)
        
        Args:
            points: [N, 3] 内存中的点云坐标 (用于最终映射 Z_base)
            source_las_path: 如果是在全局计算，传入原始文件路径以避免重复写文件
        """
        try:
            import whitebox
            from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
            import tempfile
            import os
        except ImportError:
            raise ImportError("Please install whitebox and scipy (pip install whitebox scipy)")
            
        start_time = time.time()
        self.logger.info("    Computing Z_base using WhiteboxTools (Denoise + PTD)...")
        
        # 1. 初始化并静音 WBT
        wbt = whitebox.WhiteboxTools()
        wbt.set_verbose_mode(False)  # [关键] 彻底屏蔽所有终端输出
        
        # 2. 使用安全的临时文件夹沙箱 (随用随删，不留垃圾)
        with tempfile.TemporaryDirectory() as temp_dir:
            wbt.set_working_dir(temp_dir)
            
            # 确定输入文件：如果有源文件直接用，没有就把 numpy 写入临时 las
            if source_las_path is not None and source_las_path.exists():
                input_las = str(source_las_path)
            else:
                input_las = os.path.join(temp_dir, "temp_in.las")
                # 将内存中的 points 临时写入 las (针对 z_base_on_source=False 的后备方案)
                header = laspy.LasHeader(point_format=3, version="1.2")
                header.offsets = np.min(points, axis=0)
                header.scales = np.array([0.001, 0.001, 0.001])
                temp_las = laspy.LasData(header)
                temp_las.x, temp_las.y, temp_las.z = points[:, 0], points[:, 1], points[:, 2]
                temp_las.write(input_las)
                
            clean_las = os.path.join(temp_dir, "temp_clean.las")
            ground_las = os.path.join(temp_dir, "temp_ground.las")
            
            # ================= 执行 WBT 算法链 =================
            # A. 孤立噪声点剔除
            wbt.lidar_remove_outliers(
                input_las, 
                clean_las, 
                radius=self.z_base_denoise_radius,
                elev_diff=self.z_base_denoise_elev_diff,    
                use_median=True
            )
            
            if not os.path.exists(clean_las):
                self.logger.warning("    WBT Denoise failed! Falling back to global Z-min.")
                return np.full(len(points), points[:, 2].min(), dtype=np.float32)
                
            # B. PTD 地面滤波 (参数根据你的设定)
            wbt.lidar_ground_point_filter(
                clean_las,
                ground_las,
                radius=self.z_base_ptd_radius,
                min_neighbours=0,
                slope_threshold=self.z_base_ptd_slope,
                height_threshold=self.z_base_ptd_height,
                classify=True,          # 必须为 True，将地面点分类为 2
                slope_norm=self.z_base_ptd_slope_norm,
                height_above_ground=False
            )
            # ==================================================
            
            if not os.path.exists(ground_las):
                self.logger.warning("    WBT Ground Filter failed! Falling back to global Z-min.")
                return np.full(len(points), points[:, 2].min(), dtype=np.float32)

            # 3. 读取 WBT 提取的地面点
            with laspy.open(ground_las) as fh:
                g_las = fh.read()
                # 提取分类为 2 (Ground) 的点
                ground_mask = g_las.classification == 2
                raw_ground_points = np.vstack((g_las.x[ground_mask], g_las.y[ground_mask], g_las.z[ground_mask])).T

        # <--- 离开 with 块后，temp_dir 中的所有中间 .las 文件已被 Python 自动删除！--->

        if len(raw_ground_points) < 3:
            self.logger.warning("    Too few ground points found by WBT, using global Z-min.")
            return np.full(len(points), points[:, 2].min(), dtype=np.float32)

        # 4. 骨架降采样 (保留这一步，能成倍加速 KD-Tree 构建并过滤密集噪声)
        grid_size = 2.0
        xy_grid = np.floor(raw_ground_points[:, :2] / grid_size).astype(np.int32)
        _, unique_indices = np.unique(xy_grid, axis=0, return_index=True)
        skeleton_points = raw_ground_points[unique_indices]
        
        skeleton_xy = skeleton_points[:, :2]
        skeleton_z = skeleton_points[:, 2]

        self.logger.info(f"    Building Nearest Neighbor surface from {len(skeleton_points):,} skeleton points...")

        from scipy.interpolate import NearestNDInterpolator

        # 5. 极速最近邻插值 (直接生成全覆盖的连续基准面，自带外推，无 NaN)
        nearest_interp = NearestNDInterpolator(skeleton_xy, skeleton_z)
        z_base = nearest_interp(points[:, :2])

        # ==========================================================

        self.logger.info(f"    Z_base computed in {time.time() - start_time:.2f}s")
        return z_base.astype(np.float32)

    def _compute_superpoint(self, points: np.ndarray) -> np.ndarray:
        """
        计算超点分割 (Superpoint Segmentation)

        使用基于图的分割算法，将点云划分为几何上连贯的超点块。

        Args:
            points: 点云坐标 (N, 3)

        Returns:
            超点索引数组 (N,)，每个点的超点 ID
        """
        N = len(points)

        # 应用 Z 轴敏感度调整
        if self.sp_z_sensitivity != 1.0:
            xyz_scaled = points.copy()
            xyz_scaled[:, 2] *= self.sp_z_sensitivity
        else:
            xyz_scaled = points

        # 计算法向量
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz_scaled)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.sp_normal_radius, max_nn=30
            )
        )
        pcd.orient_normals_towards_camera_location([0., 0., 0.])
        normals = np.asarray(pcd.normals).astype(np.float32)

        # 构建 KNN 拓扑图
        nbrs = NearestNeighbors(
            n_neighbors=self.sp_edge_k + 1,
            algorithm='kd_tree',
            n_jobs=1
        ).fit(xyz_scaled)
        _, indices = nbrs.kneighbors(xyz_scaled)

        source_nodes = np.repeat(np.arange(N), self.sp_edge_k)
        target_nodes = indices[:, 1:].flatten()
        edges = np.vstack((source_nodes, target_nodes)).T

        # 执行 C++ 超点分割
        vertices_tensor = torch.tensor(xyz_scaled, dtype=torch.float32)
        normals_tensor = torch.tensor(normals, dtype=torch.float32)
        edges_tensor = torch.tensor(edges, dtype=torch.int64)

        sp_idx_tensor = segment_point(
            vertices=vertices_tensor,
            normals=normals_tensor,
            edges=edges_tensor,
            kThresh=self.sp_kThresh,
            segMinVerts=self.sp_segMinVerts,
            segMaxVerts=self.sp_segMaxVerts
        )

        return sp_idx_tensor.numpy().astype(np.int32)

    def _compute_superpoints_for_all_tiles(self):
        """
        批量并行计算所有 tile 的超点分割（tile-post 模式）
        """
        # 收集所有已生成的 tile 文件
        tile_files = list(self.output_dir.glob(f'*.{self.output_format}'))

        if not tile_files:
            self.logger.warning("  No tile files found, skipping superpoint computation.")
            return

        self.logger.info(f"  Found {len(tile_files)} tile files to process")
        self.logger.info(f"  Using {self.sp_num_workers} parallel workers")

        start_time = time.time()

        # 超点参数字典
        sp_params = {
            'kThresh': self.sp_kThresh,
            'segMinVerts': self.sp_segMinVerts,
            'segMaxVerts': self.sp_segMaxVerts,
            'edge_k': self.sp_edge_k,
            'normal_radius': self.sp_normal_radius,
            'z_sensitivity': self.sp_z_sensitivity
        }

        # 并行处理
        completed = 0
        errors = []
        success_count = 0

        if self.sp_num_workers <= 1:
            # 串行模式
            self.logger.info("  Processing in serial mode...")
            for tile_path in tqdm(tile_files, desc="  Computing superpoints"):
                result = _compute_sp_for_single_tile(tile_path, sp_params)
                if result['status'] == 'success':
                    success_count += 1
                else:
                    errors.append(result)
        else:
            # 并行模式
            with ProcessPoolExecutor(max_workers=self.sp_num_workers) as executor:
                futures = {executor.submit(_compute_sp_for_single_tile, tile_path, sp_params): tile_path
                          for tile_path in tile_files}

                for future in tqdm(as_completed(futures), total=len(tile_files), desc="  Computing superpoints"):
                    result = future.result()
                    completed += 1
                    if result['status'] == 'success':
                        success_count += 1
                    else:
                        errors.append(result)
                        self.logger.error(f"    Failed: {result['filename']} - {result['error']}")

        elapsed = time.time() - start_time
        self.logger.info(f"  Superpoint computation completed in {elapsed:.2f}s")
        self.logger.info(f"  Success: {success_count}/{len(tile_files)}, Failed: {len(errors)}")

        if errors:
            self.logger.warning(f"  {len(errors)} tiles failed superpoint computation")
            for err in errors[:5]:  # 只显示前 5 个错误
                self.logger.warning(f"    - {err['filename']}: {err['error']}")

    def _save_tiles(self, las_file: Path, las_data: laspy.LasData,
                    segments: List[Tuple[np.ndarray, np.ndarray]],
                    points: np.ndarray = None, source_hag: np.ndarray = None,
                    source_z_base: np.ndarray = None, source_normals: np.ndarray = None,
                    source_superpoint: np.ndarray = None):
        """
        保存分块为 LAS/LAZ 文件

        Args:
            las_file: 原始文件路径
            las_data: 原始 LAS 数据
            segments: 分块列表，每项为 (含 buffer 的点索引, 核心边界数组 [xmin,ymin,xmax,ymax])
            points: 点云坐标 (N, 3)，用于 tile 模式计算 HAG
            source_hag: 在源点云上预计算的 HAG 值（source 模式）
            source_z_base: 在源点云上预计算的 Z_base 值（source 模式）
            source_normals: 在源点云上预计算的法向量（source 模式）
            source_superpoint: 在源点云上预计算的超点索引（source 模式）
        """
        base_name = las_file.stem
        ext = f".{self.output_format}"
        
        # 获取原始头文件信息
        src_header = las_data.header
        
        # 处理版本兼容性：LAS 1.0/1.1 升级到 1.2（以支持 extra bytes）
        version = src_header.version
        if version.major == 1 and version.minor < 2:
            version = laspy.header.Version(1, 2)
            self.logger.info(f"  Upgrading LAS version from {src_header.version} to {version}")
        
        # 判断是否需要添加 HAG 字段
        need_hag = self.calc_hag
        need_z_base = self.calc_z_base
        need_normal = self.calc_normals
        need_superpoint = self.calc_superpoint
        
        for i, (indices, core_bbox) in enumerate(tqdm(segments, desc="  Saving tiles", leave=False)):
            if len(indices) == 0:
                continue
            
            # 创建新的头文件，保持与原始一致
            header = laspy.LasHeader(
                point_format=src_header.point_format.id,
                version=version
            )
            header.scales = src_header.scales
            header.offsets = src_header.offsets
            
            # 复制 VLRs（包含坐标系信息）
            for vlr in src_header.vlrs:
                header.vlrs.append(vlr)

            # 将核心边界写入 VLR，供 CNF 推理时定位无 buffer 的有效区域
            import json
            bbox_dict = {"core_bbox": core_bbox.tolist()}
            bbox_json = json.dumps(bbox_dict).encode('utf-8')
            bbox_vlr = laspy.VLR(
                user_id="PointSpace",
                record_id=1001,
                description="Core BBox for CNF",
                record_data=bbox_json
            )
            header.vlrs.append(bbox_vlr)
            
            # 复制已有的 extra dimensions
            for extra_dim in src_header.point_format.extra_dimensions:
                header.add_extra_dim(laspy.ExtraBytesParams(
                    name=extra_dim.name,
                    type=extra_dim.dtype,
                    description=extra_dim.description,
                    offsets=extra_dim.offsets,
                    scales=extra_dim.scales
                ))
            
            # 添加 orig_idx 字段
            existing_names = [ed.name for ed in header.point_format.extra_dimensions]
            if self.save_orig_idx and 'orig_idx' not in existing_names:
                header.add_extra_dim(laspy.ExtraBytesParams(
                    name="orig_idx",
                    type=np.uint32,
                    description="Original point index"
                ))
            
            # 添加 hag 字段
            if need_hag and 'hag' not in existing_names:
                header.add_extra_dim(laspy.ExtraBytesParams(
                    name="hag",
                    type=np.float32,
                    description="Height Above Ground"
                ))

            # 注意：如果 calc_z_base 也启用，可以在这里添加 z_base 字段，类似于 hag 的处理
            if need_z_base and 'z_base' not in existing_names:
                header.add_extra_dim(laspy.ExtraBytesParams(
                    name="z_base",
                    type=np.float32,
                    description="CSF Macro Base Surface"
                ))

            if need_normal:
                if 'normal_x' not in existing_names:
                    header.add_extra_dim(laspy.ExtraBytesParams(name="normal_x", type=np.float32, description="Normal Vector X"))
                if 'normal_y' not in existing_names:
                    header.add_extra_dim(laspy.ExtraBytesParams(name="normal_y", type=np.float32, description="Normal Vector Y"))
                if 'normal_z' not in existing_names:
                    header.add_extra_dim(laspy.ExtraBytesParams(name="normal_z", type=np.float32, description="Normal Vector Z"))

            # 添加 superpoint 字段
            if need_superpoint and 'superpoint' not in existing_names:
                header.add_extra_dim(laspy.ExtraBytesParams(
                    name="superpoint",
                    type=np.int32,
                    description="Superpoint Segment ID"
                ))
            
            # 创建新的 LAS 数据
            new_las = laspy.LasData(header)
            
            # 提取子集点
            source_points = las_data.points[indices]
            
            # 复制所有维度（除了我们要单独处理的）
            for dim_name in source_points.array.dtype.names:
                if dim_name in ['orig_idx', 'hag', 'z_base', 'superpoint']:
                    continue  # 跳过，后面单独写入
                if dim_name in new_las.points.array.dtype.names:
                    new_las.points[dim_name] = source_points[dim_name]
            
            # 写入原始索引
            if self.save_orig_idx:
                new_las.orig_idx = indices.astype(np.uint32)
            
            # 写入 HAG
            if need_hag:
                if source_hag is not None:
                    # source 模式：使用预计算的 HAG
                    tile_hag = source_hag[indices].astype(np.float32)
                else:
                    # tile 模式：在每个 tile 上单独计算 HAG
                    tile_points = points[indices]
                    tile_classification = np.array(source_points['classification'])
                    tile_hag = self._compute_hag(tile_points, tile_classification)
                new_las.hag = tile_hag.astype(np.float32)

            if need_z_base:
                if self.z_base_on_source:
                    tile_z_base = source_z_base[indices].astype(np.float32)
                else:
                    tile_points = points[indices]
                    tile_z_base = self._compute_z_base(tile_points)
                new_las.z_base = tile_z_base.astype(np.float32)

            if need_normal:
                if self.normals_on_source:
                    tile_normals = source_normals[indices].astype(np.float32)
                else:
                    # 如果在局部块计算，注意边缘可能出现畸变
                    tile_cls = np.array(source_points['classification']) if self.normal_class is not None else None
                    tile_normals = self._compute_normals(points[indices], classification=tile_cls)

                new_las.normal_x = tile_normals[:, 0]
                new_las.normal_y = tile_normals[:, 1]
                new_las.normal_z = tile_normals[:, 2]

            # 写入 Superpoint
            if need_superpoint:
                if self.superpoint_on_source and source_superpoint is not None:
                    # source 模式：使用预计算的超点索引
                    tile_superpoint = source_superpoint[indices].astype(np.int32)
                    new_las.superpoint = tile_superpoint
                else:
                    # tile-post 模式：暂时填充 -1，后续批量并行计算
                    new_las.superpoint = np.full(len(indices), -1, dtype=np.int32)
            
            # 更新头文件统计信息
            new_las.update_header()
            
            # 保存文件
            out_path = self.output_dir / f"{base_name}_{i:04d}{ext}"
            new_las.write(out_path)
        
        self.logger.info(f"  Saved {len(segments)} tiles to {self.output_dir}")


if __name__ == "__main__":
    processor = LASTileProcessor(
        # 路径与格式配置
        input_path=r"E:\data\铁二院\第二批\new\nl\train",  # 原始数据路径
        output_dir=r"E:\data\铁二院\第二批\new\nl\tile\train", # 输出路径
        output_format='las',          # 输出格式

        # 分块参数
        window_size=(50.0,50.0),   # 分块大小
        overlap=True,                 # 启用重叠
        overlap_factor=1,             # 重叠因子
        min_points=5000,              # 最小点数
        max_points=None,              # 最大点数限制（None=不限制）
        save_orig_idx=True,           # 保存原始点索引
        buffer_size=0,

        # Superpoint 超点分割参数
        calc_superpoint=False,         # 启用超点分割
        superpoint_on_source=False,   # False: 先 tile 再批量并行计算（推荐，更高效）
                                      # True: 先在原始点云计算再 tile（避免边界效应）
        sp_kThresh=0.005,              # 分割阈值，越大块越大
        sp_segMinVerts=5,             # 最小超点点数
        sp_segMaxVerts=4096,          # 最大超点点数（-1 表示不限制）
        sp_edge_k=10,                 # KNN 边数
        sp_normal_radius=2.0,         # 法向量搜索半径
        sp_z_sensitivity=2.0,         # Z 轴敏感度（>1 增强高度差异）
        sp_num_workers=8,             # 超点并行计算进程数（tile-post 模式）

        # 法向量参数
        calc_normals=False,           # 启用法向量计算
        normal_on_source=True,        # 在原始点云计算法向量（推荐，避免边界效应）
        normal_k_neighbors=30,        # 法向量 K 近邻数
        normal_class=[2],              # 仅使用地面点计算法向量（如果 None 则使用全部点）

        # HAG 参数
        calc_hag=False,                # 启用 HAG 计算
        hag_ground_class=2,           # DALES: 地面点类别为1（原始DALES类别）
        hag_on_source=True,           # 在原始点云计算（推荐，避免边界效应）
        hag_k_neighbors=12,           # IDW 插值邻居数
        hag_power=2.0,                # IDW 幂次（反距离平方）

        # Z_base 参数
        calc_z_base=False,             # 启用 Z_base 计算
        z_base_on_source=True,         # 在原始点云计算（强烈建议，消除块间断层）
        z_base_denoise_radius=2.0,    # 局部搜索半径(米)
        z_base_denoise_elev_diff=2.0, # 判定为高空/地下噪声的高差阈值(米)
        z_base_ptd_radius=5.0,       # 提取地面的最大搜索半径 (大型建筑多可设大至20-30)
        z_base_ptd_slope=15.0,        # 坡度阈值 (山区可设大，如20-30；城市设小，如10)
        z_base_ptd_height=0.25,       # 贴地精度阈值 (越小越贴地，但也容易漏掉微起伏)
        z_base_ptd_slope_norm=True,   # 是否对坡度进行归一化（推荐，适应不同地形）
    )
    processor.process_all_files()
