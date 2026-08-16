"""
HPSD Point Feature Analyzer

对 HPSD Safetensors 点特征进行 PCA、MiniBatchKMeans 分析和 LAS/LAZ 赋色：
- 点云和特征按同名文件配对，并严格检查点数；
- PCA 与 KMeans 模型只在受控随机样本上拟合；
- 大数据采样使用固定内存上限和 Safetensors 行切片；
- 分析阶段支持 GPU 归一化/PCA 以及 CPU-I/O 预取流水线；
- PCA、KMeans 分别输出赋色后的 tile 点云；
- 可利用 ``orig_idx`` 对重叠 tile 去重并恢复源点顺序；
- 分析模型以 Safetensors 保存，不使用 pickle。
"""

from __future__ import annotations

import colorsys
import copy
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Sequence, Union

import laspy
import numpy as np
import torch
from laspy.vlrs.vlrlist import VLRList
from safetensors import safe_open
from safetensors.torch import save_file as save_safetensors
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from tqdm import tqdm

try:
    from pointspace.utils.logger import get_root_logger
except ImportError:
    def get_root_logger():
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
        )
        return logging.getLogger(__name__)


class HPSDFeatureAnalyzer:
    """HPSD 点特征分析、赋色和 tile 合并工具。"""

    def __init__(
        self,
        pointcloud_path: Union[str, Path],
        feature_path: Union[str, Path],
        output_dir: Union[str, Path],
        methods: Sequence[str] = ('pca', 'kmeans'),
        sample_ratio: float = 0.1,
        max_samples: int = 50000,
        pca_dim: int = 32,
        kmeans_k: int = 16,
        kmeans_batch_size: int = 65536,
        transform_batch_size: int = 65536,
        device: str = 'auto',
        prefetch_tiles: int = 1,
        core_sample_only: bool = True,
        merge_tiles: bool = True,
        overwrite: bool = False,
        random_seed: int = 42,
    ):
        self.pointcloud_path = Path(pointcloud_path)
        self.feature_path = Path(feature_path)
        self.output_dir = Path(output_dir)
        self.methods = tuple(dict.fromkeys(method.lower() for method in methods))
        self.sample_ratio = float(sample_ratio)
        self.max_samples = int(max_samples)
        self.pca_dim = int(pca_dim)
        self.kmeans_k = int(kmeans_k)
        self.kmeans_batch_size = int(kmeans_batch_size)
        self.transform_batch_size = int(transform_batch_size)
        self.device = str(device).lower()
        self.prefetch_tiles = int(prefetch_tiles)
        self.core_sample_only = bool(core_sample_only)
        self.merge_tiles = bool(merge_tiles)
        self.overwrite = bool(overwrite)
        self.random_seed = int(random_seed)
        self.logger = get_root_logger()

        invalid_methods = set(self.methods).difference({'pca', 'kmeans'})
        if invalid_methods:
            raise ValueError(f'Unsupported methods: {sorted(invalid_methods)}')
        if not self.methods:
            raise ValueError('At least one method is required')
        if not 0 < self.sample_ratio <= 1:
            raise ValueError('sample_ratio must be in (0, 1]')
        if self.max_samples <= 0 or self.pca_dim < 3 or self.kmeans_k <= 1:
            raise ValueError('max_samples, pca_dim and kmeans_k are invalid')
        if self.transform_batch_size <= 0:
            raise ValueError('transform_batch_size must be positive')
        if self.prefetch_tiles not in {0, 1}:
            raise ValueError('prefetch_tiles currently supports only 0 or 1')
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('CUDA analysis was requested but CUDA is unavailable')
        self.torch_device = torch.device(self.device)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pairs = self._pair_inputs()
        self.pca = None
        self.kmeans = None
        self.rgb_low = None
        self.rgb_high = None

    @staticmethod
    def _list_files(path: Path, suffixes):
        if path.is_file():
            if path.suffix.lower() not in suffixes:
                raise ValueError(f'Unsupported file: {path}')
            return [path]
        if not path.is_dir():
            raise FileNotFoundError(path)
        files = sorted(
            file for file in path.iterdir()
            if file.is_file() and file.suffix.lower() in suffixes
        )
        if not files:
            raise FileNotFoundError(f'No supported files found in {path}')
        return files

    def _pair_inputs(self):
        point_files = self._list_files(self.pointcloud_path, {'.las', '.laz'})
        feature_files = self._list_files(self.feature_path, {'.safetensors'})
        feature_index = {file.stem: file for file in feature_files}
        missing = [file.stem for file in point_files if file.stem not in feature_index]
        if missing:
            preview = ', '.join(missing[:8])
            raise FileNotFoundError(f'Missing same-stem HPSD features: {preview}')
        pairs = [(file, feature_index[file.stem]) for file in point_files]
        self.logger.info(f'Paired {len(pairs)} point clouds and HPSD features')
        return pairs

    @staticmethod
    def _load_feature(path: Path):
        with safe_open(str(path), framework='np') as file:
            if 'feature' not in file.keys():
                raise ValueError(f'No feature tensor in {path}')
            feature = file.get_tensor('feature')
            metadata = file.metadata() or {}
        if feature.ndim != 2:
            raise ValueError(f'Feature must use [N,C] layout: {path}')
        if metadata.get('layout', 'NC').upper() != 'NC':
            raise ValueError(f'Unsupported feature layout in {path}')
        return feature, metadata

    @staticmethod
    def _load_feature_rows(path: Path, indices):
        """只读取随机采样需要的特征行，避免为每个 tile 加载完整 ``[N,C]``。

        Safetensors 支持连续 slice，但不支持 NumPy fancy indexing。采样行很少
        时按排序后的连续 run 读取；采样比例较大时一次读取完整张量再索引，
        避免大量细碎 slice 调用。返回值固定为 float32，供 PCA/KMeans 拟合。
        """
        indices = np.sort(np.asarray(indices, dtype=np.int64))
        with safe_open(str(path), framework='np') as file:
            if 'feature' not in file.keys():
                raise ValueError(f'No feature tensor in {path}')
            feature_slice = file.get_slice('feature')
            shape = tuple(feature_slice.get_shape())
            metadata = file.metadata() or {}
            if len(shape) != 2:
                raise ValueError(f'Feature must use [N,C] layout: {path}')
            if metadata.get('layout', 'NC').upper() != 'NC':
                raise ValueError(f'Unsupported feature layout in {path}')
            if indices.size == 0:
                return np.empty((0, shape[1]), dtype=np.float32), metadata, shape
            if indices[0] < 0 or indices[-1] >= shape[0]:
                raise IndexError(f'Feature row index exceeds tensor shape in {path}')

            # 当所需行超过总行数的 5%（且至少 1024 行）时，顺序读取整块通常
            # 比数千次随机小 slice 更快；大数据集的全局预算会使单 tile 配额
            # 很小，从而自动走下面的稀疏读取路径。
            if indices.size >= max(1024, int(np.ceil(shape[0] * 0.05))):
                feature = file.get_tensor('feature')
                return (
                    np.asarray(feature[indices], dtype=np.float32),
                    metadata,
                    shape,
                )

            output = np.empty((indices.size, shape[1]), dtype=np.float32)
            run_start = 0
            while run_start < indices.size:
                run_end = run_start + 1
                while (
                    run_end < indices.size
                    and indices[run_end] == indices[run_end - 1] + 1
                ):
                    run_end += 1
                source_start = int(indices[run_start])
                source_end = int(indices[run_end - 1]) + 1
                output[run_start:run_end] = np.asarray(
                    feature_slice[source_start:source_end], dtype=np.float32
                )
                run_start = run_end
        return output, metadata, shape

    @staticmethod
    def _core_bbox(las):
        for vlr in las.header.vlrs:
            if getattr(vlr, 'user_id', None) == 'PointSpace' and getattr(vlr, 'record_id', None) == 1001:
                try:
                    return np.asarray(
                        json.loads(vlr.record_data.decode('utf-8'))['core_bbox'],
                        dtype=np.float64,
                    )
                except (KeyError, ValueError, UnicodeDecodeError):
                    return None
        return None

    @classmethod
    def _core_indices(cls, las):
        bbox = cls._core_bbox(las)
        if bbox is None:
            return np.arange(len(las.points), dtype=np.int64)
        x = np.asarray(las.x)
        y = np.asarray(las.y)
        return np.flatnonzero(
            (x >= bbox[0]) & (x <= bbox[2]) &
            (y >= bbox[1]) & (y <= bbox[3])
        )

    @staticmethod
    def _normalize(feature):
        feature = np.asarray(feature, dtype=np.float32)
        # einsum 避免 np.linalg.norm 在大型 [N,C] 数组上产生较慢的
        # 通用路径，对 1024 维 HPSD 特征的 CPU 归一化更高效。
        norm = np.sqrt(np.einsum('ij,ij->i', feature, feature))[:, None]
        return feature / np.maximum(norm, 1e-12)

    def _sample_features(self):
        """在 ``max_samples`` 硬上限内对所有 tile 做近似等额随机采样。

        旧实现先保存每个 tile 的采样块，再 concatenate 和全局截断；在大数据
        集上会于截断前产生数十 GiB 临时数组。当前实现根据“剩余预算 / 剩余
        tile”动态分配配额，并直接写入固定容量 buffer，因此峰值样本存储约为
        ``max_samples * feature_dim * 4`` 字节。
        """
        rng = np.random.default_rng(self.random_seed)
        # 随机化 tile 顺序，使 max_samples 小于 tile 数时也不会总是选择文件名
        # 排在前面的 tile。配额仍会随未使用预算动态重新分配。
        pair_order = rng.permutation(len(self.pairs))
        sample_pairs = [self.pairs[index] for index in pair_order]
        sample_buffer = None
        sample_count = 0
        feature_dim = None

        progress = tqdm(
            sample_pairs, desc='Sampling HPSD features', unit='tile'
        )
        for tile_position, (point_path, feature_path) in enumerate(progress):
            remaining_budget = self.max_samples - sample_count
            remaining_tiles = len(sample_pairs) - tile_position
            if remaining_budget <= 0:
                break

            # ceil 可保证在所有 tile 都有足够候选点时最终填满全局预算；若某块
            # 点数不足，未使用配额会自动转移给后续 tile。
            tile_budget = int(np.ceil(remaining_budget / remaining_tiles))
            with safe_open(str(feature_path), framework='np') as file:
                if 'feature' not in file.keys():
                    raise ValueError(f'No feature tensor in {feature_path}')
                feature_shape = tuple(file.get_slice('feature').get_shape())
                feature_metadata = file.metadata() or {}
            if len(feature_shape) != 2:
                raise ValueError(f'Feature must use [N,C] layout: {feature_path}')
            if feature_metadata.get('layout', 'NC').upper() != 'NC':
                raise ValueError(f'Unsupported feature layout in {feature_path}')

            num_points = feature_shape[0]
            with laspy.open(point_path) as reader:
                las_num_points = reader.header.point_count
            if num_points != las_num_points:
                raise ValueError(
                    f'Point/feature count mismatch for {point_path.name}: '
                    f'{las_num_points} != {num_points}'
                )

            if self.core_sample_only:
                candidate = self._core_indices(laspy.read(point_path))
            else:
                candidate = np.arange(num_points, dtype=np.int64)
            if candidate.size == 0 or tile_budget == 0:
                continue
            ratio_count = max(1, int(round(candidate.size * self.sample_ratio)))
            tile_sample_count = min(candidate.size, ratio_count, tile_budget)
            selected = rng.choice(
                candidate, size=tile_sample_count, replace=False
            )
            chunk, _, checked_shape = self._load_feature_rows(
                feature_path, selected
            )
            if checked_shape != feature_shape:
                raise RuntimeError(f'Feature shape changed while reading {feature_path}')

            if sample_buffer is None:
                feature_dim = feature_shape[1]
                buffer_gib = self.max_samples * feature_dim * 4 / 1024 ** 3
                self.logger.info(
                    f'Bounded sampling buffer: max_samples={self.max_samples}, '
                    f'feature_dim={feature_dim}, float32={buffer_gib:.2f} GiB'
                )
                sample_buffer = np.empty(
                    (self.max_samples, feature_dim), dtype=np.float32
                )
            elif feature_shape[1] != feature_dim:
                raise ValueError(
                    f'Feature dimension mismatch for {feature_path}: '
                    f'{feature_shape[1]} != {feature_dim}'
                )
            next_count = sample_count + chunk.shape[0]
            sample_buffer[sample_count:next_count] = chunk
            sample_count = next_count
            progress.set_postfix(samples=sample_count, refresh=False)

        if sample_buffer is None or sample_count == 0:
            raise RuntimeError('No HPSD features were sampled')
        sample = sample_buffer[:sample_count]
        # 分块原地归一化。np.linalg.norm 的中间计算也会占用内存，
        # 因此不对整个样本矩阵一次计算。
        normalize_batch_size = min(self.transform_batch_size, 16384)
        for start in range(0, sample_count, normalize_batch_size):
            end = min(start + normalize_batch_size, sample_count)
            chunk = sample[start:end]
            norm = np.linalg.norm(chunk, axis=1, keepdims=True)
            chunk /= np.maximum(norm, 1e-12)
        return sample

    def _fit_models(self, sample):
        pca_dim = min(self.pca_dim, sample.shape[0], sample.shape[1])
        if pca_dim < 3:
            raise ValueError('At least three PCA components are required for coloring')
        self.pca = PCA(
            n_components=pca_dim,
            svd_solver='randomized',
            random_state=self.random_seed,
        )
        sample_pca = self.pca.fit_transform(sample)
        self.rgb_low = np.percentile(sample_pca[:, :3], 1, axis=0).astype(np.float32)
        self.rgb_high = np.percentile(sample_pca[:, :3], 99, axis=0).astype(np.float32)
        if 'kmeans' in self.methods:
            if sample.shape[0] < self.kmeans_k:
                raise ValueError('Sample count is smaller than kmeans_k')
            self.kmeans = MiniBatchKMeans(
                n_clusters=self.kmeans_k,
                batch_size=self.kmeans_batch_size,
                n_init=3,
                random_state=self.random_seed,
            )
            self.kmeans.fit(sample_pca)
        self._save_analysis_model()

    def _save_analysis_model(self):
        tensors = {
            'pca_components': torch.from_numpy(
                self.pca.components_.astype(np.float32)
            ).contiguous(),
            'pca_mean': torch.from_numpy(
                self.pca.mean_.astype(np.float32)
            ).contiguous(),
            'pca_explained_variance': torch.from_numpy(
                self.pca.explained_variance_.astype(np.float32)
            ).contiguous(),
            'pca_rgb_low': torch.from_numpy(self.rgb_low).contiguous(),
            'pca_rgb_high': torch.from_numpy(self.rgb_high).contiguous(),
        }
        if self.kmeans is not None:
            tensors['kmeans_centers'] = torch.from_numpy(
                self.kmeans.cluster_centers_.astype(np.float32)
            ).contiguous()
        metadata = {
            'format_version': '1',
            'methods': json.dumps(self.methods),
            'pca_dim': str(self.pca.n_components_),
            'kmeans_k': str(self.kmeans_k if self.kmeans is not None else 0),
            'normalized_input': 'true',
        }
        save_safetensors(
            tensors,
            str(self.output_dir / 'analysis_model.safetensors'),
            metadata,
        )

    def _transform_feature_cpu(self, feature):
        """CPU 分块投影，并预分配输出以避免 concatenate 拷贝。"""
        pca_feature = np.empty(
            (feature.shape[0], self.pca.n_components_), dtype=np.float32
        )
        for start in range(0, feature.shape[0], self.transform_batch_size):
            end = min(start + self.transform_batch_size, feature.shape[0])
            chunk = self._normalize(feature[start:end])
            pca_feature[start:end] = self.pca.transform(chunk).astype(
                np.float32, copy=False
            )
        labels = (
            self.kmeans.predict(pca_feature).astype(np.int32, copy=False)
            if self.kmeans is not None else None
        )
        return pca_feature, labels

    def _transform_feature_cuda(self, feature):
        """GPU 分块完成 L2 归一化和 PCA，仅回传低维投影。"""
        pca_feature = np.empty(
            (feature.shape[0], self.pca.n_components_), dtype=np.float32
        )
        components = torch.as_tensor(
            self.pca.components_, dtype=torch.float32, device=self.torch_device
        )
        mean = torch.as_tensor(
            self.pca.mean_, dtype=torch.float32, device=self.torch_device
        )
        bias = -(mean @ components.T)
        with torch.inference_mode():
            for start in range(0, feature.shape[0], self.transform_batch_size):
                end = min(start + self.transform_batch_size, feature.shape[0])
                chunk = torch.from_numpy(feature[start:end]).to(
                    self.torch_device, dtype=torch.float32, non_blocking=False
                )
                chunk = torch.nn.functional.normalize(chunk, dim=1)
                projected = torch.addmm(bias, chunk, components.T)
                pca_feature[start:end] = projected.cpu().numpy()
        labels = (
            self.kmeans.predict(pca_feature).astype(np.int32, copy=False)
            if self.kmeans is not None else None
        )
        return pca_feature, labels

    def _transform_feature(self, feature):
        if self.torch_device.type == 'cuda':
            return self._transform_feature_cuda(feature)
        return self._transform_feature_cpu(feature)

    def _pca_rgb(self, pca_feature):
        value = (pca_feature[:, :3] - self.rgb_low) / np.maximum(
            self.rgb_high - self.rgb_low, 1e-12
        )
        value = np.power(np.clip(value, 0.0, 1.0), 0.85)
        return (value * 65535.0).round().astype(np.uint16)

    def _cluster_rgb(self, labels):
        palette = np.asarray(
            [
                colorsys.hsv_to_rgb((index * 0.61803398875) % 1.0, 0.72, 0.95)
                for index in range(self.kmeans_k)
            ],
            dtype=np.float32,
        )
        return (palette[labels] * 65535.0).round().astype(np.uint16)

    @staticmethod
    def _ensure_rgb(las):
        names = set(las.point_format.dimension_names)
        if {'red', 'green', 'blue'}.issubset(names):
            return las
        target_format = 7 if las.point_format.id >= 6 else 3
        return laspy.convert(las, point_format_id=target_format)

    @staticmethod
    def _ensure_kmeans_dimension(las):
        if 'hpsd_kmeans' not in set(las.point_format.dimension_names):
            las.add_extra_dim(
                laspy.ExtraBytesParams(
                    name='hpsd_kmeans',
                    type=np.int32,
                    description='HPSD KMeans cluster',
                )
            )
        return las

    def _write_colored_tile(self, source_path, output_path, rgb, labels=None):
        las = self._ensure_rgb(laspy.read(source_path))
        las.red, las.green, las.blue = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        if labels is not None:
            las = self._ensure_kmeans_dimension(las)
            las.hpsd_kmeans = labels
        output_path.parent.mkdir(parents=True, exist_ok=True)
        las.write(output_path)

    def _analyze_tiles(self):
        outputs = {method: [] for method in self.methods}
        tasks = []
        for point_path, feature_path in self.pairs:
            output_paths = {
                method: self.output_dir / method / 'tiles' / point_path.name
                for method in self.methods
            }
            for method, output_path in output_paths.items():
                outputs[method].append(output_path)
            required = {
                method for method, output_path in output_paths.items()
                if self.overwrite or not output_path.exists()
            }
            # 所有目标都已存在时，不再读取数百 MiB 的特征并重复
            # 归一化/PCA。这使 overwrite=False 可以真正支持断点续跑。
            if required:
                tasks.append((point_path, feature_path, output_paths, required))

        if not tasks:
            self.logger.info('All analyzed tiles already exist; skipping transform')
            return outputs

        self.logger.info(
            f'Analyzing {len(tasks)}/{len(self.pairs)} tiles on {self.torch_device}; '
            f'feature prefetch={self.prefetch_tiles}'
        )

        def analyze(task, feature):
            point_path, _, output_paths, required = task
            with laspy.open(point_path) as reader:
                num_points = reader.header.point_count
            if feature.shape[0] != num_points:
                raise ValueError(f'Point/feature count mismatch for {point_path.name}')
            pca_feature, labels = self._transform_feature(feature)

            write_jobs = []
            if 'pca' in required:
                write_jobs.append(
                    (output_paths['pca'], self._pca_rgb(pca_feature), None)
                )
            if 'kmeans' in required:
                write_jobs.append(
                    (
                        output_paths['kmeans'],
                        self._cluster_rgb(labels),
                        labels,
                    )
                )
            if len(write_jobs) == 1:
                self._write_colored_tile(point_path, *write_jobs[0])
            else:
                # PCA 和 KMeans 是两个独立的 LAZ 压缩任务，并行写出
                # 可以利用多核 CPU，且不阻塞下一块的特征预取。
                with ThreadPoolExecutor(max_workers=2) as writer:
                    futures = [
                        writer.submit(
                            self._write_colored_tile,
                            point_path,
                            output_path,
                            rgb,
                            job_labels,
                        )
                        for output_path, rgb, job_labels in write_jobs
                    ]
                    for future in futures:
                        future.result()

        progress = tqdm(total=len(tasks), desc='Analyzing HPSD features', unit='tile')
        if self.prefetch_tiles == 0:
            for task in tasks:
                feature, _ = self._load_feature(task[1])
                analyze(task, feature)
                progress.update()
        else:
            # 单读取线程预取下一块，与当前 tile 的 GPU 投影及
            # LAS/LAZ 写盘重叠；不扩大到多个 worker，以免内存和磁盘
            # 随机读压力过大。
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self._load_feature, tasks[0][1])
                for index, task in enumerate(tasks):
                    feature, _ = future.result()
                    if index + 1 < len(tasks):
                        future = executor.submit(
                            self._load_feature, tasks[index + 1][1]
                        )
                    analyze(task, feature)
                    progress.update()
        progress.close()
        return outputs

    @staticmethod
    def _source_name(stem):
        match = re.match(r'^(.+)_\d{4}$', stem)
        return match.group(1) if match else stem

    def _merge_group(self, tile_paths, output_path):
        point_arrays = []
        orig_indices = []
        first_las = None
        reference_dtype = None
        reference_scales = None
        reference_offsets = None
        for tile_path in tile_paths:
            las = laspy.read(tile_path)
            if 'orig_idx' not in set(las.point_format.dimension_names):
                raise ValueError(
                    f'{tile_path} has no orig_idx; reliable tile merging is impossible'
                )
            if first_las is None:
                first_las = las
                reference_dtype = las.points.array.dtype
                reference_scales = np.asarray(las.header.scales)
                reference_offsets = np.asarray(las.header.offsets)
            if las.points.array.dtype != reference_dtype:
                raise ValueError(f'Point formats differ inside source group: {tile_path}')
            if not np.array_equal(las.header.scales, reference_scales) or not np.array_equal(
                las.header.offsets, reference_offsets
            ):
                raise ValueError(f'LAS scales/offsets differ inside source group: {tile_path}')
            point_arrays.append(las.points.array.copy())
            orig_indices.append(np.asarray(las.orig_idx, dtype=np.uint64))

        all_points = np.concatenate(point_arrays)
        all_orig_idx = np.concatenate(orig_indices)
        order = np.argsort(all_orig_idx, kind='stable')
        sorted_orig_idx = all_orig_idx[order]
        keep = np.ones(order.shape[0], dtype=bool)
        keep[1:] = sorted_orig_idx[1:] != sorted_orig_idx[:-1]
        selected = all_points[order[keep]].copy()

        header = copy.deepcopy(first_las.header)
        header.vlrs = VLRList(
            [
                vlr for vlr in header.vlrs
                if not (
                    getattr(vlr, 'user_id', None) == 'PointSpace'
                    and getattr(vlr, 'record_id', None) == 1001
                )
            ]
        )
        merged = laspy.LasData(header)
        merged.points = laspy.ScaleAwarePointRecord(
            selected,
            header.point_format,
            header.scales,
            header.offsets,
        )
        merged.update_header()
        bbox = [
            float(np.min(merged.x)), float(np.min(merged.y)),
            float(np.max(merged.x)), float(np.max(merged.y)),
        ]
        merged.header.vlrs.append(
            laspy.VLR(
                user_id='PointSpace',
                record_id=1001,
                description='Merged BBox',
                record_data=json.dumps({'core_bbox': bbox}).encode('utf-8'),
            )
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        merged.write(output_path)
        return len(all_points), len(selected)

    def _merge_tiles(self, outputs):
        tasks = []
        for method, tile_paths in outputs.items():
            groups = {}
            for tile_path in tile_paths:
                groups.setdefault(self._source_name(tile_path.stem), []).append(tile_path)
            for source_name, group in groups.items():
                suffix = group[0].suffix
                output_path = self.output_dir / method / 'merged' / f'{source_name}{suffix}'
                tasks.append((method, source_name, group, output_path))

        for method, source_name, group, output_path in tqdm(
            tasks, desc='Merging analyzed tiles', unit='source'
        ):
            if output_path.exists() and not self.overwrite:
                continue
            before, after = self._merge_group(group, output_path)
            self.logger.info(
                f'  {method}/{source_name}: {before} tile points -> {after} unique points'
            )

    def process(self):
        sample = self._sample_features()
        self.logger.info(
            f'Fitting PCA/KMeans with {sample.shape[0]} x {sample.shape[1]} samples'
        )
        self._fit_models(sample)
        outputs = self._analyze_tiles()
        if self.merge_tiles:
            self._merge_tiles(outputs)
        self.logger.info(f'Analysis completed: {self.output_dir}')
        return outputs


def analyze_hpsd_features(
    pointcloud_path,
    feature_path,
    output_dir,
    methods=('pca', 'kmeans'),
    sample_ratio=0.1,
    max_samples=50000,
    pca_dim=32,
    kmeans_k=16,
    kmeans_batch_size=65536,
    transform_batch_size=65536,
    device='auto',
    prefetch_tiles=1,
    core_sample_only=True,
    merge_tiles=True,
    overwrite=False,
    random_seed=42,
):
    analyzer = HPSDFeatureAnalyzer(
        pointcloud_path=pointcloud_path,
        feature_path=feature_path,
        output_dir=output_dir,
        methods=methods,
        sample_ratio=sample_ratio,
        max_samples=max_samples,
        pca_dim=pca_dim,
        kmeans_k=kmeans_k,
        kmeans_batch_size=kmeans_batch_size,
        transform_batch_size=transform_batch_size,
        device=device,
        prefetch_tiles=prefetch_tiles,
        core_sample_only=core_sample_only,
        merge_tiles=merge_tiles,
        overwrite=overwrite,
        random_seed=random_seed,
    )
    return analyzer.process()


if __name__ == '__main__':
    analyze_hpsd_features(
        pointcloud_path=r'E:\data\云南\data\tile',
        feature_path=r'E:\data\云南\hpsd_feature\litept_v1m4',
        output_dir=r'E:\data\云南\hpsd_analysis\litept_v1m4',
        methods=('pca', 'kmeans'),
        sample_ratio=0.05,
        max_samples=200000,
        pca_dim=32,
        kmeans_k=16,
        kmeans_batch_size=16384,
        transform_batch_size=65536,
        # auto 优先使用 CUDA 进行归一化和 PCA，cpu 可强制关闭。
        device='auto',
        # 预取 1 个 tile 可重叠特征读盘与 GPU/LAS 处理；0 为串行。
        prefetch_tiles=1,
        merge_tiles=True,
        overwrite=False,
    )
