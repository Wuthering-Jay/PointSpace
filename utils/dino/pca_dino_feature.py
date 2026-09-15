"""
DINO Feature PCA Reducer

对 ``extract_dino_feature.py`` 生成的 Safetensors 特征执行全目录共享 PCA：
- 从整个输入目录中抽样文件和 patch，使用固定全局样本预算；
- 所有文件共享同一个 PCA mean/components，而不是逐文件独立拟合；
- 逐文件、分 batch 变换，避免同时加载全部特征；
- 保留原始 DINO metadata 和其他 tensor，仅替换 ``feature``；
- 输入输出目录相同时使用临时文件和 ``os.replace`` 原子覆盖；
- PCA 参数以 Safetensors 保存，便于复现和处理后续新增特征。
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from uuid import uuid4

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file as save_safetensors
from sklearn.decomposition import PCA
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


PCA_MODEL_NAME = 'dino_pca_model.safetensors'


class DINOFeaturePCAReducer:
    """
    对一个目录内的 DINO patch 特征拟合并应用共享 PCA。

    Args:
        input_path: ``extract_dino_feature.py`` 输出的 Safetensors 目录
        output_path: 降维特征目录；与 input_path 相同时原子覆盖源文件
        target_dim: PCA 输出通道数，必须小于原始特征维度
        sample_ratio: 被选中拟合文件内期望采样的 patch 比例
        max_samples: PCA 拟合的全局最大 patch 数
        max_sample_files: 最多使用多少个随机文件拟合 PCA；None 表示全部
        normalize_input: PCA 前是否对每个 patch 特征执行 L2 归一化
        whiten: 是否启用 sklearn PCA whitening
        svd_solver: sklearn PCA solver，推荐高维数据使用 ``randomized``
        iterated_power: randomized PCA 的 power iteration 次数或 ``auto``
        n_oversamples: randomized PCA 的额外采样维度
        power_iteration_normalizer: randomized PCA power iteration 归一化方式
        transform_batch_size: 每批读取和投影的 patch 数
        output_dtype: 输出 ``feature`` 类型，支持 float16/float32
        overwrite: 输出目录不同时是否覆盖已有同名文件；原地模式始终覆盖
        random_seed: 文件、patch 抽样和 PCA 的共同随机种子
        save_pca_result: 是否保存每个输入文件对应的 PCA 降维特征
        save_pca_model: 是否保存共享 PCA 参数
        pca_model_name: PCA 参数文件名；不会被当作输入特征处理
    """

    def __init__(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        target_dim: int = 128,
        sample_ratio: float = 0.05,
        max_samples: int = 100000,
        max_sample_files: Optional[int] = 64,
        normalize_input: bool = True,
        whiten: bool = False,
        svd_solver: str = 'randomized',
        iterated_power: Union[str, int] = 'auto',
        n_oversamples: int = 10,
        power_iteration_normalizer: str = 'auto',
        transform_batch_size: int = 65536,
        output_dtype: str = 'float16',
        overwrite: bool = False,
        random_seed: int = 42,
        save_pca_result: bool = True,
        save_pca_model: bool = True,
        pca_model_name: str = PCA_MODEL_NAME,
    ):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.target_dim = int(target_dim)
        self.sample_ratio = float(sample_ratio)
        self.max_samples = int(max_samples)
        self.max_sample_files = (
            None if max_sample_files is None else int(max_sample_files)
        )
        self.normalize_input = bool(normalize_input)
        self.whiten = bool(whiten)
        self.svd_solver = str(svd_solver).lower()
        self.iterated_power = iterated_power
        self.n_oversamples = int(n_oversamples)
        normalizer_key = str(power_iteration_normalizer).lower()
        self.power_iteration_normalizer = {
            'auto': 'auto',
            'none': 'none',
            'qr': 'QR',
            'lu': 'LU',
        }.get(normalizer_key, str(power_iteration_normalizer))
        self.transform_batch_size = int(transform_batch_size)
        self.output_dtype = str(output_dtype).lower()
        self.overwrite = bool(overwrite)
        self.random_seed = int(random_seed)
        self.save_pca_result = bool(save_pca_result)
        self.save_pca_model = bool(save_pca_model)
        self.pca_model_name = str(pca_model_name)
        self.logger = get_root_logger()

        self._validate_parameters()
        self.input_path = self.input_path.resolve()
        self.output_path = self.output_path.resolve()
        self.in_place = self.input_path == self.output_path
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.feature_files = self._find_feature_files()
        self.file_info, self.input_dim = self._scan_feature_headers()
        self.pca = None

    def _validate_parameters(self):
        if not self.input_path.is_dir():
            raise ValueError(f'input_path must be a directory: {self.input_path}')
        if self.target_dim <= 0:
            raise ValueError('target_dim must be positive')
        if not 0 < self.sample_ratio <= 1:
            raise ValueError('sample_ratio must be in (0, 1]')
        if self.max_samples <= 0:
            raise ValueError('max_samples must be positive')
        if self.max_sample_files is not None and self.max_sample_files <= 0:
            raise ValueError('max_sample_files must be positive or None')
        if self.transform_batch_size <= 0:
            raise ValueError('transform_batch_size must be positive')
        if self.output_dtype not in ('float16', 'float32'):
            raise ValueError("output_dtype must be 'float16' or 'float32'")
        if self.svd_solver not in ('auto', 'full', 'covariance_eigh', 'arpack', 'randomized'):
            raise ValueError('Unsupported sklearn PCA svd_solver')
        if self.n_oversamples <= 0:
            raise ValueError('n_oversamples must be positive')
        if self.power_iteration_normalizer not in ('auto', 'QR', 'LU', 'none'):
            raise ValueError('Unsupported power_iteration_normalizer')
        if not self.pca_model_name.lower().endswith('.safetensors'):
            raise ValueError('pca_model_name must end with .safetensors')

    def _find_feature_files(self) -> List[Path]:
        files = sorted(
            path for path in self.input_path.iterdir()
            if (
                path.is_file()
                and path.suffix.lower() == '.safetensors'
                and path.name != self.pca_model_name
            )
        )
        if not files:
            raise ValueError(f'No DINO Safetensors files found: {self.input_path}')
        self.logger.info(f'Found {len(files)} DINO feature files')
        return files

    @staticmethod
    def _read_header(path: Path) -> Tuple[Tuple[int, int], Dict[str, str]]:
        with safe_open(path, framework='numpy') as file:
            if 'feature' not in file.keys():
                raise ValueError(f"Safetensors file has no 'feature': {path}")
            shape = tuple(file.get_slice('feature').get_shape())
            metadata = dict(file.metadata() or {})
        if len(shape) != 2:
            raise ValueError(f'feature must have [patch, channel] shape: {path}')
        return shape, metadata

    def _scan_feature_headers(self):
        file_info = []
        input_dim = None
        total_patches = 0
        for path in tqdm(self.feature_files, desc='Scan DINO features', unit='file'):
            shape, metadata = self._read_header(path)
            if shape[0] <= 0 or shape[1] <= 0:
                raise ValueError(f'Empty DINO feature tensor: {path}')
            if input_dim is None:
                input_dim = shape[1]
            elif shape[1] != input_dim:
                raise ValueError(
                    f'Feature dimension mismatch for {path.name}: '
                    f'{shape[1]} != {input_dim}'
                )
            file_info.append({
                'path': path,
                'rows': shape[0],
                'metadata': metadata,
            })
            total_patches += shape[0]

        if self.target_dim >= input_dim:
            raise ValueError(
                f'target_dim ({self.target_dim}) must be smaller than '
                f'input feature dimension ({input_dim})'
            )
        self.logger.info(
            f'DINO feature matrix: {total_patches:,} x {input_dim}; '
            f'target_dim={self.target_dim}'
        )
        return file_info, input_dim

    @staticmethod
    def _normalize(feature: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(feature, axis=1, keepdims=True)
        return feature / np.maximum(norm, 1e-12)

    @staticmethod
    def _load_feature_rows(path: Path, indices: np.ndarray) -> np.ndarray:
        """读取抽样行；稠密抽样顺序读取整张量，稀疏抽样读取连续 runs。"""
        indices = np.sort(np.asarray(indices, dtype=np.int64))
        with safe_open(path, framework='numpy') as file:
            feature_slice = file.get_slice('feature')
            rows, channels = feature_slice.get_shape()
            if indices.size == 0:
                return np.empty((0, channels), dtype=np.float32)
            if indices[0] < 0 or indices[-1] >= rows:
                raise IndexError(f'Feature row index exceeds shape in {path}')

            if indices.size >= max(1024, int(np.ceil(rows * 0.05))):
                feature = file.get_tensor('feature')
                return np.asarray(feature[indices], dtype=np.float32)

            output = np.empty((indices.size, channels), dtype=np.float32)
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
        return output

    def _sample_features(self) -> np.ndarray:
        rng = np.random.default_rng(self.random_seed)
        file_count = len(self.file_info)
        sample_file_count = (
            file_count
            if self.max_sample_files is None
            else min(file_count, self.max_sample_files)
        )
        selected_file_indices = np.sort(
            rng.choice(file_count, size=sample_file_count, replace=False)
        )
        selected_info = [self.file_info[index] for index in selected_file_indices]
        selected_rows = np.asarray(
            [info['rows'] for info in selected_info], dtype=np.int64
        )
        available_rows = int(selected_rows.sum())
        sample_count = min(
            available_rows,
            self.max_samples,
            max(self.target_dim, int(round(available_rows * self.sample_ratio))),
        )
        if sample_count < self.target_dim:
            raise ValueError(
                f'Only {sample_count} PCA samples are available for '
                f'target_dim={self.target_dim}'
            )

        global_indices = np.sort(
            rng.choice(available_rows, size=sample_count, replace=False)
        )
        offsets = np.concatenate(([0], np.cumsum(selected_rows)))
        sample = np.empty((sample_count, self.input_dim), dtype=np.float32)
        write_start = 0
        for file_index, info in enumerate(tqdm(
            selected_info, desc='Sample PCA features', unit='file'
        )):
            begin, end = offsets[file_index], offsets[file_index + 1]
            mask = (global_indices >= begin) & (global_indices < end)
            local_indices = global_indices[mask] - begin
            if local_indices.size == 0:
                continue
            chunk = self._load_feature_rows(info['path'], local_indices)
            if self.normalize_input:
                chunk = self._normalize(chunk)
            write_end = write_start + chunk.shape[0]
            sample[write_start:write_end] = chunk
            write_start = write_end

        if write_start != sample_count:
            raise RuntimeError(
                f'PCA sample count mismatch: {write_start} != {sample_count}'
            )
        self.logger.info(
            f'PCA sample: {sample_count:,} patches from '
            f'{sample_file_count}/{file_count} files'
        )
        return sample

    def _fit_pca(self, sample: np.ndarray):
        self.logger.info(
            f'Fitting shared PCA: {sample.shape[0]:,} x {sample.shape[1]} '
            f'-> {self.target_dim}'
        )
        self.pca = PCA(
            n_components=self.target_dim,
            whiten=self.whiten,
            svd_solver=self.svd_solver,
            iterated_power=self.iterated_power,
            n_oversamples=self.n_oversamples,
            power_iteration_normalizer=self.power_iteration_normalizer,
            random_state=self.random_seed,
        )
        self.pca.fit(sample)
        self.explained_variance_ratio_sum = float(
            self.pca.explained_variance_ratio_.sum()
        )
        probe_count = min(sample.shape[0], 10000)
        probe_index = np.linspace(
            0, sample.shape[0] - 1, probe_count, dtype=np.int64
        )
        probe = sample[probe_index]
        reconstructed = self.pca.inverse_transform(self.pca.transform(probe))
        numerator = np.sum(probe * reconstructed, axis=1)
        denominator = np.linalg.norm(probe, axis=1) * np.linalg.norm(
            reconstructed, axis=1
        )
        self.reconstruction_cosine_mean = float(
            np.mean(numerator / np.maximum(denominator, 1e-12))
        )
        self.logger.info(
            'PCA quality: explained_variance_ratio_sum='
            f'{self.explained_variance_ratio_sum:.6f}, '
            'reconstruction_cosine_mean='
            f'{self.reconstruction_cosine_mean:.6f} '
            f'({probe_count:,} sampled fit rows)'
        )
        if self.save_pca_model:
            self._save_pca_model(sample.shape[0])

    def _validate_output_collisions(self):
        """避免同一输出目录中混入不同共享 PCA 生成的特征。"""
        if self.in_place or self.overwrite:
            return

        existing = []
        if self.save_pca_result:
            existing.extend(
                self.output_path / info['path'].name
                for info in self.file_info
                if (self.output_path / info['path'].name).exists()
            )
        model_path = self.output_path / self.pca_model_name
        if self.save_pca_model and model_path.exists():
            existing.append(model_path)
        if existing:
            preview = ', '.join(path.name for path in existing[:3])
            if len(existing) > 3:
                preview += f', ... ({len(existing)} files)'
            raise FileExistsError(
                f'Output already exists ({preview}); set overwrite=True or '
                'use an empty output_path to keep one consistent shared PCA'
            )

    def _save_pca_model(self, sample_count: int) -> Path:
        output = self.output_path / self.pca_model_name
        tensors = {
            'components': np.ascontiguousarray(
                self.pca.components_, dtype=np.float32
            ),
            'mean': np.ascontiguousarray(self.pca.mean_, dtype=np.float32),
            'explained_variance': np.ascontiguousarray(
                self.pca.explained_variance_, dtype=np.float32
            ),
            'explained_variance_ratio': np.ascontiguousarray(
                self.pca.explained_variance_ratio_, dtype=np.float32
            ),
            'singular_values': np.ascontiguousarray(
                self.pca.singular_values_, dtype=np.float32
            ),
        }
        metadata = {
            'schema': 'pointspace_dino_shared_pca_v1',
            'input_dim': str(self.input_dim),
            'target_dim': str(self.target_dim),
            'sample_count': str(sample_count),
            'normalize_input': str(self.normalize_input).lower(),
            'whiten': str(self.whiten).lower(),
            'svd_solver': self.svd_solver,
            'iterated_power': str(self.iterated_power),
            'n_oversamples': str(self.n_oversamples),
            'power_iteration_normalizer': self.power_iteration_normalizer,
            'random_seed': str(self.random_seed),
            'explained_variance_ratio_sum': str(
                self.explained_variance_ratio_sum
            ),
            'reconstruction_cosine_mean': str(
                self.reconstruction_cosine_mean
            ),
        }
        self._atomic_save(tensors, output, metadata)
        self.logger.info(f'Saved shared PCA model: {output}')
        return output

    @staticmethod
    def _parse_dino_metadata(raw_metadata: Dict[str, str]) -> Dict:
        encoded = raw_metadata.get('metadata', '{}')
        try:
            metadata = json.loads(encoded)
        except json.JSONDecodeError as error:
            raise ValueError('Invalid DINO metadata JSON') from error
        if not isinstance(metadata, dict):
            raise ValueError('DINO metadata must decode to a dictionary')
        return metadata

    @staticmethod
    def _atomic_save(
        tensors: Dict[str, np.ndarray],
        output: Path,
        metadata: Dict[str, str],
    ):
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f'.{output.name}.{uuid4().hex}.tmp')
        try:
            save_safetensors(tensors, temporary, metadata=metadata)
            os.replace(temporary, output)
        finally:
            if temporary.exists():
                temporary.unlink()

    def _transform_file(self, info: Dict) -> Path:
        source = info['path']
        target = self.output_path / source.name
        output_numpy_dtype = (
            np.float16 if self.output_dtype == 'float16' else np.float32
        )
        transformed = np.empty(
            (info['rows'], self.target_dim), dtype=output_numpy_dtype
        )
        other_tensors = {}
        with safe_open(source, framework='numpy') as file:
            feature_slice = file.get_slice('feature')
            for start in range(0, info['rows'], self.transform_batch_size):
                end = min(start + self.transform_batch_size, info['rows'])
                chunk = np.asarray(
                    feature_slice[start:end], dtype=np.float32
                )
                if self.normalize_input:
                    chunk = self._normalize(chunk)
                projected = self.pca.transform(chunk)
                transformed[start:end] = projected.astype(
                    output_numpy_dtype, copy=False
                )
            for key in file.keys():
                if key != 'feature':
                    # Safetensors numpy tensor may keep a memory map alive. A
                    # real copy is required before replacing the source file
                    # in-place on Windows.
                    other_tensors[key] = np.array(
                        file.get_tensor(key), copy=True, order='C'
                    )
            del feature_slice

        dino_metadata = self._parse_dino_metadata(info['metadata'])
        dino_metadata.setdefault('original_feature_dim', self.input_dim)
        dino_metadata.update({
            'feature_shape': [info['rows'], self.target_dim],
            'feature_dtype': self.output_dtype,
            'pca_schema': 'pointspace_dino_shared_pca_v1',
            'pca_target_dim': self.target_dim,
            'pca_normalize_input': self.normalize_input,
            'pca_whiten': self.whiten,
            'pca_model': self.pca_model_name if self.save_pca_model else None,
        })
        output_metadata = dict(info['metadata'])
        output_metadata['metadata'] = json.dumps(
            dino_metadata, separators=(',', ':'), ensure_ascii=False
        )
        output_metadata['pca_schema'] = 'pointspace_dino_shared_pca_v1'
        tensors = {'feature': np.ascontiguousarray(transformed)}
        tensors.update(other_tensors)
        self._atomic_save(tensors, target, output_metadata)
        return target

    def process(self) -> List[Path]:
        self._validate_output_collisions()
        sample = self._sample_features()
        self._fit_pca(sample)

        outputs = []
        if self.save_pca_result:
            for info in tqdm(
                self.file_info, desc='Transform DINO features', unit='file'
            ):
                outputs.append(self._transform_file(info))
            self.logger.info(
                f'PCA reduction completed: {len(outputs)} files -> '
                f'{self.output_path}'
            )
        else:
            self.logger.info(
                'PCA fitting completed; per-file PCA result saving is disabled'
            )
        return outputs


def reduce_dino_features_pca(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    target_dim: int = 128,
    sample_ratio: float = 0.05,
    max_samples: int = 100000,
    max_sample_files: Optional[int] = 64,
    normalize_input: bool = True,
    whiten: bool = False,
    svd_solver: str = 'randomized',
    iterated_power: Union[str, int] = 'auto',
    n_oversamples: int = 10,
    power_iteration_normalizer: str = 'auto',
    transform_batch_size: int = 65536,
    output_dtype: str = 'float16',
    overwrite: bool = False,
    random_seed: int = 42,
    save_pca_result: bool = True,
    save_pca_model: bool = True,
    pca_model_name: str = PCA_MODEL_NAME,
) -> List[Path]:
    """创建共享 PCA reducer 并处理输入目录中的全部 DINO 特征。"""
    reducer = DINOFeaturePCAReducer(
        input_path=input_path,
        output_path=output_path,
        target_dim=target_dim,
        sample_ratio=sample_ratio,
        max_samples=max_samples,
        max_sample_files=max_sample_files,
        normalize_input=normalize_input,
        whiten=whiten,
        svd_solver=svd_solver,
        iterated_power=iterated_power,
        n_oversamples=n_oversamples,
        power_iteration_normalizer=power_iteration_normalizer,
        transform_batch_size=transform_batch_size,
        output_dtype=output_dtype,
        overwrite=overwrite,
        random_seed=random_seed,
        save_pca_result=save_pca_result,
        save_pca_model=save_pca_model,
        pca_model_name=pca_model_name,
    )
    return reducer.process()


if __name__ == '__main__':
    reduce_dino_features_pca(
        input_path=r'G:\data\swiss\joint_tiles_100\dino_feature',
        output_path=r'G:\data\swiss\joint_tiles_100\dino_feature_pca128',

        target_dim=128,
        # Fit on a global sample drawn from every training tile. One million
        # 1024D FP32 rows need roughly 3.8 GiB before sklearn work buffers.
        sample_ratio=1.0,
        max_samples=1000000,
        max_sample_files=None,

        normalize_input=True,
        whiten=False,
        svd_solver='randomized',
        iterated_power=7,
        n_oversamples=64,
        power_iteration_normalizer='QR',

        transform_batch_size=65536,
        output_dtype='float32',
        overwrite=True,
        random_seed=42,
        save_pca_result=False,
        save_pca_model=True,
    )
