"""
DINOv3 Image Feature Extractor

对单幅影像或影像目录批量提取 DINOv3 patch 特征，支持：
- 原始分辨率下的重叠 tile 推理
- 多 tile batch 推理与 AMP 混合精度
- Hann 窗口融合重叠区域
- Safetensors 安全快速保存特征和关键元数据
- 为同名 correspondence Safetensors 添加一维 patch_index
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import timm
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from safetensors import safe_open
from safetensors.numpy import load_file as load_safetensors
from safetensors.numpy import save_file as save_safetensors
from sklearn.decomposition import PCA
from torchvision.transforms import functional as TF
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


DEFAULT_CHECKPOINT_DIR = (
    Path(__file__).resolve().parent
    / 'checkpoint'
    / 'vit_large_patch16_dinov3.sat493m'
)


def load_dino_feature(path: Union[str, Path]) -> Tuple[np.ndarray, Dict]:
    """安全读取本工具生成的 Safetensors 特征和关键元数据。"""
    path = Path(path)
    if path.suffix.lower() != '.safetensors':
        raise ValueError(f'Expected a .safetensors file: {path}')
    tensors = load_safetensors(path)
    if 'feature' not in tensors:
        raise ValueError(f"Safetensors file has no 'feature' tensor: {path}")
    with safe_open(path, framework='numpy') as file:
        raw_metadata = file.metadata().get('metadata', '{}')
    return tensors['feature'], json.loads(raw_metadata)


def load_dino_metadata(path: Union[str, Path]) -> Dict:
    """只读取 Safetensors 文件头中的关键元数据，不加载特征张量。"""
    path = Path(path)
    if path.suffix.lower() != '.safetensors':
        raise ValueError(f'Expected a .safetensors file: {path}')
    with safe_open(path, framework='numpy') as file:
        raw_metadata = file.metadata().get('metadata')
        if raw_metadata is None:
            raise ValueError(f'Safetensors file has no metadata: {path}')
    metadata = json.loads(raw_metadata)
    required = (
        'original_size', 'patch_size',
        'feature_shape', 'feature_grid_size',
    )
    missing = [name for name in required if name not in metadata]
    if missing:
        raise ValueError(
            f'{path} is missing metadata: {", ".join(missing)}'
        )
    return metadata


def _find_input_files(
    input_path: Union[str, Path],
    suffix: str,
    description: str,
) -> List[Path]:
    """查找单文件或文件夹内的指定格式文件。"""
    input_path = Path(input_path)
    if input_path.is_file():
        if input_path.suffix.lower() != suffix:
            raise ValueError(f'Expected a {suffix} {description}: {input_path}')
        return [input_path]
    if input_path.is_dir():
        files = sorted(
            path for path in input_path.iterdir()
            if path.is_file() and path.suffix.lower() == suffix
        )
        if not files:
            raise ValueError(f'No {suffix} {description} files found: {input_path}')
        return files
    raise ValueError(f'Invalid path: {input_path}')


def _update_correspondence_safetensors(
    source_path: Path,
    output_path: Path,
    metadata: Dict,
) -> str:
    """根据 DINO 元数据生成线性 patch 索引并原子化保存。"""
    tensors = load_safetensors(source_path)
    with safe_open(source_path, framework='numpy') as source_file:
        source_metadata = dict(source_file.metadata() or {})
    required = ('pixel_coord', 'valid')
    missing = [name for name in required if name not in tensors]
    if missing:
        raise ValueError(
            f'{source_path} is missing tensors: {", ".join(missing)}'
        )
    pixel_coord = np.asarray(tensors['pixel_coord'])
    valid = np.asarray(tensors['valid'], dtype=bool).reshape(-1)
    if pixel_coord.ndim != 2 or pixel_coord.shape[1] != 2:
        raise ValueError(f'pixel_coord must have shape [N, 2]: {source_path}')
    if len(pixel_coord) != len(valid):
        raise ValueError(f'pixel_coord and valid length mismatch: {source_path}')
    pixel_rows = pixel_coord[:, 0]
    pixel_columns = pixel_coord[:, 1]

    original_size = tuple(metadata['original_size'])
    patch_size = int(metadata['patch_size'])
    feature_shape = tuple(metadata['feature_shape'])
    feature_grid_size = tuple(metadata['feature_grid_size'])
    if len(original_size) != 2 or len(feature_shape) != 2:
        raise ValueError('Invalid DINO feature metadata dimensions')
    if len(feature_grid_size) != 2:
        raise ValueError('feature_grid_size must contain [width, height]')
    grid_width, grid_height = map(int, feature_grid_size)
    if int(feature_shape[0]) != grid_height * grid_width:
        raise ValueError('feature_shape and feature_grid_size are inconsistent')

    patch_columns = np.floor(
        pixel_columns.astype(np.float64) / patch_size
    ).astype(np.int64)
    patch_rows = np.floor(
        pixel_rows.astype(np.float64) / patch_size
    ).astype(np.int64)
    patch_inside = (
        (pixel_columns >= 0) & (pixel_columns < original_size[0])
        & (pixel_rows >= 0) & (pixel_rows < original_size[1])
        & (patch_columns >= 0) & (patch_columns < grid_width)
        & (patch_rows >= 0) & (patch_rows < grid_height)
    )
    valid &= patch_inside
    patch_index = patch_rows * grid_width + patch_columns
    patch_index = patch_index.astype(np.int32, copy=False)
    patch_index[~valid] = -1

    # 保留 tile 阶段写入的 observability 及未来扩展 tensor。DINO 更新只
    # 负责修正 patch_index/valid，不应把其他逐点信息静默删除。
    output_tensors = {
        name: np.ascontiguousarray(value)
        for name, value in tensors.items()
    }
    output_tensors.update(
        pixel_coord=np.ascontiguousarray(pixel_coord, dtype=np.int32),
        patch_index=np.ascontiguousarray(patch_index),
        valid=np.ascontiguousarray(valid),
    )
    if 'observability' in output_tensors:
        observability = np.asarray(output_tensors['observability']).reshape(-1)
        if len(observability) != len(valid):
            raise ValueError(
                f'observability and valid length mismatch: {source_path}'
            )
        if not np.all(np.isfinite(observability)):
            raise ValueError(f'observability contains non-finite values: {source_path}')
        observability = np.clip(observability, 0.0, 1.0)
        # image_valid 还可能因为正射表面过滤而为 False；这些点的低 q 对
        # 可观测性分层分析仍有意义。这里只清零确实落在 DINO 网格外的点。
        observability[~patch_inside] = 0.0
        output_tensors['observability'] = np.ascontiguousarray(
            observability, dtype=np.float16
        )

    output_metadata = dict(source_metadata)
    output_metadata.update({
        'schema': 'pointspace_image_mapping_v3',
        'coordinate_order': 'row_col',
        'feature_grid_size': json.dumps([grid_width, grid_height]),
        'patch_size': str(patch_size),
        'original_size': json.dumps(list(original_size)),
    })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f'.{output_path.name}.{os.getpid()}.tmp')
    try:
        save_safetensors(
            output_tensors,
            temporary_path,
            metadata=output_metadata,
        )
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return 'updated'


def update_dino_correspondence(
    correspondence_path: Union[str, Path],
    feature_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    raise_on_error: bool = False,
) -> List[Dict]:
    """
    使用已有 DINO 特征更新同名 correspondence Safetensors。

    该函数不会加载 DINO 模型或特征张量，也不会重新提取特征。
    output_dir=None 时原子化原地更新；指定目录时保留源文件并将更新版本
    写入新目录。raise_on_error=False 时单个坏文件记录为 failed
    并继续处理，其余文件不受影响。
    """
    correspondence_files = _find_input_files(
        correspondence_path, '.safetensors', 'correspondence'
    )
    feature_files = _find_input_files(
        feature_path, '.safetensors', 'DINO feature'
    )
    feature_by_stem = {path.stem: path for path in feature_files}
    if len(feature_by_stem) != len(feature_files):
        raise ValueError('Duplicate DINO feature stems are not supported')
    output_dir = Path(output_dir) if output_dir is not None else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for correspondence_file in tqdm(
        correspondence_files, desc='Update correspondence'
    ):
        feature_file = feature_by_stem.get(correspondence_file.stem)
        if feature_file is None:
            results.append({
                'file': correspondence_file.name,
                'status': 'skipped',
                'reason': 'matching Safetensors not found',
            })
            continue
        target = (
            output_dir / correspondence_file.name
            if output_dir is not None
            else correspondence_file
        )
        try:
            metadata = load_dino_metadata(feature_file)
            status = _update_correspondence_safetensors(
                correspondence_file, target, metadata
            )
            results.append({'file': correspondence_file.name, 'status': status})
        except Exception as error:
            if raise_on_error:
                raise
            results.append({
                'file': correspondence_file.name,
                'status': 'failed',
                'reason': str(error),
            })
    return results


def tile_positions(
    length: int,
    tile_size: int,
    overlap: int,
    patch_size: int,
) -> List[int]:
    """生成 patch 对齐且尽量均匀的 tile 起点。"""
    if length <= tile_size:
        return [0]
    stride = tile_size - overlap
    tile_count = int(np.ceil((length - tile_size) / stride)) + 1
    raw_positions = np.linspace(0, length - tile_size, tile_count)
    starts = [
        int(round(position / patch_size) * patch_size)
        for position in raw_positions
    ]
    starts[0], starts[-1] = 0, length - tile_size
    return list(dict.fromkeys(starts))


def blend_window(height: int, width: int) -> torch.Tensor:
    """生成 patch 空间的平滑正权重窗口。"""
    window_y = torch.hann_window(height, periodic=False).clamp_min(0.05)
    window_x = torch.hann_window(width, periodic=False).clamp_min(0.05)
    return window_y[:, None] * window_x[None, :]


class DINOFeatureExtractor:
    """
    DINOv3 批量影像特征提取器。

    Args:
        image_path: 输入影像文件或文件夹
        output_dir: 特征输出文件夹
        correspondence_path: 可选的同名 correspondence Safetensors 文件或文件夹
        correspondence_output_dir: 更新后映射输出文件夹；None 表示原子化原地更新
        checkpoint_dir: timm DINOv3 checkpoint 文件夹
        tile_size: 推理 tile 边长，必须能被模型 patch_size 整除
        tile_overlap: tile 重叠像素数，必须能被 patch_size 整除
        batch_size: 每次模型前向的 tile 数量
        output_dtype: 特征保存类型，'float16' 或 'float32'
        device: 推理设备；None 表示优先 CUDA
        amp: 是否启用 AMP；None 表示 CUDA 上自动启用
        amp_dtype: AMP 类型，'float16' 或 'bfloat16'
        save_pca: 是否保存每幅影像的三通道 PCA 特征可视化
        pca_output_dir: PCA 图像输出目录；None 表示 output_dir/pca
        pca_format: PCA 图像格式，默认为 'jpg'
        pca_max_samples: PCA 拟合使用的最大 patch 数，全部 patch 仍会被转换
        prefetch_batches: CPU 预处理队列最多缓存的 batch 数
    """

    IMAGE_SUFFIXES = ('.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp', '.webp')

    def __init__(
        self,
        image_path: Union[str, Path],
        output_dir: Union[str, Path],
        correspondence_path: Optional[Union[str, Path]] = None,
        correspondence_output_dir: Optional[Union[str, Path]] = None,
        checkpoint_dir: Union[str, Path] = DEFAULT_CHECKPOINT_DIR,
        tile_size: int = 768,
        tile_overlap: int = 128,
        batch_size: int = 4,
        output_dtype: str = 'float16',
        device: Optional[str] = None,
        amp: Optional[bool] = None,
        amp_dtype: str = 'float16',
        save_pca: bool = True,
        pca_output_dir: Optional[Union[str, Path]] = None,
        pca_format: str = 'jpg',
        pca_max_samples: int = 100000,
        prefetch_batches: int = 2,
    ):
        self.image_path = Path(image_path)
        self.output_dir = Path(output_dir)
        self.correspondence_path = (
            Path(correspondence_path) if correspondence_path is not None else None
        )
        self.correspondence_output_dir = (
            Path(correspondence_output_dir)
            if correspondence_output_dir is not None
            else None
        )
        self.checkpoint_dir = Path(checkpoint_dir)
        self.tile_size = int(tile_size)
        self.tile_overlap = int(tile_overlap)
        self.batch_size = int(batch_size)
        self.output_dtype = output_dtype.lower()
        self.device = torch.device(
            device or ('cuda' if torch.cuda.is_available() else 'cpu')
        )
        self.amp = self.device.type == 'cuda' if amp is None else bool(amp)
        self.amp_dtype_name = amp_dtype.lower()
        self.save_pca = bool(save_pca)
        self.pca_output_dir = (
            Path(pca_output_dir)
            if pca_output_dir is not None
            else self.output_dir / 'pca'
        )
        self.pca_format = pca_format.lower().lstrip('.')
        self.pca_max_samples = int(pca_max_samples)
        self.prefetch_batches = int(prefetch_batches)
        self.logger = get_root_logger()

        self.image_files = self._find_files(
            self.image_path, self.IMAGE_SUFFIXES, 'image'
        )
        self.correspondence_files = self._find_correspondence_files()
        correspondence_stems = [path.stem for path in self.correspondence_files]
        if len(correspondence_stems) != len(set(correspondence_stems)):
            raise ValueError('Duplicate correspondence file stems are not supported')
        self.correspondence_by_stem = {
            path.stem: path for path in self.correspondence_files
        }

        self.config = self._load_config()
        self.patch_size = self._get_patch_size()
        self.channels = int(self.config['num_features'])
        self.mean = self.config['pretrained_cfg']['mean']
        self.std = self.config['pretrained_cfg']['std']
        self.architecture = self.config['architecture']
        self._validate_parameters()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.save_pca:
            self.pca_output_dir.mkdir(parents=True, exist_ok=True)
        if self.correspondence_output_dir is not None:
            self.correspondence_output_dir.mkdir(parents=True, exist_ok=True)

        self.model = None

    @staticmethod
    def _find_files(
        input_path: Path,
        suffixes: Tuple[str, ...],
        description: str,
    ) -> List[Path]:
        """查找单个文件或文件夹内指定格式的文件。"""
        if input_path.is_file():
            if input_path.suffix.lower() not in suffixes:
                raise ValueError(
                    f'Unsupported {description} format: {input_path.suffix}'
                )
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

    def _find_correspondence_files(self) -> List[Path]:
        """查找可选的 correspondence Safetensors。"""
        if self.correspondence_path is None:
            return []
        return self._find_files(
            self.correspondence_path, ('.safetensors',), 'correspondence'
        )

    def _load_config(self) -> Dict:
        """读取 checkpoint 配置。"""
        config_path = self.checkpoint_dir / 'config.json'
        checkpoint_path = self.checkpoint_dir / 'model.safetensors'
        if not config_path.is_file():
            raise FileNotFoundError(f'Config not found: {config_path}')
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')
        with config_path.open('r', encoding='utf-8') as file:
            return json.load(file)

    def _get_patch_size(self) -> int:
        """从模型配置推断 patch size，无法推断时使用 16。"""
        architecture = str(self.config['architecture'])
        for token in architecture.split('_'):
            if token.startswith('patch') and token[5:].isdigit():
                return int(token[5:])
        return 16

    def _validate_parameters(self):
        """验证输入参数和运行环境。"""
        if self.tile_size <= 0 or self.tile_size % self.patch_size:
            raise ValueError(
                f'tile_size must be positive and divisible by {self.patch_size}'
            )
        if self.tile_overlap < 0 or self.tile_overlap >= self.tile_size:
            raise ValueError('tile_overlap must satisfy 0 <= overlap < tile_size')
        if self.tile_overlap % self.patch_size:
            raise ValueError(
                f'tile_overlap must be divisible by {self.patch_size}'
            )
        if self.batch_size < 1:
            raise ValueError('batch_size must be a positive integer')
        if self.output_dtype not in ('float16', 'float32'):
            raise ValueError("output_dtype must be 'float16' or 'float32'")
        if self.amp_dtype_name not in ('float16', 'bfloat16'):
            raise ValueError("amp_dtype must be 'float16' or 'bfloat16'")
        if self.pca_format not in ('jpg', 'jpeg', 'png'):
            raise ValueError("pca_format must be 'jpg', 'jpeg', or 'png'")
        if self.pca_max_samples < 3:
            raise ValueError('pca_max_samples must be at least 3')
        if self.prefetch_batches < 1:
            raise ValueError('prefetch_batches must be a positive integer')
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA was requested but is not available')
        if self.amp and self.device.type == 'cpu' and self.amp_dtype_name == 'float16':
            raise ValueError('CPU AMP does not support float16; use bfloat16 or amp=False')

    def _load_model(self):
        """延迟加载 DINOv3 模型。"""
        if self.model is not None:
            return
        checkpoint_path = self.checkpoint_dir / 'model.safetensors'
        self.logger.info(
            f'Loading {self.architecture} from {checkpoint_path}'
        )
        self.model = timm.create_model(
            self.architecture,
            pretrained=False,
            checkpoint_path=str(checkpoint_path),
            dynamic_img_size=True,
        ).eval().to(self.device)

    @staticmethod
    def _load_image_rgb(path: Path) -> Image.Image:
        """读取 Pillow 支持的影像并转换为 8-bit RGB。"""
        try:
            with Image.open(path) as source:
                source.seek(0)
                image = ImageOps.exif_transpose(source).copy()
        except (UnidentifiedImageError, OSError) as error:
            raise ValueError(f'Unsupported or damaged image file: {path}') from error

        if image.mode in {'I', 'I;16', 'I;16B', 'I;16L', 'F'}:
            values = np.asarray(image, dtype=np.float32)
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                raise ValueError(f'Image contains no finite values: {path}')
            low, high = np.percentile(finite, (0.1, 99.9))
            values = np.nan_to_num(values, nan=low, posinf=high, neginf=low)
            values = np.clip((values - low) / max(high - low, 1e-8), 0, 1)
            image = Image.fromarray((values * 255).astype(np.uint8), mode='L')
        return image.convert('RGB')

    def _pad_to_patch(
        self,
        image: Image.Image,
    ) -> Image.Image:
        """仅在右侧和底部复制填充到完整 patch，不改变输入影像尺寸。"""
        padded_size = (
            int(np.ceil(image.width / self.patch_size) * self.patch_size),
            int(np.ceil(image.height / self.patch_size) * self.patch_size),
        )
        if padded_size == image.size:
            return image
        values = np.asarray(image)
        padded = np.pad(
            values,
            (
                (0, padded_size[1] - image.height),
                (0, padded_size[0] - image.width),
                (0, 0),
            ),
            mode='edge',
        )
        return Image.fromarray(padded, mode='RGB')

    def process_all_images(self) -> List[Path]:
        """使用 CPU-GPU 流水线处理全部影像并返回特征文件路径。"""
        self._load_model()
        start_time = time.time()
        self.logger.info('DINOv3 Feature Extractor started')
        self.logger.info(f'  Input: {self.image_path}')
        self.logger.info(f'  Output: {self.output_dir}')
        self.logger.info(f'  Images: {len(self.image_files)}')
        self.logger.info(f'  Tile/overlap: {self.tile_size}/{self.tile_overlap}')
        self.logger.info(
            f'  Batch/device/AMP: {self.batch_size}/{self.device}/'
            f'{self.amp} ({self.amp_dtype_name})'
        )

        task_queue = queue.Queue(maxsize=self.batch_size * self.prefetch_batches)
        writer_queue = queue.Queue(maxsize=self.prefetch_batches)
        producer_done = threading.Event()
        stop_producer = threading.Event()
        producer_error = []
        writer_error = []
        states = {}
        states_lock = threading.Lock()
        output_lock = threading.Lock()
        total_tiles = self._count_total_tiles()

        def producer():
            try:
                for image_id, image_file in enumerate(self.image_files):
                    state, inference_image, positions = self._prepare_image(
                        image_file
                    )
                    with states_lock:
                        states[image_id] = state
                    for x, y in positions:
                        tile = inference_image.crop((
                            x,
                            y,
                            x + state['tile_width'],
                            y + state['tile_height'],
                        ))
                        tensor = TF.normalize(
                            TF.to_tensor(tile), self.mean, self.std
                        )
                        if self.device.type == 'cuda':
                            tensor = tensor.pin_memory()
                        task = {
                            'image_id': image_id,
                            'x': x,
                            'y': y,
                            'tensor': tensor,
                        }
                        while not stop_producer.is_set():
                            try:
                                task_queue.put(task, timeout=0.1)
                                break
                            except queue.Full:
                                continue
                        if stop_producer.is_set():
                            return
                    del inference_image
            except Exception as error:
                producer_error.append(error)
            finally:
                producer_done.set()

        output_by_id = {}

        def writer():
            while True:
                item = writer_queue.get()
                try:
                    if item is None:
                        return
                    image_id, state = item
                    if not writer_error:
                        output_path = self._finalize_image(state)
                        with output_lock:
                            output_by_id[image_id] = output_path
                except Exception as error:
                    writer_error.append(error)
                finally:
                    writer_queue.task_done()

        producer_thread = threading.Thread(
            target=producer,
            name='dino-cpu-producer',
            daemon=True,
        )
        producer_thread.start()
        writer_thread = threading.Thread(
            target=writer,
            name='dino-cpu-writer',
            daemon=True,
        )
        writer_thread.start()

        pending_by_shape = {}
        progress = tqdm(total=total_tiles, desc='DINO tile inference')
        try:
            while not producer_done.is_set() or not task_queue.empty():
                try:
                    task = task_queue.get(timeout=0.1)
                except queue.Empty:
                    if producer_error:
                        raise producer_error[0]
                    continue
                shape_key = tuple(task['tensor'].shape)
                bucket = pending_by_shape.setdefault(shape_key, [])
                bucket.append(task)
                if len(bucket) >= self.batch_size:
                    self._run_task_batch(
                        bucket[:self.batch_size], states, states_lock,
                        writer_queue, progress
                    )
                    del bucket[:self.batch_size]
                elif sum(len(items) for items in pending_by_shape.values()) >= (
                    self.batch_size * self.prefetch_batches
                ):
                    # 输入图像尺寸差异很大时，不能让各形状桶无限等待。
                    flush_bucket = max(
                        pending_by_shape.values(), key=len
                    )
                    self._run_task_batch(
                        flush_bucket, states, states_lock,
                        writer_queue, progress
                    )
                    flush_bucket.clear()

            if producer_error:
                raise producer_error[0]
            for bucket in pending_by_shape.values():
                if bucket:
                    self._run_task_batch(
                        bucket, states, states_lock, writer_queue, progress
                    )
        finally:
            progress.close()
            stop_producer.set()
            producer_thread.join()
            writer_queue.put(None)
            writer_queue.join()
            writer_thread.join()

        if writer_error:
            raise writer_error[0]

        output_paths = [output_by_id[index] for index in range(len(self.image_files))]

        self.logger.info(
            f'Processing completed: {len(output_paths)} images in '
            f'{time.time() - start_time:.2f}s'
        )
        return output_paths

    def _image_layout(self, image_file: Path) -> Dict:
        """读取影像尺寸并计算其推理 tile 布局。"""
        with Image.open(image_file) as image:
            original_size = image.size
        padded_size = (
            int(np.ceil(original_size[0] / self.patch_size) * self.patch_size),
            int(np.ceil(original_size[1] / self.patch_size) * self.patch_size),
        )
        tile_width = min(self.tile_size, padded_size[0])
        tile_height = min(self.tile_size, padded_size[1])
        x_starts = tile_positions(
            padded_size[0], tile_width, self.tile_overlap, self.patch_size
        )
        y_starts = tile_positions(
            padded_size[1], tile_height, self.tile_overlap, self.patch_size
        )
        return {
            'original_size': original_size,
            'padded_size': padded_size,
            'tile_width': tile_width,
            'tile_height': tile_height,
            'positions': [(x, y) for y in y_starts for x in x_starts],
        }

    def _count_total_tiles(self) -> int:
        """快速读取影像头并统计全局进度条的 tile 总数。"""
        return sum(
            len(self._image_layout(image_file)['positions'])
            for image_file in self.image_files
        )

    def _prepare_image(self, image_file: Path):
        """在 CPU 线程中读取影像并创建该影像的累积状态。"""
        image = self._load_image_rgb(image_file)
        original_size = image.size
        inference_image = self._pad_to_patch(image)
        padded_size = inference_image.size

        tile_width = min(self.tile_size, padded_size[0])
        tile_height = min(self.tile_size, padded_size[1])
        x_starts = tile_positions(
            padded_size[0], tile_width, self.tile_overlap, self.patch_size
        )
        y_starts = tile_positions(
            padded_size[1], tile_height, self.tile_overlap, self.patch_size
        )
        positions = [(x, y) for y in y_starts for x in x_starts]

        grid_width = padded_size[0] // self.patch_size
        grid_height = padded_size[1] // self.patch_size
        tile_grid_width = tile_width // self.patch_size
        tile_grid_height = tile_height // self.patch_size
        feature_sum = torch.zeros(
            self.channels, grid_height, grid_width, dtype=torch.float32
        )
        weight_sum = torch.zeros(1, grid_height, grid_width, dtype=torch.float32)
        weights = blend_window(tile_grid_height, tile_grid_width).unsqueeze(0)
        state = {
            'image_file': image_file,
            'original_size': original_size,
            'padded_size': padded_size,
            'tile_width': tile_width,
            'tile_height': tile_height,
            'tile_grid_width': tile_grid_width,
            'tile_grid_height': tile_grid_height,
            'tile_count': len(positions),
            'completed_tiles': 0,
            'feature_sum': feature_sum,
            'weight_sum': weight_sum,
            'weights': weights,
        }
        return state, inference_image, positions

    def _run_task_batch(
        self,
        tasks: List[Dict],
        states: Dict,
        states_lock: threading.Lock,
        writer_queue: queue.Queue,
        progress: tqdm,
    ):
        """执行可能来自不同影像、但 tile 尺寸相同的全局 batch。"""
        batch = torch.stack([task['tensor'] for task in tasks]).to(
            self.device, non_blocking=True
        )
        tile_height, tile_width = batch.shape[-2:]
        grid_height = tile_height // self.patch_size
        grid_width = tile_width // self.patch_size
        batch_features = self._forward_batch(batch, grid_height, grid_width)
        completed_image_ids = []
        with states_lock:
            for batch_index, task in enumerate(tasks):
                state = states[task['image_id']]
                patch_x = task['x'] // self.patch_size
                patch_y = task['y'] // self.patch_size
                state['feature_sum'][
                    :,
                    patch_y:patch_y + grid_height,
                    patch_x:patch_x + grid_width,
                ] += batch_features[batch_index] * state['weights']
                state['weight_sum'][
                    :,
                    patch_y:patch_y + grid_height,
                    patch_x:patch_x + grid_width,
                ] += state['weights']
                state['completed_tiles'] += 1
                if state['completed_tiles'] == state['tile_count']:
                    completed_image_ids.append(task['image_id'])
        progress.update(len(tasks))

        for image_id in completed_image_ids:
            with states_lock:
                state = states.pop(image_id)
            writer_queue.put((image_id, state))

    def _finalize_image(self, state: Dict) -> Path:
        """融合、保存一幅已完成影像并更新 correspondence。"""
        if torch.any(state['weight_sum'] == 0):
            raise RuntimeError(
                f'Tile layout left uncovered patches: {state["image_file"]}'
            )
        feature = (
            state['feature_sum'] / state['weight_sum']
        ).numpy().astype(
            np.float16 if self.output_dtype == 'float16' else np.float32,
            copy=False,
        )
        image_file = state['image_file']
        original_size = state['original_size']
        padded_size = state['padded_size']
        channels, grid_height, grid_width = feature.shape
        flat_feature = np.ascontiguousarray(
            feature.transpose(1, 2, 0).reshape(-1, channels)
        )
        metadata = {
            'feature_layout': 'PC',
            'feature_shape': list(flat_feature.shape),
            'feature_grid_size': [grid_width, grid_height],
            'feature_dtype': str(feature.dtype),
            'original_size': list(original_size),
            'padded_size': list(padded_size),
            'patch_size': self.patch_size,
            'tile_size': [state['tile_width'], state['tile_height']],
            'tile_overlap': self.tile_overlap,
            'tile_count': state['tile_count'],
        }
        if self.save_pca:
            self._save_pca_visualization(
                image_file.stem,
                feature,
                original_size,
                padded_size,
            )
        output_path = self._save_feature(image_file.stem, flat_feature, metadata)
        self._update_matching_correspondence(
            image_file.stem, original_size, grid_width, grid_height
        )
        return output_path

    def _save_pca_visualization(
        self,
        stem: str,
        feature: np.ndarray,
        original_size: Tuple[int, int],
        padded_size: Tuple[int, int],
    ) -> Path:
        """将 CHW patch 特征转换为三通道 PCA 图像并恢复到原始尺寸。"""
        channels, grid_height, grid_width = feature.shape
        vectors = (
            feature.transpose(1, 2, 0)
            .reshape(-1, channels)
            .astype(np.float32, copy=False)
        )
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.maximum(norms, 1e-8)

        sample_count = min(len(vectors), self.pca_max_samples)
        if sample_count < len(vectors):
            sample_indices = np.linspace(
                0, len(vectors) - 1, sample_count, dtype=np.int64
            )
            fit_vectors = vectors[sample_indices]
        else:
            fit_vectors = vectors

        component_count = min(3, len(fit_vectors), channels)
        pca = PCA(
            n_components=component_count,
            svd_solver='randomized' if component_count < min(fit_vectors.shape) else 'full',
            random_state=0,
        )
        projected = pca.fit(fit_vectors).transform(vectors)
        pca_map = np.zeros((len(vectors), 3), dtype=np.float32)
        pca_map[:, :component_count] = projected
        pca_map = pca_map.reshape(grid_height, grid_width, 3)

        for channel in range(3):
            low, high = np.percentile(pca_map[..., channel], (1, 99))
            pca_map[..., channel] = np.clip(
                (pca_map[..., channel] - low) / max(high - low, 1e-8),
                0,
                1,
            )
        pca_image = Image.fromarray(
            (pca_map * 255).astype(np.uint8), mode='RGB'
        )
        pca_image = pca_image.resize(padded_size, Image.Resampling.BICUBIC)
        pca_image = pca_image.crop((0, 0, original_size[0], original_size[1]))

        output_path = self.pca_output_dir / f'{stem}.{self.pca_format}'
        save_kwargs = {'quality': 95, 'subsampling': 0} if self.pca_format in ('jpg', 'jpeg') else {}
        pca_image.save(output_path, **save_kwargs)
        return output_path

    def _forward_batch(
        self,
        batch: torch.Tensor,
        grid_height: int,
        grid_width: int,
    ) -> torch.Tensor:
        """执行一个 tile batch，并返回 CPU float32 BCHW patch 特征。"""
        amp_dtype = (
            torch.float16
            if self.amp_dtype_name == 'float16'
            else torch.bfloat16
        )
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=amp_dtype,
            enabled=self.amp,
        ):
            tokens = self.model.forward_features(batch)
        patch_count = grid_height * grid_width
        if tokens.ndim != 3 or tokens.shape[1] < patch_count:
            raise RuntimeError(f'Unexpected token tensor: {tuple(tokens.shape)}')
        patch_tokens = tokens[:, -patch_count:, :]
        return (
            patch_tokens.reshape(-1, grid_height, grid_width, self.channels)
            .permute(0, 3, 1, 2)
            .float()
            .cpu()
        )

    def _save_feature(
        self,
        stem: str,
        feature: np.ndarray,
        metadata: Dict,
    ) -> Path:
        """将特征和关键元数据保存到同一 Safetensors 文件。"""
        output_path = self.output_dir / f'{stem}.safetensors'
        save_safetensors(
            {'feature': np.ascontiguousarray(feature)},
            output_path,
            metadata={
                'metadata': json.dumps(
                    metadata, separators=(',', ':'), ensure_ascii=False
                )
            },
        )
        return output_path

    def _update_matching_correspondence(
        self,
        stem: str,
        original_size: Tuple[int, int],
        grid_width: int,
        grid_height: int,
    ):
        """为同名 correspondence 添加或更新一维 patch_index。"""
        if not self.correspondence_by_stem:
            return
        source_path = self.correspondence_by_stem.get(stem)
        if source_path is None:
            self.logger.warning(f'  No matching correspondence for {stem}')
            return
        output_path = (
            self.correspondence_output_dir / source_path.name
            if self.correspondence_output_dir is not None
            else source_path
        )
        self._update_correspondence_safetensors(
            source_path,
            output_path,
            {
                'original_size': list(original_size),
                'patch_size': self.patch_size,
                'feature_shape': [grid_height * grid_width, self.channels],
                'feature_grid_size': [grid_width, grid_height],
            },
        )

    def _update_correspondence_safetensors(
        self,
        source_path: Path,
        output_path: Path,
        metadata: Dict,
    ):
        """生成并更新单个 correspondence Safetensors。"""
        return _update_correspondence_safetensors(source_path, output_path, metadata)


def extract_dino_features(
    image_path: Union[str, Path],
    output_dir: Union[str, Path],
    correspondence_path: Optional[Union[str, Path]] = None,
    correspondence_output_dir: Optional[Union[str, Path]] = None,
    checkpoint_dir: Union[str, Path] = DEFAULT_CHECKPOINT_DIR,
    tile_size: int = 768,
    tile_overlap: int = 128,
    batch_size: int = 4,
    output_dtype: str = 'float16',
    device: Optional[str] = None,
    amp: Optional[bool] = None,
    amp_dtype: str = 'float16',
    save_pca: bool = True,
    pca_output_dir: Optional[Union[str, Path]] = None,
    pca_format: str = 'jpg',
    pca_max_samples: int = 100000,
    prefetch_batches: int = 2,
) -> List[Path]:
    """创建 DINOv3 提取器并处理全部输入影像。"""
    extractor = DINOFeatureExtractor(
        image_path=image_path,
        output_dir=output_dir,
        correspondence_path=correspondence_path,
        correspondence_output_dir=correspondence_output_dir,
        checkpoint_dir=checkpoint_dir,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        batch_size=batch_size,
        output_dtype=output_dtype,
        device=device,
        amp=amp,
        amp_dtype=amp_dtype,
        save_pca=save_pca,
        pca_output_dir=pca_output_dir,
        pca_format=pca_format,
        pca_max_samples=pca_max_samples,
        prefetch_batches=prefetch_batches,
    )
    return extractor.process_all_images()


if __name__ == '__main__':
    extract_dino_features(
        image_path=r'E:\data\湖北\joint_tiles\image',
        output_dir=r'E:\data\湖北\joint_tiles\dino_feature',
        correspondence_path=r'E:\data\湖北\joint_tiles\correspondence',
        correspondence_output_dir=None,
        checkpoint_dir=DEFAULT_CHECKPOINT_DIR,

        tile_size=768,
        tile_overlap=128,
        batch_size=4,
        prefetch_batches=2,

        output_dtype='float16',
        
        amp=True,
        amp_dtype='float16',

        save_pca=True,
        pca_output_dir=r'E:\data\湖北\joint_tiles\dino_pca',
        pca_format='jpg',
        pca_max_samples=100000,
        
    )
