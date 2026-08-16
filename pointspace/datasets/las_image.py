"""读取 LAS/LAZ 点云、DINO 特征及点到 patch 映射的数据集。

``LasImageDataset`` 不读取原始影像，而是读取同名的 DINO Safetensors
特征和 correspondence Safetensors。三类文件通过不含扩展名的文件名配对。

返回 ``data_dict`` 中与影像特征有关的字段如下。这里 ``N`` 表示当前点云
的点数，``C`` 表示 DINO 特征维数，``Hf``、``Wf`` 表示 DINO patch 网格的
行数和列数，``P=Hf*Wf``。所有二维坐标和尺寸均采用 ``(row, col)`` /
``(height, width)`` 顺序。

点级字段（点采样、裁剪和体素下采样时与点云同步变化）：

    dino_pixel_coord: (N, 2), int64
        每个点在原始正射影像中的像素坐标 ``(pixel_row, pixel_col)``。
        无影像覆盖时坐标可能超出影像范围，因此使用前还应检查
        ``dino_valid``。
    dino_patch_index: (N,), int64
        每个点对应的一维 patch 索引，单样本内为
        ``patch_row * Wf + patch_col``。无效点为 -1；合批时有效索引会自动
        加上前序样本的 patch 数，可直接执行 ``dino_feature[index]``。
    dino_valid: (N,), bool
        点是否具有有效的影像 patch 对应关系；开启表面点过滤生成映射时，
        该字段还同时反映点是否为正射视角下可见的表面点。

样本级字段（点采样时保持不变）：

    dino_feature: (P, C), float16 或 float32
        按行优先顺序展平的 DINO patch 特征，布局为 [P,C]。
    dino_offset: (1,), int64
        当前样本的 patch 数 P；合批后为各样本 patch 数的累积和，语义与
        点云 ``offset`` 一致。
    dino_original_size: (1, 2), int64
        原始影像尺寸 ``(height, width)``。
    dino_padded_size: (1, 2), int64
        为适配 patch 或推理 tile 而补边后的尺寸 ``(height, width)``。
    dino_feature_size: (1, 2), int64
        有效 DINO 特征网格尺寸 ``(Hf, Wf)``。
    dino_patch_size: (1,), int64
        DINO 模型单个 patch 在原始影像上对应的像素边长。

经过 ``point_collate_fn`` 合批后，点级字段沿点维拼接，``dino_feature``
直接拼成 ``(sum(P), C)``，``dino_offset`` 变为累积边界。样本级尺寸字段
变为 ``(B, ...)``，二维网格宽高仍可由 ``dino_feature_size`` 获取。

特意不使用项目已有的 ``correspondence`` 字段名，因为该字段在其他数据集
中表示形状为 ``(N, num_images, 2)`` 的多视角像素关系，二者语义和形状不同。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from safetensors import safe_open

from pointspace.utils.logger import get_root_logger

from .builder import DATASETS
from .las import LasDataset


@DATASETS.register_module()
class LasImageDataset(LasDataset):
    """Read paired LAS/LAZ, DINO and point-to-patch Safetensors files.

    The three inputs are paired by file stem.  This dataset deliberately does
    not use the generic ``correspondence`` key: that key already means a
    multi-view ``[num_points, num_images, 2]`` pixel tensor in PointSpace.
    DINO-related point coordinates use ``(row, col)`` order so they can index
    image or feature tensors directly.  The main output fields are
    ``dino_feature`` (PC), ``dino_pixel_coord`` (N x 2),
    ``dino_patch_index`` (N), and ``dino_valid`` (N).  All ``dino_*_size``
    fields likewise use ``(height, width)`` order.

    When explicit paths are omitted, the expected layout is::

        data_root/
            pointcloud/*.las|*.laz
            dino_feature/*.safetensors
            correspondence/*.safetensors

    Cache-related arguments are intentionally absent.  LAS/LAZ and DINO
    assets are read directly for every sample.
    """

    DINO_POINT_KEYS = (
        "dino_pixel_coord",
        "dino_patch_index",
        "dino_valid",
    )

    def __init__(
        self,
        split="train",
        data_root="data/dataset",
        data_path=None,
        data_list=None,
        dino_feature_path=None,
        correspondence_path=None,
        transform=None,
        post_transform=None,
        aug_transform=None,
        test_mode=False,
        ignore_index=-1,
        loop=1,
        required_class=None,
        remap_class=False,
        class_weight=None,
        weight_sample=0.2,
        weighted_sampler=False,
        target_key=None,
    ):
        root = Path(data_root)
        if data_path is None and data_list is None:
            default_pointcloud_path = root / "pointcloud"
            if default_pointcloud_path.is_dir():
                data_path = str(default_pointcloud_path)

        self.dino_feature_path = Path(
            dino_feature_path
            if dino_feature_path is not None
            else root / "dino_feature"
        )
        self.correspondence_path = Path(
            correspondence_path
            if correspondence_path is not None
            else root / "correspondence"
        )
        self._dino_assets = {}

        super().__init__(
            split=split,
            data_root=data_root,
            data_path=data_path,
            data_list=data_list,
            transform=transform,
            post_transform=post_transform,
            aug_transform=aug_transform,
            test_mode=test_mode,
            ignore_index=ignore_index,
            loop=loop,
            required_class=required_class,
            remap_class=remap_class,
            class_weight=class_weight,
            weight_sample=weight_sample,
            weighted_sampler=weighted_sampler,
            target_key=target_key,
        )

    @staticmethod
    def _index_assets(path, suffix, description):
        path = Path(path)
        if path.is_file():
            files = [path]
        elif path.is_dir():
            files = sorted(
                file for file in path.iterdir()
                if file.is_file() and file.suffix.lower() == suffix
            )
        else:
            raise FileNotFoundError(f"{description} path does not exist: {path}")

        wrong_suffix = [file for file in files if file.suffix.lower() != suffix]
        if wrong_suffix:
            raise ValueError(f"Expected {suffix} {description}: {wrong_suffix[0]}")
        if not files:
            raise FileNotFoundError(f"No {suffix} {description} files found: {path}")

        result = {}
        for file in files:
            if file.stem in result:
                raise ValueError(
                    f"Duplicate {description} stem {file.stem!r}: "
                    f"{result[file.stem]} and {file}"
                )
            result[file.stem] = str(file)
        return result

    def get_data_list(self):
        data_list = super().get_data_list()
        feature_files = self._index_assets(
            self.dino_feature_path, ".safetensors", "DINO feature"
        )
        mapping_files = self._index_assets(
            self.correspondence_path, ".safetensors", "point-to-patch mapping"
        )

        missing = []
        assets = {}
        for las_path in data_list:
            stem = Path(las_path).stem
            missing_assets = []
            if stem not in feature_files:
                missing_assets.append("DINO feature")
            if stem not in mapping_files:
                missing_assets.append("point-to-patch mapping")
            if missing_assets:
                missing.append(f"{stem}: {', '.join(missing_assets)}")
                continue
            assets[stem] = (feature_files[stem], mapping_files[stem])

        if missing:
            preview = "; ".join(missing[:8])
            if len(missing) > 8:
                preview += f"; ... and {len(missing) - 8} more"
            raise FileNotFoundError(
                "Each LAS/LAZ sample must have same-stem DINO and mapping assets. "
                + preview
            )

        self._dino_assets = assets
        get_root_logger().info(
            "LasImageDataset: paired %d LAS/LAZ, DINO and mapping samples.",
            len(data_list),
        )
        return data_list

    @staticmethod
    def _metadata_pair(metadata, key, required=True):
        value = metadata.get(key)
        if value is None:
            if required:
                raise ValueError(f"DINO metadata is missing {key!r}")
            return None
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"DINO metadata {key!r} must contain [width, height]")
        width, height = (int(value[0]), int(value[1]))
        if width <= 0 or height <= 0:
            raise ValueError(f"DINO metadata {key!r} must be positive, got {value}")
        return width, height

    @staticmethod
    def _load_dino_feature(path):
        try:
            with safe_open(path, framework="pt", device="cpu") as file:
                if "feature" not in file.keys():
                    raise ValueError("Safetensors has no 'feature' tensor")
                feature = file.get_tensor("feature")
                raw_metadata = (file.metadata() or {}).get("metadata")
        except Exception as error:
            raise ValueError(f"Failed to read DINO feature {path}: {error}") from error

        if raw_metadata is None:
            raise ValueError(f"Safetensors has no DINO metadata: {path}")
        try:
            metadata = json.loads(raw_metadata)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid DINO metadata JSON in {path}: {error}") from error

        if feature.ndim != 2:
            raise ValueError(
                f"DINO feature must use PC layout, got shape {tuple(feature.shape)}: {path}"
            )
        metadata_shape = metadata.get("feature_shape")
        if metadata_shape is None or tuple(map(int, metadata_shape)) != tuple(feature.shape):
            raise ValueError(
                f"DINO feature_shape metadata does not match tensor {tuple(feature.shape)}: {path}"
            )
        if metadata.get("feature_layout", "PC").upper() != "PC":
            raise ValueError(f"Only PC DINO features are supported: {path}")
        return feature.contiguous(), metadata

    @staticmethod
    def _load_mapping(path, num_points):
        try:
            with safe_open(path, framework="pt", device="cpu") as file:
                required = {"pixel_coord", "patch_index", "valid"}
                missing = required.difference(file.keys())
                if missing:
                    raise ValueError(f"missing tensors: {', '.join(sorted(missing))}")
                pixel_coord = file.get_tensor("pixel_coord")
                patch_index = file.get_tensor("patch_index")
                valid = file.get_tensor("valid")
                mapping_metadata = file.metadata() or {}
        except Exception as error:
            raise ValueError(f"Failed to read mapping {path}: {error}") from error
        if pixel_coord.shape != (num_points, 2):
            raise ValueError(
                f"pixel_coord shape mismatch for {path}: expected ({num_points}, 2), "
                f"got {tuple(pixel_coord.shape)}"
            )
        if patch_index.shape != (num_points,) or valid.shape != (num_points,):
            raise ValueError(f"patch_index/valid length mismatch for {path}")
        return pixel_coord.long(), patch_index.long(), valid.bool(), mapping_metadata

    def get_data(self, idx):
        data_dict = super().get_data(idx)
        name = data_dict["name"]
        feature_path, mapping_path = self._dino_assets[name]
        feature, metadata = self._load_dino_feature(feature_path)
        pixel_coord, patch_index, valid, mapping_metadata = self._load_mapping(
            mapping_path, data_dict["coord"].shape[0]
        )

        original_width, original_height = self._metadata_pair(
            metadata, "original_size"
        )
        feature_grid_size = metadata.get("feature_grid_size")
        if not isinstance(feature_grid_size, (list, tuple)) or len(feature_grid_size) != 2:
            raise ValueError(f"Invalid feature_grid_size in {feature_path}")
        grid_width, grid_height = map(int, feature_grid_size)
        if grid_width * grid_height != feature.shape[0]:
            raise ValueError(f"Feature grid does not match feature length: {feature_path}")
        patch_size = int(metadata.get("patch_size", 0))
        if patch_size <= 0:
            raise ValueError(f"Invalid patch_size in {feature_path}: {patch_size}")
        mapping_grid = mapping_metadata.get("feature_grid_size")
        if mapping_grid is not None and json.loads(mapping_grid) != [grid_width, grid_height]:
            raise ValueError(f"Mapping and DINO feature grid mismatch: {mapping_path}")
        mapping_patch_size = mapping_metadata.get("patch_size")
        if mapping_patch_size is not None and int(mapping_patch_size) != patch_size:
            raise ValueError(f"Mapping and DINO patch_size mismatch: {mapping_path}")
        mapping_original_size = mapping_metadata.get("original_size")
        if (
            mapping_original_size is not None
            and json.loads(mapping_original_size) != [original_width, original_height]
        ):
            raise ValueError(f"Mapping and DINO original_size mismatch: {mapping_path}")
        padded_size = self._metadata_pair(metadata, "padded_size", required=False)
        if padded_size is None:
            padded_width = grid_width * patch_size
            padded_height = grid_height * patch_size
        else:
            padded_width, padded_height = padded_size

        # Sample-level shapes have a leading singleton dimension.  The project
        # collator can therefore concatenate them into [batch, ...].
        data_dict.update(
            dino_feature=feature,
            dino_pixel_coord=pixel_coord,
            dino_patch_index=patch_index,
            dino_valid=valid,
            dino_offset=np.asarray([feature.shape[0]], dtype=np.int64),
            dino_original_size=np.asarray(
                [[original_height, original_width]], dtype=np.int64
            ),
            dino_padded_size=np.asarray(
                [[padded_height, padded_width]], dtype=np.int64
            ),
            dino_feature_size=np.asarray(
                [[grid_height, grid_width]], dtype=np.int64
            ),
            dino_patch_size=np.asarray([patch_size], dtype=np.int64),
        )

        point_keys = list(data_dict.get("index_valid_keys", []))
        for key in self.DINO_POINT_KEYS:
            if key not in point_keys:
                point_keys.append(key)
        data_dict["index_valid_keys"] = point_keys
        return data_dict
