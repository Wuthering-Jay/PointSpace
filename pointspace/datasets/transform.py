"""
3D point cloud augmentation

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com), Yujia Zhang (yujia.zhang.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import random
import numbers
import scipy
import scipy.stats
from scipy.stats import binned_statistic_2d
from scipy.ndimage import convolve, gaussian_filter, map_coordinates
from scipy.interpolate import NearestNDInterpolator, RegularGridInterpolator
from scipy.spatial import cKDTree
import numpy as np
import torch
from torchvision import transforms
import copy
from collections.abc import Sequence, Mapping
from pointspace.utils.registry import Registry

TRANSFORMS = Registry("transforms")


def index_operator(data_dict, index, duplicate=False):
    # index selection operator for keys in "index_valid_keys"
    # custom these keys by "Update" transform in config
    if "index_valid_keys" not in data_dict:
        data_dict["index_valid_keys"] = [
            "coord",
            "color",
            "normal",
            "superpoint",
            "intensity",
            "echo",
            "segment",
            "instance",
        ]
    if not duplicate:
        for key in data_dict["index_valid_keys"]:
            if key in data_dict:
                data_dict[key] = data_dict[key][index]
        return data_dict
    else:
        data_dict_ = dict()
        for key in data_dict.keys():
            if key in data_dict["index_valid_keys"]:
                data_dict_[key] = data_dict[key][index]
            elif key == "index_valid_keys":
                data_dict_[key] = copy.copy(data_dict[key])
            else:
                data_dict_[key] = data_dict[key]
        return data_dict_


@TRANSFORMS.register_module()
class Collect(object):
    def __init__(self, keys, offset_keys_dict=None, optional_keys=None, **kwargs):
        """
        e.g. Collect(keys=[coord], feat_keys=[coord, color])
        """
        if offset_keys_dict is None:
            offset_keys_dict = dict(offset="coord")
        self.keys = keys
        self.offset_keys = offset_keys_dict
        self.optional_keys = set(optional_keys) if optional_keys else set()
        self.kwargs = kwargs

    def __call__(self, data_dict):
        data = dict()
        if isinstance(self.keys, str):
            self.keys = [self.keys]
        for key in self.keys:
            if key in data_dict:
                data[key] = data_dict[key]
            elif key not in self.optional_keys:
                raise KeyError(
                    f"Collect: required key '{key}' not found in data_dict. "
                    f"Available keys: {list(data_dict.keys())}"
                )
        for key, value in self.offset_keys.items():
            data[key] = torch.tensor([data_dict[value].shape[0]])
        for name, keys in self.kwargs.items():
            name = name.replace("_keys", "")
            assert isinstance(keys, Sequence)
            data[name] = torch.cat([data_dict[key].float() for key in keys], dim=1)
        return data


@TRANSFORMS.register_module()
class Copy(object):
    def __init__(self, keys_dict=None):
        if keys_dict is None:
            keys_dict = dict(coord="origin_coord", segment="origin_segment")
        self.keys_dict = keys_dict

    def __call__(self, data_dict):
        for key, value in self.keys_dict.items():
            if isinstance(data_dict[key], np.ndarray):
                data_dict[value] = data_dict[key].copy()
            elif isinstance(data_dict[key], torch.Tensor):
                data_dict[value] = data_dict[key].clone().detach()
            else:
                data_dict[value] = copy.deepcopy(data_dict[key])
        return data_dict


@TRANSFORMS.register_module()
class Update(object):
    def __init__(self, keys_dict=None):
        if keys_dict is None:
            keys_dict = dict()
        self.keys_dict = keys_dict

    def __call__(self, data_dict):
        for key, value in self.keys_dict.items():
            data_dict[key] = value
        return data_dict


@TRANSFORMS.register_module()
class ToTensor(object):
    def __call__(self, data):
        if isinstance(data, torch.Tensor):
            return data
        elif isinstance(data, str):
            # note that str is also a kind of sequence, judgement should before sequence
            return data
        elif isinstance(data, int):
            return torch.LongTensor([data])
        elif isinstance(data, float):
            return torch.FloatTensor([data])
        elif isinstance(data, np.ndarray) and np.issubdtype(data.dtype, bool):
            return torch.from_numpy(data)
        elif isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.integer):
            return torch.from_numpy(data).long()
        elif isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.floating):
            return torch.from_numpy(data).float()
        elif isinstance(data, Mapping):
            result = {sub_key: self(item) for sub_key, item in data.items()}
            return result
        elif isinstance(data, Sequence):
            result = [self(item) for item in data]
            return result
        else:
            raise TypeError(f"type {type(data)} cannot be converted to tensor.")


@TRANSFORMS.register_module()
class NormalizeColor8bit(object):
    def __call__(self, data_dict):
        if "color" in data_dict.keys():
            data_dict["color"] = data_dict["color"] / 255
        return data_dict
    
@TRANSFORMS.register_module()
class NormalizeColor16bit(object):
    def __call__(self, data_dict):
        if "color" in data_dict.keys():
            data_dict["color"] = data_dict["color"] / 65535
        return data_dict
    
@TRANSFORMS.register_module()
class RobustLogIntensity(object):
    """
    [Normalization] 针对 Intensity 的稳健对数归一化。
    解决长尾分布、量纲不统一、整体亮度漂移问题。
    """
    def __init__(self, clip_min=-3.0, clip_max=3.0):
        self.clip_min = clip_min
        self.clip_max = clip_max

    def __call__(self, data_dict):
        if "intensity" not in data_dict:
            return data_dict

        intensity = data_dict["intensity"]
        
        # 1. Log 变换 (压缩长尾)
        intensity_log = np.log(intensity + 1.0)
        
        # 2. 计算稳健统计量 (Instance-wise)
        median = np.median(intensity_log)
        q75, q25 = np.percentile(intensity_log, [75, 25])
        iqr = q75 - q25
        if iqr < 1e-6: iqr = 1.0
            
        # 3. 稳健标准化 (N(0, 1))
        intensity_norm = (intensity_log - median) / iqr
        
        # 4. 截断极值 (抑制高反/极暗噪点)
        intensity_norm = np.clip(intensity_norm, self.clip_min, self.clip_max)
        
        data_dict["intensity"] = intensity_norm
        return data_dict


def _fit_plane_least_squares(xy, z):
    """Fit z = ax + by + c by least squares."""
    if xy.shape[0] < 3:
        return np.array([0.0, 0.0, float(np.median(z) if z.size > 0 else 0.0)], dtype=np.float32)
    a = np.concatenate([xy.astype(np.float64), np.ones((xy.shape[0], 1), dtype=np.float64)], axis=1)
    coef, *_ = np.linalg.lstsq(a, z.astype(np.float64), rcond=None)
    return coef.astype(np.float32)


def _predict_plane(coef, xy):
    return (xy[:, 0] * coef[0] + xy[:, 1] * coef[1] + coef[2]).astype(np.float32)


def _local_min_mask_xy(coord, xy_grid):
    """Keep only the lowest-Z point in each XY cell."""
    if xy_grid is None or xy_grid <= 0 or coord.shape[0] == 0:
        return np.ones(coord.shape[0], dtype=bool)

    grid = np.floor(coord[:, :2] / float(xy_grid)).astype(np.int64)
    min_z = {}
    min_idx = {}
    for i in range(coord.shape[0]):
        key = (int(grid[i, 0]), int(grid[i, 1]))
        z = float(coord[i, 2])
        if key not in min_z or z < min_z[key]:
            min_z[key] = z
            min_idx[key] = i
    mask = np.zeros(coord.shape[0], dtype=bool)
    mask[list(min_idx.values())] = True
    return mask


def _compute_geometric_features_np(
    coord,
    k=25,
    k_min=5,
    add_self_as_neighbor=True,
):
    """Compute a subset of official PointFeatures on CPU with numpy/scipy."""
    n = coord.shape[0]
    coord = np.asarray(coord, dtype=np.float32)
    if n == 0:
        return {
            "linearity": np.empty((0, 1), dtype=np.float32),
            "planarity": np.empty((0, 1), dtype=np.float32),
            "scattering": np.empty((0, 1), dtype=np.float32),
            "verticality": np.empty((0, 1), dtype=np.float32),
            "normal": np.empty((0, 3), dtype=np.float32),
        }

    k_eff = int(max(1, min(k, n)))
    tree = cKDTree(coord)
    _, nn = tree.query(coord, k=k_eff)
    if k_eff == 1:
        nn = nn[:, None]

    if not add_self_as_neighbor:
        # cKDTree includes self for exact queries; drop it when asked.
        nn = nn[:, 1:] if nn.shape[1] > 1 else nn[:, :0]

    if nn.shape[1] == 0:
        zero_1 = np.zeros((n, 1), dtype=np.float32)
        zero_3 = np.zeros((n, 3), dtype=np.float32)
        return {
            "linearity": zero_1.copy(),
            "planarity": zero_1.copy(),
            "scattering": zero_1.copy(),
            "verticality": zero_1.copy(),
            "normal": zero_3,
        }

    pts = coord[nn]  # [N, K, 3]
    center = pts.mean(axis=1, keepdims=True)
    centered = pts - center
    cov = np.einsum("nki,nkj->nij", centered, centered) / max(pts.shape[1], 1)

    eigval, eigvec = np.linalg.eigh(cov)
    eigval = np.clip(eigval, a_min=0.0, a_max=None).astype(np.float32)
    eigvec = eigvec.astype(np.float32)

    normal = eigvec[:, :, 0]
    flip_mask = normal[:, 2] < 0
    normal[flip_mask] *= -1.0

    lambda_1 = np.sqrt(eigval[:, 2])
    lambda_2 = np.sqrt(eigval[:, 1])
    lambda_3 = np.sqrt(eigval[:, 0])

    denom = lambda_1 + 1e-3
    linearity = ((lambda_1 - lambda_2) / denom).reshape(-1, 1)
    planarity = ((lambda_2 - lambda_3) / denom).reshape(-1, 1)
    scattering = (lambda_3 / denom).reshape(-1, 1)

    unary = (np.abs(eigvec) * eigval[:, None, :]).sum(axis=2)
    verticality = (unary[:, 2] / (np.linalg.norm(unary, axis=1) + 1e-8)).reshape(-1, 1)
    verticality *= 2.0

    if pts.shape[1] < k_min:
        linearity.fill(0.0)
        planarity.fill(0.0)
        scattering.fill(0.0)
        verticality.fill(0.0)
        normal.fill(0.0)

    return {
        "linearity": linearity.astype(np.float32),
        "planarity": planarity.astype(np.float32),
        "scattering": scattering.astype(np.float32),
        "verticality": verticality.astype(np.float32),
        "normal": normal.astype(np.float32),
    }


@TRANSFORMS.register_module()
class PointFeatures(object):
    """Minimal PointFeatures subset aligned with official EZ-SP needs.

    Currently supports the DALES-relevant geometric keys:
    `linearity`, `planarity`, `scattering`, `verticality`, `normal`.
    """

    _SUPPORTED_KEYS = {"linearity", "planarity", "scattering", "verticality", "normal"}

    def __init__(
        self,
        keys=None,
        k_min=5,
        k=25,
        add_self_as_neighbor=True,
        overwrite=True,
    ):
        self.keys = list(keys) if keys is not None else ["linearity", "planarity", "scattering", "verticality"]
        self.k_min = k_min
        self.k = k
        self.add_self_as_neighbor = add_self_as_neighbor
        self.overwrite = overwrite

    def __call__(self, data_dict):
        if "coord" not in data_dict or len(data_dict["coord"]) == 0:
            return data_dict

        requested = [k for k in self.keys if k in self._SUPPORTED_KEYS]
        if not requested:
            return data_dict

        if not self.overwrite:
            requested = [k for k in requested if k not in data_dict]
            if not requested:
                return data_dict

        features = _compute_geometric_features_np(
            data_dict["coord"],
            k=self.k,
            k_min=self.k_min,
            add_self_as_neighbor=self.add_self_as_neighbor,
        )
        for key in requested:
            data_dict[key] = features[key]

        if "index_valid_keys" not in data_dict:
            data_dict["index_valid_keys"] = []
        for key in requested:
            if key not in data_dict["index_valid_keys"]:
                data_dict["index_valid_keys"].append(key)
        return data_dict


@TRANSFORMS.register_module()
class GroundElevation(object):
    """Approximate official GroundElevation for outdoor point clouds."""

    def __init__(
        self,
        z_threshold=None,
        verticality_threshold=None,
        xy_grid=None,
        model="ransac",
        scale=3.0,
        k=3,
    ):
        self.z_threshold = z_threshold
        self.verticality_threshold = verticality_threshold
        self.xy_grid = xy_grid
        self.model = model
        self.scale = scale
        self.k = max(1, int(k))
        assert model in ["ransac", "knn", "mlp"]

    def __call__(self, data_dict):
        if self.scale <= 0 or "coord" not in data_dict or len(data_dict["coord"]) == 0:
            return data_dict

        coord = np.asarray(data_dict["coord"], dtype=np.float32)
        mask = np.ones(coord.shape[0], dtype=bool)

        if self.z_threshold is not None:
            mask &= coord[:, 2] <= (coord[:, 2].min() + float(self.z_threshold))

        if (
            self.verticality_threshold is not None
            and 0 < self.verticality_threshold < 1
            and "verticality" in data_dict
        ):
            verticality = np.asarray(data_dict["verticality"]).reshape(-1)
            mask &= verticality < float(self.verticality_threshold)

        if self.xy_grid is not None and self.xy_grid > 0:
            mask &= _local_min_mask_xy(coord, self.xy_grid)

        ref = coord[mask]
        if ref.shape[0] == 0:
            ref = coord

        xy = coord[:, :2]
        ref_xy = ref[:, :2]
        ref_z = ref[:, 2]

        if self.model == "knn" and ref.shape[0] > 0:
            tree = cKDTree(ref_xy)
            k_eff = min(self.k, ref.shape[0])
            dist, idx = tree.query(xy, k=k_eff)
            if k_eff == 1:
                dist = dist[:, None]
                idx = idx[:, None]
            weight = 1.0 / np.maximum(dist, 1e-3)
            z_ground = (weight * ref_z[idx]).sum(axis=1) / np.maximum(weight.sum(axis=1), 1e-6)
        else:
            coef = _fit_plane_least_squares(ref_xy, ref_z)
            if self.model == "ransac" and ref.shape[0] >= 6:
                pred_ref = _predict_plane(coef, ref_xy)
                resid = np.abs(ref_z - pred_ref)
                thresh = np.percentile(resid, 70)
                inlier = resid <= max(thresh, 1e-2)
                if inlier.sum() >= 3:
                    coef = _fit_plane_least_squares(ref_xy[inlier], ref_z[inlier])
            z_ground = _predict_plane(coef, xy)

        elevation = ((coord[:, 2] - z_ground) / float(self.scale)).reshape(-1, 1).astype(np.float32)
        data_dict["elevation"] = elevation
        if "index_valid_keys" not in data_dict:
            data_dict["index_valid_keys"] = []
        if "elevation" not in data_dict["index_valid_keys"]:
            data_dict["index_valid_keys"].append("elevation")
        return data_dict


@TRANSFORMS.register_module()
class NormalizeCoord(object):
    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            # modified from pointnet2
            centroid = np.mean(data_dict["coord"], axis=0)
            data_dict["coord"] -= centroid
            if "core_bbox" in data_dict:
                data_dict["core_bbox"][0::2] -= centroid[0]
                data_dict["core_bbox"][1::2] -= centroid[1]
            m = np.max(np.sqrt(np.sum(data_dict["coord"] ** 2, axis=1)))
            data_dict["coord"] = data_dict["coord"] / m
            if "core_bbox" in data_dict:
                data_dict["core_bbox"] /= m
        return data_dict


@TRANSFORMS.register_module()
class PositiveShift(object):
    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            coord_min = np.min(data_dict["coord"], 0)
            data_dict["coord"] -= coord_min
            if "core_bbox" in data_dict:
                data_dict["core_bbox"][0::2] -= coord_min[0]
                data_dict["core_bbox"][1::2] -= coord_min[1]
        return data_dict


@TRANSFORMS.register_module()
class CenterShift(object):
    def __init__(self, apply_z=True):
        self.apply_z = apply_z

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            x_min, y_min, z_min = data_dict["coord"].min(axis=0)
            x_max, y_max, _ = data_dict["coord"].max(axis=0)
            if self.apply_z:
                shift = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2, z_min])
            else:
                shift = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2, 0])
            data_dict["coord"] -= shift
            data_dict["coord_shift"] = shift
            if "core_bbox" in data_dict:
                data_dict["core_bbox"][0::2] -= shift[0]
                data_dict["core_bbox"][1::2] -= shift[1]
        return data_dict


@TRANSFORMS.register_module()
class CentroidShift(object):
    def __init__(self, apply_z=True):
        self.apply_z = apply_z

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            centroid = np.mean(data_dict["coord"], axis=0)
            if not self.apply_z:
                centroid[2] = 0
            data_dict["coord"] -= centroid
            data_dict["coord_shift"] = centroid
            if "core_bbox" in data_dict:
                data_dict["core_bbox"][0::2] -= centroid[0]
                data_dict["core_bbox"][1::2] -= centroid[1]
        return data_dict


@TRANSFORMS.register_module()
class ZPercentileCenterShift(object):
    def __init__(self, percentile=1.0):
        self.percentile = percentile

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            coords = data_dict["coord"]
            x_min, y_min = coords[:, 0].min(), coords[:, 1].min()
            x_max, y_max = coords[:, 0].max(), coords[:, 1].max()
            z_shift_val = np.percentile(coords[:, 2], self.percentile)
            shift = np.array([
                (x_min + x_max) / 2.0,
                (y_min + y_max) / 2.0,
                z_shift_val
            ])
            data_dict["coord"] -= shift
            data_dict["coord_shift"] = shift
            if "core_bbox" in data_dict:
                data_dict["core_bbox"][0::2] -= shift[0]
                data_dict["core_bbox"][1::2] -= shift[1]

        return data_dict


@TRANSFORMS.register_module()
class RandomShift(object):
    def __init__(self, shift=((-0.2, 0.2), (-0.2, 0.2), (0, 0))):
        self.shift = shift

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            shift_x = np.random.uniform(self.shift[0][0], self.shift[0][1])
            shift_y = np.random.uniform(self.shift[1][0], self.shift[1][1])
            shift_z = np.random.uniform(self.shift[2][0], self.shift[2][1])
            data_dict["coord"] += [shift_x, shift_y, shift_z]
            if "core_bbox" in data_dict:
                data_dict["core_bbox"][0::2] += shift_x
                data_dict["core_bbox"][1::2] += shift_y
        return data_dict


@TRANSFORMS.register_module()
class PointClip(object):
    def __init__(self, point_cloud_range=(-80, -80, -3, 80, 80, 1)):
        self.point_cloud_range = point_cloud_range

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            data_dict["coord"] = np.clip(
                data_dict["coord"],
                a_min=self.point_cloud_range[:3],
                a_max=self.point_cloud_range[3:],
            )
        return data_dict


@TRANSFORMS.register_module()
class RandomDropout(object):
    def __init__(self, dropout_ratio=0.2, dropout_application_ratio=0.5):
        """
        upright_axis: axis index among x,y,z, i.e. 2 for z
        """
        self.dropout_ratio = dropout_ratio
        self.dropout_application_ratio = dropout_application_ratio

    def __call__(self, data_dict):
        if random.random() < self.dropout_application_ratio:
            n = len(data_dict["coord"])
            idx = np.random.choice(n, int(n * (1 - self.dropout_ratio)), replace=False)
            if "sampled_index" in data_dict:
                # for ScanNet data efficient, we need to make sure labeled point is sampled.
                idx = np.unique(np.append(idx, data_dict["sampled_index"]))
                mask = np.zeros_like(data_dict["segment"]).astype(bool)
                mask[data_dict["sampled_index"]] = True
                data_dict["sampled_index"] = np.where(mask[idx])[0]
            data_dict = index_operator(data_dict, idx)
        return data_dict


@TRANSFORMS.register_module()
class RandomRotate(object):
    def __init__(self, angle=None, center=None, axis="z", always_apply=False, p=0.5):
        self.angle = [-1, 1] if angle is None else angle
        self.axis = axis
        self.always_apply = always_apply
        self.p = p if not self.always_apply else 1
        self.center = center

    def __call__(self, data_dict):
        if random.random() > self.p:
            return data_dict
        angle = np.random.uniform(self.angle[0], self.angle[1]) * np.pi
        rot_cos, rot_sin = np.cos(angle), np.sin(angle)
        if self.axis == "x":
            rot_t = np.array([[1, 0, 0], [0, rot_cos, -rot_sin], [0, rot_sin, rot_cos]])
        elif self.axis == "y":
            rot_t = np.array([[rot_cos, 0, rot_sin], [0, 1, 0], [-rot_sin, 0, rot_cos]])
        elif self.axis == "z":
            rot_t = np.array([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]])
        else:
            raise NotImplementedError
        if "coord" in data_dict.keys():
            if self.center is None:
                x_min, y_min, z_min = data_dict["coord"].min(axis=0)
                x_max, y_max, z_max = data_dict["coord"].max(axis=0)
                center = [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]
            else:
                center = self.center
            data_dict["coord"] -= center
            data_dict["coord"] = np.dot(data_dict["coord"], np.transpose(rot_t))
            data_dict["coord"] += center
        if "normal" in data_dict.keys():
            data_dict["normal"] = np.dot(data_dict["normal"], np.transpose(rot_t))
        return data_dict


@TRANSFORMS.register_module()
class RandomRotateTargetAngle(object):
    def __init__(
        self, angle=(1 / 2, 1, 3 / 2), center=None, axis="z", always_apply=False, p=0.75
    ):
        self.angle = angle
        self.axis = axis
        self.always_apply = always_apply
        self.p = p if not self.always_apply else 1
        self.center = center

    def __call__(self, data_dict):
        if random.random() > self.p:
            return data_dict
        angle = np.random.choice(self.angle) * np.pi
        rot_cos, rot_sin = np.cos(angle), np.sin(angle)
        if self.axis == "x":
            rot_t = np.array([[1, 0, 0], [0, rot_cos, -rot_sin], [0, rot_sin, rot_cos]])
        elif self.axis == "y":
            rot_t = np.array([[rot_cos, 0, rot_sin], [0, 1, 0], [-rot_sin, 0, rot_cos]])
        elif self.axis == "z":
            rot_t = np.array([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]])
        else:
            raise NotImplementedError
        if "coord" in data_dict.keys():
            if self.center is None:
                x_min, y_min, z_min = data_dict["coord"].min(axis=0)
                x_max, y_max, z_max = data_dict["coord"].max(axis=0)
                center = [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]
            else:
                center = self.center
            data_dict["coord"] -= center
            data_dict["coord"] = np.dot(data_dict["coord"], np.transpose(rot_t))
            data_dict["coord"] += center
        if "normal" in data_dict.keys():
            data_dict["normal"] = np.dot(data_dict["normal"], np.transpose(rot_t))
        return data_dict


@TRANSFORMS.register_module()
class RandomScale(object):
    def __init__(self, scale=None, anisotropic=False):
        self.scale = scale if scale is not None else [0.95, 1.05]
        self.anisotropic = anisotropic

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            scale = np.random.uniform(
                self.scale[0], self.scale[1], 3 if self.anisotropic else 1
            )
            data_dict["coord"] *= scale
        return data_dict


@TRANSFORMS.register_module()
class RandomFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data_dict):
        if np.random.rand() < self.p:
            if "coord" in data_dict.keys():
                data_dict["coord"][:, 0] = -data_dict["coord"][:, 0]
            if "normal" in data_dict.keys():
                data_dict["normal"][:, 0] = -data_dict["normal"][:, 0]
        if np.random.rand() < self.p:
            if "coord" in data_dict.keys():
                data_dict["coord"][:, 1] = -data_dict["coord"][:, 1]
            if "normal" in data_dict.keys():
                data_dict["normal"][:, 1] = -data_dict["normal"][:, 1]
        return data_dict


@TRANSFORMS.register_module()
class RandomJitter(object):
    def __init__(self, sigma=0.01, clip=0.05):
        assert clip > 0
        self.sigma = sigma
        self.clip = clip

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            jitter = np.clip(
                self.sigma * np.random.randn(data_dict["coord"].shape[0], 3),
                -self.clip,
                self.clip,
            )
            data_dict["coord"] += jitter
        return data_dict


@TRANSFORMS.register_module()
class ClipGaussianJitter(object):
    def __init__(self, scalar=0.02, store_jitter=False):
        self.scalar = scalar
        self.mean = np.mean(3)
        self.cov = np.identity(3)
        self.quantile = 1.96
        self.store_jitter = store_jitter

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            jitter = np.random.multivariate_normal(
                self.mean, self.cov, data_dict["coord"].shape[0]
            )
            jitter = self.scalar * np.clip(jitter / 1.96, -1, 1)
            data_dict["coord"] += jitter
            if self.store_jitter:
                data_dict["jitter"] = jitter
        return data_dict


@TRANSFORMS.register_module()
class ChromaticAutoContrast(object):
    def __init__(self, p=0.2, blend_factor=None):
        self.p = p
        self.blend_factor = blend_factor

    def __call__(self, data_dict):
        if "color" in data_dict.keys() and np.random.rand() < self.p:
            lo = np.min(data_dict["color"], 0, keepdims=True)
            hi = np.max(data_dict["color"], 0, keepdims=True)
            diff = hi - lo
            if not np.any(diff > 0):
                return data_dict
            scale = np.divide(
                255,
                diff,
                out=np.ones_like(diff, dtype=data_dict["color"].dtype),
                where=diff > 0,
            )
            contrast_feat = (data_dict["color"][:, :3] - lo) * scale
            blend_factor = (
                np.random.rand() if self.blend_factor is None else self.blend_factor
            )
            data_dict["color"][:, :3] = (1 - blend_factor) * data_dict["color"][
                :, :3
            ] + blend_factor * contrast_feat
        return data_dict


@TRANSFORMS.register_module()
class ChromaticTranslation(object):
    def __init__(self, p=0.95, ratio=0.05):
        self.p = p
        self.ratio = ratio

    def __call__(self, data_dict):
        if "color" in data_dict.keys() and np.random.rand() < self.p:
            tr = (np.random.rand(1, 3) - 0.5) * 255 * 2 * self.ratio
            data_dict["color"][:, :3] = np.clip(tr + data_dict["color"][:, :3], 0, 255)
        return data_dict


@TRANSFORMS.register_module()
class ChromaticJitter(object):
    def __init__(self, p=0.95, std=0.005):
        self.p = p
        self.std = std

    def __call__(self, data_dict):
        if "color" in data_dict.keys() and np.random.rand() < self.p:
            noise = np.random.randn(data_dict["color"].shape[0], 3)
            noise *= self.std * 255
            data_dict["color"][:, :3] = np.clip(
                noise + data_dict["color"][:, :3], 0, 255
            )
        return data_dict


@TRANSFORMS.register_module()
class RandomColorGrayScale(object):
    def __init__(self, p):
        self.p = p

    @staticmethod
    def rgb_to_grayscale(color, num_output_channels=1):
        if color.shape[-1] < 3:
            raise TypeError(
                "Input color should have at least 3 dimensions, but found {}".format(
                    color.shape[-1]
                )
            )

        if num_output_channels not in (1, 3):
            raise ValueError("num_output_channels should be either 1 or 3")

        r, g, b = color[..., 0], color[..., 1], color[..., 2]
        gray = (0.2989 * r + 0.587 * g + 0.114 * b).astype(color.dtype)
        gray = np.expand_dims(gray, axis=-1)

        if num_output_channels == 3:
            gray = np.broadcast_to(gray, color.shape)

        return gray

    def __call__(self, data_dict):
        if np.random.rand() < self.p:
            data_dict["color"] = self.rgb_to_grayscale(data_dict["color"], 3)
        return data_dict


@TRANSFORMS.register_module()
class RandomColorJitter(object):
    """
    Random Color Jitter for 3D point cloud (refer torchvision)
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, p=0.95):
        self.brightness = self._check_input(brightness, "brightness")
        self.contrast = self._check_input(contrast, "contrast")
        self.saturation = self._check_input(saturation, "saturation")
        self.hue = self._check_input(
            hue, "hue", center=0, bound=(-0.5, 0.5), clip_first_on_zero=False
        )
        self.p = p

    @staticmethod
    def _check_input(
        value, name, center=1, bound=(0, float("inf")), clip_first_on_zero=True
    ):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError(
                    "If {} is a single number, it must be non negative.".format(name)
                )
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError(
                "{} should be a single number or a list/tuple with length 2.".format(
                    name
                )
            )

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def blend(color1, color2, ratio):
        ratio = float(ratio)
        bound = 255.0
        return (
            (ratio * color1 + (1.0 - ratio) * color2)
            .clip(0, bound)
            .astype(color1.dtype)
        )

    @staticmethod
    def rgb2hsv(rgb):
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        maxc = np.max(rgb, axis=-1)
        minc = np.min(rgb, axis=-1)
        eqc = maxc == minc
        cr = maxc - minc
        s = cr / (np.ones_like(maxc) * eqc + maxc * (1 - eqc))
        cr_divisor = np.ones_like(maxc) * eqc + cr * (1 - eqc)
        rc = (maxc - r) / cr_divisor
        gc = (maxc - g) / cr_divisor
        bc = (maxc - b) / cr_divisor

        hr = (maxc == r) * (bc - gc)
        hg = ((maxc == g) & (maxc != r)) * (2.0 + rc - bc)
        hb = ((maxc != g) & (maxc != r)) * (4.0 + gc - rc)
        h = hr + hg + hb
        h = (h / 6.0 + 1.0) % 1.0
        return np.stack((h, s, maxc), axis=-1)

    @staticmethod
    def hsv2rgb(hsv):
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        i = np.floor(h * 6.0)
        f = (h * 6.0) - i
        i = i.astype(np.int32)

        p = np.clip((v * (1.0 - s)), 0.0, 1.0)
        q = np.clip((v * (1.0 - s * f)), 0.0, 1.0)
        t = np.clip((v * (1.0 - s * (1.0 - f))), 0.0, 1.0)
        i = i % 6
        mask = np.expand_dims(i, axis=-1) == np.arange(6)

        a1 = np.stack((v, q, p, p, t, v), axis=-1)
        a2 = np.stack((t, v, v, q, p, p), axis=-1)
        a3 = np.stack((p, p, t, v, v, q), axis=-1)
        a4 = np.stack((a1, a2, a3), axis=-1)

        return np.einsum("...na, ...nab -> ...nb", mask.astype(hsv.dtype), a4)

    def adjust_brightness(self, color, brightness_factor):
        if brightness_factor < 0:
            raise ValueError(
                "brightness_factor ({}) is not non-negative.".format(brightness_factor)
            )

        return self.blend(color, np.zeros_like(color), brightness_factor)

    def adjust_contrast(self, color, contrast_factor):
        if contrast_factor < 0:
            raise ValueError(
                "contrast_factor ({}) is not non-negative.".format(contrast_factor)
            )
        mean = np.mean(RandomColorGrayScale.rgb_to_grayscale(color))
        return self.blend(color, mean, contrast_factor)

    def adjust_saturation(self, color, saturation_factor):
        if saturation_factor < 0:
            raise ValueError(
                "saturation_factor ({}) is not non-negative.".format(saturation_factor)
            )
        gray = RandomColorGrayScale.rgb_to_grayscale(color)
        return self.blend(color, gray, saturation_factor)

    def adjust_hue(self, color, hue_factor):
        if not (-0.5 <= hue_factor <= 0.5):
            raise ValueError(
                "hue_factor ({}) is not in [-0.5, 0.5].".format(hue_factor)
            )
        orig_dtype = color.dtype
        hsv = self.rgb2hsv(color / 255.0)
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        h = (h + hue_factor) % 1.0
        hsv = np.stack((h, s, v), axis=-1)
        color_hue_adj = (self.hsv2rgb(hsv) * 255.0).astype(orig_dtype)
        return color_hue_adj

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        fn_idx = torch.randperm(4)
        b = (
            None
            if brightness is None
            else np.random.uniform(brightness[0], brightness[1])
        )
        c = None if contrast is None else np.random.uniform(contrast[0], contrast[1])
        s = (
            None
            if saturation is None
            else np.random.uniform(saturation[0], saturation[1])
        )
        h = None if hue is None else np.random.uniform(hue[0], hue[1])
        return fn_idx, b, c, s, h

    def __call__(self, data_dict):
        (
            fn_idx,
            brightness_factor,
            contrast_factor,
            saturation_factor,
            hue_factor,
        ) = self.get_params(self.brightness, self.contrast, self.saturation, self.hue)

        for fn_id in fn_idx:
            if (
                fn_id == 0
                and brightness_factor is not None
                and np.random.rand() < self.p
            ):
                data_dict["color"] = self.adjust_brightness(
                    data_dict["color"], brightness_factor
                )
            elif (
                fn_id == 1 and contrast_factor is not None and np.random.rand() < self.p
            ):
                data_dict["color"] = self.adjust_contrast(
                    data_dict["color"], contrast_factor
                )
            elif (
                fn_id == 2
                and saturation_factor is not None
                and np.random.rand() < self.p
            ):
                data_dict["color"] = self.adjust_saturation(
                    data_dict["color"], saturation_factor
                )
            elif fn_id == 3 and hue_factor is not None and np.random.rand() < self.p:
                data_dict["color"] = self.adjust_hue(data_dict["color"], hue_factor)
        return data_dict


@TRANSFORMS.register_module()
class HueSaturationTranslation(object):
    @staticmethod
    def rgb_to_hsv(rgb):
        # Translated from source of colorsys.rgb_to_hsv
        # r,g,b should be a numpy arrays with values between 0 and 255
        # rgb_to_hsv returns an array of floats between 0.0 and 1.0.
        rgb = rgb.astype("float")
        hsv = np.zeros_like(rgb)
        # in case an RGBA array was passed, just copy the A channel
        hsv[..., 3:] = rgb[..., 3:]
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        maxc = np.max(rgb[..., :3], axis=-1)
        minc = np.min(rgb[..., :3], axis=-1)
        hsv[..., 2] = maxc
        mask = maxc != minc
        hsv[mask, 1] = (maxc - minc)[mask] / maxc[mask]
        rc = np.zeros_like(r)
        gc = np.zeros_like(g)
        bc = np.zeros_like(b)
        rc[mask] = (maxc - r)[mask] / (maxc - minc)[mask]
        gc[mask] = (maxc - g)[mask] / (maxc - minc)[mask]
        bc[mask] = (maxc - b)[mask] / (maxc - minc)[mask]
        hsv[..., 0] = np.select(
            [r == maxc, g == maxc], [bc - gc, 2.0 + rc - bc], default=4.0 + gc - rc
        )
        hsv[..., 0] = (hsv[..., 0] / 6.0) % 1.0
        return hsv

    @staticmethod
    def hsv_to_rgb(hsv):
        # Translated from source of colorsys.hsv_to_rgb
        # h,s should be a numpy arrays with values between 0.0 and 1.0
        # v should be a numpy array with values between 0.0 and 255.0
        # hsv_to_rgb returns an array of uints between 0 and 255.
        rgb = np.empty_like(hsv)
        rgb[..., 3:] = hsv[..., 3:]
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        i = (h * 6.0).astype("uint8")
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        i = i % 6
        conditions = [s == 0.0, i == 1, i == 2, i == 3, i == 4, i == 5]
        rgb[..., 0] = np.select(conditions, [v, q, p, p, t, v], default=v)
        rgb[..., 1] = np.select(conditions, [v, v, v, q, p, p], default=t)
        rgb[..., 2] = np.select(conditions, [v, p, t, v, v, q], default=p)
        return rgb.astype("uint8")

    def __init__(self, hue_max=0.5, saturation_max=0.2):
        self.hue_max = hue_max
        self.saturation_max = saturation_max

    def __call__(self, data_dict):
        if "color" in data_dict.keys():
            # Assume color[:, :3] is rgb
            hsv = HueSaturationTranslation.rgb_to_hsv(data_dict["color"][:, :3])
            hue_val = (np.random.rand() - 0.5) * 2 * self.hue_max
            sat_ratio = 1 + (np.random.rand() - 0.5) * 2 * self.saturation_max
            hsv[..., 0] = np.remainder(hue_val + hsv[..., 0] + 1, 1)
            hsv[..., 1] = np.clip(sat_ratio * hsv[..., 1], 0, 1)
            data_dict["color"][:, :3] = np.clip(
                HueSaturationTranslation.hsv_to_rgb(hsv), 0, 255
            )
        return data_dict


@TRANSFORMS.register_module()
class RandomDropColor(object):
    def __init__(self, drop_ratio=0.2, drop_application_ratio=0.5):
        self.drop_ratio = drop_ratio
        self.drop_application_ratio = drop_application_ratio
        self.drop_value = 0.0

    def __call__(self, data_dict):
        if (
            "color" in data_dict.keys()
            and random.random() < self.drop_application_ratio
        ):
            n = len(data_dict["color"])
            idx = np.random.choice(n, int(n * self.drop_ratio), replace=False)
            data_dict["color"][idx] = self.drop_value
        return data_dict


@TRANSFORMS.register_module()
class RandomDropNormal(object):
    def __init__(self, drop_ratio=0.2, drop_application_ratio=0.5):
        self.drop_ratio = drop_ratio
        self.drop_application_ratio = drop_application_ratio
        self.drop_value = 0.0

    def __call__(self, data_dict):
        if (
            "normal" in data_dict.keys()
            and random.random() < self.drop_application_ratio
        ):
            n = len(data_dict["normal"])
            num_to_drop = int(n * self.drop_ratio)
            idx = np.random.choice(n, num_to_drop, replace=False)
            data_dict["normal"][idx] = self.drop_value
        return data_dict


@TRANSFORMS.register_module()
class RandomDropEcho(object):
    def __init__(self, drop_ratio=0.2, drop_application_ratio=0.5):
        self.drop_ratio = drop_ratio
        self.drop_application_ratio = drop_application_ratio
        self.drop_value = 0.0

    def __call__(self, data_dict):
        if (
            "echo" in data_dict.keys()
            and random.random() < self.drop_application_ratio
        ):
            n = len(data_dict["echo"])
            num_to_drop = int(n * self.drop_ratio)
            idx = np.random.choice(n, num_to_drop, replace=False)
            data_dict["echo"][idx] = self.drop_value
        return data_dict


@TRANSFORMS.register_module()
class RandomDropIntensity(object):
    def __init__(self, drop_ratio=0.2, drop_application_ratio=0.5):
        self.drop_ratio = drop_ratio
        self.drop_application_ratio = drop_application_ratio
        self.drop_value = 0.0

    def __call__(self, data_dict):
        if (
            "intensity" in data_dict.keys()
            and random.random() < self.drop_application_ratio
        ):
            n = len(data_dict["intensity"])
            num_to_drop = int(n * self.drop_ratio)
            idx = np.random.choice(n, num_to_drop, replace=False)
            data_dict["intensity"][idx] = self.drop_value
        return data_dict


@TRANSFORMS.register_module()
class RandomColorDrop(object):
    def __init__(self, p=0.2, color_augment=0.0):
        self.p = p
        self.color_augment = color_augment

    def __call__(self, data_dict):
        if "color" in data_dict.keys() and np.random.rand() < self.p:
            data_dict["color"] *= self.color_augment
        return data_dict

    def __repr__(self):
        return "RandomColorDrop(color_augment: {}, p: {})".format(
            self.color_augment, self.p
        )


@TRANSFORMS.register_module()
class ElasticDistortion(object):
    def __init__(self, distortion_params=None):
        self.distortion_params = (
            [[0.2, 0.4], [0.8, 1.6]] if distortion_params is None else distortion_params
        )

    @staticmethod
    def elastic_distortion(coords, granularity, magnitude):
        """
        Apply elastic distortion on sparse coordinate space.
        pointcloud: numpy array of (number of points, at least 3 spatial dims)
        granularity: size of the noise grid (in same scale[m/cm] as the voxel grid)
        magnitude: noise multiplier
        """
        blurx = np.ones((3, 1, 1, 1)).astype("float32") / 3
        blury = np.ones((1, 3, 1, 1)).astype("float32") / 3
        blurz = np.ones((1, 1, 3, 1)).astype("float32") / 3
        coords_min = coords.min(0)

        # Create Gaussian noise tensor of the size given by granularity.
        noise_dim = ((coords - coords_min).max(0) // granularity).astype(int) + 3
        noise = np.random.randn(*noise_dim, 3).astype(np.float32)

        # Smoothing.
        for _ in range(2):
            noise = scipy.ndimage.filters.convolve(
                noise, blurx, mode="constant", cval=0
            )
            noise = scipy.ndimage.filters.convolve(
                noise, blury, mode="constant", cval=0
            )
            noise = scipy.ndimage.filters.convolve(
                noise, blurz, mode="constant", cval=0
            )

        # Trilinear interpolate noise filters for each spatial dimensions.
        ax = [
            np.linspace(d_min, d_max, d)
            for d_min, d_max, d in zip(
                coords_min - granularity,
                coords_min + granularity * (noise_dim - 2),
                noise_dim,
            )
        ]
        interp = RegularGridInterpolator(
            ax, noise, bounds_error=False, fill_value=0
        )
        coords += interp(coords) * magnitude
        return coords

    def __call__(self, data_dict):
        if "coord" in data_dict.keys() and self.distortion_params is not None:
            if random.random() < 0.95:
                for granularity, magnitude in self.distortion_params:
                    data_dict["coord"] = self.elastic_distortion(
                        data_dict["coord"], granularity, magnitude
                    )
        return data_dict


@TRANSFORMS.register_module()
class GridSample(object):
    def __init__(
        self,
        grid_size=0.05,
        hash_type="fnv",
        mode="train",
        return_inverse=False,
        return_grid_coord=False,
        return_min_coord=False,
        return_displacement=False,
        project_displacement=False,
    ):
        self.grid_size = grid_size
        self.hash = self.fnv_hash_vec if hash_type == "fnv" else self.ravel_hash_vec
        assert mode in ["train", "test"]
        self.mode = mode
        self.return_inverse = return_inverse
        self.return_grid_coord = return_grid_coord
        self.return_min_coord = return_min_coord
        self.return_displacement = return_displacement
        self.project_displacement = project_displacement

    def __call__(self, data_dict):
        assert "coord" in data_dict.keys()
        scaled_coord = data_dict["coord"] / np.array(self.grid_size)
        grid_coord = np.floor(scaled_coord).astype(int)
        min_coord = grid_coord.min(0)
        grid_coord -= min_coord
        scaled_coord -= min_coord
        min_coord = min_coord * np.array(self.grid_size)
        key = self.hash(grid_coord)
        idx_sort = np.argsort(key)
        key_sort = key[idx_sort]
        _, inverse, count = np.unique(key_sort, return_inverse=True, return_counts=True)
        if self.mode == "train":  # train mode
            idx_select = (
                np.cumsum(np.insert(count, 0, 0)[0:-1])
                + np.random.randint(0, count.max(), count.size) % count
            )
            idx_unique = idx_sort[idx_select]
            if "sampled_index" in data_dict:
                # for ScanNet data efficient, we need to make sure labeled point is sampled.
                idx_unique = np.unique(
                    np.append(idx_unique, data_dict["sampled_index"])
                )
                mask = np.zeros_like(data_dict["segment"]).astype(bool)
                mask[data_dict["sampled_index"]] = True
                data_dict["sampled_index"] = np.where(mask[idx_unique])[0]
            data_dict = index_operator(data_dict, idx_unique)
            # Store grid_size for later use (e.g., Point.sparsify())
            data_dict["grid_size"] = self.grid_size
            if self.return_inverse:
                data_dict["inverse"] = np.zeros_like(inverse)
                data_dict["inverse"][idx_sort] = inverse
            if self.return_grid_coord:
                data_dict["grid_coord"] = grid_coord[idx_unique]
                if "grid_coord" not in data_dict["index_valid_keys"]:
                    data_dict["index_valid_keys"].append("grid_coord")
            if self.return_min_coord:
                data_dict["min_coord"] = min_coord.reshape([1, 3])
            if self.return_displacement:
                displacement = (
                    scaled_coord - grid_coord - 0.5
                )  # [0, 1] -> [-0.5, 0.5] displacement to center
                if self.project_displacement:
                    displacement = np.sum(
                        displacement * data_dict["normal"], axis=-1, keepdims=True
                    )
                data_dict["displacement"] = displacement[idx_unique]
                if "displacement" not in data_dict["index_valid_keys"]:
                    data_dict["index_valid_keys"].append("displacement")
            return data_dict

        elif self.mode == "test":  # test mode
            data_part_list = []
            inverse_map = None
            if self.return_inverse:
                inverse_map = np.zeros_like(inverse)
                inverse_map[idx_sort] = inverse
            for i in range(count.max()):
                idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + i % count
                idx_part = idx_sort[idx_select]
                data_part = index_operator(data_dict, idx_part, duplicate=True)
                data_part["index"] = idx_part
                # Store grid_size for later use (e.g., Point.sparsify())
                data_part["grid_size"] = self.grid_size
                if inverse_map is not None and i == 0:
                    # Only one fragment needs to carry the full inverse mapping.
                    data_part["inverse"] = inverse_map
                if self.return_grid_coord:
                    data_part["grid_coord"] = grid_coord[idx_part]
                    if "grid_coord" not in data_part["index_valid_keys"]:
                        data_part["index_valid_keys"].append("grid_coord")
                if self.return_min_coord:
                    data_part["min_coord"] = min_coord.reshape([1, 3])
                if self.return_displacement:
                    displacement = (
                        scaled_coord - grid_coord - 0.5
                    )  # [0, 1] -> [-0.5, 0.5] displacement to center
                    if self.project_displacement:
                        displacement = np.sum(
                            displacement * data_dict["normal"], axis=-1, keepdims=True
                        )
                    data_part["displacement"] = displacement[idx_part]
                    if "displacement" not in data_part["index_valid_keys"]:
                        data_part["index_valid_keys"].append("displacement")
                data_part_list.append(data_part)
            return data_part_list
        else:
            raise NotImplementedError

    @staticmethod
    def ravel_hash_vec(arr):
        """
        Ravel the coordinates after subtracting the min coordinates.
        """
        assert arr.ndim == 2
        arr = arr.copy()
        arr -= arr.min(0)
        arr = arr.astype(np.uint64, copy=False)
        arr_max = arr.max(0).astype(np.uint64) + 1

        keys = np.zeros(arr.shape[0], dtype=np.uint64)
        # Fortran style indexing
        for j in range(arr.shape[1] - 1):
            keys += arr[:, j]
            keys *= arr_max[j + 1]
        keys += arr[:, -1]
        return keys

    @staticmethod
    def fnv_hash_vec(arr):
        """
        FNV64-1A
        """
        assert arr.ndim == 2
        # Floor first for negative coordinates
        arr = arr.copy()
        arr = arr.astype(np.uint64, copy=False)
        hashed_arr = np.uint64(14695981039346656037) * np.ones(
            arr.shape[0], dtype=np.uint64
        )
        for j in range(arr.shape[1]):
            hashed_arr *= np.uint64(1099511628211)
            hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
        return hashed_arr
    
def partition_voxels_for_test(idx_sort, count, max_test_loops):
    """
    为测试模式对体素化的点云进行分区 (已修正逻辑)。

    该函数将点云根据体素进行分组，然后将每个体素内的点进行分区。
    分区的数量(即最终生成的批次数)会自适应调整，但不会超过 max_test_loops。

    Args:
        idx_sort (np.ndarray): 根据体素哈希值排序后的原始点云索引。
        count (np.ndarray): 每个唯一体素中包含的点的数量。
        max_test_loops (int): 最大的测试循环次数上限。

    Returns:
        list[np.ndarray]: 一个列表，其中每个元素都是一个用于测试批次的点索引数组。
    """
    # 1. 确定实际需要的循环次数
    #    它由最密集体素的点数决定，但不能超过设定的上限。
    if count.size == 0: # 处理完全没有点或体素的极端情况
        return []
    num_actual_loops = min(count.max(), max_test_loops)

    # 2. 将排序后的点云索引，根据其所属的体素进行分组
    points_in_voxels = np.split(idx_sort, np.cumsum(count[:-1]))

    # 3. 对每个体素内的点进行再切分
    all_chunks = []
    for voxel_points in points_in_voxels:
        # 随机打乱体素内的点
        np.random.shuffle(voxel_points)
        # 使用自适应计算出的 `num_actual_loops` 作为切分数量
        chunks = np.array_split(voxel_points, num_actual_loops)
        all_chunks.append(chunks)

    # 4. 重新组合成 `num_actual_loops` 个批次的索引列表
    chunk_counts=count
    chunk_counts[chunk_counts > num_actual_loops] = num_actual_loops
    list_of_batch_indices = []
    for i in range(num_actual_loops):
        current_batch_indices = []
        for j, voxel_chunks in enumerate(all_chunks):
            current_batch_indices.append(voxel_chunks[i%chunk_counts[j]])
        
        final_indices = np.concatenate(current_batch_indices)
        list_of_batch_indices.append(final_indices)
        
    return list_of_batch_indices


@TRANSFORMS.register_module()
class GridSample_Maxloop(GridSample):
    def __init__(
        self,
        grid_size=0.05,
        hash_type="fnv",
        mode="train",
        return_inverse=False,
        return_grid_coord=False,
        return_min_coord=False,
        return_displacement=False,
        project_displacement=False,
        max_test_loops=30
    ):
        super().__init__()
        self.grid_size = grid_size
        self.hash = self.fnv_hash_vec if hash_type == "fnv" else self.ravel_hash_vec
        assert mode in ["train", "test"]
        self.mode = mode
        self.return_inverse = return_inverse
        self.return_grid_coord = return_grid_coord
        self.return_min_coord = return_min_coord
        self.return_displacement = return_displacement
        self.project_displacement = project_displacement
        self.max_test_loops = max_test_loops

    def __call__(self, data_dict):
        assert "coord" in data_dict.keys()
        # 计算规则化坐标
        self.grid_size=self.grid_size
        scaled_coord = data_dict["coord"] / np.array(self.grid_size)
        grid_coord = np.floor(scaled_coord).astype(int)
        # 计算最小网格坐标，归一化
        min_coord = grid_coord.min(0)
        grid_coord -= min_coord
        scaled_coord -= min_coord
        min_coord = min_coord * np.array(self.grid_size)
        # 获取规则坐标哈希值并排序
        key = self.hash(grid_coord)
        idx_sort = np.argsort(key)
        key_sort = key[idx_sort]
        # 计算网格索引和点数统计
        _, inverse, count = np.unique(key_sort, return_inverse=True, return_counts=True)
        if self.mode == "train":  # train mode
            # 格网中随机采样
            idx_select = (
                np.cumsum(np.insert(count, 0, 0)[0:-1])
                + np.random.randint(0, count.max(), count.size) % count
            )
            idx_unique = idx_sort[idx_select]
            if "sampled_index" in data_dict:
                # for ScanNet data efficient, we need to make sure labeled point is sampled.
                idx_unique = np.unique(
                    np.append(idx_unique, data_dict["sampled_index"])
                )
                mask = np.zeros_like(data_dict["segment"]).astype(bool)
                mask[data_dict["sampled_index"]] = True
                data_dict["sampled_index"] = np.where(mask[idx_unique])[0]
            data_dict = index_operator(data_dict, idx_unique)
            # Store grid_size for later use (e.g., Point.sparsify())
            data_dict["grid_size"] = self.grid_size
            # 若需返回逆索引 return_inverse，记录每个点在原始数据中的归属
            if self.return_inverse:
                data_dict["inverse"] = np.zeros_like(inverse)
                data_dict["inverse"][idx_sort] = inverse
            # 记录网格坐标和最小坐标
            if self.return_grid_coord:
                data_dict["grid_coord"] = grid_coord[idx_unique]
                data_dict["index_valid_keys"].append("grid_coord")
            if self.return_min_coord:
                data_dict["min_coord"] = min_coord.reshape([1, 3])
            # 点在网格内的位置和法线上的距离
            if self.return_displacement:
                displacement = (
                    scaled_coord - grid_coord - 0.5
                )  # [0, 1] -> [-0.5, 0.5] displacement to center
                if self.project_displacement:
                    displacement = np.sum(
                        displacement * data_dict["normal"], axis=-1, keepdims=True
                    )
                data_dict["displacement"] = displacement[idx_unique]
                data_dict["index_valid_keys"].append("displacement")
            return data_dict

        elif self.mode == "test":  # test mode
            # 调用已修正的分区函数
            list_of_batch_indices = partition_voxels_for_test(
                idx_sort, count, self.max_test_loops
            )
            data_part_list = []
            inverse_map = None
            if self.return_inverse:
                inverse_map = np.zeros_like(inverse)
                inverse_map[idx_sort] = inverse
            for batch_indices in list_of_batch_indices:
                if len(batch_indices) == 0:
                    continue
                
                data_part = index_operator(data_dict, batch_indices, duplicate=True)
                data_part["index"] = batch_indices
                # Store grid_size for later use (e.g., Point.sparsify())
                data_part["grid_size"] = self.grid_size
                if inverse_map is not None and not data_part_list:
                    # Only one fragment needs to carry the full inverse mapping.
                    data_part["inverse"] = inverse_map
                if self.return_grid_coord:
                    data_part["grid_coord"] = grid_coord[batch_indices]
                    data_dict["index_valid_keys"].append("grid_coord")
                if self.return_min_coord:
                    data_part["min_coord"] = min_coord.reshape([1, 3])
                if self.return_displacement:
                    displacement = (
                        scaled_coord - grid_coord - 0.5
                    )  # [0, 1] -> [-0.5, 0.5] displacement to center
                    if self.project_displacement:
                        displacement = np.sum(
                            displacement * data_dict["normal"], axis=-1, keepdims=True
                        )
                    data_dict["displacement"] = displacement[batch_indices]
                    data_dict["index_valid_keys"].append("displacement")
                data_part_list.append(data_part)
       
            return data_part_list
        
        else:
            raise NotImplementedError


@TRANSFORMS.register_module()
class GridVoxelize(object):
    """Voxelization without sampling - for EZ-SP superpoint methods.
    
    Unlike GridSample which samples one point per voxel, this transform:
    1. Voxelizes the point cloud into a grid
    2. Keeps ALL original points (no sampling)
    3. Creates voxel representations and point-to-voxel mapping
    4. Enables voxel-based feature extraction + point-level partition
    
    This is critical for superpoint methods like EZ-SP where we need:
    - Fast sparse convolution on voxels
    - Complete point cloud for accurate partition boundaries
    
    Args:
        grid_size: Voxel size for quantization
        hash_type: Hash function ('fnv' or 'ravel')
        mode: 'train' or 'test' 
        aggregation: How to aggregate point features in each voxel
                     'mean', 'max', or 'first' (default: 'mean')
        return_voxel_coord: Whether to return voxel center coordinates
        return_inverse: Whether to return point→voxel mapping (required for decoder)
        return_grid_coord: Whether to return grid coordinates
        
    Returns (in data_dict):
        coord: [N, 3] - Original point coordinates (unchanged)
        feat: [N, C] - Original point features (unchanged)
        voxel_coord: [M, 3] - Voxel center coordinates
        voxel_feat: [M, C] - Aggregated voxel features
        inverse: [N] - Point-to-voxel mapping (point_i → voxel_id)
        grid_coord: [N, 3] - Grid coordinates for each point
    """
    
    def __init__(
        self,
        grid_size=0.05,
        hash_type="fnv",
        mode="train",
        aggregation="mean",
        return_voxel_coord=True,
        return_inverse=True,
        return_grid_coord=True,
        return_min_coord=False,
        feat_keys=None,  # NEW: Specify which keys to aggregate into voxel_feat
    ):
        self.grid_size = grid_size
        self.hash = self.fnv_hash_vec if hash_type == "fnv" else self.ravel_hash_vec
        assert mode in ["train", "test"]
        self.mode = mode
        assert aggregation in ["mean", "max", "first"]
        self.aggregation = aggregation
        self.return_voxel_coord = return_voxel_coord
        self.return_inverse = return_inverse
        self.return_grid_coord = return_grid_coord
        self.return_min_coord = return_min_coord
        self.feat_keys = feat_keys  # List of keys to concat as features

    def __call__(self, data_dict):
        assert "coord" in data_dict.keys()
        
        # Build features from feat_keys if specified (like Collect does)
        if self.feat_keys is not None:
            feat = np.concatenate([data_dict[key].astype(np.float32) for key in self.feat_keys], axis=1)
        elif "feat" in data_dict.keys():
            feat = data_dict["feat"]
        else:
            feat = None  # No features to aggregate
        
        # 1. Compute grid coordinates
        scaled_coord = data_dict["coord"] / np.array(self.grid_size)
        grid_coord = np.floor(scaled_coord).astype(int)
        min_coord = grid_coord.min(0)
        grid_coord -= min_coord
        scaled_coord -= min_coord
        min_coord = min_coord * np.array(self.grid_size)
        
        # 2. Hash grid coordinates to find unique voxels
        key = self.hash(grid_coord)
        idx_sort = np.argsort(key)
        key_sort = key[idx_sort]
        unique_keys, inverse_sorted, count = np.unique(
            key_sort, return_inverse=True, return_counts=True
        )
        
        # 3. Create point→voxel mapping (inverse)
        # inverse[i] tells which voxel point i belongs to
        inverse = np.zeros_like(inverse_sorted)
        inverse[idx_sort] = inverse_sorted
        
        # 4. Find one representative point per voxel (for coord/features)
        idx_voxel_representatives = np.cumsum(np.insert(count, 0, 0)[:-1])
        idx_voxel_points = idx_sort[idx_voxel_representatives]
        
        # 5. Compute voxel center coordinates
        voxel_grid_coord = grid_coord[idx_voxel_points]
        voxel_coord = (voxel_grid_coord + 0.5) * self.grid_size + min_coord
        
        # 6. Aggregate point features into voxel features (if features available)
        if feat is not None:
            voxel_feat = self._aggregate_features(feat, inverse, len(unique_keys))
            data_dict["voxel_feat"] = voxel_feat
            if "voxel_feat" not in data_dict.get("index_valid_keys", []):
                if "index_valid_keys" not in data_dict:
                    data_dict["index_valid_keys"] = []
                data_dict["index_valid_keys"].append("voxel_feat")
        # else: No features available, voxel_feat won't be created
        
        # 7. Store voxelization results
        data_dict["grid_size"] = self.grid_size
        
        if self.return_voxel_coord:
            data_dict["voxel_coord"] = voxel_coord
            if "voxel_coord" not in data_dict.get("index_valid_keys", []):
                if "index_valid_keys" not in data_dict:
                    data_dict["index_valid_keys"] = []
                data_dict["index_valid_keys"].append("voxel_coord")
        
        if self.return_inverse:
            data_dict["inverse"] = inverse
            if "inverse" not in data_dict.get("index_valid_keys", []):
                if "index_valid_keys" not in data_dict:
                    data_dict["index_valid_keys"] = []
                data_dict["index_valid_keys"].append("inverse")
        
        if self.return_grid_coord:
            data_dict["grid_coord"] = grid_coord
            if "grid_coord" not in data_dict.get("index_valid_keys", []):
                if "index_valid_keys" not in data_dict:
                    data_dict["index_valid_keys"] = []
                data_dict["index_valid_keys"].append("grid_coord")
        
        if self.return_min_coord:
            data_dict["min_coord"] = min_coord.reshape([1, 3])
        
        # Store original point count for verification
        data_dict["num_raw_points"] = len(data_dict["coord"])
        data_dict["num_voxels"] = len(unique_keys)
        
        return data_dict
    
    def _aggregate_features(self, feat, inverse, num_voxels):
        """Aggregate point features into voxel features (VECTORIZED - no for loops).
        
        Args:
            feat: [N, C] point features
            inverse: [N] point-to-voxel mapping
            num_voxels: M number of unique voxels
            
        Returns:
            voxel_feat: [M, C] aggregated voxel features
        """
        N, C = feat.shape
        voxel_feat = np.zeros((num_voxels, C), dtype=feat.dtype)
        
        if self.aggregation == "mean":
            # Mean pooling (already vectorized)
            np.add.at(voxel_feat, inverse, feat)
            count = np.bincount(inverse, minlength=num_voxels).reshape(-1, 1)
            voxel_feat = voxel_feat / np.maximum(count, 1)
        
        elif self.aggregation == "max":
            # Max pooling (VECTORIZED using np.maximum.at)
            # Initialize with -inf so empty voxels stay at -inf (then we can fill with 0)
            voxel_feat.fill(-np.inf)
            np.maximum.at(voxel_feat, inverse, feat)
            # Replace -inf with 0 for empty voxels (if any)
            voxel_feat[np.isinf(voxel_feat)] = 0
        
        elif self.aggregation == "first":
            # First point in each voxel (VECTORIZED)
            # Sort by point index, then use unique to find first occurrence
            sort_idx = np.argsort(inverse, kind='stable')  # stable preserves order for same voxel
            sorted_inverse = inverse[sort_idx]
            sorted_feat = feat[sort_idx]
            
            # Find first occurrence of each voxel
            _, first_idx = np.unique(sorted_inverse, return_index=True)
            voxel_feat[sorted_inverse[first_idx]] = sorted_feat[first_idx]
        
        return voxel_feat

    @staticmethod
    def ravel_hash_vec(arr):
        """Ravel the coordinates after subtracting the min coordinates."""
        assert arr.ndim == 2
        arr = arr.copy()
        arr -= arr.min(0)
        arr = arr.astype(np.uint64, copy=False)
        arr_max = arr.max(0).astype(np.uint64) + 1

        keys = np.zeros(arr.shape[0], dtype=np.uint64)
        # Fortran style indexing
        for j in range(arr.shape[1] - 1):
            keys += arr[:, j]
            keys *= arr_max[j + 1]
        keys += arr[:, -1]
        return keys

    @staticmethod
    def fnv_hash_vec(arr):
        """FNV64-1A"""
        assert arr.ndim == 2
        arr = arr.copy()
        arr = arr.astype(np.uint64, copy=False)
        hashed_arr = np.uint64(14695981039346656037) * np.ones(
            arr.shape[0], dtype=np.uint64
        )
        for j in range(arr.shape[1]):
            hashed_arr *= np.uint64(1099511628211)
            hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
        return hashed_arr


@TRANSFORMS.register_module()
class SaveNodeIndex(object):
    """Save node indices for tracking points through voxelization.
    
    This is the CRITICAL first step of the official EZ-SP pipeline:
    1. SaveNodeIndex('sub') → saves [0, 1, 2, ..., N-1] into data['sub']
    2. GridSampling3D → voxelizes points, converts 'sub' to Cluster object
    3. GreedyPartition → creates NAG[L0=sub, L1=voxels, L2+=superpoints]
    
    By storing indices BEFORE voxelization, we can later backtrack:
    - Which raw points belong to which voxel (via Cluster)
    - Propagate predictions from superpoints back to raw points
    
    Following: src/transforms/sampling.py:SaveNodeIndex
    
    Args:
        key: str - Attribute name to store indices (default: 'sub')
        
    Example:
        >>> transform = SaveNodeIndex(key='sub')
        >>> data_dict = {'coord': np.random.randn(1000, 3)}
        >>> data_dict = transform(data_dict)
        >>> data_dict['sub']
        array([0, 1, 2, ..., 999])
    """
    DEFAULT_KEY = 'sub'
    
    def __init__(self, key: str = 'sub'):
        self.key = key
    
    def __call__(self, data_dict):
        """Save point indices."""
        assert "coord" in data_dict, "coord must exist before SaveNodeIndex"
        num_points = len(data_dict["coord"])
        data_dict[self.key] = np.arange(num_points, dtype=np.int64)
        
        # Mark as index-valid for proper handling
        if "index_valid_keys" not in data_dict:
            data_dict["index_valid_keys"] = []
        if self.key not in data_dict["index_valid_keys"]:
            data_dict["index_valid_keys"].append(self.key)
        
        return data_dict


@TRANSFORMS.register_module()
class GridSampling3D(object):
    """Voxel grid sampling following official EZ-SP implementation.
    
    This is the CORE voxelization transform that:
    1. Clusters points into voxels based on grid coordinates
    2. Aggregates point features (mean/last)
    3. CRITICALLY: Converts 'sub' (saved indices) to Cluster format
    
    The Cluster object preserves the mapping from voxels → raw points,
    enabling final prediction propagation: superpoints → voxels → raw points
    
    Data flow:
        Input:  N raw points with sub=[0,1,...,N-1]
        Output: M voxels with sub=Cluster(pointer, value)
                - pointer[i:i+1] gives range in value array for voxel i
                - value contains original point indices
    
    Following: src/transforms/sampling.py:GridSampling3D
    
    Performance Optimizations:
        - FNV64-style hashing for voxel grouping
        - Stable mergesort grouping
        - Early exit for empty histograms
        - Vectorized aggregation (bincount / add.at)
    
    Args:
        size: float - Voxel size (meters)
        mode: str - 'mean' for average, 'last' for random point
        hist_key: str|List - Keys to convert to histogram (e.g., 'y' for labels)
        hist_size: int|List - Histogram bins (e.g., num_classes+1)
        quantize_coords: bool - Store integer grid coordinates
        feat_keys: List[str] - Keys to aggregate into features
        
    Returns:
        data_dict with voxelized data:
            - coord: [M, 3] voxel center coordinates
            - feat: [M, C] aggregated features
            - sub: Cluster object mapping voxels→raw points
            - y: [M, num_classes+1] label histogram (if hist_key='y')
            - grid_size: float (CRITICAL: required by Point.sparsify())
    """
    
    # Keys that get converted to Cluster objects (point→voxel backtracking)
    _CLUSTER_KEYS = ['sub']
    
    # Keys that use majority voting
    _VOTING_KEYS = ['super_index', 'is_val']
    
    # Keys that use 'last' mode regardless of global mode
    _LAST_KEYS = ['batch']
    
    def __init__(
        self,
        size: float = 0.1,
        mode: str = "mean",
        hist_key=None,
        hist_size=None,
        quantize_coords: bool = True,
        feat_keys=None,
    ):
        self.size = size
        self.mode = mode
        assert mode in ["mean", "last"], f"mode must be 'mean' or 'last', got {mode}"
        
        # Histogram configuration (for label aggregation)
        hist_key = [] if hist_key is None else hist_key
        hist_size = [] if hist_size is None else hist_size
        hist_key = [hist_key] if isinstance(hist_key, str) else list(hist_key)
        hist_size = [hist_size] if isinstance(hist_size, int) else list(hist_size)
        assert len(hist_key) == len(hist_size), "hist_key and hist_size must match"
        self.bins = {k: v for k, v in zip(hist_key, hist_size)}
        
        self.quantize_coords = quantize_coords
        self.feat_keys = feat_keys
    
    @staticmethod
    def _fnv_hash(arr):
        """FNV64-1A hash for grid coordinates."""
        assert arr.ndim == 2
        arr = arr.astype(np.uint64)
        hashed = np.uint64(14695981039346656037) * np.ones(len(arr), dtype=np.uint64)
        for j in range(arr.shape[1]):
            hashed *= np.uint64(1099511628211)
            hashed = np.bitwise_xor(hashed, arr[:, j])
        return hashed

    def __call__(self, data_dict):
        assert "coord" in data_dict, "coord is required"
        coord = data_dict["coord"]
        num_points = len(coord)
        
        # 1. Compute grid coordinates - optimized
        grid_coord = np.floor(coord / self.size).astype(np.int64)
        min_grid = grid_coord.min(axis=0, keepdims=True)  # Keep dims for broadcasting
        grid_coord_offset = grid_coord - min_grid  # Broadcasting subtraction
        
        # 2. Hash to find unique voxels (faster path for this project)
        key = self._fnv_hash(grid_coord_offset)
        sort_idx = np.argsort(key, kind="mergesort")
        key_sorted = key[sort_idx]
        _, inverse_sorted, counts = np.unique(
            key_sorted, return_inverse=True, return_counts=True
        )
        num_voxels = len(counts)

        # Inverse mapping: point_i → voxel_id
        inverse = np.empty(num_points, dtype=np.int64)
        inverse[sort_idx] = inverse_sorted

        # Representative point per voxel (first in sorted order)
        first_idx = np.concatenate(([0], np.cumsum(counts[:-1])))
        unique_pos_indices = sort_idx[first_idx]
        
        # 3. Build output data_dict
        out_dict = {}
        
        # Store metadata (CRITICAL: grid_size required by Point.sparsify())
        out_dict["num_raw_points"] = num_points
        out_dict["num_voxels"] = num_voxels
        out_dict["grid_size"] = self.size  # Will be re-ensured at end
        
        # Voxel center coordinates - optimized indexing
        voxel_grid_coord = grid_coord[unique_pos_indices]
        voxel_coord = (voxel_grid_coord.astype(np.float32) + 0.5) * self.size  # Combined ops
        out_dict["coord"] = voxel_coord
        
        if self.quantize_coords:
            out_dict["grid_coord"] = (voxel_grid_coord - min_grid[0]).astype(np.int32)  # Remove keepdim
        
        # 4. Process each attribute
        index_valid_keys = ["coord"]
        if self.quantize_coords:
            index_valid_keys.append("grid_coord")
        
        # Keys ending with '_raw' should be preserved (not aggregated)
        # They store original point-level data for evaluation
        raw_keys_to_preserve = {}
        
        for key_name, item in data_dict.items():
            if key_name in ["coord", "index_valid_keys", "num_raw_points", "num_voxels", "grid_size"]:
                continue
            
            # Preserve keys ending with '_raw' - they store original point-level data
            if key_name.endswith("_raw"):
                raw_keys_to_preserve[key_name] = item
                continue
            
            # CRITICAL: Cluster keys (sub) → create Cluster mapping
            if key_name in self._CLUSTER_KEYS:
                if isinstance(item, np.ndarray) and item.ndim == 1:
                    # Create Cluster object (CSR format)
                    # pointer: [M+1], cumsum of counts
                    # value: original point indices sorted by voxel
                    pointer = np.concatenate([[0], np.cumsum(counts)])
                    value = item[sort_idx]  # Point indices in voxel order
                    out_dict[key_name] = {"pointer": pointer, "value": value}
                    # Note: Will be converted to Cluster object in ToTensor or collate
                continue
            
            # Histogram keys (y) → aggregate to histogram
            if key_name in self.bins:
                hist_size = self.bins[key_name]
                if isinstance(item, np.ndarray) and item.ndim == 1:
                    # OPTIMIZED histogram aggregation
                    # Filter valid labels once
                    valid_mask = (item >= 0) & (item < hist_size)
                    
                    if valid_mask.any():
                        valid_voxels = inverse[valid_mask]
                        valid_labels = item[valid_mask].astype(np.int64)
                        
                        # Use flat index: voxel_id * hist_size + label
                        flat_idx = valid_voxels * hist_size + valid_labels
                        
                        # Count occurrences using bincount (fastest method)
                        y_hist_flat = np.bincount(flat_idx, minlength=num_voxels * hist_size)
                        y_hist = y_hist_flat.reshape(num_voxels, hist_size).astype(np.float32)
                    else:
                        # No valid labels, create zero histogram
                        y_hist = np.zeros((num_voxels, hist_size), dtype=np.float32)
                    
                    out_dict[key_name] = y_hist
                    index_valid_keys.append(key_name)
                continue
            
            # Standard tensor aggregation
            if not isinstance(item, np.ndarray):
                out_dict[key_name] = item
                continue
            
            if item.shape[0] != num_points:
                out_dict[key_name] = item  # Not point-indexed
                continue
            
            # Voting keys (majority voting) - OPTIMIZED
            if key_name in self._VOTING_KEYS:
                # Use bincount for mode calculation (most frequent value)
                max_val = int(item.max()) + 1
                # Create flat index: voxel_id * max_val + value
                flat_idx = inverse * max_val + item.astype(np.int64)
                counts_flat = np.bincount(flat_idx, minlength=num_voxels * max_val)
                counts_2d = counts_flat.reshape(num_voxels, max_val)
                agg = counts_2d.argmax(axis=1).astype(item.dtype)
                out_dict[key_name] = agg
                index_valid_keys.append(key_name)
                continue
            
            # Last keys or 'last' mode
            if key_name in self._LAST_KEYS or self.mode == 'last':
                out_dict[key_name] = item[unique_pos_indices]
                index_valid_keys.append(key_name)
                continue
            
            # Mean aggregation (default for 'mean' mode)
            if self.mode == 'mean':
                if item.ndim == 1:
                    agg = np.zeros(num_voxels, dtype=np.float32)
                    np.add.at(agg, inverse, item.astype(np.float32))
                    agg /= np.maximum(counts, 1)
                else:
                    agg = np.zeros((num_voxels, item.shape[1]), dtype=np.float32)
                    np.add.at(agg, inverse, item.astype(np.float32))
                    agg /= np.maximum(counts, 1).reshape(-1, 1)
                out_dict[key_name] = agg
                index_valid_keys.append(key_name)
        
        # Build features if feat_keys specified
        if self.feat_keys:
            feat_list = []
            for fk in self.feat_keys:
                if fk in out_dict:
                    f = out_dict[fk]
                    if f.ndim == 1:
                        f = f.reshape(-1, 1)
                    feat_list.append(f.astype(np.float32))
            if feat_list:
                out_dict["feat"] = np.concatenate(feat_list, axis=1)
                index_valid_keys.append("feat")
        
        out_dict["index_valid_keys"] = index_valid_keys
        
        # Store inverse mapping for potential use (but main mechanism is 'sub' Cluster)
        out_dict["voxel_inverse"] = inverse
        
        # Restore preserved raw keys (original point-level data for evaluation)
        # CRITICAL: Do this BEFORE ensuring grid_size to avoid overwriting metadata
        out_dict.update(raw_keys_to_preserve)
        
        # CRITICAL FIX: Ensure grid_size is always present (required by Point.sparsify())
        # Even if it was in data_dict originally, ensure output has it
        out_dict["grid_size"] = self.size
        
        return out_dict


@TRANSFORMS.register_module()
class SphereCrop(object):
    def __init__(self, point_max=80000, sample_rate=None, mode="random"):
        self.point_max = point_max
        self.sample_rate = sample_rate
        assert mode in ["random", "center", "all", "given"]
        self.mode = mode

    def __call__(self, data_dict):
        point_max = (
            int(self.sample_rate * data_dict["coord"].shape[0])
            if self.sample_rate is not None
            else self.point_max
        )

        assert "coord" in data_dict.keys()
        if data_dict["coord"].shape[0] > point_max:
            if self.mode == "random":
                center = data_dict["coord"][
                    np.random.randint(data_dict["coord"].shape[0])
                ]
            elif self.mode == "center":
                center = data_dict["coord"][data_dict["coord"].shape[0] // 2]
            elif self.mode == "given":
                given_index = data_dict["correspondence"].reshape(
                    data_dict["correspondence"].shape[0], -1
                )
                given_index = np.all(
                    given_index != np.ones_like(given_index[0]) * -1, axis=1
                )
                given_coord = data_dict["coord"][given_index]
                if given_coord.shape[0] == 0:
                    center = data_dict["coord"][
                        np.random.randint(data_dict["coord"].shape[0])
                    ]
                else:
                    center = np.mean(given_coord, axis=0)
            else:
                raise NotImplementedError
            idx_crop = np.argsort(np.sum(np.square(data_dict["coord"] - center), 1))[
                :point_max
            ]
            data_dict = index_operator(data_dict, idx_crop)
        return data_dict


@TRANSFORMS.register_module()
class ShufflePoint(object):
    def __call__(self, data_dict):
        assert "coord" in data_dict.keys()
        shuffle_index = np.arange(data_dict["coord"].shape[0])
        np.random.shuffle(shuffle_index)
        data_dict = index_operator(data_dict, shuffle_index)
        return data_dict


@TRANSFORMS.register_module()
class CropBoundary(object):
    def __call__(self, data_dict):
        assert "segment" in data_dict
        segment = data_dict["segment"].flatten()
        mask = (segment != 0) * (segment != 1)
        data_dict = index_operator(data_dict, mask)
        return data_dict


@TRANSFORMS.register_module()
class ContrastiveViewsGenerator(object):
    def __init__(
        self,
        view_keys=("coord", "color", "normal", "origin_coord"),
        view_trans_cfg=None,
    ):
        self.view_keys = view_keys
        self.view_trans = Compose(view_trans_cfg)

    def __call__(self, data_dict):
        view1_dict = dict()
        view2_dict = dict()
        for key in self.view_keys:
            view1_dict[key] = data_dict[key].copy()
            view2_dict[key] = data_dict[key].copy()
        view1_dict = self.view_trans(view1_dict)
        view2_dict = self.view_trans(view2_dict)
        for key, value in view1_dict.items():
            data_dict["view1_" + key] = value
        for key, value in view2_dict.items():
            data_dict["view2_" + key] = value
        return data_dict


@TRANSFORMS.register_module()
class MultiViewGenerator(object):
    def __init__(
        self,
        global_view_num=2,
        global_view_scale=(0.4, 1.0),
        local_view_num=4,
        local_view_scale=(0.1, 0.4),
        global_shared_transform=None,
        global_transform=None,
        local_transform=None,
        max_size=65536,
        enc2d_max_size=102400,
        enc2d_scale=(0.8, 1),
        center_height_scale=(0, 1),
        shared_global_view=False,
        view_keys=("coord", "origin_coord", "color", "normal", "correspondence"),
        static_view_keys=("name", "img_num"),
    ):
        self.global_view_num = global_view_num
        self.global_view_scale = global_view_scale
        self.local_view_num = local_view_num
        self.local_view_scale = local_view_scale
        self.global_shared_transform = Compose(global_shared_transform)
        self.global_transform = Compose(global_transform)
        self.local_transform = Compose(local_transform)
        self.max_size = max_size
        self.enc2d_max_size = enc2d_max_size
        self.enc2d_scale = enc2d_scale
        self.center_height_scale = center_height_scale
        self.shared_global_view = shared_global_view
        self.view_keys = view_keys
        self.static_view_keys = static_view_keys
        assert "coord" in view_keys

    def get_view(self, point, center, scale, if_enc2d=False):
        coord = point["coord"]
        max_size = min(self.max_size, coord.shape[0])
        enc2d_max_size = min(self.enc2d_max_size, coord.shape[0])
        size = 0
        for _ in range(10):
            if if_enc2d:
                size = enc2d_max_size
            else:
                size = int(np.random.uniform(*scale) * max_size)
            if size > 0:
                break
        if size == 0:
            size = max(10, scale[-1] * max_size)
        assert size > 0
        index = np.argsort(np.sum(np.square(coord - center), axis=-1))[:size]
        view = dict(index=index)
        for key in point.keys():
            if key in self.view_keys:
                view[key] = point[key][index]
            if key in self.static_view_keys:
                view[key] = point[key]
        if "index_valid_keys" in point.keys():
            # inherit index_valid_keys from point
            view["index_valid_keys"] = point["index_valid_keys"]
        return view

    @staticmethod
    def match_point_image(major_view, data_dict):
        major_correspondence = major_view["correspondence"].transpose(1, 0, 2)
        correspondence = data_dict["correspondence"].transpose(1, 0, 2)
        is_all_neg1 = np.any(major_correspondence != np.array([-1, -1]), axis=(1, 2))
        indices = np.where(is_all_neg1)[0]
        img_dict = {
            "images": data_dict["images"][indices],
            "img_num": indices.shape[0],
            "major_correspondence": major_correspondence[indices].transpose(1, 0, 2),
            "correspondence": correspondence[indices].transpose(1, 0, 2),
        }
        return img_dict

    def __call__(self, data_dict):
        coord = data_dict["coord"]
        point = self.global_shared_transform(copy.deepcopy(data_dict))
        z_min = coord[:, 2].min()
        z_max = coord[:, 2].max()
        z_min_ = z_min + (z_max - z_min) * self.center_height_scale[0]
        z_max_ = z_min + (z_max - z_min) * self.center_height_scale[1]
        if "correspondence" not in data_dict.keys():
            center_mask = np.logical_and(coord[:, 2] >= z_min_, coord[:, 2] <= z_max_)
            major_center = coord[np.random.choice(np.where(center_mask)[0])]
            major_view = self.get_view(point, major_center, self.global_view_scale)
        else:
            given_index = data_dict["correspondence"].reshape(
                data_dict["correspondence"].shape[0], -1
            )
            given_index = np.all(
                given_index != np.ones_like(given_index[0]) * -1, axis=1
            )
            given_coord = data_dict["coord"][given_index]
            if given_coord.shape[0] == 0:
                center_mask = np.logical_and(
                    coord[:, 2] >= z_min_, coord[:, 2] <= z_max_
                )
                major_center = coord[np.random.choice(np.where(center_mask)[0])]
            else:
                major_center = np.mean(given_coord, axis=0)
            major_view = self.get_view(
                point, major_center, self.global_view_scale, if_enc2d=True
            )
            img_dict = self.match_point_image(major_view, data_dict)
            major_view["correspondence"] = img_dict["major_correspondence"]
            data_dict["correspondence"] = img_dict["correspondence"]
            point["correspondence"] = img_dict["correspondence"]
            data_dict["img_num"] = img_dict["img_num"]
            data_dict["images"] = img_dict["images"]
        major_coord = major_view["coord"]

        # get global views: restrict the center of left global view within the major global view
        if not self.shared_global_view:
            global_views = [
                self.get_view(
                    point=point,
                    center=major_coord[np.random.randint(major_coord.shape[0])],
                    scale=self.global_view_scale,
                )
                for _ in range(self.global_view_num - 1)
            ]
        else:
            global_views = [
                {key: value.copy() for key, value in major_view.items()}
                for _ in range(self.global_view_num - 1)
            ]

        global_views = [major_view] + global_views

        # get local views: restrict the center of local view within the major global view
        cover_mask = np.zeros_like(major_view["index"], dtype=bool)
        local_views = []
        for i in range(self.local_view_num):
            if sum(~cover_mask) == 0:
                # reset cover mask if all points are sampled
                cover_mask[:] = False
            local_view = self.get_view(
                point=data_dict,
                center=major_coord[np.random.choice(np.where(~cover_mask)[0])],
                scale=self.local_view_scale,
            )
            local_views.append(local_view)
            cover_mask[np.isin(major_view["index"], local_view["index"])] = True

        # augmentation and concat
        view_dict = {}
        for global_view in global_views:
            global_view.pop("index")
            global_view = self.global_transform(global_view)
            for key in self.view_keys:
                if f"global_{key}" in view_dict.keys():
                    view_dict[f"global_{key}"].append(global_view[key])
                else:
                    view_dict[f"global_{key}"] = [global_view[key]]
        view_dict["global_offset"] = np.cumsum(
            [data.shape[0] for data in view_dict["global_coord"]]
        )
        for local_view in local_views:
            local_view.pop("index")
            local_view = self.local_transform(local_view)
            for key in self.view_keys:
                if f"local_{key}" in view_dict.keys():
                    view_dict[f"local_{key}"].append(local_view[key])
                else:
                    view_dict[f"local_{key}"] = [local_view[key]]
        view_dict["local_offset"] = np.cumsum(
            [data.shape[0] for data in view_dict["local_coord"]]
        )

        for key in view_dict.keys():
            if "offset" not in key:
                if key in self.static_view_keys:
                    view_dict[key] = view_dict[key]
                else:
                    view_dict[key] = np.concatenate(view_dict[key], axis=0)
        data_dict.update(view_dict)
        return data_dict


@TRANSFORMS.register_module()
class InstanceParser(object):
    def __init__(self, segment_ignore_index=(-1, 0, 1), instance_ignore_index=-1):
        self.segment_ignore_index = segment_ignore_index
        self.instance_ignore_index = instance_ignore_index

    def __call__(self, data_dict):
        coord = data_dict["coord"]
        segment = data_dict["segment"]
        instance = data_dict["instance"]
        mask = ~np.in1d(segment, self.segment_ignore_index)
        # mapping ignored instance to ignore index
        instance[~mask] = self.instance_ignore_index
        # reorder left instance
        unique, inverse = np.unique(instance[mask], return_inverse=True)
        instance_num = len(unique)
        instance[mask] = inverse
        # init instance information
        centroid = np.ones((coord.shape[0], 3)) * self.instance_ignore_index
        bbox = np.ones((instance_num, 8)) * self.instance_ignore_index
        vacancy = [
            index for index in self.segment_ignore_index if index >= 0
        ]  # vacate class index

        for instance_id in range(instance_num):
            mask_ = instance == instance_id
            coord_ = coord[mask_]
            bbox_min = coord_.min(0)
            bbox_max = coord_.max(0)
            bbox_centroid = coord_.mean(0)
            bbox_center = (bbox_max + bbox_min) / 2
            bbox_size = bbox_max - bbox_min
            bbox_theta = np.zeros(1, dtype=coord_.dtype)
            bbox_class = np.array([segment[mask_][0]], dtype=coord_.dtype)
            # shift class index to fill vacate class index caused by segment ignore index
            bbox_class -= np.greater(bbox_class, vacancy).sum()

            centroid[mask_] = bbox_centroid
            bbox[instance_id] = np.concatenate(
                [bbox_center, bbox_size, bbox_theta, bbox_class]
            )  # 3 + 3 + 1 + 1 = 8
        data_dict["instance"] = instance
        data_dict["instance_centroid"] = centroid
        data_dict["bbox"] = bbox
        return data_dict


class Compose(object):
    def __init__(self, cfg=None):
        self.cfg = cfg if cfg is not None else []
        self.transforms = []
        for t_cfg in self.cfg:
            self.transforms.append(TRANSFORMS.build(t_cfg))

    def __call__(self, data_dict):
        for t in self.transforms:
            data_dict = t(data_dict)
        return data_dict


@TRANSFORMS.register_module()
class ImgToTensor(object):
    def __init__(self):
        self.totensor = transforms.ToTensor()

    def __call__(self, img):
        return self.totensor(img)


@TRANSFORMS.register_module()
class ImgGaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(
        self, *, p: float = 0.5, radius_min: float = 0.1, radius_max: float = 2.0
    ):
        # NOTE: torchvision is applying 1 - probability to return the original image
        self.p = p
        self.transform = transforms.GaussianBlur(
            kernel_size=9, sigma=(radius_min, radius_max)
        )
        super().__init__()

    def __call__(self, img):
        if np.random.rand() < self.p:
            img = self.transform(img)
        return img


@TRANSFORMS.register_module()
class ImgChromaticJitter(object):
    def __init__(self, p=0.95, std=0.005):
        self.p = p
        self.std = std

    def __call__(self, img):
        if np.random.rand() < self.p:
            noise = torch.rand(3)
            noise *= self.std
            noise = noise[:, None, None].expand_as(img)
            img += noise
            img = torch.clip(img, 0, 1)
        return img


@TRANSFORMS.register_module()
class ImgPixelContrast(object):
    def __init__(self, threshold, p=0.2):
        super().__init__()
        self.p = p
        self.threshold = threshold

    def __call__(self, img):
        if np.random.rand() < self.p:
            n, h, w = img.shape[0], img.shape[2], img.shape[3]
            num_pixels = int(self.threshold * h * w * n)
            indices = torch.randint(0, n * h * w, (num_pixels,))
            img = img.permute(0, 2, 3, 1).reshape(-1, 3)
            img[indices, :] = 255.0 - img[indices, :]
            img = img.reshape(n, h, w, 3).permute(0, 3, 1, 2)
        return img


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


@TRANSFORMS.register_module()
class Imgnormalize(object):
    def __init__(self, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD):
        super().__init__()
        self.normalize = transforms.Normalize(mean=mean, std=std)

    def __call__(self, img):
        return self.normalize(img)


@TRANSFORMS.register_module()
class ImgRandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.imgrandomhorizontalflip = transforms.RandomHorizontalFlip(p=p)

    def __call__(self, img):
        return self.imgrandomhorizontalflip(img)


@TRANSFORMS.register_module()
class ImgRandomResizedCrop(object):
    def __init__(self, size, scale, interpolation):
        super().__init__()
        self.imgrandomresizedcrop = transforms.RandomResizedCrop(
            size=size, scale=scale, interpolation=interpolation
        )

    def __call__(self, img):
        return self.imgrandomresizedcrop(img)


@TRANSFORMS.register_module()
class ImgRandomColorJitter(object):
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8):
        colorjitter = transforms.ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
        )
        super().__init__()
        self.p = p
        self.colorjitter = colorjitter

    def __call__(self, img):
        return self.colorjitter(img)


@TRANSFORMS.register_module()
class ImgRandomGrayscale(object):
    def __init__(self, p=0.1):
        super().__init__()
        self.p = p
        self.imgrandomgrayscale = transforms.RandomGrayscale(p=p)

    def __call__(self, img):
        return self.imgrandomgrayscale(img)


@TRANSFORMS.register_module()
class ImgRandomSolarize(object):
    def __init__(self, threshold, p=0.1):
        super().__init__()
        self.p = p
        self.imgrandomsolarize = transforms.RandomSolarize(threshold=threshold, p=p)

    def __call__(self, img):
        return self.imgrandomsolarize(img)


@TRANSFORMS.register_module()
class ImgAugmentation(object):
    def __init__(
        self,
        imgtransforms,
        crop_h=518,
        crop_w=518,
        patch_h=37,
        patch_w=37,
        patch_size=14,
    ):
        self.transforms = []
        self.transforms_cfg = imgtransforms
        for t_cfg in self.transforms_cfg:
            self.transforms.append(TRANSFORMS.build(t_cfg))
        self.crop_h = crop_h
        self.crop_w = crop_w
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.patch_size = patch_size
        self.crop_start = [
            random.randint(0, patch_h * patch_size - crop_h),
            random.randint(0, patch_w * patch_size - crop_w),
        ]

    def __call__(self, point):
        point["images"] = transforms.functional.crop(
            point["images"],
            top=self.crop_start[0],
            left=self.crop_start[1],
            height=self.crop_h,
            width=self.crop_w,
        )
        for id, t in enumerate(self.transforms):
            point["images"] = t(point["images"])
        correspondence = point["correspondence"]
        correspondence_shape = correspondence.shape
        correspondence = correspondence.reshape(-1, 2)
        mask = (
            (self.crop_start[0] <= correspondence[:, 0])
            & (correspondence[:, 0] < self.crop_start[0] + self.crop_h)
            & (self.crop_start[1] <= correspondence[:, 1])
            & (correspondence[:, 1] < self.crop_start[1] + self.crop_w)
        )
        correspondence[~mask] = np.array([-1, -1])
        correspondence[mask] -= np.array(self.crop_start)
        point["correspondence"] = correspondence.reshape(correspondence_shape)
        return point

@TRANSFORMS.register_module()
class TerrainImplicitSampler(object):
    """
    针对连续地形隐式重建(Continuous DEM)的多策略混合数据采样器。
    该模块会将输入的点云划分为 Support(支撑点，用于提取特征) 和 Query(查询点，用于监督隐式网络)。
    """
    def __init__(self,
                 random_ratio=0.1,             # 策略1: 纯随机抽取的比例
                 feature_ratio=0.1,            # 策略2: 基于地形特征(山脊/陡坎)抽取的比例
                 max_blocks=5,                 # 策略3: 矩形空洞的最大数量
                 block_size_range=(2.0, 15.0), # 矩形空洞的长宽边长范围(米)
                 feature_resolution=2.0,       # 极速地形特征提取的 2.5D 栅格分辨率(米)
                 max_query_ratio=0.9,          # 放开限制，允许挖走 90% 的点
                 compute_gt_low=True,          # 是否计算低频平滑真值 (双分支需要, 单分支可关闭)
                 query_max=None,
                 ground_class=None,            # 若非 None，仅从该类别的地面点中抽取 Query
                 extreme_hole_prob=0.3,        # 以此概率触发一个覆盖瓦片50%-80%范围的极端大空洞
                 ):
        self.random_ratio = random_ratio
        self.feature_ratio = feature_ratio
        self.max_blocks = max_blocks
        self.block_size_range = block_size_range
        self.feature_resolution = feature_resolution
        self.max_query_ratio = max_query_ratio
        self.compute_gt_low = compute_gt_low
        self.query_max = query_max
        self.ground_class = ground_class
        self.extreme_hole_prob = extreme_hole_prob

    def _fast_topographic_weights(self, coord):
        """
        极速计算地形特征权重 (基于 2.5D 栅格近似)，复杂度严格 O(N)
        """
        xy = coord[:, :2]
        z = coord[:, 2]

        min_x, min_y = np.min(xy[:, 0]), np.min(xy[:, 1])
        max_x, max_y = np.max(xy[:, 0]), np.max(xy[:, 1])
        
        # 1. 构建栅格 Bins
        bins_x = np.arange(min_x, max_x + self.feature_resolution, self.feature_resolution)
        bins_y = np.arange(min_y, max_y + self.feature_resolution, self.feature_resolution)
        
        # 如果点云范围极小(异常数据)，直接返回均匀分布
        if len(bins_x) < 3 or len(bins_y) < 3:
            return np.ones_like(z) / len(z)

        # 2. 极速计算格网统计量
        z_mean, x_edge, y_edge, _ = binned_statistic_2d(
            xy[:, 0], xy[:, 1], z, statistic='mean', bins=[bins_x, bins_y])
        z_max, _, _, _ = binned_statistic_2d(
            xy[:, 0], xy[:, 1], z, statistic='max', bins=[bins_x, bins_y])
        z_min, _, _, _ = binned_statistic_2d(
            xy[:, 0], xy[:, 1], z, statistic='min', bins=[bins_x, bins_y])

        # 处理空网格 (NaN)
        valid_mean = np.nanmean(z_mean)
        z_mean = np.nan_to_num(z_mean, nan=valid_mean if not np.isnan(valid_mean) else 0.0)
        z_range = np.nan_to_num(z_max - z_min, nan=0.0)

        # 3. 提取二阶特征 (拉普拉斯算子 -> 山脊/山谷)
        kernel = np.array([[ 0,  1,  0],
                           [ 1, -4,  1],
                           [ 0,  1,  0]])
        laplacian = convolve(z_mean, kernel, mode='reflect')
        curvature = np.abs(laplacian)

        # 4. 融合特征 (陡坎 + 山谷山脊)
        norm_range = z_range / (np.max(z_range) + 1e-6)
        norm_curve = curvature / (np.max(curvature) + 1e-6)
        grid_feature_map = norm_range + 1.5 * norm_curve 

        # 5. 极速映射回原始点云
        idx_x = np.clip(np.digitize(xy[:, 0], x_edge) - 1, 0, len(bins_x) - 2)
        idx_y = np.clip(np.digitize(xy[:, 1], y_edge) - 1, 0, len(bins_y) - 2)
        point_weights = grid_feature_map[idx_x, idx_y]

        # 指数激化：拉大普通平地点与地形特征点的概率差距
        point_weights = point_weights ** 2

        # 归一化为合法概率分布
        sum_weights = np.sum(point_weights)
        if sum_weights > 1e-6:
            return point_weights / sum_weights
        else:
            return np.ones_like(z) / len(z)
        
    def _robust_extract_low_frequency(self, coord, resolution=2.0, sigma=1.5):
        """
        稳健、极速且连续的低频地形真值提取 (Transform 中使用)
        """
        xy = coord[:, :2]
        z = coord[:, 2]

        # ---------------------------------------------------------
        # 1. 极端情况保底防崩 (点数太少直接返回均值平面)
        # ---------------------------------------------------------
        if len(z) < 50:
            return np.full_like(z, np.mean(z) if len(z) > 0 else 0.0)

        # ---------------------------------------------------------
        # 2. 构建带 Padding 的网格 (防止边缘点插值时越界)
        # ---------------------------------------------------------
        min_x, min_y = np.min(xy[:, 0]), np.min(xy[:, 1])
        max_x, max_y = np.max(xy[:, 0]), np.max(xy[:, 1])
        
        pad = resolution * 2.0
        bins_x = np.arange(min_x - pad, max_x + pad, resolution)
        bins_y = np.arange(min_y - pad, max_y + pad, resolution)

        # 如果范围太小，退化为均值
        if len(bins_x) < 3 or len(bins_y) < 3:
            return np.full_like(z, np.mean(z))

        # ---------------------------------------------------------
        # 3. 极速栅格化 (求网格均值)
        # ---------------------------------------------------------
        z_grid, x_edge, y_edge, _ = binned_statistic_2d(
            xy[:, 0], xy[:, 1], z, statistic='mean', bins=[bins_x, bins_y])

        # ---------------------------------------------------------
        # 4. 稳健的 NaN 空洞填补
        # ---------------------------------------------------------
        valid_mask = ~np.isnan(z_grid)
        if not np.any(valid_mask):
            return np.full_like(z, np.mean(z))

        # 仅当存在空洞时才进行插值填补，节省时间
        if not np.all(valid_mask):
            grid_x, grid_y = np.meshgrid(
                (x_edge[:-1] + x_edge[1:]) / 2, 
                (y_edge[:-1] + y_edge[1:]) / 2, 
                indexing='ij'
            )
            # 使用 NearestNDInterpolator 填补，速度极快
            valid_coords = np.column_stack((grid_x[valid_mask], grid_y[valid_mask]))
            valid_z = z_grid[valid_mask]
            interpolator = NearestNDInterpolator(valid_coords, valid_z)
            z_grid_filled = interpolator(grid_x, grid_y)
        else:
            z_grid_filled = z_grid

        # ---------------------------------------------------------
        # 5. 提取低频 (高斯滤波)
        # ---------------------------------------------------------
        z_grid_smooth = gaussian_filter(z_grid_filled, sigma=sigma)

        # ---------------------------------------------------------
        # 6. 🌟 核心：连续映射！(双线性插值消除阶梯效应)
        # ---------------------------------------------------------
        # 将实际的 X, Y 物理坐标转换为网格矩阵的“小数索引”
        idx_x = (xy[:, 0] - x_edge[0]) / resolution - 0.5
        idx_y = (xy[:, 1] - y_edge[0]) / resolution - 0.5

        # map_coordinates 底层是 C，速度极快
        # order=1 表示双线性插值，这样提取出来的高程是一个完美连续的光滑曲面
        z_low = map_coordinates(
            z_grid_smooth, 
            [idx_x, idx_y], 
            order=1, 
            mode='nearest'
        )

        return z_low.astype(np.float32)

    def __call__(self, data_dict):
        # 仅在包含监督信号时处理 (跳过推理致密网格阶段)
        if "segment" not in data_dict and "gt" not in data_dict and "query_gt" not in data_dict.get("keys", []):
            return data_dict

        coord = data_dict["coord"]
        num_points = coord.shape[0]

        if num_points < 10:
            # 极少点时给出最小合法输出，避免下游 Collect/Collate 报 KeyError
            data_dict["query_coord"] = np.empty((0, 2), dtype=np.float32)
            data_dict["query_gt"] = np.empty((0,), dtype=np.float32)
            if self.compute_gt_low:
                data_dict["query_gt_low"] = np.empty((0,), dtype=np.float32)
            return data_dict

        # ==========================================
        # 🌟 0. 全局低频真值计算与缓存
        # ==========================================
        if self.compute_gt_low:
            if "z_low" in data_dict:
                z_low_full = data_dict["z_low"]
            else:
                # 必须在破坏前对【最原始、最完整的 coord】进行计算
                z_low_full = self._robust_extract_low_frequency(
                    coord, resolution=3.0, sigma=1.5
                )
                data_dict["z_low"] = z_low_full
        else:
            z_low_full = None

        # ==========================================
        # 🌟 地面点掩码 (当 ground_class 不为 None 时，仅从地面点中抽 Query)
        # ==========================================
        if self.ground_class is not None:
            if "segment" not in data_dict:
                raise KeyError(
                    "TerrainImplicitSampler: ground_class is set but 'segment' "
                    "is missing in data_dict; cannot guarantee ground-only query points."
                )
            ground_eligible = (data_dict["segment"].flatten() == self.ground_class)
        else:
            ground_eligible = None  # 全部点均可成为 Query

        # 全局掩码，标记哪些点被挖走作为 Query
        query_mask = np.zeros(num_points, dtype=bool)

        # ==========================================
        # 策略 0: 极端大空洞 (模拟密林/水面造成的巨大地面缺失)
        # ==========================================
        if np.random.rand() < self.extreme_hole_prob:
            extent = coord.max(axis=0)[:2] - coord.min(axis=0)[:2]
            frac = np.random.uniform(0.5, 0.8)
            w = max(extent[0] * frac, 1.0)
            h = max(extent[1] * frac, 1.0)
            cx = np.random.uniform(coord[:, 0].min(), coord[:, 0].max())
            cy = np.random.uniform(coord[:, 1].min(), coord[:, 1].max())
            extreme_mask = (
                (np.abs(coord[:, 0] - cx) <= w / 2.0)
                & (np.abs(coord[:, 1] - cy) <= h / 2.0)
            )
            query_mask |= extreme_mask

        # ==========================================
        # 策略 1: 随机数量、随机长宽的矩形空洞抽取 (模拟遮挡/水面)
        # ==========================================
        num_blocks = np.random.randint(0, self.max_blocks + 1)
        for _ in range(num_blocks):
            w = np.random.uniform(self.block_size_range[0], self.block_size_range[1])
            h = np.random.uniform(self.block_size_range[0], self.block_size_range[1])
            center_idx = np.random.randint(num_points)
            center_x, center_y = coord[center_idx, 0], coord[center_idx, 1]
            
            mask_x = np.abs(coord[:, 0] - center_x) <= w / 2.0
            mask_y = np.abs(coord[:, 1] - center_y) <= h / 2.0
            query_mask |= (mask_x & mask_y)

        # ==========================================
        # 策略 2: 基于真实地形起伏（山脊、山谷、陡坎）的特征点抽取
        # ==========================================
        num_feat = int(num_points * self.feature_ratio)
        if num_feat > 0:
            probs = self._fast_topographic_weights(coord)
            feat_indices = np.random.choice(num_points, num_feat, replace=False, p=probs)
            query_mask[feat_indices] = True

        # ==========================================
        # 策略 3: 全局随机均匀抽取 (模拟激光雷达常规掉点/抽稀)
        # ==========================================
        num_rand = int(num_points * self.random_ratio)
        if num_rand > 0:
            rand_indices = np.random.choice(num_points, num_rand, replace=False)
            query_mask[rand_indices] = True

        # ==========================================
        # 🌟 限制 Query 仅来自地面点: 非地面点永远留在 Support
        # ==========================================
        if ground_eligible is not None:
            query_mask &= ground_eligible

        # ==========================================
        # 截断与保底机制：确保有足够的点送给 Backbone 提特征
        # ==========================================
        query_indices = np.where(query_mask)[0]
        max_allowed_queries = max(1, int(num_points * self.max_query_ratio))
        if self.query_max is not None:
            max_allowed_queries = min(max_allowed_queries, self.query_max)

        if len(query_indices) > max_allowed_queries:
            # 如果挖得太猛，随机归还一部分给 Support
            keep_idx = np.random.choice(len(query_indices), max_allowed_queries, replace=False)
            query_indices = query_indices[keep_idx]
            query_mask = np.zeros(num_points, dtype=bool)
            query_mask[query_indices] = True

        # 保底：若没有任何 query，随机选 1 个点作为 query
        if query_mask.sum() == 0:
            if ground_eligible is not None and ground_eligible.any():
                candidates = np.where(ground_eligible)[0]
                fallback = candidates[np.random.randint(len(candidates))]
            else:
                fallback = np.random.randint(num_points)
            query_mask[fallback] = True

        # 强校验：启用 ground_class 时，Query 必须全部为地面点
        if ground_eligible is not None and np.any(~ground_eligible[query_mask]):
            raise RuntimeError(
                "TerrainImplicitSampler invariant violated: non-ground points found in query set."
            )

        # 最终掩码
        support_mask = ~query_mask
        # 保证至少 1 个 support 点（极端保底）
        if support_mask.sum() == 0:
            support_mask[0] = True
            query_mask[0] = False

        # ==========================================
        # 🌟 第一段：提取 (index_operator 破坏数组长度之前)
        # ==========================================
        query_coord_xy = coord[query_mask, :2].copy().astype(np.float32)
        query_gt_raw   = coord[query_mask, 2].copy().astype(np.float32)

        if "normal" in data_dict:
            raw_qn = data_dict["normal"][query_mask].astype(np.float32)
            norms = np.linalg.norm(raw_qn, axis=-1, keepdims=True) + 1e-6
            query_normal_gt = raw_qn / norms
        else:
            query_normal_gt = None

        # ==========================================
        # 第二段：破坏性切片，仅保留 Support 点
        # ==========================================
        data_dict = index_operator(data_dict, np.where(support_mask)[0])

        # ==========================================
        # 第三段：干净注入
        # ==========================================
        data_dict["query_coord"] = query_coord_xy
        data_dict["query_gt"]    = query_gt_raw
        if z_low_full is not None:
            data_dict["query_gt_low"] = z_low_full[query_mask].copy().astype(np.float32)
        if query_normal_gt is not None:
            data_dict["query_normal_gt"] = query_normal_gt

        return data_dict


@TRANSFORMS.register_module()
class CategoryAwareDownsample(object):
    """Non-ground voxel downsampling while preserving 100% ground points.

    Ground points (``segment == ground_class``) are kept at full density.
    Non-ground points are voxel-downsampled to reduce computational cost
    while still providing semantic context to the backbone.

    Args:
        grid_size (float): Voxel edge length for non-ground downsampling.
        ground_class (int): LAS classification code for ground (default 2).
    """

    def __init__(self, grid_size=1.0, ground_class=2):
        self.grid_size = grid_size
        self.ground_class = ground_class

    def __call__(self, data_dict):
        if "segment" not in data_dict or "coord" not in data_dict:
            return data_dict

        coord = data_dict["coord"]
        num_points = coord.shape[0]
        if num_points < 10:
            return data_dict

        segment = data_dict["segment"].flatten()
        ground_indices = np.where(segment == self.ground_class)[0]
        non_ground_indices = np.where(segment != self.ground_class)[0]

        if len(non_ground_indices) == 0:
            # All ground — nothing to downsample
            return data_dict

        # Voxel downsample non-ground points
        ng_coord = coord[non_ground_indices]
        voxel_idx = np.floor(ng_coord / self.grid_size).astype(np.int32)
        # Produce a unique key per voxel via structured array view
        _, unique_rel_idx = np.unique(
            voxel_idx, axis=0, return_index=True
        )
        sampled_non_ground = non_ground_indices[unique_rel_idx]

        # Merge and shuffle
        keep_indices = np.concatenate([ground_indices, sampled_non_ground])
        np.random.shuffle(keep_indices)

        return index_operator(data_dict, keep_indices)


@TRANSFORMS.register_module()
class ClassFilter(object):
    """Filter point cloud to keep only points belonging to specified classes.

    Keys listed in ``index_valid_keys`` are subsetted; all other keys are
    left untouched.  If ``class_key`` is absent from the data dict the
    transform is a no-op.

    Args:
        keep_classes (list[int]): Class labels to retain.
        class_key (str): Key in data_dict containing per-point labels.
            Defaults to ``"segment"``.
    """

    def __init__(self, keep_classes, class_key="segment"):
        self.keep_classes = list(keep_classes)
        self.class_key = class_key

    def __call__(self, data_dict):
        if self.class_key not in data_dict:
            return data_dict
        labels = data_dict[self.class_key]
        mask = np.isin(labels, self.keep_classes)
        idx = np.where(mask)[0]
        return index_operator(data_dict, idx)


@TRANSFORMS.register_module()
class GridCoordinate(object):
    """Compute integer grid coordinates without downsampling.

    Adds ``grid_coord`` (int32, same shape as ``coord``) to the data dict
    and registers it in ``index_valid_keys``.  Unlike :class:`GridSample`,
    no points are removed.

    ``grid_coord[i] = floor(coord[i] / grid_size) - floor(coord / grid_size).min(axis=0)``

    Args:
        grid_size (float): Voxel edge length used for quantisation.
    """

    def __init__(self, grid_size=0.05):
        self.grid_size = grid_size

    def __call__(self, data_dict):
        assert "coord" in data_dict
        grid_coord = np.floor(data_dict["coord"] / self.grid_size).astype(np.int32)
        grid_coord -= grid_coord.min(axis=0)
        data_dict["grid_coord"] = grid_coord
        if "index_valid_keys" not in data_dict:
            data_dict["index_valid_keys"] = list(
                filter(lambda k: isinstance(data_dict[k], np.ndarray)
                       and data_dict[k].ndim > 0
                       and data_dict[k].shape[0] == data_dict["coord"].shape[0],
                       [k for k in data_dict if k != "index_valid_keys"])
            )
        if "grid_coord" not in data_dict["index_valid_keys"]:
            data_dict["index_valid_keys"].append("grid_coord")
        return data_dict
    
@TRANSFORMS.register_module()
class NonGroundSmoother(object):
    """
    实时树冠平滑器：拦截输入点云，提取局部 DSM，进行高斯滤波，
    并强行覆盖植被点的 Z 坐标，让高频噪声永远无法进入 Backbone。
    """
    def __init__(self, grid_size=1.0, sigma=2.0, ground_class=2):
        """
        :param grid_size: 内部构建迷你栅格的分辨率（如 1.0 米）
        :param sigma: 高斯滤波的强度（如 2.0，对应强力平滑）
        :param ground_class: 地面类别的标签 ID
        """
        self.grid_size = grid_size
        self.sigma = sigma
        self.ground_class = ground_class

    def __call__(self, input_dict):
        coord = input_dict["coord"]      # [N, 3]
        segment = input_dict.get("segment", None) # [N, 1]

        # 如果没有语义标签，为了安全起见，不执行强行覆盖
        if segment is None:
            return input_dict

        segment = segment.squeeze()
        
        # 1. 找到非地面点（植被、建筑等）的掩码
        non_ground_mask = (segment != self.ground_class)
        if not np.any(non_ground_mask):
            return input_dict # 全是地面，直接返回

        # 2. 构建局部的相对 2D 坐标系
        min_x, min_y = np.min(coord[:, 0]), np.min(coord[:, 1])
        max_x, max_y = np.max(coord[:, 0]), np.max(coord[:, 1])
        
        width = int(np.ceil((max_x - min_x) / self.grid_size)) + 1
        height = int(np.ceil((max_y - min_y) / self.grid_size)) + 1

        # 初始化 DSM 栅格，使用一个极低的值保底
        dsm_grid = np.full((width, height), -9999.0, dtype=np.float32)

        # 3. 将所有点投影到栅格，提取 Max Z (纯正的粗糙 DSM)
        grid_x = np.clip(((coord[:, 0] - min_x) / self.grid_size).astype(np.int32), 0, width - 1)
        grid_y = np.clip(((coord[:, 1] - min_y) / self.grid_size).astype(np.int32), 0, height - 1)
        
        # 极速向量化计算每个像素的最大 Z 值
        np.maximum.at(dsm_grid, (grid_x, grid_y), coord[:, 2])

        # 填补没有点的空洞像素（使用最近邻或全区均值，简单起见这里用非空均值）
        valid_mask = (dsm_grid > -9999.0)
        if np.any(valid_mask):
            mean_z = np.mean(dsm_grid[valid_mask])
            dsm_grid[~valid_mask] = mean_z

        # 4. 🌟 核心：执行离线级别的 2D 高斯平滑
        smoothed_dsm = gaussian_filter(dsm_grid, sigma=self.sigma)

        # 5. 映射回点云：查找每个点对应的平滑后 Z 值
        smoothed_z = smoothed_dsm[grid_x, grid_y]

        # 6. 🌟 物理覆盖：只把非地面点（树冠）的真实 Z 坐标，替换为平滑后的 Z！
        # 真实地面点保持毫米级精度不变
        coord[non_ground_mask, 2] = smoothed_z[non_ground_mask]

        input_dict["coord"] = coord
        return input_dict

@TRANSFORMS.register_module()
class ClassLabelClamp(object):
    """
    类别标签安全替换器：拦截异常的语义 ID。
    将所有小于 0 或大于等于 num_classes 的野生 ID，统一替换为指定的 clamp_id。
    """
    def __init__(self, num_classes=32, clamp_id=0):
        """
        :param num_classes: 数据集中有效类别的总数（上限排他，即有效范围 0 到 num_classes-1）
        :param clamp_id: 超出范围时赋予的默认 ID。建议使用 0 (通常代表 Unclassified) 
                         或 num_classes - 1 (需确保该类专门留作"其他/噪点"类)
        """
        self.num_classes = num_classes
        self.clamp_id = clamp_id

    def __call__(self, input_dict):
        if "segment" in input_dict and input_dict["segment"] is not None:
            segment = input_dict["segment"]
            
            # 找到所有不合法的野生 ID 的掩码 (Mask)
            invalid_mask = (segment < 0) | (segment >= self.num_classes)
            
            # 将它们统一替换为 clamp_id
            if np.any(invalid_mask):
                segment[invalid_mask] = self.clamp_id
                
            input_dict["segment"] = segment
            
        return input_dict

