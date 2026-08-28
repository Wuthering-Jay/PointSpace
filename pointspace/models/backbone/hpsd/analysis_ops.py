"""HPSD 数据覆盖审计使用的稀疏 token 统计。"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torch_scatter

from .hpsd_v1m1 import TokenPatchEdges


@dataclass(frozen=True)
class TokenTeacherStats:
    """具有 DINO teacher 的紧凑 token 统计。"""

    token: torch.Tensor
    feature: torch.Tensor
    purity: torch.Tensor
    point_count: torch.Tensor
    patch_count: torch.Tensor


def compute_token_visibility(input_to_level, valid, num_tokens):
    """由逐点 valid 计算 token 的总点数、有效点数和可视率。"""
    if input_to_level.ndim != 1 or valid.ndim != 1:
        raise ValueError("input_to_level and valid must be 1D")
    if input_to_level.shape[0] != valid.shape[0]:
        raise ValueError("input_to_level and valid lengths do not match")
    if num_tokens < 0:
        raise ValueError("num_tokens must be non-negative")

    one = torch.ones(
        input_to_level.shape[0], device=valid.device, dtype=torch.float32
    )
    total_count = torch_scatter.scatter_sum(
        one, input_to_level.long(), dim=0, dim_size=num_tokens
    )
    valid_count = torch_scatter.scatter_sum(
        valid.float(), input_to_level.long(), dim=0, dim_size=num_tokens
    )
    visibility = valid_count / total_count.clamp_min(1.0)
    return visibility, valid_count, total_count


def _edge_weight(point_count, mode, dtype):
    if mode == "uniform":
        return torch.ones_like(point_count, dtype=dtype)
    if mode == "count":
        return point_count.to(dtype)
    if mode == "sqrt_count":
        return torch.sqrt(point_count.to(dtype))
    raise ValueError("edge_weight must be one of 'uniform', 'count', or 'sqrt_count'")


def aggregate_patch_teacher_to_tokens(teacher, edges, edge_weight="sqrt_count"):
    """把 token-patch 边上的 DINO 特征聚合为紧凑 token teacher。"""
    if teacher.ndim != 2:
        raise ValueError("teacher must have shape [P, C]")
    if not isinstance(edges, TokenPatchEdges):
        raise TypeError("edges must be TokenPatchEdges")
    if edges.num_edges == 0:
        empty_index = edges.token
        empty_float = teacher.new_empty(0, dtype=torch.float32)
        return TokenTeacherStats(
            token=empty_index,
            feature=teacher[:0].float(),
            purity=empty_float,
            point_count=empty_float,
            patch_count=empty_float,
        )

    teacher_float = F.normalize(teacher.float(), dim=-1, eps=1e-12)
    weight = _edge_weight(edges.point_count, edge_weight, torch.float32)
    used_token, edge_to_token = torch.unique(
        edges.token.long(), sorted=True, return_inverse=True
    )
    token_sum = torch_scatter.scatter_sum(
        teacher_float[edges.patch.long()] * weight.unsqueeze(1),
        edge_to_token,
        dim=0,
        dim_size=used_token.shape[0],
    )
    weight_sum = torch_scatter.scatter_sum(
        weight, edge_to_token, dim=0, dim_size=used_token.shape[0]
    )
    token_teacher = F.normalize(
        token_sum / weight_sum.clamp_min(1e-12).unsqueeze(1),
        dim=-1,
        eps=1e-12,
    )
    edge_cosine = torch.sum(
        teacher_float[edges.patch.long()] * token_teacher[edge_to_token], dim=-1
    )
    purity = torch_scatter.scatter_sum(
        edge_cosine * weight,
        edge_to_token,
        dim=0,
        dim_size=used_token.shape[0],
    ) / weight_sum.clamp_min(1e-12)
    point_count = torch_scatter.scatter_sum(
        edges.point_count.float(),
        edge_to_token,
        dim=0,
        dim_size=used_token.shape[0],
    )
    patch_count = torch_scatter.scatter_sum(
        torch.ones_like(weight),
        edge_to_token,
        dim=0,
        dim_size=used_token.shape[0],
    )
    return TokenTeacherStats(
        token=used_token,
        feature=token_teacher,
        purity=purity.clamp(min=-1.0, max=1.0),
        point_count=point_count,
        patch_count=patch_count,
    )
