"""VRSR 使用的稀疏 token 统计与有界检索算子。"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torch_scatter

from ..hpsd.hpsd_v1m1 import TokenPatchEdges


@dataclass(frozen=True)
class TokenTeacherStats:
    """具有 DINO teacher 的紧凑 token 统计。

    ``token`` 为原始 token 行号；其余字段第一维均与 ``token`` 对齐。只为
    有 token-patch 边的 token 保存 1024 维 teacher，避免创建稠密 ``[T,1024]``。
    """

    token: torch.Tensor
    feature: torch.Tensor
    purity: torch.Tensor
    point_count: torch.Tensor
    patch_count: torch.Tensor


def compute_token_visibility(input_to_level, valid, num_tokens):
    """由逐点 valid 计算 token 的总点数、有效点数和连续可视率。"""
    if input_to_level.ndim != 1 or valid.ndim != 1:
        raise ValueError("input_to_level and valid must be 1D")
    if input_to_level.shape[0] != valid.shape[0]:
        raise ValueError("input_to_level and valid lengths do not match")
    if num_tokens < 0:
        raise ValueError("num_tokens must be non-negative")

    dtype = torch.float32
    one = torch.ones(input_to_level.shape[0], device=valid.device, dtype=dtype)
    total_count = torch_scatter.scatter_sum(
        one, input_to_level.long(), dim=0, dim_size=num_tokens
    )
    valid_count = torch_scatter.scatter_sum(
        valid.to(dtype), input_to_level.long(), dim=0, dim_size=num_tokens
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
    """把稀疏 token-patch 边上的 DINO 特征聚合为紧凑 token teacher。

    purity 是每条 patch teacher 与其 token 聚合中心 cosine 的加权均值；它衡量
    一个 token 内多个 patch 的语义一致程度，而不是可视点比例。
    """
    if teacher.ndim != 2:
        raise ValueError("teacher must have shape [P, C]")
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
    purity = purity.clamp(min=-1.0, max=1.0)
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
        purity=purity,
        point_count=point_count,
        patch_count=patch_count,
    )


def height_stratified_cap(index, coord, max_count):
    """沿 z 均匀覆盖地确定性截取 token，避免只保留低行号区域。"""
    if max_count is None or index.numel() <= int(max_count):
        return index
    max_count = int(max_count)
    if max_count <= 0:
        return index[:0]
    order = torch.argsort(coord[index, 2])
    position = torch.linspace(
        0,
        order.numel() - 1,
        steps=max_count,
        device=index.device,
    ).round().long()
    return index[order[position]]


def chunked_topk_cosine(query, key, topk, chunk_size=256):
    """以有界临时矩阵计算归一化特征的 cosine Top-K。"""
    if query.ndim != 2 or key.ndim != 2 or query.shape[1] != key.shape[1]:
        raise ValueError("query and key must be [N, C] with equal channels")
    if int(topk) <= 0 or int(chunk_size) <= 0:
        raise ValueError("topk and chunk_size must be positive")
    k = min(int(topk), key.shape[0])
    if query.shape[0] == 0 or k == 0:
        return (
            query.new_empty((query.shape[0], 0)),
            torch.empty((query.shape[0], 0), device=query.device, dtype=torch.long),
        )
    values = []
    indices = []
    for start in range(0, query.shape[0], int(chunk_size)):
        similarity = query[start : start + int(chunk_size)] @ key.transpose(0, 1)
        value, index = torch.topk(similarity, k=k, dim=1, largest=True, sorted=True)
        values.append(value)
        indices.append(index)
    return torch.cat(values, dim=0), torch.cat(indices, dim=0)
