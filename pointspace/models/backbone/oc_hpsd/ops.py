"""OC-HPSD 的连续可观测度、路由边和结构化 masking 算子。"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter



@dataclass(frozen=True)
class RoutedTokenPatchEdges:
    """一次去重后同时保存 anchor 与 simulated-missing 支持量。

    ``token``、``patch`` 的形状为 ``[E]``。其余 tensor 也为 ``[E]``，
    分别记录两条监督路由的支持点数和连续可观测度之和。
    """

    token: torch.Tensor
    patch: torch.Tensor
    anchor_count: torch.Tensor
    masked_count: torch.Tensor
    anchor_q_sum: torch.Tensor
    masked_q_sum: torch.Tensor

    @property
    def num_edges(self):
        return int(self.token.numel())

    def route_count(self, route):
        if route == "anchor":
            return self.anchor_count
        if route == "masked":
            return self.masked_count
        raise ValueError("route must be 'anchor' or 'masked'")

    def route_q_sum(self, route):
        if route == "anchor":
            return self.anchor_q_sum
        if route == "masked":
            return self.masked_q_sum
        raise ValueError("route must be 'anchor' or 'masked'")

    def route_weight(self, route, mode="sqrt_count", dtype=torch.float32):
        """返回融合支持数量和连续 q 的 edge 权重。"""
        count = self.route_count(route).to(dtype)
        q_sum = self.route_q_sum(route).to(dtype)
        if mode == "uniform":
            return torch.where(count > 0, q_sum / count.clamp_min(1.0), q_sum)
        if mode == "count":
            return q_sum
        if mode == "sqrt_count":
            return torch.where(
                count > 0,
                q_sum / torch.sqrt(count.clamp_min(1.0)),
                q_sum,
            )
        raise ValueError("edge_weight must be one of 'uniform', 'count', or 'sqrt_count'")


@dataclass(frozen=True)
class TokenRouteStats:
    """目标层 token 的逐点总数、有效数与 simulated-missing 统计。"""

    total_count: torch.Tensor
    valid_count: torch.Tensor
    masked_count: torch.Tensor
    masked_q_sum: torch.Tensor | None


@dataclass(frozen=True)
class MaskSelectionStats:
    """结构化 masking 的候选点数和受约束后的目标预算。"""

    candidate_count: torch.Tensor
    requested_count: torch.Tensor


def build_routed_token_patch_edges(
    input_to_level,
    patch_index,
    valid,
    observability,
    simulated_mask,
    num_tokens,
    num_patches,
    validate_mapping=False,
):
    """一次构造去重 token-patch edge，并统计两条监督路由。"""
    tensors = (input_to_level, patch_index, valid, observability, simulated_mask)
    if any(tensor.ndim != 1 for tensor in tensors):
        raise ValueError("All routed edge inputs must be 1D")
    if len({int(tensor.shape[0]) for tensor in tensors}) != 1:
        raise ValueError("Routed edge input lengths do not match")
    if num_tokens < 0 or num_patches < 0:
        raise ValueError("num_tokens and num_patches must be non-negative")
    if not torch.isfinite(observability).all():
        raise ValueError("observability contains non-finite values")
    if torch.any((observability < 0) | (observability > 1)):
        raise ValueError("observability must be within [0, 1]")

    relation = valid.bool() & (patch_index >= 0)
    token = input_to_level[relation].long()
    patch = patch_index[relation].long()
    if token.numel() == 0:
        empty_index = input_to_level.new_empty(0, dtype=torch.long)
        empty_float = observability.new_empty(0, dtype=torch.float32)
        return RoutedTokenPatchEdges(
            token=empty_index,
            patch=empty_index,
            anchor_count=empty_float,
            masked_count=empty_float,
            anchor_q_sum=empty_float,
            masked_q_sum=empty_float,
        )

    if validate_mapping:
        if int(token.min()) < 0 or int(token.max()) >= num_tokens:
            raise ValueError("Point-to-token mapping contains an invalid token index")
        if int(patch.min()) < 0 or int(patch.max()) >= num_patches:
            raise ValueError("Point-to-patch mapping contains an invalid patch index")

    edge_key = token * num_patches + patch
    unique_key, point_to_edge = torch.unique(
        edge_key, sorted=True, return_inverse=True
    )
    edge_count = int(unique_key.shape[0])
    masked = simulated_mask[relation].bool()
    q = observability[relation].float()
    anchor_float = (~masked).float()
    masked_float = masked.float()
    anchor_count = torch_scatter.scatter_sum(
        anchor_float, point_to_edge, dim=0, dim_size=edge_count
    )
    masked_count = torch_scatter.scatter_sum(
        masked_float, point_to_edge, dim=0, dim_size=edge_count
    )
    anchor_q_sum = torch_scatter.scatter_sum(
        q * anchor_float, point_to_edge, dim=0, dim_size=edge_count
    )
    masked_q_sum = torch_scatter.scatter_sum(
        q * masked_float, point_to_edge, dim=0, dim_size=edge_count
    )
    return RoutedTokenPatchEdges(
        token=torch.div(unique_key, num_patches, rounding_mode="floor"),
        patch=torch.remainder(unique_key, num_patches),
        anchor_count=anchor_count,
        masked_count=masked_count,
        anchor_q_sum=anchor_q_sum,
        masked_q_sum=masked_q_sum,
    )


def aggregate_tokens_to_patches_routed(
    token_feature,
    edges,
    route="anchor",
    edge_weight="sqrt_count",
):
    """按指定路由把低维 token feature 聚合到实际使用的 patch。"""
    weight = edges.route_weight(route, edge_weight, token_feature.dtype)
    keep = weight > 0
    if not torch.any(keep):
        return (
            token_feature[:0],
            edges.patch[:0],
            token_feature.new_empty(0),
        )

    token = edges.token[keep]
    patch = edges.patch[keep]
    weight = weight[keep]
    used_patch, edge_to_patch = torch.unique(
        patch, sorted=True, return_inverse=True
    )
    patch_feature = torch_scatter.scatter_sum(
        token_feature[token] * weight.unsqueeze(1),
        edge_to_patch,
        dim=0,
        dim_size=used_patch.shape[0],
    )
    patch_weight = torch_scatter.scatter_sum(
        weight, edge_to_patch, dim=0, dim_size=used_patch.shape[0]
    )
    patch_feature = patch_feature / patch_weight.clamp_min(1e-12).unsqueeze(1)
    return patch_feature, used_patch, patch_weight


def aggregate_teacher_to_tokens_routed(
    teacher,
    edges,
    route="masked",
    edge_weight="sqrt_count",
):
    """只为指定路由实际引用的 token 聚合归一化 DINO teacher。"""
    weight = edges.route_weight(route, edge_weight, torch.float32)
    keep = weight > 0
    if not torch.any(keep):
        return edges.token[:0], teacher[:0].float(), teacher.new_empty(0).float()

    token = edges.token[keep]
    patch = edges.patch[keep]
    weight = weight[keep]
    used_token, edge_to_token = torch.unique(
        token, sorted=True, return_inverse=True
    )
    teacher_float = F.normalize(teacher.float(), dim=-1, eps=1e-12)
    token_sum = torch_scatter.scatter_sum(
        teacher_float[patch] * weight.unsqueeze(1),
        edge_to_token,
        dim=0,
        dim_size=used_token.shape[0],
    )
    token_weight = torch_scatter.scatter_sum(
        weight, edge_to_token, dim=0, dim_size=used_token.shape[0]
    )
    token_teacher = F.normalize(
        token_sum / token_weight.clamp_min(1e-12).unsqueeze(1),
        dim=-1,
        eps=1e-12,
    )
    return used_token, token_teacher, token_weight


def compute_token_route_stats(
    input_to_level,
    valid,
    simulated_mask,
    num_tokens,
    observability=None,
):
    """由逐点路由计算 token 支持数量，不依赖 token-patch 去重。"""
    tensors = (input_to_level, valid, simulated_mask)
    if any(tensor.ndim != 1 for tensor in tensors):
        raise ValueError("Token route inputs must be 1D")
    if observability is not None and observability.ndim != 1:
        raise ValueError("observability must be 1D")
    if len({int(tensor.shape[0]) for tensor in tensors}) != 1:
        raise ValueError("Token route input lengths do not match")
    if (
        observability is not None
        and observability.shape[0] != input_to_level.shape[0]
    ):
        raise ValueError("Token route input lengths do not match")
    one = torch.ones(
        input_to_level.shape[0], device=input_to_level.device, dtype=torch.float32
    )
    total_count = torch_scatter.scatter_sum(
        one, input_to_level.long(), dim=0, dim_size=num_tokens
    )
    valid_count = torch_scatter.scatter_sum(
        valid.float(), input_to_level.long(), dim=0, dim_size=num_tokens
    )
    masked_count = torch_scatter.scatter_sum(
        (valid.bool() & simulated_mask.bool()).float(),
        input_to_level.long(),
        dim=0,
        dim_size=num_tokens,
    )
    masked_q_sum = None
    if observability is not None:
        masked_q_sum = torch_scatter.scatter_sum(
            observability.float() * (valid.bool() & simulated_mask.bool()).float(),
            input_to_level.long(),
            dim=0,
            dim_size=num_tokens,
        )
    return TokenRouteStats(
        total_count=total_count,
        valid_count=valid_count,
        masked_count=masked_count,
        masked_q_sum=masked_q_sum,
    )


class GeometryGuidedMaskGenerator(nn.Module):
    """按 XY block 和垂向跨度选择高可信可视点进行输入 masking。"""

    def __init__(
        self,
        block_size=4.0,
        min_observability=0.6,
        min_vertical_span=1.0,
        min_anchor_points=64,
        min_anchor_ratio=0.65,
        max_mask_points=8192,
        fallback_random_block=True,
        fill_partial_block=False,
    ):
        super().__init__()
        self.block_size = float(block_size)
        self.min_observability = float(min_observability)
        self.min_vertical_span = float(min_vertical_span)
        self.min_anchor_points = int(min_anchor_points)
        self.min_anchor_ratio = float(min_anchor_ratio)
        self.max_mask_points = (
            None if max_mask_points is None else int(max_mask_points)
        )
        self.fallback_random_block = bool(fallback_random_block)
        self.fill_partial_block = bool(fill_partial_block)
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if not 0 <= self.min_observability <= 1:
            raise ValueError("min_observability must be within [0, 1]")
        if self.min_vertical_span < 0 or self.min_anchor_points < 0:
            raise ValueError("vertical span and min anchor points must be non-negative")
        if not 0 <= self.min_anchor_ratio <= 1:
            raise ValueError("min_anchor_ratio must be within [0, 1]")
        if self.max_mask_points is not None and self.max_mask_points < 1:
            raise ValueError("max_mask_points must be positive or None")

    @torch.no_grad()
    def _sample_budget(self, candidate_count, mask_rate):
        required_anchor = max(
            self.min_anchor_points,
            int(candidate_count * self.min_anchor_ratio + 0.999999),
        )
        budget = min(
            int(candidate_count * mask_rate),
            max(candidate_count - required_anchor, 0),
        )
        if self.max_mask_points is not None:
            budget = min(budget, self.max_mask_points)
        return budget

    def forward(
        self,
        coord,
        batch,
        valid,
        observability,
        mask_rate,
        return_stats=False,
    ):
        if coord.ndim != 2 or coord.shape[1] != 3:
            raise ValueError("coord must have shape [N, 3]")
        if any(tensor.ndim != 1 for tensor in (batch, valid, observability)):
            raise ValueError("batch, valid and observability must be 1D")
        if not coord.shape[0] == batch.shape[0] == valid.shape[0] == observability.shape[0]:
            raise ValueError("Mask generator input lengths do not match")
        rate = float(mask_rate)
        result = torch.zeros(coord.shape[0], device=coord.device, dtype=torch.bool)
        if rate <= 0 or coord.shape[0] == 0:
            if not return_stats:
                return result
            zero = coord.new_zeros((), dtype=torch.long)
            return result, MaskSelectionStats(zero, zero)
        if rate > 1:
            raise ValueError("mask_rate must be within [0, 1]")

        candidate = valid.bool() & (observability >= self.min_observability)
        requested_count = 0
        for sample in torch.unique(batch, sorted=True):
            sample_index = torch.nonzero(batch == sample, as_tuple=False).flatten()
            sample_candidate = candidate[sample_index]
            candidate_count = int(sample_candidate.sum())
            mask_budget = self._sample_budget(candidate_count, rate)
            requested_count += mask_budget
            if mask_budget <= 0:
                continue

            sample_coord = coord[sample_index]
            block_coord = torch.floor(
                sample_coord[:, :2] / self.block_size
            ).long()
            _, point_to_block = torch.unique(
                block_coord, dim=0, sorted=True, return_inverse=True
            )
            num_blocks = int(point_to_block.max()) + 1
            candidate_per_block = torch_scatter.scatter_sum(
                sample_candidate.float(),
                point_to_block,
                dim=0,
                dim_size=num_blocks,
            )
            z_min, _ = torch_scatter.scatter_min(
                sample_coord[:, 2], point_to_block, dim=0, dim_size=num_blocks
            )
            z_max, _ = torch_scatter.scatter_max(
                sample_coord[:, 2], point_to_block, dim=0, dim_size=num_blocks
            )
            eligible = (candidate_per_block > 0) & (
                (z_max - z_min) >= self.min_vertical_span
            )
            if not torch.any(eligible) and self.fallback_random_block:
                eligible = candidate_per_block > 0
            eligible_index = torch.nonzero(eligible, as_tuple=False).flatten()
            if eligible_index.numel() == 0:
                continue

            order = eligible_index[
                torch.randperm(eligible_index.numel(), device=coord.device)
            ]
            ordered_count = candidate_per_block[order].long()
            cumulative = torch.cumsum(ordered_count, dim=0)
            selected_count = int((cumulative <= mask_budget).sum())
            selected = order[:selected_count]
            selected_lookup = torch.zeros(
                num_blocks, device=coord.device, dtype=torch.bool
            )
            selected_lookup[selected] = True
            local_mask = sample_candidate & selected_lookup[point_to_block]
            if self.fill_partial_block and selected_count < order.numel():
                full_count = (
                    int(cumulative[selected_count - 1])
                    if selected_count > 0
                    else 0
                )
                remaining = mask_budget - full_count
                if remaining > 0:
                    boundary = order[selected_count]
                    boundary_point = torch.nonzero(
                        sample_candidate & (point_to_block == boundary),
                        as_tuple=False,
                    ).flatten()
                    boundary_point = boundary_point[
                        torch.randperm(boundary_point.numel(), device=coord.device)[
                            :remaining
                        ]
                    ]
                    local_mask[boundary_point] = True
            if not torch.any(local_mask):
                continue
            result[sample_index[local_mask]] = True
        if not return_stats:
            return result
        return result, MaskSelectionStats(
            candidate_count=candidate.sum(),
            requested_count=coord.new_tensor(requested_count, dtype=torch.long),
        )
