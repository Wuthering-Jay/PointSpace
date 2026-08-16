"""HPSD：将正射影像 DINO patch 特征蒸馏到三维点云 encoder。

本文件只负责三维层级特征与 DINO patch 之间的稀疏蒸馏关系，不负责读取
影像或运行 DINO。DINO 特征和逐点对应关系由 ``LasImageDataset`` 读取。

主要张量约定如下：

``N``
    当前 batch 的输入点数。
``T``
    当前蒸馏层的三维 token 数，通常远小于 ``N``。
``P``
    batch 内全部 DINO patch 数，特征 ``dino_feature`` 的形状为 ``[P,Cd]``。
``E``
    去重后的 token-patch 边数。一条边可以由多个原始点共同支持。
``U``
    当前 batch 中实际获得点云监督的 DINO patch 数，``U <= P``。
``C3``
    当前蒸馏层的三维通道数。
``Cd``
    DINO teacher 通道数，当前默认保持原生 1024 维。

训练路径首先通过 ``input_to_level: [N]`` 和 ``dino_patch_index: [N]`` 构造
``E`` 条稀疏关系，再在低维 ``C3`` 空间聚合为 ``[U,C3]``，最后只为这
``U`` 个 patch 生成 ``[U,Cd]`` student prediction。这样不会创建显存开销
很大的逐点 ``[N,1024]`` 训练张量。

测试导出路径不需要 DINO 和 correspondence。它把目标层 token 特征投影并
按照 ``input_to_level`` gather 回当前 fragment 的输入点顺序，随后由
``HPSDFeatureTester`` 合并多个 GridSample fragment。
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter

from pointspace.models.builder import MODELS, build_model
from pointspace.models.utils.misc import offset2batch


@dataclass(frozen=True)
class TokenPatchEdges:
    """一个三维层级与 DINO patch 之间的去重稀疏边。

    Attributes:
        token: ``[E]``，每条边对应的三维 token 行号。
        patch: ``[E]``，每条边对应的 batch 全局 DINO patch 行号。
        point_count: ``[E]``，支持该 token-patch 关系的原始点数量。
    """

    token: torch.Tensor
    patch: torch.Tensor
    point_count: torch.Tensor

    @property
    def num_edges(self):
        return int(self.token.numel())


def build_token_patch_edges(
    input_to_level,
    patch_index,
    valid,
    num_tokens,
    num_patches,
    validate_mapping=False,
):
    """从逐点映射构造唯一的 ``(token, patch)`` 边及其支持点数。

    无效点由 ``valid=False`` 或 ``patch_index=-1`` 表示，会在建边前移除。
    对相同 ``(token, patch)`` 的多个点只保留一条边，并记录在
    ``point_count`` 中。该过程保留一个 token 对多个 patch、一个 patch 对
    多个 token 的完整多对多关系，不退化为最近邻 patch。
    """
    if input_to_level.ndim != 1 or patch_index.ndim != 1 or valid.ndim != 1:
        raise ValueError("input_to_level, patch_index and valid must be 1D")
    if not input_to_level.shape[0] == patch_index.shape[0] == valid.shape[0]:
        raise ValueError("Point-to-token and point-to-patch lengths do not match")
    if num_tokens < 0 or num_patches < 0:
        raise ValueError("num_tokens and num_patches must be non-negative")

    # correspondence 是逐点记录，只有 valid 且 patch 非负的点参与蒸馏。
    relation_mask = valid.bool() & (patch_index >= 0)
    token = input_to_level[relation_mask].long()
    patch = patch_index[relation_mask].long()
    if token.numel() == 0:
        empty = input_to_level.new_empty(0, dtype=torch.long)
        return TokenPatchEdges(token=empty, patch=empty, point_count=empty)

    if validate_mapping:
        if int(token.min()) < 0 or int(token.max()) >= num_tokens:
            raise ValueError("Point-to-token mapping contains an invalid token index")
        if int(patch.min()) < 0 or int(patch.max()) >= num_patches:
            raise ValueError("Point-to-patch mapping contains an invalid patch index")

    # 将二维索引编码为无碰撞的一维键，比 torch.unique(dim=0) 少一次 [N,2]
    # 临时张量分配。解码后仍分别保存 token 和 patch。
    edge_key = token * num_patches + patch
    unique_key, point_count = torch.unique(edge_key, sorted=True, return_counts=True)
    return TokenPatchEdges(
        token=torch.div(unique_key, num_patches, rounding_mode="floor"),
        patch=torch.remainder(unique_key, num_patches),
        point_count=point_count,
    )


def aggregate_tokens_to_patches(token_feat, edges, edge_weight="sqrt_count"):
    """在低维三维特征空间内，把多条 token 边聚合到被引用 patch。

    Args:
        token_feat: ``[T,C3]``，某一 encoder 层的原生 token 特征。
        edges: 去重后的 ``E`` 条 token-patch 边。
        edge_weight: ``uniform`` 为每条边等权；``count`` 按支持点数加权；
            ``sqrt_count`` 按支持点数平方根加权，默认用于减弱高密度区域偏置。

    Returns:
        patch_feat: ``[U,C3]``，每个有效 patch 的聚合三维特征。
        used_patch: ``[U]``，patch_feat 对应的 batch 全局 patch 行号。
        patch_weight: ``[U]``，每个 patch 的边权总和。
    """
    if edges.num_edges == 0:
        return (
            token_feat.new_empty((0, token_feat.shape[1])),
            edges.patch,
            token_feat.new_empty(0),
        )

    if edge_weight == "uniform":
        weight = torch.ones_like(edges.point_count, dtype=token_feat.dtype)
    elif edge_weight == "count":
        weight = edges.point_count.to(token_feat.dtype)
    elif edge_weight == "sqrt_count":
        weight = torch.sqrt(edges.point_count.to(token_feat.dtype))
    else:
        raise ValueError("edge_weight must be one of 'uniform', 'count', or 'sqrt_count'")

    # edge_to_patch 把每条稀疏边映射到紧凑的 U 个 used patch 行号。
    used_patch, edge_to_patch = torch.unique(edges.patch, sorted=True, return_inverse=True)
    weighted_feat = token_feat[edges.token] * weight.unsqueeze(1)
    patch_feat = torch_scatter.scatter_sum(
        weighted_feat, edge_to_patch, dim=0, dim_size=used_patch.shape[0]
    )
    patch_weight = torch_scatter.scatter_sum(
        weight, edge_to_patch, dim=0, dim_size=used_patch.shape[0]
    )
    patch_feat = patch_feat / patch_weight.clamp_min(1e-12).unsqueeze(1)
    return patch_feat, used_patch, patch_weight


@MODELS.register_module("HPSD-v1m1")
class HierarchicalPatchSetDistiller(nn.Module):
    """将原生 DINO patch teacher 特征蒸馏到 PTV3 或 LitePT。

    Args:
        backbone: 支持 ``return_hierarchy=True`` 的 encoder-only 配置。
        distill_levels: 各 encoder 层是否独立接受 DINO 蒸馏。
        distill_loss_weights: 各层蒸馏 loss 权重，未启用层的值被忽略。
        level_channels: 各 encoder level 的通道数，必须与 backbone 一致。
        teacher_channels: DINO teacher 通道数，默认 1024。
        edge_weight: token-patch 聚合权重策略。
        sample_balanced: 是否先在每个样本内平均 patch loss，再对样本平均。
        validate_mapping: 是否在建边时执行同步式范围检查；离线映射可信时可关。
        projector_hidden_channels: MLP 中间层通道数。

    训练输入除常规 ``coord/grid_coord/feat/offset`` 外，还需要：

    - ``dino_feature: [P,Cd]``；
    - ``dino_patch_index: [N]``，有效值为 batch 全局 patch 行号；
    - ``dino_valid: [N]``；
    - ``dino_offset: [B]``，每个样本 patch 数的累计边界。
    """

    def __init__(
        self,
        backbone,
        distill_levels=(False, False, True, True, True),
        distill_loss_weights=(0.0, 0.0, 1.0, 0.5, 0.25),
        level_channels=None,
        teacher_channels=1024,
        edge_weight="sqrt_count",
        sample_balanced=True,
        validate_mapping=False,
        projector_hidden_channels=1024,
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        self.teacher_channels = int(teacher_channels)
        self.edge_weight = edge_weight
        self.sample_balanced = bool(sample_balanced)
        self.validate_mapping = bool(validate_mapping)
        self.projector_hidden_channels = int(projector_hidden_channels)

        if level_channels is None:
            raise ValueError("level_channels is required")
        self.level_channels = tuple(int(channel) for channel in level_channels)
        self.distill_levels = tuple(bool(enabled) for enabled in distill_levels)
        self.distill_loss_weights = tuple(
            float(weight) for weight in distill_loss_weights
        )
        if len(self.distill_levels) != len(self.level_channels):
            raise ValueError(
                "distill_levels and level_channels must have the same length"
            )
        if len(self.distill_loss_weights) != len(self.level_channels):
            raise ValueError(
                "distill_loss_weights and level_channels must have the same length"
            )
        self.active_levels = tuple(
            level for level, enabled in enumerate(self.distill_levels) if enabled
        )
        if not self.active_levels:
            raise ValueError("At least one distill level must be enabled")
        if any(weight < 0 for weight in self.distill_loss_weights):
            raise ValueError("distill_loss_weights must be non-negative")
        if any(self.distill_loss_weights[level] <= 0 for level in self.active_levels):
            raise ValueError("Enabled distill levels must have positive loss weights")
        if self.teacher_channels <= 0:
            raise ValueError("teacher_channels must be positive")
        if edge_weight not in {"uniform", "count", "sqrt_count"}:
            raise ValueError("edge_weight must be one of 'uniform', 'count', or 'sqrt_count'")
        if self.projector_hidden_channels <= 0:
            raise ValueError("projector_hidden_channels must be positive")

        # 每层先在自身 C3 空间聚合到 patch，再独立映射到 Cd，
        # 避免深层因通道数更多而在 concat 中获得隐式高权重。
        self.student_projectors = nn.ModuleDict(
            {
                str(level): nn.Sequential(
                    nn.LayerNorm(self.level_channels[level]),
                    nn.Linear(
                        self.level_channels[level],
                        self.projector_hidden_channels,
                    ),
                    nn.GELU(),
                    nn.Linear(
                        self.projector_hidden_channels, self.teacher_channels
                    ),
                )
                for level in self.active_levels
            }
        )

    def _encode(self, input_dict):
        """运行 backbone，并校验所有已启用蒸馏层。"""
        point, hierarchy = self.backbone(input_dict, return_hierarchy=True)
        if len(hierarchy) != len(self.level_channels):
            raise ValueError(
                f"Backbone returned {len(hierarchy)} levels, but "
                f"{len(self.level_channels)} were configured"
            )
        for level_index in self.active_levels:
            level_feat = hierarchy[level_index].point.feat
            expected_channels = self.level_channels[level_index]
            if level_feat.ndim != 2 or level_feat.shape[1] != expected_channels:
                raise ValueError(
                    f"Encoder level {level_index} feature shape "
                    f"{tuple(level_feat.shape)} does not match configured channels "
                    f"{expected_channels}"
                )
        return point, hierarchy

    def extract_point_feature(
        self,
        input_dict,
        feature_source="projected",
        feature_level=2,
        normalize=True,
    ):
        """按当前 fragment 的 backbone 输入点顺序返回逐点特征。

        ``projected`` 导出指定层的 DINO 对齐 ``[N,Cd]`` 特征；
        ``backbone`` 导出该层原生 ``[N,C3]`` 特征。投影只在 ``T``
        个 token 上执行，然后通过 ``input_to_level`` gather 到点。这里的 N 是
        当前 fragment 点数，整张 tile 的原序合并由 tester 完成。
        """
        feature_level = int(feature_level)
        if feature_level not in self.active_levels:
            raise ValueError(
                f"feature_level={feature_level} is not enabled by distill_levels"
            )
        _, hierarchy = self._encode(input_dict)
        level = hierarchy[feature_level]
        level_feat = level.point.feat
        if feature_source == "projected":
            token_feature = self.student_projectors[str(feature_level)](level_feat)
        elif feature_source == "backbone":
            token_feature = level_feat
        else:
            raise ValueError("feature_source must be 'projected' or 'backbone'")
        point_feature = token_feature[level.input_to_level]
        if normalize:
            point_feature = F.normalize(point_feature.float(), dim=-1, eps=1e-12)
        return point_feature

    @staticmethod
    def _sample_balanced_mean(patch_loss, patch_index, dino_offset):
        """先按样本平均 patch loss，再对具有有效监督的样本等权平均。"""
        if patch_loss.numel() == 0:
            return patch_loss.sum()
        patch_batch = offset2batch(dino_offset.long())[patch_index]
        batch_size = int(dino_offset.shape[0])
        loss_sum = torch_scatter.scatter_sum(
            patch_loss, patch_batch, dim=0, dim_size=batch_size
        )
        loss_count = torch_scatter.scatter_sum(
            torch.ones_like(patch_loss), patch_batch, dim=0, dim_size=batch_size
        )
        valid_sample = loss_count > 0
        if not torch.any(valid_sample):
            return patch_loss.sum()
        return (loss_sum[valid_sample] / loss_count[valid_sample]).mean()

    def forward(
        self,
        input_dict,
        return_point=False,
        return_point_feature=False,
        feature_source="projected",
        feature_level=2,
        normalize_feature=True,
    ):
        """根据 ``return_point_feature`` 在训练蒸馏和测试导出路径间切换。

        训练返回 ``loss``、``patch_loss`` 和 token/edge/patch 数量；导出路径
        只返回 ``point_feature``，因此测试数据不需要携带任何 DINO 字段。
        """
        # 特征导出是独立路径：不建 token-patch 边，也不计算 teacher loss。
        if return_point_feature:
            return {
                "point_feature": self.extract_point_feature(
                    input_dict,
                    feature_source=feature_source,
                    feature_level=feature_level,
                    normalize=normalize_feature,
                )
            }

        required = {"dino_feature", "dino_patch_index", "dino_valid", "dino_offset"}
        missing = required.difference(input_dict)
        if missing:
            raise KeyError(f"HPSD input is missing fields: {sorted(missing)}")

        point, hierarchy = self._encode(input_dict)
        dino_feature = input_dict["dino_feature"]
        if dino_feature.ndim != 2 or dino_feature.shape[1] != self.teacher_channels:
            raise ValueError(
                f"Expected DINO feature shape [P, {self.teacher_channels}], got "
                f"{tuple(dino_feature.shape)}"
            )

        # 各层独立建立 token-patch 图、聚合和映射。各层共享同一
        # teacher，但不共享 projector，因而每层都必须单独学会对齐。
        teacher = F.normalize(dino_feature.float(), dim=-1)
        weighted_losses = []
        result = {}
        total_weight = sum(
            self.distill_loss_weights[level] for level in self.active_levels
        )
        for level_index in self.active_levels:
            level = hierarchy[level_index]
            level_feat = level.point.feat
            edges = build_token_patch_edges(
                input_to_level=level.input_to_level,
                patch_index=input_dict["dino_patch_index"],
                valid=input_dict["dino_valid"],
                num_tokens=level_feat.shape[0],
                num_patches=dino_feature.shape[0],
                validate_mapping=self.validate_mapping,
            )
            patch_feat, used_patch, _ = aggregate_tokens_to_patches(
                level_feat, edges, edge_weight=self.edge_weight
            )
            projector = self.student_projectors[str(level_index)]

            if patch_feat.shape[0] == 0:
                # 零监督 batch 仍让每层 projector 与对应 backbone stage
                # 连入计算图，避免 DDP 将其误判为 unused parameter。
                patch_pred = projector(level_feat[:0])
                level_loss = patch_pred.float().sum() * 0.0
            else:
                patch_pred = projector(patch_feat)
                student = F.normalize(patch_pred.float(), dim=-1)
                loss_per_patch = 1.0 - torch.sum(
                    student * teacher[used_patch], dim=-1
                )
                if self.sample_balanced:
                    level_loss = self._sample_balanced_mean(
                        loss_per_patch, used_patch, input_dict["dino_offset"]
                    )
                else:
                    level_loss = loss_per_patch.mean()

            weight = self.distill_loss_weights[level_index]
            weighted_losses.append(level_loss * weight)
            # 分层指标用于观察粗细尺度的收敛差异；训练器只使用
            # result["loss"] 反向，不会重复累加这些日志值。
            result[f"hpsd_level_{level_index}_loss"] = level_loss.detach()
            result[f"hpsd_level_{level_index}_tokens"] = level_feat.new_tensor(
                level_feat.shape[0]
            )
            result[f"hpsd_level_{level_index}_edges"] = level_feat.new_tensor(
                edges.num_edges
            )
            result[f"hpsd_level_{level_index}_patches"] = level_feat.new_tensor(
                used_patch.shape[0]
            )

        loss = torch.stack(weighted_losses).sum() / total_weight
        result["loss"] = loss
        result["patch_loss"] = loss.detach()
        if return_point:
            result["point"] = point
            result["hierarchy"] = hierarchy
        return result
