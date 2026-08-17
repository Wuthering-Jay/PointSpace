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
    目标层及更深层 concat 后的三维通道数。
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
            # 保留与 token_feat 的计算图连接，使零监督 batch 也能
            # 为 backbone 和 projector 生成零梯度。
            token_feat[:0],
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


def fuse_hierarchy_features(hierarchy, target_level):
    """把目标层及其后所有深层特征对齐后按通道拼接。

    ``pooling_inverse`` 表示细层 token 所属的下一粗层 token。逐层组合该
    映射，可将更深层特征无插值地复制回目标层，最终返回形状
    ``[T, sum(level_channels[target_level:])]`` 的层级表示。
    """
    if target_level < 0 or target_level >= len(hierarchy):
        raise ValueError(f"Invalid target_level {target_level}")
    target = hierarchy[target_level].point
    target_to_level = torch.arange(
        target.feat.shape[0], device=target.feat.device, dtype=torch.long
    )
    features = [target.feat]
    for level in hierarchy[target_level + 1 :]:
        child = level.point
        if "pooling_inverse" not in child.keys():
            raise RuntimeError(
                f"Encoder level {level.level} has no pooling_inverse for feature fusion"
            )
        target_to_level = child.pooling_inverse.long()[target_to_level]
        features.append(child.feat[target_to_level])
    return torch.cat(features, dim=1)


@MODELS.register_module("HPSD-v1m1")
class HierarchicalPatchSetDistiller(nn.Module):
    """将原生 DINO patch teacher 特征蒸馏到 PTV3 或 LitePT。

    Args:
        backbone: 支持 ``return_hierarchy=True`` 的 encoder-only 配置。
        distill_level: concat 的目标层级，0 为输入分辨率。
        level_channels: 各 encoder level 的通道数，必须与 backbone 一致。
        teacher_channels: DINO teacher 通道数，默认 1024。
        edge_weight: token-patch 聚合权重策略。
        sample_balanced: 是否先在每个样本内平均 patch loss，再对样本平均。
        validate_mapping: 是否在建边时执行同步式范围检查；离线映射可信时可关。
        fuse_deeper_features: 是否把目标层后的深层特征对齐并 concat。
        projector_hidden_channels: 轻量 MLP projector 的隐藏通道数。

    训练输入除常规 ``coord/grid_coord/feat/offset`` 外，还需要：

    - ``dino_feature: [P,Cd]``；
    - ``dino_patch_index: [N]``，有效值为 batch 全局 patch 行号；
    - ``dino_valid: [N]``；
    - ``dino_offset: [B]``，每个样本 patch 数的累计边界。
    """

    def __init__(
        self,
        backbone,
        distill_level=2,
        level_channels=None,
        teacher_channels=1024,
        edge_weight="sqrt_count",
        sample_balanced=True,
        validate_mapping=False,
        fuse_deeper_features=True,
        projector_hidden_channels=1024,
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        self.distill_level = int(distill_level)
        self.teacher_channels = int(teacher_channels)
        self.edge_weight = edge_weight
        self.sample_balanced = bool(sample_balanced)
        self.validate_mapping = bool(validate_mapping)
        self.fuse_deeper_features = bool(fuse_deeper_features)
        self.projector_hidden_channels = int(projector_hidden_channels)

        if self.distill_level < 0:
            raise ValueError("distill_level must be non-negative")
        if level_channels is None:
            raise ValueError("level_channels is required")
        self.level_channels = tuple(int(channel) for channel in level_channels)
        if self.distill_level >= len(self.level_channels):
            raise ValueError(
                f"distill_level={self.distill_level} is outside "
                f"level_channels with {len(self.level_channels)} levels"
            )
        if self.teacher_channels <= 0:
            raise ValueError("teacher_channels must be positive")
        if edge_weight not in {"uniform", "count", "sqrt_count"}:
            raise ValueError("edge_weight must be one of 'uniform', 'count', or 'sqrt_count'")
        if self.projector_hidden_channels <= 0:
            raise ValueError("projector_hidden_channels must be positive")

        self.projector_in_channels = (
            sum(self.level_channels[self.distill_level :])
            if self.fuse_deeper_features
            else self.level_channels[self.distill_level]
        )
        # 只保留 MLP projector。它仅作用于 U 个有效 patch，而不是全部点。
        self.student_projector = nn.Sequential(
            nn.LayerNorm(self.projector_in_channels),
            nn.Linear(self.projector_in_channels, self.projector_hidden_channels),
            nn.GELU(),
            nn.Linear(self.projector_hidden_channels, self.teacher_channels),
        )

    def _encode(self, input_dict):
        """运行 backbone，并返回目标层及 concat 后的层级特征。"""
        point, hierarchy = self.backbone(input_dict, return_hierarchy=True)
        if self.distill_level >= len(hierarchy):
            raise ValueError(
                f"Backbone returned {len(hierarchy)} levels, but "
                f"distill_level={self.distill_level} was requested"
            )
        level = hierarchy[self.distill_level]
        level_feat = level.point.feat
        expected_channels = self.level_channels[self.distill_level]
        if level_feat.ndim != 2 or level_feat.shape[1] != expected_channels:
            raise ValueError(
                f"Encoder level {self.distill_level} feature shape "
                f"{tuple(level_feat.shape)} does not match configured channels "
                f"{expected_channels}"
            )
        distill_feat = (
            fuse_hierarchy_features(hierarchy, self.distill_level)
            if self.fuse_deeper_features
            else level_feat
        )
        if distill_feat.shape[1] != self.projector_in_channels:
            raise ValueError(
                f"Fused encoder feature has {distill_feat.shape[1]} channels, "
                f"but projector expects {self.projector_in_channels}"
            )
        return point, hierarchy, level, distill_feat

    def extract_point_feature(
        self,
        input_dict,
        feature_source="projected",
        normalize=True,
    ):
        """按当前 fragment 的 backbone 输入点顺序返回逐点特征。

        ``projected`` 导出 DINO 对齐 ``[N,Cd]`` 特征；``backbone`` 导出
        concat 后的 ``[N,C3]`` 特征。投影仅在目标层 token 上执行。
        """
        _, _, level, distill_feat = self._encode(input_dict)
        if feature_source == "projected":
            token_feature = self.student_projector(distill_feat)
        elif feature_source == "backbone":
            token_feature = distill_feat
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
        normalize_feature=True,
    ):
        """根据 ``return_point_feature`` 在训练蒸馏和测试导出路径间切换。

        训练返回 ``loss`` 以及紧凑的 ``tok/edge/patch`` 统计项；
        导出路径只返回 ``point_feature``，因此测试数据不需要携带任何 DINO 字段。
        """
        # 特征导出是独立路径：不建 token-patch 边，也不计算 teacher loss。
        if return_point_feature:
            return {
                "point_feature": self.extract_point_feature(
                    input_dict,
                    feature_source=feature_source,
                    normalize=normalize_feature,
                )
            }

        required = {"dino_feature", "dino_patch_index", "dino_valid", "dino_offset"}
        missing = required.difference(input_dict)
        if missing:
            raise KeyError(f"HPSD input is missing fields: {sorted(missing)}")

        point, hierarchy, level, distill_feat = self._encode(input_dict)
        level_feat = level.point.feat
        dino_feature = input_dict["dino_feature"]
        if dino_feature.ndim != 2 or dino_feature.shape[1] != self.teacher_channels:
            raise ValueError(
                f"Expected DINO feature shape [P, {self.teacher_channels}], got "
                f"{tuple(dino_feature.shape)}"
            )

        # 目标层与全部深层特征先对齐 concat，再只建一次边、聚合一次 patch。
        teacher = F.normalize(dino_feature.float(), dim=-1)
        edges = build_token_patch_edges(
            input_to_level=level.input_to_level,
            patch_index=input_dict["dino_patch_index"],
            valid=input_dict["dino_valid"],
            num_tokens=level_feat.shape[0],
            num_patches=dino_feature.shape[0],
            validate_mapping=self.validate_mapping,
        )
        patch_feat, used_patch, _ = aggregate_tokens_to_patches(
            distill_feat, edges, edge_weight=self.edge_weight
        )
        patch_pred = self.student_projector(patch_feat)
        if patch_feat.shape[0] == 0:
            # distill_feat[:0] 保留 concat backbone 与 projector 的计算图连接。
            loss = patch_pred.float().sum() * 0.0
        else:
            student = F.normalize(patch_pred.float(), dim=-1)
            loss_per_patch = 1.0 - torch.sum(
                student * teacher[used_patch], dim=-1
            )
            if self.sample_balanced:
                loss = self._sample_balanced_mean(
                    loss_per_patch, used_patch, input_dict["dino_offset"]
                )
            else:
                loss = loss_per_patch.mean()

        # 统计键刻意采用简称，避免 InformationWriter 的单行训练日志过长。
        # loss 是训练器反向传播所需的唯一损失；不再重复返回 patch_loss。
        result = {
            "loss": loss,
            "tok": level_feat.new_tensor(level_feat.shape[0]),
            "edge": level_feat.new_tensor(edges.num_edges),
            "patch": level_feat.new_tensor(used_patch.shape[0]),
        }
        if return_point:
            result["point"] = point
            result["hierarchy"] = hierarchy
        return result
