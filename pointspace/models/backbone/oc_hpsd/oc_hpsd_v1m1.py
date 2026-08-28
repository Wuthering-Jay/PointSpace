"""一次训练的观测条件 HPSD 与上下文语义补全。"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter

from pointspace.models.builder import MODELS
from pointspace.models.utils.misc import offset2batch

from ..hpsd.hpsd_v1m1 import HierarchicalPatchSetDistiller
from .ops import (
    GeometryGuidedMaskGenerator,
    aggregate_teacher_to_tokens_routed,
    aggregate_tokens_to_patches_routed,
    build_routed_token_patch_edges,
    compute_token_route_stats,
)


@MODELS.register_module("OC-HPSD-v1m1")
class ObservationConditionedHPSD(HierarchicalPatchSetDistiller):
    """在 HPSD 可靠锚定之外，以真实输入 masking 训练 CSC。

    训练时，高可信可视点被划分为未遮蔽 anchor 和 simulated-missing 两路。
    Anchor 使用 observation-weighted patch-set HPSD；simulated-missing 点的
    embedded feature 由 backbone mask token 替换，再由 F3/F4 等深层上下文
    恢复其真实 DINO teacher。真实不可视点不构造伪 teacher。

    ``set_train_progress`` 由 ``ObservationCurriculumHook`` 在同一个训练 run
    内更新。测试特征导出直接复用 HPSD，不生成 mask 或 CSC 张量。
    """

    def __init__(
        self,
        masking=None,
        completion_hidden_channels=1024,
        completion_min_points=1,
        completion_min_mask_fraction=0.5,
        completion_soft_weight=False,
        completion_full_weight_fraction=0.5,
        report_compact_stats=False,
        mask_rate=0.30,
        lambda_csc=0.20,
        curriculum_start=0.10,
        curriculum_warmup=0.10,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not self.fuse_deeper_features:
            raise ValueError("OC-HPSD requires fuse_deeper_features=True")
        if self.distill_level + 1 >= len(self.level_channels):
            raise ValueError("OC-HPSD requires at least one level deeper than distill_level")

        self.context_in_channels = sum(
            self.level_channels[self.distill_level + 1 :]
        )
        self.completion_hidden_channels = int(completion_hidden_channels)
        self.completion_min_points = int(completion_min_points)
        self.completion_min_mask_fraction = float(completion_min_mask_fraction)
        self.completion_soft_weight = bool(completion_soft_weight)
        self.completion_full_weight_fraction = float(
            completion_full_weight_fraction
        )
        self.report_compact_stats = bool(report_compact_stats)
        self.target_mask_rate = float(mask_rate)
        self.target_lambda_csc = float(lambda_csc)
        self.curriculum_start = float(curriculum_start)
        self.curriculum_warmup = float(curriculum_warmup)
        if self.completion_hidden_channels <= 0 or self.completion_min_points < 1:
            raise ValueError("Completion hidden channels and min points must be positive")
        if not 0 <= self.completion_min_mask_fraction <= 1:
            raise ValueError("completion_min_mask_fraction must be within [0, 1]")
        if not 0 < self.completion_full_weight_fraction <= 1:
            raise ValueError("completion_full_weight_fraction must be within (0, 1]")
        if not 0 <= self.target_mask_rate <= 1:
            raise ValueError("mask_rate must be within [0, 1]")
        if self.target_lambda_csc < 0:
            raise ValueError("lambda_csc must be non-negative")
        if not 0 <= self.curriculum_start <= 1 or self.curriculum_warmup < 0:
            raise ValueError("Invalid curriculum fractions")
        if self.curriculum_start + self.curriculum_warmup > 1:
            raise ValueError("curriculum_start + curriculum_warmup cannot exceed 1")

        self.mask_generator = GeometryGuidedMaskGenerator(
            **({} if masking is None else masking)
        )
        self.completion_projector = nn.Sequential(
            nn.LayerNorm(self.context_in_channels),
            nn.Linear(self.context_in_channels, self.completion_hidden_channels),
            nn.GELU(),
            nn.Linear(self.completion_hidden_channels, self.teacher_channels),
        )
        self._train_progress = 1.0

        embedding = getattr(self.backbone, "embedding", None)
        if (
            embedding is not None
            and hasattr(embedding, "mask_token")
            and embedding.mask_token is None
        ):
            raise ValueError(
                "OC-HPSD backbone must be configured with mask_token=True"
            )

    def set_train_progress(self, progress):
        """设置当前 run 的归一化进度；断点恢复时可由 hook 重建。"""
        self._train_progress = min(max(float(progress), 0.0), 1.0)

    def _curriculum_scale(self):
        progress = self._train_progress
        if progress <= self.curriculum_start:
            return 0.0
        if self.curriculum_warmup == 0:
            return 1.0
        return min(
            max(
                (progress - self.curriculum_start) / self.curriculum_warmup,
                0.0,
            ),
            1.0,
        )

    @staticmethod
    def _sample_balanced_token_mean(loss, token, token_batch):
        if loss.numel() == 0:
            return loss.sum()
        batch = token_batch[token]
        used_batch, token_to_batch = torch.unique(
            batch, sorted=True, return_inverse=True
        )
        loss_sum = torch_scatter.scatter_sum(
            loss, token_to_batch, dim=0, dim_size=used_batch.shape[0]
        )
        loss_count = torch_scatter.scatter_sum(
            torch.ones_like(loss),
            token_to_batch,
            dim=0,
            dim_size=used_batch.shape[0],
        )
        return (loss_sum / loss_count.clamp_min(1.0)).mean()

    @staticmethod
    def _sample_balanced_weighted_token_mean(loss, weight, token, token_batch):
        """先在每个样本内归一化连续权重，再对样本等权平均。"""
        if loss.numel() == 0:
            return loss.sum()
        batch = token_batch[token]
        used_batch, token_to_batch = torch.unique(
            batch, sorted=True, return_inverse=True
        )
        loss_sum = torch_scatter.scatter_sum(
            loss * weight,
            token_to_batch,
            dim=0,
            dim_size=used_batch.shape[0],
        )
        weight_sum = torch_scatter.scatter_sum(
            weight,
            token_to_batch,
            dim=0,
            dim_size=used_batch.shape[0],
        )
        return (loss_sum / weight_sum.clamp_min(1e-12)).mean()

    def _observation_forward(self, input_dict, return_point=False):
        required = {
            "coord",
            "offset",
            "dino_feature",
            "image_patch_index",
            "image_valid",
            "dino_offset",
        }
        missing = required.difference(input_dict)
        if missing:
            raise KeyError(f"OC-HPSD input is missing fields: {sorted(missing)}")

        valid = input_dict["image_valid"].bool()
        observability = input_dict.get("image_observability")
        if observability is None:
            observability = valid.float()
        else:
            observability = observability.float()
        input_batch = offset2batch(input_dict["offset"].long())
        scale = self._curriculum_scale()
        current_mask_rate = self.target_mask_rate * scale
        mask_result = self.mask_generator(
            coord=input_dict["coord"],
            batch=input_batch,
            valid=valid,
            observability=observability,
            mask_rate=current_mask_rate,
            return_stats=self.report_compact_stats,
        )
        if self.report_compact_stats:
            simulated_mask, mask_stats = mask_result
        else:
            simulated_mask = mask_result
            mask_stats = None

        masked_input = dict(input_dict)
        masked_input["mask"] = simulated_mask
        point, hierarchy, level, distill_feat = self._encode(masked_input)
        level_feature = level.point.feat
        teacher = input_dict["dino_feature"]
        if teacher.ndim != 2 or teacher.shape[1] != self.teacher_channels:
            raise ValueError(
                f"Expected DINO feature shape [P, {self.teacher_channels}], got "
                f"{tuple(teacher.shape)}"
            )
        teacher = F.normalize(teacher.float(), dim=-1, eps=1e-12)

        edges = build_routed_token_patch_edges(
            input_to_level=level.input_to_level,
            patch_index=input_dict["image_patch_index"],
            valid=valid,
            observability=observability,
            simulated_mask=simulated_mask,
            num_tokens=level_feature.shape[0],
            num_patches=teacher.shape[0],
            validate_mapping=self.validate_mapping,
        )

        patch_feature, used_patch, _ = aggregate_tokens_to_patches_routed(
            distill_feat,
            edges,
            route="anchor",
            edge_weight=self.edge_weight,
        )
        if patch_feature.shape[0] == 0:
            patch_prediction = self.student_projector(patch_feature)
            hpsd_loss = patch_prediction.float().sum() * 0.0
        else:
            patch_loss = self._projector_cosine_loss(
                self.student_projector,
                patch_feature,
                teacher[used_patch],
            )
            if self.sample_balanced:
                hpsd_loss = self._sample_balanced_mean(
                    patch_loss, used_patch, input_dict["dino_offset"]
                )
            else:
                hpsd_loss = patch_loss.mean()

        used_token, token_teacher, _ = aggregate_teacher_to_tokens_routed(
            teacher,
            edges,
            route="masked",
            edge_weight=self.edge_weight,
        )
        route_stats = compute_token_route_stats(
            level.input_to_level,
            valid,
            simulated_mask,
            level_feature.shape[0],
            observability=observability if self.completion_soft_weight else None,
        )
        if used_token.numel() > 0:
            mask_fraction = (
                route_stats.masked_count[used_token]
                / route_stats.valid_count[used_token].clamp_min(1.0)
            )
            target_keep = (
                route_stats.masked_count[used_token] >= self.completion_min_points
            ) & (mask_fraction >= self.completion_min_mask_fraction)
            target_token = used_token[target_keep]
            target_teacher = token_teacher[target_keep]
            if self.completion_soft_weight:
                masked_count = route_stats.masked_count[target_token]
                masked_mean_q = (
                    route_stats.masked_q_sum[target_token]
                    / masked_count.clamp_min(1.0)
                )
                target_weight = (
                    mask_fraction[target_keep]
                    / self.completion_full_weight_fraction
                ).clamp(max=1.0)
                target_weight = (
                    target_weight
                    * masked_mean_q
                    * torch.sqrt(masked_count.clamp_min(1.0))
                )
        else:
            target_token = used_token
            target_teacher = token_teacher
            if self.completion_soft_weight:
                target_weight = token_teacher.new_empty(0)

        # distill_feat 的前一段为 F2，后续通道依次为上采样后的 F3/F4。
        context_feature = distill_feat[:, self.level_channels[self.distill_level] :]
        if target_token.numel() == 0:
            completion_prediction = self.completion_projector(
                context_feature[target_token]
            )
            csc_loss = completion_prediction.float().sum() * 0.0
        else:
            token_loss = self._projector_cosine_loss(
                self.completion_projector,
                context_feature[target_token],
                target_teacher,
            )
            if self.completion_soft_weight and self.sample_balanced:
                csc_loss = self._sample_balanced_weighted_token_mean(
                    token_loss,
                    target_weight,
                    target_token,
                    level.point.batch.long(),
                )
            elif self.completion_soft_weight:
                csc_loss = (
                    token_loss * target_weight
                ).sum() / target_weight.sum().clamp_min(1e-12)
            elif self.sample_balanced:
                csc_loss = self._sample_balanced_token_mean(
                    token_loss, target_token, level.point.batch.long()
                )
            else:
                csc_loss = token_loss.mean()

        current_lambda = self.target_lambda_csc * scale
        total_loss = hpsd_loss + current_lambda * csc_loss
        anchor_edge_count = (edges.anchor_count > 0).sum()
        result = {
            "loss": total_loss,
            "hpsd": hpsd_loss.detach(),
            "csc": csc_loss.detach(),
            "mr": level_feature.new_tensor(current_mask_rate),
            "tok": level_feature.new_tensor(level_feature.shape[0]),
            "edge": anchor_edge_count.to(level_feature.dtype),
            "patch": level_feature.new_tensor(used_patch.shape[0]),
            "anc": (~simulated_mask & valid).sum().to(level_feature.dtype),
            "msk": simulated_mask.sum().to(level_feature.dtype),
            "ctok": level_feature.new_tensor(target_token.shape[0]),
        }
        if self.report_compact_stats:
            masked_count = simulated_mask.sum().to(level_feature.dtype)
            candidate_count = mask_stats.candidate_count.to(level_feature.dtype)
            requested_count = mask_stats.requested_count.to(level_feature.dtype)
            result.pop("anc")
            result["mr"] = masked_count / candidate_count.clamp_min(1.0)
            result["mu"] = masked_count / requested_count.clamp_min(1.0)
            result["cr"] = level_feature.new_tensor(target_token.shape[0]) / max(
                int(used_token.shape[0]), 1
            )
        if return_point:
            result["point"] = point
            result["hierarchy"] = hierarchy
        return result

    def forward(
        self,
        input_dict,
        return_point=False,
        return_point_feature=False,
        feature_source="projected",
        normalize_feature=True,
    ):
        if return_point_feature:
            return super().forward(
                input_dict,
                return_point=return_point,
                return_point_feature=True,
                feature_source=feature_source,
                normalize_feature=normalize_feature,
            )
        return self._observation_forward(input_dict, return_point=return_point)
