"""在保持 HPSD 原始蒸馏不变的前提下，为正射不可视 token 传播监督。"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from pointspace.models.builder import MODELS

from ..hpsd.hpsd_v1m1 import HierarchicalPatchSetDistiller
from .ops import (
    aggregate_patch_teacher_to_tokens,
    chunked_topk_cosine,
    compute_token_visibility,
    height_stratified_cap,
)


class VisibilityReliableSupervisor(nn.Module):
    """DINO 锚定的低维校准与样本内不可视 token 软传播分支。"""

    def __init__(
        self,
        in_channels,
        teacher_channels=1024,
        propagation_channels=128,
        hidden_channels=256,
        projection_seed=3407,
        mode="local",
        source_q=0.6,
        target_q=0.0,
        min_source_points=4,
        min_source_patches=1,
        source_purity=None,
        topk=8,
        temperature=0.1,
        max_sources=512,
        max_targets=1024,
        query_chunk_size=256,
        lambda_cal=0.05,
        lambda_local=0.02,
        edge_weight="sqrt_count",
    ):
        super().__init__()
        self.mode = str(mode)
        if self.mode not in {"calibrate", "local"}:
            raise ValueError("mode must be 'calibrate' or 'local'")
        self.source_q = float(source_q)
        self.target_q = float(target_q)
        self.min_source_points = int(min_source_points)
        self.min_source_patches = int(min_source_patches)
        self.source_purity = None if source_purity is None else float(source_purity)
        self.topk = int(topk)
        self.temperature = float(temperature)
        self.max_sources = None if max_sources is None else int(max_sources)
        self.max_targets = None if max_targets is None else int(max_targets)
        self.query_chunk_size = int(query_chunk_size)
        self.lambda_cal = float(lambda_cal)
        self.lambda_local = float(lambda_local)
        self.edge_weight = edge_weight
        self.propagation_channels = int(propagation_channels)

        if in_channels <= 0 or teacher_channels <= 0:
            raise ValueError("in_channels and teacher_channels must be positive")
        if self.propagation_channels <= 0 or hidden_channels <= 0:
            raise ValueError("propagation and hidden channels must be positive")
        if self.propagation_channels > int(teacher_channels):
            raise ValueError("propagation_channels cannot exceed teacher_channels")
        if not 0.0 <= self.source_q <= 1.0 or not 0.0 <= self.target_q <= 1.0:
            raise ValueError("source_q and target_q must be within [0, 1]")
        if self.source_purity is not None and not -1.0 <= self.source_purity <= 1.0:
            raise ValueError("source_purity must be within [-1, 1]")
        if self.topk <= 0 or self.query_chunk_size <= 0:
            raise ValueError("topk and query_chunk_size must be positive")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.lambda_cal < 0 or self.lambda_local < 0:
            raise ValueError("loss weights must be non-negative")

        self.prop_head = nn.Sequential(
            nn.LayerNorm(int(in_channels)),
            nn.Linear(int(in_channels), int(hidden_channels)),
            nn.GELU(),
            nn.Linear(int(hidden_channels), self.propagation_channels),
        )

        # 固定正交投影定义稳定的 teacher 低维坐标系，并随 checkpoint 保存。
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(projection_seed))
        random_matrix = torch.randn(
            int(teacher_channels), self.propagation_channels, generator=generator
        )
        projection, _ = torch.linalg.qr(random_matrix, mode="reduced")
        self.register_buffer("teacher_projection", projection.contiguous())

    @staticmethod
    def _zero_loss(feature):
        return feature.float().sum() * 0.0

    def _build_sources(self, stats, visibility):
        if stats.token.numel() == 0:
            return stats.token
        keep = visibility[stats.token] >= self.source_q
        keep &= stats.point_count >= self.min_source_points
        keep &= stats.patch_count >= self.min_source_patches
        if self.source_purity is not None:
            keep &= stats.purity >= self.source_purity
        return stats.token[keep]

    def _calibration_loss(self, student, stats, source_token):
        if source_token.numel() == 0:
            return self._zero_loss(student), student.new_tensor(0.0)
        # stats.token 已排序；searchsorted 将 source 原始行号映射到紧凑 teacher 行。
        compact_index = torch.searchsorted(stats.token, source_token)
        teacher128 = F.normalize(
            stats.feature[compact_index] @ self.teacher_projection.float(),
            dim=-1,
            eps=1e-12,
        )
        cosine = torch.sum(student[source_token].float() * teacher128, dim=-1)
        return (1.0 - cosine).mean(), cosine.detach().mean()

    def _local_loss(self, student, source_token, target_token, token_batch, coord):
        zero = self._zero_loss(student)
        if self.mode != "local" or source_token.numel() == 0 or target_token.numel() == 0:
            return zero, 0, student.new_tensor(0.0)

        sample_losses = []
        entropy_sum = student.new_tensor(0.0)
        accepted = 0
        for sample in torch.unique(token_batch, sorted=True):
            source = source_token[token_batch[source_token] == sample]
            target = target_token[token_batch[target_token] == sample]
            if source.numel() == 0 or target.numel() == 0:
                continue
            source = height_stratified_cap(source, coord, self.max_sources)
            target = height_stratified_cap(target, coord, self.max_targets)
            target_train = student[target]
            with torch.no_grad():
                similarity, neighbor = chunked_topk_cosine(
                    target_train.detach().float(),
                    student[source].detach().float(),
                    topk=self.topk,
                    chunk_size=self.query_chunk_size,
                )
                weight = torch.softmax(similarity / self.temperature, dim=-1)
                reference = F.normalize(
                    torch.sum(
                        weight.unsqueeze(-1)
                        * student[source][neighbor].detach().float(),
                        dim=1,
                    ),
                    dim=-1,
                    eps=1e-12,
                )
                if weight.shape[1] > 1:
                    entropy = -torch.sum(
                        weight * torch.log(weight.clamp_min(1e-12)), dim=-1
                    ) / math.log(weight.shape[1])
                else:
                    entropy = torch.zeros_like(weight[:, 0])
            sample_losses.append(
                (1.0 - torch.sum(target_train.float() * reference, dim=-1)).mean()
            )
            entropy_sum = entropy_sum + entropy.sum()
            accepted += int(target.shape[0])

        if not sample_losses:
            return zero, 0, student.new_tensor(0.0)
        return (
            torch.stack(sample_losses).mean(),
            accepted,
            entropy_sum / max(accepted, 1),
        )

    def forward(self, context, input_dict):
        level = context.level
        num_tokens = int(context.distill_feat.shape[0])
        visibility, valid_count, total_count = compute_token_visibility(
            level.input_to_level,
            input_dict["dino_valid"],
            num_tokens,
        )
        teacher_stats = aggregate_patch_teacher_to_tokens(
            context.teacher, context.edges, edge_weight=self.edge_weight
        )
        source_token = self._build_sources(teacher_stats, visibility)
        target_token = torch.nonzero(
            (visibility <= self.target_q) & (total_count > 0), as_tuple=False
        ).flatten()

        student = F.normalize(self.prop_head(context.distill_feat).float(), dim=-1)
        cal_loss, projection_cosine = self._calibration_loss(
            student, teacher_stats, source_token
        )
        token_batch = level.point.batch.long()
        local_loss, accepted, neighbor_entropy = self._local_loss(
            student,
            source_token,
            target_token,
            token_batch,
            level.point.coord,
        )
        total_loss = self.lambda_cal * cal_loss
        if self.mode == "local":
            total_loss = total_loss + self.lambda_local * local_loss
        return {
            "loss": total_loss,
            "cal": cal_loss,
            "local": local_loss,
            "source_count": student.new_tensor(source_token.numel()),
            "target_count": student.new_tensor(target_token.numel()),
            "accepted_count": student.new_tensor(accepted),
            "projection_cosine": projection_cosine,
            "neighbor_entropy": neighbor_entropy.detach(),
            "valid_token_count": student.new_tensor(int((valid_count > 0).sum())),
        }


@MODELS.register_module("HPSD-VRSR-v1m1")
class HPSDVRSRDistiller(HierarchicalPatchSetDistiller):
    """保持 HPSD 主损失和导出路径不变，并附加训练期 VRSR 分支。"""

    def __init__(self, vrsr=None, **kwargs):
        super().__init__(**kwargs)
        self.vrsr = VisibilityReliableSupervisor(
            in_channels=self.projector_in_channels,
            teacher_channels=self.teacher_channels,
            edge_weight=self.edge_weight,
            **({} if vrsr is None else vrsr),
        )

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

        hpsd_result, context = self.forward_train(
            input_dict, return_point=return_point, return_context=True
        )
        vrsr_result = self.vrsr(context, input_dict)
        result = dict(hpsd_result)
        result["loss"] = hpsd_result["loss"] + vrsr_result["loss"]
        # 只返回标量简称，避免 InformationWriter 的单行日志过长。
        result.update(
            hpsd=hpsd_result["loss"].detach(),
            cal=vrsr_result["cal"].detach(),
            loc=vrsr_result["local"].detach(),
            src=vrsr_result["source_count"].detach(),
            tgt=vrsr_result["target_count"].detach(),
            acc=vrsr_result["accepted_count"].detach(),
            pcos=vrsr_result["projection_cosine"].detach(),
            ent=vrsr_result["neighbor_entropy"].detach(),
        )
        return result
