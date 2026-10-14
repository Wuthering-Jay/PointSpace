"""Sonata mask/roll/local-view distillation with a LitePT backbone.

The original three objectives and two-global/four-local view topology are
preserved. Point matching may be capped for ALS memory, without changing the
target or loss on the retained correspondences.
"""

from copy import deepcopy
import math

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

import pointops
from pointspace.models.builder import MODELS, build_model
from pointspace.models.modules import PointModel


class PrototypeHead(nn.Module):
    def __init__(self, channels, hidden, embedding, prototypes):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(channels, hidden), nn.GELU(), nn.Linear(hidden, embedding))
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.trunc_normal_(layer.weight, std=0.02)
                nn.init.zeros_(layer.bias)
        self.prototypes = nn.utils.parametrizations.weight_norm(
            nn.Linear(embedding, prototypes, bias=False))
        self.prototypes.parametrizations.weight.original0.data.fill_(1)
        self.prototypes.parametrizations.weight.original0.requires_grad_(False)

    def forward(self, x):
        return self.prototypes(F.normalize(self.mlp(x).float(), dim=-1))


@MODELS.register_module("Sonata-LitePT-v1m1")
class SonataLitePT(PointModel):
    def __init__(
        self, backbone, backbone_out_channels=72, head_hidden_channels=4096,
        head_embed_channels=512, head_num_prototypes=4096,
        mask_group_size=4.0, mask_ratio=0.5,
        mask_group_size_start=None, mask_ratio_start=None,
        teacher_temp_start=None, warmup_ratio=0.05,
        num_local_views=4, mask_loss_weight=0.25,
        roll_mask_loss_weight=0.25, unmask_loss_weight=0.5,
        matching_max_radius=0.75, matching_max_pair=2048,
        head_chunk_size=256,
        teacher_temp=0.07, student_temp=0.1, sinkhorn_iterations=3,
        momentum_base=0.996, momentum_final=1.0,
    ):
        super().__init__()
        if backbone.get("type") != "LitePT-v1m4" or backbone.get("enc_mode") is not False:
            raise ValueError("Sonata-LitePT-v1m1 requires full LitePT-v1m4")
        if backbone_out_channels != backbone["dec_channels"][0]:
            raise ValueError("backbone_out_channels must match LitePT P0 width")
        if not (0 < mask_ratio < 1) or min(mask_group_size, matching_max_radius,
                                           matching_max_pair, teacher_temp,
                                           student_temp, sinkhorn_iterations) <= 0:
            raise ValueError("Invalid Sonata mask/matching/temperature settings")
        self.student = nn.ModuleDict(dict(
            backbone=build_model(backbone),
            mask_head=PrototypeHead(backbone_out_channels, head_hidden_channels,
                                    head_embed_channels, head_num_prototypes),
            unmask_head=PrototypeHead(backbone_out_channels, head_hidden_channels,
                                      head_embed_channels, head_num_prototypes),
        ))
        self.teacher = deepcopy(self.student)
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.mask_token = nn.Parameter(torch.zeros(1, backbone["in_channels"]))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.mask_group_size = float(mask_group_size)
        self.mask_ratio = float(mask_ratio)
        self.mask_group_size_base = float(mask_group_size)
        self.mask_ratio_base = float(mask_ratio)
        self.mask_group_size_start = float(mask_group_size_start if mask_group_size_start is not None else mask_group_size)
        self.mask_ratio_start = float(mask_ratio_start if mask_ratio_start is not None else mask_ratio)
        self.teacher_temp_start = float(teacher_temp_start if teacher_temp_start is not None else teacher_temp)
        self.teacher_temp_base = float(teacher_temp)
        self.warmup_ratio = float(warmup_ratio)
        self.num_local_views = int(num_local_views)
        self.mask_loss_weight = float(mask_loss_weight)
        self.roll_mask_loss_weight = float(roll_mask_loss_weight)
        self.unmask_loss_weight = float(unmask_loss_weight)
        if self.num_local_views < 0 or min(mask_loss_weight, roll_mask_loss_weight,
                                            unmask_loss_weight) < 0:
            raise ValueError("Invalid Sonata view count or loss weights")
        if self.num_local_views == 0 and self.unmask_loss_weight:
            raise ValueError("Unmasked local-view loss requires local views")
        self.matching_max_radius = float(matching_max_radius)
        self.matching_max_pair = int(matching_max_pair)
        self.head_chunk_size = int(head_chunk_size)
        if self.head_chunk_size <= 0:
            raise ValueError("head_chunk_size must be positive")
        self.teacher_temp = float(teacher_temp)
        self.student_temp = float(student_temp)
        self.sinkhorn_iterations = int(sinkhorn_iterations)
        self.momentum_base = float(momentum_base)
        self.momentum_final = float(momentum_final)
        self._ema_updates = 0
        self._total_updates = 1

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        return self

    def before_train(self):
        trainer = self.trainer
        accumulation = trainer.cfg.gradient_accumulation_steps
        self._total_updates = max(1, int(trainer.cfg.scheduler.total_steps))
        self._ema_updates = int(trainer.start_epoch) * len(trainer.train_loader) // accumulation

    def after_step(self):
        # ModelHook runs for every micro-batch. Only update teacher after a
        # completed optimizer step; otherwise gradient accumulation changes EMA.
        if (self.trainer._gradient_accumulation_counter != 0
                or not getattr(self.trainer, "_optimizer_step_succeeded", True)):
            return
        progress = min(1.0, self._ema_updates / self._total_updates)
        momentum = self.momentum_final - (self.momentum_final - self.momentum_base) * (1 + math.cos(math.pi * progress)) / 2
        with torch.no_grad():
            for teacher, student in zip(self.teacher.parameters(), self.student.parameters()):
                teacher.lerp_(student, 1 - momentum)
            for teacher, student in zip(self.teacher.buffers(), self.student.buffers()):
                teacher.copy_(student)
        self._ema_updates += 1

    def before_step(self):
        warmup_updates = max(1, int(self._total_updates * self.warmup_ratio))
        progress = min(1.0, self._ema_updates / warmup_updates)
        self.mask_group_size = (self.mask_group_size_start + progress *
                                (self.mask_group_size_base - self.mask_group_size_start))
        self.mask_ratio = (self.mask_ratio_start + progress *
                           (self.mask_ratio_base - self.mask_ratio_start))
        self.teacher_temp = (self.teacher_temp_start + progress *
                             (self.teacher_temp_base - self.teacher_temp_start))

    def _mask(self, view):
        coord, offset = view["coord"], view["offset"]
        mask = torch.zeros(len(coord), dtype=torch.bool, device=coord.device)
        start = 0
        for end in offset.tolist():
            block = coord[start:end].float()
            cell = torch.floor((block - block.amin(0)) / self.mask_group_size).long()
            _, inverse = torch.unique(cell, dim=0, return_inverse=True)
            count = int(inverse.max()) + 1
            chosen = torch.randperm(count, device=coord.device)[:max(1, int(count * self.mask_ratio))]
            mask[start:end] = torch.isin(inverse, chosen)
            start = end
        return mask

    @staticmethod
    def _sinkhorn(logits, temperature, iterations):
        # A single scalar shift preserves the assignment ratios.
        scores = logits.float() / temperature
        maximum = scores.amax()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
        scores = scores - maximum
        q = scores.exp().t()
        total = q.sum()
        samples = torch.tensor(q.shape[1], device=q.device, dtype=q.dtype)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(total)
            dist.all_reduce(samples)
        q /= total.clamp_min(1e-12)
        prototypes = q.shape[0]
        for _ in range(iterations):
            row_sum = q.sum(dim=1, keepdim=True)
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(row_sum)
            q /= row_sum.clamp_min(1e-12)
            q /= prototypes
            q /= q.sum(dim=0, keepdim=True).clamp_min(1e-12)
            q /= samples
        return (q * samples).t()

    @torch.no_grad()
    def _pairs(self, first, second):
        c1, c2 = first["origin_coord"].float(), second["origin_coord"].float()
        o1, o2 = first["offset"].int(), second["offset"].int()
        if len(o1) != len(o2):
            raise ValueError("Sonata views must contain the same source blocks")
        # The common origin_coord frame is produced before independent view augmentation.
        nearest = pointops.knn_query_idx(1, c2.contiguous(), o2,
                                         c1.contiguous(), o1).long().flatten()
        valid = (c1 - c2[nearest]).square().sum(-1) <= self.matching_max_radius ** 2
        left, right = [], []
        start = 0
        for end in o1.tolist():
            idx = torch.nonzero(valid[start:end], as_tuple=False).flatten() + start
            if len(idx) > self.matching_max_pair:
                idx = idx[torch.randperm(len(idx), device=idx.device)[:self.matching_max_pair]]
            left.append(idx)
            right.append(nearest[idx])
            start = end
        return torch.cat(left), torch.cat(right), int(valid.sum())

    def _assignment_loss(self, student_feature, teacher_feature,
                         student_index, teacher_index, head):
        if dist.is_available() and dist.is_initialized():
            available = student_index.new_tensor(student_index.numel())
            dist.all_reduce(available, op=dist.ReduceOp.MIN)
            if available.item() == 0:
                raise RuntimeError("Sonata has no matched points on at least one rank")
        if student_index.numel() == 0:
            return None
        with torch.no_grad():
            target = self.teacher[head](teacher_feature[teacher_index])
            assignment = self._sinkhorn(
                target, self.teacher_temp, self.sinkhorn_iterations)
        def chunk_loss(feature, soft_target):
            prediction = self.student[head](feature)
            return -(soft_target * F.log_softmax(
                prediction.float() / self.student_temp, dim=-1)).sum()
        total = student_feature.new_zeros((), dtype=torch.float32)
        for index_chunk, target_chunk in zip(
                student_index.split(self.head_chunk_size),
                assignment.split(self.head_chunk_size)):
            feature_chunk = student_feature[index_chunk]
            if self.training:
                total = total + checkpoint(
                    chunk_loss, feature_chunk, target_chunk,
                    use_reentrant=False, preserve_rng_state=False)
            else:
                total = total + chunk_loss(feature_chunk, target_chunk)
        return total / student_index.numel()

    @staticmethod
    def _view(data_dict, prefix):
        return {key[len(prefix):]: value for key, value in data_dict.items()
                if key.startswith(prefix)}

    def forward(self, data_dict, return_point_feature=False,
                feature_source="backbone", normalize_feature=False,
                _return_student_view1=False):
        if return_point_feature:
            if feature_source not in ("backbone", "student", "teacher"):
                raise ValueError("Sonata feature_source must be backbone/student/teacher")
            module = self.teacher if feature_source == "teacher" else self.student
            feature = module["backbone"](data_dict).feat
            if "feature_point_mask" in data_dict:
                feature = feature[data_dict["feature_point_mask"]]
            if normalize_feature:
                feature = F.normalize(feature.float(), dim=-1)
            return {"point_feature": feature}

        views = [self._view(data_dict, prefix) for prefix in ("view1_", "view2_")]
        local_views = [self._view(data_dict, f"local{number + 1}_")
                       for number in range(self.num_local_views)]
        if any(not view for view in views + local_views):
            raise ValueError("Sonata requires two global and the configured local views")
        masks = [self._mask(view) for view in views]
        with torch.no_grad():
            teacher_features = [self.teacher["backbone"](view).feat for view in views]
        student_features = []
        for view, mask in zip(views, masks):
            masked_view = dict(view)
            masked_view["feat"] = torch.where(mask[:, None],
                self.mask_token.to(view["feat"].dtype), view["feat"])
            student_features.append(self.student["backbone"](masked_view).feat)

        def mean_or_zero(losses):
            return (torch.stack(losses).mean() if losses else
                    student_features[0].sum() * 0)

        # Masked global view is predicted from its *own* unmasked EMA view.
        mask_losses = []
        if self.mask_loss_weight:
            for student_feature, teacher_feature, view in zip(
                    student_features, teacher_features, views):
                index = torch.randperm(len(student_feature), device=student_feature.device)
                index = index[:self.matching_max_pair]
                mask_losses.append(self._assignment_loss(
                    student_feature, teacher_feature, index, index, "mask_head"))
        loss_mask = mean_or_zero(mask_losses)

        # Roll swaps the two *teacher* global views within each source tile.
        roll_losses = []
        pair_count = 0
        candidate_count = 0
        if self.roll_mask_loss_weight:
            for source, target in ((0, 1), (1, 0)):
                left, right, candidates = self._pairs(views[source], views[target])
                pair_count += len(left)
                candidate_count += candidates
                value = self._assignment_loss(
                    student_features[source], teacher_features[target],
                    left, right, "mask_head")
                if value is not None:
                    roll_losses.append(value)
        loss_roll = mean_or_zero(roll_losses)

        # All local students are matched to the first, principal global EMA view.
        local_losses = []
        if self.unmask_loss_weight:
            for local_view in local_views:
                local_feature = self.student["backbone"](local_view).feat
                left, right, candidates = self._pairs(local_view, views[0])
                pair_count += len(left)
                candidate_count += candidates
                value = self._assignment_loss(
                    local_feature, teacher_features[0], left, right,
                    "unmask_head")
                if value is not None:
                    local_losses.append(value)
        loss_unmask = mean_or_zero(local_losses)
        if self.roll_mask_loss_weight and not roll_losses:
            raise RuntimeError("Sonata found no rolled global-view correspondences")
        if self.unmask_loss_weight and not local_losses:
            raise RuntimeError("Sonata found no local-to-global correspondences")
        loss = (self.mask_loss_weight * loss_mask +
                self.roll_mask_loss_weight * loss_roll +
                self.unmask_loss_weight * loss_unmask)
        output = dict(loss=loss, mask=loss_mask.detach(), roll=loss_roll.detach(),
                      unmask=loss_unmask.detach(),
                      matched_pairs=views[0]["offset"].new_tensor(pair_count),
                      matching_fraction=views[0]["coord"].new_tensor(
                          pair_count / max(candidate_count, 1)),
                      mask_fraction=torch.stack([m.float().mean() for m in masks]).mean())
        if _return_student_view1:
            output["_student_view1_feature"] = student_features[0]
        return output
