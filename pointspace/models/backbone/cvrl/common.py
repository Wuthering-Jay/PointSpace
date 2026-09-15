"""Shared visual-return backbone and training machinery for CVRL."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from pointspace.models.backbone.vpsd.vpsd_v1m1 import (
    build_compact_hierarchy,
)
from pointspace.models.losses import build_criteria
from pointspace.models.backbone.rrcl.ops import (
    NUM_RETURN_ROLES,
    build_knn_return_context,
)

from pointspace.models.builder import build_model


CVRL_MODES = frozenset({"visual", "return", "full"})


def _normalize_cvrl_mode(mode):
    mode = str(mode).lower()
    if mode == "backbone":
        mode = "full"
    if mode not in CVRL_MODES:
        raise ValueError(f"CVRL mode must be one of {sorted(CVRL_MODES)}")
    return mode


def _mode_branches(mode):
    mode = _normalize_cvrl_mode(mode)
    return mode in {"visual", "full"}, mode in {"return", "full"}


class VisualReturnBackboneBase(nn.Module):
    """Persistent ``[visual | return]`` backbone used after pretraining."""

    FEATURE_SOURCES = frozenset({"backbone", "full", "visual", "return"})

    def __init__(
        self,
        return_backbone,
        level_channels=(24, 48, 96, 192, 384),
        distill_level=2,
        visual_channels=128,
        return_channels=96,
        mode="full",
    ):
        super().__init__()
        self.return_backbone = build_model(return_backbone)
        if not hasattr(self.return_backbone, "encode") or not hasattr(
            self.return_backbone, "decode_hierarchy"
        ):
            raise TypeError(
                "CVRL backbone requires encode/decode_hierarchy interfaces"
            )
        if bool(getattr(self.return_backbone, "enc_mode", True)):
            raise ValueError("CVRL backbone requires a full encoder-decoder")

        self.level_channels = tuple(int(channel) for channel in level_channels)
        self.distill_level = int(distill_level)
        self.visual_channels = int(visual_channels)
        self.return_channels = int(return_channels)
        self.mode = _normalize_cvrl_mode(mode)
        if len(self.level_channels) != int(self.return_backbone.num_stages):
            raise ValueError("level_channels must describe every encoder stage")
        if not 0 <= self.distill_level < len(self.level_channels):
            raise ValueError("distill_level is outside level_channels")

        self.semantic_in_channels = sum(
            self.level_channels[self.distill_level :]
        )
        self.return_norm = nn.LayerNorm(self.return_channels)
        visual_available, return_available = _mode_branches(self.mode)
        self.register_buffer(
            "_cvrl_trained_branches",
            torch.tensor(
                [visual_available, return_available], dtype=torch.uint8
            ),
            persistent=True,
        )
        self._update_out_channels()

    def _update_out_channels(self):
        if self.mode == "visual":
            self.out_channels = self.visual_channels
        elif self.mode == "return":
            self.out_channels = self.return_channels
        else:
            self.out_channels = self.visual_channels + self.return_channels

    def set_mode(self, mode, reset_availability=False):
        """Select the default output and optionally initialize checkpoint metadata."""
        self.mode = _normalize_cvrl_mode(mode)
        self._update_out_channels()
        if reset_availability:
            visual, return_branch = _mode_branches(self.mode)
            self._cvrl_trained_branches.copy_(
                self._cvrl_trained_branches.new_tensor(
                    [visual, return_branch]
                )
            )

    def mark_mode_trained(self, mode):
        """Persist which branches have actually received pretraining supervision."""
        visual, return_branch = _mode_branches(mode)
        trained = self._cvrl_trained_branches.new_tensor(
            [visual, return_branch]
        )
        self._cvrl_trained_branches.copy_(
            torch.maximum(self._cvrl_trained_branches, trained)
        )

    def assert_mode_available(self, mode):
        """Reject feature extraction from an absent/untrained checkpoint branch."""
        mode = _normalize_cvrl_mode(mode)
        visual, return_branch = _mode_branches(mode)
        available = self._cvrl_trained_branches.detach().cpu().tolist()
        missing = []
        if visual and not available[0]:
            missing.append("visual")
        if return_branch and not available[1]:
            missing.append("return")
        if missing:
            raise RuntimeError(
                f"CVRL checkpoint does not contain trained {missing} branch "
                f"weights; requested mode='{mode}', availability="
                f"visual:{bool(available[0])}/return:{bool(available[1])}."
            )
        return mode

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        availability_key = prefix + "_cvrl_trained_branches"
        if availability_key not in state_dict:
            # Checkpoints created before branch metadata always trained both
            # branches. Inject metadata so they remain strictly loadable.
            state_dict[availability_key] = self._cvrl_trained_branches.new_ones(2)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def encode_hierarchy(self, input_dict):
        """Encode once and expose the native P2/P3/P4 hierarchy."""
        deepest, hierarchy = self.return_backbone.encode(input_dict)
        compact = build_compact_hierarchy(
            deepest,
            target_level=self.distill_level,
            expected_levels=len(self.level_channels),
            validate_mapping=False,
        )
        selected_points = compact.points[self.distill_level :]
        selected_channels = self.level_channels[self.distill_level :]
        for point, channels in zip(selected_points, selected_channels):
            if point.feat.ndim != 2 or point.feat.shape[1] != channels:
                raise ValueError(
                    f"Encoder feature {tuple(point.feat.shape)} does not match "
                    f"configured channels {channels}"
                )
        level_features = tuple(point.feat for point in selected_points)
        return hierarchy, compact, level_features

    def fuse_token_hierarchy(self, compact, level_features):
        """Map deeper native features to P2 and concatenate all channels."""
        if len(level_features) != len(compact.target_to_levels):
            raise ValueError("native hierarchy feature/mapping counts differ")
        features = [level_features[0]]
        for feature, target_to_level in zip(
            level_features[1:], compact.target_to_levels[1:]
        ):
            if target_to_level is None:
                raise ValueError("deeper hierarchy level requires a P2 mapping")
            features.append(feature[target_to_level])
        fused = torch.cat(features, dim=-1)
        if fused.shape[1] != self.semantic_in_channels:
            raise RuntimeError(
                f"Fused hierarchy has {fused.shape[1]} channels, expected "
                f"{self.semantic_in_channels}"
            )
        return fused

    def project_visual_tokens(self, compact, level_features):
        fused = self.fuse_token_hierarchy(compact, level_features)
        visual_token = self.semantic_projector(fused)
        visual_point = visual_token[compact.input_to_target]
        return visual_token, visual_point

    def decode_return(self, hierarchy, decoder_detach_stages=()):
        decoded = self.return_backbone.decode_hierarchy(
            hierarchy, detach_stages=decoder_detach_stages
        )
        if decoded.feat.ndim != 2 or decoded.feat.shape[1] != self.return_channels:
            raise ValueError(
                f"Return decoder feature {tuple(decoded.feat.shape)} does not "
                f"match return_channels={self.return_channels}"
            )
        return decoded, self.return_norm(decoded.feat)

    def forward(
        self,
        input_dict,
        return_branches=False,
        decoder_detach_stages=(),
        mode=None,
    ):
        mode = self.assert_mode_available(self.mode if mode is None else mode)
        use_visual, use_return = _mode_branches(mode)
        if use_visual:
            hierarchy, compact, level_features = self.encode_hierarchy(
                input_dict
            )
        else:
            _, hierarchy = self.return_backbone.encode(input_dict)
            compact = level_features = None
        visual_token = visual_point = None
        return_feature = None
        if use_visual:
            visual_token, visual_point = self.project_visual_tokens(
                compact, level_features
            )
        if use_return:
            point, return_feature = self.decode_return(
                hierarchy, decoder_detach_stages=decoder_detach_stages
            )
        else:
            point = compact.points[0]

        if mode == "visual":
            point.feat = visual_point
        elif mode == "return":
            point.feat = return_feature
        else:
            point.feat = torch.cat((visual_point, return_feature), dim=-1)
        if "sparse_conv_feat" in point.keys():
            point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(
                point.feat
            )
        if return_branches:
            return point, {
                "compact": compact,
                "level_features": level_features,
                "visual_token": visual_token,
                "visual_feature": visual_point,
                "return_feature": return_feature,
            }
        return point

    def extract_point_feature(
        self,
        input_dict,
        feature_source="backbone",
        normalize=False,
    ):
        if feature_source not in self.FEATURE_SOURCES:
            raise ValueError(
                f"feature_source must be one of {sorted(self.FEATURE_SOURCES)}"
            )
        mode = self.mode if feature_source == "backbone" else feature_source
        point = self.forward(input_dict, mode=mode)
        feature = point.feat
        if normalize:
            feature = F.normalize(feature.float(), dim=-1, eps=1e-12)
        return feature


class ComplementaryVisualReturnLearningBase(nn.Module):
    """Aggregate native hierarchy features before semantic projection."""

    def __init__(
        self,
        backbone,
        criteria,
        backbone_out_channels=None,
        visual_channels=128,
        return_channels=96,
        predictor_checkpoint=True,
        edge_weight="sqrt_count",
        sample_balanced=True,
        validate_mapping=False,
        aggregate_channel_chunk_size=64,
        visual_loss_chunk_size=8192,
        knn_k=32,
        lambda_vpsd=1.0,
        lambda_return=1.0,
        lambda_ctx=1.0,
        head_dropout=0.1,
        ignore_index=-1,
        decoder_detach_stages=(2, 3, 4),
        mode="full",
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        if not isinstance(self.backbone, VisualReturnBackboneBase):
            raise TypeError("CVRL requires a compatible visual-return backbone")
        self.mode = _normalize_cvrl_mode(mode)
        self.use_visual, self.use_return = _mode_branches(self.mode)
        self.backbone.set_mode(self.mode, reset_availability=True)
        self.visual_channels = int(visual_channels)
        self.return_channels = int(return_channels)
        self.backbone_out_channels = self.backbone.out_channels
        if (
            backbone_out_channels is not None
            and int(backbone_out_channels) != self.backbone_out_channels
        ):
            raise ValueError(
                f"backbone_out_channels must be {self.backbone_out_channels} "
                f"for mode='{self.mode}'"
            )
        if self.visual_channels != self.backbone.visual_channels:
            raise ValueError("visual_channels disagrees with the CVRL backbone")
        if self.return_channels != self.backbone.return_channels:
            raise ValueError("return_channels disagrees with the CVRL backbone")
        self.predictor_checkpoint = bool(predictor_checkpoint)
        self.criteria = build_criteria(criteria)
        if not self.use_return:
            criteria_modules = getattr(
                self.criteria, "criteria", [self.criteria]
            )
            for criterion in criteria_modules:
                if hasattr(criterion, "auto_class_weight"):
                    criterion.auto_class_weight = False
        self.edge_weight = str(edge_weight)
        self.sample_balanced = bool(sample_balanced)
        self.validate_mapping = bool(validate_mapping)
        self.aggregate_channel_chunk_size = int(aggregate_channel_chunk_size)
        self.visual_loss_chunk_size = int(visual_loss_chunk_size)
        self.knn_k = int(knn_k)
        self.lambda_vpsd = float(lambda_vpsd)
        self.lambda_return = float(lambda_return)
        self.lambda_ctx = float(lambda_ctx)
        self.ignore_index = int(ignore_index)
        self.decoder_detach_stages = tuple(
            sorted({int(stage) for stage in decoder_detach_stages})
        )
        if self.edge_weight not in {"uniform", "count", "sqrt_count"}:
            raise ValueError("Unsupported VPSD edge_weight")
        if self.aggregate_channel_chunk_size <= 0:
            raise ValueError("aggregate_channel_chunk_size must be positive")
        if self.visual_loss_chunk_size <= 0:
            raise ValueError("visual_loss_chunk_size must be positive")
        invalid_detach = set(self.decoder_detach_stages).difference(
            range(len(self.backbone.level_channels))
        )
        if invalid_detach:
            raise ValueError(
                f"Invalid decoder_detach_stages: {sorted(invalid_detach)}"
            )

        self.role_head = self._build_return_head(
            self.return_channels, head_dropout
        )
        self.context_head = self._build_return_head(
            self.return_channels, head_dropout
        )
        self._freeze_inactive_branch_parameters()

    @staticmethod
    def _set_requires_grad(module, enabled):
        for parameter in module.parameters():
            parameter.requires_grad_(enabled)

    def _freeze_inactive_branch_parameters(self):
        """Avoid DDP unused-parameter failures in single-branch training."""
        if not self.use_visual:
            self._set_requires_grad(self.backbone.semantic_projector, False)
        if not self.use_return:
            self._set_requires_grad(self.backbone.return_backbone.dec, False)
            self._set_requires_grad(self.backbone.return_norm, False)
            self._set_requires_grad(self.role_head, False)
            self._set_requires_grad(self.context_head, False)

    @staticmethod
    def _build_return_head(channels, dropout):
        return nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels, NUM_RETURN_ROLES),
        )

    @staticmethod
    def _normalize_epsilon(dtype):
        return 1e-6 if dtype == torch.float16 else 1e-12

    def _visual_loss(self, input_dict, compact, level_features):
        raise NotImplementedError

    def _diagnose_task_gradients(self, visual_loss, return_loss, output):
        """Optional joint-task diagnostic implemented by CVRL variants."""
        return None

    def extract_point_feature(
        self,
        input_dict,
        feature_source="backbone",
        normalize=False,
    ):
        return self.backbone.extract_point_feature(
            input_dict,
            feature_source=feature_source,
            normalize=normalize,
        )

    def forward(
        self,
        input_dict,
        return_point_feature=False,
        feature_source="backbone",
        normalize_feature=False,
    ):
        if return_point_feature:
            return {
                "point_feature": self.extract_point_feature(
                    input_dict,
                    feature_source=feature_source,
                    normalize=normalize_feature,
                )
            }

        self.backbone.mark_mode_trained(self.mode)

        role_target = context_target = context_valid = None
        if self.use_return:
            required = {"return_role", "coord", "offset"}
            missing = required.difference(input_dict)
            if missing:
                raise KeyError(
                    f"CVRL return input is missing fields: {sorted(missing)}"
                )
            role_target = input_dict["return_role"].reshape(-1).long()
            context_target, context_valid = build_knn_return_context(
                coord=input_dict["coord"],
                offset=input_dict["offset"],
                return_role=role_target,
                k=self.knn_k,
                ignore_index=self.ignore_index,
            )

        if self.use_visual:
            hierarchy, compact, level_features = self.backbone.encode_hierarchy(
                input_dict
            )
            loss = level_features[0].sum() * 0.0
        else:
            deepest, hierarchy = self.backbone.return_backbone.encode(input_dict)
            compact = level_features = None
            loss = deepest.feat.sum() * 0.0
        output = {}
        if self.use_visual:
            loss_vpsd, visual_stats = self._visual_loss(
                input_dict, compact, level_features
            )
            loss = loss + self.lambda_vpsd * loss_vpsd
            output.update(
                {"loss/vpsd": loss_vpsd.detach(), **visual_stats}
            )

        if self.use_return:
            detach_stages = (
                self.decoder_detach_stages if self.use_visual else ()
            )
            _, return_feature = self.backbone.decode_return(
                hierarchy,
                decoder_detach_stages=detach_stages,
            )
            role_logits = self.role_head(return_feature)
            context_logits = self.context_head(return_feature)

            role_valid = role_target != self.ignore_index
            if role_valid.any():
                loss_role = self.criteria(role_logits, role_target)
            else:
                loss_role = role_logits.sum() * 0.0
            self.criteria._last_individual_losses.clear()

            if context_valid.any():
                log_prob = F.log_softmax(
                    context_logits[context_valid].float(), dim=-1
                )
                loss_ctx = -(
                    context_target[context_valid] * log_prob
                ).sum(dim=-1).mean()
            else:
                loss_ctx = context_logits.sum() * 0.0

            loss_return = loss_role + self.lambda_ctx * loss_ctx
            if self.use_visual:
                self._diagnose_task_gradients(
                    loss_vpsd, loss_return, output
                )
            loss = loss + self.lambda_return * loss_return
            output.update(
                {
                    "loss/role": loss_role.detach(),
                    "loss/ctx": loss_ctx.detach(),
                }
            )

        output = {"loss": loss, **output}
        if not self.training and self.use_return:
            output.update(
                role_logits=role_logits,
                context_logits=context_logits,
                context_target=context_target,
                context_valid=context_valid,
            )
        return output
