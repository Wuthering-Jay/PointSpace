"""CVRL-v1m1: complementary visual-return learning with accumulated PCGrad."""

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from pointspace.models.backbone.vpsd.vpsd_v1m1 import (
    VisibilityAwarePatchSetDistiller,
)
from pointspace.models.backbone.vpsd.vpsd_v1m1 import (
    aggregate_hierarchy_to_patches,
    build_token_patch_edges_compact,
)
from pointspace.models.backbone.vpsd.projection import (
    FixedPCAReduction,
    NativeDINOProjector,
)
from pointspace.models.builder import MODELS

from .common import (
    ComplementaryVisualReturnLearningBase,
    VisualReturnBackboneBase,
)


@MODELS.register_module("CVRL-v1m1-backbone")
class NativeVisualReturnBackbone(VisualReturnBackboneBase):
    """Persistent ``[PCA(native visual) | return]`` CVRL backbone."""

    def __init__(
        self,
        *args,
        native_channels=1024,
        native_hidden_channels=1024,
        pca_path=None,
        require_pca=True,
        pca_output_normalize=True,
        pca_output_scale=None,
        visual_token_chunk_size=8192,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.native_channels = int(native_channels)
        self.visual_token_chunk_size = int(visual_token_chunk_size)
        if self.native_channels <= 0 or self.visual_token_chunk_size <= 0:
            raise ValueError("native_channels and chunk size must be positive")

        # The native projector remains part of the downstream backbone; PCA is
        # only the deterministic compact visual readout.
        self.semantic_projector = NativeDINOProjector(
            in_channels=self.semantic_in_channels,
            hidden_channels=native_hidden_channels,
            out_channels=self.native_channels,
        )
        self.visual_projection = FixedPCAReduction(
            input_channels=self.native_channels,
            output_channels=self.visual_channels,
            pca_path=pca_path,
            require_pca=(require_pca and self.mode in {"visual", "full"}),
            output_normalize=pca_output_normalize,
            output_scale=pca_output_scale,
        )

    def project_visual_tokens(self, compact, level_features):
        fused = self.fuse_token_hierarchy(compact, level_features)
        visual_chunks = []
        for start in range(0, fused.shape[0], self.visual_token_chunk_size):
            end = min(start + self.visual_token_chunk_size, fused.shape[0])
            native = self.semantic_projector(fused[start:end])
            visual_chunks.append(self.visual_projection(native))
        visual_token = torch.cat(visual_chunks, dim=0)
        visual_point = visual_token[compact.input_to_target]
        return visual_token, visual_point

    def project_native_visual_tokens(
        self, compact, level_features, feature_point_mask=None
    ):
        """Return the pre-PCA 1024D visual representation for audits only."""
        fused = self.fuse_token_hierarchy(compact, level_features)
        native_chunks = []
        for start in range(0, fused.shape[0], self.visual_token_chunk_size):
            end = min(start + self.visual_token_chunk_size, fused.shape[0])
            native_chunks.append(self.semantic_projector(fused[start:end]))
        native_token = torch.cat(native_chunks, dim=0)
        input_to_target = compact.input_to_target
        if feature_point_mask is not None:
            input_to_target = input_to_target[feature_point_mask]
        return native_token, native_token[input_to_target]

    def extract_point_feature(
        self, input_dict, feature_source="backbone", normalize=False
    ):
        if feature_source != "visual_native":
            return super().extract_point_feature(
                input_dict,
                feature_source=feature_source,
                normalize=normalize,
            )
        self.assert_mode_available("visual")
        _, compact, level_features = self.encode_hierarchy(input_dict)
        feature_point_mask = input_dict.get("feature_point_mask")
        if feature_point_mask is not None:
            if (
                feature_point_mask.ndim != 1
                or feature_point_mask.shape[0] != compact.input_to_target.shape[0]
                or feature_point_mask.dtype != torch.bool
            ):
                raise ValueError("feature_point_mask must be a point-aligned bool tensor")
        _, feature = self.project_native_visual_tokens(
            compact, level_features, feature_point_mask=feature_point_mask
        )
        if normalize:
            feature = F.normalize(feature.float(), dim=-1, eps=1e-12)
        return feature


@MODELS.register_module("CVRL-v1m1")
class ComplementaryVisualReturnLearning(
    ComplementaryVisualReturnLearningBase
):
    """Official CVRL with native VPSD, RRCL and accumulated symmetric PCGrad.

    The trainer accumulates task gradients over the complete optimizer window
    before applying symmetric PCGrad to the shared stem/P0/P1 parameters.
    Task-private parameters retain their ordinary gradients.
    """

    required_backbone_type = "CVRL-v1m1-backbone"
    _PCGRAD_STAGES = ("stem", "P0", "P1")
    pcgrad_eps = 1e-12

    def __init__(
        self,
        *args,
        native_teacher_channels=1024,
        projector_checkpoint=True,
        return_warmup_ratio=0.1,
        pcgrad_start_ratio=0.1,
        pcgrad_log_interval=100,
        **kwargs,
    ):
        backbone_cfg = kwargs.get("backbone")
        if backbone_cfg is None and args:
            backbone_cfg = args[0]
        if (
            not isinstance(backbone_cfg, dict)
            or backbone_cfg.get("type") != self.required_backbone_type
        ):
            raise TypeError(
                f"{type(self).__name__} requires "
                f"{self.required_backbone_type}"
            )

        kwargs["predictor_checkpoint"] = bool(projector_checkpoint)
        super().__init__(*args, **kwargs)
        if not isinstance(self.backbone, NativeVisualReturnBackbone):
            raise TypeError("CVRL-v1m1 requires NativeVisualReturnBackbone")
        self.teacher_channels = int(native_teacher_channels)
        self.lambda_return_max = float(self.lambda_return)
        self.return_warmup_ratio = float(return_warmup_ratio)
        self.pcgrad_start_ratio = float(pcgrad_start_ratio)
        self.pcgrad_log_interval = int(pcgrad_log_interval)
        self.training_progress = 0.0
        self.training_step = 0
        if self.teacher_channels != self.backbone.native_channels:
            raise ValueError(
                "native_teacher_channels disagrees with backbone.native_channels"
            )
        if not 0 <= self.return_warmup_ratio <= 1:
            raise ValueError("return_warmup_ratio must be in [0, 1]")
        if not 0 <= self.pcgrad_start_ratio <= 1:
            raise ValueError("pcgrad_start_ratio must be in [0, 1]")
        if self.pcgrad_log_interval <= 0:
            raise ValueError("pcgrad_log_interval must be positive")

    def set_training_progress(self, progress, step):
        """Receive resume-safe progress from ``CVRLTrainingSchedule``."""
        self.training_progress = min(max(float(progress), 0.0), 1.0)
        self.training_step = int(step)

    def _return_weight(self):
        if self.return_warmup_ratio <= 0:
            return self.lambda_return_max
        scale = min(self.training_progress / self.return_warmup_ratio, 1.0)
        return self.lambda_return_max * scale

    def _shared_parameter_groups(self):
        groups = {"stem": [], "P0": [], "P1": []}
        for name, parameter in self.backbone.return_backbone.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith("embedding."):
                groups["stem"].append(parameter)
            elif name.startswith("enc.enc0."):
                groups["P0"].append(parameter)
            elif name.startswith("enc.enc1."):
                groups["P1"].append(parameter)
        return groups

    def pcgrad_parameter_groups(self):
        """Return the fixed shared stages used by official CVRL."""
        available = self._shared_parameter_groups()
        groups = {stage: available[stage] for stage in self._PCGRAD_STAGES}
        empty = [stage for stage, parameters in groups.items() if not parameters]
        if empty:
            raise RuntimeError(
                f"PCGrad found no trainable parameters for stages: {empty}"
            )
        return groups

    def pcgrad_is_active(self):
        return (
            self.training
            and self.mode == "full"
            and self.training_progress >= self.pcgrad_start_ratio
        )

    def pcgrad_should_log(self):
        return (
            self.pcgrad_is_active()
            and (self.training_step + 1) % self.pcgrad_log_interval == 0
        )

    def _diagnose_task_gradients(self, visual_loss, return_loss, output):
        """Expose graph-bearing task losses to the accumulation-aware trainer."""
        if not self.pcgrad_is_active() or not torch.is_grad_enabled():
            return
        output["_pcgrad_visual_loss"] = self.lambda_vpsd * visual_loss
        output["_pcgrad_return_loss"] = self.lambda_return * return_loss
        output["_pcgrad_log"] = self.pcgrad_should_log()

    def _native_chunk_loss(self, patch_feature, raw_target):
        def compute(input_feature, target_feature):
            prediction = self.backbone.semantic_projector(input_feature)
            dtype = prediction.dtype
            eps = self._normalize_epsilon(dtype)
            student = F.normalize(prediction.to(dtype), dim=-1, eps=eps)
            teacher = F.normalize(target_feature.to(dtype), dim=-1, eps=eps)
            return 1.0 - torch.sum(student * teacher, dim=-1)

        if self.predictor_checkpoint and self.training and torch.is_grad_enabled():
            return checkpoint(
                compute,
                patch_feature,
                raw_target,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        return compute(patch_feature, raw_target)

    def _visual_loss(self, input_dict, compact, level_features):
        required = {
            "dino_feature",
            "image_patch_index",
            "image_valid",
            "dino_offset",
        }
        missing = required.difference(input_dict)
        if missing:
            raise KeyError(f"CVRL visual input is missing fields: {sorted(missing)}")
        teacher = input_dict["dino_feature"]
        if teacher.ndim != 2 or teacher.shape[1] != self.teacher_channels:
            raise ValueError(
                f"Expected native DINO [P,{self.teacher_channels}], got "
                f"{tuple(teacher.shape)}"
            )

        edges = build_token_patch_edges_compact(
            input_to_level=compact.input_to_target,
            patch_index=input_dict["image_patch_index"],
            valid=input_dict["image_valid"],
            num_tokens=level_features[0].shape[0],
            num_patches=teacher.shape[0],
            validate_mapping=self.validate_mapping,
        )
        patch_feature, used_patch, _ = aggregate_hierarchy_to_patches(
            level_features,
            compact.target_to_levels,
            edges,
            edge_weight=self.edge_weight,
            channel_chunk_size=self.aggregate_channel_chunk_size,
        )
        if patch_feature.shape[0] == 0:
            prediction = self.backbone.semantic_projector(patch_feature)
            native_loss = prediction.float().sum() * 0.0
        else:
            native_parts = []
            all_used = used_patch.shape[0] == teacher.shape[0]
            for start in range(0, patch_feature.shape[0], self.visual_loss_chunk_size):
                end = min(start + self.visual_loss_chunk_size, patch_feature.shape[0])
                target = (
                    teacher[start:end]
                    if all_used
                    else teacher[used_patch[start:end]]
                )
                native_parts.append(
                    self._native_chunk_loss(patch_feature[start:end], target)
                )
            native_per_patch = torch.cat(native_parts, dim=0)
            if self.sample_balanced:
                native_loss = VisibilityAwarePatchSetDistiller._sample_balanced_mean(
                    native_per_patch, used_patch, input_dict["dino_offset"]
                )
            else:
                native_loss = native_per_patch.mean()

        visual_loss = native_loss
        stats = {
            "tok": level_features[0].new_tensor(level_features[0].shape[0]),
            "edge": level_features[0].new_tensor(edges.num_edges),
            "patch": level_features[0].new_tensor(used_patch.shape[0]),
        }
        return visual_loss, stats

    def forward(self, *args, **kwargs):
        self.lambda_return = self._return_weight()
        output = super().forward(*args, **kwargs)
        if not kwargs.get("return_point_feature", False):
            output["weight/return"] = output["loss"].new_tensor(
                self.lambda_return
            )
        return output
