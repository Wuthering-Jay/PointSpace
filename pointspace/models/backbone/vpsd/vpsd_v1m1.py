"""Memory-efficient VPSD with loss-equivalent sparse hierarchy reduction."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torch_scatter
from torch.utils.checkpoint import checkpoint

from pointspace.models.builder import MODELS

from .hierarchy import build_encoder_hierarchy
from .projection import FixedPCAReduction
from .common import (
    VisibilityAwarePatchSetDistillerBase,
    TokenPatchEdges,
    aggregate_tokens_to_patches,
    build_token_patch_edges,
    fuse_hierarchy_features,
)


@dataclass(frozen=True)
class CompactHierarchy:
    """Only retain mappings required by one VPSD distillation level."""

    point: object
    points: tuple
    input_to_target: torch.Tensor
    target_to_levels: tuple


def build_compact_hierarchy(
    deepest_point,
    target_level,
    expected_levels=None,
    validate_mapping=False,
):
    """Build ``input -> target`` once and short ``target -> deeper`` maps."""
    coarse_to_fine = []
    point = deepest_point
    visited = set()
    while True:
        object_id = id(point)
        if object_id in visited:
            raise RuntimeError("Cycle detected in pooling_parent hierarchy")
        visited.add(object_id)
        coarse_to_fine.append(point)
        if "pooling_parent" not in point.keys():
            break
        point = point.pooling_parent

    points = tuple(reversed(coarse_to_fine))
    if expected_levels is not None and len(points) != int(expected_levels):
        raise RuntimeError(
            f"Expected {expected_levels} encoder levels, found {len(points)}"
        )
    if target_level < 0 or target_level >= len(points):
        raise ValueError(f"Invalid target_level {target_level}")

    num_input = int(points[0].feat.shape[0])
    input_to_target = torch.arange(
        num_input, device=points[0].feat.device, dtype=torch.long
    )
    input_batch = points[0].batch if "batch" in points[0].keys() else None
    for level in range(1, target_level + 1):
        child = points[level]
        if "pooling_inverse" not in child.keys():
            raise RuntimeError(f"Encoder level {level} has no pooling_inverse")
        inverse = child.pooling_inverse.long()
        if inverse.shape[0] != points[level - 1].feat.shape[0]:
            raise RuntimeError(
                f"Invalid pooling_inverse length at encoder level {level}"
            )
        input_to_target = inverse[input_to_target]

    if validate_mapping and input_to_target.numel() > 0:
        target = points[target_level]
        if (
            int(input_to_target.min()) < 0
            or int(input_to_target.max()) >= target.feat.shape[0]
        ):
            raise RuntimeError("Out-of-range input-to-target mapping")
        if input_batch is not None and not torch.equal(
            target.batch[input_to_target], input_batch
        ):
            raise RuntimeError("Input-to-target mapping crosses batch samples")

    # None is the identity map for the target level and avoids another [T]
    # tensor. Every following tensor maps target tokens directly to that level.
    target_to_levels = [None]
    target_to_level = None
    for level in range(target_level + 1, len(points)):
        inverse = points[level].pooling_inverse.long()
        target_to_level = (
            inverse if target_to_level is None else inverse[target_to_level]
        )
        target_to_levels.append(target_to_level)

    return CompactHierarchy(
        point=deepest_point,
        points=points,
        input_to_target=input_to_target,
        target_to_levels=tuple(target_to_levels),
    )


def build_token_patch_edges_compact(
    input_to_level,
    patch_index,
    valid,
    num_tokens,
    num_patches,
    validate_mapping=False,
    observability=None,
):
    """Build the same v1m1 edges with fewer simultaneous ``[N]`` buffers."""
    if input_to_level.ndim != 1 or patch_index.ndim != 1 or valid.ndim != 1:
        raise ValueError("input_to_level, patch_index and valid must be 1D")
    if not input_to_level.shape[0] == patch_index.shape[0] == valid.shape[0]:
        raise ValueError("Point-to-token and point-to-patch lengths do not match")
    if observability is not None:
        if observability.ndim != 1 or observability.shape[0] != valid.shape[0]:
            raise ValueError("observability must be 1D and point-aligned")
        if validate_mapping:
            if not torch.isfinite(observability).all():
                raise ValueError("observability contains non-finite values")
            if torch.any((observability < 0) | (observability > 1)):
                raise ValueError("observability must be within [0, 1]")
    if num_tokens < 0 or num_patches < 0:
        raise ValueError("num_tokens and num_patches must be non-negative")

    relation_mask = valid.bool() & (patch_index >= 0)
    edge_key = input_to_level[relation_mask].long()
    if edge_key.numel() == 0:
        empty = input_to_level.new_empty(0, dtype=torch.long)
        empty_observability = (
            None
            if observability is None
            else observability.new_empty(0, dtype=torch.float32)
        )
        return TokenPatchEdges(
            token=empty,
            patch=empty,
            point_count=empty,
            observability_sum=empty_observability,
        )

    patch = patch_index[relation_mask].long()
    relation_observability = (
        None
        if observability is None
        else observability[relation_mask].float()
    )
    if validate_mapping:
        if int(edge_key.min()) < 0 or int(edge_key.max()) >= num_tokens:
            raise ValueError("Point-to-token mapping contains an invalid token index")
        if int(patch.min()) < 0 or int(patch.max()) >= num_patches:
            raise ValueError("Point-to-patch mapping contains an invalid patch index")
    del relation_mask
    edge_key.mul_(num_patches).add_(patch)
    del patch

    if relation_observability is None:
        unique_key, point_count = torch.unique(
            edge_key, sorted=True, return_counts=True
        )
        observability_sum = None
    else:
        unique_key, edge_inverse, point_count = torch.unique(
            edge_key,
            sorted=True,
            return_inverse=True,
            return_counts=True,
        )
        observability_sum = torch_scatter.scatter_sum(
            relation_observability,
            edge_inverse,
            dim=0,
            dim_size=unique_key.shape[0],
        )
    return TokenPatchEdges(
        token=torch.div(unique_key, num_patches, rounding_mode="floor"),
        patch=torch.remainder(unique_key, num_patches),
        point_count=point_count,
        observability_sum=observability_sum,
    )


def aggregate_hierarchy_to_patches(
    level_features,
    target_to_levels,
    edges,
    edge_weight="sqrt_count",
    channel_chunk_size=64,
    observability_weight=False,
    observability_power=1.0,
):
    """Aggregate channels directly from each hierarchy level to used patches."""
    if len(level_features) != len(target_to_levels):
        raise ValueError("level_features and target_to_levels lengths differ")
    if channel_chunk_size <= 0:
        raise ValueError("channel_chunk_size must be positive")
    if edges.num_edges == 0:
        empty_feature = torch.cat(
            [feature[:0] for feature in level_features], dim=1
        )
        return empty_feature, edges.patch, empty_feature.new_empty(0)

    reference = level_features[0]
    if edge_weight == "uniform":
        weight = torch.ones_like(edges.point_count, dtype=reference.dtype)
    elif edge_weight == "count":
        weight = edges.point_count.to(reference.dtype)
    elif edge_weight == "sqrt_count":
        weight = torch.sqrt(edges.point_count.to(reference.dtype))
    else:
        raise ValueError("edge_weight must be one of 'uniform', 'count', or 'sqrt_count'")

    if observability_weight:
        if edges.observability_sum is None:
            raise ValueError(
                "observability_weight=True requires edge observability_sum"
            )
        if observability_power <= 0:
            raise ValueError("observability_power must be positive")
        edge_observability = edges.observability_sum.to(reference.dtype) / (
            edges.point_count.to(reference.dtype).clamp_min(1)
        )
        weight = weight * edge_observability.clamp(0, 1).pow(
            float(observability_power)
        )

    used_patch, edge_to_patch = torch.unique(
        edges.patch, sorted=True, return_inverse=True
    )
    patch_count = int(used_patch.shape[0])
    patch_weight = torch_scatter.scatter_sum(
        weight, edge_to_patch, dim=0, dim_size=patch_count
    )
    denominator = patch_weight.clamp_min(1e-12).unsqueeze(1)
    reduced_levels = []
    for feature, target_to_level in zip(level_features, target_to_levels):
        edge_token = (
            edges.token
            if target_to_level is None
            else target_to_level[edges.token]
        )
        reduced_channels = []
        for start in range(0, feature.shape[1], channel_chunk_size):
            end = min(start + channel_chunk_size, feature.shape[1])
            # Advanced indexing creates a private tensor, so in-place weighting
            # is safe and avoids a second [E, channel_chunk_size] allocation.
            weighted = feature[edge_token, start:end]
            weighted.mul_(weight.unsqueeze(1))
            reduced = torch_scatter.scatter_sum(
                weighted,
                edge_to_patch,
                dim=0,
                dim_size=patch_count,
            )
            reduced_channels.append(reduced / denominator)
        reduced_levels.append(torch.cat(reduced_channels, dim=1))
    return torch.cat(reduced_levels, dim=1), used_patch, patch_weight


@MODELS.register_module("VPSD-v1m1")
class VisibilityAwarePatchSetDistiller(VisibilityAwarePatchSetDistillerBase):
    """Memory-efficient visibility-aware patch-set distillation."""

    def __init__(
        self,
        *args,
        aggregate_channel_chunk_size=64,
        projector_patch_chunk_size=8192,
        normalize_dtype="amp",
        visual_channels=128,
        pca_path=None,
        require_pca=True,
        pca_output_normalize=True,
        pca_output_scale=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.aggregate_channel_chunk_size = int(aggregate_channel_chunk_size)
        self.projector_patch_chunk_size = int(projector_patch_chunk_size)
        self.normalize_dtype = str(normalize_dtype).lower()
        self.visual_channels = int(visual_channels)
        if self.aggregate_channel_chunk_size <= 0:
            raise ValueError("aggregate_channel_chunk_size must be positive")
        if self.projector_patch_chunk_size <= 0:
            raise ValueError("projector_patch_chunk_size must be positive")
        if self.normalize_dtype not in {"amp", "float32", "bfloat16", "float16"}:
            raise ValueError(
                "normalize_dtype must be amp, float32, bfloat16, or float16"
            )
        self.visual_projection = FixedPCAReduction(
            input_channels=self.teacher_channels,
            output_channels=self.visual_channels,
            pca_path=pca_path,
            require_pca=require_pca,
            output_normalize=pca_output_normalize,
            output_scale=pca_output_scale,
        )

    def _encode_compact(self, input_dict):
        point = self.backbone(input_dict)
        compact = build_compact_hierarchy(
            point,
            target_level=self.distill_level,
            expected_levels=len(self.level_channels),
            validate_mapping=self.validate_mapping,
        )
        selected_points = compact.points[self.distill_level :]
        expected_channels = (
            self.level_channels[self.distill_level :]
            if self.fuse_deeper_features
            else self.level_channels[self.distill_level : self.distill_level + 1]
        )
        if not self.fuse_deeper_features:
            selected_points = selected_points[:1]
            compact = CompactHierarchy(
                point=compact.point,
                points=compact.points,
                input_to_target=compact.input_to_target,
                target_to_levels=(None,),
            )
        for point_level, channels in zip(selected_points, expected_channels):
            if point_level.feat.ndim != 2 or point_level.feat.shape[1] != channels:
                raise ValueError(
                    f"Encoder feature shape {tuple(point_level.feat.shape)} "
                    f"does not match configured channels {channels}"
                )
        return compact, tuple(point_level.feat for point_level in selected_points)

    def _resolve_normalize_dtype(self, prediction):
        if self.normalize_dtype == "amp":
            return prediction.dtype
        return {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[self.normalize_dtype]

    @staticmethod
    def _normalize_epsilon(dtype):
        # BF16 has the same exponent width as FP32, so the original epsilon is
        # representable. FP16 needs a larger non-zero guard for zero vectors.
        return 1e-6 if dtype == torch.float16 else 1e-12

    def _projector_cosine_loss_v2(self, feature, raw_target):
        def compute(input_feature, target_feature):
            prediction = self.student_projector(input_feature)
            dtype = self._resolve_normalize_dtype(prediction)
            eps = self._normalize_epsilon(dtype)
            student = F.normalize(prediction.to(dtype), dim=-1, eps=eps)
            teacher = F.normalize(target_feature.to(dtype), dim=-1, eps=eps)
            return 1.0 - torch.sum(student * teacher, dim=-1)

        if self.projector_checkpoint and self.training and torch.is_grad_enabled():
            return checkpoint(
                compute,
                feature,
                raw_target,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        return compute(feature, raw_target)

    def _chunked_projector_loss(self, patch_feature, dino_feature, used_patch):
        losses = []
        all_teacher_patches_used = used_patch.shape[0] == dino_feature.shape[0]
        for start in range(0, patch_feature.shape[0], self.projector_patch_chunk_size):
            end = min(
                start + self.projector_patch_chunk_size, patch_feature.shape[0]
            )
            # CompactImagePatches guarantees that every retained teacher patch
            # is referenced. Since used_patch is sorted, U == P then implies
            # used_patch == arange(P), without a GPU synchronization check.
            if all_teacher_patches_used:
                raw_target = dino_feature[start:end]
            else:
                raw_target = dino_feature[used_patch[start:end]]
            losses.append(
                self._projector_cosine_loss_v2(
                    patch_feature[start:end], raw_target
                )
            )
        return torch.cat(losses, dim=0)

    def _fused_target_feature(self, compact, level_features):
        features = [level_features[0]]
        for feature, target_to_level in zip(
            level_features[1:], compact.target_to_levels[1:]
        ):
            features.append(feature[target_to_level])
        return torch.cat(features, dim=1) if len(features) > 1 else features[0]

    def extract_point_feature(
        self,
        input_dict,
        feature_source="visual",
        normalize=True,
    ):
        compact, level_features = self._encode_compact(input_dict)
        target_feature = self._fused_target_feature(compact, level_features)
        if feature_source in {"visual", "projected"}:
            native_feature = self.student_projector(target_feature)
            token_feature = self.visual_projection(native_feature)
        elif feature_source == "visual_native":
            token_feature = self.student_projector(target_feature)
        elif feature_source == "backbone":
            token_feature = target_feature
        else:
            raise ValueError(
                "feature_source must be visual/projected, visual_native, or backbone"
            )
        input_to_target = compact.input_to_target
        feature_point_mask = input_dict.get("feature_point_mask")
        if feature_point_mask is not None:
            if (
                feature_point_mask.ndim != 1
                or feature_point_mask.shape[0] != input_to_target.shape[0]
                or feature_point_mask.dtype != torch.bool
            ):
                raise ValueError("feature_point_mask must be a point-aligned bool tensor")
            input_to_target = input_to_target[feature_point_mask]
        point_feature = token_feature[input_to_target]
        if normalize:
            dtype = self._resolve_normalize_dtype(point_feature)
            point_feature = F.normalize(
                point_feature.to(dtype),
                dim=-1,
                eps=self._normalize_epsilon(dtype),
            )
        return point_feature

    def forward_train(self, input_dict, return_point=False):
        required = {"dino_feature", "image_patch_index", "image_valid", "dino_offset"}
        missing = required.difference(input_dict)
        if missing:
            raise KeyError(f"VPSD input is missing fields: {sorted(missing)}")

        compact, level_features = self._encode_compact(input_dict)
        target_feature = level_features[0]
        dino_feature = input_dict["dino_feature"]
        if dino_feature.ndim != 2 or dino_feature.shape[1] != self.teacher_channels:
            raise ValueError(
                f"Expected DINO feature shape [P, {self.teacher_channels}], got "
                f"{tuple(dino_feature.shape)}"
            )

        edges = build_token_patch_edges_compact(
            input_to_level=compact.input_to_target,
            patch_index=input_dict["image_patch_index"],
            valid=input_dict["image_valid"],
            num_tokens=target_feature.shape[0],
            num_patches=dino_feature.shape[0],
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
            prediction = self.student_projector(patch_feature)
            loss = prediction.float().sum() * 0.0
        else:
            loss_per_patch = self._chunked_projector_loss(
                patch_feature, dino_feature, used_patch
            )
            if self.sample_balanced:
                loss = self._sample_balanced_mean(
                    loss_per_patch, used_patch, input_dict["dino_offset"]
                )
            else:
                loss = loss_per_patch.mean()

        result = {
            "loss": loss,
            "tok": target_feature.new_tensor(target_feature.shape[0]),
            "edge": target_feature.new_tensor(edges.num_edges),
            "patch": target_feature.new_tensor(used_patch.shape[0]),
        }
        if return_point:
            result["point"] = compact.point
            result["hierarchy"] = build_encoder_hierarchy(
                compact.point, expected_levels=len(self.level_channels)
            )
        return result


@MODELS.register_module("VPSD-v1m1-backbone")
class VisibilityAwarePatchSetBackbone(VisibilityAwarePatchSetDistiller):
    """Downstream backbone exposing VPSD's compact point representation.

    The pretraining module normally enters its DINO distillation path when
    called directly.  Downstream semantic segmentation has no image teacher,
    so this adapter makes the retained 128-D visual representation the normal
    ``forward`` output while preserving all pretrained parameter names below
    the adapter.  It is intentionally used for strict VPSD linear probing;
    decoder probing and full fine-tuning transfer the VPSD encoder into the
    native LitePT encoder-decoder instead.
    """

    def __init__(self, *args, feature_source="visual", normalize=False, **kwargs):
        super().__init__(*args, **kwargs)
        if feature_source not in {"visual", "projected", "visual_native", "backbone"}:
            raise ValueError("Unsupported VPSD downstream feature_source")
        self.feature_source = feature_source
        self.normalize = bool(normalize)
        self.out_channels = {
            "visual": self.visual_channels,
            "projected": self.visual_channels,
            "visual_native": self.teacher_channels,
            "backbone": self.projector_in_channels,
        }[feature_source]

    def _encode_compact(self, input_dict):
        """Use the encoder hierarchy while retaining a configured decoder.

        Downstream configs intentionally instantiate the same complete LitePT
        encoder-decoder for every method.  VPSD's visual feature is defined on
        P2/P3/P4, however, so strict linear probing must not route it through
        the randomly initialized downstream decoder.
        """
        deepest, _ = self.backbone.encode(input_dict)
        compact = build_compact_hierarchy(
            deepest,
            target_level=self.distill_level,
            expected_levels=len(self.level_channels),
            validate_mapping=self.validate_mapping,
        )
        selected_points = compact.points[self.distill_level :]
        expected_channels = (
            self.level_channels[self.distill_level :]
            if self.fuse_deeper_features
            else self.level_channels[self.distill_level : self.distill_level + 1]
        )
        if not self.fuse_deeper_features:
            selected_points = selected_points[:1]
            compact = CompactHierarchy(
                point=compact.point,
                points=compact.points,
                input_to_target=compact.input_to_target,
                target_to_levels=(None,),
            )
        for point_level, channels in zip(selected_points, expected_channels):
            if point_level.feat.ndim != 2 or point_level.feat.shape[1] != channels:
                raise ValueError(
                    f"Encoder feature shape {tuple(point_level.feat.shape)} "
                    f"does not match configured channels {channels}"
                )
        return compact, tuple(point_level.feat for point_level in selected_points)

    def forward(self, input_dict):
        return self.extract_point_feature(
            input_dict,
            feature_source=self.feature_source,
            normalize=self.normalize,
        )
