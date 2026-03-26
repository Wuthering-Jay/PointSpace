"""
EZSP Partition Module for PointSpace

This module provides the training infrastructure for EZ-SP partition learning.
It combines TinySparseCNN feature extraction with contrastive learning to
train point embeddings that enable semantic-aware superpoint clustering.

Usage:
- Phase 1 (Partition Training): Train this module with contrastive loss
  to learn features good for graph clustering.
- Phase 2 (Semantic Training): Use the trained features with the full
  SPT model for semantic segmentation.

Reference: EZ-SP (https://arxiv.org/abs/2402.04991)
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, Optional, Any, List, Tuple

from pointspace.models.builder import MODELS, build_model
from pointspace.models.losses.builder import LOSSES
from pointspace.models.modules import PointModule
from pointspace.models.utils import offset2batch

from pointspace.models.ezsp.tiny_sparse_cnn import TinySparseCNN
from pointspace.models.ezsp.partition import GPUGreedyPartition
from pointspace.models.ezsp.loss import EZSPContrastiveLoss


def build_partition_criteria(cfg_list):
    """Build partition criteria directly (not using standard Criteria class).

    EZ-SP contrastive loss has a different interface than standard losses.
    """
    if cfg_list is None:
        return None

    criteria = nn.ModuleList()
    for cfg in cfg_list:
        criteria.append(LOSSES.build(cfg=cfg))
    return criteria


@MODELS.register_module("EZSPPartitionSegmentor")
class EZSPPartitionSegmentor(nn.Module):
    """Segmentor for EZ-SP partition learning stage.

    This module trains point features via contrastive learning such that
    same-class points have similar features and different-class points
    have dissimilar features. The learned features can then be used for
    GPU-accelerated graph-based superpoint clustering.

    Training Flow:
        Point Cloud -> TinySparseCNN -> Features -> ContrastiveLoss

    Validation Flow:
        Point Cloud -> TinySparseCNN -> Features -> GPUGreedyPartition -> Superpoints

    Args:
        backbone: Config for TinySparseCNN (or any feature backbone)
        partition_criteria: Config for contrastive loss
        partition_cfg: Config for GPUGreedyPartition (used during validation)
        compute_partition_on_val: Whether to compute actual partition during validation

    Forward Input:
        point dict with keys:
        - feat: (N, C) input features
        - coord: (N, 3) coordinates
        - grid_coord: (N, 3) grid coordinates
        - offset: (B,) batch offsets
        - segment: (N,) ground truth labels (for training)

    Forward Output:
        dict with keys:
        - loss: total loss (during training)
        - l_partition: partition contrastive loss
        - seg_logits: placeholder for compatibility (zeros)
        - partition_result: partition info (during validation if enabled)
    """

    def __init__(
        self,
        backbone: Dict = None,
        partition_criteria: List[Dict] = None,
        partition_cfg: Dict = None,
        compute_partition_on_val: bool = True,
    ):
        super().__init__()

        # Build backbone (TinySparseCNN by default)
        if backbone is None:
            backbone = dict(type="TinySparseCNN", in_channels=9)
        self.backbone = build_model(backbone)

        # Build partition loss (using custom builder for EZSP interface)
        self.partition_criteria = build_partition_criteria(partition_criteria)

        # Build partition module for validation
        self.compute_partition_on_val = compute_partition_on_val
        self.partition = None
        if partition_cfg is not None and compute_partition_on_val:
            self.partition = GPUGreedyPartition(**partition_cfg)

    def forward(self, input_dict: Dict) -> Dict:
        """Forward pass.

        Args:
            input_dict: Point dict with feat, coord, grid_coord, offset, segment

        Returns:
            Output dict with loss and predictions
        """
        # Extract point features
        point = self.backbone(input_dict)

        # Get features and metadata
        feat = point.feat
        coord = point.coord
        offset = point.offset
        segment = input_dict.get("segment", None)

        output = {
            "feat": feat,
            "seg_logits": torch.zeros(feat.shape[0], 1, device=feat.device),  # Placeholder
        }

        # Training: compute contrastive loss
        if self.training and self.partition_criteria is not None and len(self.partition_criteria) > 0:
            if segment is not None:
                total_loss = 0.0
                for criterion in self.partition_criteria:
                    loss = criterion(
                        feat=feat,
                        pos=coord,
                        segment=segment,
                        offset=offset,
                    )
                    total_loss = total_loss + loss
                output["loss"] = total_loss
                output["l_partition"] = total_loss.detach()
            else:
                # No labels provided
                output["loss"] = feat.sum() * 0.0
                output["l_partition"] = output["loss"]

        # Validation: optionally compute partition
        if not self.training and self.partition is not None and self.compute_partition_on_val:
            batch = offset2batch(offset)
            partition_result = self.partition(
                feat=feat,
                pos=coord,
                batch=batch,
            )
            output["partition_result"] = partition_result
            output["super_index"] = partition_result["super_index"]
            output["num_superpoints"] = partition_result["num_superpoints"]

        return output


@MODELS.register_module("EZSPBackbone")
class EZSPBackbone(PointModule):
    """EZ-SP feature extraction backbone.

    This module wraps TinySparseCNN as a backbone that can be used with
    other segmentors. It extracts features suitable for graph-based
    partition and semantic segmentation.

    Args:
        in_channels: Input feature dimension
        channels: List of channel sizes for conv blocks
        kernel_sizes: List of kernel sizes
        **kwargs: Additional args for TinySparseCNN
    """

    def __init__(
        self,
        in_channels: int = 9,
        channels: List[int] = None,
        kernel_sizes: List[int] = None,
        **kwargs,
    ):
        super().__init__()

        self.cnn = TinySparseCNN(
            in_channels=in_channels,
            channels=channels,
            kernel_sizes=kernel_sizes,
            **kwargs,
        )

    @property
    def out_channels(self):
        return self.cnn.out_dim

    def forward(self, point):
        return self.cnn(point)


@MODELS.register_module("EZSPPartitionTrainer")
class EZSPPartitionTrainer(nn.Module):
    """Trainer module for EZ-SP partition learning.

    This is a simplified training module that focuses purely on learning
    good features for partitioning. It can be used standalone or as part
    of a larger pipeline.

    The forward pass:
    1. Extracts features using TinySparseCNN
    2. Computes contrastive loss on graph edges
    3. Returns loss for training

    Args:
        in_channels: Input feature dimension
        cnn_channels: CNN channel sizes
        cnn_kernel_sizes: CNN kernel sizes
        loss_temperature: Affinity temperature
        loss_gamma: Focal loss gamma
        adaptive_ratio: Adaptive sampling ratio
        num_classes: Number of semantic classes
        k_neighbors: KNN neighbors for graph
    """

    def __init__(
        self,
        in_channels: int = 9,
        cnn_channels: List[int] = None,
        cnn_kernel_sizes: List[int] = None,
        loss_temperature: float = 1.0,
        loss_gamma: float = 1.0,
        adaptive_ratio: float = 0.7,
        num_classes: int = None,
        k_neighbors: int = 8,
    ):
        super().__init__()

        # Feature extractor
        self.backbone = TinySparseCNN(
            in_channels=in_channels,
            channels=cnn_channels,
            kernel_sizes=cnn_kernel_sizes,
        )

        # Contrastive loss
        self.criterion = EZSPContrastiveLoss(
            affinity_temperature=loss_temperature,
            focal_gamma=loss_gamma,
            adaptive_sampling_ratio=adaptive_ratio,
            num_classes=num_classes,
            k=k_neighbors,
        )

    @property
    def out_channels(self):
        return self.backbone.out_dim

    def forward(self, input_dict: Dict) -> Dict:
        """Forward pass for training.

        Args:
            input_dict: Point dict with feat, coord, grid_coord, offset, segment

        Returns:
            Dict with loss and features
        """
        # Extract features
        point = self.backbone(input_dict)
        feat = point.feat

        output = {"feat": feat}

        # Compute loss during training
        if self.training:
            coord = point.coord
            offset = point.offset
            segment = input_dict.get("segment", None)

            if segment is not None:
                loss = self.criterion(
                    feat=feat,
                    pos=coord,
                    segment=segment,
                    offset=offset,
                )
            else:
                loss = feat.sum() * 0.0

            output["loss"] = loss
            output["l_partition"] = loss.detach()

        return output

    def extract_features(self, input_dict: Dict) -> Tensor:
        """Extract features without computing loss.

        Useful for inference or when using pre-trained features.
        """
        point = self.backbone(input_dict)
        return point.feat


def build_ezsp_partition_model(
    in_channels: int = 9,
    cnn_channels: List[int] = None,
    cnn_kernel_sizes: List[int] = None,
    partition_reg: float = 0.02,
    partition_min_size: int = 30,
    partition_k: int = 8,
    loss_temperature: float = 1.0,
    loss_gamma: float = 1.0,
    adaptive_ratio: float = 0.7,
    num_classes: int = None,
) -> EZSPPartitionSegmentor:
    """Factory function to create EZ-SP partition model with common config.

    Args:
        in_channels: Input feature dimension
        cnn_channels: CNN channel sizes (default: [32, 32, 32])
        cnn_kernel_sizes: CNN kernel sizes (default: [7, 3, 3])
        partition_reg: Regularization for partition
        partition_min_size: Minimum superpoint size
        partition_k: KNN neighbors
        loss_temperature: Affinity temperature
        loss_gamma: Focal loss gamma
        adaptive_ratio: Adaptive sampling ratio
        num_classes: Number of semantic classes

    Returns:
        Configured EZSPPartitionSegmentor
    """
    if cnn_channels is None:
        cnn_channels = [32, 32, 32]
    if cnn_kernel_sizes is None:
        cnn_kernel_sizes = [7, 3, 3]

    backbone_cfg = dict(
        type="TinySparseCNN",
        in_channels=in_channels,
        channels=cnn_channels,
        kernel_sizes=cnn_kernel_sizes,
    )

    partition_criteria_cfg = [
        dict(
            type="EZSPContrastiveLoss",
            affinity_temperature=loss_temperature,
            focal_gamma=loss_gamma,
            adaptive_sampling_ratio=adaptive_ratio,
            num_classes=num_classes,
            k=partition_k,
            loss_weight=1.0,
        )
    ]

    partition_cfg = dict(
        reg=partition_reg,
        min_size=partition_min_size,
        k=partition_k,
    )

    return EZSPPartitionSegmentor(
        backbone=backbone_cfg,
        partition_criteria=partition_criteria_cfg,
        partition_cfg=partition_cfg,
    )
