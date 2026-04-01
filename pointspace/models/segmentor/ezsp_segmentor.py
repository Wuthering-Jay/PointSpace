"""
EZSPPartitionSegmentor - Two-Stage Segmentor for EZ-SP

This module implements the main segmentor for EZ-SP, supporting two training stages:
1. Partition learning: Train CNN to learn point features for good superpoint partitions
2. Semantic segmentation: Train transformer on superpoint graphs

Author: PointSpace Team
"""

from typing import Dict, Optional, Union
import warnings

import torch
import torch.nn as nn

from pointspace.models.builder import MODELS, build_model
from pointspace.models.losses import build_criteria
from pointspace.models.utils.structure import Point
from pointspace.models.backbone.ezsp.superpoint_hierarchy import SuperpointHierarchy


@MODELS.register_module()
class EZSPPartitionSegmentor(nn.Module):
    """
    EZ-SP Two-Stage Partition Segmentor

    Stage 1 (training_partition_stage=True):
        Input → SparseCNN → Point Embeddings → GreedyPartition → PartitionCriterion
        Goal: Learn good point features for partition boundary alignment

    Stage 2 (training_partition_stage=False):
        Input → Pretrained SparseCNN → Point Embeddings → GreedyPartition → Transformer → Semantic Labels
        Goal: Semantic segmentation on superpoint graphs

    Args:
        training_partition_stage: bool - Current training stage
        num_classes: int - Number of semantic classes
        sparse_cnn: dict - SparseCNN configuration
        partition_module: dict - GreedyContourPriorPartition configuration
        partition_criterion: dict - PartitionCriterion configuration (Stage 1)
        transformer: dict - Transformer configuration (Stage 2)
        criteria: dict | list - Semantic segmentation loss configuration (Stage 2)
        freeze_cnn: bool - Whether to freeze CNN in Stage 2
        backbone_out_channels: int - Output channels of SparseCNN (for seg_head)

    Input:
        input_dict: Dict containing:
            - coord: [N, 3] point coordinates
            - feat: [N, C_in] input features
            - grid_coord: [N, 3] voxelized coordinates
            - offset: [B] cumulative point counts
            - segment: [N] optional GT labels

    Output:
        Dict containing:
            - loss: Scalar loss (training)
            - seg_logits: [N, num_classes] (Stage 2)
            - nag: SuperpointHierarchy (evaluation)
    """

    def __init__(
        self,
        training_partition_stage: bool = True,
        num_classes: int = 13,
        sparse_cnn: Optional[dict] = None,
        partition_module: Optional[dict] = None,
        partition_criterion: Optional[dict] = None,
        transformer: Optional[dict] = None,
        criteria: Optional[Union[dict, list]] = None,
        freeze_cnn: bool = True,
        backbone_out_channels: int = 32,
        allow_transformer_fallback: bool = True,
    ):
        super().__init__()

        self.training_partition_stage = training_partition_stage
        self.num_classes = num_classes
        self.freeze_cnn = freeze_cnn
        self.backbone_out_channels = backbone_out_channels
        self.allow_transformer_fallback = allow_transformer_fallback

        # SparseCNN (required for both stages)
        if sparse_cnn is not None:
            self.sparse_cnn = build_model(sparse_cnn)
        else:
            # Default SparseCNN
            self.sparse_cnn = build_model(
                dict(
                    type="EZ-SparseCNN",
                    in_channels=6,
                    channels=[32, 32, 32],
                    norm="gn",
                )
            )

        # Partition module (required for both stages)
        if partition_module is not None:
            self.partition_module = build_model(partition_module)
        else:
            # Default partition module
            self.partition_module = build_model(
                dict(
                    type="GreedyContourPriorPartition",
                    reg=2e-2,
                    min_size=[5, 30, 90],
                    k_adjacency=10,
                )
            )

        if training_partition_stage:
            # Stage 1: Partition criterion
            if partition_criterion is not None:
                # Use LOSSES.build for single criterion
                from pointspace.models.losses.builder import LOSSES
                self.partition_criterion = LOSSES.build(partition_criterion)
            else:
                from pointspace.models.losses.partition_criterion import PartitionCriterion
                self.partition_criterion = PartitionCriterion(
                    num_classes=num_classes,
                )
            # No transformer/seg_head in Stage 1
            self.transformer = None
            self.seg_head = None
            self.criteria = None
        else:
            # Stage 2: Transformer + Semantic segmentation
            self.partition_criterion = None

            if transformer is not None:
                self.transformer = build_model(transformer)
            else:
                # Default: use simple SPT transformer
                from pointspace.models.backbone.ezsp.ezsp_transformer import (
                    EZSPTransformerSimple,
                )
                warnings.warn(
                    "No transformer config provided. Using default EZSPTransformerSimple. "
                    "For better performance, configure a custom transformer.",
                    stacklevel=2,
                )
                self.transformer = EZSPTransformerSimple(
                    num_classes=num_classes,
                    in_channels=backbone_out_channels,
                    hidden_dim=64,
                    num_heads=4,
                    num_blocks=2,
                )

            # Transformer has built-in seg_head
            self.seg_head = None

            if criteria is not None:
                self.criteria = build_criteria(criteria)
            else:
                # CRITICAL: SPT/EZ-SP uses ignore_index = num_classes (NOT -1!)
                # This follows the histogram convention where void/ignored labels
                # are placed in the (num_classes)-th column
                self.criteria = build_criteria(
                    [
                        dict(
                            type="CrossEntropyLoss", 
                            loss_weight=1.0, 
                            ignore_index=num_classes  # Use num_classes, not -1
                        ),
                    ]
                )

            # Freeze CNN in Stage 2
            if freeze_cnn:
                for param in self.sparse_cnn.parameters():
                    param.requires_grad = False

    def forward(self, input_dict: Dict) -> Dict:
        """
        Forward pass

        Data flow:
            1. input_dict contains raw coord, feat, grid_coord, etc.
            2. SparseCNN extracts point embeddings
            3. GreedyPartition performs dynamic partition (GPU KNN inside)
            4. Compute loss based on stage
        """
        # Build Point object
        point = Point(input_dict)

        # ========== Step 1: SparseCNN Feature Extraction ==========
        point = self.sparse_cnn(point)
        # point.feat: [N, 32] CNN embeddings

        # ========== Step 2: Dynamic Partition ==========
        # GPU KNN + Greedy partition (adjacency graph built inside partition_module)
        nag = self.partition_module(
            pos=point.coord,
            x=point.feat,
            offset=point.offset,
            y=input_dict.get("segment"),
        )

        # ========== Step 3: Stage-specific Processing ==========
        if self.training_partition_stage:
            return self._forward_partition_stage(nag, input_dict)
        else:
            return self._forward_semantic_stage(nag, point, input_dict)

    def _forward_partition_stage(
        self, nag: SuperpointHierarchy, input_dict: Dict
    ) -> Dict:
        """Stage 1: Partition learning"""
        if self.training:
            # Training: compute partition loss
            loss, partition_output = self.partition_criterion(nag)
            return {
                "loss": loss,
                "n_inter_edge": partition_output.get("n_inter_edge", 0),
                "n_intra_edge": partition_output.get("n_intra_edge", 0),
                "mean_affinity_intra": partition_output.get("mean_affinity_intra", 0),
                "mean_affinity_inter": partition_output.get("mean_affinity_inter", 0),
            }
        else:
            # Validation: compute oracle mIoU
            return self._compute_partition_metrics(nag, input_dict)

    def _forward_semantic_stage(
        self, nag: SuperpointHierarchy, point: Point, input_dict: Dict
    ) -> Dict:
        """Stage 2: Semantic segmentation"""
        # Transformer on NAG (returns point-level logits)
        seg_logits = self.transformer(nag)

        # Ensure correct shape
        assert seg_logits.shape[0] == point.coord.shape[0], (
            f"Logits shape mismatch: got {seg_logits.shape[0]} points, "
            f"expected {point.coord.shape[0]}"
        )

        result = {"seg_logits": seg_logits}

        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            result["loss"] = loss
        elif "segment" in input_dict:
            loss = self.criteria(seg_logits, input_dict["segment"])
            result["loss"] = loss

        return result

    def _compute_partition_metrics(
        self, nag: SuperpointHierarchy, input_dict: Dict
    ) -> Dict:
        """Compute partition quality metrics (Oracle mIoU)"""
        result = {"nag": nag}

        # Check if we have labels
        level1 = nag[1] if nag.num_levels > 1 else nag[0]
        y = level1.get("y")

        if y is None or "segment" not in input_dict:
            return result

        # Oracle: each superpoint takes majority label
        y_hist = y[:, : self.num_classes] if y.shape[1] > self.num_classes else y
        y_oracle = y_hist.argmax(dim=1)

        # Propagate back to raw points
        if nag.num_levels > 1:
            super_index = nag[0]["super_index"]
            if super_index is not None:
                y_pred = y_oracle[super_index]
            else:
                y_pred = y_oracle
        else:
            y_pred = y_oracle

        y_true = input_dict["segment"]

        result["y_pred"] = y_pred
        result["y_true"] = y_true

        # Compute accuracy
        valid_mask = y_true >= 0
        if valid_mask.any():
            acc = (y_pred[valid_mask] == y_true[valid_mask]).float().mean()
            result["oracle_acc"] = acc

        return result

    def load_stage1_weights(self, checkpoint_path: str, strict: bool = False):
        """
        Load Stage 1 (partition learning) weights for Stage 2

        Args:
            checkpoint_path: Path to Stage 1 checkpoint
            strict: Whether to strictly match keys
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Handle different checkpoint formats
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        # Filter to only sparse_cnn weights
        cnn_state_dict = {}
        for k, v in state_dict.items():
            # Handle module. prefix from DDP
            if k.startswith("module."):
                k = k[7:]
            if k.startswith("sparse_cnn."):
                cnn_state_dict[k[11:]] = v  # Remove 'sparse_cnn.' prefix

        if cnn_state_dict:
            self.sparse_cnn.load_state_dict(cnn_state_dict, strict=strict)
            print(f"Loaded {len(cnn_state_dict)} CNN weights from {checkpoint_path}")
        else:
            print(f"Warning: No sparse_cnn weights found in {checkpoint_path}")


@MODELS.register_module()
class EZSPPartitionSegmentorV2(EZSPPartitionSegmentor):
    """
    EZSPPartitionSegmentor V2 with additional features

    Adds:
    - Support for multi-scale features
    - Auxiliary losses
    - Feature visualization hooks
    """

    def __init__(
        self,
        training_partition_stage: bool = True,
        num_classes: int = 13,
        sparse_cnn: Optional[dict] = None,
        partition_module: Optional[dict] = None,
        partition_criterion: Optional[dict] = None,
        transformer: Optional[dict] = None,
        criteria: Optional[Union[dict, list]] = None,
        freeze_cnn: bool = True,
        backbone_out_channels: int = 32,
        aux_loss_weight: float = 0.0,
        allow_transformer_fallback: bool = True,
    ):
        super().__init__(
            training_partition_stage=training_partition_stage,
            num_classes=num_classes,
            sparse_cnn=sparse_cnn,
            partition_module=partition_module,
            partition_criterion=partition_criterion,
            transformer=transformer,
            criteria=criteria,
            freeze_cnn=freeze_cnn,
            backbone_out_channels=backbone_out_channels,
            allow_transformer_fallback=allow_transformer_fallback,
        )
        self.aux_loss_weight = aux_loss_weight

        # Auxiliary prediction head for deep supervision
        if aux_loss_weight > 0 and not training_partition_stage:
            self.aux_head = nn.Linear(backbone_out_channels, num_classes)
        else:
            self.aux_head = None

    def _forward_semantic_stage(
        self, nag: SuperpointHierarchy, point: Point, input_dict: Dict
    ) -> Dict:
        result = super()._forward_semantic_stage(nag, point, input_dict)

        # Add auxiliary loss
        if self.aux_head is not None and self.training:
            aux_logits = self.aux_head(point.feat)
            aux_loss = self.criteria(aux_logits, input_dict["segment"])
            result["loss"] = result["loss"] + self.aux_loss_weight * aux_loss
            result["aux_loss"] = aux_loss

        return result
