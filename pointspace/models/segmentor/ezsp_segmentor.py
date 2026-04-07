"""
EZSPPartitionSegmentor - Two-Stage Segmentor for EZ-SP

This module implements the main segmentor for EZ-SP, supporting two training stages:
1. Partition learning: Train CNN to learn point features for good superpoint partitions
2. Semantic segmentation: Train transformer on superpoint graphs

Architecture (Official High-Efficiency Strategy):
    L0 (raw points) ─[sub Cluster]─> L1 (voxels) ─[partition]─> L2+ (superpoints)
    
    - GPU core computation (SparseCNN, Partition, Transformer) runs on L1+ only
    - L0 raw points are preserved via 'sub' Cluster mapping for final evaluation
    - Final predictions propagate back to L0 via super_index chain

Author: PointSpace Team
"""

from typing import Dict, Optional, Union
import warnings

import torch
import torch.nn as nn

from pointspace.models.builder import MODELS, build_model
from pointspace.models.losses import build_criteria
from pointspace.models.utils.structure import Point
from pointspace.models.backbone.ezsp.superpoint_hierarchy import (
    SuperpointHierarchy, 
    Cluster,
    SuperpointLevel,
)
from pointspace.models.backbone.ezsp.spt.weight_init import apply_weight_init
from pointspace.utils.logger import get_root_logger


@MODELS.register_module()
class EZSPPartitionSegmentor(nn.Module):
    """
    EZ-SP Two-Stage Partition Segmentor (Official High-Efficiency Architecture)

    Stage 1 (training_partition_stage=True):
        Voxels(L1) → SparseCNN → Voxel Embeddings → GreedyPartition(L1) → PartitionCriterion
        Goal: Learn good voxel features for partition boundary alignment

    Stage 2 (training_partition_stage=False):
        Voxels(L1) → Pretrained SparseCNN → Voxel Embeddings → GreedyPartition → Transformer → Semantic Labels
        Goal: Semantic segmentation on superpoint graphs

    Key Design (matching official EZ-SP):
        - SparseCNN and Partition run on VOXELS (L1), NOT raw points
        - Raw points (L0) are preserved via 'sub' Cluster mapping for final evaluation
        - Predictions propagate back to L0 via sub.to_super_index() (voxel→point)
        - No VoxelToPointDecoder needed!

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

    Input (from GridSampling3D transform):
        input_dict: Dict containing:
            - coord: [M, 3] voxel center coordinates
            - feat: [M, C_in] voxel features
            - grid_coord: [M, 3] voxelized integer coordinates
            - offset: [B] cumulative voxel counts
            - sub: dict{'pointer', 'value'} → voxel→raw_point Cluster mapping
            - segment: [M, num_classes+1] label histogram (for training)
            - segment_raw: [N] raw point labels (for validation metrics)
            - num_raw_points: int - Original point count

    Output:
        Dict containing:
            - loss: Scalar loss (training)
            - seg_logits: [N, num_classes] (Stage 2, propagated to L0)
            - y_pred/y_true: Oracle predictions (validation, at L0)
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
        # Legacy parameters (ignored, kept for config compatibility)
        use_voxel_to_point: bool = False,
        voxel_to_point_decoder: Optional[dict] = None,
    ):
        super().__init__()

        self.training_partition_stage = training_partition_stage
        self.num_classes = num_classes
        self.freeze_cnn = freeze_cnn
        self.backbone_out_channels = backbone_out_channels
        self.allow_transformer_fallback = allow_transformer_fallback

        # Legacy warning
        if use_voxel_to_point or voxel_to_point_decoder is not None:
            warnings.warn(
                "use_voxel_to_point and voxel_to_point_decoder are deprecated. "
                "The official EZ-SP architecture uses voxel-level (L1) partition, "
                "not point-level. These parameters will be ignored.",
                DeprecationWarning,
                stacklevel=2,
            )

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

            # Apply weight initialization to Transformer
            # This follows official SPT implementation to improve FP16/AMP stability
            logger = get_root_logger()
            logger.info("Applying Xavier weight initialization to Transformer...")
            apply_weight_init(self.transformer, linear='xavier_uniform', rpe='xavier_uniform')
            
            # Initialize seg_head if it exists (only for Stage2 without transformer head)
            if hasattr(self, 'seg_head') and self.seg_head is not None:
                apply_weight_init(self.seg_head, linear='xavier_uniform')
            
            # Freeze CNN in Stage 2
            if freeze_cnn:
                logger.info("Freezing SparseCNN weights for Stage 2 training...")
                self.sparse_cnn.freeze()
                
                # Verify freeze worked
                frozen_params = sum(1 for p in self.sparse_cnn.parameters() if not p.requires_grad)
                total_params = sum(1 for _ in self.sparse_cnn.parameters())
                logger.info(f"SparseCNN frozen: {frozen_params}/{total_params} parameters")
                
                if frozen_params != total_params:
                    logger.warning(
                        f"Warning: Not all CNN parameters were frozen! "
                        f"({frozen_params}/{total_params})"
                    )

    def forward(self, input_dict: Dict) -> Dict:
        """
        Forward pass (Official High-Efficiency Architecture)

        Data flow:
            1. Input already contains VOXEL-level data (from GridSampling3D)
               - coord: [M, 3] voxel coordinates
               - feat: [M, C] voxel features
               - sub: Cluster mapping voxels → raw points
            2. SparseCNN extracts voxel embeddings
            3. GreedyPartition runs on VOXELS (not raw points!)
            4. Compute loss based on stage
            5. For evaluation: propagate predictions back to L0 via sub mapping
        """
        device = input_dict.get("coord").device if isinstance(input_dict.get("coord"), torch.Tensor) else "cuda"
        
        # Build Point object from voxelized data
        point = Point(input_dict)
        
        # ========== Extract 'sub' Cluster (voxel→raw_point mapping) ==========
        sub_cluster = None
        if "sub" in input_dict:
            sub_data = input_dict["sub"]
            if isinstance(sub_data, dict):
                sub_cluster = Cluster.from_dict(sub_data, device=device)
            elif isinstance(sub_data, Cluster):
                sub_cluster = sub_data
        
        num_raw_points = input_dict.get("num_raw_points", None)
        
        # ========== Step 1: SparseCNN Feature Extraction on Voxels ==========
        # This operates on [M, C_in] voxel features → [M, C_out] voxel embeddings
        point = self.sparse_cnn(point)
        # point.feat: [M, C] voxel embeddings (M = num_voxels)

        # ========== Step 2: Dynamic Partition on Voxels (NOT raw points!) ==========
        # GPU KNN + Greedy partition operates on M voxels, much faster than N points
        # NAG structure: [L1: voxels, L2+: superpoints]
        
        # Prepare label histogram for partition
        # The 'segment' key should contain [M, num_classes+1] histogram from GridSampling3D
        y_hist = input_dict.get("segment", None)
        if y_hist is not None and isinstance(y_hist, torch.Tensor):
            # Ensure it's histogram format (2D)
            if y_hist.dim() == 1:
                # Convert scalar labels to histogram
                y_hist = self._labels_to_histogram(y_hist, self.num_classes + 1)
        
        nag = self.partition_module(
            pos=point.coord,  # [M, 3] voxel positions
            x=point.feat,     # [M, C] voxel embeddings
            offset=point.offset,
            batch=input_dict.get("batch", getattr(point, "batch", None)),
            y=y_hist,         # [M, num_classes+1] voxel label histogram
        )
        
        # Store sub_cluster info for later use (but don't prepend L0 yet)
        # L0 prepending happens only when needed for evaluation/prediction propagation
        
        # ========== Step 3: Stage-specific Processing ==========
        if self.training_partition_stage:
            # Training partition stage: work on voxels (L1 = nag[0])
            return self._forward_partition_stage(nag, input_dict, sub_cluster)
        else:
            # Semantic stage or evaluation: may need L0 prepending
            if sub_cluster is not None and num_raw_points is not None:
                self._prepend_raw_point_level(nag, sub_cluster, num_raw_points)
            return self._forward_semantic_stage(nag, point, input_dict, sub_cluster)
    
    def _labels_to_histogram(self, labels: torch.Tensor, num_bins: int) -> torch.Tensor:
        """Convert scalar labels to one-hot histogram."""
        device = labels.device
        N = labels.shape[0]
        hist = torch.zeros(N, num_bins, device=device, dtype=torch.float32)
        valid_mask = (labels >= 0) & (labels < num_bins)
        if valid_mask.any():
            hist[torch.arange(N, device=device)[valid_mask], labels[valid_mask].long()] = 1.0
        return hist
    
    def _prepend_raw_point_level(
        self, 
        nag: SuperpointHierarchy, 
        sub_cluster: Cluster, 
        num_raw_points: int
    ):
        """
        Prepend L0 (raw points) level to NAG structure.
        
        After this:
            nag[0] = L0 (raw points, virtual level with super_index to L1)
            nag[1] = L1 (voxels, original nag[0])
            nag[2+] = L2+ (superpoints, original nag[1:])
        """
        device = nag[0]["pos"].device
        
        # Create super_index for L0→L1 (raw_point → voxel mapping)
        # This is the inverse of sub_cluster
        l0_super_index = sub_cluster.to_super_index()
        
        # Create L0 level (minimal data for raw points)
        l0_level = SuperpointLevel()
        l0_level["super_index"] = l0_super_index  # [N] raw_point → voxel
        l0_level["num_points"] = num_raw_points
        # Note: We don't store pos/feat for L0 to save memory
        # They can be retrieved from input_dict if needed
        
        # Store sub cluster in L1 level for backtracking
        nag[0]["sub"] = sub_cluster
        
        # Prepend L0 to NAG (using levels list, not _list)
        nag.levels.insert(0, l0_level)
        nag.start_i_level = 0  # Now L0 is the atom level
    
    def _forward_partition_stage(
        self, nag: SuperpointHierarchy, input_dict: Dict, sub_cluster: Optional[Cluster]
    ) -> Dict:
        """Stage 1: Partition learning on voxels (L1)
        
        In this stage:
        - nag[0] = L1 (voxels with features x, pos, edge_index)
        - nag[1:] = L2+ (superpoints)
        No L0 prepending during training - partition criterion works directly on voxels.
        """
        if self.training:
            # Training: compute partition loss on voxel level (nag[0])
            loss, partition_output = self.partition_criterion(nag)
            return {
                "loss": loss,
                "n_inter": torch.tensor(partition_output.get("n_inter_edge", 0), dtype=torch.float32),
                "n_intra": torch.tensor(partition_output.get("n_intra_edge", 0), dtype=torch.float32),
                "m_aff_intra": torch.tensor(partition_output.get("mean_affinity_intra", 0.0), dtype=torch.float32),
                "m_aff_inter": torch.tensor(partition_output.get("mean_affinity_inter", 0.0), dtype=torch.float32),
            }
        else:
            # Validation: prepend L0 and compute oracle mIoU at raw point level
            num_raw_points = input_dict.get("num_raw_points", None)
            if sub_cluster is not None and num_raw_points is not None:
                self._prepend_raw_point_level(nag, sub_cluster, num_raw_points)
            return self._compute_partition_metrics(nag, input_dict, sub_cluster)

    def _forward_semantic_stage(
        self, nag: SuperpointHierarchy, point: Point, input_dict: Dict, sub_cluster: Optional[Cluster]
    ) -> Dict:
        """Stage 2: Semantic segmentation on superpoint graphs
        
        Predictions are computed at superpoint level, then propagated:
            L2+ (superpoints) → L1 (voxels) → L0 (raw points)
        
        Critical: Loss is computed at SUPERPOINT level (not voxel/point level)!
        """
        # 1. Transformer outputs superpoint-level logits
        # NAG structure: [L0 (raw points), L1 (voxels), L2+ (superpoints)]
        # Transformer operates on L1+ and returns superpoint-level predictions
        seg_logits_superpoint = self.transformer(nag)  # [num_superpoints_L1, num_classes]
        
        # If multi-stage output (list), use finest level (L1)
        if isinstance(seg_logits_superpoint, list):
            seg_logits_superpoint = seg_logits_superpoint[0]
        
        result = {}
        
        # 2. Compute loss at SUPERPOINT level when labels are available
        # (both train and eval, so evaluator can always log val_loss)
        if "segment" in input_dict:
            # Extract superpoint-level label histogram from NAG
            # NAG[1] contains voxel-level data (L1)
            # In NAG terminology, L1 voxels ARE the "superpoints" for the first partition level
            y_hist_L1 = nag[1].get('y')  # [num_superpoints_L1, num_classes+1]
            
            if y_hist_L1 is None:
                # Fallback: use voxel-level labels from input_dict
                y_voxel = input_dict.get("segment")
                if y_voxel is None:
                    raise ValueError("No labels found for training! NAG[1]['y'] is None and input_dict['segment'] is missing.")
                
                # If y_voxel is histogram format, use it directly
                if y_voxel.dim() == 2 and y_voxel.shape[1] == self.num_classes + 1:
                    y_hist_L1 = y_voxel
                else:
                    # If y_voxel is hard labels, convert to one-hot histogram
                    # (This should not happen if NAG is properly constructed)
                    get_root_logger().warning("NAG[1]['y'] is None, using input_dict['segment'] as fallback")
                    if y_voxel.dim() == 1:
                        # Hard labels: convert to histogram
                        import torch.nn.functional as F
                        y_hist_L1 = F.one_hot(y_voxel.long(), num_classes=self.num_classes + 1).float()
                    else:
                        # Already histogram
                        y_hist_L1 = y_voxel
            
            # Extract target labels (argmax of histogram, excluding ignore class)
            # y_hist_L1 shape: [num_superpoints, num_classes+1]
            # Last column (index num_classes) is the ignore/void class
            y_target = y_hist_L1[:, :self.num_classes].argmax(dim=1)  # [num_superpoints]
            
            # Compute loss at superpoint level
            loss = self.criteria(seg_logits_superpoint, y_target)
            result["loss"] = loss
            
            # Store superpoint logits for debugging
            result["seg_logits_superpoint"] = seg_logits_superpoint
        
        # 3. Propagate predictions to point level (for evaluation and visualization)
        # This is ONLY for output, NOT for loss computation!
        if sub_cluster is not None:
            # Use sub_cluster mapping: L1 (voxels) -> L0 (raw points)
            l0_super_index = sub_cluster.to_super_index()
            seg_logits_l0 = seg_logits_superpoint[l0_super_index]  # [N, num_classes]
            result["seg_logits"] = seg_logits_l0
        else:
            # Fallback: use voxel_inverse if available
            voxel_inverse = input_dict.get("voxel_inverse")
            if voxel_inverse is not None:
                if isinstance(voxel_inverse, torch.Tensor):
                    seg_logits_l0 = seg_logits_superpoint[voxel_inverse]
                else:
                    seg_logits_l0 = seg_logits_superpoint[torch.from_numpy(voxel_inverse).long().to(seg_logits_superpoint.device)]
                result["seg_logits"] = seg_logits_l0
            else:
                # No mapping available, return superpoint-level logits
                # (This may cause dimension mismatch in evaluation, but prevents crash)
                result["seg_logits"] = seg_logits_superpoint
                get_root_logger().warning("No sub_cluster or voxel_inverse mapping found, returning superpoint-level logits")
        
        # Also store voxel-level logits (same as superpoint in current NAG structure)
        result["seg_logits_voxel"] = seg_logits_superpoint

        return result

    def _compute_partition_metrics(
        self, nag: SuperpointHierarchy, input_dict: Dict, sub_cluster: Optional[Cluster]
    ) -> Dict:
        """Compute partition quality metrics (Oracle mIoU) at L0 (raw point level).
        
        ⚠️ DDP-Safe: Returns only Tensors, no NAG objects!
        
        Oracle computation:
            1. Get label histogram at L1 (voxel level)
            2. Propagate majority label L2→L1→L0
            3. Compare with raw point ground truth
        
        Returns:
            dict containing:
                - y_pred: Oracle predictions at raw point level
                - y_true: Ground truth labels at raw point level
                - oracle_acc: Oracle accuracy
                - superpoint_labels: Per-point superpoint assignment
        """
        result = {}
        
        # Debug: Check NAG structure after L0 prepending
        # After prepend: nag[0]=L0(virtual), nag[1]=L1(voxels), nag[2]=L2(superpoints)
        
        # ========== Get label histogram from appropriate level ==========
        # With L0 prepended: L1 is voxels (at index 1), L2+ are superpoints
        # Labels ('y') are at L1 (voxel level)
        y = None
        if nag.num_levels > 1:
            y = nag[1].get("y")  # L1 = voxels after prepending
        if y is None and nag.num_levels > 0:
            y = nag[0].get("y")  # Fallback: try L0
        
        # ========== Superpoint Labels Output (at L0) ==========
        # Propagate all levels of superpoint IDs to raw points (L0)
        if nag.num_levels > 1 and sub_cluster is not None:
            # L0→L1 mapping
            l0_super_index = sub_cluster.to_super_index()  # [N] point → voxel_id
            
            # Build superpoint ID at each level, propagated to L0 (raw points)
            superpoint_labels_multi = []
            
            # Level 1: Voxel IDs at point level
            # Each point gets its voxel ID
            voxel_ids_l0 = l0_super_index  # [N] point → voxel_id
            superpoint_labels_multi.append(voxel_ids_l0)
            
            # Level 2+: Superpoint IDs at point level
            # Chain propagation: L0 → L1 → L2 → ...
            current_mapping = l0_super_index  # Start from L0→L1
            
            for level in range(1, nag.num_levels):
                # Get L[level] → L[level+1] mapping
                level_super_index = nag[level].get("super_index")
                if level_super_index is not None:
                    # Chain: current_mapping gives L0→L[level]
                    # level_super_index gives L[level]→L[level+1]
                    # Result: L0→L[level+1]
                    current_mapping = level_super_index[current_mapping]
                    superpoint_labels_multi.append(current_mapping)
                else:
                    break
            
            # Final superpoint ID (highest level)
            if superpoint_labels_multi:
                result["superpoint_labels"] = superpoint_labels_multi[-1]  # [N] raw_point → highest_superpoint_id
                result["superpoint_labels_multi"] = superpoint_labels_multi  # List of [N] tensors, one per level
        

        # ========== Oracle mIoU Computation at L0 ==========
        # Need raw point labels for evaluation
        y_raw = input_dict.get("segment_raw")
        if y_raw is None:
            # Fallback: try segment (might be voxel-level histogram)
            y_raw = input_dict.get("segment")
        
        if y is None or y_raw is None:
            # Can't compute oracle without labels
            return result

        # Oracle: each superpoint/voxel takes majority label
        y_hist = y[:, :self.num_classes] if y.dim() == 2 and y.shape[1] > self.num_classes else y
        if y_hist.dim() == 2:
            y_oracle_l1 = y_hist.argmax(dim=1)  # [M] voxel-level oracle
        else:
            y_oracle_l1 = y_hist
        
        # If we have L2 superpoints, propagate oracle from L2→L1
        if nag.num_levels > 2:
            y_l2 = nag[2].get("y")
            if y_l2 is not None:
                y_hist_l2 = y_l2[:, :self.num_classes] if y_l2.dim() == 2 and y_l2.shape[1] > self.num_classes else y_l2
                if y_hist_l2.dim() == 2:
                    y_oracle_l2 = y_hist_l2.argmax(dim=1)
                    # Propagate L2→L1
                    l1_super_index = nag[1].get("super_index")
                    if l1_super_index is not None:
                        y_oracle_l1 = y_oracle_l2[l1_super_index]

        # Propagate L1→L0 (voxel → raw_point)
        if sub_cluster is not None:
            l0_super_index = sub_cluster.to_super_index()
            y_pred = y_oracle_l1[l0_super_index]
        else:
            # Fallback to voxel_inverse
            voxel_inverse = input_dict.get("voxel_inverse")
            if voxel_inverse is not None:
                if isinstance(voxel_inverse, torch.Tensor):
                    y_pred = y_oracle_l1[voxel_inverse]
                else:
                    y_pred = y_oracle_l1[torch.from_numpy(voxel_inverse).long().to(y_oracle_l1.device)]
            else:
                y_pred = y_oracle_l1  # Can't propagate, return voxel-level

        # Ground truth at raw point level
        if isinstance(y_raw, torch.Tensor):
            y_true = y_raw
        else:
            y_true = torch.from_numpy(y_raw).long().to(y_pred.device)
        
        # Handle 2D histogram ground truth
        if y_true.dim() == 2:
            # segment_raw is still histogram (shouldn't happen, but handle it)
            y_true = y_true[:, :self.num_classes].argmax(dim=1)
        
        # Ensure same size
        if y_pred.shape[0] != y_true.shape[0]:
            # Size mismatch, likely y_true is voxel-level
            # Debug: log the mismatch
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                f"[_compute_partition_metrics] Size mismatch detected:\n"
                f"  y_pred shape: {y_pred.shape}\n"
                f"  y_true shape: {y_true.shape}\n"
                f"  sub_cluster: {sub_cluster is not None}\n"
                f"  voxel_inverse: {input_dict.get('voxel_inverse') is not None}\n"
                f"  segment_raw: {input_dict.get('segment_raw') is not None}"
            )
            result["y_pred_voxel"] = y_pred
            result["y_true_voxel"] = y_true
            return result

        result["y_pred"] = y_pred
        result["y_true"] = y_true
        
        # Compute accuracy on valid labels (excluding ignore_index)
        valid_mask = (y_true >= 0) & (y_true < self.num_classes)
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
