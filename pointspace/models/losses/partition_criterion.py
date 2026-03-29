"""
PartitionCriterion - Edge Classification Loss for Partition Learning

This module implements the partition learning loss function for EZ-SP.
The core idea is to classify edges as intra-class (same label) or inter-class
(different labels), encouraging the CNN to learn features that respect
semantic boundaries.

Reference: Superpoint Transformer (src/loss/partition_criterion.py)

Author: PointSpace Team
"""

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from pointspace.models.losses.builder import LOSSES, build_criteria
from pointspace.models.backbone.ezsp.superpoint_hierarchy import SuperpointHierarchy


class SPTBinaryFocalLoss(nn.Module):
    """
    Binary Focal Loss matching original SPT implementation.
    
    Expects probability input p ∈ (0, 1), not logits.
    
    Reference: src/loss/focal.py BinaryFocalLoss
    """
    
    def __init__(
        self,
        gamma: float = 1.0,
        alpha: float = 0.5,
        epsilon: float = 1e-6,
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # Weight for positive class (intra-edge)
        self.epsilon = epsilon
    
    def forward(self, p: Tensor, y: Tensor) -> Tensor:
        """
        Args:
            p: (N,) Predicted probabilities (affinity) in range (0, 1)
            y: (N,) Target labels (True=intra-edge, False=inter-edge)
        
        Returns:
            Scalar loss
        """
        # Convert y to boolean if needed
        y = y.bool()
        
        # Transform p based on target:
        # If y=True (intra), we want high p, so use p directly
        # If y=False (inter), we want low p, so use 1-p
        factor = 2 * y.float() - 1  # True -> 1, False -> -1
        p_transformed = (~y).float() + p * factor
        
        # Clamp for numerical stability
        p_transformed = self.epsilon + (1 - 2 * self.epsilon) * p_transformed
        
        # Class weights: alpha for positive (intra), 1-alpha for negative (inter)
        weight = y.float() * self.alpha + (~y).float() * (1 - self.alpha)
        
        # Focal loss: -weight * (1-p)^gamma * log(p)
        focal_term = (1 - p_transformed) ** self.gamma
        loss = -focal_term * torch.log(p_transformed) * weight
        
        return loss.mean()


@LOSSES.register_module()
class PartitionCriterion(nn.Module):
    """
    Partition Learning Edge Classification Loss

    Core idea:
        Transform the partition problem into edge classification:
        - INTER_EDGE (cross-class): target=0, should separate
        - INTRA_EDGE (same-class): target=1, should merge

    Affinity computation:
        affinity = exp(-||X_i - X_j|| / temperature)

    Loss function:
        loss = SPTBinaryFocalLoss(affinity, target)
        
    Note: Uses probability-based focal loss matching original SPT implementation.

    Args:
        gamma: float - Focal loss gamma parameter
        alpha: float - Weight for intra-edge class (0.5 = balanced)
        temperature: float - Affinity temperature parameter
        adaptive_sampling: bool - Whether to use adaptive edge sampling
        adaptive_sampling_ratio: float - Target ratio for minority class
        num_classes: int - Number of semantic classes
        loss_weight: float - Overall loss weight
        sharding: int | None - Sharding for large graphs (memory optimization)

    Input:
        nag: SuperpointHierarchy object containing:
            - level[0].x: [N, C] point features
            - level[0].edge_index: [2, E] edge indices
            - level[0].y: [N, num_classes] label histogram

    Output:
        loss: Tensor - Scalar loss value
        output: dict - Statistics (n_intra_edge, n_inter_edge, mean_affinity, etc.)
    """
    
    # Class constants matching original implementation
    INTER_EDGE_LABEL = 0
    INTRA_EDGE_LABEL = 1

    def __init__(
        self,
        gamma: float = 1.0,
        alpha: float = 0.5,
        temperature: float = 1.0,
        adaptive_sampling: bool = True,
        adaptive_sampling_ratio: float = 0.9,
        num_classes: int = 13,
        loss_weight: float = 1.0,
        sharding: Optional[int] = None,
        # Legacy parameters for backwards compatibility
        loss_function: Optional[dict] = None,
    ):
        super().__init__()
        
        # Use SPT-style BinaryFocalLoss (probability input, not logits)
        self.loss_fn = SPTBinaryFocalLoss(gamma=gamma, alpha=alpha)
        
        self.temperature = temperature
        self.adaptive_sampling = adaptive_sampling
        self.adaptive_sampling_ratio = adaptive_sampling_ratio
        self.num_classes = num_classes
        self.loss_weight = loss_weight
        self.sharding = sharding

    def forward(
        self,
        nag: SuperpointHierarchy,
    ) -> Tuple[Tensor, Dict]:
        """
        Compute partition loss following original SPT implementation.

        Args:
            nag: SuperpointHierarchy object

        Returns:
            loss: Scalar loss tensor
            output: Dict with statistics
        """
        level0 = nag[0]
        x = level0["x"]
        edge_index = level0["edge_index"]
        y = level0.get("y")

        if y is None:
            raise ValueError("Level 0 missing 'y' (labels) for partition criterion")
        
        if edge_index.numel() == 0:
            return self._fake_loss(x.device)

        # ========== A.1) Filter edges ==========
        
        # Remove self-loops (following original implementation)
        mask_self_loops = edge_index[0] == edge_index[1]
        edge_index = edge_index[:, ~mask_self_loops]
        
        if edge_index.numel() == 0:
            return self._fake_loss(x.device)
        
        src, dst = edge_index[0], edge_index[1]

        # Get majority class for each node (histogram -> label)
        if y.dim() == 2:
            majority_class_count, y_labels = y[:, :self.num_classes].max(dim=1)
        else:
            majority_class_count = torch.ones_like(y)
            y_labels = y
        
        # Discard edges containing pure void voxels (all points were void)
        mask_void_voxels = majority_class_count == 0
        mask_void_edges = mask_void_voxels[src] | mask_void_voxels[dst]
        edge_index = edge_index[:, ~mask_void_edges]
        
        if edge_index.numel() == 0:
            return self._fake_loss(x.device)
        
        src, dst = edge_index[0], edge_index[1]

        # ========== A.2) Compute target affinity ==========
        # Intra-edge (same class) = 1, Inter-edge (different class) = 0
        target_affinity = (y_labels[src] == y_labels[dst]).int()
        n_inter_edge = (target_affinity == self.INTER_EDGE_LABEL).sum().item()
        n_intra_edge = (target_affinity == self.INTRA_EDGE_LABEL).sum().item()
        
        if n_inter_edge == 0:
            return self._fake_loss(x.device)

        # ========== A.3) Adaptive sampling ==========
        if self.training and self.adaptive_sampling_ratio is not None:
            sampled_indices = self._binary_adaptive_sampling(
                target_affinity, 
                minority_class=self.INTER_EDGE_LABEL
            )
            
            if sampled_indices.numel() == 0:
                return self._fake_loss(x.device)
            
            edge_index = edge_index[:, sampled_indices]
            target_affinity = target_affinity[sampled_indices]
            src, dst = edge_index[0], edge_index[1]

        # ========== B) Predict edge affinity ==========
        predicted_affinity = self._features_to_edge_affinity(x, edge_index)
        
        # ========== C) Compute loss ==========
        # Directly pass affinity (probability) to loss function - matching SPT
        loss = self.loss_fn(predicted_affinity, target_affinity.bool())
        loss = loss * self.loss_weight

        # ========== D) Statistics for logging ==========
        with torch.no_grad():
            intra_mask = target_affinity == self.INTRA_EDGE_LABEL
            inter_mask = target_affinity == self.INTER_EDGE_LABEL
            mean_affinity_intra = (
                predicted_affinity[intra_mask].mean().item() if intra_mask.any() else 0.0
            )
            mean_affinity_inter = (
                predicted_affinity[inter_mask].mean().item() if inter_mask.any() else 0.0
            )

        output = {
            "loss": loss.detach(),
            "n_intra_edge": n_intra_edge,
            "n_inter_edge": n_inter_edge,
            "mean_affinity_intra": mean_affinity_intra,
            "mean_affinity_inter": mean_affinity_inter,
            "affinity_gap": mean_affinity_intra - mean_affinity_inter,
        }

        return loss, output
    
    def _fake_loss(self, device) -> Tuple[Tensor, Dict]:
        """Return fake loss when no valid edges exist."""
        fake_loss = torch.tensor(0.0, device=device, requires_grad=True)
        output = {
            "loss": torch.tensor(0.0, device=device),
            "n_intra_edge": 0,
            "n_inter_edge": 0,
            "mean_affinity_intra": 0.0,
            "mean_affinity_inter": 0.0,
            "affinity_gap": 0.0,
        }
        return fake_loss, output
    
    def _features_to_edge_affinity(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """
        Compute edge affinity from features.
        
        affinity = exp(-||x_i - x_j|| / temperature)
        """
        distances = self._compute_edge_distances(x, edge_index)
        affinity = torch.exp(-distances / self.temperature)
        return affinity
    
    def _compute_edge_distances(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Compute L2 distances for all edges, with optional sharding."""
        if self.sharding is None or edge_index.shape[1] <= self.sharding:
            return (x[edge_index[0]] - x[edge_index[1]]).norm(dim=1)
        
        # Sharding for memory efficiency on large graphs
        distances = []
        num_edges = edge_index.shape[1]
        for start in range(0, num_edges, self.sharding):
            end = min(start + self.sharding, num_edges)
            edge_batch = edge_index[:, start:end]
            dist_batch = (x[edge_batch[0]] - x[edge_batch[1]]).norm(dim=1)
            distances.append(dist_batch)
        return torch.cat(distances)

    def _binary_adaptive_sampling(
        self, 
        y: Tensor, 
        minority_class: int = 0
    ) -> Tensor:
        """
        Adaptive sampling of edges to balance inter/intra edges.
        
        Following original SPT implementation:
        - Take all samples from minority class
        - Sample from majority class so minority = adaptive_sampling_ratio of total
        
        Args:
            y: Edge labels (0=inter, 1=intra)
            minority_class: Which class is minority (typically inter=0)
        
        Returns:
            Tensor of sampled edge indices
        """
        count_minority = (y == minority_class).sum()
        count_majority = y.shape[0] - count_minority
        
        if count_minority == 0:
            return torch.tensor([], dtype=torch.long, device=y.device)
        
        # Take all minority samples
        sample_minority = torch.where(y == minority_class)[0]
        
        # Sample n_sample_majority from majority class
        # So that minority / total = adaptive_sampling_ratio
        # => n_majority = (1/ratio - 1) * n_minority
        n_sample_majority = int((1.0 / self.adaptive_sampling_ratio - 1) * count_minority)
        n_sample_majority = min(n_sample_majority, count_majority.item())
        
        if n_sample_majority <= 0:
            return sample_minority
        
        majority_indices = torch.where(y != minority_class)[0]
        perm = torch.randperm(len(majority_indices), device=y.device)[:n_sample_majority]
        sample_majority = majority_indices[perm]
        
        sampled_indices = torch.cat([sample_minority, sample_majority])
        return sampled_indices


@LOSSES.register_module()
class PartitionCriterionV2(PartitionCriterion):
    """
    Partition Criterion V2 with additional features

    Adds:
    - Hard negative mining
    - Contrastive loss variant
    - Per-class statistics
    """

    def __init__(
        self,
        gamma: float = 1.0,
        alpha: float = 0.5,
        temperature: float = 1.0,
        adaptive_sampling: bool = True,
        adaptive_sampling_ratio: float = 0.9,
        num_classes: int = 13,
        loss_weight: float = 1.0,
        sharding: Optional[int] = None,
        hard_negative_ratio: float = 0.0,
        margin: float = 0.5,
        loss_function: Optional[dict] = None,
    ):
        super().__init__(
            gamma=gamma,
            alpha=alpha,
            temperature=temperature,
            adaptive_sampling=adaptive_sampling,
            adaptive_sampling_ratio=adaptive_sampling_ratio,
            num_classes=num_classes,
            loss_weight=loss_weight,
            sharding=sharding,
            loss_function=loss_function,
        )
        self.hard_negative_ratio = hard_negative_ratio
        self.margin = margin

    def forward(
        self,
        nag: SuperpointHierarchy,
    ) -> Tuple[Tensor, Dict]:
        # Get base loss and output
        loss, output = super().forward(nag)

        # Add hard negative mining if enabled
        if self.hard_negative_ratio > 0 and self.training:
            hn_loss = self._hard_negative_loss(nag)
            loss = loss + self.hard_negative_ratio * hn_loss
            output["hard_negative_loss"] = hn_loss.detach()

        return loss, output

    def _hard_negative_loss(self, nag: SuperpointHierarchy) -> Tensor:
        """
        Hard negative mining loss

        Focuses on edges that are wrongly classified (high affinity inter-edges)
        """
        level0 = nag[0]
        x = level0["x"]
        edge_index = level0["edge_index"]
        y = level0.get("y")

        if y is None:
            return torch.tensor(0.0, device=x.device)

        src, dst = edge_index[0], edge_index[1]

        # Get labels
        if y.dim() == 2:
            y_src = y[src, : self.num_classes].argmax(dim=1)
            y_dst = y[dst, : self.num_classes].argmax(dim=1)
        else:
            y_src, y_dst = y[src], y[dst]

        # Find inter-class edges
        inter_mask = (y_src != y_dst) & (y_src >= 0) & (y_dst >= 0)

        if not inter_mask.any():
            return torch.tensor(0.0, device=x.device)

        # Compute affinity for inter edges
        inter_src = src[inter_mask]
        inter_dst = dst[inter_mask]
        feat_dist = (x[inter_src] - x[inter_dst]).norm(dim=1)
        affinity = torch.exp(-feat_dist / self.temperature)

        # Penalize high affinity inter edges (should be low)
        # Hinge loss: max(0, affinity - margin)
        hn_loss = F.relu(affinity - self.margin).mean()

        return hn_loss
