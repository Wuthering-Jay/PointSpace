"""
EZSP Contrastive Loss for Partition Learning

This loss trains point features such that points within the same semantic
class have high affinity (similar features) while points across different
classes have low affinity (dissimilar features).

The learned features are then used by GPUGreedyPartition to create
superpoints that respect semantic boundaries.

Reference: EZ-SP (https://arxiv.org/abs/2402.04991)
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple, Dict
from torch_geometric.nn import knn_graph

from pointspace.models.losses.builder import LOSSES


class BinaryFocalLoss(nn.Module):
    """Binary Focal Loss for imbalanced edge classification.

    Focal loss helps when there's class imbalance between inter-class
    and intra-class edges.

    Args:
        gamma: Focusing parameter (higher = more focus on hard examples)
        weight: Weight for positive class (intra-class edges)
        epsilon: Small constant for numerical stability
    """

    def __init__(
        self,
        gamma: float = 1.0,
        weight: float = 0.5,
        epsilon: float = 1e-6,
    ):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.epsilon = epsilon

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Args:
            pred: Predicted probabilities for positive class (N,)
            target: Boolean target labels (N,)

        Returns:
            Scalar loss
        """
        target = target.float()

        # Convert to probability of correct class
        # True -> p, False -> 1-p
        factor = 2 * target - 1  # True -> 1, False -> -1
        p = (1 - target) + pred * factor

        # Clamp for stability
        p = self.epsilon + (1 - 2 * self.epsilon) * p

        # Class-balanced weights
        weight = target * self.weight + (1 - target) * (1 - self.weight)

        # Focal loss: -(1-p)^gamma * log(p)
        loss = -(1 - p) ** self.gamma * torch.log(p)
        loss = loss * weight

        return loss.mean()


def compute_edge_distances(
    feat: Tensor,
    edge_index: Tensor,
    sharding: Optional[int] = None,
) -> Tensor:
    """Compute Euclidean distances for edges.

    Args:
        feat: Node features (N, D)
        edge_index: Edge indices (2, E)
        sharding: Process edges in chunks (for memory)

    Returns:
        Edge distances (E,)
    """
    src, dst = edge_index

    if sharding is None or sharding <= 0:
        # Process all at once
        diff = feat[src] - feat[dst]
        return diff.norm(dim=-1)
    else:
        # Process in chunks
        num_edges = edge_index.shape[1]
        if isinstance(sharding, float) and 0 < sharding < 1:
            chunk_size = max(1, int(num_edges * sharding))
        else:
            chunk_size = int(sharding)

        distances = []
        for i in range(0, num_edges, chunk_size):
            chunk_src = src[i:i + chunk_size]
            chunk_dst = dst[i:i + chunk_size]
            diff = feat[chunk_src] - feat[chunk_dst]
            distances.append(diff.norm(dim=-1))

        return torch.cat(distances)


@LOSSES.register_module()
class EZSPContrastiveLoss(nn.Module):
    """Contrastive loss for EZ-SP partition learning.

    Trains features such that same-class points have high affinity
    and different-class points have low affinity. This enables
    the partition algorithm to create semantically meaningful superpoints.

    The affinity between two points is: exp(-||f_i - f_j|| / T)
    where f_i, f_j are the learned features and T is the temperature.

    Args:
        affinity_temperature: Temperature for affinity computation
        focal_gamma: Focal loss gamma parameter
        adaptive_sampling_ratio: Ratio of minority class after sampling.
            If None, no adaptive sampling. Default: 0.9
        num_classes: Number of semantic classes (excluding void)
        k: Number of KNN neighbors for graph construction
        r_max: Maximum neighbor distance
        loss_weight: Weight for this loss term
        train_only: Only compute during training
        sharding: Process edges in chunks (memory optimization)

    Forward Args:
        feat: Point features (N, D)
        pos: Point positions (N, 3)
        segment: Segment labels (N,) - class labels for each point
        offset: Batch offsets (B,)

    Returns:
        Scalar loss value
    """

    INTER_EDGE = 0  # Different class
    INTRA_EDGE = 1  # Same class

    def __init__(
        self,
        affinity_temperature: float = 1.0,
        focal_gamma: float = 1.0,
        adaptive_sampling_ratio: Optional[float] = 0.9,
        num_classes: int = None,
        k: int = 8,
        r_max: Optional[float] = None,
        loss_weight: float = 1.0,
        train_only: bool = True,
        sharding: Optional[int] = None,
    ):
        super().__init__()

        self.affinity_temperature = affinity_temperature
        self.adaptive_sampling_ratio = adaptive_sampling_ratio
        self.num_classes = num_classes
        self.k = k
        self.r_max = r_max
        self.loss_weight = loss_weight
        self.train_only = train_only
        self.sharding = sharding

        self.loss_fn = BinaryFocalLoss(gamma=focal_gamma)

    def build_knn_graph(
        self,
        pos: Tensor,
        batch: Optional[Tensor] = None,
    ) -> Tensor:
        """Build KNN graph for loss computation."""
        edge_index = knn_graph(pos, k=self.k, batch=batch, loop=False)

        if self.r_max is not None:
            src, dst = edge_index
            dist = (pos[src] - pos[dst]).norm(dim=-1)
            mask = dist <= self.r_max
            edge_index = edge_index[:, mask]

        return edge_index

    def features_to_affinity(self, feat: Tensor, edge_index: Tensor) -> Tensor:
        """Convert features to edge affinities."""
        distances = compute_edge_distances(feat, edge_index, self.sharding)
        affinity = torch.exp(-distances / self.affinity_temperature)
        return affinity

    def adaptive_sample_edges(
        self,
        edge_index: Tensor,
        target: Tensor,
        minority_class: int = 0,
    ) -> Tuple[Tensor, Tensor]:
        """Adaptively sample edges to balance classes.

        Keeps all minority class edges and samples majority class to achieve
        desired ratio.
        """
        count_minority = (target == minority_class).sum()
        count_majority = target.shape[0] - count_minority

        # Sample from majority class
        n_sample_majority = int(
            (1.0 / self.adaptive_sampling_ratio - 1) * count_minority
        )
        n_sample_majority = min(n_sample_majority, count_majority)

        # Get indices
        minority_idx = torch.where(target == minority_class)[0]
        majority_idx = torch.where(target != minority_class)[0]

        # Random sample majority
        if n_sample_majority < majority_idx.shape[0]:
            perm = torch.randperm(majority_idx.shape[0], device=target.device)
            majority_idx = majority_idx[perm[:n_sample_majority]]

        # Combine
        sampled_idx = torch.cat([minority_idx, majority_idx])

        return edge_index[:, sampled_idx], target[sampled_idx]

    def forward(
        self,
        feat: Tensor,
        pos: Tensor,
        segment: Tensor,
        offset: Optional[Tensor] = None,
        edge_index: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """Compute contrastive loss.

        Args:
            feat: Point features (N, D)
            pos: Point positions (N, 3)
            segment: Ground truth segment labels (N,)
            offset: Batch offsets (B,)
            edge_index: Pre-computed graph (optional)

        Returns:
            Scalar loss
        """
        # Skip during eval if train_only
        if self.train_only and not self.training:
            return feat.sum() * 0.0

        device = feat.device

        # Build graph if not provided
        if edge_index is None:
            batch = None
            if offset is not None:
                from pointspace.models.utils import offset2batch
                batch = offset2batch(offset)
            edge_index = self.build_knn_graph(pos, batch)

        if edge_index.numel() == 0:
            return feat.sum() * 0.0

        # Remove self-loops
        mask = edge_index[0] != edge_index[1]
        edge_index = edge_index[:, mask]

        src, dst = edge_index

        # Handle void class (typically -1 or highest class index)
        # Remove edges where either endpoint is void
        if self.num_classes is not None:
            void_mask = (segment[src] >= self.num_classes) | (segment[dst] >= self.num_classes)
            valid_mask = ~void_mask
            edge_index = edge_index[:, valid_mask]
            src, dst = edge_index

        if edge_index.numel() == 0:
            return feat.sum() * 0.0

        # Compute target: 1 = same class (intra), 0 = different class (inter)
        target = (segment[src] == segment[dst]).long()

        # Check if we have any inter-class edges
        n_inter = (target == self.INTER_EDGE).sum()
        if n_inter == 0:
            return feat.sum() * 0.0

        # Adaptive sampling during training
        if self.training and self.adaptive_sampling_ratio is not None:
            edge_index, target = self.adaptive_sample_edges(
                edge_index, target, minority_class=self.INTER_EDGE
            )

        if edge_index.numel() == 0:
            return feat.sum() * 0.0

        # Compute predicted affinity
        pred_affinity = self.features_to_affinity(feat, edge_index)

        # Compute loss
        loss = self.loss_fn(pred_affinity, target.bool())

        return loss * self.loss_weight


@LOSSES.register_module()
class EZSPPartitionLoss(EZSPContrastiveLoss):
    """Alias for EZSPContrastiveLoss.

    Named to match the PartitionCriterion from the original SPT codebase.
    """
    pass
