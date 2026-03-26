"""
EZSP (Easy Superpoint) Module for PointSpace

This module provides GPU-accelerated superpoint segmentation capabilities,
adapted from the Superpoint Transformer project.

Key components:
- TinySparseCNN: Lightweight sparse CNN for feature extraction
- GPUGreedyPartition: GPU-based graph clustering
- EZSPContrastiveLoss: Contrastive learning loss for partition training
- Utils: Bridge functions between PointSpace offset and NAG ptr formats

Reference:
- EZ-SP: https://arxiv.org/abs/2402.04991
- SPT: https://arxiv.org/abs/2306.08045
"""

from pointspace.models.ezsp.utils import (
    offset_to_ptr,
    ptr_to_offset,
    sizes_to_ptr,
    ptr_to_sizes,
    sizes_to_offset,
    offset_to_sizes,
    indices_to_ptr,
    ptr_to_batch,
    batch_to_ptr,
    super_index_to_sub_ptr,
    compute_super_index,
)

from pointspace.models.ezsp.tiny_sparse_cnn import (
    TinySparseCNN,
    TinySparseCNNEncoder,
    SpConvBlock,
)

from pointspace.models.ezsp.partition import (
    GPUGreedyPartition,
    HierarchicalPartition,
    scatter_mean_weighted,
)

from pointspace.models.ezsp.loss import (
    EZSPContrastiveLoss,
    EZSPPartitionLoss,
    BinaryFocalLoss,
    compute_edge_distances,
)

from pointspace.models.ezsp.segmentor import (
    EZSPPartitionSegmentor,
    EZSPBackbone,
    EZSPPartitionTrainer,
    build_ezsp_partition_model,
)

__all__ = [
    # Utils
    "offset_to_ptr",
    "ptr_to_offset",
    "sizes_to_ptr",
    "ptr_to_sizes",
    "sizes_to_offset",
    "offset_to_sizes",
    "indices_to_ptr",
    "ptr_to_batch",
    "batch_to_ptr",
    "super_index_to_sub_ptr",
    "compute_super_index",
    # CNN
    "TinySparseCNN",
    "TinySparseCNNEncoder",
    "SpConvBlock",
    # Partition
    "GPUGreedyPartition",
    "HierarchicalPartition",
    "scatter_mean_weighted",
    # Loss
    "EZSPContrastiveLoss",
    "EZSPPartitionLoss",
    "BinaryFocalLoss",
    "compute_edge_distances",
    # Segmentor
    "EZSPPartitionSegmentor",
    "EZSPBackbone",
    "EZSPPartitionTrainer",
    "build_ezsp_partition_model",
]
