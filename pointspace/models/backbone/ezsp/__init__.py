"""
EZ-SP (Easy Superpoints) Module for PointSpace

This module implements the EZ-SP architecture for end-to-end learnable
superpoint-based point cloud semantic segmentation.

Components:
    - SparseCNN: Sparse convolutional feature extractor
    - GraphNorm: Graph-aware normalization layer
    - GreedyContourPriorPartition: GPU-based hierarchical partition
    - SuperpointHierarchy: Multi-level superpoint data structure
    - EZSPTransformer: SPT-based semantic segmentation transformer

Author: PointSpace Team
"""

from pointspace.models.backbone.ezsp.graph_norm import GraphNorm, GraphNorm1d
from pointspace.models.backbone.ezsp.sparse_cnn import SparseCNN, SparseCNNv2
from pointspace.models.backbone.ezsp.superpoint_hierarchy import (
    Cluster,
    SuperpointLevel,
    SuperpointHierarchy,
)
from pointspace.models.backbone.ezsp.graph_partition import (
    GreedyContourPriorPartition,
)
from pointspace.models.backbone.ezsp.hierarchy_graph_transform import (
    HierarchyGraphTransform,
)
# from pointspace.models.backbone.ezsp.voxel_to_point_decoder import (
#     VoxelToPointDecoder,
#     LightweightVoxelToPointDecoder,
# )
from pointspace.models.backbone.ezsp.ezsp_transformer import EZSPTransformer

__all__ = [
    # Normalization
    "GraphNorm",
    "GraphNorm1d",
    # Feature extraction
    "SparseCNN",
    "SparseCNNv2",
    # Data structures
    "Cluster",
    "SuperpointLevel",
    "SuperpointHierarchy",
    # Partition
    "GreedyContourPriorPartition",
    "HierarchyGraphTransform",
    # "GreedyContourPriorPartitionSimple",
    # Voxel-to-point decoders
    # "VoxelToPointDecoder",
    # "LightweightVoxelToPointDecoder",
    # Transformer for semantic segmentation
    "EZSPTransformer",
]
