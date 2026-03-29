"""
SPT (Superpoint Transformer) Components for EZ-SP

This submodule contains the Transformer-based components for superpoint
graph processing in EZ-SP Stage 2.

Components:
    - MLP, FFN, Classifier: Basic building blocks
    - BatchNorm, UnitSphereNorm, GroupNorm: Normalization layers
    - DropPath: Stochastic depth
    - SelfAttentionBlock: Multi-head self-attention with RPE
    - TransformerBlock: Pre-norm residual transformer
    - Stage, DownNFuseStage, UpNFuseStage, PointStage: Multi-scale processing
    - Pool operators: SumPool, MeanPool, MaxPool, MinPool, AttentivePool
    - Fusion operators: CatFusion, AdditiveFusion, IndexUnpool

Reference: Superpoint Transformer (src/nn/, src/models/components/spt.py)

Author: PointSpace Team
"""

from pointspace.models.backbone.ezsp.spt.mlp import MLP, FFN, Classifier
from pointspace.models.backbone.ezsp.spt.norm import (
    BatchNorm,
    UnitSphereNorm,
    GroupNorm,
    LayerNorm,
    InstanceNorm,
    GraphNorm,
    INDEX_BASED_NORMS,
)
from pointspace.models.backbone.ezsp.spt.dropout import DropPath, drop_path
from pointspace.models.backbone.ezsp.spt.attention import SelfAttentionBlock
from pointspace.models.backbone.ezsp.spt.transformer import TransformerBlock
from pointspace.models.backbone.ezsp.spt.pool import (
    pool_factory,
    SumPool,
    MeanPool,
    MaxPool,
    MinPool,
    StdPool,
    AttentivePool,
    AttentivePoolWithLearntQueries,
    BaseAttentivePool,
)
from pointspace.models.backbone.ezsp.spt.fusion import (
    fusion_factory,
    BaseFusion,
    CatFusion,
    AdditiveFusion,
    TakeFirstFusion,
    TakeSecondFusion,
    IndexUnpool,
)
from pointspace.models.backbone.ezsp.spt.stage import (
    Stage,
    DownNFuseStage,
    UpNFuseStage,
    PointStage,
)
from pointspace.models.backbone.ezsp.spt.spt import SPT
from pointspace.models.backbone.ezsp.spt.utils import init_weights, build_qk_scale_func


__all__ = [
    # MLP
    "MLP",
    "FFN",
    "Classifier",
    # Normalization
    "BatchNorm",
    "UnitSphereNorm",
    "GroupNorm",
    "LayerNorm",
    "InstanceNorm",
    "GraphNorm",
    "INDEX_BASED_NORMS",
    # Dropout
    "DropPath",
    "drop_path",
    # Attention
    "SelfAttentionBlock",
    # Transformer
    "TransformerBlock",
    # Pooling
    "pool_factory",
    "SumPool",
    "MeanPool",
    "MaxPool",
    "MinPool",
    "StdPool",
    "AttentivePool",
    "AttentivePoolWithLearntQueries",
    "BaseAttentivePool",
    # Fusion
    "fusion_factory",
    "BaseFusion",
    "CatFusion",
    "AdditiveFusion",
    "TakeFirstFusion",
    "TakeSecondFusion",
    "IndexUnpool",
    # Stages
    "Stage",
    "DownNFuseStage",
    "UpNFuseStage",
    "PointStage",
    # SPT Network
    "SPT",
    # Utils
    "init_weights",
    "build_qk_scale_func",
]
