"""
SparseCNN - Sparse Convolutional Neural Network for EZ-SP

This module implements a lightweight SparseCNN using spconv for point cloud
feature extraction. It is the first stage of EZ-SP that learns point embeddings
for partition learning.

The network architecture follows the original SPT implementation but uses
spconv instead of torchsparse.

Author: PointSpace Team
"""

from typing import List, Optional, Union
import torch
import torch.nn as nn
import spconv.pytorch as spconv
from torch import Tensor

from pointspace.models.builder import MODELS
from pointspace.models.utils.structure import Point
from pointspace.models.modules import PointModule
from pointspace.models.backbone.ezsp.graph_norm import GraphNorm


class SparseConvBlock(spconv.SparseModule):
    """
    Single Sparse Convolution Block

    Architecture: SubMConv3d -> Norm -> Activation -> (Residual)

    Args:
        in_channels: int - Input channels
        out_channels: int - Output channels
        kernel_size: int - Convolution kernel size
        dilation: int - Dilation rate
        norm: str - Normalization type: 'bn' (BatchNorm) | 'gn' (GraphNorm)
        norm_eps: float - Epsilon for normalization (helps with AMP stability)
        activation: str - Activation type: 'relu' | 'leakyrelu'
        residual: bool - Whether to use residual connection
        indice_key: str - Key for sparse convolution indices (for reuse)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        norm: str = "gn",
        norm_eps: float = 1e-3,  # Larger eps for AMP stability
        activation: str = "relu",
        residual: bool = True,
        indice_key: Optional[str] = None,
    ):
        super().__init__()
        self.residual = residual and (in_channels == out_channels)
        self.norm_type = norm

        # Submanifold sparse convolution
        padding = dilation * (kernel_size // 2)
        self.conv = spconv.SubMConv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False,
            indice_key=indice_key,
        )

        # Normalization
        if norm == "gn":
            self.norm = GraphNorm(out_channels, eps=norm_eps)
        elif norm == "bn":
            self.norm = nn.BatchNorm1d(out_channels, eps=norm_eps, momentum=0.01)
        elif norm == "none" or norm is None:
            self.norm = None
        else:
            self.norm = nn.BatchNorm1d(out_channels, eps=norm_eps, momentum=0.01)

        # Activation
        if activation == "relu":
            self.act = nn.ReLU(inplace=True)
        elif activation == "leakyrelu":
            self.act = nn.LeakyReLU(0.1, inplace=True)
        elif activation == "none" or activation is None:
            self.act = nn.Identity()
        else:
            self.act = nn.Identity()

        # Shortcut projection (if channels differ)
        if self.residual:
            self.shortcut = nn.Identity()
        elif residual:
            # Need projection for residual when channels differ
            self.shortcut = spconv.SubMConv3d(
                in_channels, out_channels, kernel_size=1, bias=False
            )
            self.residual = True
        else:
            self.shortcut = None

    def forward(
        self, x: spconv.SparseConvTensor, batch: Optional[Tensor] = None
    ) -> spconv.SparseConvTensor:
        """
        Forward pass

        Args:
            x: SparseConvTensor input
            batch: (N,) batch indices for GraphNorm

        Returns:
            SparseConvTensor output
        """
        identity = x.features if self.residual else None

        # Convolution
        out = self.conv(x)
        feat = out.features

        # Normalization
        if self.norm is not None:
            if self.norm_type == "gn" and batch is not None:
                feat = self.norm(feat, batch)
            elif isinstance(self.norm, nn.BatchNorm1d):
                feat = self.norm(feat)
            elif self.norm_type == "gn":
                # GraphNorm without batch - fall back to instance-like behavior
                feat = self.norm(feat, torch.zeros(feat.shape[0], dtype=torch.long, device=feat.device))

        # Activation
        feat = self.act(feat)

        # Residual
        if self.residual:
            if isinstance(self.shortcut, spconv.SubMConv3d):
                identity = self.shortcut(x).features
            feat = feat + identity

        return out.replace_feature(feat)


@MODELS.register_module("EZ-SparseCNN")
class SparseCNN(PointModule):
    """
    EZ-SP Sparse CNN Feature Extractor

    A lightweight 3-layer sparse CNN that extracts point embeddings for
    partition learning. Uses spconv for efficient sparse convolutions.

    Architecture:
        Input projection -> [ConvBlock x N] -> Output features

    The network processes Point objects and updates their 'feat' attribute
    with learned embeddings.

    Args:
        in_channels: int - Number of input feature channels
        channels: List[int] - Channel sizes for each conv block, default [32, 32, 32]
        kernel_size: int - Convolution kernel size, default 3
        dilation: int - Dilation rate, default 1
        norm: str - Normalization: 'gn' (GraphNorm) | 'bn' (BatchNorm1d)
        norm_eps: float - Epsilon for normalization (default 1e-3 for AMP stability)
        activation: str - Activation: 'relu' | 'leakyrelu'
        residual: bool - Use residual connections within blocks
        global_residual: bool - Add input to output (requires same channels)

    Input:
        point: Point object containing:
            - coord: [N, 3] point coordinates
            - feat: [N, in_channels] input features
            - grid_coord: [N, 3] voxelized coordinates
            - batch: [N] batch indices
            - offset: [B] cumulative point counts

    Output:
        point: Point object with updated feat: [N, channels[-1]]

    Example:
        >>> cnn = SparseCNN(in_channels=6, channels=[32, 32, 32])
        >>> point = Point(coord=coord, feat=feat, grid_coord=grid_coord, ...)
        >>> point = cnn(point)
        >>> embeddings = point.feat  # [N, 32]
    """

    def __init__(
        self,
        in_channels: int,
        channels: List[int] = [32, 32, 32],
        kernel_size: int = 3,
        dilation: int = 1,
        norm: str = "gn",
        norm_eps: float = 1e-3,  # Larger eps for AMP stability
        activation: str = "relu",
        residual: bool = True,
        global_residual: bool = False,
        last_norm: bool = True,
        last_activation: bool = True,
        frozen: bool = False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.channels = channels
        self.out_channels = channels[-1]
        self.global_residual = global_residual
        self.norm_type = norm
        self.norm_eps = norm_eps
        self._frozen = frozen

        # Input projection if channels don't match
        if in_channels != channels[0]:
            self.input_proj = nn.Linear(in_channels, channels[0])
        else:
            self.input_proj = nn.Identity()

        # Build sparse conv blocks
        self.blocks = nn.ModuleList()
        prev_ch = channels[0]
        num_blocks = len(channels)

        for i, ch in enumerate(channels):
            is_last = (i == num_blocks - 1)
            block = SparseConvBlock(
                in_channels=prev_ch,
                out_channels=ch,
                kernel_size=kernel_size,
                dilation=dilation,
                norm=norm if (last_norm or not is_last) else "none",
                norm_eps=norm_eps,
                activation=activation if (last_activation or not is_last) else "none",
                residual=residual,
                indice_key=f"ezsp_subm{i}",
            )
            self.blocks.append(block)
            prev_ch = ch

        # Global residual projection
        if global_residual:
            if in_channels != channels[-1]:
                self.global_proj = nn.Linear(in_channels, channels[-1])
            else:
                self.global_proj = nn.Identity()

        # Apply frozen state if requested
        if frozen:
            self.freeze()

    @property
    def frozen(self) -> bool:
        """Whether the CNN is frozen"""
        return self._frozen

    def freeze(self):
        """Freeze all parameters"""
        if not self._frozen:
            for param in self.parameters():
                param.requires_grad = False
            self._frozen = True

    def unfreeze(self):
        """Unfreeze all parameters"""
        if self._frozen:
            for param in self.parameters():
                param.requires_grad = True
            self._frozen = False

    def forward(self, point: Point) -> Point:
        """
        Forward pass with train/eval dtype adaptation
        
        Strategy:
        - Training mode: Use FP16 (fast, memory efficient)
        - Eval mode: Force FP32 (avoid spconv algorithm errors)
        
        This solves:
        - Training: 100% success with FP16 (speed + memory)
        - Eval: 100% success with FP32 (stability)
        - No dtype conversion overhead in training (most iterations)

        Args:
            point: Point object with coord, feat, grid_coord, batch, offset

        Returns:
            point: Point object with updated feat embeddings
        """
        # Eval mode: Force FP32 to avoid spconv kernel issues
        # Training mode: Keep AMP setting (FP16) for speed
        if not self.training:
            # Evaluation/inference: force float32
            with torch.amp.autocast('cuda', enabled=False):
                return self._forward_impl(point)
        else:
            # Training: use global AMP setting (FP16 if enabled)
            return self._forward_impl(point)
    
    def _forward_impl(self, point: Point) -> Point:
        """Actual forward implementation (dtype-agnostic).
        
        In the official EZ-SP architecture:
        - Data comes from GridSampling3D already at voxel level
        - coord: [M, 3] voxel center coordinates
        - feat: [M, C] aggregated voxel features
        - grid_coord: [M, 3] integer grid coordinates (for sparsify)
        
        No voxel_mode detection needed - just run standard sparse conv.
        """
        # NOTE: Legacy voxel_mode code removed. In official EZ-SP architecture,
        # GridSampling3D handles voxelization in the data pipeline. SparseCNN
        # receives voxel-level data directly (coord=voxel centers, feat=voxel features).
        
        # Store original features for global residual (BEFORE projection)
        feat_input = None
        if self.global_residual:
            feat_input = point.feat.clone()

        # Input projection
        point.feat = self.input_proj(point.feat)

        # Sparsify (create SparseConvTensor)
        point.sparsify()

        # Forward through conv blocks
        sparse_feat = point.sparse_conv_feat
        batch = point.batch

        for block in self.blocks:
            sparse_feat = block(sparse_feat, batch)

        # Extract features from sparse tensor
        point.feat = sparse_feat.features
        point.sparse_conv_feat = sparse_feat

        # Global residual
        if feat_input is not None:
            point.feat = point.feat + self.global_proj(feat_input)

        return point


@MODELS.register_module("EZ-SparseCNN-v2")
class SparseCNNv2(SparseCNN):
    """
    SparseCNN v2 with additional features

    Adds:
    - Optional dropout
    - Output normalization
    - Configurable final activation
    """

    def __init__(
        self,
        in_channels: int,
        channels: List[int] = [32, 32, 32],
        kernel_size: int = 3,
        dilation: int = 1,
        norm: str = "gn",
        norm_eps: float = 1e-3,
        activation: str = "relu",
        residual: bool = True,
        global_residual: bool = False,
        dropout: float = 0.0,
        output_norm: bool = False,
        output_activation: bool = False,
    ):
        super().__init__(
            in_channels=in_channels,
            channels=channels,
            kernel_size=kernel_size,
            dilation=dilation,
            norm=norm,
            norm_eps=norm_eps,
            activation=activation,
            residual=residual,
            global_residual=global_residual,
        )

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if output_norm:
            if norm == "gn":
                self.output_norm = GraphNorm(channels[-1], eps=norm_eps)
            else:
                self.output_norm = nn.BatchNorm1d(channels[-1], eps=norm_eps, momentum=0.01)
        else:
            self.output_norm = None

        if output_activation:
            if activation == "relu":
                self.output_act = nn.ReLU(inplace=True)
            elif activation == "leakyrelu":
                self.output_act = nn.LeakyReLU(0.1, inplace=True)
            else:
                self.output_act = nn.Identity()
        else:
            self.output_act = None

    def forward(self, point: Point) -> Point:
        # Call parent forward
        point = super().forward(point)

        # Additional processing
        feat = point.feat

        # Dropout
        feat = self.dropout(feat)

        # Output norm
        if self.output_norm is not None:
            if isinstance(self.output_norm, GraphNorm):
                feat = self.output_norm(feat, point.batch)
            else:
                feat = self.output_norm(feat)

        # Output activation
        if self.output_act is not None:
            feat = self.output_act(feat)

        point.feat = feat
        return point
