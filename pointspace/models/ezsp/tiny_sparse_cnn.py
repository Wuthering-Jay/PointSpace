"""
TinySparseCNN for EZ-SP Feature Extraction

A lightweight sparse CNN for point cloud feature extraction, used in the partition
learning stage of EZ-SP. The network is intentionally small (32→32→32) to keep
the partition learning efficient while providing sufficient features for graph
clustering.

Reference: EZ-SP (https://arxiv.org/abs/2402.04991)
"""

import torch
import torch.nn as nn
import spconv.pytorch as spconv
from functools import partial
from typing import List, Union, Optional

from pointspace.models.builder import MODELS
from pointspace.models.utils import offset2batch


class SpConvBlock(spconv.SparseModule):
    """Single sparse convolution block: Conv -> Norm -> Activation

    Uses SubMConv3d to preserve the sparsity pattern.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        norm_fn=None,
        activation=None,
        bias: bool = False,
        indice_key: str = None,
    ):
        super().__init__()

        self.conv = spconv.SubMConv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=bias,
            indice_key=indice_key,
        )

        self.norm = norm_fn(out_channels) if norm_fn is not None else None
        self.activation = activation

    def forward(self, x: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        x = self.conv(x)
        if self.norm is not None:
            x = x.replace_feature(self.norm(x.features))
        if self.activation is not None:
            x = x.replace_feature(self.activation(x.features))
        return x


@MODELS.register_module("TinySparseCNN")
class TinySparseCNN(nn.Module):
    """Lightweight Sparse CNN for EZ-SP partition learning.

    This network extracts features from point clouds for use in graph-based
    partition algorithms. It's designed to be very lightweight while providing
    sufficient discriminative power for clustering.

    Default configuration (from EZ-SP paper):
        - 3 conv blocks with 32 channels each
        - kernel sizes: [7, 3, 3]
        - GroupNorm/BatchNorm + LeakyReLU

    Args:
        in_channels: Input feature dimension
        channels: List of channel sizes for each conv block. Default: [32, 32, 32]
        kernel_sizes: List of kernel sizes for each conv block. Default: [7, 3, 3]
        norm_fn: Normalization function factory. Default: BatchNorm1d
        activation: Activation function. Default: LeakyReLU
        last_norm: Whether to apply normalization in the last layer
        last_activation: Whether to apply activation in the last layer

    Input:
        point: Point dict with keys:
            - feat: (N, in_channels) input features
            - grid_coord: (N, 3) grid coordinates
            - offset: (B,) batch offsets
            - sparse_shape: sparse tensor shape (optional, computed if missing)

    Output:
        point: Point dict with updated 'feat' containing CNN features
    """

    def __init__(
        self,
        in_channels: int,
        channels: List[int] = None,
        kernel_sizes: Union[int, List[int]] = None,
        norm_fn=None,
        activation=None,
        last_norm: bool = True,
        last_activation: bool = True,
    ):
        super().__init__()

        # Default EZ-SP configuration
        if channels is None:
            channels = [32, 32, 32]
        if kernel_sizes is None:
            kernel_sizes = [7, 3, 3]
        if norm_fn is None:
            norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        if activation is None:
            activation = nn.LeakyReLU(inplace=True)

        # Ensure kernel_sizes is a list
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * len(channels)

        assert len(channels) == len(kernel_sizes), \
            f"channels ({len(channels)}) and kernel_sizes ({len(kernel_sizes)}) must have same length"

        self.in_channels = in_channels
        self.channels = channels
        self.kernel_sizes = kernel_sizes
        self.out_channels = channels[-1]

        # Build conv blocks
        blocks = []
        prev_channels = in_channels

        for i, (ch, ks) in enumerate(zip(channels, kernel_sizes)):
            is_last = (i == len(channels) - 1)
            blocks.append(
                SpConvBlock(
                    in_channels=prev_channels,
                    out_channels=ch,
                    kernel_size=ks,
                    norm_fn=norm_fn if (not is_last or last_norm) else None,
                    activation=activation if (not is_last or last_activation) else None,
                    bias=(norm_fn is None),
                    indice_key=f"tiny_cnn_{i}",
                )
            )
            prev_channels = ch

        self.blocks = nn.ModuleList(blocks)

    @property
    def out_dim(self):
        return self.out_channels

    def forward(self, point):
        """Forward pass.

        Args:
            point: Point dict containing feat, grid_coord, offset, etc.

        Returns:
            Updated point dict with CNN features
        """
        feat = point.feat
        grid_coord = point.grid_coord
        offset = point.offset

        # Get batch indices
        batch = offset2batch(offset)

        # Compute sparse shape if not provided
        if "sparse_shape" not in point.keys():
            sparse_shape = torch.add(
                torch.max(grid_coord, dim=0).values, 1
            ).tolist()
        else:
            sparse_shape = point.sparse_shape

        # Create sparse tensor coordinates: [batch, x, y, z]
        coords = torch.cat([batch.unsqueeze(-1).int(), grid_coord.int()], dim=-1)

        # Create SparseConvTensor
        sparse_feat = spconv.SparseConvTensor(
            features=feat,
            indices=coords.contiguous(),
            spatial_shape=sparse_shape,
            batch_size=len(offset),
        )

        # Apply conv blocks
        for block in self.blocks:
            sparse_feat = block(sparse_feat)

        # Update point features
        point.feat = sparse_feat.features
        point.sparse_conv_feat = sparse_feat

        return point


@MODELS.register_module("TinySparseCNNEncoder")
class TinySparseCNNEncoder(TinySparseCNN):
    """Encoder wrapper for TinySparseCNN that only outputs features.

    Same as TinySparseCNN but returns only the feature tensor instead of
    the full point dict. Useful when integrating with other models.
    """

    def forward(self, point):
        point = super().forward(point)
        return point.feat
