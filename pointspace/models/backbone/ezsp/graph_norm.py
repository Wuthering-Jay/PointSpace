"""
GraphNorm - Graph Normalization for Point Cloud

This module implements GraphNorm which normalizes features per-graph (batch),
unlike BatchNorm which normalizes across the entire batch.

This is essential for point cloud processing where different superpoints/clusters
have vastly different sizes, and BatchNorm would destroy local feature distributions.

Author: PointSpace Team
"""

import torch
import torch.nn as nn
from torch import Tensor
from torch_scatter import scatter_mean


class GraphNorm(nn.Module):
    """
    Graph Normalization Layer

    Normalizes features independently for each graph/batch, handling variable-sized
    point clouds properly.

    Unlike BatchNorm which computes statistics across the entire batch,
    GraphNorm computes mean and variance for each sample separately.

    Formula:
        x_norm = (x - mean_per_graph) / std_per_graph
        output = weight * x_norm + bias  (if affine=True)

    Args:
        num_features: int - Number of input features (channels)
        eps: float - Small constant for numerical stability, default 1e-5
        affine: bool - If True, use learnable weight and bias, default True
        momentum: float - Momentum for running stats (not used, for API compatibility)

    Shape:
        - Input: (N, C) where N is total points, C is num_features
        - batch: (N,) batch index for each point
        - Output: (N, C)

    Example:
        >>> norm = GraphNorm(64)
        >>> x = torch.randn(1000, 64)
        >>> batch = torch.zeros(1000, dtype=torch.long)
        >>> batch[500:] = 1  # Two graphs with 500 points each
        >>> out = norm(x, batch)
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        affine: bool = True,
        momentum: float = 0.1,  # For API compatibility, not used
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters"""
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, x: Tensor, batch: Tensor) -> Tensor:
        """
        Forward pass

        Args:
            x: (N, C) Point features
            batch: (N,) Batch index for each point (0, 0, ..., 1, 1, ..., B-1)

        Returns:
            x_norm: (N, C) Normalized features
        """
        # Compute per-graph mean: [B, C]
        mean = scatter_mean(x, batch, dim=0)

        # Center features
        x_centered = x - mean[batch]

        # Compute per-graph variance: [B, C]
        var = scatter_mean(x_centered**2, batch, dim=0)

        # Normalize
        std = (var + self.eps).sqrt()
        x_norm = x_centered / std[batch]

        # Apply affine transformation
        if self.affine:
            x_norm = x_norm * self.weight + self.bias

        return x_norm

    def extra_repr(self) -> str:
        return f"{self.num_features}, eps={self.eps}, affine={self.affine}"


class GraphNorm1d(GraphNorm):
    """Alias for GraphNorm, for API consistency with nn.BatchNorm1d"""

    pass
