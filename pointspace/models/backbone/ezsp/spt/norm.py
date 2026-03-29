"""
Normalization Components for SPT.

This module provides normalization layers used in the Superpoint Transformer.

Reference: src/nn/norm.py from Superpoint Transformer

Author: PointSpace Team
"""

import torch
from torch import nn
from torch_scatter import scatter
from torch_geometric.nn.norm import LayerNorm, InstanceNorm, GraphNorm
from torch_geometric.nn.inits import ones, zeros
from torch_geometric.utils import degree
from typing import Optional


__all__ = [
    "BatchNorm",
    "UnitSphereNorm",
    "GroupNorm",
    "LayerNorm",
    "InstanceNorm",
    "GraphNorm",
    "INDEX_BASED_NORMS",
]


class BatchNorm(nn.Module):
    """BatchNorm for graph/point cloud data.

    Handles both sparse [N, D] and dense [B, N, D] inputs efficiently.

    Credits: torch-points3d
    """

    def __init__(self, num_features: int, **kwargs):
        """Initialize BatchNorm.

        Args:
            num_features: Number of input features.
            **kwargs: Additional arguments passed to nn.BatchNorm1d.
        """
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(num_features, **kwargs)

    def _forward_dense(self, x: torch.Tensor) -> torch.Tensor:
        """Forward for dense [B, N, D] tensors."""
        return self.batch_norm(x.permute(0, 2, 1)).permute(0, 2, 1)

    def _forward_sparse(self, x: torch.Tensor) -> torch.Tensor:
        """Forward for sparse [N, D] tensors.

        BatchNorm1D is not optimized for 2D tensors. The first dimension
        is supposed to be the batch and therefore not very large. So we
        introduce a custom version that leverages BatchNorm1D in a more
        optimized way.
        """
        x = x.unsqueeze(2)
        x = x.transpose(0, 2)
        x = self.batch_norm(x)
        x = x.transpose(0, 2)
        return x.squeeze(dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape [N, D] or [B, N, D].

        Returns:
            Normalized tensor of same shape.
        """
        if x.dim() == 2:
            return self._forward_sparse(x)
        elif x.dim() == 3:
            return self._forward_dense(x)
        else:
            raise ValueError(f"Unsupported number of dimensions: {x.dim()}")


def scatter_mean_weighted(
    src: torch.Tensor,
    idx: torch.Tensor,
    w: torch.Tensor,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    """Compute weighted mean using scatter operations.

    Args:
        src: Source tensor.
        idx: Index tensor for scatter.
        w: Weight tensor.
        dim_size: Size of output dimension.

    Returns:
        Weighted mean tensor.
    """
    w = w.float().view(-1, 1)
    weighted_src = src * w
    sum_weighted = scatter(weighted_src, idx, dim=0, dim_size=dim_size, reduce="add")
    sum_w = scatter(w, idx, dim=0, dim_size=dim_size, reduce="add").clamp_(min=1e-6)
    return sum_weighted / sum_w


class UnitSphereNorm(nn.Module):
    """Normalize positions of same-segment nodes in a unit sphere of diameter 1.

    This is useful for normalizing local coordinates within superpoints.

    Args:
        log_diameter: Whether the returned diameter should be log-normalized.
            This may be useful if using the diameter as a feature in downstream
            learning tasks.
    """

    def __init__(self, log_diameter: bool = False):
        """Initialize UnitSphereNorm.

        Args:
            log_diameter: Whether to return log-normalized diameter.
        """
        super().__init__()
        self.log_diameter = log_diameter

    def forward(
        self,
        pos: torch.Tensor,
        idx: Optional[torch.Tensor] = None,
        w: Optional[torch.Tensor] = None,
        num_super: Optional[int] = None,
    ) -> tuple:
        """Forward pass.

        Args:
            pos: Position tensor of shape [N, 3].
            idx: Segment indices of shape [N]. If None, all points are treated
                as one segment.
            w: Optional weights of shape [N].
            num_super: Number of segments (for scatter dim_size).

        Returns:
            Tuple of (normalized_pos, diameter).
        """
        # Normalization
        if idx is None:
            pos, diameter = self._forward(pos, w=w)
        else:
            pos, diameter = self._forward_scatter(pos, idx, w=w, num_super=num_super)

        # Log-normalize the diameter if required
        if self.log_diameter:
            diameter = torch.log(diameter + 1)

        return pos, diameter

    def _forward(
        self,
        pos: torch.Tensor,
        w: Optional[torch.Tensor] = None,
    ) -> tuple:
        """Forward without scatter operations.

        Applies the sphere normalization on all pos coordinates together.
        """
        # Compute the diameter (max span along main axes)
        min_ = pos.min(dim=0).values
        max_ = pos.max(dim=0).values
        diameter = (max_ - min_).max()

        # Compute the center of the nodes
        if w is None:
            center = pos.mean(dim=0)
        else:
            w_sum = w.float().sum()
            w_sum = 1 if w_sum == 0 else w_sum
            center = (pos * w.view(-1, 1).float()).sum(dim=0) / w_sum
        center = center.view(1, -1)

        # Unit-sphere normalization
        pos = (pos - center) / (diameter + 1e-2)

        return pos, diameter.view(1, 1)

    def _forward_scatter(
        self,
        pos: torch.Tensor,
        idx: torch.Tensor,
        w: Optional[torch.Tensor] = None,
        num_super: Optional[int] = None,
    ) -> tuple:
        """Forward with scatter operations.

        Applies the sphere normalization for each segment separately.
        """
        # Compute the diameter (max span along main axes)
        min_segment = scatter(pos, idx, dim=0, dim_size=num_super, reduce="min")
        max_segment = scatter(pos, idx, dim=0, dim_size=num_super, reduce="max")
        diameter_segment = (max_segment - min_segment).max(dim=1).values

        # Compute the center of the nodes
        if w is None:
            center_segment = scatter(pos, idx, dim=0, dim_size=num_super, reduce="mean")
        else:
            center_segment = scatter_mean_weighted(pos, idx, w, dim_size=num_super)

        # Compute per-node center and diameter
        center = center_segment[idx]
        diameter = diameter_segment[idx]

        # Unit-sphere normalization
        pos = (pos - center) / (diameter.view(-1, 1) + 1e-2)

        return pos, diameter_segment.view(-1, 1)


class GroupNorm(torch.nn.Module):
    """Group normalization on graphs.

    Supports two modes:
    - 'graph': Normalize input nodes based on the graph they belong to.
    - 'node': Apply BatchNorm on each node separately.

    Args:
        in_channels: Number of input channels.
        num_groups: Number of groups. Must be a divider of in_channels.
        eps: Small constant for numerical stability.
        affine: Whether to use learnable affine parameters.
        mode: Normalization mode ('graph' or 'node').
    """

    def __init__(
        self,
        in_channels: int,
        num_groups: int = 4,
        eps: float = 1e-5,
        affine: bool = True,
        mode: str = "graph",
    ):
        """Initialize GroupNorm.

        Args:
            in_channels: Number of input channels.
            num_groups: Number of groups. Must divide in_channels evenly.
            eps: Small constant for numerical stability.
            affine: Whether to use learnable affine parameters.
            mode: 'graph' for graph-wise norm, 'node' for node-wise norm.
        """
        super().__init__()

        assert in_channels % num_groups == 0, (
            f"in_channels ({in_channels}) must be a multiple of "
            f"num_groups ({num_groups})"
        )
        self.in_channels = in_channels
        self.num_groups = num_groups
        self.group_channels = in_channels // num_groups
        self.eps = eps
        self.mode = mode

        if affine:
            self.weight = nn.Parameter(torch.Tensor(in_channels))
            self.bias = nn.Parameter(torch.Tensor(in_channels))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters."""
        ones(self.weight)
        zeros(self.bias)

    def forward(
        self,
        x: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input features of shape [N, in_channels].
            batch: Optional batch indices of shape [N].

        Returns:
            Normalized features of shape [N, in_channels].
        """
        if self.mode == "graph":
            # If graph-wise normalization mode and 'batch' is not provided,
            # we consider all input nodes to belong to the same graph
            if batch is None:
                batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)

            # Separate group features using a new dimension
            x = x.view(-1, self.num_groups, self.group_channels)

            # Compute the number of items in each group normalization
            batch_size = int(batch.max()) + 1
            norm = degree(batch, batch_size, dtype=x.dtype).clamp_(min=1)
            norm = norm.mul_(self.group_channels).view(-1, 1, 1)

            # Compute the groupwise mean
            mean = (
                scatter(x, batch, dim=0, dim_size=batch_size, reduce="add").sum(
                    dim=-1, keepdim=True
                )
                / norm
            )

            # Groupwise mean-centering
            x = x - mean.index_select(0, batch)

            # Compute the groupwise variance
            var = (
                scatter(x * x, batch, dim=0, dim_size=batch_size, reduce="add").sum(
                    dim=-1, keepdim=True
                )
                / norm
            )

            # Groupwise std scaling
            out = x / (var + self.eps).sqrt().index_select(0, batch)

            # Restore input shape
            out = out.view(-1, self.in_channels)

            # Apply learnable mean and variance to each channel
            if self.weight is not None and self.bias is not None:
                out = out * self.weight + self.bias

            return out

        # GroupNorm in a node wise fashion
        if self.mode == "node":
            if batch is None:
                out = nn.functional.group_norm(
                    x, self.num_groups, weight=self.weight, bias=self.bias, eps=self.eps
                )
                return out

        raise ValueError(f"Unknown normalization mode: {self.mode}")

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(in_channels={self.in_channels}, "
            f"num_groups={self.num_groups}, mode={self.mode})"
        )


# Tuple of normalization layers that require batch index
INDEX_BASED_NORMS = (LayerNorm, InstanceNorm, GraphNorm, GroupNorm)
