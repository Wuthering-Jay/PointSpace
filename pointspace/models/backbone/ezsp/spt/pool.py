"""
Pooling Operators for SPT.

This module provides pooling operations for aggregating features from
child nodes to parent nodes in the superpoint hierarchy.

Reference: src/nn/pool.py from Superpoint Transformer

Author: PointSpace Team
"""

import torch
from torch import nn
from torch_geometric.nn.aggr import (
    SumAggregation,
    MeanAggregation,
    MaxAggregation,
    MinAggregation,
    StdAggregation,
)
from torch_scatter import scatter_sum
from torch_geometric.utils import softmax
from typing import Optional, Union, Type

from pointspace.models.backbone.ezsp.spt.utils import build_qk_scale_func


__all__ = [
    "pool_factory",
    "SumPool",
    "MeanPool",
    "MaxPool",
    "MinPool",
    "StdPool",
    "AttentivePool",
    "AttentivePoolWithLearntQueries",
    "BaseAttentivePool",
    "AggregationPoolMixIn",
]


def pool_factory(
    pool: Union[str, nn.Module, Type[nn.Module]],
    *args,
    **kwargs,
) -> nn.Module:
    """Build a Pool module from string or existing module.

    This helper is intended to be used in SPT and Stage constructors.

    Args:
        pool: Pooling specification. Can be:
            - 'max', 'min', 'mean', 'sum', 'std': String for standard aggregations
            - An existing pool module
            - A class to instantiate with args/kwargs

    Returns:
        Pooling module.
    """
    if isinstance(pool, (AggregationPoolMixIn, BaseAttentivePool)):
        return pool
    if pool == "max":
        return MaxPool()
    if pool == "min":
        return MinPool()
    if pool == "mean":
        return MeanPool()
    if pool == "sum":
        return SumPool()
    if pool == "std":
        return StdPool()
    return pool(*args, **kwargs)


class AggregationPoolMixIn:
    """MixIn to convert torch-geometric Aggregation modules into Pool modules.

    Provides a consistent forward signature for pooling operations.

    Forward signature:
        __call__(x_child, x_parent, index, edge_attr=None, num_pool=None)

    Args:
        x_child: Features of shape (Nc, Cc) for children nodes.
        x_parent: Not used for standard aggregations.
        index: LongTensor of shape (Nc,) indicating parent for each child.
        edge_attr: Not used for standard aggregations.
        num_pool: Number of parent nodes Np. If None, inferred from index.max() + 1.
    """

    def __call__(
        self,
        x_child: torch.Tensor,
        x_parent: torch.Tensor,
        index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        num_pool: Optional[int] = None,
    ) -> torch.Tensor:
        return super().__call__(x_child, index=index, dim_size=num_pool)


class SumPool(AggregationPoolMixIn, SumAggregation):
    """Sum pooling."""
    pass


class MeanPool(AggregationPoolMixIn, MeanAggregation):
    """Mean pooling."""
    pass


class MaxPool(AggregationPoolMixIn, MaxAggregation):
    """Max pooling."""
    pass


class MinPool(AggregationPoolMixIn, MinAggregation):
    """Min pooling."""
    pass


class StdPool(AggregationPoolMixIn, StdAggregation):
    """Standard deviation pooling."""
    pass


class BaseAttentivePool(nn.Module):
    """Base class for attentive pooling operations.

    This class implements attention-based pooling from child nodes to parent
    nodes. Subclasses must implement `_get_query()` to define how queries
    are computed from parent features.

    Args:
        dim: Feature dimension.
        num_heads: Number of attention heads.
        in_dim: Input feature dimension (if different from dim).
        out_dim: Output feature dimension (if different from dim).
        qkv_bias: Whether Q, K, V projections have bias.
        qk_dim: Query/key dimension per head.
        qk_scale: Query-key scaling mode.
        attn_drop: Dropout on attention weights.
        drop: Dropout on output features.
        in_rpe_dim: Input dimension for RPE.
        k_rpe: Whether to apply RPE to keys.
        q_rpe: Whether to apply RPE to queries.
        v_rpe: Whether to apply RPE to values.
        heads_share_rpe: Whether heads share RPE parameters.
    """

    def __init__(
        self,
        dim: Optional[int] = None,
        num_heads: int = 1,
        in_dim: Optional[int] = None,
        out_dim: Optional[int] = None,
        qkv_bias: bool = True,
        qk_dim: int = 8,
        qk_scale: Optional[Union[float, str]] = None,
        attn_drop: Optional[float] = None,
        drop: Optional[float] = None,
        in_rpe_dim: int = 9,
        k_rpe: bool = False,
        q_rpe: bool = False,
        v_rpe: bool = False,
        heads_share_rpe: bool = False,
    ):
        super().__init__()

        assert dim % num_heads == 0, "dim must be a multiple of num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.qk_dim = qk_dim
        self.qk_scale = build_qk_scale_func(dim, num_heads, qk_scale)
        self.heads_share_rpe = heads_share_rpe

        # Key-Value projection
        self.kv = nn.Linear(dim, qk_dim * num_heads + dim, bias=qkv_bias)

        # Build RPE encoders
        rpe_dim = qk_dim if heads_share_rpe else qk_dim * num_heads

        if not isinstance(k_rpe, bool):
            self.k_rpe = k_rpe
        else:
            self.k_rpe = nn.Linear(in_rpe_dim, rpe_dim) if k_rpe else None

        if not isinstance(q_rpe, bool):
            self.q_rpe = q_rpe
        else:
            self.q_rpe = nn.Linear(in_rpe_dim, rpe_dim) if q_rpe else None

        if v_rpe:
            raise NotImplementedError("v_rpe not yet implemented")

        # Optional input/output projections
        self.in_proj = nn.Linear(in_dim, dim) if in_dim is not None else None
        self.out_proj = nn.Linear(dim, out_dim) if out_dim is not None else None

        # Dropout layers
        self.attn_drop = (
            nn.Dropout(attn_drop) if attn_drop is not None and attn_drop > 0 else None
        )
        self.out_drop = (
            nn.Dropout(drop) if drop is not None and drop > 0 else None
        )

    def forward(
        self,
        x_child: torch.Tensor,
        x_parent: torch.Tensor,
        index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        num_pool: Optional[int] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x_child: Child node features of shape (Nc, Cc).
            x_parent: Parent node features of shape (Np, Cp).
            index: LongTensor of shape (Nc,) indicating parent for each child.
            edge_attr: Optional edge attributes of shape (Nc, F) for RPE.
            num_pool: Number of parent nodes. If None, inferred from x_parent.

        Returns:
            Pooled features of shape (Np, out_dim or dim).
        """
        Nc = x_child.shape[0]
        Np = x_parent.shape[0] if num_pool is None else num_pool
        H = self.num_heads
        D = self.qk_dim
        DH = D * H

        # Optional linear projection of input features
        if self.in_proj is not None:
            x_child = self.in_proj(x_child)

        # Compute queries from parent features
        q = self._get_query(x_parent)  # [Np, DH]

        # Compute keys and values from child features
        kv = self.kv(x_child)  # [Nc, DH + C]

        # Expand queries and separate keys and values
        q = q[index].view(Nc, H, D)  # [Nc, H, D]
        k = kv[:, :DH].view(Nc, H, D)  # [Nc, H, D]
        v = kv[:, DH:].view(Nc, H, -1)  # [Nc, H, C // H]

        # Apply scaling on queries
        q = q * self.qk_scale(index)

        # RPE for keys
        if self.k_rpe is not None and edge_attr is not None:
            rpe = self.k_rpe(edge_attr)
            if self.heads_share_rpe:
                rpe = rpe.repeat(1, H)
            k = k + rpe.view(Nc, H, -1)

        # RPE for queries
        if self.q_rpe is not None and edge_attr is not None:
            rpe = self.q_rpe(edge_attr)
            if self.heads_share_rpe:
                rpe = rpe.repeat(1, H)
            q = q + rpe.view(Nc, H, -1)

        # Compute compatibility scores from query-key products
        compat = torch.einsum("nhd, nhd -> nh", q, k)  # [Nc, H]

        # Compute attention scores with scaled softmax
        attn = softmax(compat, index=index, dim=0, num_nodes=Np)  # [Nc, H]

        # Optional attention dropout
        if self.attn_drop is not None:
            attn = self.attn_drop(attn)

        # Apply attention to values
        x = (v * attn.unsqueeze(-1)).view(Nc, self.dim)  # [Nc, C]
        x = scatter_sum(x, index, dim=0, dim_size=Np)  # [Np, C]

        # Optional output projection
        if self.out_proj is not None:
            x = self.out_proj(x)  # [Np, out_dim]

        # Optional output dropout
        if self.out_drop is not None:
            x = self.out_drop(x)  # [Np, C or out_dim]

        return x

    def _get_query(self, x_parent: torch.Tensor) -> torch.Tensor:
        """Compute queries from parent features.

        Subclasses must implement this method.

        Args:
            x_parent: Parent node features of shape (Np, Cp).

        Returns:
            Queries of shape (Np, D * H).
        """
        raise NotImplementedError

    def extra_repr(self) -> str:
        return f"dim={self.dim}, num_heads={self.num_heads}"


class AttentivePool(BaseAttentivePool):
    """Attentive pooling with queries computed from parent features.

    Queries are computed by projecting parent features through a linear layer.
    """

    def __init__(
        self,
        dim: Optional[int] = None,
        q_in_dim: Optional[int] = None,
        num_heads: int = 1,
        in_dim: Optional[int] = None,
        out_dim: Optional[int] = None,
        qkv_bias: bool = True,
        qk_dim: int = 8,
        qk_scale: Optional[Union[float, str]] = None,
        attn_drop: Optional[float] = None,
        drop: Optional[float] = None,
        in_rpe_dim: int = 9,
        k_rpe: bool = False,
        q_rpe: bool = False,
        v_rpe: bool = False,
        heads_share_rpe: bool = False,
    ):
        """Initialize AttentivePool.

        Args:
            q_in_dim: Input dimension for query projection.
            Other args: See BaseAttentivePool.
        """
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            in_dim=in_dim,
            out_dim=out_dim,
            qkv_bias=qkv_bias,
            qk_dim=qk_dim,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            drop=drop,
            in_rpe_dim=in_rpe_dim,
            k_rpe=k_rpe,
            q_rpe=q_rpe,
            v_rpe=v_rpe,
            heads_share_rpe=heads_share_rpe,
        )

        # Query projection from parent features
        self.q = nn.Linear(q_in_dim, qk_dim * num_heads, bias=qkv_bias)

    def _get_query(self, x_parent: torch.Tensor) -> torch.Tensor:
        """Build queries from input parent features.

        Args:
            x_parent: Parent node features of shape (Np, Cp).

        Returns:
            Queries of shape (Np, D * H).
        """
        return self.q(x_parent)


class AttentivePoolWithLearntQueries(BaseAttentivePool):
    """Attentive pooling with learnable query parameters.

    Each head learns its own query, and all parent nodes use the same queries.
    """

    def __init__(
        self,
        dim: Optional[int] = None,
        num_heads: int = 1,
        in_dim: Optional[int] = None,
        out_dim: Optional[int] = None,
        qkv_bias: bool = True,
        qk_dim: int = 8,
        qk_scale: Optional[Union[float, str]] = None,
        attn_drop: Optional[float] = None,
        drop: Optional[float] = None,
        in_rpe_dim: int = 18,
        k_rpe: bool = False,
        q_rpe: bool = False,
        v_rpe: bool = False,
        heads_share_rpe: bool = False,
    ):
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            in_dim=in_dim,
            out_dim=out_dim,
            qkv_bias=qkv_bias,
            qk_dim=qk_dim,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            drop=drop,
            in_rpe_dim=in_rpe_dim,
            k_rpe=k_rpe,
            q_rpe=q_rpe,
            v_rpe=v_rpe,
            heads_share_rpe=heads_share_rpe,
        )

        # Learnable query parameters
        self.q = nn.Parameter(torch.zeros(qk_dim * num_heads))
        nn.init.trunc_normal_(self.q, std=0.02)

    def _get_query(self, x_parent: torch.Tensor) -> torch.Tensor:
        """Build queries from learnable parameters.

        Parent features are only used to get the number of parent nodes.

        Args:
            x_parent: Parent node features of shape (Np, Cp).

        Returns:
            Queries of shape (Np, D * H).
        """
        Np = x_parent.shape[0]
        return self.q.repeat(Np, 1)
