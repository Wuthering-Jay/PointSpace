"""
Self-Attention Block for SPT.

This module provides the multi-head self-attention mechanism with
relative positional encoding (RPE) support for the Superpoint Transformer.

Reference: src/nn/attention.py from Superpoint Transformer

Author: PointSpace Team
"""

import torch
from torch import nn
from torch_scatter import scatter_sum
from torch_geometric.utils import softmax
from typing import Optional, Union

from pointspace.models.backbone.ezsp.spt.utils import build_qk_scale_func


__all__ = ["SelfAttentionBlock"]


class SelfAttentionBlock(nn.Module):
    """Self-attention block for graph-based transformers.

    Implements multi-head self-attention with optional relative positional
    encodings (RPE) computed from edge features and/or node feature differences.

    This block is designed to be used in a residual fashion within TransformerBlock.

    Inspired by: https://github.com/microsoft/Swin-Transformer

    Args:
        dim: Dimension of the feature space on which the attention operates.
        num_heads: Number of attention heads.
        in_dim: Input feature dimension. If specified, features are projected
            from in_dim to dim with a linear layer.
        out_dim: Output feature dimension. If specified, features are projected
            from dim to out_dim with a linear layer.
        qkv_bias: Whether Q, K, V linear layers should have bias.
        qk_dim: Dimension of queries and keys per head.
        qk_scale: Scaling mode for query-key product. See build_qk_scale_func.
        attn_drop: Dropout probability on attention weights.
        drop: Dropout probability on output features.
        in_rpe_dim: Dimension of edge features for RPE computation.
        k_rpe: Whether to apply RPE to keys from edge features.
        q_rpe: Whether to apply RPE to queries from edge features.
        v_rpe: Whether to apply RPE to values from edge features.
        k_delta_rpe: Whether to apply RPE to keys from node feature differences.
        q_delta_rpe: Whether to apply RPE to queries from node feature differences.
        qk_share_rpe: Whether Q and K share RPE parameters.
        q_on_minus_rpe: Whether to compute Q RPE on negative features.
        heads_share_rpe: Whether heads share RPE parameters.
    """

    def __init__(
        self,
        dim: int,
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
        k_delta_rpe: bool = False,
        q_delta_rpe: bool = False,
        qk_share_rpe: bool = False,
        q_on_minus_rpe: bool = False,
        heads_share_rpe: bool = False,
    ):
        """Initialize SelfAttentionBlock."""
        super().__init__()

        assert dim % num_heads == 0, "dim must be a multiple of num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.qk_dim = qk_dim
        self.qk_scale = build_qk_scale_func(dim, num_heads, qk_scale)
        self.heads_share_rpe = heads_share_rpe

        # QKV projection: [Q: qk_dim * num_heads, K: qk_dim * num_heads, V: dim]
        self.qkv = nn.Linear(dim, qk_dim * 2 * num_heads + dim, bias=qkv_bias)

        # Build RPE encoders, with optional weight sharing across heads
        qk_rpe_dim = qk_dim if heads_share_rpe else qk_dim * num_heads
        v_rpe_dim = dim // num_heads if heads_share_rpe else dim

        # RPE for keys from edge features
        if not isinstance(k_rpe, bool):
            self.k_rpe = k_rpe
        else:
            self.k_rpe = nn.Linear(in_rpe_dim, qk_rpe_dim) if k_rpe else None

        # RPE for queries from edge features
        if not isinstance(q_rpe, bool):
            self.q_rpe = q_rpe
        else:
            self.q_rpe = (
                nn.Linear(in_rpe_dim, qk_rpe_dim)
                if q_rpe and not (k_rpe and qk_share_rpe)
                else None
            )

        # RPE for keys from node feature differences
        if not isinstance(k_delta_rpe, bool):
            self.k_delta_rpe = k_delta_rpe
        else:
            self.k_delta_rpe = nn.Linear(dim, qk_rpe_dim) if k_delta_rpe else None

        # RPE for queries from node feature differences
        if not isinstance(q_delta_rpe, bool):
            self.q_delta_rpe = q_delta_rpe
        else:
            self.q_delta_rpe = (
                nn.Linear(dim, qk_rpe_dim)
                if q_delta_rpe and not (k_delta_rpe and qk_share_rpe)
                else None
            )

        self.qk_share_rpe = qk_share_rpe
        self.q_on_minus_rpe = q_on_minus_rpe

        # RPE for values from edge features
        if not isinstance(v_rpe, bool):
            self.v_rpe = v_rpe
        else:
            self.v_rpe = nn.Linear(in_rpe_dim, v_rpe_dim) if v_rpe else None

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
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Node features of shape (N, Cx).
            edge_index: Edge indices of shape (2, E). Source indicates the
                querying element, target indicates key elements.
            edge_attr: Optional edge attributes of shape (E, Ce) for RPE.

        Returns:
            Updated node features of shape (N, out_dim or dim).
        """
        N = x.shape[0]
        E = edge_index.shape[1]
        H = self.num_heads
        D = self.qk_dim
        DH = D * H

        # Optional linear projection of input features
        if self.in_proj is not None:
            x = self.in_proj(x)

        # Compute queries, keys and values
        qkv = self.qkv(x)

        # Separate Q, K, V
        q = qkv[:, :DH].view(N, H, D)  # [N, H, D]
        k = qkv[:, DH : 2 * DH].view(N, H, D)  # [N, H, D]
        v = qkv[:, 2 * DH :].view(N, H, -1)  # [N, H, dim // H]

        # Expand Q, K, V to edges
        s = edge_index[0]  # source (query) [E]
        t = edge_index[1]  # target (key) [E]
        q = q[s]  # [E, H, D]
        k = k[t]  # [E, H, D]
        v = v[t]  # [E, H, dim // H]

        # Apply scaling on queries
        q = q * self.qk_scale(s)

        # RPE from edge features for keys
        if self.k_rpe is not None and edge_attr is not None:
            rpe = self.k_rpe(edge_attr)
            if self.heads_share_rpe:
                rpe = rpe.repeat(1, H)
            k = k + rpe.view(E, H, -1)

        # RPE from edge features for queries
        if self.q_rpe is not None and edge_attr is not None:
            if self.q_on_minus_rpe:
                rpe = self.q_rpe(-edge_attr)
            else:
                rpe = self.q_rpe(edge_attr)
            if self.heads_share_rpe:
                rpe = rpe.repeat(1, H)
            q = q + rpe.view(E, H, -1)
        elif self.k_rpe is not None and self.qk_share_rpe and edge_attr is not None:
            if self.q_on_minus_rpe:
                rpe = self.k_rpe(-edge_attr)
            else:
                rpe = self.k_rpe(edge_attr)
            if self.heads_share_rpe:
                rpe = rpe.repeat(1, H)
            q = q + rpe.view(E, H, -1)

        # RPE from node delta features for keys
        if self.k_delta_rpe is not None:
            rpe = self.k_delta_rpe(x[edge_index[1]] - x[edge_index[0]])
            if self.heads_share_rpe:
                rpe = rpe.repeat(1, H)
            k = k + rpe.view(E, H, -1)

        # RPE from node delta features for queries
        if self.q_delta_rpe is not None:
            if self.q_on_minus_rpe:
                rpe = self.q_delta_rpe(x[edge_index[0]] - x[edge_index[1]])
            else:
                rpe = self.q_delta_rpe(x[edge_index[1]] - x[edge_index[0]])
            if self.heads_share_rpe:
                rpe = rpe.repeat(1, H)
            q = q + rpe.view(E, H, -1)
        elif (
            self.k_delta_rpe is not None
            and self.qk_share_rpe
            and edge_attr is not None
        ):
            if self.q_on_minus_rpe:
                rpe = self.k_delta_rpe(x[edge_index[0]] - x[edge_index[1]])
            else:
                rpe = self.k_delta_rpe(x[edge_index[1]] - x[edge_index[0]])
            if self.heads_share_rpe:
                rpe = rpe.repeat(1, H)
            q = q + rpe.view(E, H, -1)

        # RPE from edge features for values
        if self.v_rpe is not None and edge_attr is not None:
            rpe = self.v_rpe(edge_attr)
            if self.heads_share_rpe:
                rpe = rpe.repeat(1, H)
            v = v + rpe.view(E, H, -1)

        # Compute compatibility scores from query-key products
        compat = torch.einsum("ehd, ehd -> eh", q, k)  # [E, H]

        # Compute attention scores with scaled softmax
        attn = softmax(compat, index=s, dim=0, num_nodes=N)  # [E, H]

        # Optional attention dropout
        if self.attn_drop is not None:
            attn = self.attn_drop(attn)

        # Apply attention to values
        x = (v * attn.unsqueeze(-1)).view(E, self.dim)  # [E, dim]
        x = scatter_sum(x, s, dim=0, dim_size=N)  # [N, dim]

        # Optional output projection
        if self.out_proj is not None:
            x = self.out_proj(x)  # [N, out_dim]

        # Optional output dropout
        if self.out_drop is not None:
            x = self.out_drop(x)  # [N, dim or out_dim]

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, num_heads={self.num_heads}"
