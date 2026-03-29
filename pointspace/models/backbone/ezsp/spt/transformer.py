"""
Transformer Block for SPT.

This module provides the pre-norm residual transformer block used in
the Superpoint Transformer architecture.

Reference: src/nn/transformer.py from Superpoint Transformer

Author: PointSpace Team
"""

import torch
from torch import nn
from typing import Optional, Tuple, Type, Union
from torch_geometric.nn.norm import LayerNorm

from pointspace.models.backbone.ezsp.spt.attention import SelfAttentionBlock
from pointspace.models.backbone.ezsp.spt.mlp import FFN
from pointspace.models.backbone.ezsp.spt.dropout import DropPath
from pointspace.models.backbone.ezsp.spt.norm import INDEX_BASED_NORMS


__all__ = ["TransformerBlock"]


class TransformerBlock(nn.Module):
    """Base block of the Transformer architecture with pre-norm residual connections.

    Architecture diagram:
        x ---------------- + ---------------- + -->
            \\             |   \\              |
             -- N -- SA --     -- N -- FFN --

    Where:
        - N: Normalization (pre-norm style)
        - SA: Self-Attention
        - FFN: Feed-Forward Network

    Inspired by: https://github.com/microsoft/Swin-Transformer

    Args:
        dim: Feature dimension.
        num_heads: Number of attention heads.
        qkv_bias: Whether Q, K, V projections have bias.
        qk_dim: Query/key dimension per head.
        qk_scale: Query-key scaling mode.
        in_rpe_dim: Input dimension for RPE.
        ffn_ratio: FFN hidden dimension = dim * ffn_ratio.
        attn_drop: Dropout on attention weights.
        residual_drop: Dropout on SA and FFN outputs.
        drop_path: Stochastic depth probability.
        activation: Activation function for FFN.
        norm: Normalization layer class.
        pre_norm: Whether to use pre-norm (True) or post-norm (False).
        no_sa: Disable self-attention branch.
        no_ffn: Disable FFN branch.
        k_rpe, q_rpe, v_rpe: RPE from edge features.
        k_delta_rpe, q_delta_rpe: RPE from node feature differences.
        qk_share_rpe: Share RPE parameters between Q and K.
        q_on_minus_rpe: Compute Q RPE on negative features.
        heads_share_rpe: Share RPE parameters across heads.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 1,
        qkv_bias: bool = True,
        qk_dim: int = 8,
        qk_scale: Optional[Union[float, str]] = None,
        in_rpe_dim: int = 18,
        ffn_ratio: int = 4,
        attn_drop: Optional[float] = None,
        residual_drop: Optional[float] = None,
        drop_path: Optional[float] = None,
        activation: Optional[nn.Module] = None,
        norm: Type[nn.Module] = LayerNorm,
        pre_norm: bool = True,
        no_sa: bool = False,
        no_ffn: bool = False,
        k_rpe: bool = False,
        q_rpe: bool = False,
        v_rpe: bool = False,
        k_delta_rpe: bool = False,
        q_delta_rpe: bool = False,
        qk_share_rpe: bool = False,
        q_on_minus_rpe: bool = False,
        heads_share_rpe: bool = False,
    ):
        """Initialize TransformerBlock."""
        super().__init__()

        if activation is None:
            activation = nn.LeakyReLU()

        self.dim = dim
        self.pre_norm = pre_norm

        # Self-Attention residual branch
        self.no_sa = no_sa
        if not no_sa:
            self.sa_norm = norm(dim)
            self.sa = SelfAttentionBlock(
                dim,
                num_heads=num_heads,
                in_dim=None,
                out_dim=dim,
                qkv_bias=qkv_bias,
                qk_dim=qk_dim,
                qk_scale=qk_scale,
                in_rpe_dim=in_rpe_dim,
                attn_drop=attn_drop,
                drop=residual_drop,
                k_rpe=k_rpe,
                q_rpe=q_rpe,
                v_rpe=v_rpe,
                k_delta_rpe=k_delta_rpe,
                q_delta_rpe=q_delta_rpe,
                qk_share_rpe=qk_share_rpe,
                q_on_minus_rpe=q_on_minus_rpe,
                heads_share_rpe=heads_share_rpe,
            )

        # Feed-Forward Network residual branch
        self.no_ffn = no_ffn
        if not no_ffn:
            self.ffn_norm = norm(dim)
            self.ffn_ratio = ffn_ratio
            self.ffn = FFN(
                dim,
                hidden_dim=int(dim * ffn_ratio),
                activation=activation,
                drop=residual_drop,
            )

        # Optional DropPath for stochastic depth
        self.drop_path = (
            DropPath(drop_path)
            if drop_path is not None and drop_path > 0
            else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
        norm_index: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.

        Args:
            x: Node features of shape (N, dim).
            norm_index: Node indices for LayerNorm of shape (N,).
            edge_index: Edge indices of shape (2, E) for self-attention.
            edge_attr: Edge attributes of shape (E, F) for RPE.

        Returns:
            Tuple of (output features, norm_index, edge_index).
        """
        assert x.dim() == 2, "x should be a 2D Tensor"
        assert x.is_floating_point(), "x should be a FloatTensor"
        assert norm_index.dim() == 1 and norm_index.shape[0] == x.shape[0], (
            "norm_index should be a 1D LongTensor with same length as x"
        )
        assert edge_index is None or (
            edge_index.dim() == 2 and not edge_index.is_floating_point()
        ), "edge_index should be a 2D LongTensor"
        assert edge_attr is None or (
            edge_attr.dim() == 2 and edge_attr.shape[0] == edge_index.shape[1]
        ), "edge_attr should be a 2D Tensor with E rows"

        # Keep track of x for residual connection
        shortcut = x

        # Self-Attention residual branch (skip if no edges)
        if self.no_sa or edge_index is None or edge_index.shape[1] == 0:
            pass
        elif self.pre_norm:
            x = self._forward_norm(self.sa_norm, x, norm_index)
            x = self.sa(x, edge_index, edge_attr=edge_attr)
            x = shortcut + self.drop_path(x)
        else:
            x = self.sa(x, edge_index, edge_attr=edge_attr)
            x = self.drop_path(x)
            x = self._forward_norm(self.sa_norm, shortcut + x, norm_index)

        # Update shortcut for FFN residual (version >= 2.2.0 behavior)
        shortcut = x

        # Feed-Forward Network residual branch
        if not self.no_ffn and self.pre_norm:
            x = self._forward_norm(self.ffn_norm, x, norm_index)
            x = self.ffn(x)
            x = shortcut + self.drop_path(x)
        elif not self.no_ffn and not self.pre_norm:
            x = self.ffn(x)
            x = self.drop_path(x)
            x = self._forward_norm(self.ffn_norm, shortcut + x, norm_index)

        return x, norm_index, edge_index

    @staticmethod
    def _forward_norm(
        norm: nn.Module,
        x: torch.Tensor,
        norm_index: torch.Tensor,
    ) -> torch.Tensor:
        """Helper for forward pass on norm modules.

        Some modules require an index, while others don't.
        """
        if isinstance(norm, INDEX_BASED_NORMS):
            return norm(x, batch=norm_index)
        return norm(x)
