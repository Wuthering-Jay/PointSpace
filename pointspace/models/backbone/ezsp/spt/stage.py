"""
Stage Modules for SPT.

This module provides the Stage, DownNFuseStage, and UpNFuseStage components
for the UNet-like architecture in Superpoint Transformer.

Reference: src/nn/stage.py from Superpoint Transformer

Author: PointSpace Team
"""

import torch
from torch import nn
from typing import List, Optional, Tuple, Type, Union, Dict, Any

from pointspace.models.backbone.ezsp.spt.mlp import MLP
from pointspace.models.backbone.ezsp.spt.transformer import TransformerBlock
from pointspace.models.backbone.ezsp.spt.norm import BatchNorm, UnitSphereNorm
from pointspace.models.backbone.ezsp.spt.pool import pool_factory
from pointspace.models.backbone.ezsp.spt.fusion import (
    CatFusion,
    fusion_factory,
    IndexUnpool,
)


__all__ = ["Stage", "DownNFuseStage", "UpNFuseStage", "PointStage"]


def _build_shared_rpe_encoders(
    rpe: Union[bool, nn.Module],
    num_blocks: int,
    num_heads: int,
    in_dim: int,
    out_dim: int,
    blocks_share: bool,
    heads_share: bool,
) -> List[Union[bool, nn.Module]]:
    """Build RPE encoders for Stage.

    This helper makes shared encoder construction easier. Setting blocks_share=True
    will make all blocks use the same RPE encoder.

    Args:
        rpe: Whether to use RPE, or an existing encoder module.
        num_blocks: Number of transformer blocks.
        num_heads: Number of attention heads.
        in_dim: Input dimension for RPE.
        out_dim: Output dimension for RPE (per head).
        blocks_share: Whether blocks share RPE encoders.
        heads_share: Whether heads share RPE encoders.

    Returns:
        List of RPE encoders (or bool) for each block.
    """
    if not isinstance(rpe, bool):
        assert blocks_share, (
            "If a module is passed for the RPE encoder, blocks_share should be True"
        )
        return [rpe] * num_blocks

    if not heads_share:
        out_dim = out_dim * num_heads

    if blocks_share and rpe:
        return [nn.Linear(in_dim, out_dim)] * num_blocks

    return [rpe] * num_blocks


class Stage(nn.Module):
    """A Stage has the following structure:

        x  -- PosInjection -- in_MLP -- TransformerBlock -- out_MLP -->
                    |       (optional)   (* num_blocks)   (optional)
        pos -- UnitSphereNorm
    (optional)

    Args:
        dim: Number of channels for the TransformerBlock.
        num_blocks: Number of TransformerBlocks in the Stage.
        num_heads: Number of attention heads.
        in_mlp: Channels for input MLP. Last channel must match dim.
        out_mlp: Channels for output MLP. First channel must match dim.
        mlp_activation: Activation for MLPs.
        mlp_norm: Normalization for MLPs.
        mlp_drop: Dropout rate for MLPs.
        use_pos: Whether to concatenate normalized position to features.
        use_diameter: Whether to concatenate diameter to features.
        use_diameter_parent: Whether to concatenate parent diameter.
        qk_dim: Query/key dimension.
        k_rpe, q_rpe: RPE from edge features.
        k_delta_rpe, q_delta_rpe: RPE from node feature differences.
        qk_share_rpe: Share RPE between Q and K.
        q_on_minus_rpe: Compute Q RPE on negative features.
        blocks_share_rpe: Share RPE across blocks.
        heads_share_rpe: Share RPE across heads.
        **transformer_kwargs: Additional TransformerBlock arguments.
    """

    def __init__(
        self,
        dim: int,
        num_blocks: int = 1,
        num_heads: int = 1,
        in_rpe_dim: int = 18,
        in_mlp: Optional[List[int]] = None,
        out_mlp: Optional[List[int]] = None,
        mlp_activation: Optional[nn.Module] = None,
        mlp_norm: Type[nn.Module] = BatchNorm,
        mlp_drop: Optional[float] = None,
        use_pos: bool = True,
        use_diameter: bool = False,
        use_diameter_parent: bool = False,
        qk_dim: int = 8,
        k_rpe: bool = False,
        q_rpe: bool = False,
        k_delta_rpe: bool = False,
        q_delta_rpe: bool = False,
        qk_share_rpe: bool = False,
        q_on_minus_rpe: bool = False,
        blocks_share_rpe: bool = False,
        heads_share_rpe: bool = False,
        **transformer_kwargs: Any,
    ):
        super().__init__()

        if mlp_activation is None:
            mlp_activation = nn.LeakyReLU()

        self.dim = dim
        self.num_blocks = num_blocks
        self.num_heads = num_heads

        # MLP to change input channel size
        if in_mlp is not None:
            assert in_mlp[-1] == dim, f"in_mlp[-1]={in_mlp[-1]} must match dim={dim}"
            self.in_mlp = MLP(
                in_mlp,
                activation=mlp_activation,
                norm=mlp_norm,
                drop=mlp_drop,
            )
        else:
            self.in_mlp = None

        # MLP to change output channel size
        if out_mlp is not None:
            assert out_mlp[0] == dim, f"out_mlp[0]={out_mlp[0]} must match dim={dim}"
            self.out_mlp = MLP(
                out_mlp,
                activation=mlp_activation,
                norm=mlp_norm,
                drop=mlp_drop,
            )
        else:
            self.out_mlp = None

        # Transformer blocks
        if num_blocks > 0:
            # Build shared RPE encoders if needed
            k_rpe_blocks = _build_shared_rpe_encoders(
                k_rpe, num_blocks, num_heads, in_rpe_dim, qk_dim, blocks_share_rpe, heads_share_rpe
            )
            k_delta_rpe_blocks = _build_shared_rpe_encoders(
                k_delta_rpe, num_blocks, num_heads, dim, qk_dim, blocks_share_rpe, heads_share_rpe
            )
            # Q RPE only if not sharing with K
            q_rpe_blocks = _build_shared_rpe_encoders(
                q_rpe and not (k_rpe and qk_share_rpe),
                num_blocks, num_heads, in_rpe_dim, qk_dim, blocks_share_rpe, heads_share_rpe
            )
            q_delta_rpe_blocks = _build_shared_rpe_encoders(
                q_delta_rpe and not (k_delta_rpe and qk_share_rpe),
                num_blocks, num_heads, dim, qk_dim, blocks_share_rpe, heads_share_rpe
            )

            self.transformer_blocks = nn.ModuleList(
                TransformerBlock(
                    dim,
                    num_heads=num_heads,
                    in_rpe_dim=in_rpe_dim,
                    qk_dim=qk_dim,
                    k_rpe=k_rpe_block,
                    q_rpe=q_rpe_block,
                    k_delta_rpe=k_delta_rpe_block,
                    q_delta_rpe=q_delta_rpe_block,
                    qk_share_rpe=qk_share_rpe,
                    q_on_minus_rpe=q_on_minus_rpe,
                    heads_share_rpe=heads_share_rpe,
                    **transformer_kwargs,
                )
                for k_rpe_block, q_rpe_block, k_delta_rpe_block, q_delta_rpe_block
                in zip(k_rpe_blocks, q_rpe_blocks, k_delta_rpe_blocks, q_delta_rpe_blocks)
            )
        else:
            self.transformer_blocks = None

        # UnitSphereNorm for position normalization
        self.pos_norm = UnitSphereNorm()

        # Fusion operator to combine positions with features
        self.feature_fusion = CatFusion()
        self.use_pos = use_pos
        self.use_diameter = use_diameter
        self.use_diameter_parent = use_diameter_parent

    @property
    def out_dim(self) -> int:
        """Output dimension of the stage."""
        if self.out_mlp is not None:
            return self.out_mlp.out_dim
        if self.transformer_blocks is not None:
            return self.transformer_blocks[-1].dim
        if self.in_mlp is not None:
            return self.in_mlp.out_dim
        return self.dim

    def forward(
        self,
        x: Optional[torch.Tensor],
        norm_index: torch.Tensor,
        pos: Optional[torch.Tensor] = None,
        diameter: Optional[torch.Tensor] = None,
        node_size: Optional[torch.Tensor] = None,
        super_index: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.

        Args:
            x: Node features of shape (N, C).
            norm_index: Batch indices for normalization.
            pos: Node positions of shape (N, 3).
            diameter: Node diameters of shape (N, 1).
            node_size: Node sizes (weights for centroid computation).
            super_index: Parent segment indices.
            edge_index: Edge indices of shape (2, E).
            edge_attr: Edge attributes of shape (E, F).

        Returns:
            Tuple of (output features, parent diameter).
        """
        # Recover info from input
        if x is not None:
            N = x.shape[0]
            dtype = x.dtype
            device = x.device
        elif pos is not None:
            N = pos.shape[0]
            dtype = pos.dtype
            device = pos.device
        elif diameter is not None:
            N = diameter.shape[0]
            dtype = diameter.dtype
            device = diameter.device
        elif super_index is not None:
            N = super_index.shape[0]
            dtype = edge_attr.dtype if edge_attr is not None else torch.float
            device = super_index.device
        else:
            raise ValueError("Could not infer basic info from input arguments")

        # Append normalized coordinates to node features
        if pos is not None:
            normalized_pos, diameter_parent = self.pos_norm(
                pos, super_index, w=node_size
            )
            if self.use_pos:
                x = self.feature_fusion(normalized_pos, x)
        else:
            diameter_parent = None

        # Inject parent segment diameter if needed
        if self.use_diameter:
            diam = (
                diameter
                if diameter is not None
                else torch.zeros((N, 1), dtype=dtype, device=device)
            )
            x = self.feature_fusion(diam, x)

        if self.use_diameter_parent:
            if diameter_parent is None:
                diam = torch.zeros((N, 1), dtype=dtype, device=device)
            elif super_index is None:
                diam = diameter_parent.repeat(N, 1)
            else:
                diam = diameter_parent[super_index]
            x = self.feature_fusion(diam, x)

        # Input MLP
        if self.in_mlp is not None:
            first_linear = None
            for module in self.in_mlp.mlp:
                if isinstance(module, nn.Linear):
                    first_linear = module
                    break
            if first_linear is not None and x.shape[1] != first_linear.in_features:
                expected_in = first_linear.in_features
                if x.shape[1] > expected_in:
                    x = x[:, :expected_in]
                else:
                    pad = x.new_zeros((x.shape[0], expected_in - x.shape[1]))
                    x = torch.cat([x, pad], dim=1)
            x = self.in_mlp(x, batch=norm_index)

        # Transformer blocks
        if self.transformer_blocks is not None:
            for block in self.transformer_blocks:
                x, norm_index, edge_index = block(
                    x, norm_index, edge_index=edge_index, edge_attr=edge_attr
                )

        # Output MLP
        if self.out_mlp is not None:
            x = self.out_mlp(x, batch=norm_index)

        return x, diameter_parent


class DownNFuseStage(Stage):
    """A Stage preceded by pooling and fusion for downsampling.

    Structure:
        x1 ------- Fusion -- Stage -->
                     |
        x2 -- Pool --

    Args:
        pool: Pooling mode ('max', 'min', 'mean', 'sum', 'std').
        fusion: Fusion mode ('cat', 'residual', 'first', 'second').
        Other args: See Stage.
    """

    def __init__(
        self,
        *args,
        pool: Union[str, nn.Module] = "max",
        fusion: str = "cat",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # Pooling operator
        # Accept either a pool object or a string
        if isinstance(pool, nn.Module):
            self.down_pool_block = pool
        else:
            self.down_pool_block = pool_factory(pool)

        # Fusion operator
        self.fusion = fusion_factory(fusion)

    def forward(
        self,
        x_parent: Optional[torch.Tensor],
        x_child: torch.Tensor,
        norm_index: torch.Tensor,
        pool_index: torch.Tensor,
        pos: Optional[torch.Tensor] = None,
        diameter: Optional[torch.Tensor] = None,
        node_size: Optional[torch.Tensor] = None,
        super_index: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        v_edge_attr: Optional[torch.Tensor] = None,
        num_super: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.

        Args:
            x_parent: Parent node features (from previous level).
            x_child: Child node features to pool.
            norm_index: Batch indices for normalization.
            pool_index: Indices mapping children to parents.
            pos: Node positions.
            diameter: Node diameters.
            node_size: Node sizes.
            super_index: Super-segment indices.
            edge_index: Edge indices for attention.
            edge_attr: Edge attributes for attention RPE.
            v_edge_attr: Vertical edge attributes for pooling RPE.
            num_super: Number of parent nodes.

        Returns:
            Tuple of (output features, parent diameter).
        """
        # Auto-align parent feature dim for attentive query projection.
        if (
            x_parent is not None
            and hasattr(self.down_pool_block, "q")
            and isinstance(self.down_pool_block.q, nn.Linear)
        ):
            expected_q_in = self.down_pool_block.q.in_features
            current_q_in = x_parent.shape[1]
            if current_q_in != expected_q_in:
                if current_q_in > expected_q_in:
                    x_parent = x_parent[:, :expected_q_in]
                else:
                    pad = x_parent.new_zeros((x_parent.shape[0], expected_q_in - current_q_in))
                    x_parent = torch.cat([x_parent, pad], dim=1)

        # Pool child features
        x_pooled = self.down_pool_block(
            x_child, x_parent, pool_index, edge_attr=v_edge_attr, num_pool=num_super
        )

        # Auto-align cat-fusion inputs to match in_mlp expectation.
        # This avoids brittle config coupling when handcrafted feature dims vary
        # (e.g. with use_pos / dataset-specific feature sets).
        if isinstance(self.fusion, CatFusion) and x_parent is not None and self.in_mlp is not None:
            first_linear = None
            for m in self.in_mlp.mlp:
                if isinstance(m, nn.Linear):
                    first_linear = m
                    break
            if first_linear is not None:
                expected_in = first_linear.in_features
                # Stage.forward will append extra dims before in_mlp:
                # - use_pos -> +3
                # - use_diameter -> +1
                # - use_diameter_parent -> +1
                pre_inject = 0
                if self.use_pos and pos is not None:
                    pre_inject += 3
                if self.use_diameter:
                    pre_inject += 1
                if self.use_diameter_parent:
                    pre_inject += 1
                expected_fused = max(0, expected_in - pre_inject)
                # Keep pooled branch as-is when possible; adjust parent branch first.
                expected_parent = max(0, expected_fused - x_pooled.shape[1])
                if x_parent.shape[1] != expected_parent:
                    if x_parent.shape[1] > expected_parent:
                        x_parent = x_parent[:, :expected_parent]
                    else:
                        pad = x_parent.new_zeros((x_parent.shape[0], expected_parent - x_parent.shape[1]))
                        x_parent = torch.cat([x_parent, pad], dim=1)

                # Final guard: if total still mismatched, trim/pad pooled branch too.
                total = x_parent.shape[1] + x_pooled.shape[1]
                if total != expected_fused:
                    target_pooled = max(0, expected_fused - x_parent.shape[1])
                    if x_pooled.shape[1] > target_pooled:
                        x_pooled = x_pooled[:, :target_pooled]
                    else:
                        pad = x_pooled.new_zeros((x_pooled.shape[0], target_pooled - x_pooled.shape[1]))
                        x_pooled = torch.cat([x_pooled, pad], dim=1)

        # Fuse parent and pooled child features
        x_fused = self.fusion(x_parent, x_pooled)

        # Stage forward
        return super().forward(
            x_fused,
            norm_index,
            pos=pos,
            node_size=node_size,
            super_index=super_index,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )


class UpNFuseStage(Stage):
    """A Stage preceded by unpooling and fusion for upsampling.

    Structure:
        x1 --------- Fusion -- Stage -->
                       |
        x2 -- Unpool --

    Used in the UNet-like decoder.

    Args:
        unpool: Unpooling mode (currently only 'index' supported).
        fusion: Fusion mode ('cat', 'residual', 'first', 'second').
        Other args: See Stage.
    """

    def __init__(
        self,
        *args,
        unpool: str = "index",
        fusion: str = "cat",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # Unpooling operator
        if unpool == "index":
            self.unpool = IndexUnpool()
        else:
            raise NotImplementedError(f"Unknown unpool='{unpool}' mode")

        # Fusion operator
        self.fusion = fusion_factory(fusion)

    def forward(
        self,
        x_child: Optional[torch.Tensor],
        x_parent: torch.Tensor,
        norm_index: torch.Tensor,
        unpool_index: torch.Tensor,
        pos: Optional[torch.Tensor] = None,
        diameter: Optional[torch.Tensor] = None,
        node_size: Optional[torch.Tensor] = None,
        super_index: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.

        Args:
            x_child: Child node features (from encoder skip connection).
            x_parent: Parent node features to unpool.
            norm_index: Batch indices for normalization.
            unpool_index: Indices mapping parents to children.
            pos: Node positions.
            diameter: Node diameters.
            node_size: Node sizes.
            super_index: Super-segment indices.
            edge_index: Edge indices for attention.
            edge_attr: Edge attributes for attention RPE.

        Returns:
            Tuple of (output features, parent diameter).
        """
        # Unpool parent features
        x_unpool = self.unpool(x_parent, unpool_index)

        # Fuse unpooled parent and child features
        x_fused = self.fusion(x_child, x_unpool)

        # Stage forward
        return super().forward(
            x_fused,
            norm_index,
            pos=pos,
            node_size=node_size,
            super_index=super_index,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )


class PointStage(Stage):
    """A Stage specifically designed for operating on raw points.

    Similar to the point-wise part of PointNet, operating on Level-1 segments.

    Structure (basic):
        x --> Concatenation --> in_MLP -->
                   ^
                   |
        pos --> UnitSphereNorm

    Args:
        in_mlp: Channels for the input MLP.
        mlp_activation: Activation for the MLP.
        mlp_norm: Normalization for the MLP.
        mlp_drop: Dropout rate for the MLP.
        use_pos: Whether to use normalized positions.
        use_diameter_parent: Whether to use parent diameter.
    """

    def __init__(
        self,
        in_mlp: List[int],
        mlp_activation: Optional[nn.Module] = None,
        mlp_norm: Type[nn.Module] = BatchNorm,
        mlp_drop: Optional[float] = None,
        use_pos: bool = True,
        use_diameter_parent: bool = False,
        cnn_blocks: bool = False,
        point_mlp_on_cnn_feats: bool = False,
    ):
        if mlp_activation is None:
            mlp_activation = nn.LeakyReLU()

        assert in_mlp is None or len(in_mlp) > 1, (
            "in_mlp should be a list of channels of length >= 2"
        )
        if point_mlp_on_cnn_feats and not cnn_blocks:
            raise ValueError(
                "point_mlp_on_cnn_feats=True requires cnn_blocks=True so the "
                "PointStage can treat `x` as the CNN branch."
            )

        super().__init__(
            in_mlp[-1] if in_mlp is not None else None,
            num_blocks=0,
            in_mlp=in_mlp,
            out_mlp=None,
            mlp_activation=mlp_activation,
            mlp_norm=mlp_norm,
            mlp_drop=mlp_drop,
            use_pos=use_pos,
            use_diameter=False,
            use_diameter_parent=use_diameter_parent,
        )
        self.cnn_blocks = cnn_blocks
        self.point_mlp_on_cnn_feats = point_mlp_on_cnn_feats

    def forward(
        self,
        x: Optional[torch.Tensor],
        norm_index: torch.Tensor,
        pos: Optional[torch.Tensor] = None,
        diameter: Optional[torch.Tensor] = None,
        node_size: Optional[torch.Tensor] = None,
        super_index: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        x_mlp: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.

        Args:
            x: Point features.
            norm_index: Batch indices.
            pos: Point positions.
            Other args: See Stage.forward.

        Returns:
            Tuple of (output features, parent diameter).
        """
        if self.cnn_blocks:
            if self.point_mlp_on_cnn_feats:
                x_fused = self.feature_fusion(x, x_mlp) if x_mlp is not None else x
                return super().forward(
                    x_fused,
                    norm_index,
                    pos,
                    diameter,
                    node_size,
                    super_index,
                    edge_index,
                    edge_attr,
                )

            if x_mlp is None and self.in_mlp is not None:
                raise ValueError(
                    "PointStage needs handcrafted features in `x_mlp` when "
                    "point_mlp_on_cnn_feats=False and an input MLP is configured."
                )

            x_mlp_out, diameter_parent = super().forward(
                x_mlp,
                norm_index,
                pos,
                diameter,
                node_size,
                super_index,
                edge_index,
                edge_attr,
            )
            x_out = self.feature_fusion(x, x_mlp_out) if x is not None else x_mlp_out
            return x_out, diameter_parent

        return super().forward(
            x, norm_index, pos, diameter, node_size, super_index, edge_index, edge_attr
        )
