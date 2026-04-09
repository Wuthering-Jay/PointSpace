"""
Superpoint Transformer (SPT) Network for EZ-SP.

This module provides the main SPT network - a UNet-like architecture
that processes superpoint hierarchies for semantic segmentation.

Reference: src/models/components/spt.py from Superpoint Transformer

Author: PointSpace Team
"""

import torch
from torch import nn
from typing import List, Optional, Tuple, Type, Union, Any, Dict
from torch_geometric.nn.norm import LayerNorm

from pointspace.models.backbone.ezsp.spt.mlp import MLP
from pointspace.models.backbone.ezsp.spt.norm import BatchNorm
from pointspace.models.backbone.ezsp.spt.pool import pool_factory, BaseAttentivePool
from pointspace.models.backbone.ezsp.spt.fusion import CatFusion
from pointspace.models.backbone.ezsp.spt.stage import (
    Stage,
    DownNFuseStage,
    UpNFuseStage,
    PointStage,
)


__all__ = ["SPT"]


def listify_with_reference(*args):
    """Convert arguments to lists with consistent length.

    Uses the first non-None argument as reference for length.
    """
    # Find reference length
    ref_len = None
    for arg in args:
        if isinstance(arg, (list, tuple)) and arg is not None:
            ref_len = len(arg)
            break

    if ref_len is None:
        return args

    # Convert all to lists of same length
    result = []
    for arg in args:
        if arg is None:
            result.append([None] * ref_len)
        elif isinstance(arg, (list, tuple)):
            result.append(list(arg))
        else:
            result.append([arg] * ref_len)

    return result


def _build_mlps(
    mlp_channels: Optional[List[int]],
    num_mlps: int,
    activation: nn.Module,
    norm: Type[nn.Module],
    share: bool,
) -> Optional[nn.ModuleList]:
    """Build MLPs for handcrafted feature processing.

    Args:
        mlp_channels: Channel sizes for MLPs.
        num_mlps: Number of MLPs to create.
        activation: Activation function.
        norm: Normalization layer class.
        share: Whether to share parameters across MLPs.

    Returns:
        ModuleList of MLPs, or None if mlp_channels is None.
    """
    if mlp_channels is None or len(mlp_channels) < 2:
        return None

    if share:
        mlp = MLP(mlp_channels, activation=activation, norm=norm)
        return nn.ModuleList([mlp] * num_mlps)
    else:
        return nn.ModuleList([
            MLP(mlp_channels, activation=activation, norm=norm)
            for _ in range(num_mlps)
        ])


def _build_shared_rpe_encoders(
    rpe: bool,
    num_stages: int,
    in_dim: int,
    out_dim: int,
    stages_share: bool,
) -> List[Union[bool, nn.Module]]:
    """Build RPE encoders for SPT stages.

    Args:
        rpe: Whether to use RPE.
        num_stages: Number of stages.
        in_dim: Input dimension.
        out_dim: Output dimension.
        stages_share: Whether stages share RPE encoders.

    Returns:
        List of RPE encoders (or bool) for each stage.
    """
    if not rpe:
        return [False] * num_stages
    if stages_share:
        encoder = nn.Linear(in_dim, out_dim)
        return [encoder] * num_stages
    return [True] * num_stages


class SPT(nn.Module):
    """Superpoint Transformer - A UNet-like architecture for superpoint processing.

    Architecture:
        p_0, x_0 --------- PointStage
                               \\
        p_1, x_1, e_1 -- DownNFuseStage_1 ------- UpNFuseStage_1 --> out_1
                                \\                       |
        p_2, x_2, e_2 -- DownNFuseStage_2 ------- UpNFuseStage_2 --> out_2
                                \\                       |
                               ...                     ...

    Where:
        - p_i: Node positions at level i
        - x_i: Node features at level i
        - e_i: Edge features at level i
        - out_i: Output features at level i

    Args:
        point_mlp: Channels for PointStage input MLP.
        point_drop: Dropout for PointStage.
        nano: If True, skip PointStage and only use superpoints.

        down_dim: Feature dimension for each DownNFuseStage.
        down_in_mlp: Input MLP channels for each DownNFuseStage.
        down_out_mlp: Output MLP channels for each DownNFuseStage.
        down_mlp_drop: Dropout for DownNFuseStage MLPs.
        down_num_heads: Attention heads for each DownNFuseStage.
        down_num_blocks: Transformer blocks for each DownNFuseStage.
        down_ffn_ratio: FFN expansion ratio for each DownNFuseStage.
        down_residual_drop: Dropout on attention output.
        down_attn_drop: Dropout on attention weights.
        down_drop_path: Stochastic depth probability.

        up_dim: Feature dimension for each UpNFuseStage.
        up_in_mlp, up_out_mlp, up_mlp_drop, up_num_heads, up_num_blocks,
        up_ffn_ratio, up_residual_drop, up_attn_drop, up_drop_path: Same as down.

        node_mlp: Channels for node feature MLPs.
        h_edge_mlp: Channels for horizontal edge MLPs.
        v_edge_mlp: Channels for vertical edge MLPs.
        mlp_activation: Activation for MLPs.
        mlp_norm: Normalization for MLPs.

        qk_dim: Query/key dimension.
        qkv_bias: Bias in QKV projections.
        qk_scale: Query-key scaling mode.
        in_rpe_dim: Input dimension for RPE.
        activation: Activation for FFN.
        norm: Normalization for transformer.
        pre_norm: Use pre-norm residual.
        no_sa: Disable self-attention.
        no_ffn: Disable FFN.
        k_rpe, q_rpe, v_rpe: RPE from edge features.
        k_delta_rpe, q_delta_rpe: RPE from node feature differences.
        qk_share_rpe: Share RPE between Q and K.
        q_on_minus_rpe: Compute Q RPE on negative features.

        share_hf_mlps: Share handcrafted feature MLPs across stages.
        stages_share_rpe: Share RPE across stages.
        blocks_share_rpe: Share RPE across blocks.
        heads_share_rpe: Share RPE across heads.

        use_pos: Use normalized positions.
        use_node_hf: Use handcrafted node features.
        use_diameter: Use node diameter.
        use_diameter_parent: Use parent diameter.
        pool: Pooling mode for DownNFuseStage.
        unpool: Unpooling mode for UpNFuseStage.
        fusion: Fusion mode.
        norm_mode: Normalization mode ('graph', 'node', 'segment').
        output_stage_wise: Return features for all levels.
    """

    def __init__(
        self,
        # PointStage params
        point_mlp: Optional[List[int]] = None,
        point_drop: Optional[float] = None,
        nano: bool = False,
        point_cnn_blocks: bool = False,
        point_mlp_on_cnn_feats: bool = False,
        # Down stage params
        down_dim: Optional[List[int]] = None,
        down_pool_dim: Optional[List[int]] = None,
        down_in_mlp: Optional[List[List[int]]] = None,
        down_out_mlp: Optional[List[List[int]]] = None,
        down_mlp_drop: Optional[List[float]] = None,
        down_num_heads: Union[int, List[int]] = 1,
        down_num_blocks: Union[int, List[int]] = 0,
        down_ffn_ratio: Union[float, List[float]] = 4,
        down_residual_drop: Optional[List[float]] = None,
        down_attn_drop: Optional[List[float]] = None,
        down_drop_path: Optional[List[float]] = None,
        # Up stage params
        up_dim: Optional[List[int]] = None,
        up_in_mlp: Optional[List[List[int]]] = None,
        up_out_mlp: Optional[List[List[int]]] = None,
        up_mlp_drop: Optional[List[float]] = None,
        up_num_heads: Union[int, List[int]] = 1,
        up_num_blocks: Union[int, List[int]] = 0,
        up_ffn_ratio: Union[float, List[float]] = 4,
        up_residual_drop: Optional[List[float]] = None,
        up_attn_drop: Optional[List[float]] = None,
        up_drop_path: Optional[List[float]] = None,
        # Handcrafted feature MLPs
        node_mlp: Optional[List[int]] = None,
        h_edge_mlp: Optional[List[int]] = None,
        v_edge_mlp: Optional[List[int]] = None,
        mlp_activation: Optional[nn.Module] = None,
        mlp_norm: Type[nn.Module] = BatchNorm,
        # Transformer params
        qk_dim: int = 8,
        qkv_bias: bool = True,
        qk_scale: Optional[str] = None,
        in_rpe_dim: int = 18,
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
        # Sharing params
        share_hf_mlps: bool = False,
        stages_share_rpe: bool = False,
        blocks_share_rpe: bool = False,
        heads_share_rpe: bool = False,
        # Feature usage
        use_pos: bool = True,
        use_node_hf: bool = True,
        use_diameter: bool = False,
        use_diameter_parent: bool = False,
        # Pool/unpool/fusion
        pool: Union[str, List[str]] = "max",
        unpool: str = "index",
        fusion: str = "cat",
        # Output
        norm_mode: str = "graph",
        output_stage_wise: bool = False,
        # Segmentation head
        num_classes: Optional[int] = None,
        add_seg_head: bool = False,
    ):
        super().__init__()

        if mlp_activation is None:
            mlp_activation = nn.LeakyReLU()
        if activation is None:
            activation = nn.LeakyReLU()

        self.nano = nano
        self.use_pos = use_pos
        self.use_node_hf = use_node_hf
        self.use_diameter = use_diameter
        self.use_diameter_parent = use_diameter_parent
        self.norm_mode = norm_mode
        self.stages_share_rpe = stages_share_rpe
        self.blocks_share_rpe = blocks_share_rpe
        self.heads_share_rpe = heads_share_rpe
        self.output_stage_wise = output_stage_wise

        # Convert to lists
        (
            down_dim,
            down_pool_dim,
            down_in_mlp,
            down_out_mlp,
            down_mlp_drop,
            down_num_heads,
            down_num_blocks,
            down_ffn_ratio,
            down_residual_drop,
            down_attn_drop,
            down_drop_path,
            pool,
        ) = listify_with_reference(
            down_dim,
            down_pool_dim,
            down_in_mlp,
            down_out_mlp,
            down_mlp_drop,
            down_num_heads,
            down_num_blocks,
            down_ffn_ratio,
            down_residual_drop,
            down_attn_drop,
            down_drop_path,
            pool,
        )

        (
            up_dim,
            up_in_mlp,
            up_out_mlp,
            up_mlp_drop,
            up_num_heads,
            up_num_blocks,
            up_ffn_ratio,
            up_residual_drop,
            up_attn_drop,
            up_drop_path,
        ) = listify_with_reference(
            up_dim,
            up_in_mlp,
            up_out_mlp,
            up_mlp_drop,
            up_num_heads,
            up_num_blocks,
            up_ffn_ratio,
            up_residual_drop,
            up_attn_drop,
            up_drop_path,
        )

        # Architecture parameters
        num_down = len(down_dim) - int(nano) if down_dim else 0
        num_up = len(up_dim) if up_dim else 0
        needs_h_edge_hf = any(x > 0 for x in (down_num_blocks or []) + (up_num_blocks or []))
        
        # Pre-build pool objects with correct dimensions
        # Note: pool list has len(down_dim) elements, but we only use elements from start_idx onward
        pool_objects = [None] * len(pool)  # Initialize with None
        if num_down > 0 and down_pool_dim:
            start_idx = int(nano)
            for i in range(start_idx, len(pool)):
                # Map pool index to down_pool_dim index
                # For nano=True: i=1,2 maps to down_pool_dim[0,1]
                # For nano=False: i=0,1,2,... maps to down_pool_dim[0,1,2,...]
                pool_dim_idx = i - start_idx
                if pool_dim_idx < len(down_pool_dim):
                    pool_dim = down_pool_dim[pool_dim_idx]
                    
                    # Determine child feature dimension for pooling
                    # For first DownNFuseStage (i=start_idx): x_child comes from first_stage
                    if i == start_idx:
                        child_dim = down_dim[0] if nano else down_dim[0]
                    else:
                        child_dim = down_dim[i - 1]

                    # Create pool with explicit attentive dimensions:
                    # - dim / output dim follows child stream
                    # - q_in_dim is inferred lazily from x_parent at runtime to
                    #   tolerate dataset/use_pos-driven handcrafted feature changes
                    if isinstance(pool[i], str) and pool[i] == "attentive":
                        pool_objects[i] = pool_factory(
                            pool[i],
                            child_dim,
                            in_dim=child_dim,
                        )
                    else:
                        pool_objects[i] = pool_factory(pool[i], pool_dim)
                else:
                    # Fallback if down_pool_dim is shorter than expected
                    pool_objects[i] = pool_factory(pool[i])
            needs_v_edge_hf = pool_objects[start_idx] is not None and isinstance(
                pool_objects[start_idx], BaseAttentivePool
            )
        else:
            # Fallback: create pool objects without dimensions
            pool_objects = [pool_factory(p) for p in pool]
            needs_v_edge_hf = False

        # Build handcrafted feature MLPs
        node_mlp_channels = node_mlp if use_node_hf else None
        self.node_mlps = _build_mlps(
            node_mlp_channels,
            num_down + int(nano),
            mlp_activation,
            mlp_norm,
            share_hf_mlps,
        )

        h_edge_mlp_channels = h_edge_mlp if needs_h_edge_hf else None
        self.h_edge_mlps = _build_mlps(
            h_edge_mlp_channels,
            num_down + int(nano),
            mlp_activation,
            mlp_norm,
            share_hf_mlps,
        )

        v_edge_mlp_channels = v_edge_mlp if needs_v_edge_hf else None
        self.v_edge_mlps = _build_mlps(
            v_edge_mlp_channels,
            num_down,
            mlp_activation,
            mlp_norm,
            share_hf_mlps,
        )

        # First stage (PointStage or nano Stage)
        if nano:
            self.first_stage = Stage(
                down_dim[0],
                num_blocks=down_num_blocks[0],
                in_mlp=down_in_mlp[0],
                out_mlp=down_out_mlp[0],
                mlp_activation=mlp_activation,
                mlp_norm=mlp_norm,
                mlp_drop=down_mlp_drop[0],
                num_heads=down_num_heads[0],
                qk_dim=qk_dim,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                in_rpe_dim=in_rpe_dim,
                ffn_ratio=down_ffn_ratio[0],
                residual_drop=down_residual_drop[0],
                attn_drop=down_attn_drop[0],
                drop_path=down_drop_path[0],
                activation=activation,
                norm=norm,
                pre_norm=pre_norm,
                no_sa=no_sa,
                no_ffn=no_ffn,
                k_rpe=k_rpe,
                q_rpe=q_rpe,
                v_rpe=v_rpe,
                k_delta_rpe=k_delta_rpe,
                q_delta_rpe=q_delta_rpe,
                qk_share_rpe=qk_share_rpe,
                q_on_minus_rpe=q_on_minus_rpe,
                use_pos=use_pos,
                use_diameter=use_diameter,
                use_diameter_parent=use_diameter_parent,
                blocks_share_rpe=blocks_share_rpe,
                heads_share_rpe=heads_share_rpe,
            )
        else:
            self.first_stage = PointStage(
                point_mlp,
                mlp_activation=mlp_activation,
                mlp_norm=mlp_norm,
                mlp_drop=point_drop,
                use_pos=use_pos,
                use_diameter_parent=use_diameter_parent,
                cnn_blocks=point_cnn_blocks,
                point_mlp_on_cnn_feats=point_mlp_on_cnn_feats,
            )

        # Feature fusion operator
        self.feature_fusion = CatFusion()

        # Down stages
        if num_down > 0:
            down_k_rpe = _build_shared_rpe_encoders(
                k_rpe, num_down, in_rpe_dim, qk_dim, stages_share_rpe
            )
            down_q_rpe = _build_shared_rpe_encoders(
                q_rpe and not (k_rpe and qk_share_rpe),
                num_down, in_rpe_dim, qk_dim, stages_share_rpe,
            )

            if nano:
                down_k_rpe = [None] + down_k_rpe
                down_q_rpe = [None] + down_q_rpe

            self.down_stages = nn.ModuleList()
            start_idx = int(nano)
            for i_down in range(start_idx, len(down_dim)):
                self.down_stages.append(
                    DownNFuseStage(
                        down_dim[i_down],
                        num_blocks=down_num_blocks[i_down],
                        in_mlp=down_in_mlp[i_down],
                        out_mlp=down_out_mlp[i_down],
                        mlp_activation=mlp_activation,
                        mlp_norm=mlp_norm,
                        mlp_drop=down_mlp_drop[i_down],
                        num_heads=down_num_heads[i_down],
                        qk_dim=qk_dim,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        in_rpe_dim=in_rpe_dim,
                        ffn_ratio=down_ffn_ratio[i_down],
                        residual_drop=down_residual_drop[i_down],
                        attn_drop=down_attn_drop[i_down],
                        drop_path=down_drop_path[i_down],
                        activation=activation,
                        norm=norm,
                        pre_norm=pre_norm,
                        no_sa=no_sa,
                        no_ffn=no_ffn,
                        k_rpe=down_k_rpe[i_down],
                        q_rpe=down_q_rpe[i_down],
                        v_rpe=v_rpe,
                        k_delta_rpe=k_delta_rpe,
                        q_delta_rpe=q_delta_rpe,
                        qk_share_rpe=qk_share_rpe,
                        q_on_minus_rpe=q_on_minus_rpe,
                        pool=pool_objects[i_down],  # Use pre-built pool object
                        fusion=fusion,
                        use_pos=use_pos,
                        use_diameter=use_diameter,
                        use_diameter_parent=use_diameter_parent,
                        blocks_share_rpe=blocks_share_rpe,
                        heads_share_rpe=heads_share_rpe,
                    )
                )
        else:
            self.down_stages = None

        # Up stages
        if num_up > 0:
            up_k_rpe = _build_shared_rpe_encoders(
                k_rpe, num_up, in_rpe_dim, qk_dim, stages_share_rpe
            )
            up_q_rpe = _build_shared_rpe_encoders(
                q_rpe and not (k_rpe and qk_share_rpe),
                num_up, in_rpe_dim, qk_dim, stages_share_rpe,
            )

            self.up_stages = nn.ModuleList()
            for i_up in range(num_up):
                self.up_stages.append(
                    UpNFuseStage(
                        up_dim[i_up],
                        num_blocks=up_num_blocks[i_up],
                        in_mlp=up_in_mlp[i_up],
                        out_mlp=up_out_mlp[i_up],
                        mlp_activation=mlp_activation,
                        mlp_norm=mlp_norm,
                        mlp_drop=up_mlp_drop[i_up],
                        num_heads=up_num_heads[i_up],
                        qk_dim=qk_dim,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        in_rpe_dim=in_rpe_dim,
                        ffn_ratio=up_ffn_ratio[i_up],
                        residual_drop=up_residual_drop[i_up],
                        attn_drop=up_attn_drop[i_up],
                        drop_path=up_drop_path[i_up],
                        activation=activation,
                        norm=norm,
                        pre_norm=pre_norm,
                        no_sa=no_sa,
                        no_ffn=no_ffn,
                        k_rpe=up_k_rpe[i_up],
                        q_rpe=up_q_rpe[i_up],
                        v_rpe=v_rpe,
                        k_delta_rpe=k_delta_rpe,
                        q_delta_rpe=q_delta_rpe,
                        qk_share_rpe=qk_share_rpe,
                        q_on_minus_rpe=q_on_minus_rpe,
                        unpool=unpool,
                        fusion=fusion,
                        use_pos=use_pos,
                        use_diameter=use_diameter,
                        use_diameter_parent=use_diameter_parent,
                        blocks_share_rpe=blocks_share_rpe,
                        heads_share_rpe=heads_share_rpe,
                    )
                )
        else:
            self.up_stages = None
        
        # Segmentation head (optional)
        if add_seg_head and num_classes is not None:
            out_channels = self.out_dim
            if isinstance(out_channels, list):
                # Multi-stage output: create head for each level
                self.seg_head = nn.ModuleList([
                    nn.Linear(dim, num_classes) for dim in out_channels
                ])
            else:
                # Single-stage output
                self.seg_head = nn.Linear(out_channels, num_classes)
        else:
            self.seg_head = None

    @property
    def num_down_stages(self) -> int:
        return len(self.down_stages) if self.down_stages is not None else 0

    @property
    def num_up_stages(self) -> int:
        return len(self.up_stages) if self.up_stages is not None else 0

    @property
    def out_dim(self) -> Union[int, List[int]]:
        """Output feature dimension(s)."""
        if self.output_stage_wise:
            out_dim = [stage.out_dim for stage in self.up_stages][::-1]
            out_dim += [self.down_stages[-1].out_dim]
            return out_dim
        if self.up_stages is not None:
            return self.up_stages[-1].out_dim
        if self.down_stages is not None:
            return self.down_stages[-1].out_dim
        return self.first_stage.out_dim

    def forward(
        self,
        hierarchy: "SuperpointHierarchy",
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Forward pass on a superpoint hierarchy.

        Args:
            hierarchy: SuperpointHierarchy containing multi-level superpoint data.

        Returns:
            Output features. If output_stage_wise=True, returns list of tensors
            for each level. Otherwise, returns single tensor for finest level.
        """
        # Get first level data
        start_level = 1 if self.nano else 0
        num_levels = hierarchy.num_levels

        # Apply first MLPs on handcrafted features (for nano mode)
        if self.nano:
            level_data = hierarchy.get_level(1)
            if self.node_mlps is not None and self.node_mlps[0] is not None:
                norm_index = self._get_norm_index(level_data)
                level_data["x"] = self.node_mlps[0](level_data["x"], batch=norm_index)
            if self.h_edge_mlps is not None and self.h_edge_mlps[0] is not None:
                norm_index = self._get_norm_index(level_data)
                edge_norm_idx = norm_index[level_data["edge_index"][0]]
                level_data["edge_attr"] = self.h_edge_mlps[0](
                    level_data["edge_attr"], batch=edge_norm_idx
                )

        # Forward first stage
        first_data = hierarchy.get_level(start_level)
        x, diameter = self._forward_first_stage(first_data)

        # Store diameter for next level
        if start_level + 1 < num_levels:
            hierarchy.get_level(start_level + 1)["diameter"] = diameter

        # Down stages
        down_outputs = []
        if self.nano:
            down_outputs.append(x)

        if self.down_stages is not None:
            node_mlp_idx = int(self.nano)
            for i_stage, stage in enumerate(self.down_stages):
                i_level = i_stage + 1 + int(self.nano)

                # Get level data
                level_data = hierarchy.get_level(i_level)
                prev_level_data = hierarchy.get_level(i_level - 1)

                # Process handcrafted features
                if self.node_mlps is not None and self.node_mlps[node_mlp_idx] is not None:
                    norm_index = self._get_norm_index(level_data)
                    if level_data.get("x") is not None:
                        level_data["x"] = self.node_mlps[node_mlp_idx](
                            level_data["x"], batch=norm_index
                        )

                if self.h_edge_mlps is not None and self.h_edge_mlps[node_mlp_idx] is not None:
                    norm_index = self._get_norm_index(level_data)
                    edge_norm_idx = norm_index[level_data["edge_index"][0]]
                    if level_data.get("edge_attr") is not None:
                        level_data["edge_attr"] = self.h_edge_mlps[node_mlp_idx](
                            level_data["edge_attr"], batch=edge_norm_idx
                        )

                if (
                    self.v_edge_mlps is not None
                    and i_stage < len(self.v_edge_mlps)
                    and self.v_edge_mlps[i_stage] is not None
                ):
                    norm_index = self._get_norm_index(prev_level_data)
                    if prev_level_data.get("v_edge_attr") is not None:
                        prev_level_data["v_edge_attr"] = self.v_edge_mlps[i_stage](
                            prev_level_data["v_edge_attr"], batch=norm_index
                        )

                node_mlp_idx += 1

                # Forward down stage
                x, diameter = self._forward_down_stage(
                    stage, hierarchy, i_level, x
                )
                down_outputs.append(x)

                # Store diameter for next level
                if i_level + 1 < num_levels:
                    hierarchy.get_level(i_level + 1)["diameter"] = diameter

        # Up stages
        up_outputs = []
        if self.up_stages is not None:
            for i_stage, stage in enumerate(self.up_stages):
                i_level = self.num_down_stages - i_stage - 1 + int(self.nano)
                x_skip = down_outputs[-(2 + i_stage)]

                x, _ = self._forward_up_stage(
                    stage, hierarchy, i_level, x, x_skip
                )
                up_outputs.append(x)

        # Output
        if self.output_stage_wise:
            out = [x] + up_outputs[::-1][1:] + [down_outputs[-1]]
            # Apply seg_head if present
            if self.seg_head is not None:
                if isinstance(self.seg_head, nn.ModuleList):
                    out = [head(feat) for head, feat in zip(self.seg_head, out)]
                else:
                    out = [self.seg_head(feat) for feat in out]
            return out

        # Apply seg_head if present (single-stage output)
        if self.seg_head is not None and not isinstance(self.seg_head, nn.ModuleList):
            x = self.seg_head(x)
        
        return x

    def _get_norm_index(self, level_data: Dict[str, Any]) -> torch.Tensor:
        """Get normalization indices based on norm_mode."""
        if self.norm_mode == "graph":
            return level_data.get("batch", torch.zeros(
                level_data["pos"].shape[0], dtype=torch.long, device=level_data["pos"].device
            ))
        elif self.norm_mode == "segment":
            return level_data.get("super_index", level_data.get("batch", torch.zeros(
                level_data["pos"].shape[0], dtype=torch.long, device=level_data["pos"].device
            )))
        else:  # node
            return torch.arange(
                level_data["pos"].shape[0], dtype=torch.long, device=level_data["pos"].device
            )

    def _forward_first_stage(
        self,
        data: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward first stage."""
        norm_index = self._get_norm_index(data)
        x = data.get("x")
        point_hf = data.get("point_hf")
        if isinstance(self.first_stage, PointStage):
            return self.first_stage(
                x,
                norm_index,
                pos=data.get("pos"),
                diameter=None,
                node_size=data.get("node_size"),
                super_index=data.get("super_index"),
                edge_index=data.get("edge_index"),
                edge_attr=data.get("edge_attr"),
                x_mlp=point_hf,
            )
        first_x = point_hf if point_hf is not None else x
        return self.first_stage(
            first_x,
            norm_index,
            pos=data.get("pos"),
            diameter=None,
            node_size=data.get("node_size"),
            super_index=data.get("super_index"),
            edge_index=data.get("edge_index"),
            edge_attr=data.get("edge_attr"),
        )

    def _forward_down_stage(
        self,
        stage: DownNFuseStage,
        hierarchy: "SuperpointHierarchy",
        i_level: int,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward down stage."""
        level_data = hierarchy.get_level(i_level)
        prev_level_data = hierarchy.get_level(i_level - 1)
        is_last = i_level == hierarchy.num_levels - 1

        x_hf = level_data.get("node_hf") if self.use_node_hf else None
        norm_index = self._get_norm_index(level_data)

        return stage(
            x_hf,
            x,
            norm_index,
            prev_level_data.get("super_index"),
            pos=level_data.get("pos"),
            diameter=level_data.get("diameter"),
            node_size=level_data.get("node_size"),
            super_index=level_data.get("super_index") if not is_last else None,
            edge_index=level_data.get("edge_index"),
            edge_attr=level_data.get("edge_attr"),
            v_edge_attr=prev_level_data.get("v_edge_attr"),
            num_super=level_data.get("num_nodes"),
        )

    def _forward_up_stage(
        self,
        stage: UpNFuseStage,
        hierarchy: "SuperpointHierarchy",
        i_level: int,
        x: torch.Tensor,
        x_skip: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward up stage."""
        level_data = hierarchy.get_level(i_level)

        x_hf = level_data.get("node_hf") if self.use_node_hf else None
        x_skip_fused = self.feature_fusion(x_skip, x_hf)
        norm_index = self._get_norm_index(level_data)

        return stage(
            x_skip_fused,
            x,
            norm_index,
            level_data.get("super_index"),
            pos=level_data.get("pos"),
            diameter=None,
            node_size=level_data.get("node_size"),
            super_index=level_data.get("super_index"),
            edge_index=level_data.get("edge_index"),
            edge_attr=level_data.get("edge_attr"),
        )
