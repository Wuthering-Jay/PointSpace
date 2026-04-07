"""
SPT Wrapper for EZ-SP Integration.

This module provides a wrapper around the SPT network to make it compatible
with the SuperpointHierarchy structure used in PointSpace's EZ-SP implementation.

Author: PointSpace Team
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Union

from pointspace.models.backbone.ezsp.spt.spt import SPT
from pointspace.models.backbone.ezsp.superpoint_hierarchy import SuperpointHierarchy
from pointspace.models.builder import MODELS


@MODELS.register_module()
class EZSPTransformer(nn.Module):
    """
    Wrapper around SPT network for EZ-SP Stage 2.

    This module adapts the SPT network to work with PointSpace's SuperpointHierarchy
    structure instead of the original NAG (Nested Attribute Graph) structure.

    Args:
        num_classes: Number of semantic classes for output.
        in_channels: Input feature dimension (from SparseCNN).
        point_mlp: Channels for PointStage MLP.
        point_drop: Dropout for PointStage.
        nano: If True, skip PointStage (superpoint-only processing).
        
        down_dim: Feature dimensions for each DownNFuseStage.
        down_in_mlp: Input MLP channels for each DownNFuseStage.
        down_out_mlp: Output MLP channels for each DownNFuseStage.
        down_num_heads: Number of attention heads for each stage.
        down_num_blocks: Number of transformer blocks for each stage.
        
        up_dim: Feature dimensions for each UpNFuseStage.
        up_in_mlp: Input MLP channels for each UpNFuseStage.
        up_out_mlp: Output MLP channels for each UpNFuseStage.
        up_num_heads: Number of attention heads for each stage.
        up_num_blocks: Number of transformer blocks for each stage.
        
        use_pos: Whether to use normalized positions.
        pool: Pooling mode for downsampling.
        fusion: Feature fusion mode.
        
        **spt_kwargs: Additional arguments passed to SPT network.

    Forward:
        Input: SuperpointHierarchy with point/superpoint features and graphs.
        Output: Semantic logits for each point.
    """

    def __init__(
        self,
        num_classes: int = 13,
        in_channels: int = 32,
        # PointStage params
        point_mlp: Optional[List[int]] = None,
        point_drop: Optional[float] = None,
        nano: bool = False,
        # Down stage params
        down_dim: Optional[List[int]] = None,
        down_in_mlp: Optional[List[List[int]]] = None,
        down_out_mlp: Optional[List[List[int]]] = None,
        down_num_heads: Union[int, List[int]] = 1,
        down_num_blocks: Union[int, List[int]] = 1,
        down_ffn_ratio: Union[float, List[float]] = 4,
        down_residual_drop: Optional[List[float]] = None,
        down_attn_drop: Optional[List[float]] = None,
        down_drop_path: Optional[List[float]] = None,
        # Up stage params
        up_dim: Optional[List[int]] = None,
        up_in_mlp: Optional[List[List[int]]] = None,
        up_out_mlp: Optional[List[List[int]]] = None,
        up_num_heads: Union[int, List[int]] = 1,
        up_num_blocks: Union[int, List[int]] = 1,
        up_ffn_ratio: Union[float, List[float]] = 4,
        up_residual_drop: Optional[List[float]] = None,
        up_attn_drop: Optional[List[float]] = None,
        up_drop_path: Optional[List[float]] = None,
        # Feature settings
        use_pos: bool = True,
        pool: Union[str, List[str]] = "max",
        fusion: str = "cat",
        output_stage_wise: bool = False,
        **spt_kwargs,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.nano = nano
        self.output_stage_wise = output_stage_wise

        # Set default MLP if not provided
        if point_mlp is None and not nano:
            # Default: project from in_channels to first down_dim
            first_dim = down_dim[0] if down_dim else 64
            point_mlp = [in_channels, first_dim]

        # Set default down_dim if not provided
        if down_dim is None:
            down_dim = [64, 128, 256]  # 3 down stages

        # Set default in_mlp to handle input feature dimension
        if down_in_mlp is None:
            if nano:
                # Nano mode: first stage gets features directly
                down_in_mlp = [[in_channels, down_dim[0]]]
                # Subsequent stages receive concatenated features (parent + pooled child)
                # For stage i (i >= 1):
                #   - parent features: down_dim[i-1] (from previous stage output)
                #   - pooled child features: down_dim[i-1] (pooled from same level)
                #   - concatenated: 2 * down_dim[i-1]
                if fusion == "cat":
                    for i in range(1, len(down_dim)):
                        down_in_mlp.append([2 * down_dim[i-1], down_dim[i]])
                else:
                    # For additive fusion, dimension stays same
                    down_in_mlp += [[d, d] for d in down_dim[1:]]
            else:
                # Normal mode: PointStage handles initial projection
                down_in_mlp = [[d, d] for d in down_dim]

        # Set default up stages (typically one less than down)
        if up_dim is None and len(down_dim) > 1:
            up_dim = down_dim[:-1][::-1]  # Reverse and remove last
        
        if up_in_mlp is None and up_dim is not None:
            # Concatenation fusion doubles the channels
            if fusion == "cat":
                up_in_mlp = [[d * 2, d] for d in up_dim]
            else:
                up_in_mlp = [[d, d] for d in up_dim]

        # Build SPT network
        self.spt = SPT(
            point_mlp=point_mlp,
            point_drop=point_drop,
            nano=nano,
            down_dim=down_dim,
            down_in_mlp=down_in_mlp,
            down_out_mlp=down_out_mlp,
            down_num_heads=down_num_heads,
            down_num_blocks=down_num_blocks,
            down_ffn_ratio=down_ffn_ratio,
            down_residual_drop=down_residual_drop,
            down_attn_drop=down_attn_drop,
            down_drop_path=down_drop_path,
            up_dim=up_dim,
            up_in_mlp=up_in_mlp,
            up_out_mlp=up_out_mlp,
            up_num_heads=up_num_heads,
            up_num_blocks=up_num_blocks,
            up_ffn_ratio=up_ffn_ratio,
            up_residual_drop=up_residual_drop,
            up_attn_drop=up_attn_drop,
            up_drop_path=up_drop_path,
            use_pos=use_pos,
            pool=pool,
            fusion=fusion,
            output_stage_wise=output_stage_wise,
            # Add segmentation head directly in SPT
            num_classes=num_classes,
            add_seg_head=True,
            **spt_kwargs,
        )

    def forward(self, nag: SuperpointHierarchy) -> torch.Tensor:
        """
        Forward pass through SPT network.

        Args:
            nag: SuperpointHierarchy with multi-level superpoint data.
                Expected to have:
                - Level 0 (points): pos, x (features), super_index
                - Level 1+ (superpoints): pos, x, edge_index, edge_attr, super_index

        Returns:
            seg_logits_superpoint: Semantic logits at SUPERPOINT level.
                Shape: [N_superpoints_L1, num_classes]
                
        Note:
            This returns superpoint-level logits, NOT point-level!
            The segmentor is responsible for propagating to points for evaluation.
            Loss should be computed at superpoint level.
        """
        # Forward through SPT (includes seg_head)
        seg_logits_superpoint = self.spt(nag)
        
        # Return superpoint-level logits
        # Shape: [num_superpoints_L1, num_classes]
        return seg_logits_superpoint


@MODELS.register_module()
class EZSPTransformerSimple(nn.Module):
    """
    Simplified SPT Transformer for quick experimentation.

    This is a minimal configuration with:
    - 1 DownNFuseStage
    - 1 UpNFuseStage
    - Simple pooling and fusion

    Args:
        num_classes: Number of semantic classes.
        in_channels: Input feature dimension.
        hidden_dim: Hidden feature dimension.
        num_heads: Number of attention heads.
        num_blocks: Number of transformer blocks per stage.
        use_pos: Whether to use normalized positions.
    """

    def __init__(
        self,
        num_classes: int = 13,
        in_channels: int = 32,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_blocks: int = 2,
        use_pos: bool = True,
    ):
        super().__init__()

        self.transformer = EZSPTransformer(
            num_classes=num_classes,
            in_channels=in_channels,
            nano=False,
            point_mlp=[in_channels, hidden_dim],
            down_dim=[hidden_dim, hidden_dim * 2],
            down_num_heads=[num_heads, num_heads],
            down_num_blocks=[num_blocks, num_blocks],
            up_dim=[hidden_dim],
            up_num_heads=[num_heads],
            up_num_blocks=[num_blocks],
            use_pos=use_pos,
            pool="max",
            fusion="cat",
        )

    def forward(self, nag: SuperpointHierarchy) -> torch.Tensor:
        return self.transformer(nag)
