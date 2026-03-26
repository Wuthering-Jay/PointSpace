"""
GPU Greedy Partition for EZ-SP

Wrapper around torch-graph-components for GPU-accelerated graph clustering.
This module provides the core partition algorithm used in EZ-SP for
creating superpoint hierarchies.

Reference: EZ-SP (https://arxiv.org/abs/2402.04991)
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple, Optional, Dict, Any
from torch_scatter import scatter_sum, scatter_mean
from torch_geometric.nn import knn_graph

try:
    from torch_graph_components import merge_components_by_contour_prior
    from torch_graph_components.merge import component_graph
    TORCH_GRAPH_COMPONENTS_AVAILABLE = True
except ImportError:
    TORCH_GRAPH_COMPONENTS_AVAILABLE = False
    merge_components_by_contour_prior = None
    component_graph = None

from pointspace.models.ezsp.utils import sizes_to_ptr


def scatter_mean_weighted(x: Tensor, index: Tensor, weights: Tensor) -> Tensor:
    """Compute weighted scatter mean.

    Args:
        x: Values tensor (N, D)
        index: Segment indices (N,)
        weights: Weights for each value (N,)

    Returns:
        Weighted means for each segment (num_segments, D)
    """
    if x.dim() == 1:
        x = x.unsqueeze(-1)

    weights = weights.float()
    if weights.dim() == 1:
        weights = weights.unsqueeze(-1)

    # Weighted sum
    weighted_x = x * weights
    sum_weighted = scatter_sum(weighted_x, index, dim=0)

    # Sum of weights per segment
    sum_weights = scatter_sum(weights, index, dim=0)

    # Avoid division by zero
    sum_weights = sum_weights.clamp(min=1e-8)

    return sum_weighted / sum_weights


class GPUGreedyPartition(nn.Module):
    """GPU-accelerated greedy graph partition.

    Uses the contour prior energy function for merging graph components.
    This is the core algorithm of EZ-SP that replaces CPU-based Cut-Pursuit.

    The algorithm:
    1. Builds a KNN graph from point positions
    2. Computes edge weights based on feature similarity
    3. Iteratively merges components based on energy minimization
    4. Returns point-to-superpoint assignments

    Args:
        reg: Regularization strength (controls partition coarseness). Default: 0.02
        min_size: Minimum superpoint size. Default: 30
        k: Number of KNN neighbors. Default: 8
        r_max: Maximum neighbor distance. Default: None (unlimited)
        edge_weight_mode: Mode for computing edge weights.
            Options: 'unit', 'inverse_distance', 'affinity'. Default: 'unit'
        edge_reduce: How to reduce duplicate edges. Default: 'add'
        max_iterations: Maximum merging iterations. Default: -1 (unlimited)
        connect_isolated: Whether to connect isolated nodes. Default: False
        w_adjacency: Weight for adjacency edges. Default: 0.0
        verbose: Print algorithm info. Default: False
    """

    EDGE_WEIGHT_MODES = ['unit', 'inverse_distance', 'affinity']

    def __init__(
        self,
        reg: float = 0.02,
        min_size: int = 30,
        k: int = 8,
        r_max: Optional[float] = None,
        edge_weight_mode: str = 'unit',
        edge_reduce: str = 'add',
        max_iterations: int = -1,
        connect_isolated: bool = False,
        w_adjacency: float = 0.0,
        sharding: Optional[int] = None,
        verbose: bool = False,
    ):
        super().__init__()

        if not TORCH_GRAPH_COMPONENTS_AVAILABLE:
            raise ImportError(
                "torch-graph-components is required for GPUGreedyPartition. "
                "Install with: pip install torch-graph-components"
            )

        assert edge_weight_mode in self.EDGE_WEIGHT_MODES, \
            f"Invalid edge_weight_mode: {edge_weight_mode}. Options: {self.EDGE_WEIGHT_MODES}"

        self.reg = reg
        self.min_size = min_size
        self.k = k
        self.r_max = r_max
        self.edge_weight_mode = edge_weight_mode
        self.edge_reduce = edge_reduce
        self.max_iterations = max_iterations
        self.connect_isolated = connect_isolated
        self.w_adjacency = w_adjacency
        self.sharding = sharding
        self.verbose = verbose

    def build_knn_graph(
        self,
        pos: Tensor,
        batch: Optional[Tensor] = None,
    ) -> Tensor:
        """Build KNN graph from positions.

        Args:
            pos: Point positions (N, 3)
            batch: Batch indices (N,). If None, treats as single batch.

        Returns:
            edge_index: Graph edges (2, E)
        """
        edge_index = knn_graph(
            pos,
            k=self.k,
            batch=batch,
            loop=False,  # No self-loops
        )

        # Filter by r_max if specified
        if self.r_max is not None:
            src, dst = edge_index
            dist = (pos[src] - pos[dst]).norm(dim=-1)
            mask = dist <= self.r_max
            edge_index = edge_index[:, mask]

        return edge_index

    def compute_edge_weights(
        self,
        feat: Tensor,
        pos: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        """Compute edge weights based on feature/position similarity.

        Args:
            feat: Point features (N, D)
            pos: Point positions (N, 3)
            edge_index: Graph edges (2, E)

        Returns:
            edge_weights: Weights for each edge (E,)
        """
        src, dst = edge_index

        if self.edge_weight_mode == 'unit':
            # Uniform weights
            return torch.ones(edge_index.shape[1], device=edge_index.device)

        elif self.edge_weight_mode == 'inverse_distance':
            # Inverse distance weighting
            dist = (pos[src] - pos[dst]).norm(dim=-1)
            return 1.0 / (dist + 1e-8)

        elif self.edge_weight_mode == 'affinity':
            # Feature affinity (cosine similarity)
            feat_src = feat[src]
            feat_dst = feat[dst]
            # Normalize
            feat_src = feat_src / (feat_src.norm(dim=-1, keepdim=True) + 1e-8)
            feat_dst = feat_dst / (feat_dst.norm(dim=-1, keepdim=True) + 1e-8)
            # Cosine similarity -> affinity
            sim = (feat_src * feat_dst).sum(dim=-1)
            return (sim + 1) / 2  # Map from [-1, 1] to [0, 1]

        else:
            raise ValueError(f"Unknown edge_weight_mode: {self.edge_weight_mode}")

    def forward(
        self,
        feat: Tensor,
        pos: Tensor,
        batch: Optional[Tensor] = None,
        edge_index: Optional[Tensor] = None,
        edge_attr: Optional[Tensor] = None,
        node_size: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Compute graph partition.

        Args:
            feat: Point features (N, D)
            pos: Point positions (N, 3)
            batch: Batch indices (N,). If None, treats as single batch.
            edge_index: Pre-computed graph edges (2, E). If None, builds KNN graph.
            edge_attr: Pre-computed edge weights (E,). If None, computes from mode.
            node_size: Size of each node (N,). If None, all nodes have size 1.

        Returns:
            Dictionary containing:
            - super_index: Point-to-superpoint mapping (N,)
            - num_superpoints: Number of superpoints
            - super_feat: Superpoint features (num_superpoints, D)
            - super_pos: Superpoint positions (num_superpoints, 3)
            - super_size: Superpoint sizes (num_superpoints,)
            - super_edge_index: Superpoint graph edges (2, E')
            - super_edge_attr: Superpoint edge weights (E',)
        """
        N = feat.shape[0]
        device = feat.device

        # Build graph if not provided
        if edge_index is None:
            edge_index = self.build_knn_graph(pos, batch)

        # Compute edge weights if not provided
        if edge_attr is None:
            edge_attr = self.compute_edge_weights(feat, pos, edge_index)

        # Default node sizes
        if node_size is None:
            node_size = torch.ones(N, device=device, dtype=torch.long)

        # Initial component assignment (each point is its own component)
        I = torch.arange(N, device=device)

        # Compute component features and sizes
        S_cp = node_size.clone()
        X_cp = feat.clone()
        P_cp = pos.clone() if self.connect_isolated else None

        # Build component graph
        E_cp, W_cp = component_graph(I, edge_index, edge_attr, no_self_loops=False)

        # Run greedy partition
        I_merged, iterations, (X_merged, S_merged, E_out, W_out, P_merged) = \
            merge_components_by_contour_prior(
                X=X_cp,
                S=S_cp,
                E=E_cp,
                W=W_cp,
                reg=self.reg,
                min_size=self.min_size,
                merge_only_small=False,
                P=P_cp,
                k=self.k if self.connect_isolated else -1,
                w_adjacency=self.w_adjacency,
                depth=0,
                max_iterations=self.max_iterations,
                sharding=self.sharding,
                reduce=self.edge_reduce,
                verbose=self.verbose,
            )

        # Compute final positions if not done in merge
        if P_merged is None:
            P_merged = scatter_mean_weighted(pos, I_merged, node_size.float())

        # Build super_index (maps points to superpoints)
        super_index = I_merged

        if self.verbose:
            num_sp = super_index.max().item() + 1
            print(f"GPUGreedyPartition: {N} points -> {num_sp} superpoints "
                  f"({iterations} iterations)")

        return {
            'super_index': super_index,
            'num_superpoints': super_index.max().item() + 1,
            'super_feat': X_merged,
            'super_pos': P_merged,
            'super_size': S_merged,
            'super_edge_index': E_out,
            'super_edge_attr': W_out,
        }


class HierarchicalPartition(nn.Module):
    """Multi-level hierarchical partition.

    Creates a hierarchy of superpoints at multiple scales by running
    GPUGreedyPartition multiple times with increasing min_size.

    Args:
        reg: Regularization values for each level
        min_sizes: Minimum sizes for each level (e.g., [5, 30, 90])
        k: KNN neighbors
        **kwargs: Additional args passed to GPUGreedyPartition
    """

    def __init__(
        self,
        reg: float = 0.02,
        min_sizes: Tuple[int, ...] = (5, 30, 90),
        k: int = 8,
        **kwargs,
    ):
        super().__init__()

        self.num_levels = len(min_sizes)
        self.partitions = nn.ModuleList([
            GPUGreedyPartition(reg=reg, min_size=ms, k=k, **kwargs)
            for ms in min_sizes
        ])

    def forward(
        self,
        feat: Tensor,
        pos: Tensor,
        batch: Optional[Tensor] = None,
    ) -> Dict[str, Any]:
        """Compute hierarchical partition.

        Returns:
            Dictionary with partition info at each level
        """
        results = []
        current_feat = feat
        current_pos = pos
        current_batch = batch

        for i, partition in enumerate(self.partitions):
            result = partition(
                feat=current_feat,
                pos=current_pos,
                batch=current_batch,
            )
            results.append(result)

            # Use superpoint features/positions for next level
            current_feat = result['super_feat']
            current_pos = result['super_pos']
            # Batch indices need to be recomputed for superpoints
            if current_batch is not None:
                current_batch = current_batch[result['super_index']]

        return {
            'levels': results,
            'num_levels': self.num_levels,
        }
