"""
GreedyContourPriorPartition - GPU-based Greedy Superpoint Partition

This module implements the core partition algorithm of EZ-SP, which uses
learned CNN features to partition point clouds into hierarchical superpoints.

Key features:
- GPU-based KNN graph construction using pointops
- Greedy component merging with contour prior energy function
- Multi-level hierarchical partitioning

Author: PointSpace Team
"""

from typing import Dict, List, Optional, Tuple, Union

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_scatter import scatter_sum, scatter_mean, scatter_min, scatter_std

from pointspace.models.builder import MODELS
from pointspace.models.backbone.ezsp.superpoint_hierarchy import (
    Cluster,
    SuperpointLevel,
    SuperpointHierarchy,
)


def scatter_mean_weighted(
    x: Tensor,
    index: Tensor,
    weights: Tensor,
    dim: int = 0,
) -> Tensor:
    """
    Weighted scatter mean

    Args:
        x: [N, C] Features
        index: [N] Indices
        weights: [N] Weights
        dim: Scatter dimension

    Returns:
        [M, C] Weighted mean features
    """
    weighted_x = x * weights.unsqueeze(-1).float()
    sum_weighted = scatter_sum(weighted_x, index, dim=dim)
    sum_weights = scatter_sum(weights.float(), index, dim=dim).unsqueeze(-1).clamp(min=1e-6)
    return sum_weighted / sum_weights


def compute_segment_normal(
    pos: Tensor,
    super_index: Tensor,
    node_size: Tensor,
    num_segments: int,
) -> Tensor:
    """
    Compute normal vector for each segment using PCA of constituent points.
    
    Args:
        pos: [N, 3] Point positions
        super_index: [N] Segment index for each point
        node_size: [N] Weight for each point (typically 1 for level-0 points)
        num_segments: Number of segments
    
    Returns:
        normal: [M, 3] Normal vector for each segment (unit length, pointing towards Z+)
    """
    device = pos.device
    
    # Compute segment centroids (weighted mean position)
    centroid = scatter_mean_weighted(pos, super_index, node_size, dim=0)  # [M, 3]
    
    # Compute covariance matrix for each segment
    # Cov = E[(x - μ)(x - μ)^T]
    pos_centered = pos - centroid[super_index]  # [N, 3]
    
    # Weight the centered positions
    pos_weighted = pos_centered * node_size.unsqueeze(-1).float().sqrt()  # [N, 3]
    
    # Compute covariance using scatter operations
    # We need to compute the 3x3 covariance matrix for each segment
    # Cov[i,j] = sum_k w_k * (x_k[i] - μ[i]) * (x_k[j] - μ[j])
    
    # Create all 9 components of covariance matrix
    xx = scatter_sum(pos_weighted[:, 0] * pos_weighted[:, 0], super_index, dim=0)  # [M]
    xy = scatter_sum(pos_weighted[:, 0] * pos_weighted[:, 1], super_index, dim=0)
    xz = scatter_sum(pos_weighted[:, 0] * pos_weighted[:, 2], super_index, dim=0)
    yy = scatter_sum(pos_weighted[:, 1] * pos_weighted[:, 1], super_index, dim=0)
    yz = scatter_sum(pos_weighted[:, 1] * pos_weighted[:, 2], super_index, dim=0)
    zz = scatter_sum(pos_weighted[:, 2] * pos_weighted[:, 2], super_index, dim=0)
    
    # Normalize by sum of weights
    sum_weights = scatter_sum(node_size.float(), super_index, dim=0).clamp(min=1e-6)  # [M]
    cov_xx = xx / sum_weights
    cov_xy = xy / sum_weights
    cov_xz = xz / sum_weights
    cov_yy = yy / sum_weights
    cov_yz = yz / sum_weights
    cov_zz = zz / sum_weights
    
    # Build covariance matrices [M, 3, 3]
    cov = torch.zeros(num_segments, 3, 3, device=device, dtype=pos.dtype)
    cov[:, 0, 0] = cov_xx
    cov[:, 0, 1] = cov_xy
    cov[:, 0, 2] = cov_xz
    cov[:, 1, 0] = cov_xy  # symmetric
    cov[:, 1, 1] = cov_yy
    cov[:, 1, 2] = cov_yz
    cov[:, 2, 0] = cov_xz  # symmetric
    cov[:, 2, 1] = cov_yz  # symmetric
    cov[:, 2, 2] = cov_zz
    
    # Compute eigenvalues and eigenvectors
    # The normal is the eigenvector corresponding to the smallest eigenvalue
    try:
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)  # [M, 3], [M, 3, 3]
        # Smallest eigenvalue is at index 0 after eigh (sorted ascending)
        normal = eigenvectors[:, :, 0]  # [M, 3]
    except Exception:
        # Fallback: use Z+ direction if eigenvalue computation fails
        normal = torch.zeros(num_segments, 3, device=device, dtype=pos.dtype)
        normal[:, 2] = 1.0
    
    # Normalize to unit length
    normal = F.normalize(normal, p=2, dim=1)
    
    # Orient normals towards Z+ by convention (following official implementation)
    flip_mask = normal[:, 2] < 0
    normal[flip_mask] *= -1
    
    return normal


def compute_point_normal_from_knn(pos: Tensor, neighbor_idx: Tensor) -> Tensor:
    """Compute per-point normals from KNN neighborhoods via PCA."""
    device = pos.device
    n, k = neighbor_idx.shape
    if n == 0:
        return torch.empty(0, 3, device=device, dtype=pos.dtype)

    safe_nn = neighbor_idx.clone()
    invalid = safe_nn < 0
    if invalid.any():
        self_idx = torch.arange(n, device=device).view(-1, 1).expand(n, k)
        safe_nn[invalid] = self_idx[invalid]

    neigh = pos[safe_nn]  # [N, K, 3]
    center = neigh.mean(dim=1, keepdim=True)
    centered = neigh - center
    cov = centered.transpose(1, 2).bmm(centered) / max(k, 1)  # [N, 3, 3]

    try:
        _, evec = torch.linalg.eigh(cov)
        normal = evec[:, :, 0]
    except Exception:
        normal = torch.zeros(n, 3, device=device, dtype=pos.dtype)
        normal[:, 2] = 1.0

    normal = F.normalize(normal, p=2, dim=1)
    flip_mask = normal[:, 2] < 0
    normal[flip_mask] *= -1
    return normal


@MODELS.register_module()
class GreedyContourPriorPartition(nn.Module):
    """
    Greedy Contour Prior Partition Module

    Partitions point clouds into hierarchical superpoints using learned CNN
    features and a greedy energy-based merging algorithm.

    The partition is performed entirely on GPU, with KNN graph construction
    using the fast pointops CUDA kernels.

    Energy Function:
        E = Σ_i ||X_i - μ_i||² + reg * Σ_(i,j)∈E w_ij * [μ_i ≠ μ_j]

        - First term: Intra-component consistency
        - Second term: Boundary smoothness (contour prior)

    Args:
        reg: float | List[float] - Regularization strength, typical 2e-2
            Larger → coarser partition → fewer/larger superpoints
        min_size: int | List[int] - Minimum superpoint size per level, typical [5, 30, 90]
        k_adjacency: int - Number of KNN neighbors for graph construction
        spatial_weight: float | None - Spatial coordinate weight
            None: Pure feature partition (EZ-SP default)
            float: x ← [x, spatial_weight * pos]
        edge_weight_mode: str - Edge weight computation mode
            'unit': 1 (no weighting)
            'exp_neg_latent_distance': exp(-||x_i - x_j|| / d_0)
            'affinity_latent_distance': affinity form
        d_0: float | None - Reference distance, None = auto (mean)
        w_adjacency: float - Weight for newly created edges (isolated nodes)
        max_iterations: int - Max merge iterations, -1 = unlimited
        edge_reduce: str - Edge reduce mode: 'add', 'mean', 'max', 'min'

    Input:
        pos: [N, 3] - Point coordinates
        x: [N, C] - CNN point embeddings (key input!)
        offset: [B] - Cumulative point counts (for GPU KNN batch isolation)
        y: [N] | [N, num_classes] | None - Optional GT labels

    Output:
        SuperpointHierarchy - Multi-level superpoint graph structure

    Note:
        Adjacency graph is built dynamically via GPU KNN in forward(),
        no need to pre-compute edge_index.
    """

    _EDGE_WEIGHT_MODES = [
        "unit",
        "inverse_distance",
        "exp_neg_distance",
        "exp_neg_latent_distance",
        "affinity_latent_distance",
    ]

    def __init__(
        self,
        reg: Union[float, List[float]] = 2e-2,
        min_size: Union[int, List[int]] = [5, 30, 90],
        k_adjacency: int = 10,
        spatial_weight: Optional[float] = None,
        edge_weight_mode: str = "unit",
        d_0: Optional[float] = None,
        w_adjacency: float = 0.0,
        max_iterations: int = -1,
        edge_reduce: str = "add",
        build_edge_features: bool = True,
        build_vertical_features: bool = True,
    ):
        super().__init__()

        # Normalize parameters to lists
        if isinstance(min_size, list):
            num_levels = len(min_size)
        elif isinstance(reg, list):
            num_levels = len(reg)
        else:
            num_levels = 1

        self.reg = reg if isinstance(reg, list) else [reg] * num_levels
        self.min_size = min_size if isinstance(min_size, list) else [min_size] * num_levels

        assert len(self.reg) == len(self.min_size), (
            f"reg ({len(self.reg)}) and min_size ({len(self.min_size)}) "
            f"must have same length"
        )

        self.k_adjacency = k_adjacency
        self.spatial_weight = spatial_weight
        self.edge_weight_mode = edge_weight_mode
        self.d_0 = d_0
        self.w_adjacency = w_adjacency
        self.max_iterations = max_iterations
        self.edge_reduce = edge_reduce
        self.build_edge_features = build_edge_features
        self.build_vertical_features = build_vertical_features

        assert edge_weight_mode in self._EDGE_WEIGHT_MODES, (
            f"Invalid edge_weight_mode: {edge_weight_mode}, "
            f"valid options: {self._EDGE_WEIGHT_MODES}"
        )

    def forward(
        self,
        pos: Tensor,
        x: Tensor,
        offset: Tensor,
        batch: Optional[Tensor] = None,
        y: Optional[Tensor] = None,
    ) -> SuperpointHierarchy:
        """
        Execute hierarchical partition

        Args:
            pos: [N, 3] Point positions
            x: [N, D] Point features
            offset: [B] Cumulative point counts per batch
            batch: [N] Optional pre-computed batch indices (for performance)
            y: [N] or [N, C] Optional ground truth labels

        Data flow:
            1. GPU KNN graph construction (using pointops, auto batch isolation)
            2. Level 0 → Level 1 → ... → Level L

            Each Level:
                1. Compute edge weights (based on feature distance)
                2. Optional: Concatenate spatial coordinates to features
                3. Call torch-graph-components for greedy merging
                4. Build next level data
        """
        device = pos.device
        num_points = pos.shape[0]

        # ========== GPU KNN Graph Construction ==========
        from libs.pointops.functions import knn_query

        neighbor_idx, neighbor_dist = knn_query(self.k_adjacency, pos, offset)
        edge_index = self._neighbor_idx_to_edge_index(neighbor_idx)

        # Compute batch indices (use pre-computed if available for performance)
        if batch is not None:
            # ✅ Performance optimization: use pre-computed batch
            batch = batch.to(device)
        else:
            # Fallback: compute on-the-fly
            batch = self._offset_to_batch(offset, num_points, device)

        # Process labels to histogram format
        y_hist = None
        if y is not None:
            y_hist = self._prepare_label_histogram(y, device)

        # Initialize Level 0 data
        log_size_0 = torch.zeros(num_points, device=device, dtype=pos.dtype)
        data = {
            "pos": pos,
            "x": x,
            "edge_index": edge_index,
            "batch": batch,
            "offset": offset,
            "node_size": torch.ones(num_points, device=device, dtype=torch.long),
            "normal": compute_point_normal_from_knn(pos, neighbor_idx)
            if (self.build_edge_features or self.build_vertical_features)
            else None,
            "log_size": log_size_0,
            "log_length": log_size_0 / 3.0,
            "log_surface": log_size_0 * 2.0 / 3.0,
            "log_volume": log_size_0,
            "super_index": None,
            "y": y_hist,
            "v_edge_attr": None,  # Will be computed when building Level 1
        }

        data_list = [data]

        # Hierarchical partition
        for level, (reg, min_size) in enumerate(zip(self.reg, self.min_size)):
            d = data_list[level]

            # 1. Compute edge weights
            edge_weight = self._compute_edge_weights(d["x"], d["edge_index"])
            d["edge_weight"] = edge_weight
            d["edge_attr"] = (
                self._compute_horizontal_edge_attr(
                    pos=d["pos"],
                    edge_index=d["edge_index"],
                    normal=d["normal"],
                    log_length=d["log_length"],
                    log_surface=d["log_surface"],
                    log_volume=d["log_volume"],
                    log_size=d["log_size"],
                )
                if self.build_edge_features
                else None
            )

            # 2. Optional: Concatenate spatial coordinates
            x_partition = d["x"]
            if self.spatial_weight is not None and self.spatial_weight > 0:
                x_partition = torch.cat(
                    [x_partition, d["pos"] * self.spatial_weight], dim=1
                )

            # 3. Call component merging
            super_index, merged_data = self._merge_components(
                x=x_partition,
                pos=d["pos"],
                node_size=d["node_size"],
                edge_index=d["edge_index"],
                edge_weight=edge_weight,
                reg=reg,
                min_size=min_size,
                y=d.get("y"),
                batch=d.get("batch"),
                normal_child=d.get("normal"),
                log_length_child=d.get("log_length"),
                log_surface_child=d.get("log_surface"),
                log_volume_child=d.get("log_volume"),
                log_size_child=d.get("log_size"),
            )

            # 4. Update current level with super_index and v_edge_attr
            d["super_index"] = super_index
            # Store v_edge_attr in the child level (for use in DownNFuseStage)
            d["v_edge_attr"] = merged_data.pop("v_edge_attr", None)
            data_list[level] = d

            # 5. Add new level
            data_list.append(merged_data)

        return SuperpointHierarchy(data_list)

    def _offset_to_batch(self, offset: Tensor, num_points: int, device) -> Tensor:
        """Convert cumulative offset to per-point batch indices."""
        if offset.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=device)

        boundaries = torch.cat([offset.new_zeros(1), offset])
        counts = (boundaries[1:] - boundaries[:-1]).clamp(min=0)
        batch = torch.arange(offset.numel(), device=device).repeat_interleave(counts)

        # Safety fallback for malformed offset.
        if batch.numel() != num_points:
            batch = torch.zeros(num_points, dtype=torch.long, device=device)
            for i in range(len(offset)):
                start = 0 if i == 0 else offset[i - 1].item()
                end = offset[i].item()
                batch[start:end] = i

        return batch

    def _prepare_label_histogram(self, y: Tensor, device) -> Tensor:
        """Convert labels to histogram format"""
        if y.dim() == 1:
            # Single label → histogram
            valid_mask = y >= 0
            num_classes = max(y[valid_mask].max().item() + 1, 1) if valid_mask.any() else 1
            y_hist = torch.zeros(y.shape[0], num_classes, device=device)
            if valid_mask.any():
                y_hist[valid_mask] = F.one_hot(
                    y[valid_mask].long(), num_classes=num_classes
                ).float()
            return y_hist
        else:
            return y.float()

    def _neighbor_idx_to_edge_index(self, neighbor_idx: Tensor) -> Tensor:
        """
        Convert KNN neighbor indices to edge_index format

        Args:
            neighbor_idx: [N, K] K neighbors for each point

        Returns:
            edge_index: [2, E] Edge indices (invalid edges removed)
        """
        N, K = neighbor_idx.shape
        device = neighbor_idx.device

        # Build source node indices [N, K]
        src = torch.arange(N, device=device).unsqueeze(1).expand(N, K)

        # Flatten
        src = src.reshape(-1)  # [N*K]
        dst = neighbor_idx.reshape(-1)  # [N*K]

        # Filter invalid edges (neighbor_idx == -1 means invalid)
        valid_mask = dst >= 0
        src = src[valid_mask]
        dst = dst[valid_mask]

        # Combine to edge_index
        edge_index = torch.stack([src, dst], dim=0)

        return edge_index

    def _compute_edge_weights(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """
        Compute edge weights based on features

        Args:
            x: [N, C] Point features
            edge_index: [2, E] Edge indices

        Returns:
            edge_attr: [E] Edge weights
        """
        src, dst = edge_index[0], edge_index[1]

        if self.edge_weight_mode == "unit":
            return torch.ones(edge_index.shape[1], device=x.device)

        # Compute feature distance
        latent_dist = (x[src] - x[dst]).norm(dim=1)
        if self.d_0 is not None:
            d_0 = torch.as_tensor(self.d_0, dtype=latent_dist.dtype, device=x.device)
        else:
            d_0 = latent_dist.detach().mean()
        d_0 = d_0.clamp_min(1e-6)

        if self.edge_weight_mode == "exp_neg_latent_distance":
            return torch.exp(-latent_dist / d_0)
        elif self.edge_weight_mode == "affinity_latent_distance":
            d_neg_exp = torch.exp(-latent_dist / d_0)
            eps = 1e-6
            return d_neg_exp / (1 - d_neg_exp + eps)
        elif self.edge_weight_mode == "inverse_distance":
            return 1 / (1 + latent_dist / d_0)
        elif self.edge_weight_mode == "exp_neg_distance":
            # Use spatial distance instead
            # This mode should be used with pos, not x
            return torch.exp(-latent_dist / d_0)
        else:
            raise ValueError(f"Unknown edge_weight_mode: {self.edge_weight_mode}")

    def _merge_components(
        self,
        x: Tensor,
        pos: Tensor,
        node_size: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor,
        reg: float,
        min_size: int,
        y: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
        normal_child: Optional[Tensor] = None,
        log_length_child: Optional[Tensor] = None,
        log_surface_child: Optional[Tensor] = None,
        log_volume_child: Optional[Tensor] = None,
        log_size_child: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict]:
        """
        Call torch-graph-components for component merging

        Args:
            x: [N, C] Node features
            pos: [N, 3] Node positions
            node_size: [N] Size of each node (num points it contains)
            edge_index: [2, E] Edge indices
            edge_weight: [E] Scalar edge weights for contour-prior merging
            reg: Regularization strength
            min_size: Minimum superpoint size
            y: Optional [N, num_classes] Label histograms
            batch: Optional [N] Batch indices for each node

        Returns:
            super_index: [N] Superpoint ID for each point
            merged_data: dict New level data including batch and offset
        """
        from torch_graph_components import merge_components_by_contour_prior
        from torch_graph_components.merge import component_graph

        device = x.device
        num_nodes = x.shape[0]

        # Initial state: each point is its own component
        I = torch.arange(num_nodes, device=device)
        S = node_size.float()

        # Compute component graph (removes self-loops)
        E_cp, W_cp = component_graph(I, edge_index, edge_weight * reg, no_self_loops=True)

        # Prepare position for isolated node handling
        P = pos if self.w_adjacency > 0 else None

        # Merge components
        try:
            I_merged, iterations, (X_merged, S_merged, E_merged, W_merged, P_merged) = (
                merge_components_by_contour_prior(
                    x,
                    S,
                    E_cp,
                    W_cp,
                    reg,
                    min_size,
                    merge_only_small=False,
                    P=P,
                    k=self.k_adjacency if self.w_adjacency > 0 else -1,
                    w_adjacency=self.w_adjacency,
                    depth=0,
                    max_iterations=self.max_iterations,
                    sharding=None,
                    reduce=self.edge_reduce,
                    verbose=False,
                )
            )
        except Exception as e:
            # Fallback: if merging fails, keep original components
            warnings.warn(
                f"Component merging failed, fallback to identity partition: {e}",
                stacklevel=2,
            )
            I_merged = I
            X_merged = x
            S_merged = S
            E_merged = E_cp
            W_merged = W_cp
            P_merged = pos

        # Compute merged positions if not computed
        if P_merged is None or P_merged.shape[0] != X_merged.shape[0]:
            P_merged = scatter_mean_weighted(pos, I_merged, node_size)

        # Build Cluster object
        super_index = I_merged
        sub = Cluster.from_super_index(super_index, num_nodes)

        num_super = X_merged.shape[0]

        # Compute geometric attributes for merged superpoints
        normal_merged = (
            compute_segment_normal(pos, super_index, node_size, num_super)
            if (self.build_edge_features or self.build_vertical_features)
            else None
        )
        log_size_merged = torch.log(S_merged.clamp(min=1.0))
        log_length_merged = log_size_merged / 3.0
        log_surface_merged = log_size_merged * 2.0 / 3.0
        log_volume_merged = log_size_merged

        # Aggregate labels
        y_merged = None
        if y is not None:
            y_merged = scatter_sum(y, super_index, dim=0)

        # Compute merged batch indices
        # Each superpoint inherits the batch of its constituent nodes
        # (all nodes in a superpoint should have the same batch)
        batch_merged = None
        offset_merged = None
        if batch is not None:
            num_superpoints = X_merged.shape[0]
            # Use scatter to get the batch of each superpoint
            # Since all nodes in a superpoint should have the same batch,
            # we can use any reduction (min, max, or mode). We use min for efficiency.
            batch_merged = scatter_min(batch, super_index, dim=0, dim_size=num_superpoints)[0]

            # Compute offset from batch
            num_batches = batch.max().item() + 1 if batch.numel() > 0 else 1
            # Count superpoints per batch
            batch_counts = torch.zeros(num_batches, dtype=torch.long, device=device)
            batch_counts.scatter_add_(0, batch_merged, torch.ones_like(batch_merged))
            offset_merged = batch_counts.cumsum(0)

        # Compute vertical edge attributes (v_edge_attr)
        # These are features for each child->parent edge used in attentive pooling
        # v_edge_attr[i] describes the relationship between child node i and its parent
        v_edge_attr = (
            self._compute_vertical_edge_attr(
                pos_child=pos,
                pos_parent=P_merged,
                super_index=super_index,
                node_size_child=node_size,
                node_size_parent=S_merged,
                normal_child=normal_child,
                normal_parent=normal_merged,
                log_length_child=log_length_child,
                log_length_parent=log_length_merged,
                log_surface_child=log_surface_child,
                log_surface_parent=log_surface_merged,
                log_volume_child=log_volume_child,
                log_volume_parent=log_volume_merged,
                log_size_child=log_size_child,
                log_size_parent=log_size_merged,
            )
            if self.build_vertical_features
            else None
        )

        # Compute horizontal edge features for merged graph (18D)
        h_edge_attr = (
            self._compute_horizontal_edge_attr(
                pos=P_merged,
                edge_index=E_merged,
                normal=normal_merged,
                log_length=log_length_merged,
                log_surface=log_surface_merged,
                log_volume=log_volume_merged,
                log_size=log_size_merged,
            )
            if self.build_edge_features
            else None
        )

        merged_data = {
            "pos": P_merged,
            "x": X_merged,
            "node_size": S_merged.long(),
            "edge_index": E_merged,
            "edge_weight": W_merged / reg if reg > 0 else W_merged,  # scalar weights for partition only
            "edge_attr": h_edge_attr,  # 18D handcrafted edge features for SPT RPE
            "normal": normal_merged,
            "log_size": log_size_merged,
            "log_length": log_length_merged,
            "log_surface": log_surface_merged,
            "log_volume": log_volume_merged,
            "sub": sub,
            "super_index": None,
            "y": y_merged,
            "batch": batch_merged,
            "offset": offset_merged,
            "v_edge_attr": v_edge_attr,  # Vertical edge attributes for attentive pooling
        }

        return super_index, merged_data

    def _compute_vertical_edge_attr(
        self,
        pos_child: Tensor,
        pos_parent: Tensor,
        super_index: Tensor,
        node_size_child: Tensor,
        node_size_parent: Tensor,
        normal_child: Optional[Tensor] = None,
        normal_parent: Optional[Tensor] = None,
        log_length_child: Optional[Tensor] = None,
        log_length_parent: Optional[Tensor] = None,
        log_surface_child: Optional[Tensor] = None,
        log_surface_parent: Optional[Tensor] = None,
        log_volume_child: Optional[Tensor] = None,
        log_volume_parent: Optional[Tensor] = None,
        log_size_child: Optional[Tensor] = None,
        log_size_parent: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute vertical edge attributes for child->parent edges

        These features describe the relationship between each child node and
        its parent superpoint, used for relative positional encoding in
        attentive pooling.

        Features computed (official SPT-compatible, 9D):
        - centroid_dir: Direction from child to parent centroid (3D, normalized)
        - centroid_dist: Distance from child to parent centroid (1D, sqrt-distance)
        - normal_angle: Cosine angle between child and parent normals (1D)
        - log_length: parent-child log length ratio (1D)
        - log_surface: parent-child log surface ratio (1D)
        - log_volume: parent-child log volume ratio (1D)
        - log_size: parent-child log size ratio (1D)

        Args:
            pos_child: [N_child, 3] Child node positions
            pos_parent: [N_parent, 3] Parent node positions
            super_index: [N_child] Maps child to parent
            node_size_child: [N_child] Size of each child
            node_size_parent: [N_parent] Size of each parent

        Returns:
            v_edge_attr: [N_child, 9] Vertical edge features
        """
        device = pos_child.device
        n_child = pos_child.shape[0]

        # Get parent positions for each child
        parent_pos = pos_parent[super_index]  # [N_child, 3]

        # Direction from child to parent (normalized)
        delta = parent_pos - pos_child  # [N_child, 3]
        dist = delta.norm(dim=1, keepdim=True).clamp(min=1e-6)  # [N_child, 1]
        centroid_dir = delta / dist  # [N_child, 3]
        centroid_dist = dist.sqrt()

        # Handle NaN (when child == parent position)
        centroid_dir = torch.nan_to_num(centroid_dir, nan=0.0)

        # normal_angle
        if normal_child is not None and normal_parent is not None:
            normal_angle = (normal_child * normal_parent[super_index]).sum(dim=1).abs().unsqueeze(1)
        else:
            normal_angle = torch.zeros((n_child, 1), device=device, dtype=pos_child.dtype)

        # Log-ratio features
        if (
            log_size_parent is None
            or log_size_child is None
            or log_length_parent is None
            or log_length_child is None
            or log_surface_parent is None
            or log_surface_child is None
            or log_volume_parent is None
            or log_volume_child is None
        ):
            parent_size = node_size_parent[super_index].float().clamp(min=1.0)
            child_size = node_size_child.float().clamp(min=1.0)
            log_size = torch.log(parent_size) - torch.log(child_size)
            log_size = log_size.unsqueeze(1)
            log_length = log_size / 3.0
            log_surface = log_size * 2.0 / 3.0
            log_volume = log_size
        else:
            log_length = (log_length_parent[super_index] - log_length_child).unsqueeze(1)
            log_surface = (log_surface_parent[super_index] - log_surface_child).unsqueeze(1)
            log_volume = (log_volume_parent[super_index] - log_volume_child).unsqueeze(1)
            log_size = (log_size_parent[super_index] - log_size_child).unsqueeze(1)

        # Concatenate features (9D)
        v_edge_attr = torch.cat(
            [
                centroid_dir,   # 3
                centroid_dist,  # 1
                normal_angle,   # 1
                log_length,     # 1
                log_surface,    # 1
                log_volume,     # 1
                log_size,       # 1
            ],
            dim=1,
        )

        return v_edge_attr

    def _compute_horizontal_edge_attr(
        self,
        pos: Tensor,
        edge_index: Tensor,
        normal: Optional[Tensor],
        log_length: Optional[Tensor],
        log_surface: Optional[Tensor],
        log_volume: Optional[Tensor],
        log_size: Optional[Tensor],
    ) -> Tensor:
        """
        Compute official SPT-compatible horizontal edge features (18D).

        Feature layout:
        [mean_off(3), std_off(3), mean_dist(1), angle_source(1), angle_target(1),
         centroid_dir(3), centroid_dist(1), normal_angle(1), log_length(1),
         log_surface(1), log_volume(1), log_size(1)]
        """
        e = edge_index.shape[1]
        if e == 0:
            return pos.new_zeros((0, 18))

        src, dst = edge_index[0], edge_index[1]
        delta = pos[dst] - pos[src]  # [E, 3]
        dist = delta.norm(dim=1, keepdim=True).clamp(min=1e-6)
        direction = torch.nan_to_num(delta / dist, nan=0.0).clamp(-1, 1)
        dist_sqrt = dist.sqrt()

        mean_off = delta
        std_off = torch.zeros_like(delta)
        mean_dist = dist_sqrt

        if normal is None:
            angle_source = pos.new_zeros((e, 1))
            angle_target = pos.new_zeros((e, 1))
            normal_angle = pos.new_zeros((e, 1))
        else:
            angle_source = (direction * normal[src]).sum(dim=1).abs().unsqueeze(1)
            angle_target = (direction * normal[dst]).sum(dim=1).abs().unsqueeze(1)
            normal_angle = (normal[src] * normal[dst]).sum(dim=1).abs().unsqueeze(1)

        if log_size is None:
            z = pos.new_zeros((e, 1))
            log_length_diff = z
            log_surface_diff = z
            log_volume_diff = z
            log_size_diff = z
        else:
            log_length_diff = (log_length[src] - log_length[dst]).unsqueeze(1)
            log_surface_diff = (log_surface[src] - log_surface[dst]).unsqueeze(1)
            log_volume_diff = (log_volume[src] - log_volume[dst]).unsqueeze(1)
            log_size_diff = (log_size[src] - log_size[dst]).unsqueeze(1)

        return torch.cat(
            [
                mean_off,         # 3
                std_off,          # 3
                mean_dist,        # 1
                angle_source,     # 1
                angle_target,     # 1
                direction,        # 3
                dist_sqrt,        # 1
                normal_angle,     # 1
                log_length_diff,  # 1
                log_surface_diff, # 1
                log_volume_diff,  # 1
                log_size_diff,    # 1
            ],
            dim=1,
        )


@MODELS.register_module()
class GreedyContourPriorPartitionSimple(nn.Module):
    """
    Simplified partition module for testing without torch-graph-components

    Uses a simple connected components approach instead of energy-based merging.
    Useful for debugging and when torch-graph-components is not available.
    """

    def __init__(
        self,
        k_adjacency: int = 10,
        grid_size: float = 0.1,
        num_levels: int = 3,
    ):
        super().__init__()
        self.k_adjacency = k_adjacency
        self.grid_size = grid_size
        self.num_levels = num_levels

    def forward(
        self,
        pos: Tensor,
        x: Tensor,
        offset: Tensor,
        batch: Optional[Tensor] = None,
        y: Optional[Tensor] = None,
    ) -> SuperpointHierarchy:
        """Simple grid-based partition"""
        device = pos.device
        num_points = pos.shape[0]

        # Build KNN graph for edge_index (needed for PartitionCriterion)
        try:
            from libs.pointops.functions import knn_query
            neighbor_idx, _ = knn_query(self.k_adjacency, pos, offset)
            edge_index = self._neighbor_idx_to_edge_index(neighbor_idx)
        except Exception:
            # Fallback: no edge_index
            edge_index = torch.zeros(2, 0, dtype=torch.long, device=device)

        # Prepare y histogram
        y_hist = None
        if y is not None:
            if y.dim() == 1:
                valid_mask = y >= 0
                num_classes = max(y[valid_mask].max().item() + 1, 1) if valid_mask.any() else 1
                y_hist = torch.zeros(num_points, num_classes, device=device)
                if valid_mask.any():
                    import torch.nn.functional as F
                    y_hist[valid_mask] = F.one_hot(y[valid_mask].long(), num_classes).float()
            else:
                y_hist = y.float()

        # Level 0: raw points
        data_list = [
            {
                "pos": pos,
                "x": x,
                "edge_index": edge_index,
                "node_size": torch.ones(num_points, device=device, dtype=torch.long),
                "super_index": None,
                "y": y_hist,
            }
        ]

        current_pos = pos
        current_x = x
        current_size = torch.ones(num_points, device=device, dtype=torch.long)
        current_y = y_hist

        for level in range(self.num_levels):
            grid = self.grid_size * (2**level)

            # Grid-based clustering
            grid_coord = torch.floor(current_pos / grid).long()
            # Unique grid cells
            _, super_index = torch.unique(
                grid_coord, dim=0, return_inverse=True
            )

            # Update previous level
            data_list[-1]["super_index"] = super_index

            # Aggregate to new level
            num_superpoints = super_index.max().item() + 1
            new_pos = scatter_mean(current_pos, super_index, dim=0)
            new_x = scatter_mean(current_x, super_index, dim=0)
            new_size = scatter_sum(current_size, super_index, dim=0)
            new_y = scatter_sum(current_y, super_index, dim=0) if current_y is not None else None

            sub = Cluster.from_super_index(super_index, current_pos.shape[0])

            data_list.append(
                {
                    "pos": new_pos,
                    "x": new_x,
                    "node_size": new_size,
                    "sub": sub,
                    "super_index": None,
                    "y": new_y,
                }
            )

            current_pos = new_pos
            current_x = new_x
            current_size = new_size
            current_y = new_y

        return SuperpointHierarchy(data_list)

    def _neighbor_idx_to_edge_index(self, neighbor_idx: Tensor) -> Tensor:
        """Convert KNN neighbor indices to edge_index format"""
        N, K = neighbor_idx.shape
        device = neighbor_idx.device

        src = torch.arange(N, device=device).unsqueeze(1).expand(N, K)
        src = src.reshape(-1)
        dst = neighbor_idx.reshape(-1)

        valid_mask = dst >= 0
        src = src[valid_mask]
        dst = dst[valid_mask]

        return torch.stack([src, dst], dim=0)
