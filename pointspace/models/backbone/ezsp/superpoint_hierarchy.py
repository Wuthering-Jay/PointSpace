"""
SuperpointHierarchy - Hierarchical Superpoint Graph Structure

This module implements a hierarchical superpoint data structure equivalent to
the NAG (Nested Attributed Graph) in the original SPT implementation.

It stores multi-level superpoint graphs, from raw points (Level 0) to
coarsest superpoints (Level L).

Author: PointSpace Team
"""

from typing import Dict, List, Optional, Union
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_scatter import scatter_sum, scatter_mean


class Cluster:
    """
    CSR-format Cluster Membership Storage

    Efficiently stores which points belong to which cluster using
    Compressed Sparse Row (CSR) format.

    For N points grouped into M clusters:
    - pointer: [M+1] Start position of each cluster in value array
    - value: [N] Point indices sorted by cluster membership

    Example:
        Cluster 0 contains points [0, 2, 5]
        Cluster 1 contains points [1, 3, 4]

        pointer = [0, 3, 6]
        value = [0, 2, 5, 1, 3, 4]

        To get cluster i's members: value[pointer[i]:pointer[i+1]]

    Attributes:
        pointer: Tensor [M+1] - Cluster start indices
        value: Tensor [N] - Point indices

    Methods:
        from_super_index: Create from super_index mapping
        __getitem__: Get members of cluster i
        num_clusters: Number of clusters
        to: Move to device
    """

    def __init__(self, pointer: Tensor, value: Tensor):
        """
        Initialize Cluster

        Args:
            pointer: [M+1] Start positions
            value: [N] Point indices
        """
        self.pointer = pointer
        self.value = value

    @classmethod
    def from_super_index(cls, super_index: Tensor, num_points: Optional[int] = None) -> "Cluster":
        """
        Create Cluster from super_index mapping

        Args:
            super_index: [N] Cluster assignment for each point
            num_points: Optional total number of points (for validation)

        Returns:
            Cluster object
        """
        device = super_index.device

        if num_points is not None:
            assert super_index.shape[0] == num_points

        num_clusters = super_index.max().item() + 1
        N = super_index.shape[0]

        # Count points per cluster
        sizes = torch.zeros(num_clusters, dtype=torch.long, device=device)
        sizes.scatter_add_(0, super_index, torch.ones(N, device=device, dtype=torch.long))

        # Build pointer (cumsum of sizes)
        pointer = torch.zeros(num_clusters + 1, dtype=torch.long, device=device)
        pointer[1:] = sizes.cumsum(0)

        # Build value (point indices sorted by cluster)
        # Sort points by their cluster assignment
        sorted_idx = super_index.argsort(stable=True)
        value = sorted_idx

        return cls(pointer, value)
    
    @classmethod
    def from_dict(cls, d: dict, device=None) -> "Cluster":
        """
        Create Cluster from dict (typically from GridSampling3D transform).
        
        Args:
            d: dict with 'pointer' and 'value' keys (numpy arrays or tensors)
            device: Optional device to place tensors
            
        Returns:
            Cluster object
        """
        pointer = d["pointer"]
        value = d["value"]
        
        if isinstance(pointer, (list, tuple)):
            import numpy as np
            pointer = np.array(pointer)
        if isinstance(value, (list, tuple)):
            import numpy as np
            value = np.array(value)
        
        if not isinstance(pointer, Tensor):
            pointer = torch.from_numpy(pointer).long()
        if not isinstance(value, Tensor):
            value = torch.from_numpy(value).long()
        
        if device is not None:
            pointer = pointer.to(device)
            value = value.to(device)
        
        return cls(pointer, value)
    
    def to_super_index(self) -> Tensor:
        """
        Convert Cluster to super_index format.
        
        Returns:
            super_index: [num_points] mapping each point to its cluster
        """
        device = self.pointer.device
        num_points = self.value.shape[0]
        num_clusters = self.num_clusters
        
        # Create cluster indices repeated by their sizes
        sizes = self.pointer[1:] - self.pointer[:-1]
        
        # Safety check: sizes should be non-negative
        if (sizes < 0).any():
            raise RuntimeError(
                f"Cluster has invalid pointer (negative sizes). "
                f"pointer shape: {self.pointer.shape}, "
                f"min size: {sizes.min().item()}, max size: {sizes.max().item()}"
            )
        
        cluster_idx = torch.arange(num_clusters, device=device)
        repeated = cluster_idx.repeat_interleave(sizes.long())
        
        # Map back to original point order using value as indices
        # value[i] tells us which original point index is at position i
        super_index = torch.empty(num_points, dtype=torch.long, device=device)
        super_index[self.value.long()] = repeated
        
        return super_index

    def __getitem__(self, idx: Union[int, Tensor]) -> Tensor:
        """
        Get members of cluster(s)

        Args:
            idx: int or Tensor - Cluster index/indices

        Returns:
            Tensor of point indices belonging to the cluster(s)
        """
        if isinstance(idx, int):
            start = self.pointer[idx].item()
            end = self.pointer[idx + 1].item()
            return self.value[start:end]
        else:
            # Batch indexing - return list of tensors
            results = []
            for i in idx.tolist():
                start = self.pointer[i].item()
                end = self.pointer[i + 1].item()
                results.append(self.value[start:end])
            return results

    @property
    def num_clusters(self) -> int:
        """Number of clusters"""
        return len(self.pointer) - 1

    def sizes(self) -> Tensor:
        """Get size of each cluster"""
        return self.pointer[1:] - self.pointer[:-1]

    def to(self, device) -> "Cluster":
        """Move to device"""
        return Cluster(self.pointer.to(device), self.value.to(device))

    def __repr__(self) -> str:
        return f"Cluster(num_clusters={self.num_clusters}, total_points={len(self.value)})"


class SuperpointLevel(dict):
    """
    Single Level of Superpoint Hierarchy

    Stores data for one level of the hierarchical partition.

    Standard Attributes:
        pos: [N_l, 3] - Superpoint positions (centroids)
        x: [N_l, C] - Superpoint features
        node_size: [N_l] - Number of points in each superpoint
        edge_index: [2, E_l] - Graph edges between superpoints
        edge_attr: [E_l] - Edge attributes/weights
        super_index: [N_l] - Mapping to next (coarser) level
        sub: Cluster - Indices of children from previous (finer) level
        y: [N_l, num_classes] - Label histogram (optional)
        batch: [N_l] - Batch index for each superpoint (optional)

    Properties:
        num_points: Number of superpoints in this level
        num_edges: Number of edges in the graph
        device: Device of tensors
    """

    @property
    def num_points(self) -> int:
        """Number of superpoints/nodes in this level"""
        if "pos" in self:
            return self["pos"].shape[0]
        elif "x" in self:
            return self["x"].shape[0]
        return 0

    @property
    def num_edges(self) -> int:
        """Number of edges in graph"""
        if "edge_index" in self and self["edge_index"] is not None:
            return self["edge_index"].shape[1]
        return 0

    @property
    def device(self):
        """Device of tensors"""
        for v in self.values():
            if isinstance(v, Tensor):
                return v.device
        return torch.device("cpu")

    def to(self, device) -> "SuperpointLevel":
        """Move all tensors to device"""
        new_level = SuperpointLevel()
        for k, v in self.items():
            if isinstance(v, Tensor):
                new_level[k] = v.to(device)
            elif isinstance(v, Cluster):
                new_level[k] = v.to(device)
            else:
                new_level[k] = v
        return new_level


class SuperpointHierarchy:
    """
    Hierarchical Superpoint Graph Structure (NAG equivalent)

    Stores multi-level superpoint graphs from raw points to coarsest partition.

    Structure:
        Level 0: Raw points
        Level 1: First level superpoints (aggregated from Level 0)
        Level 2: Second level superpoints (aggregated from Level 1)
        ...
        Level L: Coarsest superpoints

    Index Relationships:
        super_index[l]: Maps Level l elements to their Level l+1 superpoint
        sub[l]: Cluster object mapping Level l superpoints to Level l-1 children

    Attributes:
        levels: List[SuperpointLevel] - Data for each level
        num_levels: int - Number of levels

    Methods:
        __getitem__: Get level by index
        __len__: Number of levels
        to: Move to device
        get_level_ratios: Compute compression ratios between levels
        propagate_labels_to_points: Propagate predictions from coarse to fine
    """

    def __init__(self, data_list: List[Dict]):
        """
        Initialize from list of level data dictionaries

        Args:
            data_list: List of dicts, one per level, containing level attributes
        """
        self.levels = [SuperpointLevel(d) for d in data_list]

    @property
    def num_levels(self) -> int:
        """Number of levels in hierarchy"""
        return len(self.levels)

    def __getitem__(self, idx: int) -> SuperpointLevel:
        """Get level by index (-1 for last level)"""
        return self.levels[idx]

    def get_level(self, idx: int) -> SuperpointLevel:
        """
        Get level by index (NAG compatibility alias for __getitem__)

        Args:
            idx: Level index (0 = points, 1+ = superpoints)

        Returns:
            SuperpointLevel at given index
        """
        return self.levels[idx]

    def __len__(self) -> int:
        """Number of levels"""
        return self.num_levels

    def __iter__(self):
        """Iterate over levels"""
        return iter(self.levels)

    @property
    def device(self):
        """Device of hierarchy"""
        return self.levels[0].device

    def to(self, device) -> "SuperpointHierarchy":
        """Move all levels to device"""
        new_levels = [level.to(device) for level in self.levels]
        # Reconstruct with moved data
        new_hierarchy = SuperpointHierarchy.__new__(SuperpointHierarchy)
        new_hierarchy.levels = new_levels
        return new_hierarchy

    def get_level_ratios(self) -> List[float]:
        """
        Compute compression ratios between consecutive levels

        Returns:
            List of ratios: [N_0/N_1, N_1/N_2, ..., N_{L-1}/N_L]
        """
        ratios = []
        for i in range(1, self.num_levels):
            n_fine = self.levels[i - 1].num_points
            n_coarse = max(self.levels[i].num_points, 1)
            ratios.append(n_fine / n_coarse)
        return ratios

    def propagate_labels_to_points(
        self,
        level_preds: Tensor,
        from_level: int = -1,
        to_level: int = 0,
    ) -> Tensor:
        """
        Propagate predictions from a coarse level down to a finer level

        Args:
            level_preds: [N_from, C] Predictions at from_level
            from_level: int - Source level index (-1 for last)
            to_level: int - Target level index (0 for raw points)

        Returns:
            Tensor [N_to, C] - Predictions at target level
        """
        if from_level < 0:
            from_level = self.num_levels + from_level

        if to_level < 0:
            to_level = self.num_levels + to_level

        assert from_level > to_level, "Can only propagate from coarse to fine"

        preds = level_preds

        # Propagate from coarse to fine through super_index
        for l in range(from_level, to_level, -1):
            super_index = self.levels[l - 1].get("super_index")
            if super_index is None:
                raise ValueError(f"Level {l-1} missing super_index for propagation")
            preds = preds[super_index]

        return preds

    def aggregate_labels_to_superpoints(
        self,
        point_labels: Tensor,
        to_level: int = 1,
        mode: str = "sum",
    ) -> Tensor:
        """
        Aggregate point labels up to a superpoint level

        Args:
            point_labels: [N_0, C] Labels at Level 0 (raw points)
            to_level: int - Target superpoint level
            mode: str - 'sum' | 'mean' aggregation mode

        Returns:
            Tensor [N_to, C] - Aggregated labels
        """
        labels = point_labels

        for l in range(to_level):
            super_index = self.levels[l].get("super_index")
            if super_index is None:
                raise ValueError(f"Level {l} missing super_index for aggregation")

            if mode == "sum":
                labels = scatter_sum(labels, super_index, dim=0)
            else:
                labels = scatter_mean(labels, super_index, dim=0)

        return labels

    def get_semantic_oracle(self, level: int = 1, num_classes: int = None) -> Tensor:
        """
        Compute oracle labels for superpoints (majority vote)

        Used to evaluate partition quality.

        Args:
            level: int - Superpoint level
            num_classes: int - Number of classes (inferred if None)

        Returns:
            Tensor [N_level] - Oracle labels for each superpoint
        """
        level_data = self.levels[level]
        y = level_data.get("y")

        if y is None:
            raise ValueError(f"Level {level} missing 'y' (label histogram)")

        if num_classes is not None:
            y = y[:, :num_classes]

        return y.argmax(dim=1)

    def summary(self) -> str:
        """Get string summary of hierarchy"""
        lines = [f"SuperpointHierarchy with {self.num_levels} levels:"]
        for i, level in enumerate(self.levels):
            lines.append(
                f"  Level {i}: {level.num_points} nodes, {level.num_edges} edges"
            )
        ratios = self.get_level_ratios()
        lines.append(f"  Compression ratios: {[f'{r:.1f}' for r in ratios]}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()
