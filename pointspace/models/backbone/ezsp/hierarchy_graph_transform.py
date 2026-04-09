"""
Online hierarchy graph transforms for EZ-SP semantic training.

These transforms operate on the partitioned SuperpointHierarchy directly,
which keeps graph construction on the model side instead of moving it back
to the CPU dataloader path.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn

from pointspace.models.builder import MODELS
from pointspace.models.backbone.ezsp.graph_partition import (
    make_horizontal_graph_bidirectional,
)
from pointspace.models.backbone.ezsp.superpoint_hierarchy import (
    Cluster,
    SuperpointHierarchy,
    SuperpointLevel,
)


def _expand_level_param(
    value: Union[int, float, Sequence[Union[int, float]]],
    num_levels: int,
) -> List[Union[int, float]]:
    if isinstance(value, (list, tuple)):
        out = list(value)
        if len(out) >= num_levels:
            return out[:num_levels]
        return out + [out[-1]] * (num_levels - len(out))
    return [value] * num_levels


def _truncated_noise_like(x: Tensor, sigma: float, trunc: float) -> Tensor:
    if sigma <= 0:
        return torch.zeros_like(x)
    if trunc > 0:
        return torch.nn.init.trunc_normal_(
            torch.empty_like(x),
            mean=0.0,
            std=sigma,
            a=-trunc,
            b=trunc,
        )
    return torch.randn_like(x) * sigma


def _recompute_offset(batch: Optional[Tensor]) -> Optional[Tensor]:
    if batch is None or batch.numel() == 0:
        return None
    num_batches = int(batch.max().item()) + 1
    counts = torch.bincount(batch.long(), minlength=num_batches)
    return counts.cumsum(0)


@MODELS.register_module()
class HierarchyGraphTransform(nn.Module):
    """Lightweight online NAG-style graph augmentations for PointSpace EZ-SP."""

    def __init__(
        self,
        enabled: bool = True,
        training_only: bool = True,
        apply_levels: str = "1+",
        max_nodes: Union[int, Sequence[int]] = 0,
        max_edges: Union[int, Sequence[int]] = 0,
        n_min_edges: Union[int, Sequence[int]] = 0,
        n_max_edges: Union[int, Sequence[int]] = 0,
        add_self_loops: bool = False,
        pos_jitter_std: Union[float, Sequence[float]] = 0.0,
        pos_jitter_trunc: Union[float, Sequence[float]] = 0.0,
        edge_attr_jitter_std: Union[float, Sequence[float]] = 0.0,
        edge_attr_jitter_trunc: Union[float, Sequence[float]] = 0.0,
    ):
        super().__init__()
        self.enabled = enabled
        self.training_only = training_only
        self.apply_levels = apply_levels
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.n_min_edges = n_min_edges
        self.n_max_edges = n_max_edges
        self.add_self_loops = add_self_loops
        self.pos_jitter_std = pos_jitter_std
        self.pos_jitter_trunc = pos_jitter_trunc
        self.edge_attr_jitter_std = edge_attr_jitter_std
        self.edge_attr_jitter_trunc = edge_attr_jitter_trunc

    def forward(self, hierarchy: SuperpointHierarchy) -> SuperpointHierarchy:
        if not self.enabled:
            return hierarchy
        if self.training_only and not self.training:
            return hierarchy

        out = self._clone_hierarchy(hierarchy)
        level_ids = self._resolve_levels(out.num_levels)
        max_nodes = _expand_level_param(self.max_nodes, out.num_levels)
        max_edges = _expand_level_param(self.max_edges, out.num_levels)
        n_min_edges = _expand_level_param(self.n_min_edges, out.num_levels)
        n_max_edges = _expand_level_param(self.n_max_edges, out.num_levels)

        for level_idx in level_ids:
            num_nodes = int(max_nodes[level_idx])
            if num_nodes > 0 and out[level_idx].num_points > num_nodes:
                out = self._restrict_level_nodes(out, level_idx, num_nodes)

        for level_idx in level_ids:
            self._materialize_bidirectional_edges(out[level_idx])
            self._sample_edges_by_source(
                out[level_idx],
                int(n_min_edges[level_idx]),
                int(n_max_edges[level_idx]),
            )
            self._restrict_level_edges(out[level_idx], int(max_edges[level_idx]))
            if self.add_self_loops:
                self._add_self_loops(out[level_idx])

        pos_jitter_std = _expand_level_param(self.pos_jitter_std, out.num_levels)
        pos_jitter_trunc = _expand_level_param(self.pos_jitter_trunc, out.num_levels)
        edge_jitter_std = _expand_level_param(self.edge_attr_jitter_std, out.num_levels)
        edge_jitter_trunc = _expand_level_param(self.edge_attr_jitter_trunc, out.num_levels)
        for level_idx in level_ids:
            self._jitter_level(
                out[level_idx],
                pos_sigma=float(pos_jitter_std[level_idx]),
                pos_trunc=float(pos_jitter_trunc[level_idx]),
                edge_sigma=float(edge_jitter_std[level_idx]),
                edge_trunc=float(edge_jitter_trunc[level_idx]),
            )

        return out

    def _materialize_bidirectional_edges(self, level: SuperpointLevel) -> None:
        if level.get("edge_index") is None:
            return
        edge_index, edge_attr = make_horizontal_graph_bidirectional(
            level["edge_index"],
            level.get("edge_attr"),
        )
        level["edge_index"] = edge_index
        if edge_attr is not None:
            level["edge_attr"] = edge_attr

    def _resolve_levels(self, num_levels: int) -> List[int]:
        if isinstance(self.apply_levels, int):
            return [self.apply_levels]
        if self.apply_levels == "1+":
            return list(range(1, num_levels))
        if self.apply_levels == "all":
            return list(range(num_levels))
        if self.apply_levels == "0+":
            return list(range(num_levels))
        raise ValueError(f"Unsupported apply_levels={self.apply_levels}")

    def _clone_hierarchy(self, hierarchy: SuperpointHierarchy) -> SuperpointHierarchy:
        levels: List[SuperpointLevel] = []
        for level in hierarchy.levels:
            cloned = SuperpointLevel()
            for key, value in level.items():
                if isinstance(value, Tensor):
                    cloned[key] = value.clone()
                elif isinstance(value, Cluster):
                    cloned[key] = Cluster(value.pointer.clone(), value.value.clone())
                else:
                    cloned[key] = value
            levels.append(cloned)
        return SuperpointHierarchy(levels)

    def _restrict_level_nodes(
        self,
        hierarchy: SuperpointHierarchy,
        level_idx: int,
        num_nodes: int,
    ) -> SuperpointHierarchy:
        device = hierarchy.device
        weights = torch.ones(hierarchy[level_idx].num_points, device=device)
        selected = torch.multinomial(weights, num_nodes, replacement=False)
        keep_masks = self._expand_keep_masks(hierarchy, level_idx, selected)
        return self._apply_keep_masks(hierarchy, keep_masks)

    def _expand_keep_masks(
        self,
        hierarchy: SuperpointHierarchy,
        level_idx: int,
        selected: Tensor,
    ) -> List[Tensor]:
        keep_masks: List[Optional[Tensor]] = [None] * hierarchy.num_levels
        mask = torch.zeros(hierarchy[level_idx].num_points, dtype=torch.bool, device=hierarchy.device)
        mask[selected] = True
        keep_masks[level_idx] = mask

        for child_level in range(level_idx - 1, -1, -1):
            super_index = hierarchy[child_level].get("super_index")
            if super_index is None:
                keep_masks[child_level] = torch.ones(
                    hierarchy[child_level].num_points,
                    dtype=torch.bool,
                    device=hierarchy.device,
                )
                continue
            keep_masks[child_level] = keep_masks[child_level + 1][super_index]

        for current_level in range(level_idx, hierarchy.num_levels - 1):
            super_index = hierarchy[current_level].get("super_index")
            if super_index is None:
                keep_masks[current_level + 1] = torch.ones(
                    hierarchy[current_level + 1].num_points,
                    dtype=torch.bool,
                    device=hierarchy.device,
                )
                continue
            parent_mask = torch.zeros(
                hierarchy[current_level + 1].num_points,
                dtype=torch.bool,
                device=hierarchy.device,
            )
            parent_mask[torch.unique(super_index[keep_masks[current_level]])] = True
            keep_masks[current_level + 1] = parent_mask

        return [m if m is not None else torch.ones(
            hierarchy[i].num_points, dtype=torch.bool, device=hierarchy.device
        ) for i, m in enumerate(keep_masks)]

    def _apply_keep_masks(
        self,
        hierarchy: SuperpointHierarchy,
        keep_masks: List[Tensor],
    ) -> SuperpointHierarchy:
        old_to_new: List[Tensor] = []
        keep_indices: List[Tensor] = []
        for level_idx, level in enumerate(hierarchy.levels):
            keep_idx = keep_masks[level_idx].nonzero(as_tuple=False).view(-1)
            keep_indices.append(keep_idx)
            mapping = torch.full(
                (level.num_points,),
                -1,
                dtype=torch.long,
                device=hierarchy.device,
            )
            mapping[keep_idx] = torch.arange(keep_idx.numel(), device=hierarchy.device)
            old_to_new.append(mapping)

        new_levels: List[SuperpointLevel] = []
        for level_idx, level in enumerate(hierarchy.levels):
            keep_idx = keep_indices[level_idx]
            keep_mask = keep_masks[level_idx]
            new_level = SuperpointLevel()
            edge_mask = None
            if level.get("edge_index") is not None:
                src, dst = level["edge_index"][0], level["edge_index"][1]
                edge_mask = keep_mask[src] & keep_mask[dst]
                new_level["edge_index"] = torch.stack(
                    [old_to_new[level_idx][src[edge_mask]], old_to_new[level_idx][dst[edge_mask]]],
                    dim=0,
                )

            for key, value in level.items():
                if key in {"edge_index", "sub"}:
                    continue
                if isinstance(value, Tensor):
                    if key == "edge_attr" or (edge_mask is not None and value.shape[0] == level.num_edges):
                        if edge_mask is not None:
                            new_level[key] = value[edge_mask]
                        continue
                    if value.dim() > 0 and value.shape[0] == level.num_points:
                        new_level[key] = value[keep_idx]
                    else:
                        new_level[key] = value.clone()
                elif isinstance(value, Cluster):
                    continue
                else:
                    new_level[key] = value

            if level_idx < hierarchy.num_levels - 1 and level.get("super_index") is not None:
                mapped = old_to_new[level_idx + 1][level["super_index"][keep_idx]]
                new_level["super_index"] = mapped

            if "batch" in new_level and isinstance(new_level["batch"], Tensor):
                new_level["offset"] = _recompute_offset(new_level["batch"])
            elif "offset" in new_level and new_level["offset"] is not None:
                new_level["offset"] = None

            new_levels.append(new_level)

        for level_idx in range(1, len(new_levels)):
            prev_level = new_levels[level_idx - 1]
            if prev_level.get("super_index") is not None:
                new_levels[level_idx]["sub"] = Cluster.from_super_index(
                    prev_level["super_index"],
                    num_points=prev_level.num_points,
                )

        return SuperpointHierarchy(new_levels)

    def _sample_edges_by_source(
        self,
        level: SuperpointLevel,
        n_min: int,
        n_max: int,
    ) -> None:
        if level.get("edge_index") is None or n_max <= 0 or level.num_edges == 0:
            return
        src = level["edge_index"][0]
        keep = []
        for node_id in torch.unique(src):
            edge_ids = (src == node_id).nonzero(as_tuple=False).view(-1)
            num_edges = edge_ids.numel()
            target = min(num_edges, n_max)
            if n_min > 0:
                target = max(min(num_edges, n_min), target)
            if target >= num_edges:
                keep.append(edge_ids)
                continue
            weights = torch.ones(num_edges, device=edge_ids.device)
            chosen = torch.multinomial(weights, target, replacement=False)
            keep.append(edge_ids[chosen])
        if not keep:
            return
        edge_keep = torch.unique(torch.cat(keep, dim=0), sorted=True)
        self._apply_edge_subset(level, edge_keep)

    def _restrict_level_edges(self, level: SuperpointLevel, num_edges: int) -> None:
        if level.get("edge_index") is None or num_edges <= 0 or level.num_edges <= num_edges:
            return
        weights = torch.ones(level.num_edges, device=level.device)
        keep = torch.multinomial(weights, num_edges, replacement=False)
        keep = torch.sort(keep).values
        self._apply_edge_subset(level, keep)

    def _apply_edge_subset(self, level: SuperpointLevel, keep: Tensor) -> None:
        level["edge_index"] = level["edge_index"][:, keep]
        for key in ["edge_attr", "edge_weight"]:
            value = level.get(key)
            if isinstance(value, Tensor):
                level[key] = value[keep]

    def _add_self_loops(self, level: SuperpointLevel) -> None:
        if level.get("edge_index") is None:
            return
        device = level.device
        num_nodes = level.num_points
        edge_index = level["edge_index"]
        existing = edge_index[0] == edge_index[1]
        loop_nodes = edge_index[0][existing] if existing.any() else torch.empty(0, device=device, dtype=torch.long)
        missing_mask = torch.ones(num_nodes, dtype=torch.bool, device=device)
        if loop_nodes.numel() > 0:
            missing_mask[loop_nodes] = False
        missing = missing_mask.nonzero(as_tuple=False).view(-1)
        if missing.numel() == 0:
            return
        loops = torch.stack([missing, missing], dim=0)
        level["edge_index"] = torch.cat([edge_index, loops], dim=1)
        if isinstance(level.get("edge_attr"), Tensor):
            pad = level["edge_attr"].new_zeros((missing.numel(), level["edge_attr"].shape[1]))
            level["edge_attr"] = torch.cat([level["edge_attr"], pad], dim=0)
        if isinstance(level.get("edge_weight"), Tensor):
            pad = level["edge_weight"].new_zeros((missing.numel(),) + level["edge_weight"].shape[1:])
            level["edge_weight"] = torch.cat([level["edge_weight"], pad], dim=0)

    def _jitter_level(
        self,
        level: SuperpointLevel,
        pos_sigma: float,
        pos_trunc: float,
        edge_sigma: float,
        edge_trunc: float,
    ) -> None:
        if pos_sigma > 0 and isinstance(level.get("pos"), Tensor):
            level["pos"] = level["pos"] + _truncated_noise_like(level["pos"], pos_sigma, pos_trunc)
        if edge_sigma > 0 and isinstance(level.get("edge_attr"), Tensor):
            level["edge_attr"] = level["edge_attr"] + _truncated_noise_like(
                level["edge_attr"], edge_sigma, edge_trunc
            )
