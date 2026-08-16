"""Shared encoder hierarchy utilities for HPSD backbones."""

from dataclasses import dataclass

import torch

from pointspace.models.utils.structure import Point


@dataclass(frozen=True)
class HierarchyLevel:
    """One encoder level and the mapping from input points to its tokens."""

    point: Point
    input_to_level: torch.Tensor
    level: int


def build_encoder_hierarchy(deepest_point, expected_levels=None):
    """Build a fine-to-coarse hierarchy without mutating the pooling trace."""
    coarse_to_fine = []
    point = deepest_point
    visited = set()
    while True:
        object_id = id(point)
        if object_id in visited:
            raise RuntimeError("Cycle detected in pooling_parent hierarchy")
        visited.add(object_id)
        coarse_to_fine.append(point)
        if "pooling_parent" not in point.keys():
            break
        point = point.pooling_parent

    points = list(reversed(coarse_to_fine))
    if expected_levels is not None and len(points) != int(expected_levels):
        raise RuntimeError(
            f"Expected {expected_levels} encoder levels, found {len(points)}. "
            "HPSD backbones require encoder-only mode and traceable pooling."
        )

    num_input_points = int(points[0].feat.shape[0])
    input_to_level = torch.arange(
        num_input_points, device=points[0].feat.device, dtype=torch.long
    )
    hierarchy = [
        HierarchyLevel(point=points[0], input_to_level=input_to_level, level=0)
    ]

    input_batch = points[0].batch
    for level, child in enumerate(points[1:], start=1):
        if "pooling_inverse" not in child.keys():
            raise RuntimeError(f"Encoder level {level} has no pooling_inverse")
        inverse = child.pooling_inverse.long()
        if inverse.shape[0] != points[level - 1].feat.shape[0]:
            raise RuntimeError(
                f"Invalid pooling_inverse length at encoder level {level}: "
                f"got {inverse.shape[0]}, expected {points[level - 1].feat.shape[0]}"
            )
        input_to_level = inverse[input_to_level]
        if input_to_level.numel() > 0:
            if int(input_to_level.min()) < 0 or int(input_to_level.max()) >= len(child.feat):
                raise RuntimeError(f"Out-of-range input mapping at encoder level {level}")
            if not torch.equal(child.batch[input_to_level], input_batch):
                raise RuntimeError(f"Encoder level {level} mapping crosses batch samples")
        hierarchy.append(
            HierarchyLevel(
                point=child,
                input_to_level=input_to_level,
                level=level,
            )
        )
    return tuple(hierarchy)

