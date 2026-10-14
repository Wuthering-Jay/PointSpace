"""Pure geometry observability primitives for CVRL section 4.3."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from pointspace.utils.surface_visibility import compute_surface_visibility


P_INV = 0
P_AMB = 1
P_VIS = 2
REGION_NAMES = {P_INV: "P_inv", P_AMB: "P_amb", P_VIS: "P_vis"}


def compute_q_geo(
    points: np.ndarray,
    surface_cell_size="auto",
    surface_radius="auto",
    surface_z_tolerance: float = 0.10,
    reference_resolution: float = 0.0,
    surface_oversample: int = 2,
    surface_num_offsets: int = 4,
    surface_adaptive_radius: bool = True,
    surface_max_radius_cells: int = 2,
    surface_slope_compensation: bool = True,
    surface_max_slope_degrees: float = 60.0,
) -> np.ndarray:
    """Return continuous top-down observability using geometry only."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape [N, 3]")
    result = compute_surface_visibility(
        points=points,
        return_number=None,
        surface_cell_size=surface_cell_size,
        surface_radius=surface_radius,
        surface_z_tolerance=surface_z_tolerance,
        reference_resolution=reference_resolution,
        surface_oversample=surface_oversample,
        surface_num_offsets=surface_num_offsets,
        surface_adaptive_radius=surface_adaptive_radius,
        surface_max_radius_cells=surface_max_radius_cells,
        surface_visibility_threshold=0.5,
        echo_decay=0.0,
        surface_slope_compensation=surface_slope_compensation,
        surface_max_slope_degrees=surface_max_slope_degrees,
        first_return_bonus=0.0,
    )
    q_geo = np.asarray(result.geometric_observability, dtype=np.float32)
    if q_geo.shape != (points.shape[0],):
        raise RuntimeError("Geometry visibility returned an invalid shape")
    if not np.isfinite(q_geo).all() or np.any((q_geo < 0) | (q_geo > 1)):
        raise RuntimeError("q_geo must be finite and bounded by [0, 1]")
    return q_geo


def partition_q_geo(
    q_geo: np.ndarray,
    invisible_max: float = 0.20,
    visible_min: float = 0.60,
) -> np.ndarray:
    """Map q_geo to non-overlapping P_inv/P_amb/P_vis regions."""
    if not 0 <= invisible_max < visible_min <= 1:
        raise ValueError(
            "Thresholds must satisfy 0 <= invisible_max < visible_min <= 1"
        )
    q_geo = np.asarray(q_geo, dtype=np.float32).reshape(-1)
    if not np.isfinite(q_geo).all():
        raise ValueError("q_geo contains non-finite values")
    region = np.full(q_geo.shape, P_AMB, dtype=np.uint8)
    region[q_geo <= invisible_max] = P_INV
    region[q_geo >= visible_min] = P_VIS
    return region


def estimate_height_above_ground(
    points: np.ndarray,
    classification: np.ndarray,
    ground_class: int = 2,
    cell_size: float = 5.0,
) -> np.ndarray:
    """Estimate height above a nearest-cell LAS ground surface."""
    points = np.asarray(points, dtype=np.float64)
    classification = np.asarray(classification).reshape(-1)
    if points.shape[0] != classification.size:
        raise ValueError("points/classification length mismatch")
    if points.shape[0] == 0:
        return np.empty(0, dtype=np.float32)
    ground = classification == int(ground_class)
    if not ground.any():
        baseline = float(np.percentile(points[:, 2], 2))
        return np.maximum(points[:, 2] - baseline, 0).astype(np.float32)

    origin = points[:, :2].min(axis=0)
    cell = np.floor((points[:, :2] - origin) / float(cell_size)).astype(np.int64)
    ground_cell = cell[ground]
    ground_z = points[ground, 2]
    unique_cell, inverse = np.unique(ground_cell, axis=0, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    sorted_inverse = inverse[order]
    boundaries = np.r_[0, np.flatnonzero(np.diff(sorted_inverse)) + 1, order.size]
    cell_ground = np.empty(unique_cell.shape[0], dtype=np.float64)
    for index in range(unique_cell.shape[0]):
        values = ground_z[order[boundaries[index] : boundaries[index + 1]]]
        cell_ground[index] = np.percentile(values, 10)

    tree = cKDTree(unique_cell.astype(np.float64))
    # File-level processing is already parallel. Avoid nested all-core pools.
    _, nearest = tree.query(cell.astype(np.float64), k=1, workers=1)
    height = points[:, 2] - cell_ground[np.asarray(nearest, dtype=np.int64)]
    return np.maximum(height, 0).astype(np.float32)
