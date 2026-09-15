"""Fast adaptive orthographic visibility for irregular ALS point clouds."""

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
from scipy.ndimage import maximum_filter


@dataclass(frozen=True)
class SurfaceVisibilityResult:
    """Hard visibility and its continuous geometric/return confidence."""

    visible: np.ndarray
    observability: np.ndarray
    geometric_observability: np.ndarray
    return_observability: np.ndarray
    cell_size: float
    base_radius: float
    width: int
    height: int


def _is_auto(value):
    return value is None or (
        isinstance(value, str) and value.lower() == "auto"
    )


def _validate_auto_number(value, name, allow_zero):
    if _is_auto(value):
        return
    if isinstance(value, str):
        raise ValueError(f"{name} must be None, 'auto', or a number")
    value = float(value)
    if value < 0 or (value == 0 and not allow_zero):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {qualifier}")


def _coarse_local_spacing(points, density_mask=None):
    """Estimate per-point XY spacing with one linear coarse-grid pass."""
    xy_min = np.min(points[:, :2], axis=0)
    extent = np.maximum(np.max(points[:, :2], axis=0) - xy_min, 0.0)
    if len(points) <= 1 or np.any(extent <= 0):
        return np.ones(len(points), dtype=np.float32), 1.0

    target_bins = int(np.clip(np.sqrt(len(points) / 256.0), 8, 128))
    bin_width = extent[0] / target_bins
    bin_height = extent[1] / target_bins
    columns = np.floor(
        (points[:, 0] - xy_min[0]) / bin_width
    ).astype(np.int64)
    rows = np.floor(
        (points[:, 1] - xy_min[1]) / bin_height
    ).astype(np.int64)
    np.clip(columns, 0, target_bins - 1, out=columns)
    np.clip(rows, 0, target_bins - 1, out=rows)
    flat = rows * target_bins + columns

    if density_mask is None or not np.any(density_mask):
        density_mask = np.ones(len(points), dtype=bool)
    counts = np.bincount(
        flat[density_mask], minlength=target_bins * target_bins
    )
    occupied = counts[counts > 0]
    fallback_count = float(np.median(occupied)) if occupied.size else 1.0
    local_count = counts[flat].astype(np.float64)
    local_count[local_count <= 0] = fallback_count
    spacing = np.sqrt((bin_width * bin_height) / local_count)
    global_spacing = float(np.median(spacing[density_mask]))
    if not np.isfinite(global_spacing) or global_spacing <= 0:
        global_spacing = 1.0
    return spacing.astype(np.float32), global_spacing


def _disk_footprint(radius):
    axis = np.arange(-radius, radius + 1)
    row, column = np.meshgrid(axis, axis, indexing="ij")
    return row * row + column * column <= radius * radius


def _slope_compensated_max_filter(
    seed,
    radius,
    cell_size,
    max_slope_tangent,
):
    """Dilate a height grid while allowing plausible sloped surfaces.

    A raw max filter propagates a ridge height horizontally and therefore
    treats lower samples on the same pitched roof as occluded. Here each
    propagated candidate loses ``tan(theta) * horizontal_distance``. Vertical
    discontinuities remain much larger than this allowance and still occlude
    facade/interior points.
    """
    if radius <= 0:
        return seed
    height, width = seed.shape
    result = seed.copy()
    for delta_row in range(-radius, radius + 1):
        for delta_column in range(-radius, radius + 1):
            distance_cells = np.hypot(delta_row, delta_column)
            if distance_cells == 0 or distance_cells > radius:
                continue
            if delta_row >= 0:
                source_rows = slice(0, height - delta_row)
                target_rows = slice(delta_row, height)
            else:
                source_rows = slice(-delta_row, height)
                target_rows = slice(0, height + delta_row)
            if delta_column >= 0:
                source_columns = slice(0, width - delta_column)
                target_columns = slice(delta_column, width)
            else:
                source_columns = slice(-delta_column, width)
                target_columns = slice(0, width + delta_column)
            height_allowance = (
                max_slope_tangent * cell_size * distance_cells
            )
            target = result[target_rows, target_columns]
            candidate = (
                seed[source_rows, source_columns] - height_allowance
            )
            np.maximum(target, candidate, out=target)
    return result


def compute_surface_visibility(
    points: np.ndarray,
    return_number: Optional[np.ndarray] = None,
    surface_cell_size: Optional[Union[float, str]] = "auto",
    surface_radius: Optional[Union[float, str]] = "auto",
    surface_z_tolerance: float = 0.15,
    reference_resolution: float = 0.0,
    surface_oversample: int = 2,
    surface_num_offsets: int = 4,
    surface_adaptive_radius: bool = True,
    surface_max_radius_cells: int = 2,
    surface_visibility_threshold: float = 0.25,
    echo_decay: float = 0.7,
    surface_slope_compensation: bool = True,
    surface_max_slope_degrees: float = 60.0,
    first_return_bonus: float = 0.15,
    max_grid_cells: int = 25000000,
) -> SurfaceVisibilityResult:
    """Compute multi-offset adaptive-splat visibility and observability.

    Local density is estimated from first/single returns when available.
    Geometry confidence is averaged over shifted Z-buffers and multiplied by
    ``exp(-echo_decay * (return_number - 1))``. First returns then receive the
    bounded prior ``q <- q + first_return_bonus * (1 - q)``. The hard mask is
    obtained only once, from
    ``observability >= surface_visibility_threshold``.
    """
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape [N, 3]")
    if not np.isfinite(points).all():
        raise ValueError("point coordinates must all be finite")
    _validate_auto_number(surface_cell_size, "surface_cell_size", False)
    _validate_auto_number(surface_radius, "surface_radius", True)
    if surface_z_tolerance < 0:
        raise ValueError("surface_z_tolerance must be non-negative")
    if reference_resolution < 0 or echo_decay < 0:
        raise ValueError("reference_resolution and echo_decay must be non-negative")
    if not isinstance(surface_oversample, int) or surface_oversample < 1:
        raise ValueError("surface_oversample must be a positive integer")
    if surface_num_offsets not in {1, 4}:
        raise ValueError("surface_num_offsets must be 1 or 4")
    if not isinstance(surface_max_radius_cells, int) or surface_max_radius_cells < 0:
        raise ValueError("surface_max_radius_cells must be a non-negative integer")
    if not 0 <= surface_visibility_threshold <= 1:
        raise ValueError("surface_visibility_threshold must be in [0, 1]")
    if not isinstance(surface_slope_compensation, (bool, np.bool_)):
        raise TypeError("surface_slope_compensation must be a bool")
    if not 0 <= surface_max_slope_degrees < 90:
        raise ValueError("surface_max_slope_degrees must be in [0, 90)")
    if not 0 <= first_return_bonus < 1:
        raise ValueError("first_return_bonus must be in [0, 1)")
    if max_grid_cells < 1:
        raise ValueError("max_grid_cells must be positive")
    if len(points) == 0:
        empty_float = np.empty(0, dtype=np.float32)
        return SurfaceVisibilityResult(
            visible=np.empty(0, dtype=bool),
            observability=empty_float,
            geometric_observability=empty_float.copy(),
            return_observability=empty_float.copy(),
            cell_size=0.0,
            base_radius=0.0,
            width=0,
            height=0,
        )

    if return_number is not None:
        return_number = np.asarray(return_number).reshape(-1)
        if return_number.shape[0] != len(points):
            raise ValueError("return_number and points must have the same length")
        if not np.isfinite(return_number).all():
            raise ValueError("return_number must be finite")
        density_mask = return_number == 1
    else:
        density_mask = None
    local_spacing, global_spacing = _coarse_local_spacing(
        points, density_mask=density_mask
    )

    nominal_cell_size = (
        max(float(reference_resolution), global_spacing)
        if _is_auto(surface_cell_size)
        else float(surface_cell_size)
    )
    if nominal_cell_size <= 0:
        nominal_cell_size = 1.0
    cell_size = nominal_cell_size / surface_oversample
    xy_min = np.min(points[:, :2], axis=0)
    extent = np.maximum(np.max(points[:, :2], axis=0) - xy_min, 0.0)

    def grid_shape(candidate_cell_size):
        return (
            max(1, int(np.floor(extent[0] / candidate_cell_size)) + 2),
            max(1, int(np.floor(extent[1] / candidate_cell_size)) + 2),
        )

    width, height = grid_shape(cell_size)
    if width * height > max_grid_cells:
        # Binary search also handles near-degenerate strips where area-based
        # scaling fails because one XY extent is zero or extremely small.
        lower = cell_size
        upper = max(float(np.max(extent)), cell_size)
        while np.prod(grid_shape(upper), dtype=np.int64) > max_grid_cells:
            upper *= 2.0
        for _ in range(48):
            middle = (lower + upper) * 0.5
            candidate_width, candidate_height = grid_shape(middle)
            if candidate_width * candidate_height > max_grid_cells:
                lower = middle
            else:
                upper = middle
        cell_size = upper
        width, height = grid_shape(cell_size)

    base_radius = (
        cell_size if _is_auto(surface_radius) else float(surface_radius)
    )
    if surface_adaptive_radius and base_radius > 0:
        adaptive_radius = base_radius * local_spacing / global_spacing
    else:
        adaptive_radius = np.full(len(points), base_radius, dtype=np.float32)
    radius_cells = np.rint(adaptive_radius / cell_size).astype(np.int16)
    np.clip(
        radius_cells,
        0,
        surface_max_radius_cells,
        out=radius_cells,
    )

    offsets = (
        ((0.0, 0.0),)
        if surface_num_offsets == 1
        else ((0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5))
    )
    geometric_sum = np.zeros(len(points), dtype=np.float32)
    point_z = points[:, 2].astype(np.float32, copy=False)
    max_slope_tangent = float(
        np.tan(np.deg2rad(surface_max_slope_degrees))
    )
    for column_offset, row_offset in offsets:
        columns = np.floor(
            (points[:, 0] - xy_min[0]) / cell_size + column_offset
        ).astype(np.int64)
        rows = np.floor(
            (points[:, 1] - xy_min[1]) / cell_size + row_offset
        ).astype(np.int64)
        np.clip(columns, 0, width - 1, out=columns)
        np.clip(rows, 0, height - 1, out=rows)
        flat_indices = rows * width + columns
        surface = np.full(width * height, -np.inf, dtype=np.float32)

        for radius_cell in np.unique(radius_cells):
            mask = radius_cells == radius_cell
            seed = np.full(width * height, -np.inf, dtype=np.float32)
            np.maximum.at(seed, flat_indices[mask], point_z[mask])
            seed = seed.reshape(height, width)
            if radius_cell > 0:
                if surface_slope_compensation:
                    seed = _slope_compensated_max_filter(
                        seed,
                        int(radius_cell),
                        cell_size,
                        max_slope_tangent,
                    )
                else:
                    seed = maximum_filter(
                        seed,
                        footprint=_disk_footprint(int(radius_cell)),
                        mode="constant",
                        cval=-np.inf,
                    )
            np.maximum(surface, seed.reshape(-1), out=surface)

        surface_z = surface[flat_indices]
        height_gap = np.maximum(surface_z - point_z, 0.0)
        if surface_z_tolerance > 0:
            # A first return exactly ``surface_z_tolerance`` below the splat
            # surface has confidence 0.25, matching the default hard cutoff.
            score = np.exp(
                np.log(0.25) * (height_gap / surface_z_tolerance) ** 2
            ).astype(np.float32)
        else:
            score = (height_gap <= 1e-6).astype(np.float32)
        geometric_sum += score

    geometric = geometric_sum / float(len(offsets))
    if return_number is None:
        return_observability = np.ones(len(points), dtype=np.float32)
    else:
        echo_depth = np.maximum(return_number.astype(np.float32) - 1.0, 0.0)
        return_observability = np.exp(-echo_decay * echo_depth).astype(np.float32)
    observability = geometric * return_observability
    if return_number is not None and first_return_bonus > 0:
        # First/single returns are strong surface evidence, but the bounded
        # residual form cannot turn a clearly occluded point directly visible.
        first_return = return_number == 1
        observability[first_return] += first_return_bonus * (
            1.0 - observability[first_return]
        )
    observability = np.clip(observability, 0.0, 1.0).astype(np.float32)
    visible = observability >= surface_visibility_threshold
    return SurfaceVisibilityResult(
        visible=visible,
        observability=observability,
        geometric_observability=geometric.astype(np.float32),
        return_observability=return_observability,
        cell_size=float(cell_size),
        base_radius=float(base_radius),
        width=width,
        height=height,
    )
