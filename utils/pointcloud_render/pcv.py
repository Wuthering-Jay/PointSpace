"""Portion of Visible Sky (PCV) for point clouds.

This is an independent NumPy implementation of the ShadeVis/PCV geometric
idea: sample a hemisphere uniformly, render an orthographic z-buffer from each
direction, and count how often every point lies on the visible front layer.
The resulting value is a view-independent per-point scalar in ``[0, 1]``.
"""

from __future__ import annotations

import numpy as np

from .numpy_renderer import render_point_cloud_axis, render_point_cloud_image


def sample_hemisphere_directions(number_of_directions=128, dtype=np.float32):
    """Generate equal-solid-angle Fibonacci samples over the +Z hemisphere."""
    number_of_directions = int(number_of_directions)
    if number_of_directions < 1:
        raise ValueError("number_of_directions must be positive")
    index = np.arange(number_of_directions, dtype=np.float64)
    z = (index + 0.5) / number_of_directions
    azimuth = index * (np.pi * (3.0 - np.sqrt(5.0)))
    radius = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    directions = np.column_stack(
        (radius * np.cos(azimuth), radius * np.sin(azimuth), z)
    )
    return directions.astype(dtype, copy=False)


def _view_basis(direction):
    direction = np.asarray(direction, dtype=np.float32)
    direction /= max(float(np.linalg.norm(direction)), 1e-12)
    helper = (
        np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
        if abs(float(direction[2])) < 0.90
        else np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    )
    right = np.cross(helper, direction)
    right /= max(float(np.linalg.norm(right)), 1e-12)
    up = np.cross(direction, right)
    return right, up, direction


def _circular_offsets(radius):
    radius = int(radius)
    return np.asarray(
        [
            (dx, dy)
            for dy in range(-radius, radius + 1)
            for dx in range(-radius, radius + 1)
            if dx * dx + dy * dy <= (radius + 0.25) ** 2
        ],
        dtype=np.int32,
    )


def compute_pcv(
    points,
    number_of_directions=128,
    resolution=1024,
    point_radius=0,
    depth_tolerance=None,
    directions=None,
    projection_center=None,
    pixel_size=None,
    strict_bounds=False,
):
    """Compute a per-point Portion of Visible Sky value.

    Args:
        points: ``[N, 3]`` XYZ coordinates.
        number_of_directions: number of equal-area +Z hemisphere samples.
        resolution: square orthographic depth-buffer resolution.
        point_radius: circular splat radius in buffer pixels.  Zero matches a
            one-pixel OpenGL point.  Increase it only for sparse clouds, since
            oversized splats systematically underestimate visible sky.
        depth_tolerance: world-space tolerance for accepting points near the
            front layer.  The default follows the projected pixel footprint.
        directions: optional custom ``[M, 3]`` directions.  When supplied,
            ``number_of_directions`` is ignored.
        projection_center: optional fixed XYZ origin shared across nested
            crops.  Use with ``pixel_size`` for a controlled extent study.
        pixel_size: optional fixed world units per depth-buffer pixel.  If
            supplied, the pixel grid is anchored to integer buffer centers,
            so enlarging an even-resolution buffer does not shift its grid.
        strict_bounds: raise if any projected points fall outside the buffer.

    Returns:
        Float32 array of shape ``[N]`` in ``[0, 1]``.
    """
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape [N, 3], got {points.shape}")
    if points.shape[0] == 0:
        return np.empty(0, dtype=np.float32)
    if not np.isfinite(points).all():
        raise ValueError("points contain NaN or infinite coordinates")
    resolution = int(resolution)
    if resolution < 32:
        raise ValueError("resolution must be at least 32")
    point_radius = int(point_radius)
    if point_radius < 0:
        raise ValueError("point_radius must be non-negative")

    if projection_center is None:
        projection_center = np.median(points, axis=0)
    projection_center = np.asarray(projection_center, dtype=np.float64)
    if projection_center.shape != (3,):
        raise ValueError("projection_center must have shape [3]")
    centered = (points - projection_center).astype(np.float32, copy=False)
    extent = float(np.max(np.linalg.norm(centered, axis=1)))
    extent = max(extent * 1.005, 1e-6)
    if pixel_size is None:
        scale = (resolution - 3.0) / (2.0 * extent)
        center_pixel = (resolution - 1.0) * 0.5
    else:
        pixel_size = float(pixel_size)
        if pixel_size <= 0:
            raise ValueError("pixel_size must be positive")
        scale = 1.0 / pixel_size
        center_pixel = resolution // 2
    if depth_tolerance is None:
        pixel_footprint = (
            pixel_size
            if pixel_size is not None
            else 2.0 * extent / max(resolution - 3, 1)
        )
        depth_tolerance = max(0.35 * pixel_footprint, 2e-4 * extent)
    depth_tolerance = float(depth_tolerance)
    if depth_tolerance < 0:
        raise ValueError("depth_tolerance must be non-negative")

    if directions is None:
        directions = sample_hemisphere_directions(number_of_directions)
    else:
        directions = np.asarray(directions, dtype=np.float32)
        if directions.ndim != 2 or directions.shape[1] != 3:
            raise ValueError("directions must have shape [M, 3]")
        if directions.shape[0] == 0:
            raise ValueError("directions must not be empty")
        if np.any(directions[:, 2] < -1e-7):
            raise ValueError("PCV directions must lie on the +Z hemisphere")
        norm = np.linalg.norm(directions, axis=1, keepdims=True)
        directions = directions / np.maximum(norm, 1e-12)

    offsets = _circular_offsets(point_radius)
    visibility_count = np.zeros(centered.shape[0], dtype=np.uint32)
    z_buffer = np.empty(resolution * resolution, dtype=np.float32)
    source_depth = np.empty((centered.shape[0], offsets.shape[0]), dtype=np.float32)

    for direction in directions:
        right, up, view = _view_basis(direction)
        projected_x = centered @ right
        projected_y = centered @ up
        depth = centered @ view
        if pixel_size is None:
            px = np.rint(projected_x * scale + center_pixel).astype(np.int32)
            py = np.rint(projected_y * scale + center_pixel).astype(np.int32)
        else:
            px = np.floor(projected_x * scale).astype(np.int32) + center_pixel
            py = np.floor(projected_y * scale).astype(np.int32) + center_pixel
        inside = (
            (px >= 0)
            & (px < resolution)
            & (py >= 0)
            & (py < resolution)
        )
        if strict_bounds and not np.all(inside):
            raise ValueError(
                f"PCV buffer too small: {(~inside).sum():,} projected points "
                f"outside resolution={resolution}; increase resolution or "
                "pixel_size"
            )

        splat_x = px[:, None] + offsets[None, :, 0]
        splat_y = py[:, None] + offsets[None, :, 1]
        splat_valid = (
            inside[:, None]
            & (splat_x >= 0)
            & (splat_x < resolution)
            & (splat_y >= 0)
            & (splat_y < resolution)
        )
        pixel = (splat_y * resolution + splat_x)[splat_valid]
        source_depth[:] = depth[:, None]
        repeated_depth = source_depth[splat_valid]
        z_buffer.fill(-np.inf)
        np.maximum.at(z_buffer, pixel, repeated_depth)

        center_index = py * resolution + px
        visible = np.zeros(centered.shape[0], dtype=bool)
        visible[inside] = (
            depth[inside]
            >= z_buffer[center_index[inside]] - depth_tolerance
        )
        visibility_count += visible

    return (visibility_count.astype(np.float32) / directions.shape[0]).astype(
        np.float32, copy=False
    )


def pcv_to_illumination(
    pcv,
    ambient_floor=0.26,
    gamma=0.95,
    normalize=True,
    percentiles=(1.0, 99.0),
):
    """Map physical PCV fractions to a bounded display illumination."""
    pcv = np.asarray(pcv, dtype=np.float32)
    value = np.clip(pcv, 0.0, 1.0)
    if normalize and value.size:
        low, high = np.percentile(value, percentiles)
        if high > low + 1e-6:
            value = np.clip((value - low) / (high - low), 0.0, 1.0)
    ambient_floor = float(np.clip(ambient_floor, 0.0, 1.0))
    return ambient_floor + (1.0 - ambient_floor) * np.power(value, gamma)


def modulate_rgba_with_pcv(
    rgba,
    pcv,
    ambient_floor=0.26,
    gamma=0.95,
    normalize=True,
):
    """Fuse PCV illumination with arbitrary scalar/class/elevation RGBA."""
    rgba = np.asarray(rgba, dtype=np.float32).copy()
    if rgba.ndim != 2 or rgba.shape[1] != 4:
        raise ValueError(f"rgba must have shape [N, 4], got {rgba.shape}")
    pcv = np.asarray(pcv, dtype=np.float32)
    if pcv.shape != (rgba.shape[0],):
        raise ValueError("pcv length must match rgba")
    illumination = pcv_to_illumination(
        pcv,
        ambient_floor=ambient_floor,
        gamma=gamma,
        normalize=normalize,
    )
    rgb = np.clip(rgba[:, :3], 0.0, 1.0)
    linear = np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4,
    )
    linear *= illumination[:, None]
    rgba[:, :3] = np.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * np.power(np.maximum(linear, 0.0), 1.0 / 2.4) - 0.055,
    )
    rgba[:, :3] = np.clip(rgba[:, :3], 0.0, 1.0)
    return rgba


def render_pcv_point_cloud_image(
    x,
    y,
    z,
    rgba,
    camera=None,
    width=920,
    number_of_directions=128,
    pcv_resolution=1024,
    pcv_point_radius=0,
    ambient_floor=0.26,
    gamma=0.95,
    post_style="pcv",
    edl_strength=4.0,
    render_point_scale=0.86,
    render_min_radius=1,
    render_max_radius=5,
    render_fractional_splat=True,
):
    """Compute PCV, fuse it with colors, and return a static RGBA image."""
    points = np.column_stack((x, y, z))
    pcv = compute_pcv(
        points,
        number_of_directions=number_of_directions,
        resolution=pcv_resolution,
        point_radius=pcv_point_radius,
    )
    shaded = modulate_rgba_with_pcv(
        rgba, pcv, ambient_floor=ambient_floor, gamma=gamma
    )
    if post_style not in {"pcv", "pcv_edl", "pcv_lit_edl"}:
        raise ValueError(
            "post_style must be 'pcv', 'pcv_edl', or 'pcv_lit_edl'"
        )
    rendered = render_point_cloud_image(
        x,
        y,
        z,
        shaded,
        camera=camera,
        width=width,
        strength=edl_strength,
        style=post_style,
        point_scale=render_point_scale,
        min_point_radius=render_min_radius,
        max_point_radius=render_max_radius,
        fractional_splat=render_fractional_splat,
    )
    return (*rendered, pcv)


def render_pcv_point_cloud_axis(
    ax,
    x,
    y,
    z,
    rgba,
    camera=None,
    width=920,
    number_of_directions=128,
    pcv_resolution=1024,
    pcv_point_radius=0,
    ambient_floor=0.26,
    gamma=0.95,
    post_style="pcv",
    edl_strength=4.0,
    render_point_scale=0.86,
    render_min_radius=1,
    render_max_radius=5,
    render_fractional_splat=True,
):
    """Compute PCV and render it directly into a Matplotlib axis."""
    image, camera, projection, radius, pcv = render_pcv_point_cloud_image(
        x,
        y,
        z,
        rgba,
        camera=camera,
        width=width,
        number_of_directions=number_of_directions,
        pcv_resolution=pcv_resolution,
        pcv_point_radius=pcv_point_radius,
        ambient_floor=ambient_floor,
        gamma=gamma,
        post_style=post_style,
        edl_strength=edl_strength,
        render_point_scale=render_point_scale,
        render_min_radius=render_min_radius,
        render_max_radius=render_max_radius,
        render_fractional_splat=render_fractional_splat,
    )
    ax.imshow(image, origin="lower", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return camera, projection, radius, pcv
