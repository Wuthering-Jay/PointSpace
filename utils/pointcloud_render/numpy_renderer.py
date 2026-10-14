"""Publication-oriented static point-cloud rendering with NumPy and SciPy.

The renderer is deliberately deterministic and independent of an OpenGL
context.  It uses orthographic circular splats and a z-buffer, followed by a
hue-preserving Lit-EDL pass.  Lit-EDL keeps geometric edge enhancement separate
from broad surface illumination: it does not cast short screen-space shadows,
which tend to form black blobs along roofs and tree crowns in sparse point
clouds.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def camera_basis(x, y, azimuth=None, elevation=72.0):
    """Return a stable near-nadir camera aligned to the dominant XY axis."""
    x = np.asarray(x)
    y = np.asarray(y)
    if azimuth is None:
        xy = np.column_stack((x - np.median(x), y - np.median(y)))
        if xy.shape[0] > 20000:
            xy = xy[:: max(1, xy.shape[0] // 20000)]
        covariance = np.cov(xy, rowvar=False)
        principal = np.linalg.eigh(covariance)[1][:, -1]
        azimuth = np.degrees(np.arctan2(principal[1], principal[0])) + 38.0
    azimuth = np.radians(azimuth)
    elevation = np.radians(elevation)
    right = np.asarray([-np.sin(azimuth), np.cos(azimuth), 0.0])
    view = np.asarray(
        [
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        ]
    )
    up = np.cross(view, right)
    return right, up, view


def project_points(x, y, z, camera=None):
    """Project world coordinates to orthographic camera coordinates."""
    if camera is None:
        right, up, view = camera_basis(x, y)
        center = np.asarray([np.median(x), np.median(y), np.median(z)])
        camera = (right, up, view, center)
    else:
        right, up, view, center = camera
    points = np.column_stack((x, y, z)) - center
    return points @ right, points @ up, points @ view, camera


def to_image_pixels(x, y, z, camera, projection):
    """Map world coordinates into the pixel system returned by the renderer."""
    u, v, _, _ = project_points(x, y, z, camera)
    u_low, u_span, v_low, v_span, width, height, margin = projection
    px = (u - u_low) / u_span * (width - 2 * margin - 1) + margin
    py = (v - v_low) / v_span * (height - 2 * margin - 1) + margin
    return px, py


def _shift_no_wrap(array, dy, dx, fill_value=np.nan):
    shifted = np.full_like(array, fill_value)
    source_y = slice(max(0, -dy), min(array.shape[0], array.shape[0] - dy))
    source_x = slice(max(0, -dx), min(array.shape[1], array.shape[1] - dx))
    target_y = slice(max(0, dy), min(array.shape[0], array.shape[0] + dy))
    target_x = slice(max(0, dx), min(array.shape[1], array.shape[1] + dx))
    shifted[target_y, target_x] = array[source_y, source_x]
    return shifted


def _masked_gaussian(values, valid, sigma):
    numerator = ndi.gaussian_filter(
        np.where(valid, values, 0.0).astype(np.float32), sigma=sigma, mode="nearest"
    )
    denominator = ndi.gaussian_filter(
        valid.astype(np.float32), sigma=sigma, mode="nearest"
    )
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 1e-4,
    )


def _masked_bilateral(values, valid, spatial_sigma=1.6):
    """Fast masked cross-bilateral pass for surface-normal estimation."""
    values = np.asarray(values, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    adjacent_differences = []
    for dy, dx in ((1, 0), (0, 1)):
        neighbor = _shift_no_wrap(values, dy, dx)
        pair = valid & np.isfinite(neighbor)
        difference = np.abs(values[pair] - neighbor[pair])
        if difference.size > 100000:
            difference = difference[:: max(1, difference.size // 100000)]
        if difference.size:
            adjacent_differences.append(difference)
    if adjacent_differences:
        sampled = np.concatenate(adjacent_differences)
        range_sigma = float(np.percentile(sampled, 72.0)) * 1.8
    else:
        range_sigma = 0.01
    range_sigma = float(np.clip(range_sigma, 0.006, 0.055))

    numerator = np.where(valid, values, 0.0).astype(np.float32)
    denominator = valid.astype(np.float32)
    offsets = (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
        (-2, 0),
        (2, 0),
        (0, -2),
        (0, 2),
    )
    for dy, dx in offsets:
        neighbor = _shift_no_wrap(values, dy, dx)
        pair = valid & np.isfinite(neighbor)
        spatial_weight = np.exp(
            -(dx * dx + dy * dy) / (2.0 * spatial_sigma * spatial_sigma)
        )
        range_weight = np.exp(
            -np.square(np.where(pair, neighbor - values, 0.0))
            / (2.0 * range_sigma * range_sigma)
        )
        weight = np.where(pair, spatial_weight * range_weight, 0.0).astype(
            np.float32
        )
        numerator += weight * np.where(pair, neighbor, 0.0)
        denominator += weight
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 1e-5,
    )


def _robust_unit(values, valid, percentile=95.0):
    selected = values[valid & np.isfinite(values) & (values > 0)]
    if selected.size == 0:
        return np.zeros_like(values, dtype=np.float32)
    scale = float(np.percentile(selected, percentile))
    return np.clip(values / max(scale, 1e-6), 0.0, 1.0).astype(np.float32)


def _robust_centered(values, valid, percentile=97.0):
    selected = np.abs(values[valid & np.isfinite(values)])
    if selected.size == 0:
        return np.zeros_like(values, dtype=np.float32)
    scale = float(np.percentile(selected, percentile))
    return np.clip(values / max(scale, 1e-6), -1.0, 1.0).astype(np.float32)


def apply_lit_edl(image, normalized_depth, occupied, radius, strength):
    """Apply the formal hue-preserving Lit-EDL shading pass.

    Small and broad depth discontinuities produce restrained EDL outlines,
    while fine/coarse depth surfaces provide normal-based key and fill light.
    Coarse canopy relief adds readable crown-scale structure in forests.  The
    illumination floor intentionally prevents the outline-attached black blobs
    produced by screen-space cast-shadow approximations.
    """
    depth = np.asarray(normalized_depth, dtype=np.float32)
    occupied = np.asarray(occupied, dtype=bool)
    if not occupied.any():
        return image

    # Edge enhancement is intentionally weak at the smallest scale in rough
    # areas.  Forests then retain crown boundaries without turning every return
    # discontinuity into a dark speck.
    fine_edge = np.zeros_like(depth, dtype=np.float32)
    broad_edge = np.zeros_like(depth, dtype=np.float32)
    for distance, target in (
        (1, fine_edge),
        (2, fine_edge),
        (5, broad_edge),
        (9, broad_edge),
    ):
        for dy, dx in (
            (-distance, 0),
            (distance, 0),
            (0, -distance),
            (0, distance),
        ):
            neighbor = _shift_no_wrap(depth, dy, dx)
            valid = occupied & np.isfinite(neighbor)
            target[:] = np.maximum(
                target,
                np.where(valid, np.abs(depth - neighbor), 0.0),
            )
    fine_edge = _robust_unit(fine_edge, occupied, 97.0)
    broad_edge = _robust_unit(broad_edge, occupied, 97.0)
    local_roughness = _masked_gaussian(
        fine_edge, occupied, sigma=max(4.0, 2.5 * radius)
    )
    local_roughness = _robust_unit(local_roughness, occupied, 96.0)

    # Smooth man-made surfaces receive crisp, symmetric outlines.  Locally
    # rough canopy areas automatically exchange point-scale edges for broad
    # crown boundaries, preventing speckled black clumps.
    fine_weight = np.clip(0.62 - 0.46 * local_roughness, 0.16, 0.62)
    edge = fine_weight * fine_edge + (1.0 - fine_weight) * broad_edge
    edge_amount = 0.27 - 0.11 * local_roughness
    edl = 1.0 - edge_amount * (1.0 - np.exp(-0.40 * strength * edge))

    # Multi-scale depth relief: fine geometry keeps roof/facade structure,
    # while the coarse field turns dense forest into readable crown masses.
    fine_sigma = max(2.5, 1.35 * radius)
    coarse_sigma = max(7.0, 4.5 * radius)
    very_coarse_sigma = max(16.0, 9.0 * radius)
    bilateral_surface = _masked_bilateral(depth, occupied)
    fine_surface = _masked_gaussian(
        bilateral_surface, occupied, sigma=max(1.2, 0.55 * fine_sigma)
    )
    coarse_surface = _masked_gaussian(depth, occupied, sigma=coarse_sigma)
    very_coarse = _masked_gaussian(depth, occupied, sigma=very_coarse_sigma)

    fine_gy, fine_gx = np.gradient(fine_surface)
    coarse_gy, coarse_gx = np.gradient(coarse_surface)
    gradient_mix = np.clip(0.72 - 0.42 * local_roughness, 0.30, 0.72)
    gradient_x = gradient_mix * fine_gx + (1.0 - gradient_mix) * coarse_gx
    gradient_y = gradient_mix * fine_gy + (1.0 - gradient_mix) * coarse_gy
    gradient = np.hypot(gradient_x, gradient_y)
    gradient_scale = float(np.percentile(gradient[occupied], 92.0))
    normal_gain = 1.15 / max(gradient_scale, 1e-6)
    normal_x = -gradient_x * normal_gain
    normal_y = -gradient_y * normal_gain
    normal_z = np.ones_like(normal_x)
    normal_length = np.sqrt(normal_x**2 + normal_y**2 + normal_z**2)

    # A strong key and an opposing fill light give shape without creating a
    # black side.  This is broad surface illumination, not a cast shadow.
    key = np.asarray([-0.55, 0.60, 0.58], dtype=np.float32)
    fill = np.asarray([0.42, -0.46, 0.78], dtype=np.float32)
    key /= np.linalg.norm(key)
    fill /= np.linalg.norm(fill)

    def lambert(light):
        return np.clip(
            (
                normal_x * light[0]
                + normal_y * light[1]
                + normal_z * light[2]
            )
            / np.maximum(normal_length, 1e-6),
            0.0,
            1.0,
        )

    surface_light = 0.57 + 0.34 * lambert(key) + 0.13 * lambert(fill)

    # Crown/terrain-scale high-pass relief is signed: high masses brighten on
    # the key side and valleys darken mildly.  Its zero-centered form avoids
    # the glued-on dark silhouettes caused by one-sided occlusion terms.
    canopy_relief = _robust_centered(coarse_surface - very_coarse, occupied, 97.5)
    relief_gain = 0.075 + 0.115 * local_roughness
    relief_light = 1.0 + relief_gain * canopy_relief

    # Only a broad, low-opacity cavity term remains.  It is derived from the
    # very-coarse surface and cannot trace individual point/roof outlines.
    cavity = _robust_unit(
        np.maximum(very_coarse - coarse_surface, 0.0), occupied, 98.0
    )
    broad_ambient = 1.0 - 0.055 * cavity

    illumination = edl * surface_light * relief_light * broad_ambient
    median_light = float(np.median(illumination[occupied]))
    illumination = illumination / max(median_light, 1e-6)
    # Robust clipping controls isolated canopy spikes before applying a modest
    # contrast boost.  The 0.68 floor is the main anti-black-clump guarantee.
    low, high = np.percentile(illumination[occupied], [1.0, 99.0])
    illumination = np.clip(illumination, low, high)
    illumination = 1.0 + 1.30 * (illumination - 1.0)
    illumination = np.clip(illumination, 0.68, 1.13)

    rgb = np.clip(image[..., :3], 0.0, 1.0)
    linear = np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4,
    )
    # Multiplication alone cannot reveal highlights in saturated scientific
    # colormaps (e.g. viridis yellow is already near 1).  Shadows remain
    # multiplicative, while highlights use a restrained screen blend toward
    # white.  This exposes canopy relief without altering scalar-data hues.
    light = illumination[occupied, None]
    occupied_rgb = linear[occupied]
    shadow_factor = np.minimum(light, 1.0)
    occupied_rgb = occupied_rgb * shadow_factor
    highlight = np.maximum(light - 1.0, 0.0)
    occupied_rgb = occupied_rgb + (1.0 - occupied_rgb) * (1.35 * highlight)
    linear[occupied] = np.clip(occupied_rgb, 0.0, 1.0)
    rgb = np.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * np.power(np.maximum(linear, 0.0), 1.0 / 2.4) - 0.055,
    )
    image[..., :3][occupied] = np.clip(rgb[occupied], 0.0, 1.0)
    return image


def apply_pcv_edl(image, normalized_depth, occupied, radius, strength=4.0):
    """Add only adaptive symmetric depth edges to PCV-modulated colors.

    PCV already supplies view-independent ambient occlusion.  This pass avoids
    the surface lighting, cavity shading and directional terms from Lit-EDL,
    preventing double illumination while restoring crisp man-made boundaries.
    """
    depth = np.asarray(normalized_depth, dtype=np.float32)
    occupied = np.asarray(occupied, dtype=bool)
    if not occupied.any():
        return image

    fine_edge = np.zeros_like(depth, dtype=np.float32)
    broad_edge = np.zeros_like(depth, dtype=np.float32)
    for distance, target in (
        (1, fine_edge),
        (2, fine_edge),
        (5, broad_edge),
        (9, broad_edge),
    ):
        for dy, dx in (
            (-distance, 0),
            (distance, 0),
            (0, -distance),
            (0, distance),
        ):
            neighbor = _shift_no_wrap(depth, dy, dx)
            valid = occupied & np.isfinite(neighbor)
            target[:] = np.maximum(
                target,
                np.where(valid, np.abs(depth - neighbor), 0.0),
            )
    fine_edge = _robust_unit(fine_edge, occupied, 97.5)
    broad_edge = _robust_unit(broad_edge, occupied, 97.5)
    local_roughness = _masked_gaussian(
        fine_edge, occupied, sigma=max(4.0, 2.5 * radius)
    )
    local_roughness = _robust_unit(local_roughness, occupied, 96.0)

    # Smooth roofs/roads retain narrow lines.  Canopies use mostly broad edges
    # with a smaller amplitude so individual returns do not become black dots.
    fine_weight = np.clip(0.72 - 0.54 * local_roughness, 0.18, 0.72)
    edge = fine_weight * fine_edge + (1.0 - fine_weight) * broad_edge
    edge_amount = 0.22 - 0.09 * local_roughness
    shade = 1.0 - edge_amount * (1.0 - np.exp(-0.42 * strength * edge))
    shade = np.clip(shade, 0.78, 1.0)

    rgb = np.clip(image[..., :3], 0.0, 1.0)
    linear = np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4,
    )
    linear[occupied] *= shade[occupied, None]
    rgb = np.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * np.power(np.maximum(linear, 0.0), 1.0 / 2.4) - 0.055,
    )
    image[..., :3][occupied] = np.clip(rgb[occupied], 0.0, 1.0)
    return image


def _apply_classic_edl(image, normalized_depth, occupied, strength):
    edge = np.zeros_like(normalized_depth, dtype=np.float32)
    for dy, dx in (
        (-2, 0),
        (2, 0),
        (0, -2),
        (0, 2),
        (-1, -1),
        (1, 1),
        (-1, 1),
        (1, -1),
    ):
        neighbor = _shift_no_wrap(normalized_depth, dy, dx)
        valid = occupied & np.isfinite(neighbor)
        edge = np.maximum(
            edge,
            np.where(valid, np.abs(normalized_depth - neighbor), 0.0),
        )
    shade = 0.30 + 0.70 * np.exp(-strength * edge)
    image[..., :3][occupied] *= shade[occupied, None]
    image[..., :3] = np.clip(image[..., :3], 0.0, 1.0)
    return image


def _fill_small_projection_gaps(image, depth_image, occupied, radius):
    """Expand visible splats only across tiny sampling gaps.

    This compensates for clustered forest sampling, where a global point-count
    estimate chooses a small splat despite irregular crown-level holes.  The
    nearest visible sample supplies both color and depth, so the subsequent
    lighting pass still sees a coherent surface.  Large physical gaps and
    background remain untouched.
    """
    distance, nearest = ndi.distance_transform_edt(
        ~occupied, return_distances=True, return_indices=True
    )
    fill_limit = float(np.clip(1.15 + 0.20 * radius, 1.25, 2.0))
    fill = (~occupied) & (distance <= fill_limit)
    if not fill.any():
        return image, depth_image, occupied
    nearest_y = nearest[0][fill]
    nearest_x = nearest[1][fill]
    image[fill] = image[nearest_y, nearest_x]
    depth_image[fill] = depth_image[nearest_y, nearest_x]
    return image, depth_image, np.isfinite(depth_image)


def render_point_cloud_image(
    x,
    y,
    z,
    rgba,
    camera=None,
    width=920,
    strength=5.5,
    style="lit_edl",
    point_scale=0.95,
    min_point_radius=1,
    max_point_radius=5,
    fractional_splat=False,
    square_frame=False,
    return_depth=False,
    projection_bounds=None,
):
    """Rasterize a point cloud into an RGBA image.

    Args:
        style: ``"lit_edl"`` for the formal renderer, ``"classic_edl"`` for
            the legacy baseline, or ``"pcv"`` when colors have already been
            modulated by a per-point PCV field and only rasterization plus
            tiny-gap coverage correction is desired. ``"pcv_edl"`` adds a
            restrained edge-only EDL pass to those pre-modulated colors.
            ``"pcv_lit_edl"`` applies the full shadow-free Lit-EDL surface
            lighting and is intended for final qualitative figures.
        point_scale: multiplier applied to the projected mean point spacing.
            It controls display splats only and is independent of the PCV
            z-buffer point radius.
        fractional_splat: use deterministic opaque coverage at the continuous
            density-scaled radius. The default keeps legacy integer splats.
        square_frame: use an equal-scale square projection instead of the
            default aspect-ratio projection with a capped image height.
        return_depth: also return the final per-pixel camera-depth buffer.
        projection_bounds: optional fixed ``(u_low, u_high, v_low, v_high)``
            camera-plane extent, useful for matched-view comparisons.

    Returns:
        ``(image, camera, projection, radius)`` by default, or the same tuple
        followed by ``depth_image`` when ``return_depth=True``.
    """
    if style not in {
        "lit_edl",
        "classic_edl",
        "pcv",
        "pcv_edl",
        "pcv_lit_edl",
    }:
        raise ValueError(f"Unsupported point-cloud render style: {style!r}")
    point_scale = float(point_scale)
    min_point_radius = int(min_point_radius)
    max_point_radius = int(max_point_radius)
    if point_scale <= 0:
        raise ValueError("point_scale must be positive")
    if min_point_radius < 0 or max_point_radius < min_point_radius:
        raise ValueError("invalid min/max point radius")

    u, v, depth, camera = project_points(x, y, z, camera)
    if projection_bounds is None:
        u_low, u_high = np.quantile(u, [0.002, 0.998])
        v_low, v_high = np.quantile(v, [0.002, 0.998])
    else:
        bounds = np.asarray(projection_bounds, dtype=np.float64)
        if (bounds.shape != (4,) or not np.isfinite(bounds).all()
                or bounds[1] <= bounds[0] or bounds[3] <= bounds[2]):
            raise ValueError("projection_bounds must be finite increasing u/v bounds")
        u_low, u_high, v_low, v_high = bounds
    u_span = max(float(u_high - u_low), 1e-6)
    v_span = max(float(v_high - v_low), 1e-6)
    if square_frame:
        span = max(u_span, v_span)
        u_low = (u_low + u_high - span) / 2
        v_low = (v_low + v_high - span) / 2
        u_span = v_span = span
        height = width
    else:
        height = int(np.clip(width * v_span / u_span, 360, 720))
    margin = 8
    px = np.rint(
        (u - u_low) / u_span * (width - 2 * margin - 1) + margin
    ).astype(np.int32)
    py = np.rint(
        (v - v_low) / v_span * (height - 2 * margin - 1) + margin
    ).astype(np.int32)
    valid = (px >= 0) & (px < width) & (py >= 0) & (py < height)
    px, py, depth = px[valid], py[valid], depth[valid]
    rgba = np.asarray(rgba, dtype=np.float32)[valid]
    source = np.arange(px.size, dtype=np.int64)

    projected_area = max((width - 2 * margin) * (height - 2 * margin), 1)
    mean_spacing = np.sqrt(projected_area / max(len(px), 1))
    continuous_radius = float(
        np.clip(point_scale * mean_spacing, min_point_radius, max_point_radius)
    )
    radius = (
        continuous_radius
        if fractional_splat
        else int(np.rint(continuous_radius))
    )
    footprint_limit = radius + (0.75 if fractional_splat else 0.25)
    offset_limit = int(np.ceil(footprint_limit))
    offsets = np.asarray(
        [
            (dx, dy)
            for dy in range(-offset_limit, offset_limit + 1)
            for dx in range(-offset_limit, offset_limit + 1)
            if dx * dx + dy * dy <= footprint_limit ** 2
        ],
        dtype=np.int32,
    )
    if fractional_splat:
        offset_coverage = np.clip(
            (radius + 0.75 - np.linalg.norm(offsets, axis=1)) / 0.75,
            0.0,
            1.0,
        ).astype(np.float32)
    splat_x = px[:, None] + offsets[None, :, 0]
    splat_y = py[:, None] + offsets[None, :, 1]
    splat_valid = (
        (splat_x >= 0)
        & (splat_x < width)
        & (splat_y >= 0)
        & (splat_y < height)
    )
    pixel = (splat_y * width + splat_x)[splat_valid]
    repeated_depth = np.broadcast_to(depth[:, None], splat_x.shape)[splat_valid]
    repeated_source = np.broadcast_to(source[:, None], splat_x.shape)[splat_valid]
    if fractional_splat:
        repeated_coverage = np.broadcast_to(
            offset_coverage[None, :], splat_x.shape
        )[splat_valid]
        candidate_y, candidate_x = np.divmod(pixel, width)
        # Dither the fractional rim before the z-buffer, allowing a lower
        # point to fill a pixel rejected for the closest point.
        pixel_hash = (
            candidate_x.astype(np.uint32) * np.uint32(374761393)
            + candidate_y.astype(np.uint32) * np.uint32(668265263)
            + np.uint32(42)
        )
        pixel_hash = (
            pixel_hash ^ (pixel_hash >> np.uint32(13))
        ) * np.uint32(1274126177)
        pixel_hash ^= pixel_hash >> np.uint32(16)
        covered = repeated_coverage > (
            pixel_hash.astype(np.float64) / 4294967296.0
        )
        pixel = pixel[covered]
        repeated_source = repeated_source[covered]
        repeated_depth = repeated_depth[covered]
    order = np.lexsort((repeated_depth, pixel))
    sorted_pixel = pixel[order]
    keep = np.r_[sorted_pixel[1:] != sorted_pixel[:-1], True]
    winner = order[keep]
    pixel = pixel[winner]
    repeated_source = repeated_source[winner]
    repeated_depth = repeated_depth[winner]
    image = np.ones((height, width, 4), dtype=np.float32)
    depth_image = np.full((height, width), np.nan, dtype=np.float32)
    yy, xx = np.divmod(pixel, width)
    image[yy, xx] = rgba[repeated_source]
    depth_image[yy, xx] = repeated_depth
    occupied = np.isfinite(depth_image)
    if style in {"lit_edl", "pcv", "pcv_edl", "pcv_lit_edl"} and occupied.any():
        image, depth_image, occupied = _fill_small_projection_gaps(
            image, depth_image, occupied, radius
        )
    if occupied.any():
        d_low, d_high = np.nanquantile(depth_image, [0.02, 0.98])
        normalized = (depth_image - d_low) / max(float(d_high - d_low), 1e-6)
        if style in {"lit_edl", "pcv_lit_edl"}:
            image = apply_lit_edl(
                image, normalized, occupied, radius=radius, strength=strength
            )
        elif style == "pcv_edl":
            image = apply_pcv_edl(
                image, normalized, occupied, radius=radius, strength=strength
            )
        elif style == "classic_edl":
            image = _apply_classic_edl(image, normalized, occupied, strength)
    projection = (u_low, u_span, v_low, v_span, width, height, margin)
    if return_depth:
        return image, camera, projection, radius, depth_image
    return image, camera, projection, radius


def render_point_cloud_axis(
    ax,
    x,
    y,
    z,
    rgba,
    camera=None,
    width=920,
    strength=5.5,
    style="lit_edl",
    point_scale=0.95,
    min_point_radius=1,
    max_point_radius=5,
    fractional_splat=False,
):
    """Render directly into a Matplotlib axis."""
    image, camera, projection, radius = render_point_cloud_image(
        x,
        y,
        z,
        rgba,
        camera=camera,
        width=width,
        strength=strength,
        style=style,
        point_scale=point_scale,
        min_point_radius=min_point_radius,
        max_point_radius=max_point_radius,
        fractional_splat=fractional_splat,
    )
    ax.imshow(image, origin="lower", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return camera, projection, radius
