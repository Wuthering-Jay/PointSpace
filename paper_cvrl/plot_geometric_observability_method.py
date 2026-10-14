"""Plot geometric observability and its VPSD supervision partition.

Run directly with the pointspace environment. The geometry and q_geo values
come from one real Swiss-Audit LAS tile and its full-resolution visibility
sidecar. No semantic labels, return attributes, or image features enter the
figure. Edit the constants below to use a different tile or transect.
"""

from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path

import laspy
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.collections import LineCollection
from matplotlib.path import Path as MplPath
from safetensors import safe_open
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from utils.pointcloud_render import (
    camera_basis,
    render_point_cloud_image,
    to_image_pixels,
)

AUDIT_ROOT = Path(r"G:\data\swiss\joint_tiles_100\Swiss-Audit")
STEM = "swisssurface3d_2024_2679-1248_2056_5728.copc_0033"
PROFILE_START = np.array([2679300.0, 1248386.0], dtype=np.float64)
PROFILE_END = np.array([2679392.0, 1248306.0], dtype=np.float64)
PROFILE_HALF_WIDTH_M = 0.75
SECTION_EXTENSION_M = 8.0
SECTION_Z_BELOW_M = 3.0
SECTION_Z_ABOVE_M = 5.0
ENVELOPE_STEP_M = 0.4
ENVELOPE_RADIUS_M = 1.1
MAX_PROFILE_POINTS = 16_000
TOP_RENDER_WIDTH = 1000
TOP_POINT_SCALE = 1.08
VIEW_AZIMUTH_DEG = -125.0
VIEW_ELEVATION_DEG = 32.0
SEED = 43
OUTPUT = ROOT / "paper_cvrl/results/method/figure_geometric_observability.png"
METADATA = OUTPUT.with_suffix(".json")


def load_real_tile():
    cloud_path = AUDIT_ROOT / "pointcloud" / f"{STEM}.las"
    if not cloud_path.exists():
        cloud_path = cloud_path.with_suffix(".laz")
    sidecar = AUDIT_ROOT / "visibility" / f"{STEM}.safetensors"
    if not cloud_path.is_file() or not sidecar.is_file():
        raise FileNotFoundError(f"Missing source point cloud or q_geo: {cloud_path}, {sidecar}")
    cloud = laspy.read(cloud_path)
    with safe_open(str(sidecar), framework="np") as handle:
        q = np.asarray(handle.get_tensor("q_geo"), dtype=np.float32)
    xyz = np.column_stack((cloud.x, cloud.y, cloud.z))
    if q.shape != (xyz.shape[0],) or not np.isfinite(q).all():
        raise ValueError("Full-resolution XYZ and q_geo are not aligned or finite")
    if np.any((q < 0) | (q > 1)):
        raise ValueError("q_geo must be in [0, 1]")
    return cloud_path, xyz, q


def load_thresholds():
    summary = json.loads(
        (AUDIT_ROOT / "analysis/q_geo_summary.json").read_text(encoding="utf-8")
    )
    import re

    low = float(re.search(r"<=\s*([0-9.]+)", summary["thresholds"]["P_inv"]).group(1))
    high = float(re.search(r">=\s*([0-9.]+)", summary["thresholds"]["P_vis"]).group(1))
    if not 0 <= low < high <= 1:
        raise ValueError("Invalid visibility partition thresholds")
    return low, high


def profile_geometry(xyz):
    direction = PROFILE_END - PROFILE_START
    length = float(np.linalg.norm(direction))
    unit = direction / length
    displacement = xyz[:, :2] - PROFILE_START
    along = displacement @ unit
    across = np.abs(displacement[:, 0] * unit[1] - displacement[:, 1] * unit[0])
    mask = (along >= 0) & (along <= length) & (across <= PROFILE_HALF_WIDTH_M)
    indices = np.flatnonzero(mask)
    if len(indices) < 100:
        raise ValueError("The selected real transect contains too few ALS points")
    return along, indices, length


def top_surface_envelope(xyz, length):
    """Orthographic upper envelope sampled from every point in the LAS tile.

    A circular XY neighborhood at every transect position acts as a local
    max-Z buffer. The gentle Gaussian filter removes isolated spikes only for
    drawing the curve; raw max-Z values are retained for gap calculations.
    """
    stations = np.arange(0, length + ENVELOPE_STEP_M, ENVELOPE_STEP_M)
    xy = PROFILE_START + stations[:, None] * (PROFILE_END - PROFILE_START) / length
    tree = cKDTree(xyz[:, :2])
    neighbors = tree.query_ball_point(xy, r=ENVELOPE_RADIUS_M, workers=-1)
    raw = np.asarray(
        [np.max(xyz[ids, 2]) if ids else np.nan for ids in neighbors],
        dtype=np.float64,
    )
    valid = np.isfinite(raw)
    if valid.sum() < 0.95 * len(raw):
        raise ValueError("Insufficient full-cloud support for the surface envelope")
    raw[~valid] = np.interp(stations[~valid], stations[valid], raw[valid])
    display = gaussian_filter1d(raw, sigma=1.0, mode="nearest")
    return stations, raw, display


def representative_points(xyz, q, indices, along, stations, raw_top, low, high):
    z = xyz[indices, 2]
    # The local 3D Z-buffer is evaluated at every selected point's station.
    local_top = np.interp(along[indices], stations, raw_top)
    gap = np.maximum(local_top - z, 0.0)
    inner = (along[indices] > 8) & (along[indices] < stations[-1] - 8)
    definitions = (
        # The upper point must truly sit on a roof/canopy-height surface.
        ("surface", inner & (q[indices] >= high) & (gap < 0.45)
         & (z > np.quantile(z, 0.75)) & (along[indices] > 48)
         & (along[indices] < 64), 0.05, 56.0),
        ("intermediate", inner & (q[indices] > low) & (q[indices] < high)
         & (gap > 1.0) & (z > 415) & (along[indices] > 12)
         & (along[indices] < 29), 3.0, 20.0),
        ("occluded", inner & (q[indices] <= low) & (gap > 6.0)
         & (along[indices] > 76) & (along[indices] < 96), 12.0, 84.0),
    )
    selected = []
    for name, candidates, target_gap, target_station in definitions:
        found = np.flatnonzero(candidates)
        if not len(found):
            raise ValueError(f"No genuine {name} point fits the current transect")
        # Favor a legible spread along the real transect, without reading GT.
        occupancy = np.abs(along[indices[found]] - target_station)
        score = np.abs(gap[found] - target_gap) + 0.035 * occupancy
        chosen = int(found[np.argmin(score)])
        selected.append({
            "kind": name,
            "point_index": int(indices[chosen]),
            "distance_m": float(along[indices[chosen]]),
            "elevation_m": float(z[chosen]),
            "top_elevation_m": float(local_top[chosen]),
            "delta_z_m": float(gap[chosen]),
            "q_geo": float(q[indices[chosen]]),
        })
    return selected


def sampled_indices(indices, limit, rng):
    if len(indices) <= limit:
        return indices
    return np.sort(rng.choice(indices, size=limit, replace=False))


def draw_partition(fig, low, high, cmap, norm):
    """Vertical continuous scale with the two supervision thresholds."""
    bar_ax = fig.add_axes((0.953, 0.28, 0.019, 0.65))
    scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    colorbar = fig.colorbar(scalar, cax=bar_ax, ticks=[0.0, low, high, 1.0])
    colorbar.ax.set_yticklabels(["0", f"{low:g}", f"{high:g}", "1"])
    colorbar.ax.tick_params(labelsize=12.5, length=3, width=0.65, pad=5)
    colorbar.outline.set_linewidth(0.6)
    for threshold in (low, high):
        colorbar.ax.axhline(threshold, color="#252B30", linewidth=1.05)


def draw_xyz_box(ax, xyz, xy_min, camera, projection, margin):
    """Project a real-world XYZ bounding cuboid around the ALS points."""
    xy_max = xyz[:, :2].max(axis=0) - xy_min
    z_low, z_high = np.quantile(xyz[:, 2], [0.001, 0.999])
    z_low -= 4.0
    z_high += 6.0
    limits = ((-9.0, xy_max[0] + 9.0),
              (-9.0, xy_max[1] + 9.0),
              (float(z_low), float(z_high)))
    keys = list(product((0, 1), repeat=3))
    vertices = np.array([[limits[d][bit] for d, bit in enumerate(key)]
                         for key in keys])
    px, py = to_image_pixels(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                             camera, projection)
    projected = np.column_stack((px - margin, py - margin))
    lookup = dict(zip(keys, range(len(keys))))
    lower_keys = [key for key in keys if key[2] == 0]
    depths = [float((vertices[lookup[key]] - camera[3]) @ camera[2])
              for key in lower_keys]
    rear_key = lower_keys[int(np.argmin(depths))]

    def project_line(points):
        points = np.asarray(points, dtype=np.float64)
        sx, sy = to_image_pixels(points[:, 0], points[:, 1], points[:, 2],
                                  camera, projection)
        return np.column_stack((sx - margin, sy - margin))

    floor = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
    rear_x = [(rear_key[0], 0, 0), (rear_key[0], 1, 0),
              (rear_key[0], 1, 1), (rear_key[0], 0, 1)]
    rear_y = [(0, rear_key[1], 0), (1, rear_key[1], 0),
              (1, rear_key[1], 1), (0, rear_key[1], 1)]
    for face, face_color in ((floor, "#F0F2F3"),
                             (rear_x, "#F5F6F7"),
                             (rear_y, "#F3F5F6")):
        corners = projected[[lookup[key] for key in face]]
        ax.fill(corners[:, 0], corners[:, 1], color=face_color,
                linewidth=0, zorder=0.1)

    # Grid lines lie on the floor and two rear walls, behind real points.
    # X/Y values are local metres; Z remains absolute elevation in metres.
    for dimension in (0, 1):
        other = 1 - dimension
        for value in (25.0, 50.0, 75.0):
            if not limits[dimension][0] < value < limits[dimension][1]:
                continue
            line = np.array([
                [limits[0][0], limits[1][0], limits[2][0]],
                [limits[0][0], limits[1][0], limits[2][0]],
            ])
            line[:, dimension] = value
            line[:, other] = limits[other]
            projected_line = project_line(line)
            ax.plot(projected_line[:, 0], projected_line[:, 1],
                    color="#B8C1C7", linewidth=0.75, alpha=0.85,
                    linestyle=(0, (2, 2.5)), zorder=0.5)
            wall = line.copy()
            wall[:, other] = limits[other][rear_key[other]]
            wall[:, 2] = limits[2]
            wall_line = project_line(wall)
            ax.plot(wall_line[:, 0], wall_line[:, 1],
                    color="#B8C1C7", linewidth=0.75, alpha=0.85,
                    linestyle=(0, (2, 2.5)), zorder=0.5)
    for elevation in np.arange(np.ceil(z_low / 10) * 10, z_high, 10):
        for fixed_dimension in (0, 1):
            varying = 1 - fixed_dimension
            line = np.array([
                [limits[0][rear_key[0]], limits[1][rear_key[1]], elevation],
                [limits[0][rear_key[0]], limits[1][rear_key[1]], elevation],
            ])
            line[:, varying] = limits[varying]
            wall_line = project_line(line)
            ax.plot(wall_line[:, 0], wall_line[:, 1],
                    color="#B8C1C7", linewidth=0.75, alpha=0.85,
                    linestyle=(0, (2, 2.5)), zorder=0.5)
    for key in keys:
        for dimension in range(3):
            if key[dimension] != 0:
                continue
            other = tuple(1 - bit if d == dimension else bit
                          for d, bit in enumerate(key))
            ends = projected[[lookup[key], lookup[other]]]
            ax.plot(ends[:, 0], ends[:, 1], color="#B9C1C7",
                    linewidth=0.75, alpha=0.7, zorder=1)
    # Place X/Y ticks on the exposed lower rim and Z ticks on the outer-left
    # vertical rim. No numeric labels cross the interior point cloud.
    floor_key = min(lower_keys, key=lambda key: projected[lookup[key], 1])
    outer_z_key = min(lower_keys, key=lambda key: projected[lookup[key], 0])
    center = projected.mean(axis=0)
    for dimension, label in enumerate("XYZ"):
        origin_key = floor_key if dimension < 2 else outer_z_key
        origin = projected[lookup[origin_key]]
        other = tuple(1 - bit if d == dimension else bit
                      for d, bit in enumerate(origin_key))
        tip = projected[lookup[other]]
        ax.plot([origin[0], tip[0]], [origin[1], tip[1]],
                color="#1F2529", linewidth=1.0, zorder=1.5)
        direction = (tip - origin) / max(np.linalg.norm(tip - origin), 1.0)
        perpendicular = np.array([-direction[1], direction[0]])
        outward = perpendicular.copy()
        if np.dot(outward, tip - center) < 0:
            outward *= -1
        text_xy = (origin + 0.73 * (tip - origin) + 48.0 * outward
                   if dimension < 2 else tip + 47.0 * direction)
        ax.text(*text_xy, f"{label} (m)", fontsize=12.5, fontweight="bold",
                color="#111111", ha="center", va="center", zorder=4)
        tick_values = ((0.0, 50.0, 100.0) if dimension < 2
                       else tuple(10.0 * np.arange(
                           np.ceil(limits[2][0] / 10.0) + 1,
                           np.floor(limits[2][1] / 10.0) + 1)))
        for value in tick_values:
            if not limits[dimension][0] <= value <= limits[dimension][1]:
                continue
            point = vertices[lookup[origin_key]].copy()
            point[dimension] = value
            tick = project_line(point[None])[0]
            outward = perpendicular.copy()
            if np.dot(outward, tick - center) < 0:
                outward *= -1
            tick_end = tick + 8.0 * outward
            ax.plot([tick[0], tick_end[0]], [tick[1], tick_end[1]],
                    color="#272727", linewidth=0.75, zorder=1.6)
            text_at = tick + (37.0 if dimension < 2 else 30.0) * outward
            ax.text(*text_at, f"{int(value)}", fontsize=10.0,
                    color="#343434", ha="center", va="center", zorder=4)
    return projected


def composite_section_plane(image, point_depth, corners, corner_depth):
    """Depth-test a translucent section against the rasterized real points."""
    height, width = point_depth.shape
    x0 = max(0, int(np.floor(corners[:, 0].min())))
    x1 = min(width, int(np.ceil(corners[:, 0].max())) + 1)
    y0 = max(0, int(np.floor(corners[:, 1].min())))
    y1 = min(height, int(np.ceil(corners[:, 1].max())) + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    inside = MplPath(corners).contains_points(
        np.column_stack((xx.ravel() + 0.5, yy.ravel() + 0.5))
    ).reshape(xx.shape)
    coefficients = np.linalg.lstsq(
        np.column_stack((corners[[0, 1, 3]], np.ones(3))),
        corner_depth[[0, 1, 3]], rcond=None,
    )[0]
    surface_depth = coefficients[0] * xx + coefficients[1] * yy + coefficients[2]
    local_depth = point_depth[y0:y1, x0:x1]
    in_front = np.isfinite(local_depth) & (local_depth > surface_depth + 0.12)
    visible_plane = inside & ~in_front
    local_image = image[y0:y1, x0:x1]
    alpha = np.where(np.isfinite(local_depth), 0.50, 0.37)
    local_image[..., :3][visible_plane] = (
        (1.0 - alpha[visible_plane, None])
        * local_image[..., :3][visible_plane]
        + alpha[visible_plane, None] * np.array([0.52, 0.72, 0.84])
    )
    image[..., 3] = np.isfinite(point_depth).astype(np.float32)
    local_image[..., 3][visible_plane] = 1.0
    return image


def draw_visible_plane_edges(ax, corners, corner_depth, point_depth):
    """Occlude the section outline wherever real points are in front."""
    height, width = point_depth.shape
    segments = []
    for index in range(4):
        next_index = (index + 1) % 4
        start, end = corners[index], corners[next_index]
        length = float(np.linalg.norm(end - start))
        t = np.linspace(0.0, 1.0, max(2, int(np.ceil(length * 1.5))))
        sampled = start[None] + t[:, None] * (end - start)[None]
        mid = (sampled[:-1] + sampled[1:]) / 2.0
        mid_depth = (corner_depth[index]
                     + (t[:-1] + t[1:]) / 2.0
                     * (corner_depth[next_index] - corner_depth[index]))
        ix = np.rint(mid[:, 0]).astype(int)
        iy = np.rint(mid[:, 1]).astype(int)
        in_bounds = (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)
        visible = in_bounds.copy()
        sample_depth = point_depth[iy[in_bounds], ix[in_bounds]]
        visible[in_bounds] = (~np.isfinite(sample_depth)
                              | (mid_depth[in_bounds] >= sample_depth - 0.12))
        dash_distance = (t[:-1] + t[1:]) * 0.5 * length
        visible &= (dash_distance % 20.0) < 11.0
        segments.extend(np.stack((sampled[:-1][visible],
                                  sampled[1:][visible]), axis=1))
    # Dash spacing follows distance along each actual 3D edge; only visible
    # dashes survive the point-depth test.
    ax.add_collection(LineCollection(segments, colors="#111111",
                                     linewidths=1.45, zorder=6))


def main():
    source, xyz, q = load_real_tile()
    low, high = load_thresholds()
    along, section, length = profile_geometry(xyz)
    stations, raw_top, drawn_top = top_surface_envelope(xyz, length)
    markers = representative_points(xyz, q, section, along, stations, raw_top, low, high)
    rng = np.random.default_rng(SEED)
    profile_indices = sampled_indices(section, MAX_PROFILE_POINTS, rng)

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 12.5, "axes.linewidth": 0.7,
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })
    # The established q_geo values concentrate at 0 and 1. Saturated blue-to-
    # vermilion endpoints make the low-observability returns legible without
    # changing any measured score or the fixed 0--1 colorbar domain.
    cmap = colors.LinearSegmentedColormap.from_list(
        "observability_blue_vermilion",
        [(0.0, "#173B8C"), (0.2, "#04A7C9"),
         (0.6, "#FFD13D"), (1.0, "#E83C2E")],
        N=256,
    )
    norm = colors.Normalize(vmin=0, vmax=1)
    fig = plt.figure(figsize=(13.8, 5.35), facecolor="white")
    ax_a = fig.add_axes((0.060, 0.29, 0.470, 0.62))
    ax_b = fig.add_axes((0.570, 0.245, 0.365, 0.73))

    # Put low-q returns on top so the sparse interior points remain visible
    # through the dense, mostly high-q upper surfaces.
    order = profile_indices[np.argsort(q[profile_indices])[::-1]]
    ax_a.scatter(along[order], xyz[order, 2], c=q[order], cmap=cmap, norm=norm,
                 s=3.6, alpha=0.93, linewidths=0, rasterized=True, zorder=2)
    ax_a.plot(stations, drawn_top, color="#8C949A", linewidth=0.95,
              zorder=4, label="Local upper envelope")
    for mark in markers:
        x = mark["distance_m"]
        z = mark["elevation_m"]
        ax_a.scatter([x], [z], s=76, color=cmap(norm(mark["q_geo"])),
                     edgecolor="#20252A", linewidth=0.8, zorder=7)
    ax_a.set_xlim(0, length)
    all_z = xyz[section, 2]
    ax_a.set_ylim(float(np.quantile(all_z, 0.002) - 1.4),
                  float(np.quantile(all_z, 0.998) + 2.8))
    ax_a.set_xlabel("Distance A–A′ (m)", fontsize=14)
    ax_a.set_ylabel("Elevation (m)")
    ax_a.grid(True, color="#D7DCE0", linewidth=0.5, alpha=0.58, zorder=0)
    ax_a.legend(loc="upper right", frameon=False, fontsize=12)

    # Render every measured XYZ point from an oblique orthographic camera.
    # The renderer uses physical Z for occlusion; the `pcv` raster style adds
    # no color shading and retains the measured continuous q_geo palette.
    xy_min = xyz[:, :2].min(axis=0)
    relative_xy = xyz[:, :2] - xy_min
    rgba = cmap(norm(q)).astype(np.float32)
    right, up, view = camera_basis(
        relative_xy[:, 0], relative_xy[:, 1],
        azimuth=VIEW_AZIMUTH_DEG, elevation=VIEW_ELEVATION_DEG,
    )
    camera = (
        right, up, view,
        np.median(np.column_stack((relative_xy, xyz[:, 2])), axis=0),
    )
    image, _camera, projection, point_radius, depth_image = render_point_cloud_image(
        relative_xy[:, 0], relative_xy[:, 1], xyz[:, 2], rgba,
        camera=camera, width=TOP_RENDER_WIDTH, style="pcv",
        point_scale=TOP_POINT_SCALE, min_point_radius=1,
        max_point_radius=7, fractional_splat=True, square_frame=True,
        return_depth=True,
    )
    _u_low, _u_span, _v_low, _v_span, _width, _height, margin = projection
    cropped = image[margin:-margin, margin:-margin].copy()
    cropped_depth = depth_image[margin:-margin, margin:-margin]
    canvas_pad = 85
    cropped = np.pad(cropped, ((canvas_pad, canvas_pad),
                               (canvas_pad, canvas_pad), (0, 0)),
                     constant_values=1.0)
    cropped_depth = np.pad(cropped_depth,
                           ((canvas_pad, canvas_pad),
                            (canvas_pad, canvas_pad)),
                           constant_values=np.nan)
    profile_unit = (PROFILE_END - PROFILE_START) / length
    section_xy = np.vstack((
        PROFILE_START - SECTION_EXTENSION_M * profile_unit,
        PROFILE_END + SECTION_EXTENSION_M * profile_unit,
    )) - xy_min
    lower_z = float(np.quantile(xyz[section, 2], 0.002)
                    - SECTION_Z_BELOW_M)
    upper_z = float(np.quantile(xyz[section, 2], 0.998)
                    + SECTION_Z_ABOVE_M)
    plane_xy = section_xy[[0, 1, 1, 0]]
    plane_z = np.array([lower_z, lower_z, upper_z, upper_z])
    plane_x, plane_y = to_image_pixels(
        plane_xy[:, 0], plane_xy[:, 1], plane_z, camera, projection,
    )
    plane_pixels = np.column_stack((plane_x - margin + canvas_pad,
                                    plane_y - margin + canvas_pad))
    plane_world = np.column_stack((plane_xy, plane_z))
    plane_depth = (plane_world - camera[3]) @ camera[2]
    cropped = composite_section_plane(
        cropped, cropped_depth, plane_pixels, plane_depth,
    )
    box_pixels = draw_xyz_box(ax_b, xyz, xy_min, camera, projection,
                              margin - canvas_pad)
    ax_b.imshow(cropped, origin="lower", interpolation="nearest", zorder=2)
    draw_visible_plane_edges(ax_b, plane_pixels, plane_depth, cropped_depth)
    label_xy = (plane_pixels[3]
                + 0.075 * (plane_pixels[2] - plane_pixels[3])
                + 0.22 * (plane_pixels[0] - plane_pixels[3]))
    ax_b.text(*label_xy, "Profile A–A′", ha="left", va="center",
              fontsize=10.8, fontweight="bold", color="#111111",
              bbox=dict(facecolor="white", alpha=0.82,
                        edgecolor="none", pad=1.2), zorder=7)
    bounds = np.vstack((box_pixels, plane_pixels))
    padding = 24.0
    ax_b.set_xlim(float(bounds[:, 0].min() - padding),
                  float(bounds[:, 0].max() + padding))
    ax_b.set_ylim(float(bounds[:, 1].min() - padding),
                  float(bounds[:, 1].max() + padding))
    ax_b.set_aspect("equal", adjustable="box")
    ax_b.set_xticks([])
    ax_b.set_yticks([])
    for spine in ax_b.spines.values():
        spine.set_visible(False)
    draw_partition(fig, low, high, cmap, norm)
    fig.text(0.295, 0.165, "(a) Geometric observability in an ALS profile",
             ha="center", va="center", fontsize=14.5, fontweight="bold")
    fig.text(0.752, 0.165, "(b) Spatial observability and supervision partition",
             ha="center", va="center", fontsize=14.5, fontweight="bold")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=450, facecolor="white",
                bbox_inches="tight", pad_inches=0.09)
    plt.close(fig)
    metadata = {
        "schema": "cvrl_geometric_observability_method_v1",
        "source_las": str(source),
        "q_geo_source": str(AUDIT_ROOT / "visibility" / f"{STEM}.safetensors"),
        "full_tile_points": int(len(xyz)),
        "profile_full_points": int(len(section)),
        "top_display_points": int(len(xyz)),
        "profile_display_points": int(len(profile_indices)),
        "top_renderer": {
            "name": "adaptive circular point splats with physical-height occlusion",
            "style": "pcv raster, no added PCV or EDL shading",
            "azimuth_degrees": VIEW_AZIMUTH_DEG,
            "elevation_degrees": VIEW_ELEVATION_DEG,
            "overlap_priority": "frontmost real XYZ point per projected pixel",
            "width_pixels": TOP_RENDER_WIDTH,
            "point_scale": TOP_POINT_SCALE,
            "point_radius_pixels": float(point_radius),
            "fractional_splat": True,
        },
        "profile_endpoints_xy": [PROFILE_START.tolist(), PROFILE_END.tolist()],
        "profile_half_width_m": PROFILE_HALF_WIDTH_M,
        "section_plane_elevation_m": [lower_z, upper_z],
        "section_extension_m_each_end": SECTION_EXTENSION_M,
        "section_plane_style": "camera-depth-tested translucent sheet and occlusion-tested black outline",
        "envelope": "maximum Z from complete tile within 1.1 m XY radius of 0.4 m transect stations; Gaussian-smoothed only for line display",
        "tau_low": low, "tau_high": high,
        "display_colormap": "continuous saturated blue--cyan--gold--vermilion, fixed q_geo range [0,1]",
        "representative_points": markers,
    }
    METADATA.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {OUTPUT}")
    print(f"Saved provenance {METADATA}")


if __name__ == "__main__":
    main()
