"""Section 4.3 projection-conflict figures (IDE/module-level settings)."""

from __future__ import annotations

import json
from pathlib import Path

import laspy
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from matplotlib import patheffects
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from PIL import Image
from safetensors import safe_open

from plot_cvrl_43_results import (
    AUDIT_ROOT,
    CLASS_NAMES,
    OUTPUT_DIR,
    REPORT_FIGURE_DIR,
    PAPER_CLASS_NAMES,
    PALETTE,
    REGION_COLORS,
    _set_publication_style,
)
from utils.pointcloud_render import (
    compute_pcv,
    modulate_rgba_with_pcv,
    render_point_cloud_image,
    to_image_pixels,
)


CONFLICT_PATH = AUDIT_ROOT / "analysis" / "section_43_projection_conflict.json"
CASE_ROOT = Path(r"G:\data\swiss\joint_tiles")
IMAGE_ROOT = CASE_ROOT / "image"
CASE_VISIBILITY_ROOT = OUTPUT_DIR / "case_visibility_200m"
CASE_VISIBILITY_SCHEMA = "pointspace_section43_200m_case_visibility_v1"
CASES = (
    (
        "Under-canopy ground",
        "swisssurface3d_2023_2615-1179_2056_5728_0002",
        2,
    ),
    (
        "Building facade",
        "swisssurface3d_2024_2679-1248_2056_5728.copc_0006",
        26,
    ),
)
# Explicit illustrative transect for the forest/field boundary. Pixel pairs
# are (row, column) in the original 2000 px orthophoto; the affine fit below
# maps them back to the exact ALS XY line.
CASE_SECTION_PIXELS = {
    "swisssurface3d_2023_2615-1179_2056_5728_0002": (
        (100.0, 100.0), (1050.0, 1250.0)
    ),
}
# Post-hoc audit examples: each selected 16 px DINO patch has a high-purity
# visible LAS surface proxy and many projected P_inv points of another class.
# These IDs are validated against the same support/purity rule as Fig. 6(a).
CASE_CONFLICT_PATCHES = {
    "swisssurface3d_2023_2615-1179_2056_5728_0002": (4157, 3),
    "swisssurface3d_2024_2679-1248_2056_5728.copc_0006": (7311, 6),
}
CLASS_COLORS = {
    1: "#A0A4AA",
    2: "#B7A780",
    3: "#39865A",
    6: "#C67C5E",
    17: "#D2A451",
    26: "#7768AE",
    27: "#7768AE",
}
VISIBILITY_REGION_COLORS = (
    REGION_COLORS["P_inv"], REGION_COLORS["P_amb"], REGION_COLORS["P_vis"],
)
PROFILE_HALF_WIDTH_M = 0.25
PROFILE_HALF_LENGTH_M = 80.0


def _interval_error(value):
    return np.asarray([[value["conflict_rate"] - value["ci95"][0]],
                       [value["ci95"][1] - value["conflict_rate"]]]) * 100


def plot_conflict_rates(data):
    summary = data["summary"]
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.1), constrained_layout=True)
    regions = ("P_vis", "P_amb", "P_inv")
    colors = (PALETTE["blue"], PALETTE["gold"], PALETTE["vermillion"])
    y = np.arange(len(regions))
    values = np.asarray([summary["all"][name]["conflict_rate"] for name in regions]) * 100
    lower = np.asarray([summary["all"][name]["ci95"][0] for name in regions]) * 100
    upper = np.asarray([summary["all"][name]["ci95"][1] for name in regions]) * 100
    axes[0].barh(y, values, color=colors, height=0.55)
    axes[0].errorbar(
        values, y, xerr=np.vstack((values - lower, upper - values)),
        fmt="none", ecolor="#292929", capsize=3, linewidth=1,
    )
    axes[0].set_yticks(y, [r"$P_{vis}$", r"$P_{amb}$", r"$P_{inv}$"])
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 55)
    axes[0].set_xlabel("Patch-class conflict (%)")
    axes[0].set_title("(a) All eligible audit points")
    axes[0].grid(axis="x", alpha=0.35)
    for yi, value in enumerate(values):
        axes[0].text(value + 2, yi, f"{value:.1f}%", va="center", fontsize=10)

    classes = (2, 3, 6, 17)
    class_y = np.arange(len(classes))
    for offset, region, label, color in (
        (-0.11, "P_vis", r"$P_{vis}$", PALETTE["blue"]),
        (0.11, "P_inv", r"$P_{inv}$", PALETTE["vermillion"]),
    ):
        values = [summary[str(cls)][region] for cls in classes]
        axes[1].errorbar(
            [value["conflict_rate"] * 100 for value in values],
            class_y + offset,
            xerr=np.concatenate([_interval_error(value) for value in values], axis=1),
            fmt="o", capsize=3, markersize=6, color=color, label=label,
        )
    axes[1].set_yticks(class_y, [CLASS_NAMES[str(cls)] for cls in classes])
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 100)
    axes[1].set_xlabel("Patch-class conflict (%)")
    axes[1].set_title("(b) Selected LAS classes")
    axes[1].grid(axis="x", alpha=0.35)
    axes[1].legend(loc="lower right", fontsize=10)
    REPORT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(REPORT_FIGURE_DIR / "figure_43_projection_conflict.png")
    plt.close(fig)


def _read_case(stem):
    source = CASE_ROOT / "pointcloud" / f"{stem}.las"
    cloud = laspy.read(source)
    with safe_open(
        str(CASE_ROOT / "correspondence" / f"{stem}.safetensors"),
        framework="np",
    ) as handle:
        pixel = np.asarray(handle.get_tensor("pixel_coord"))
    with safe_open(
        str(CASE_VISIBILITY_ROOT / f"{stem}.safetensors"),
        framework="np",
    ) as handle:
        region = np.asarray(handle.get_tensor("region"))
        metadata = handle.metadata() or {}
    if (metadata.get("schema") != CASE_VISIBILITY_SCHEMA
            or metadata.get("source_bytes") != str(source.stat().st_size)):
        raise ValueError(f"{stem}: stale or incompatible 200 m visibility sidecar")
    xyz = np.column_stack(
        (np.asarray(cloud.x), np.asarray(cloud.y), np.asarray(cloud.z))
    )
    if len(region) != len(xyz) or len(pixel) != len(xyz):
        raise ValueError(f"{stem}: point/visibility/correspondence counts differ")
    return xyz, np.asarray(cloud.classification), region, pixel


def _select_center(pixel, classification, region, class_id, image_shape):
    target = (classification == class_id) & (region == 0)
    if target.sum() == 0:
        raise ValueError(f"No P_inv class {class_id} points for case")
    # Keep the illustrative transect away from image/tile edges when possible.
    margin = max(185, int(0.35 * min(image_shape[:2])))
    interior = (
        target
        & (pixel[:, 0] >= margin)
        & (pixel[:, 0] < image_shape[0] - margin)
        & (pixel[:, 1] >= margin)
        & (pixel[:, 1] < image_shape[1] - margin)
    )
    if interior.sum() >= 100:
        target = interior
    block = 60
    ncols = int(np.ceil(image_shape[1] / block))
    cell = (pixel[target, 0] // block) * ncols + pixel[target, 1] // block
    best = np.bincount(cell).argmax()
    selected = np.flatnonzero(target)[cell == best]
    return np.median(pixel[selected], axis=0), selected


def _sample(index, limit, seed):
    if len(index) <= limit:
        return index
    return np.sort(np.random.default_rng(seed).choice(index, limit, replace=False))


def _conflict_patch(stem, classification, region, pixel, target_class):
    """Verify a concrete example using the Fig. 6(a) LAS surface-proxy rule."""
    patch_id, expected_proxy = CASE_CONFLICT_PATCHES[stem]
    with safe_open(
        str(CASE_ROOT / "correspondence" / f"{stem}.safetensors"),
        framework="np",
    ) as handle:
        patch_index = np.asarray(handle.get_tensor("patch_index"))
        valid = np.asarray(handle.get_tensor("valid"), dtype=bool)
        metadata = handle.metadata() or {}
    if len(patch_index) != len(classification) or len(valid) != len(pixel):
        raise ValueError(f"{stem}: correspondence point count mismatch")
    patch_size = int(metadata["patch_size"])
    grid_cols, grid_rows = json.loads(metadata["feature_grid_size"])
    patch_count = grid_cols * grid_rows
    if not (0 <= patch_id < patch_count):
        raise ValueError(f"{stem}: example patch outside correspondence grid")
    projected_patch = (
        (pixel[:, 0] // patch_size) * grid_cols + pixel[:, 1] // patch_size
    )
    actual = patch_index >= 0
    if not np.array_equal(projected_patch[actual], patch_index[actual]):
        raise ValueError(f"{stem}: pixel/patch mapping mismatch")
    surface = valid & (patch_index == patch_id) & (classification >= 0)
    surface_classes = np.bincount(classification[surface].astype(int))
    support = int(surface_classes.sum())
    proxy_class = int(surface_classes.argmax()) if support else -1
    purity = float(surface_classes.max() / support) if support else 0.0
    if support < 5 or purity < 0.70 or proxy_class != expected_proxy:
        raise ValueError(
            f"{stem}: example surface proxy changed: class={proxy_class}, "
            f"support={support}, purity={purity:.3f}"
        )
    target_index = np.flatnonzero(
        (projected_patch == patch_id)
        & (classification == target_class)
        & (region == 0)
    )
    if not len(target_index):
        raise ValueError(f"{stem}: no P_inv target points in example patch")
    return {
        "id": patch_id,
        "size": patch_size,
        "row": (patch_id // grid_cols) * patch_size,
        "col": (patch_id % grid_cols) * patch_size,
        "proxy_class": proxy_class,
        "support": support,
        "purity": purity,
        "target_index": target_index,
    }


def _annotate_conflict_orthophoto(ax, image, pixel, patch, target_class,
                                  row_lo, col_lo):
    """Locate the exact patch, then magnify it just enough to resolve targets."""
    row, col, size = patch["row"], patch["col"], patch["size"]
    gold, red = PALETTE["gold"], PALETTE["vermillion"]
    ax.add_patch(Rectangle(
        (col - col_lo - 0.5, row - row_lo - 0.5), size, size,
        facecolor=gold, alpha=0.28, edgecolor="none", zorder=10,
    ))
    ax.add_patch(Rectangle(
        (col - col_lo - 0.5, row - row_lo - 0.5), size, size,
        facecolor="none", edgecolor="white", linewidth=2.0, zorder=11,
    ))
    marker_x = col - col_lo + size / 2
    marker_y = row - row_lo + size / 2
    for edgecolor, linewidth, zorder in (
        ("white", 3.2, 12), (red, 1.9, 13)
    ):
        ax.plot(
            marker_x, marker_y, marker="o", linestyle="none",
            markersize=14, markerfacecolor="none",
            markeredgecolor=edgecolor, markeredgewidth=linewidth,
            zorder=zorder,
        )

    # A 16 px patch is only ~8 px in the full 200 m panel. The small in-panel
    # inset keeps its true footprint while revealing the same pixels and points.
    pad = 56
    center_row, center_col = row + size // 2, col + size // 2
    crop_row = max(0, center_row - pad)
    crop_col = max(0, center_col - pad)
    crop_bottom = min(image.shape[0], center_row + pad)
    crop_right = min(image.shape[1], center_col + pad)
    inset = ax.inset_axes((0.515, 0.025, 0.465, 0.465), zorder=20)
    inset.imshow(image[crop_row:crop_bottom, crop_col:crop_right])
    inset.add_patch(Rectangle(
        (col - crop_col - 0.5, row - crop_row - 0.5), size, size,
        facecolor=gold, alpha=0.30, edgecolor="none", zorder=3,
    ))
    inset.add_patch(Rectangle(
        (col - crop_col - 0.5, row - crop_row - 0.5), size, size,
        facecolor="none", edgecolor="white", linewidth=1.4, zorder=4,
    ))
    target_pixels = np.unique(pixel[patch["target_index"]], axis=0)
    target_pixels = target_pixels[
        (target_pixels[:, 0] >= crop_row) & (target_pixels[:, 0] < crop_bottom)
        & (target_pixels[:, 1] >= crop_col) & (target_pixels[:, 1] < crop_right)
    ]
    if len(target_pixels) > 14:
        target_pixels = target_pixels[np.linspace(
            0, len(target_pixels) - 1, 14, dtype=int
        )]
    inset.scatter(
        target_pixels[:, 1] - crop_col, target_pixels[:, 0] - crop_row,
        s=24, color=red, edgecolors="white", linewidths=0.65,
        zorder=5,
    )
    inset.text(
        0.02, 0.02,
        f"Surface proxy:\n{PAPER_CLASS_NAMES[str(patch['proxy_class'])]}",
        transform=inset.transAxes, ha="left", va="bottom", fontsize=12.5,
        color="#172530", bbox={"facecolor": "white", "alpha": 0.94,
                               "edgecolor": "none", "pad": 2.5}, zorder=7,
    )
    inset.text(
        0.98, 0.98,
        rf"$P_{{inv}}$ target:" + "\n" + PAPER_CLASS_NAMES[str(target_class)],
        transform=inset.transAxes, ha="right", va="top", fontsize=12.5,
        color=red, bbox={"facecolor": "white", "alpha": 0.94,
                         "edgecolor": "none", "pad": 2.5}, zorder=7,
    )
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_color("white")
        spine.set_linewidth(1.6)
    arrow_destination = (0.54, 0.49) if target_class == 2 else (0.75, 0.49)
    arrow = ax.annotate(
        "", xy=arrow_destination, xycoords="axes fraction",
        xytext=(marker_x, marker_y), textcoords="data",
        arrowprops={"arrowstyle": "-|>", "color": red, "linewidth": 1.8,
                    "mutation_scale": 15, "shrinkA": 10, "shrinkB": 3},
        zorder=19,
    )
    arrow.arrow_patch.set_path_effects([
        patheffects.Stroke(linewidth=3.8, foreground="white"),
        patheffects.Normal(),
    ])


def fit_xy_to_pixel_affine(xy, pixel):
    """Fit row/column coordinates without losing precision at Swiss map offsets."""
    xy = np.asarray(xy, dtype=np.float64)
    pixel = np.asarray(pixel, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2 or pixel.shape != xy.shape:
        raise ValueError("xy and pixel must both be [N, 2]")
    origin = xy.mean(axis=0)
    design = np.column_stack((xy - origin, np.ones(len(xy))))
    affine = np.linalg.lstsq(design, pixel, rcond=None)[0]
    residual = np.linalg.norm(pixel - design @ affine, axis=1)
    return origin, affine, float(np.median(residual))


def _section_line(ax, x, y, *, fontsize):
    """Mark the same dashed A-A' transect in the image and ALS top view."""
    line, = ax.plot(
        x, y, color="white", linewidth=2.5, linestyle=(0, (5, 4)), zorder=8,
    )
    line.set_path_effects([
        patheffects.Stroke(linewidth=4.1, foreground="#21313E"),
        patheffects.Normal(),
    ])
    midpoint = np.asarray([(x[0] + x[1]) / 2, (y[0] + y[1]) / 2])
    label = ax.annotate(
        "A-A'", xy=midpoint, xytext=(5, -5), textcoords="offset points",
        ha="left", va="top", color="white", fontsize=fontsize,
        fontweight="bold", zorder=9,
    )
    label.set_path_effects([
        patheffects.Stroke(linewidth=2.8, foreground="#21313E"),
        patheffects.Normal(),
    ])


def plot_projection_cases(
    data,
    *,
    fig=None,
    axes=None,
    row_labels=None,
    column_titles=None,
    output_path=None,
    show_case_count=True,
    font_scale=1.0,
    legend_anchor=None,
):
    """Draw the two spatial examples, optionally into a larger composite figure."""
    standalone = fig is None
    if standalone:
        fig, axes = plt.subplots(2, 3, figsize=(14.7, 8.7), constrained_layout=True)
    if axes is None or np.shape(axes) not in ((2, 3), (2, 4)):
        raise ValueError("axes must be a 2 x 3 or 2 x 4 array")
    show_visibility_profile = np.shape(axes)[1] == 4
    if row_labels is None:
        row_labels = [case[0] for case in CASES]
    if column_titles is None:
        column_titles = (
            ("Orthophoto", "ALS top view", "Elevation profile: class",
             "Elevation profile: visibility") if show_visibility_profile
            else ("Orthophoto", "ALS top view", "Elevation profile")
        )
    if len(column_titles) != np.shape(axes)[1]:
        raise ValueError("column_titles must match the number of axes columns")
    tile_lookup = {tile["stem"]: tile for tile in data["tiles"]}
    for row, (label, stem, class_id) in enumerate(CASES):
        xyz, classification, region, pixel = _read_case(stem)
        image = np.asarray(Image.open(IMAGE_ROOT / f"{stem}.jpg").convert("RGB"))
        conflict_patch = _conflict_patch(
            stem, classification, region, pixel, class_id
        )
        center_pixel, center_index = _select_center(
            pixel, classification, region, class_id, image.shape
        )
        row_center, col_center = center_pixel
        half = 1000
        row_lo = int(np.clip(row_center - half, 0, max(image.shape[0] - 2 * half, 0)))
        col_lo = int(np.clip(col_center - half, 0, max(image.shape[1] - 2 * half, 0)))
        row_hi = min(row_lo + 2 * half, image.shape[0])
        col_hi = min(col_lo + 2 * half, image.shape[1])
        in_crop = (
            (pixel[:, 0] >= row_lo) & (pixel[:, 0] < row_hi)
            & (pixel[:, 1] >= col_lo) & (pixel[:, 1] < col_hi)
        )
        target_inv = in_crop & (classification == class_id) & (region == 0)

        center_xy = np.median(xyz[center_index, :2], axis=0)
        local_target = xyz[np.flatnonzero(target_inv), :2]
        local_target = local_target[
            np.linalg.norm(local_target - center_xy, axis=1) < 18
        ]
        if len(local_target) >= 3:
            eigenvalues, eigenvectors = np.linalg.eigh(
                np.cov(local_target, rowvar=False)
            )
            tangent = eigenvectors[:, np.argmax(eigenvalues)]
            direction = np.asarray([-tangent[1], tangent[0]]) if class_id == 26 else tangent
        else:
            direction = np.asarray([1.0, 0.0])
        normal = np.asarray([-direction[1], direction[0]])
        endpoints = center_xy[None, :] + np.asarray(
            [-PROFILE_HALF_LENGTH_M, PROFILE_HALF_LENGTH_M]
        )[:, None] * direction
        fit_index = _sample(np.flatnonzero(in_crop), 8000, 67)
        xy_origin, affine, residual_px = fit_xy_to_pixel_affine(
            xyz[fit_index, :2], pixel[fit_index]
        )
        if residual_px > 2.0:
            raise ValueError(
                f"{stem}: XYZ-to-image fit residual {residual_px:.2f} px"
            )
        endpoint_pixels = np.column_stack((
            endpoints - xy_origin, np.ones(2)
        )) @ affine
        section_half_length = PROFILE_HALF_LENGTH_M
        if stem in CASE_SECTION_PIXELS:
            endpoint_pixels = np.asarray(CASE_SECTION_PIXELS[stem], dtype=np.float64)
            endpoints = xy_origin + (endpoint_pixels - affine[2]) @ np.linalg.inv(affine[:2])
            center_xy = endpoints.mean(axis=0)
            vector = endpoints[1] - endpoints[0]
            section_half_length = float(np.linalg.norm(vector) / 2)
            direction = vector / np.linalg.norm(vector)
            normal = np.asarray([-direction[1], direction[0]])

        ax = axes[row, 0]
        ax.imshow(image[row_lo:row_hi, col_lo:col_hi])
        _section_line(
            ax, endpoint_pixels[:, 1] - col_lo,
            endpoint_pixels[:, 0] - row_lo, fontsize=10 * font_scale,
        )
        _annotate_conflict_orthophoto(
            ax, image, pixel, conflict_patch, class_id, row_lo, col_lo
        )
        ax.set_xlim(-0.5, col_hi - col_lo - 0.5)
        ax.set_ylim(row_hi - row_lo - 0.5, -0.5)
        ax.set_box_aspect(1)
        ax.set_xticks([])
        ax.set_yticks([])

        ax = axes[row, 1]
        crop_index = np.flatnonzero(in_crop)
        display_local = _sample(np.arange(len(crop_index)), 120000, 63)
        display_index = crop_index[display_local]
        pcv = compute_pcv(
            xyz[display_index].astype(np.float32, copy=False),
            number_of_directions=128, resolution=512, point_radius=0,
        )
        rgba = np.asarray([
            mcolors.to_rgba(CLASS_COLORS.get(int(cls), "#A0A4AA"))
            for cls in classification[display_index]
        ], dtype=np.float32)
        shaded = modulate_rgba_with_pcv(rgba, pcv, normalize=False)
        right_xy = affine[:2, 1].astype(np.float64)
        right_xy /= np.linalg.norm(right_xy)
        up_xy = -affine[:2, 0].astype(np.float64)
        up_xy -= (up_xy @ right_xy) * right_xy
        up_xy /= np.linalg.norm(up_xy)
        camera = (
            np.r_[right_xy, 0.0], np.r_[up_xy, 0.0],
            np.asarray([0.0, 0.0, 1.0]),
            np.median(xyz[crop_index], axis=0),
        )
        rendered, camera, projection, _radius = render_point_cloud_image(
            xyz[display_index, 0], xyz[display_index, 1],
            xyz[display_index, 2], shaded,
            camera=camera, width=820, strength=4.0,
            style="pcv_edl", point_scale=0.86,
            fractional_splat=True, max_point_radius=7,
            square_frame=True,
        )
        ax.imshow(rendered, origin="lower", interpolation="nearest")
        z_line = np.full(2, np.median(xyz[crop_index, 2]))
        line_x, line_y = to_image_pixels(
            endpoints[:, 0], endpoints[:, 1], z_line,
            camera, projection,
        )
        _section_line(ax, line_x, line_y, fontsize=10 * font_scale)
        target_xyz = np.median(xyz[conflict_patch["target_index"]], axis=0)
        target_x, target_y = to_image_pixels(
            np.asarray([target_xyz[0]]), np.asarray([target_xyz[1]]),
            np.asarray([target_xyz[2]]), camera, projection,
        )
        ax.plot(
            target_x, target_y, marker="o", linestyle="none",
            markersize=14, markerfacecolor="none", markeredgecolor="white",
            markeredgewidth=3.2, zorder=12,
        )
        ax.plot(
            target_x, target_y, marker="o", linestyle="none",
            markersize=14, markerfacecolor="none",
            markeredgecolor=PALETTE["vermillion"],
            markeredgewidth=1.9, zorder=13,
        )
        ax.set_xlim(-0.5, rendered.shape[1] - 0.5)
        ax.set_ylim(-0.5, rendered.shape[0] - 0.5)
        ax.set_aspect("equal")
        ax.set_box_aspect(1)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        delta = xyz[:, :2] - center_xy
        along = delta @ direction
        across = delta @ normal
        profile = in_crop & (np.abs(along) <= section_half_length) & (np.abs(across) <= PROFILE_HALF_WIDTH_M)
        profile_index = _sample(np.flatnonzero(profile), 25000, 65)
        ax = axes[row, 2]
        if not len(profile_index):
            raise ValueError(f"{stem}: no points within the 0.5 m section slab")
        low, high = np.percentile(xyz[profile_index, 2], [0.2, 99.8])
        z_limits = (low - 1.5, high + 1.5)
        ax.scatter(
            along[profile_index], xyz[profile_index, 2],
            s=6.0,
            c=[CLASS_COLORS.get(int(cls), "#A0A4AA") for cls in classification[profile_index]],
            alpha=0.84, rasterized=True, zorder=3,
        )
        ax.set_xlim(-section_half_length, section_half_length)
        ax.set_ylim(*z_limits)
        ax.set_xlabel("Distance A-A' (m)")
        ax.set_ylabel("Elevation (m)")
        ax.grid(alpha=0.25)
        if show_visibility_profile:
            visibility_ax = axes[row, 3]
            visibility_ax.scatter(
                along[profile_index], xyz[profile_index, 2],
                s=6.0,
                c=[VISIBILITY_REGION_COLORS[int(code)] for code in region[profile_index]],
                alpha=0.84, rasterized=True,
            )
            visibility_ax.set_xlim(ax.get_xlim())
            visibility_ax.set_ylim(ax.get_ylim())
            visibility_ax.set_xlabel("Distance A-A' (m)")
            visibility_ax.grid(alpha=0.25)
            visibility_ax.tick_params(axis="y", labelleft=False)
        if show_case_count and stem in tile_lookup:
            ax.text(
                0.02, 0.96,
                f"P_inv target: {tile_lookup[stem]['P_inv_ground_conflict'] if class_id == 2 else tile_lookup[stem]['P_inv_facade_conflict']} "
                "conflicting audit points in tile",
                transform=ax.transAxes, va="top", ha="left", fontsize=9 * font_scale,
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
            )

        axes[row, 0].set_ylabel(row_labels[row], fontsize=11 * font_scale)

    for col, title in enumerate(column_titles):
        axes[0, col].set_title(title, fontsize=11 * font_scale)
    legend_handles = [
        *(Line2D([0], [0], marker="o", linestyle="", color=color,
                  markersize=9 * font_scale, label=PAPER_CLASS_NAMES[str(cls)])
          for cls, color in ((1, CLASS_COLORS[1]), (2, CLASS_COLORS[2]), (3, CLASS_COLORS[3]),
                             (6, CLASS_COLORS[6]), (26, CLASS_COLORS[26]))),
    ]
    if show_visibility_profile:
        legend_handles.extend(
            Line2D([0], [0], marker="o", linestyle="", color=REGION_COLORS[name],
                   markersize=9 * font_scale, label=rf"$P_{{{name[2:]}}}$")
            for name in ("P_vis", "P_amb", "P_inv")
        )
    legend_options = {
        "handles": legend_handles,
        "loc": "outside lower center" if legend_anchor is None else "lower center",
        "ncol": len(legend_handles),
        "fontsize": 10 * font_scale,
        "handlelength": 1.0,
        "handletextpad": 0.4,
        "columnspacing": 1.2,
    }
    if legend_anchor is not None:
        legend_options["bbox_to_anchor"] = legend_anchor
    fig.legend(**legend_options)
    if standalone:
        REPORT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path or REPORT_FIGURE_DIR / "figure_43_projection_mismatch_cases.png")
        plt.close(fig)


def main():
    _set_publication_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = json.loads(CONFLICT_PATH.read_text(encoding="utf-8"))
    plot_conflict_rates(data)
    plot_projection_cases(data)


if __name__ == "__main__":
    main()
