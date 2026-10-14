"""Compare PCV context extents on nested crops of the same Swiss source LAS.

All settings are module-level for direct IDE execution.  Raw source files are
read once per scene, and no LAS data are modified.  Full-extent figures compare
raw PCV, PCV-modulated colors, edge-only EDL, and Lit-EDL.  Additional figures
hold the central 100 m view fixed while varying its PCV context.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import laspy
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from matplotlib.patches import Patch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from utils.pointcloud_render import (  # noqa: E402
    compute_pcv,
    modulate_rgba_with_pcv,
    render_point_cloud_axis,
)


AUDIT_ROOT = Path(r"G:\data\swiss\joint_tiles_100\Swiss-Audit")
RAW_ROOT = Path(r"G:\data\swiss\swiss_cvrl_70km2")
OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "section_43"
ENVIRONMENTS = ("open_ground", "urban", "dense_vegetation")
EXTENTS_METERS = (50, 100, 200, 400)
FIXED_CORE_METERS = 100
PCV_DIRECTIONS = 128
PCV_PIXELS_PER_METER = 2.0
PCV_MIN_RESOLUTION = 512
PCV_MAX_RESOLUTION = 2048
PCV_POINT_RADIUS = 0
DISPLAY_WIDTH = 760
DISPLAY_MAX_POINTS = 50_000
DISPLAY_POINT_SCALE = 0.95
EDL_STRENGTH = 3.0
RANDOM_SEED = 42

CLASS_COLORS = {
    1: "#969BA4",  # unclassified
    2: "#C7B78E",  # ground
    3: "#4C8B5F",  # vegetation
    4: "#4C8B5F",
    5: "#4C8B5F",
    6: "#C97756",  # roof
    9: "#5E96BE",  # water
    14: "#8B74A9",  # wire/transport
    15: "#8B74A9",
    17: "#D2A451",  # bridge
    26: "#C97756",
    27: "#D2A451",
}


def _set_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9,
            "axes.titlesize": 10,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 320,
            "savefig.bbox": "tight",
        }
    )


def _select_scenes():
    summary = json.loads(
        (AUDIT_ROOT / "analysis" / "q_geo_summary.json").read_text(
            encoding="utf-8"
        )
    )
    source_files = {
        path.stem: path
        for path in RAW_ROOT.rglob("*.las")
    }
    source_files.update(
        {path.stem: path for path in RAW_ROOT.rglob("*.laz")}
    )
    scenes = []
    for environment in ENVIRONMENTS:
        candidates = [
            tile
            for tile in summary["tile_stats"]
            if tile["environment"] == environment
        ]
        candidates.sort(
            key=lambda tile: tile["region_count"][0] / tile["points"]
        )
        tile = candidates[len(candidates) // 2]
        stem = tile["stem"]
        source_stem = stem.rsplit("_", 1)[0]
        if source_stem not in source_files:
            raise FileNotFoundError(f"Source LAS not found for {stem}")
        with laspy.open(AUDIT_ROOT / "q_geo_pointcloud" / f"{stem}.laz") as reader:
            minimum, maximum = reader.header.mins, reader.header.maxs
        center_xy = (minimum[:2] + maximum[:2]) / 2.0
        scenes.append((environment, source_files[source_stem], center_xy))
    return scenes


def _read_largest_crop(path, center_xy, largest_extent):
    """Stream the 1 km LAS and retain only the largest comparison square."""
    center_x, center_y = center_xy
    half = largest_extent / 2.0
    xyz_chunks = []
    class_chunks = []
    with laspy.open(path) as reader:
        for chunk in reader.chunk_iterator(1_000_000):
            x = np.asarray(chunk.x)
            y = np.asarray(chunk.y)
            keep = (
                (x >= center_x - half)
                & (x < center_x + half)
                & (y >= center_y - half)
                & (y < center_y + half)
            )
            if not np.any(keep):
                continue
            xyz_chunks.append(
                np.column_stack(
                    (x[keep] - center_x, y[keep] - center_y, np.asarray(chunk.z)[keep])
                ).astype(np.float32)
            )
            class_chunks.append(
                np.asarray(chunk.classification)[keep].astype(np.uint8)
            )
    if not xyz_chunks:
        raise ValueError(f"No points found around {center_xy} in {path}")
    xyz = np.concatenate(xyz_chunks, axis=0)
    classes = np.concatenate(class_chunks, axis=0)
    return xyz, classes


def _crop_mask(xyz, extent):
    half = extent / 2.0
    return (
        (xyz[:, 0] >= -half)
        & (xyz[:, 0] < half)
        & (xyz[:, 1] >= -half)
        & (xyz[:, 1] < half)
    )


def _class_rgba(classes):
    rgba = np.tile(
        np.asarray(mcolors.to_rgba("#969BA4"), dtype=np.float32),
        (len(classes), 1),
    )
    for class_id, color in CLASS_COLORS.items():
        rgba[classes == class_id] = mcolors.to_rgba(color)
    return rgba


def _sample_indices(indices, rng):
    if len(indices) <= DISPLAY_MAX_POINTS:
        return indices
    return np.sort(rng.choice(indices, DISPLAY_MAX_POINTS, replace=False))


def _render(ax, xyz, rgba, camera=None, style="pcv_lit_edl"):
    camera, _, radius = render_point_cloud_axis(
        ax,
        xyz[:, 0],
        xyz[:, 1],
        xyz[:, 2],
        rgba,
        camera=camera,
        width=DISPLAY_WIDTH,
        strength=EDL_STRENGTH,
        style=style,
        point_scale=DISPLAY_POINT_SCALE,
    )
    return camera, radius


def _annotate(ax, text):
    ax.text(
        0.025,
        0.025,
        text,
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=7.5,
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.82,
        },
    )


def run():
    _set_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scenes = _select_scenes()
    rows = []
    overview_fig, overview_axes = plt.subplots(
        len(scenes), len(EXTENTS_METERS),
        figsize=(16.8, 3.5 * len(scenes)), constrained_layout=True,
    )
    pure_fig, pure_axes = plt.subplots(
        len(scenes), len(EXTENTS_METERS),
        figsize=(16.8, 3.5 * len(scenes)), constrained_layout=True,
    )
    color_fig, color_axes = plt.subplots(
        len(scenes), len(EXTENTS_METERS),
        figsize=(16.8, 3.5 * len(scenes)), constrained_layout=True,
    )
    edge_fig, edge_axes = plt.subplots(
        len(scenes), len(EXTENTS_METERS),
        figsize=(16.8, 3.5 * len(scenes)), constrained_layout=True,
    )
    context_extents = tuple(
        extent for extent in EXTENTS_METERS if extent >= FIXED_CORE_METERS
    )
    core_fig, core_axes = plt.subplots(
        len(scenes), len(context_extents),
        figsize=(13.0, 3.5 * len(scenes)), constrained_layout=True,
    )
    raw_fig, raw_axes = plt.subplots(
        len(scenes), len(context_extents),
        figsize=(13.0, 3.5 * len(scenes)), constrained_layout=True,
    )

    for row_index, (environment, path, center_xy) in enumerate(scenes):
        print(f"[{environment}] Reading {path.name}", flush=True)
        xyz, classes = _read_largest_crop(
            path, center_xy, max(EXTENTS_METERS)
        )
        projection_center = np.asarray(
            [0.0, 0.0, float(np.median(xyz[:, 2]))], dtype=np.float32
        )
        core_indices = np.flatnonzero(_crop_mask(xyz, FIXED_CORE_METERS))
        display_rng = np.random.default_rng(RANDOM_SEED + row_index)
        core_display_indices = _sample_indices(core_indices, display_rng)
        print(
            f"[{environment}] largest crop: {len(xyz):,} points; "
            f"fixed core: {len(core_indices):,} points",
            flush=True,
        )

        for column_index, extent in enumerate(EXTENTS_METERS):
            indices = np.flatnonzero(_crop_mask(xyz, extent))
            points = xyz[indices]
            resolution = int(
                np.clip(
                    np.ceil(extent * PCV_PIXELS_PER_METER * 2.2 / 64) * 64,
                    PCV_MIN_RESOLUTION,
                    PCV_MAX_RESOLUTION,
                )
            )
            print(
                f"[{environment}] {extent} m: {len(indices):,} points, "
                f"PCV buffer {resolution}x{resolution}",
                flush=True,
            )
            pcv = compute_pcv(
                points,
                number_of_directions=PCV_DIRECTIONS,
                resolution=resolution,
                point_radius=PCV_POINT_RADIUS,
                projection_center=projection_center,
                pixel_size=1.0 / PCV_PIXELS_PER_METER,
                strict_bounds=True,
            )
            display_indices = _sample_indices(indices, display_rng)
            display_local = np.searchsorted(indices, display_indices)
            display_xyz = xyz[display_indices]
            display_rgba = modulate_rgba_with_pcv(
                _class_rgba(classes[display_indices]),
                pcv[display_local],
                normalize=False,
            )
            camera, radius = _render(
                overview_axes[row_index, column_index],
                display_xyz,
                display_rgba,
            )
            _annotate(
                overview_axes[row_index, column_index],
                f"{len(indices):,} source points | r={radius} px",
            )
            pure_camera, pure_radius = _render(
                pure_axes[row_index, column_index],
                display_xyz,
                plt.get_cmap("gray")(pcv[display_local]),
                camera=camera,
                style="pcv",
            )
            assert pure_camera is camera
            _annotate(
                pure_axes[row_index, column_index],
                f"mean PCV={float(np.mean(pcv)):.3f} | r={pure_radius} px",
            )
            for axes, style in (
                (color_axes, "pcv"),
                (edge_axes, "pcv_edl"),
            ):
                _, style_radius = _render(
                    axes[row_index, column_index],
                    display_xyz,
                    display_rgba,
                    camera=camera,
                    style=style,
                )
                _annotate(
                    axes[row_index, column_index],
                    f"same points | r={style_radius} px",
                )

            core_pcv = None
            if extent >= FIXED_CORE_METERS:
                core_local_all = np.searchsorted(indices, core_indices)
                core_pcv = pcv[core_local_all]
                core_display_local = np.searchsorted(
                    indices, core_display_indices
                )
                core_display_pcv = pcv[core_display_local]
                core_rgba = modulate_rgba_with_pcv(
                    _class_rgba(classes[core_display_indices]),
                    core_display_pcv,
                    normalize=False,
                )
                core_column = context_extents.index(extent)
                _, core_radius = _render(
                    core_axes[row_index, core_column],
                    xyz[core_display_indices],
                    core_rgba,
                )
                _annotate(
                    core_axes[row_index, core_column],
                    f"same {len(core_display_indices):,} core points | r={core_radius} px",
                )
                _, _, raw_radius = render_point_cloud_axis(
                    raw_axes[row_index, core_column],
                    xyz[core_display_indices, 0],
                    xyz[core_display_indices, 1],
                    xyz[core_display_indices, 2],
                    plt.get_cmap("gray")(core_display_pcv),
                    width=DISPLAY_WIDTH,
                    style="pcv",
                    point_scale=DISPLAY_POINT_SCALE,
                )
                _annotate(
                    raw_axes[row_index, core_column],
                    f"mean PCV={float(np.mean(core_pcv)):.3f} | r={raw_radius} px",
                )

            rows.append(
                {
                    "environment": environment,
                    "source": str(path),
                    "extent_m": extent,
                    "points": len(indices),
                    "pcv_resolution": resolution,
                    "pcv_mean": float(np.mean(pcv)),
                    "pcv_p10": float(np.percentile(pcv, 10)),
                    "pcv_median": float(np.median(pcv)),
                    "pcv_p90": float(np.percentile(pcv, 90)),
                    "core_pcv_mean": (
                        float(np.mean(core_pcv)) if core_pcv is not None else ""
                    ),
                }
            )
            del pcv, points

        for axes in (overview_axes, pure_axes, color_axes, edge_axes):
            axes[row_index, 0].set_ylabel(
                environment.replace("_", " ").title(),
                rotation=0, ha="right", va="center", labelpad=12,
            )
        core_axes[row_index, 0].set_ylabel(
            environment.replace("_", " ").title(),
            rotation=0, ha="right", va="center", labelpad=12,
        )
        raw_axes[row_index, 0].set_ylabel(
            environment.replace("_", " ").title(),
            rotation=0, ha="right", va="center", labelpad=12,
        )
        del xyz, classes

    for index, extent in enumerate(EXTENTS_METERS):
        for axes in (overview_axes, pure_axes, color_axes, edge_axes):
            axes[0, index].set_title(
                f"({chr(ord('a') + index)}) {extent} × {extent} m"
            )
    for index, extent in enumerate(context_extents):
        core_axes[0, index].set_title(
            f"({chr(ord('a') + index)}) PCV context {extent} × {extent} m"
        )
        raw_axes[0, index].set_title(
            f"({chr(ord('a') + index)}) PCV context {extent} × {extent} m"
        )

    legend = [
        Patch(color=CLASS_COLORS[class_id], label=label)
        for class_id, label in (
            (2, "Ground"), (3, "Vegetation"), (6, "Building"),
            (9, "Water"), (17, "Bridge"),
        )
    ]
    for figure in (overview_fig, color_fig, edge_fig):
        figure.legend(
            handles=legend, loc="lower center", ncol=5,
            bbox_to_anchor=(0.5, -0.025), frameon=False,
        )
    pure_fig.colorbar(
        plt.cm.ScalarMappable(
            norm=mcolors.Normalize(0.0, 1.0), cmap="gray"
        ),
        ax=pure_axes,
        fraction=0.013,
        pad=0.012,
        label="Portion of visible sky",
    )
    core_fig.legend(
        handles=legend, loc="lower center", ncol=5,
        bbox_to_anchor=(0.5, -0.025), frameon=False,
    )
    raw_fig.colorbar(
        plt.cm.ScalarMappable(
            norm=mcolors.Normalize(0.0, 1.0), cmap="gray"
        ),
        ax=raw_axes,
        fraction=0.013,
        pad=0.012,
        label="Portion of visible sky",
    )
    for name, figure in (
        ("figure_43_pcv_extent_overview", overview_fig),
        ("figure_43_pcv_extent_pure", pure_fig),
        ("figure_43_pcv_extent_color", color_fig),
        ("figure_43_pcv_extent_edge_edl", edge_fig),
        ("figure_43_pcv_extent_fixed_core", core_fig),
        ("figure_43_pcv_extent_fixed_core_raw", raw_fig),
    ):
        figure.savefig(OUTPUT_DIR / f"{name}.png")
        plt.close(figure)

    metrics_path = OUTPUT_DIR / "pcv_extent_comparison_metrics.csv"
    with metrics_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved comparison figures and {metrics_path}", flush=True)


if __name__ == "__main__":
    run()
