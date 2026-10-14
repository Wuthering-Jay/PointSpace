"""Compare display splat sizes for 100 m PCV + edge-only EDL.

Run directly in an IDE.  PCV, sampled points, colors, and camera stay fixed
within each scene; only the density-scaled display point radius changes.
"""

from __future__ import annotations

import csv

import matplotlib.pyplot as plt
import numpy as np

from plot_pcv_extent_comparison import (
    DISPLAY_WIDTH,
    EDL_STRENGTH,
    OUTPUT_DIR,
    PCV_DIRECTIONS,
    PCV_PIXELS_PER_METER,
    PCV_POINT_RADIUS,
    RANDOM_SEED,
    _annotate,
    _class_rgba,
    _read_largest_crop,
    _sample_indices,
    _select_scenes,
    _set_style,
)
from utils.pointcloud_render import (
    compute_pcv,
    modulate_rgba_with_pcv,
    render_point_cloud_axis,
)


EXTENT_METERS = 100
POINT_SCALES = (0.40, 0.65, 0.95, 1.30, 1.70)


def _render_row(
    axes, xyz, rgba, scales, environment, *, fractional_splat=False
):
    camera = None
    radii = []
    for ax, scale in zip(axes, scales):
        camera, _, radius = render_point_cloud_axis(
            ax,
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
            rgba,
            camera=camera,
            width=DISPLAY_WIDTH,
            strength=EDL_STRENGTH,
            style="pcv_edl",
            point_scale=scale,
            min_point_radius=1,
            max_point_radius=7,
            fractional_splat=fractional_splat,
        )
        _annotate(ax, f"scale={scale:.2f} | radius={radius:.2f} px")
        radii.append(radius)
    axes[0].set_ylabel(
        environment.replace("_", " ").title(),
        rotation=0,
        ha="right",
        va="center",
        labelpad=12,
    )
    return radii


def _save(fig, stem):
    fig.savefig(OUTPUT_DIR / f"{stem}.png")
    plt.close(fig)


def run(
    point_scales=POINT_SCALES,
    *,
    fractional_splat=False,
    output_suffix="",
):
    _set_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scenes = _select_scenes()
    overview, overview_axes = plt.subplots(
        len(scenes),
        len(point_scales),
        figsize=(4.1 * len(point_scales), 3.7 * len(scenes)),
        constrained_layout=True,
    )
    rows = []

    for row_index, (environment, source, center_xy) in enumerate(scenes):
        print(f"[{environment}] Reading 100 m crop from {source.name}", flush=True)
        xyz, classes = _read_largest_crop(source, center_xy, EXTENT_METERS)
        resolution = 512
        pcv = compute_pcv(
            xyz,
            number_of_directions=PCV_DIRECTIONS,
            resolution=resolution,
            point_radius=PCV_POINT_RADIUS,
            projection_center=np.asarray(
                [0.0, 0.0, float(np.median(xyz[:, 2]))],
                dtype=np.float32,
            ),
            pixel_size=1.0 / PCV_PIXELS_PER_METER,
            strict_bounds=True,
        )
        sample = _sample_indices(
            np.arange(len(xyz)),
            np.random.default_rng(RANDOM_SEED + row_index),
        )
        display_xyz = xyz[sample]
        display_rgba = modulate_rgba_with_pcv(
            _class_rgba(classes[sample]),
            pcv[sample],
            normalize=False,
        )
        radii = _render_row(
            overview_axes[row_index],
            display_xyz,
            display_rgba,
            point_scales,
            environment,
            fractional_splat=fractional_splat,
        )

        detail, detail_axes = plt.subplots(
            1, len(point_scales),
            figsize=(4.1 * len(point_scales), 3.8),
            constrained_layout=True,
        )
        detail_radii = _render_row(
            detail_axes,
            display_xyz,
            display_rgba,
            point_scales,
            environment,
            fractional_splat=fractional_splat,
        )
        assert radii == detail_radii
        for col, scale in enumerate(point_scales):
            detail_axes[col].set_title(
                f"({chr(ord('a') + col)}) scale={scale:.2f}, "
                f"r={radii[col]:.2f} px"
            )
            rows.append(
                {
                    "environment": environment,
                    "extent_m": EXTENT_METERS,
                    "source_points": len(xyz),
                    "display_points": len(sample),
                    "point_scale": scale,
                    "display_radius_px": radii[col],
                    "pcv_mean": float(np.mean(pcv)),
                }
            )
        _save(
            detail,
            f"figure_43_pcv_edge_edl_point_size_{environment}{output_suffix}",
        )
        print(
            f"[{environment}] {len(xyz):,} source points, "
            f"{len(sample):,} displayed, radii={radii}",
            flush=True,
        )

    for col, scale in enumerate(point_scales):
        overview_axes[0, col].set_title(
            f"({chr(ord('a') + col)}) scale={scale:.2f}"
        )
    _save(
        overview,
        f"figure_43_pcv_edge_edl_point_size_comparison{output_suffix}",
    )

    with (OUTPUT_DIR / f"pcv_edge_edl_point_size_metrics{output_suffix}.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("Saved PCV + edge-only EDL point-size comparison", flush=True)


if __name__ == "__main__":
    run()
