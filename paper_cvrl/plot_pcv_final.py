"""Final 100 m PCV + edge-only EDL figure at point_scale=0.86."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from plot_pcv_extent_comparison import (
    CLASS_COLORS,
    DISPLAY_WIDTH,
    EDL_STRENGTH,
    OUTPUT_DIR,
    PCV_DIRECTIONS,
    PCV_PIXELS_PER_METER,
    PCV_POINT_RADIUS,
    RANDOM_SEED,
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
POINT_SCALE = 0.86
REPORT_FIGURE_DIR = OUTPUT_DIR / "report_figures"


def main():
    _set_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    scenes = _select_scenes()
    fig, axes = plt.subplots(
        1, len(scenes), figsize=(14.4, 4.8), constrained_layout=True
    )
    for index, (environment, source, center_xy) in enumerate(scenes):
        xyz, classes = _read_largest_crop(source, center_xy, EXTENT_METERS)
        pcv = compute_pcv(
            xyz,
            number_of_directions=PCV_DIRECTIONS,
            resolution=512,
            point_radius=PCV_POINT_RADIUS,
            projection_center=np.asarray(
                [0.0, 0.0, float(np.median(xyz[:, 2]))],
                dtype=np.float32,
            ),
            pixel_size=1.0 / PCV_PIXELS_PER_METER,
            strict_bounds=True,
        )
        chosen = _sample_indices(
            np.arange(len(xyz)),
            np.random.default_rng(RANDOM_SEED + index),
        )
        rgba = modulate_rgba_with_pcv(
            _class_rgba(classes[chosen]), pcv[chosen], normalize=False
        )
        _, _, radius = render_point_cloud_axis(
            axes[index],
            xyz[chosen, 0], xyz[chosen, 1], xyz[chosen, 2],
            rgba,
            width=DISPLAY_WIDTH,
            strength=EDL_STRENGTH,
            style="pcv_edl",
            point_scale=POINT_SCALE,
            fractional_splat=True,
            max_point_radius=7,
        )
        axes[index].set_title(
            f"({chr(ord('a') + index)}) "
            f"{environment.replace('_', ' ').title()}"
        )
        print(
            f"{environment}: {len(xyz):,} source points, "
            f"{len(chosen):,} displayed, radius={radius:.2f} px",
            flush=True,
        )
    fig.legend(
        handles=[
            Patch(color=CLASS_COLORS[class_id], label=label)
            for class_id, label in (
                (2, "Ground"), (3, "Vegetation"), (6, "Building"),
                (9, "Water"), (17, "Bridge"),
            )
        ],
        loc="outside lower center", ncol=5, frameon=False,
    )
    fig.savefig(REPORT_FIGURE_DIR / "figure_43_pcv_final.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
