"""Create the two Section 4.3 main-text figures as PNG files.

The data, source tile choices, and colors are shared with the full Swiss-Audit
report; this script only changes the paper layout. Run directly from an IDE.
"""

from __future__ import annotations

import json

import laspy
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from safetensors import safe_open

from plot_cvrl_43_results import (
    AUDIT_ROOT,
    OUTPUT_DIR,
    PAPER_CLASS_NAMES,
    REGION_COLORS,
    _set_publication_style,
)
from plot_projection_conflict import CONFLICT_PATH, plot_projection_cases


Q_GEO_PATH = AUDIT_ROOT / "analysis" / "q_geo_summary.json"
FIG5_PATH = OUTPUT_DIR / "figure_5_visibility_bottleneck.png"
FIG6_PATH = OUTPUT_DIR / "figure_6_projection_conflict_mechanism.png"
REGIONS = ("P_inv", "P_amb", "P_vis")
REGION_LABELS = (r"$P_{inv}$", r"$P_{amb}$", r"$P_{vis}$")
FIG6_PANEL_SIZE_IN = 3.25
FIG6_PANEL_X_IN = (0.65, 4.80, 8.95, 13.10)
LIGHT_GRID = "#E2E7EB"


SCENE_ORDER = (
    "Tall-building", "Low-rise built", "Dense canopy",
    "Sparse vegetation", "Flat bare ground", "Mixed / other",
)


def _ground_relief(stem):
    path = AUDIT_ROOT / "pointcloud" / f"{stem}.las"
    if not path.exists():
        path = path.with_suffix(".laz")
    cloud = laspy.read(path)
    ground = np.asarray(cloud.z)[np.asarray(cloud.classification) == 2]
    if ground.size < 100:
        return float("inf")
    return float(np.quantile(ground, 0.98) - np.quantile(ground, 0.02))


def _scene_group(tile):
    """Mutually exclusive, point-composition-based audit strata."""
    total = tile["points"]
    counts = tile["class_stats"]
    fraction = lambda key: counts.get(str(key), {}).get("points", 0) / total
    vegetation, roof, ground = fraction(3), fraction(6), fraction(2)
    if roof >= 0.15:
        path = AUDIT_ROOT / "visibility" / f"{tile['stem']}.safetensors"
        with safe_open(str(path), framework="np") as handle:
            classes = handle.get_tensor("classification")
            height = handle.get_tensor("height_agl")
        roof_p90 = float(np.quantile(height[classes == 6], 0.9))
        return "Tall-building" if roof_p90 >= 15.0 else "Low-rise built"
    if vegetation >= 0.60:
        return "Dense canopy"
    if (ground >= 0.75 and vegetation < 0.05 and roof < 0.05
            and _ground_relief(tile["stem"]) <= 5.0):
        return "Flat bare ground"
    if 0.15 <= vegetation < 0.60 and ground < 0.75:
        return "Sparse vegetation"
    return "Mixed / other"


def _scene_stats(summary):
    grouped = {name: np.zeros(3, dtype=np.int64) for name in SCENE_ORDER}
    tile_counts = {name: 0 for name in SCENE_ORDER}
    for tile in summary["tile_stats"]:
        group = _scene_group(tile)
        grouped[group] += np.asarray(tile["region_count"], dtype=np.int64)
        tile_counts[group] += 1
    result = []
    for name in SCENE_ORDER:
        counts = grouped[name]
        if not counts.sum():
            continue
        result.append((name, {
            **{f"{region}_ratio": counts[index] / counts.sum()
               for index, region in enumerate(REGIONS)},
            "points": int(counts.sum()),
            "tiles": tile_counts[name],
        }))
    return result


def _stacked_axis(ax, items, *, rotation=48):
    x = np.arange(len(items))
    bottom = np.zeros(len(items), dtype=float)
    for region in REGIONS:
        values = np.asarray([entry[f"{region}_ratio"] for _, entry in items]) * 100
        ax.bar(
            x, values, bottom=bottom, width=0.72,
            color=REGION_COLORS[region], linewidth=0,
        )
        bottom += values
    ax.set_xticks(x, [label for label, _ in items], rotation=rotation, ha="right")
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 25))
    ax.set_ylabel("Point proportion (%)", labelpad=8)
    ax.grid(axis="y", color=LIGHT_GRID, linewidth=0.8, alpha=1.0)
    ax.tick_params(axis="x", length=0, labelsize=16)
    ax.tick_params(axis="y", labelsize=17)
    ax.yaxis.label.set_size(20)
    ax.spines["bottom"].set_visible(False)


def _panel_caption(fig, slot, label):
    ax = fig.add_subplot(slot)
    ax.axis("off")
    ax.text(
        0.5, 0.5, label, ha="center", va="center",
        fontsize=23, fontweight="semibold", transform=ax.transAxes,
    )


def plot_figure_5(summary):
    scenes = _scene_stats(summary)
    class_ids = (2, 3, 6, 9, 14, 15, 17, 26, 27)
    classes = [
        (PAPER_CLASS_NAMES[str(class_id)], summary["class_stats"][str(class_id)])
        for class_id in class_ids
    ]
    height_names = ("0-0.5", "0.5-2", "2-5", "5-15", "15-30", ">30")
    heights = [
        (f"{name} m", summary["height_stats"][str(index)])
        for index, name in enumerate(height_names)
    ]

    fig = plt.figure(figsize=(19.2, 8.0), layout="constrained")
    grid = fig.add_gridspec(
        2, 3, height_ratios=(1.0, 0.095),
        width_ratios=(1.20, 1.63, 1.0),
    )
    axes = [fig.add_subplot(grid[0, col]) for col in range(3)]
    _stacked_axis(axes[0], scenes, rotation=46)
    _stacked_axis(axes[1], classes, rotation=52)
    _stacked_axis(axes[2], heights, rotation=43)
    for col, label in enumerate((
        "(a) Scene composition", "(b) ALS class",
        "(c) Height above local ground",
    )):
        _panel_caption(fig, grid[1, col], label)
    fig.legend(
        handles=[
            Patch(facecolor=REGION_COLORS[region], label=label)
            for region, label in zip(REGIONS, REGION_LABELS)
        ],
        loc="outside upper center", ncol=3, fontsize=21,
        handlelength=1.9, columnspacing=2.4,
    )
    fig.savefig(FIG5_PATH, dpi=320)
    plt.close(fig)


def plot_figure_6(data):
    # Keep the four columns compact horizontally while giving the quantitative
    # panel and the two qualitative rows enough independent vertical space.
    width, height = 17.0, 12.1
    fig = plt.figure(figsize=(width, height))
    panel_size = FIG6_PANEL_SIZE_IN
    panel_x = FIG6_PANEL_X_IN
    panel_y = (5.35, 1.45)
    rate_x, rate_y, rate_width, rate_height = 1.10, 9.90, 15.25, 1.30
    rate_ax = fig.add_axes((
        rate_x / width, rate_y / height,
        rate_width / width, rate_height / height,
    ))
    case_axes = np.asarray([
        [fig.add_axes((x / width, y / height, panel_size / width, panel_size / height))
         for x in panel_x]
        for y in panel_y
    ])

    order = ("P_vis", "P_amb", "P_inv")
    summary = data["summary"]["all"]
    y = np.arange(3)
    value = np.asarray([summary[key]["conflict_rate"] for key in order]) * 100
    lower = np.asarray([summary[key]["ci95"][0] for key in order]) * 100
    upper = np.asarray([summary[key]["ci95"][1] for key in order]) * 100
    rate_ax.barh(
        y, value, height=0.58,
        color=[REGION_COLORS[key] for key in order],
    )
    rate_ax.errorbar(
        value, y, xerr=np.vstack((value - lower, upper - value)),
        fmt="none", ecolor="#243344", elinewidth=1.6, capsize=6,
    )
    rate_ax.set_yticks(y, [r"$P_{vis}$", r"$P_{amb}$", r"$P_{inv}$"])
    rate_ax.invert_yaxis()
    rate_ax.set_xlim(0, 53)
    rate_ax.set_xlabel("Patch-label conflict (%)", fontsize=22, labelpad=8)
    rate_ax.tick_params(axis="both", labelsize=20)
    rate_ax.grid(axis="x", color=LIGHT_GRID, linewidth=0.8, alpha=1.0)
    for row, number in enumerate(value):
        rate_ax.text(46.4, row, f"{number:.2f}%", va="center", fontsize=20)

    plot_projection_cases(
        data, fig=fig, axes=case_axes,
        row_labels=("", ""),
        column_titles=(
            "Orthophoto", "ALS top view",
            "Elevation profile: class", "Elevation profile: visibility",
        ),
        show_case_count=False,
        font_scale=1.75,
        legend_anchor=(0.5, 0.015),
    )
    # Match the profile grids to the paper-wide grid treatment used by Fig. 7.
    for profile_ax in case_axes[:, 2:].flat:
        profile_ax.grid(color=LIGHT_GRID, linewidth=0.8, alpha=1.0)
    for row, label in enumerate((
        "(a) Projection conflict", "(b) Under-canopy ground",
        "(c) Building facade",
    )):
        center_y = (rate_y + rate_height / 2) if row == 0 else panel_y[row - 1] + panel_size / 2
        fig.text(
            0.018, center_y / height, label,
            rotation=90, ha="center", va="center",
            fontsize=20, fontweight="semibold",
        )
    fig.savefig(FIG6_PATH, dpi=320)
    plt.close(fig)


def main():
    _set_publication_style()
    plt.rcParams.update({
        "font.size": 21,
        "axes.labelsize": 21,
        "xtick.labelsize": 19,
        "ytick.labelsize": 19,
    })
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = json.loads(Q_GEO_PATH.read_text(encoding="utf-8"))
    conflict = json.loads(CONFLICT_PATH.read_text(encoding="utf-8"))
    plot_figure_5(summary)
    plot_figure_6(conflict)
    print(FIG5_PATH)
    print(FIG6_PATH)


if __name__ == "__main__":
    main()
