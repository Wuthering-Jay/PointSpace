"""Render the quantitative figures used by the CVRL Section 4.3 report."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import laspy
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from utils.pointcloud_render import (
    compute_pcv,
    modulate_rgba_with_pcv,
    render_point_cloud_axis,
    to_image_pixels,
)
from paper_cvrl.section43_metrics import metric_miou


AUDIT_ROOT = Path(r"G:\data\swiss\joint_tiles_100\Swiss-Audit")
OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "section_43"
REPORT_FIGURE_DIR = OUTPUT_DIR / "report_figures"
CLASS_NAMES = {
    "1": "Unclassified",
    "2": "Ground",
    "3": "Vegetation",
    "6": "Building roof",
    "9": "Water",
    "14": "Wire / transport line",
    "15": "Transmission tower",
    "17": "Bridge deck",
    "26": "Building facade",
    "27": "Bridge pier & stay cable",
}

# Compact paper labels are shared by Fig. 5 bars and Fig. 6 semantic legend.
# Keep the full LAS names above for analysis tables and archival outputs.
PAPER_CLASS_NAMES = {
    "1": "Unclassified",
    "2": "Ground",
    "3": "Vegetation",
    "6": "Building roof",
    "9": "Water",
    "14": "Wire / line",
    "15": "Tower",
    "17": "Bridge deck",
    "26": "Building facade",
    "27": "Bridge pier / cable",
}

# Publication-oriented, color-blind-safe palette.  Times New Roman is used for
# the paper figures and falls back to a metrically similar serif font when the
# script is run on a machine without the Microsoft font installed.
PALETTE = {
    "blue": "#3B6FB6",
    "vermillion": "#C44E52",
    "gold": "#D6A84B",
    "teal": "#2A9D8F",
    "purple": "#7768AE",
    "cyan": "#4C9FAD",
    "gray": "#8A8F98",
}
REGION_COLORS = {
    "P_inv": PALETTE["vermillion"],
    "P_amb": PALETTE["gold"],
    "P_vis": PALETTE["blue"],
}


def _set_publication_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.titleweight": "semibold",
            "legend.fontsize": 8.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "axes.axisbelow": True,
            "grid.color": "#D9DDE3",
            "grid.linewidth": 0.55,
            "grid.alpha": 0.75,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 320,
            "savefig.bbox": "tight",
            "ps.fonttype": 42,
        }
    )


def _clean_axis(ax, grid=True):
    if grid:
        ax.grid(axis="y")
    ax.tick_params(direction="out", length=3, width=0.7)


def _render_edl(ax, x, y, z, rgba, camera=None, width=920, strength=5.5):
    return render_point_cloud_axis(
        ax,
        x,
        y,
        z,
        rgba,
        camera=camera,
        width=width,
        strength=strength,
        style="classic_edl",
    )


def _render_lit_edl(ax, x, y, z, rgba, camera=None, width=920, strength=5.5):
    return render_point_cloud_axis(
        ax,
        x,
        y,
        z,
        rgba,
        camera=camera,
        width=width,
        strength=strength,
        style="lit_edl",
    )


def _stacked(ax, labels, stats, title, horizontal=False):
    x = np.arange(len(labels))
    bottom = np.zeros(len(labels))
    for region in ("P_inv", "P_amb", "P_vis"):
        values = np.asarray([stats[label][f"{region}_ratio"] for label in labels]) * 100
        display_region = rf"$P_{{{region[2:]}}}$"
        if horizontal:
            ax.barh(x, values, left=bottom, label=display_region, color=REGION_COLORS[region])
        else:
            ax.bar(x, values, bottom=bottom, label=display_region, color=REGION_COLORS[region])
        bottom += values
    if horizontal:
        ax.set_yticks(x, labels)
        ax.set_xlabel("Point proportion (%)")
        ax.set_xlim(0, 100)
        ax.invert_yaxis()
        ax.grid(axis="x")
    else:
        ax.set_xticks(x, labels)
        ax.set_ylabel("Point proportion (%)")
        ax.set_ylim(0, 100)
        ax.grid(axis="y")
    ax.set_title(title)


def plot_visibility(q):
    fig, axes = plt.subplots(1, 3, figsize=(18.2, 6.0), constrained_layout=True)
    # The audit selection also has infrastructure-enriched and tiny residual
    # strata; those are sampling priorities, not peer scene types. Keep this
    # panel at one compositional level and disclose omitted strata in caption.
    scene_names = {
        "dense_vegetation": "Vegetation-dominant",
        "suburban": "Low-building mosaic",
        "urban": "Building-dominant",
        "open_ground": "Ground-dominant",
    }
    environments = sorted(
        scene_names,
        key=lambda key: q["environment_stats"][key]["P_inv_ratio"],
        reverse=True,
    )
    environment_labels = [
        f"{scene_names[key]} (n={q['environment_stats'][key]['tiles']})"
        for key in environments
    ]
    environment_stats = {
        display: q["environment_stats"][source]
        for display, source in zip(environment_labels, environments)
    }
    _stacked(
        axes[0], environment_labels, environment_stats,
        "(a) Primary scene composition", horizontal=True,
    )
    classes = sorted(q["class_stats"], key=int)
    class_labels = [CLASS_NAMES.get(value, value) for value in classes]
    class_stats = {CLASS_NAMES.get(key, key): value for key, value in q["class_stats"].items()}
    _stacked(
        axes[1], class_labels, class_stats,
        "(b) LAS class", horizontal=True,
    )
    bins = ["0-0.5 m", "0.5-2 m", "2-5 m", "5-15 m", "15-30 m", ">30 m"]
    height_stats = {
        label: q["height_stats"][str(index)]
        for index, label in enumerate(bins)
    }
    _stacked(
        axes[2], bins, height_stats,
        "(c) Height above local ground", horizontal=True,
    )
    for ax in axes:
        ax.tick_params(axis="both", labelsize=10)
        ax.set_xlabel("Point proportion (%)", fontsize=11)
        ax.title.set_fontsize(12)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="outside upper center", ncol=3,
        fontsize=12, handlelength=1.9, columnspacing=2.5,
    )
    fig.savefig(REPORT_FIGURE_DIR / "figure_43_visibility_bottleneck.png")
    plt.close(fig)


def plot_internalization(report):
    order = [
        "dino_native1024", "dino_pca128", "vpsd_native1024",
        "vpsd_pca128", "cvrl_visual_native1024", "cvrl_visual_pca128",
    ]
    labels = ["DINO\n1024", "DINO\n128", "VPSD\n1024", "VPSD\n128", "CVRL-V\n1024", "CVRL-V\n128"]
    linear = [metric_miou(report["internalization"][key]["linear"]["all"]) for key in order]
    knn = [metric_miou(report["internalization"][key]["knn"]["all"]) for key in order]
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.6), constrained_layout=True)
    x = np.arange(len(order)); width = 0.37
    axes[0].bar(x - width / 2, linear, width, label="Linear probe", color=PALETTE["blue"])
    axes[0].bar(x + width / 2, knn, width, label="Cosine kNN", color=PALETTE["teal"])
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("mIoU")
    axes[0].set_ylim(0, 0.62)
    axes[0].set_title("(a) Semantic internalization")
    axes[0].legend()
    _clean_axis(axes[0])
    cosine = report["paired_cosine"]
    cosine_order = list(cosine)
    cosine_labels = ["VPSD-1024", "CVRL-V-1024", "VPSD-128", "CVRL-V-128"]
    means = [cosine[key]["mean"] for key in cosine_order]
    p10 = [cosine[key]["p10"] for key in cosine_order]
    p90 = [cosine[key]["p90"] for key in cosine_order]
    axes[1].bar(np.arange(4), means, color=[PALETTE["purple"], PALETTE["gold"]] * 2)
    axes[1].errorbar(
        np.arange(4), means,
        yerr=[np.asarray(means) - np.asarray(p10), np.asarray(p90) - np.asarray(means)],
        fmt="none", color="black", capsize=4,
    )
    axes[1].set_xticks(np.arange(4), cosine_labels)
    axes[1].set_ylabel("Cosine to matched DINO teacher")
    axes[1].set_ylim(0, 0.9)
    axes[1].set_title("(b) Teacher alignment (P10–mean–P90)")
    _clean_axis(axes[1])
    fig.savefig(REPORT_FIGURE_DIR / "figure_43_semantic_internalization.png")
    plt.close(fig)


def plot_regional(report):
    preferred = [
        ("vpsd_scratch_pca128", "Scratch-VPSD-128"),
        ("vpsd_pca128", "VPSD-128"),
        ("rrcl_independent72", "RRCL-72"),
        ("cvrl_visual_pca128", "CVRL-V-128"),
        ("cvrl_return72", "CVRL-R-72"),
        ("cvrl_full200", "CVRL-full-200"),
    ]
    pairs = [pair for pair in preferred if pair[0] in report["regional_compensation"]]
    sources = [pair[0] for pair in pairs]
    labels = [pair[1] for pair in pairs]
    regions = ["all", "P_inv", "P_amb", "P_vis"]
    colors = [PALETTE["gray"], PALETTE["vermillion"], PALETTE["gold"], PALETTE["blue"]]
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 9.0), constrained_layout=True)
    x = np.arange(len(sources)); width = 0.19
    for index, region in enumerate(regions):
        values = [metric_miou(report["regional_compensation"][source]["linear"][region]) for source in sources]
        display = "All" if region == "all" else rf"$P_{{{region[2:]}}}$"
        axes[0, 0].bar(x + (index - 1.5) * width, values, width, label=display, color=colors[index])
    axes[0, 0].set_xticks(x, labels)
    axes[0, 0].set_ylim(0.0, 0.62)
    axes[0, 0].set_ylabel("mIoU")
    axes[0, 0].set_title("(a) Frozen linear probe")
    axes[0, 0].legend(ncol=2)
    _clean_axis(axes[0, 0])
    for index, region in enumerate(regions):
        values = [metric_miou(report["regional_compensation"][source]["knn"][region]) for source in sources]
        axes[0, 1].bar(x + (index - 1.5) * width, values, width, label=region, color=colors[index])
    axes[0, 1].set_xticks(x, labels)
    axes[0, 1].set_ylim(0.0, 0.55)
    axes[0, 1].set_ylabel("mIoU")
    axes[0, 1].set_title("(b) Cosine kNN")
    _clean_axis(axes[0, 1])
    visual = report["regional_compensation"]["cvrl_visual_pca128"]["linear"]
    full = report["regional_compensation"]["cvrl_full200"]["linear"]
    vpsd = report["regional_compensation"]["vpsd_pca128"]["linear"]
    axes[1, 0].bar(
        np.arange(4) - 0.18,
        [(metric_miou(full[r]) - metric_miou(visual[r])) * 100 for r in regions],
        0.36, label="Full − CVRL visual", color=PALETTE["purple"],
    )
    axes[1, 0].bar(
        np.arange(4) + 0.18,
        [(metric_miou(full[r]) - metric_miou(vpsd[r])) * 100 for r in regions],
        0.36, label="Full − VPSD", color=PALETTE["cyan"],
    )
    axes[1, 0].axhline(0, color="black", linewidth=0.8)
    axes[1, 0].set_xticks(
        np.arange(4), ["All", r"$P_{inv}$", r"$P_{amb}$", r"$P_{vis}$"]
    )
    axes[1, 0].set_ylabel("mIoU gain (percentage points)")
    axes[1, 0].set_title("(c) Complementarity gain")
    axes[1, 0].legend()
    _clean_axis(axes[1, 0])
    classes = sorted(
        report["regional_compensation"]["cvrl_full200"]["linear"]["P_inv"]["per_class"],
        key=int,
    )
    class_x = np.arange(len(classes))
    for index, source in enumerate(sources):
        values = [
            report["regional_compensation"][source]["linear"]["P_inv"]["per_class"][class_id]["F1"]
            for class_id in classes
        ]
        axes[1, 1].plot(class_x, values, marker="o", label=labels[index])
    axes[1, 1].set_xticks(class_x, [CLASS_NAMES.get(value, value) for value in classes], rotation=35, ha="right")
    axes[1, 1].set_ylabel("P_inv class F1")
    axes[1, 1].set_title(r"(d) Class-wise $P_{inv}$ behavior")
    axes[1, 1].legend(fontsize=7.5, ncol=2)
    _clean_axis(axes[1, 1])
    fig.savefig(REPORT_FIGURE_DIR / "figure_43_regional_compensation.png")
    plt.close(fig)


def plot_pinv_class_complementarity(report):
    """Separate return's class-specific strengths from its overall mIoU."""
    classes = (2, 3, 6, 17, 26)
    methods = (
        ("vpsd_pca128", "VPSD-128", PALETTE["blue"]),
        ("rrcl_independent72", "RRCL-72", PALETTE["teal"]),
        ("cvrl_visual_pca128", "CVRL-V-128", PALETTE["gold"]),
        ("cvrl_full200", "CVRL-full-200", PALETTE["vermillion"]),
    )
    bootstrap = report["paired_block_bootstrap"]["class_comparisons"]
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.2), constrained_layout=True)
    y = np.arange(len(classes))
    for method_index, (key, label, color) in enumerate(methods):
        values = [
            report["regional_compensation"][key]["linear"]["P_inv"]["per_class"][str(cls)]["F1"]
            for cls in classes
        ]
        axes[0].barh(
            y + (method_index - 1.5) * 0.18, values,
            0.17, label=label, color=color,
        )
    axes[0].set_yticks(y, [CLASS_NAMES[str(cls)] for cls in classes])
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 1)
    axes[0].set_xlabel(r"Frozen linear $P_{inv}$ F1")
    axes[0].set_title(r"(a) Class-wise $P_{inv}$ separability")
    axes[0].legend(ncol=2, loc="lower right", fontsize=9)
    axes[0].grid(axis="x", alpha=0.3)

    for offset, comparison, label, color in (
        (-0.10, "rrcl_independent_minus_vpsd", "RRCL - VPSD", PALETTE["teal"]),
        (0.10, "full_minus_vpsd", "CVRL-full - VPSD", PALETTE["vermillion"]),
    ):
        values = [bootstrap[comparison][str(cls)] for cls in classes]
        center = np.asarray([value["point_estimate"] for value in values]) * 100
        low = np.asarray([value["ci95_low"] for value in values]) * 100
        high = np.asarray([value["ci95_high"] for value in values]) * 100
        axes[1].errorbar(
            center, y + offset,
            xerr=np.vstack((center - low, high - center)),
            fmt="o", markersize=6, capsize=3,
            label=label, color=color,
        )
    axes[1].axvline(0, color="#444444", linewidth=0.8)
    axes[1].set_yticks(y, [CLASS_NAMES[str(cls)] for cls in classes])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("F1 difference versus VPSD (percentage points)")
    axes[1].set_title("(b) Paired source-block bootstrap 95% CI")
    axes[1].legend(loc="upper left", fontsize=9)
    axes[1].grid(axis="x", alpha=0.3)
    fig.savefig(REPORT_FIGURE_DIR / "figure_43_pinv_class_complementarity.png")
    plt.close(fig)


def plot_q_geo_qualitative_examples(q, render_points=_render_edl, suffix=""):
    """Render near-nadir EDL examples backing the qualitative visual audit."""
    environments = [
        "open_ground",
        "urban",
        "suburban",
        "dense_vegetation",
        "rare_infrastructure",
    ]
    selected = []
    for environment in environments:
        candidates = [
            tile for tile in q["tile_stats"] if tile["environment"] == environment
        ]
        candidates.sort(key=lambda tile: tile["region_count"][0] / tile["points"])
        if candidates:
            selected.append(candidates[len(candidates) // 2])
    fig, axes = plt.subplots(
        len(selected), 2, figsize=(11.8, 2.75 * len(selected)), constrained_layout=True
    )
    continuous_cmap = plt.get_cmap("viridis")
    region_rgba = np.asarray(
        [mcolors.to_rgba(REGION_COLORS[key]) for key in ("P_inv", "P_amb", "P_vis")]
    )
    rng = np.random.default_rng(42)
    for row, tile in enumerate(selected):
        path = AUDIT_ROOT / "q_geo_pointcloud" / f"{tile['stem']}.laz"
        cloud = laspy.read(path)
        count = len(cloud.points)
        index = (
            np.arange(count)
            if count <= 50000
            else np.sort(rng.choice(count, size=50000, replace=False))
        )
        x = np.asarray(cloud.x)[index]
        y = np.asarray(cloud.y)[index]
        z = np.asarray(cloud.z)[index]
        q_geo = np.asarray(cloud.q_geo)[index]
        region = np.asarray(cloud.q_geo_region)[index]
        camera, _, point_size = render_points(
            axes[row, 0], x, y, z, continuous_cmap(np.clip(q_geo, 0, 1)),
            width=920, strength=5.5,
        )
        render_points(
            axes[row, 1], x, y, z, region_rgba[region],
            camera=camera, width=920, strength=5.5,
        )
        ratio = tile["region_count"][0] / tile["points"] * 100
        axes[row, 0].set_ylabel(
            f"{tile['environment'].replace('_', ' ').title()}\n"
            rf"$P_{{inv}}$={ratio:.1f}%",
            rotation=0, ha="right", va="center", labelpad=12,
        )
    axes[0, 0].set_title(r"(a) Continuous geometry-only $q_{geo}$")
    axes[0, 1].set_title(r"(b) Frozen $P_{inv}/P_{amb}/P_{vis}$ partition")
    scalar = plt.cm.ScalarMappable(norm=mcolors.Normalize(0, 1), cmap=continuous_cmap)
    fig.colorbar(
        scalar, ax=axes[:, 0], fraction=0.018, pad=0.015,
        label=r"Geometry observability $q_{geo}$",
    )
    axes[0, 1].legend(
        handles=[
            Patch(color=REGION_COLORS["P_inv"], label=r"$P_{inv}$"),
            Patch(color=REGION_COLORS["P_amb"], label=r"$P_{amb}$"),
            Patch(color=REGION_COLORS["P_vis"], label=r"$P_{vis}$"),
        ],
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        ncol=1,
    )
    fig.savefig(REPORT_FIGURE_DIR / f"figure_43_q_geo_qualitative_examples{suffix}.png")
    plt.close(fig)


def plot_lit_edl_comparison(q):
    """Compare classic EDL and the formal Lit-EDL renderer."""
    environments = ["open_ground", "urban", "dense_vegetation"]
    selected = []
    for environment in environments:
        candidates = [
            tile for tile in q["tile_stats"] if tile["environment"] == environment
        ]
        candidates.sort(key=lambda tile: tile["region_count"][0] / tile["points"])
        if candidates:
            selected.append(candidates[len(candidates) // 2])
    continuous_cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(
        len(selected), 3, figsize=(14.8, 3.45 * len(selected)), constrained_layout=True
    )
    rng = np.random.default_rng(42)
    for row, tile in enumerate(selected):
        cloud = laspy.read(
            AUDIT_ROOT / "q_geo_pointcloud" / f"{tile['stem']}.laz"
        )
        count = len(cloud.points)
        index = (
            np.arange(count)
            if count <= 50000
            else np.sort(rng.choice(count, size=50000, replace=False))
        )
        x, y, z = (
            np.asarray(cloud.x)[index],
            np.asarray(cloud.y)[index],
            np.asarray(cloud.z)[index],
        )
        q_geo = np.asarray(cloud.q_geo)[index]
        region = np.asarray(cloud.q_geo_region)[index]
        rgba = continuous_cmap(np.clip(q_geo, 0, 1))
        camera, _, _ = _render_edl(
            axes[row, 0], x, y, z, rgba, width=820, strength=5.5
        )
        _render_lit_edl(
            axes[row, 1], x, y, z, rgba,
            camera=camera, width=820, strength=5.5,
        )
        _render_lit_edl(
            axes[row, 2], x, y, z, region_rgba[region],
            camera=camera, width=820, strength=5.5,
        )
        ratio = tile["region_count"][0] / tile["points"] * 100
        axes[row, 0].set_ylabel(
            f"{tile['environment'].replace('_', ' ').title()}\n"
            rf"$P_{{inv}}$={ratio:.1f}%",
            rotation=0, ha="right", va="center", labelpad=12,
        )
    axes[0, 0].set_title("(a) Classic EDL")
    axes[0, 1].set_title("(b) Lit-EDL")
    axes[0, 2].set_title(r"(c) Lit-EDL partition")
    scalar = plt.cm.ScalarMappable(
        norm=mcolors.Normalize(0, 1), cmap=continuous_cmap
    )
    fig.colorbar(
        scalar, ax=axes[:, :2], fraction=0.014, pad=0.012,
        label=r"Geometry observability $q_{geo}$",
    )
    axes[0, 2].legend(
        handles=[
            Patch(color=REGION_COLORS["P_inv"], label=r"$P_{inv}$"),
            Patch(color=REGION_COLORS["P_amb"], label=r"$P_{amb}$"),
            Patch(color=REGION_COLORS["P_vis"], label=r"$P_{vis}$"),
        ],
        loc="upper left", bbox_to_anchor=(1.01, 1.0), ncol=1,
    )
    fig.savefig(OUTPUT_DIR / "figure_43_lit_edl_comparison.png")
    plt.close(fig)


def plot_pcv_comparison(q):
    """Compare Lit-EDL with view-independent hemispherical PCV shading."""
    environments = ["open_ground", "urban", "dense_vegetation"]
    selected = []
    for environment in environments:
        candidates = [
            tile for tile in q["tile_stats"] if tile["environment"] == environment
        ]
        candidates.sort(key=lambda tile: tile["region_count"][0] / tile["points"])
        if candidates:
            selected.append(candidates[len(candidates) // 2])

    continuous_cmap = plt.get_cmap("viridis")
    gray_cmap = plt.get_cmap("gray")
    region_rgba = np.asarray(
        [mcolors.to_rgba(REGION_COLORS[key]) for key in ("P_inv", "P_amb", "P_vis")]
    )
    fig, axes = plt.subplots(
        len(selected), 5, figsize=(21.8, 3.35 * len(selected)), constrained_layout=True
    )
    rng = np.random.default_rng(42)
    for row, tile in enumerate(selected):
        cloud = laspy.read(
            AUDIT_ROOT / "q_geo_pointcloud" / f"{tile['stem']}.laz"
        )
        all_x = np.asarray(cloud.x)
        all_y = np.asarray(cloud.y)
        all_z = np.asarray(cloud.z)
        all_points = np.column_stack((all_x, all_y, all_z))
        # PCV must see the complete local scene.  Computing it after the 50k
        # publication subsample would create projection holes and false sky.
        all_pcv = compute_pcv(
            all_points,
            number_of_directions=128,
            resolution=1024,
            point_radius=0,
        )

        count = len(cloud.points)
        index = (
            np.arange(count)
            if count <= 50000
            else np.sort(rng.choice(count, size=50000, replace=False))
        )
        x, y, z = all_x[index], all_y[index], all_z[index]
        q_geo = np.asarray(cloud.q_geo)[index]
        pcv = all_pcv[index]
        q_geo_rgba = continuous_cmap(np.clip(q_geo, 0, 1))
        pcv_q_geo_rgba = modulate_rgba_with_pcv(q_geo_rgba, pcv)

        camera, _, _ = _render_lit_edl(
            axes[row, 0], x, y, z, q_geo_rgba, width=760, strength=5.5
        )
        raw_pcv_rgba = gray_cmap(np.clip(pcv, 0.0, 1.0))
        render_point_cloud_axis(
            axes[row, 1], x, y, z, raw_pcv_rgba,
            camera=camera, width=760, style="pcv",
        )
        render_point_cloud_axis(
            axes[row, 2], x, y, z,
            pcv_q_geo_rgba,
            camera=camera, width=760, style="pcv",
        )
        render_point_cloud_axis(
            axes[row, 3], x, y, z,
            pcv_q_geo_rgba,
            camera=camera, width=760, strength=4.0, style="pcv_edl",
        )
        render_point_cloud_axis(
            axes[row, 4], x, y, z,
            pcv_q_geo_rgba,
            camera=camera, width=760, strength=3.0, style="pcv_lit_edl",
        )
        ratio = tile["region_count"][0] / tile["points"] * 100
        axes[row, 0].set_ylabel(
            f"{tile['environment'].replace('_', ' ').title()}\n"
            rf"$P_{{inv}}$={ratio:.1f}%",
            rotation=0, ha="right", va="center", labelpad=12,
        )

    axes[0, 0].set_title(r"(a) Lit-EDL $q_{geo}$")
    axes[0, 1].set_title("(b) Raw PCV")
    axes[0, 2].set_title(r"(c) PCV $\times$ $q_{geo}$")
    axes[0, 3].set_title(r"(d) PCV $+$ edge-only EDL")
    axes[0, 4].set_title(r"(e) PCV $+$ full Lit-EDL")
    pcv_scalar = plt.cm.ScalarMappable(
        norm=mcolors.Normalize(0, 1), cmap=gray_cmap
    )
    fig.colorbar(
        pcv_scalar,
        ax=axes[:, 1],
        fraction=0.014,
        pad=0.012,
        label="Portion of visible sky",
    )
    fig.savefig(OUTPUT_DIR / "figure_43_pcv_edl_comparison.png")
    plt.close(fig)


def plot_pcv_point_size_comparison(q):
    """Compare adaptive display splat levels under identical PCV-Lit-EDL."""
    environments = ["open_ground", "urban", "dense_vegetation"]
    selected = []
    for environment in environments:
        candidates = [
            tile for tile in q["tile_stats"] if tile["environment"] == environment
        ]
        candidates.sort(key=lambda tile: tile["region_count"][0] / tile["points"])
        if candidates:
            selected.append(candidates[len(candidates) // 2])

    levels = [
        ("Very small", 0.40),
        ("Small", 0.65),
        ("Balanced", 0.95),
        ("Large", 1.30),
        ("Very large", 1.70),
    ]
    fig, axes = plt.subplots(
        len(selected),
        len(levels),
        figsize=(20.2, 3.35 * len(selected)),
        constrained_layout=True,
    )
    continuous_cmap = plt.get_cmap("viridis")
    rng = np.random.default_rng(42)
    for row, tile in enumerate(selected):
        cloud = laspy.read(
            AUDIT_ROOT / "q_geo_pointcloud" / f"{tile['stem']}.laz"
        )
        all_x = np.asarray(cloud.x)
        all_y = np.asarray(cloud.y)
        all_z = np.asarray(cloud.z)
        all_pcv = compute_pcv(
            np.column_stack((all_x, all_y, all_z)),
            number_of_directions=128,
            resolution=1024,
            point_radius=0,
        )
        count = len(cloud.points)
        index = (
            np.arange(count)
            if count <= 50000
            else np.sort(rng.choice(count, size=50000, replace=False))
        )
        x, y, z = all_x[index], all_y[index], all_z[index]
        q_geo_rgba = continuous_cmap(
            np.clip(np.asarray(cloud.q_geo)[index], 0.0, 1.0)
        )
        shaded_rgba = modulate_rgba_with_pcv(q_geo_rgba, all_pcv[index])

        camera = None
        for column, (_, point_scale) in enumerate(levels):
            camera, _, radius = render_point_cloud_axis(
                axes[row, column],
                x,
                y,
                z,
                shaded_rgba,
                camera=camera,
                width=720,
                strength=3.0,
                style="pcv_lit_edl",
                point_scale=point_scale,
                min_point_radius=1,
                max_point_radius=7,
            )
            axes[row, column].text(
                0.025,
                0.025,
                f"radius={radius} px",
                transform=axes[row, column].transAxes,
                ha="left",
                va="bottom",
                fontsize=8,
                color="black",
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.78,
                },
            )
        ratio = tile["region_count"][0] / tile["points"] * 100
        axes[row, 0].set_ylabel(
            f"{tile['environment'].replace('_', ' ').title()}\n"
            rf"$P_{{inv}}$={ratio:.1f}%",
            rotation=0,
            ha="right",
            va="center",
            labelpad=12,
        )

    for column, (name, point_scale) in enumerate(levels):
        axes[0, column].set_title(
            f"({chr(ord('a') + column)}) {name}\nscale={point_scale:.2f}"
        )
    fig.savefig(OUTPUT_DIR / "figure_43_pcv_lit_edl_point_size_comparison.png")
    plt.close(fig)


def plot_mismatch_cases():
    """Show where CVRL-full repairs representative VPSD P_inv errors."""
    render_points = _render_edl
    prediction_dir = AUDIT_ROOT / "analysis" / "predictions"
    vpsd_path = prediction_dir / "regional_vpsd_pca128_linear.npz"
    full_path = prediction_dir / "regional_cvrl_full200_linear.npz"
    if not vpsd_path.is_file() or not full_path.is_file():
        return
    with np.load(vpsd_path) as source:
        vpsd = {key: source[key] for key in source.files}
    with np.load(full_path) as source:
        full = {key: source[key] for key in source.files}
    for key in ("stem", "point_index", "target", "region"):
        if not np.array_equal(vpsd[key], full[key]):
            raise ValueError(f"VPSD/CVRL prediction rows differ for {key}")

    cases = [(3, "Vegetation"), (6, "Building roof"), (17, "Bridge deck")]
    chosen = []
    for class_id, class_name in cases:
        recovered = (
            (vpsd["target"] == class_id)
            & (vpsd["region"] == 0)
            & (vpsd["prediction"] != class_id)
            & (full["prediction"] == class_id)
        )
        stems, counts = np.unique(vpsd["stem"][recovered], return_counts=True)
        if stems.size:
            chosen.append((class_id, class_name, stems[np.argmax(counts)]))
    if not chosen:
        return

    classes = sorted(CLASS_NAMES, key=int)
    class_to_color = {
        int(value): plt.get_cmap("tab10")(index % 10)
        for index, value in enumerate(classes)
    }
    region_colors = np.asarray(
        [mcolors.to_rgba(REGION_COLORS[key]) for key in ("P_inv", "P_amb", "P_vis")]
    )
    fig, axes = plt.subplots(
        len(chosen), 4, figsize=(14.5, 3.25 * len(chosen))
    )
    if len(chosen) == 1:
        axes = axes[None, :]
    rng = np.random.default_rng(43)
    used_class_ids = set()
    for row, (class_id, class_name, stem) in enumerate(chosen):
        cloud = laspy.read(AUDIT_ROOT / "q_geo_pointcloud" / f"{stem}.laz")
        x_all, y_all, z_all = np.asarray(cloud.x), np.asarray(cloud.y), np.asarray(cloud.z)
        background = (
            np.arange(len(cloud.points))
            if len(cloud.points) <= 40000
            else np.sort(rng.choice(len(cloud.points), 40000, replace=False))
        )
        tile_mask = vpsd["stem"] == stem
        indices = vpsd["point_index"][tile_mask].astype(np.int64)
        target = vpsd["target"][tile_mask]
        region = vpsd["region"][tile_mask]
        pred_v = vpsd["prediction"][tile_mask]
        pred_f = full["prediction"][tile_mask]
        used_class_ids.update(np.unique(np.r_[target, pred_v, pred_f]).astype(int).tolist())
        recovered = (
            (target == class_id)
            & (region == 0)
            & (pred_v != class_id)
            & (pred_f == class_id)
        )
        render_index = np.union1d(background, indices)
        test_position = np.searchsorted(render_index, indices)
        render_x, render_y, render_z = (
            x_all[render_index], y_all[render_index], z_all[render_index]
        )
        camera, _, _ = render_points(
            axes[row, 0], render_x, render_y, render_z,
            region_colors[np.asarray(cloud.q_geo_region)[render_index]],
            width=760,
        )
        for axis, values in zip(axes[row, 1:], (target, pred_v, pred_f)):
            rgba = np.tile(mcolors.to_rgba("#D7D9DD"), (render_index.size, 1))
            rgba[test_position] = np.asarray(
                [class_to_color.get(int(value), (0, 0, 0, 1)) for value in values]
            )
            _, projection, _ = render_points(
                axis, render_x, render_y, render_z, rgba,
                camera=camera, width=760,
            )
            recovered_x, recovered_y = to_image_pixels(
                x_all[indices[recovered]], y_all[indices[recovered]],
                z_all[indices[recovered]], camera, projection,
            )
            axis.scatter(
                recovered_x, recovered_y,
                facecolors="none", edgecolors="#111111", s=24, linewidths=0.65,
            )
        axes[row, 0].set_ylabel(
            f"{class_name}\nrecovered = {int(recovered.sum())}",
            rotation=0, ha="right", va="center", labelpad=10,
        )
    for axis, title in zip(
        axes[0],
        (
            r"(a) $q_{geo}$ region",
            "(b) Ground truth",
            "(c) VPSD prediction",
            "(d) CVRL-full prediction",
        ),
    ):
        axis.set_title(title)
    used_classes = sorted(used_class_ids)
    legend_handles = [
        Line2D(
            [0], [0], marker="o", linestyle="", markersize=6,
            color=class_to_color[int(class_id)],
            label=CLASS_NAMES.get(str(class_id), str(class_id)),
        )
        for class_id in used_classes
    ]
    legend_handles.append(
        Line2D(
            [0], [0], marker="o", linestyle="", markersize=8,
            markerfacecolor="none", markeredgecolor="black",
            label="VPSD error corrected by CVRL-full",
        )
    )
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=min(6, len(legend_handles)),
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.subplots_adjust(left=0.10, right=0.995, top=0.94, bottom=0.14, wspace=0.04, hspace=0.10)
    fig.savefig(REPORT_FIGURE_DIR / "figure_43_representative_error_recovery.png")
    plt.close(fig)


def write_key_csv(report):
    path = OUTPUT_DIR / "section_43_key_metrics.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["analysis", "source", "region", "metric", "value", "points"])
        for source, result in report["internalization"].items():
            for probe in ("linear", "knn"):
                for region, metrics in result[probe].items():
                    writer.writerow(["internalization", source, region, f"{probe}_mIoU", metric_miou(metrics), metrics["points"]])
        for source, result in report["regional_compensation"].items():
            for probe in ("linear", "knn"):
                for region, metrics in result[probe].items():
                    writer.writerow(["regional", source, region, f"{probe}_mIoU", metric_miou(metrics), metrics["points"]])


def main():
    _set_publication_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    report = json.loads((AUDIT_ROOT / "analysis" / "section_43_report.json").read_text(encoding="utf-8"))
    q_geo = json.loads((AUDIT_ROOT / "analysis" / "q_geo_summary.json").read_text(encoding="utf-8"))
    plot_visibility(q_geo)
    plot_q_geo_qualitative_examples(q_geo)
    plot_internalization(report)
    plot_regional(report)
    plot_pinv_class_complementarity(report)
    plot_mismatch_cases()
    write_key_csv(report)


if __name__ == "__main__":
    main()
