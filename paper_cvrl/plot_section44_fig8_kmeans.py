"""Render a KMeans counterpart to the frozen-feature PCA-color Fig. 8.

KMeans is fitted separately in each region and each original L2-normalized
feature space. Semantic labels are used only for the Reference row. Run from
the IDE after ``export_section44_fig8_features.py``.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans

from plot_cvrl_43_results import _set_publication_style
from plot_section44_fig8_pca_rgb import (
    METHOD_DIMS,
    OTHER,
    PCV_DIRECTIONS,
    PCV_RESOLUTION,
    PROFILE_HALF_WIDTH_M,
    RANDOM_SEED,
    RESULT_DIR,
    SCENES,
    SEMANTIC,
    TOP_POINT_SCALE,
    TOP_RENDER_WIDTH,
    MAX_TOP_POINTS,
    _top_view,
    _vertical_profile,
    extract_profile,
    load_feature,
    load_region,
    sample_ids,
    semantic_colors,
)
from utils.pointcloud_render import compute_pcv


PNG_PATH = RESULT_DIR / "figure_8_kmeans_frozen_representations.png"
META_PATH = RESULT_DIR / "figure_8_kmeans_frozen_representations.json"
N_CLUSTERS = 8
FIT_POINTS_PER_REGION = 55_000
CLUSTER_COLORS = np.asarray([
    "#4E79A7", "#E15759", "#59A14F", "#B07AA1",
    "#F28E2B", "#76B7B2", "#EDC948", "#9C755F",
])


def fit_region_kmeans(region: str, method: str, n_points: int):
    feature = load_feature(region, method, n_points)
    ids = sample_ids(n_points, FIT_POINTS_PER_REGION, RANDOM_SEED)
    model = KMeans(
        n_clusters=N_CLUSTERS, n_init=5, max_iter=200,
        random_state=RANDOM_SEED, algorithm="lloyd",
    )
    model.fit(feature[ids])
    original = model.predict(feature)
    counts = np.bincount(original, minlength=N_CLUSTERS)
    descending = np.argsort(-counts, kind="stable")
    old_to_rank = np.empty(N_CLUSTERS, dtype=np.int32)
    old_to_rank[descending] = np.arange(N_CLUSTERS)
    clusters = old_to_rank[original]
    metadata = {
        "method": method,
        "region": region,
        "input_dim": METHOD_DIMS[method],
        "fit_points": int(ids.size),
        "cluster_counts_descending": counts[descending].tolist(),
        "original_id_to_size_rank": old_to_rank.tolist(),
        "inertia_fit": float(model.inertia_),
    }
    return CLUSTER_COLORS[clusters], metadata


def plot_kmeans_figure() -> None:
    _set_publication_style()
    plt.rcParams.update({
        "font.size": 8.5, "axes.titlesize": 9,
        "axes.labelsize": 8.5, "savefig.dpi": 600,
    })
    regions = {name: load_region(name) for name in ("urban", "forest")}
    metadata = {
        "schema": "section44_fig8_kmeans_v1",
        "seed": RANDOM_SEED,
        "n_clusters": N_CLUSTERS,
        "fit_space": "original frozen feature, per-point L2 normalized",
        "fit_scope": "independent for each method and region",
        "cvrl_display_branch_scaling": False,
        "cluster_color_rule": "cluster IDs ranked by descending point count within each method and region",
        "cluster_colors": CLUSTER_COLORS.tolist(),
        "cluster_color_comparability": "IDs and colors do not denote the same structure across methods or regions",
        "regions": {},
        "methods": {},
    }
    for name, data in regions.items():
        xyz = data["xyz"]
        ids = sample_ids(data["n_points"], MAX_TOP_POINTS, RANDOM_SEED)
        data["top_ids"] = ids[np.argsort(xyz[ids, 2], kind="stable")]
        data["camera"] = (
            np.asarray([1.0, 0.0, 0.0]),
            np.asarray([0.0, 1.0, 0.0]),
            np.asarray([0.0, 0.0, 1.0]),
            np.median(xyz[data["top_ids"]], axis=0),
        )
        data["pcv"] = compute_pcv(
            xyz[data["top_ids"]].astype(np.float32, copy=False),
            number_of_directions=PCV_DIRECTIONS,
            resolution=PCV_RESOLUTION,
            point_radius=0,
        )
        profile = extract_profile(
            xyz, SCENES[name]["profile"], PROFILE_HALF_WIDTH_M[name],
        )
        data["profile"] = profile
        profile_z = xyz[profile["mask"], 2]
        z_low, z_high = np.quantile(profile_z, [0.002, 0.998])
        data["profile_zlim"] = (float(z_low - 0.8), float(z_high + 0.8))
        metadata["regions"][name] = {
            "stem": data["stem"],
            "points": data["n_points"],
            "top_view_display_points": int(ids.size),
            "profile_endpoints_xy": SCENES[name]["profile"],
            "profile_half_width_m": PROFILE_HALF_WIDTH_M[name],
            "profile_points": int(profile["mask"].sum()),
            "profile_zlim_m": data["profile_zlim"],
            "top_renderer": {
                "style": "pcv_edl",
                "point_scale": TOP_POINT_SCALE,
                "width_px": TOP_RENDER_WIDTH,
            },
        }

    colors = {}
    for method in METHOD_DIMS:
        colors[method] = {}
        metadata["methods"][method] = {}
        for region, data in regions.items():
            colors[method][region], detail = fit_region_kmeans(
                region, method, data["n_points"],
            )
            metadata["methods"][method][region] = detail

    reference_colors = {
        region: semantic_colors(data["label"])
        for region, data in regions.items()
    }
    fig = plt.figure(figsize=(7.70, 7.80), facecolor="white")
    grid = fig.add_gridspec(
        5, 4, left=0.075, right=0.99, top=0.915, bottom=0.055,
        width_ratios=(1.0, 1.0, 1.0, 1.0),
        height_ratios=(1.0, 0.10, 1.0, 1.0, 1.0),
        hspace=0.05, wspace=0.38,
    )
    titles = (
        "Urban region\nTop view", "Urban region\nProfile A–A′",
        "Forest region\nTop view", "Forest region\nProfile B–B′",
    )
    row_names = ("Reference", "VPSD", "RRCL", "CVRL")
    for row, row_name in enumerate(row_names):
        for col in range(4):
            region = "urban" if col < 2 else "forest"
            data = regions[region]
            point_colors = (reference_colors[region] if row == 0
                            else colors[row_name.lower()][region])
            ax = fig.add_subplot(grid[row + (row > 0), col])
            if row == 0:
                ax.set_title(titles[col], fontsize=8.6,
                             weight="semibold", pad=5)
            if col % 2 == 0:
                _top_view(ax, data, point_colors,
                          data["profile"], region, row)
            else:
                _vertical_profile(ax, data, point_colors,
                                  data["profile"], region, row)
            if col == 0:
                ax.text(-0.15, 0.5, row_name, transform=ax.transAxes,
                        ha="right", va="center", rotation=90,
                        fontsize=9.3, weight="bold")

    legend = [
        Line2D([0], [0], marker="o", ls="", markersize=5.7,
               markerfacecolor=color, markeredgecolor="none", label=label)
        for label, color in [*SEMANTIC.values(), OTHER]
    ]
    legend_ax = fig.add_subplot(grid[1, :])
    legend_ax.axis("off")
    legend_ax.legend(handles=legend, loc="center", ncol=5, frameon=False,
                     fontsize=7.7, columnspacing=1.1, handletextpad=0.35)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_PATH, dpi=600, facecolor="white")
    plt.close(fig)
    META_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {PNG_PATH}")


if __name__ == "__main__":
    plot_kmeans_figure()
