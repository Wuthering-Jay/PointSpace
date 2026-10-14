"""Render Section 4.4 Fig. 8 from frozen point features.

Run ``export_section44_fig8_features.py`` first. The same point IDs are used
throughout. Each method and region gets its own 3D PCA display transform.
Reference labels orient its color axes for display; they do not fit the PCA.
"""

from __future__ import annotations

import json

import laspy
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
from safetensors import safe_open
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

from export_section44_fig8_features import OUTPUT_ROOT, SCENE_STEM, FOREST_STEM
from plot_cvrl_43_results import _set_publication_style
from utils.pointcloud_render import (
    compute_pcv,
    modulate_rgba_with_pcv,
    render_point_cloud_image,
    to_image_pixels,
)


RESULT_DIR = OUTPUT_ROOT.parent
PNG_PATH = RESULT_DIR / "figure_8_pca_rgb_frozen_representations.png"
PDF_PATH = RESULT_DIR / "figure_8_pca_rgb_frozen_representations.pdf"
META_PATH = RESULT_DIR / "figure_8_pca_rgb_frozen_representations.json"

RANDOM_SEED = 42
PCA_FIT_PER_REGION = 55_000
PCA_TRANSFORM_CHUNK = 90_000
RGB_CLIP_PERCENTILES = (1.0, 99.0)
MAX_TOP_POINTS = 150_000
PROFILE_POINT_SIZE = 3.0
PROFILE_HALF_WIDTH_M = {"urban": 0.75, "forest": 0.75}
TOP_RENDER_WIDTH = 760
TOP_POINT_SCALE = 0.86
PCV_DIRECTIONS = 128
PCV_RESOLUTION = 512
EXPORT_PDF = True

SCENES = {
    "urban": {
        "stem": SCENE_STEM,
        "profile": ((2679300.0, 1248386.0), (2679392.0, 1248306.0)),
        "profile_name": "A–A′",
    },
    "forest": {
        "stem": FOREST_STEM,
        "profile": ((2616302.0, 1179630.0), (2616398.0, 1179655.0)),
        "profile_name": "B–B′",
    },
}

SEMANTIC = {
    2: ("Ground", "#C7A46A"),
    3: ("Vegetation", "#3D9C59"),
    6: ("Building roof", "#DC6248"),
    26: ("Building facade", "#8461A7"),
}
OTHER = ("Other", "#8A8F98")
METHOD_DIMS = {"vpsd": 128, "rrcl": 72, "cvrl": 200}
CVRL_BRANCH_COMPONENTS = {
    "urban": {"visual": 3, "return": 1},
    "forest": {"visual": 3, "return": 3},
}
CVRL_BRANCH_DISPLAY_GAIN = {
    "urban": {"visual": 1.0, "return": 1.0},
    "forest": {"visual": 1.0, "return": 1.35},
}
# CIELAB anchors fix the otherwise arbitrary orientation of each PCA display.
# Labels select one rigid orientation of the same 3D embedding; they do not
# color individual points or affect the frozen features/PCA fit.
COLOR_ANCHORS_LAB = {
    2: (67.0, 7.0, 32.0),       # Ground: tan
    3: (56.0, -35.0, 26.0),    # Vegetation: green
    6: (58.0, 44.0, 30.0),     # Roof: warm red
    26: (48.0, 28.0, -29.0),   # Facade: purple
}


def _lab_to_srgb(lab: np.ndarray) -> np.ndarray:
    """Vectorized D65 CIELAB -> sRGB; only a display transform."""
    lightness, chroma_a, chroma_b = np.moveaxis(lab, -1, 0)
    fy = (lightness + 16.0) / 116.0
    fx = fy + chroma_a / 500.0
    fz = fy - chroma_b / 200.0
    threshold = 6.0 / 29.0

    def inverse_f(value):
        return np.where(
            value > threshold, value ** 3,
            3.0 * threshold ** 2 * (value - 4.0 / 29.0),
        )

    xyz = np.stack((0.95047 * inverse_f(fx), inverse_f(fy),
                    1.08883 * inverse_f(fz)), axis=-1)
    linear_rgb = xyz @ np.asarray([
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252],
    ], dtype=np.float32).T
    linear_rgb = np.maximum(linear_rgb, 0.0)
    srgb = np.where(linear_rgb <= 0.0031308, 12.92 * linear_rgb,
                    1.055 * np.power(linear_rgb, 1.0 / 2.4) - 0.055)
    return np.clip(srgb, 0.0, 1.0).astype(np.float32)


def _stretch_pca(unit_pcs: np.ndarray) -> np.ndarray:
    centered = 2.0 * unit_pcs - 1.0
    return np.sign(centered) * np.abs(centered) ** 0.72


def _semantic_orientation(
    stretched_pcs: np.ndarray, labels: np.ndarray, fit_ids: np.ndarray,
    region: str,
) -> tuple[np.ndarray, dict]:
    """Choose a rigid color-space orientation from class centroids only."""
    classes = (2, 3, 6, 26) if region == "urban" else (2, 3)
    fit_labels = labels[fit_ids]
    source = np.stack([
        stretched_pcs[fit_ids][fit_labels == class_id].mean(axis=0)
        for class_id in classes
    ])
    # Target coordinates correspond to a*, b*, L* respectively.
    target = np.stack([
        (COLOR_ANCHORS_LAB[class_id][1] / 75.0,
         COLOR_ANCHORS_LAB[class_id][2] / 70.0,
         (COLOR_ANCHORS_LAB[class_id][0] - 61.0) / 19.0)
        for class_id in classes
    ])
    weights = np.asarray(
        [0.5 if class_id == 26 else 1.0 for class_id in classes]
    )
    source_center = np.average(source, axis=0, weights=weights)
    target_center = np.average(target, axis=0, weights=weights)
    weight_root = np.sqrt(weights[:, None])
    rotation, _ = orthogonal_procrustes(
        (source - source_center) * weight_root,
        (target - target_center) * weight_root,
    )
    offset = target_center - source_center @ rotation
    oriented = stretched_pcs @ rotation + offset
    achieved = source @ rotation + offset
    metadata = {
        "anchor_classes": list(classes),
        "anchor_point_counts": {
            str(class_id): int(np.count_nonzero(fit_labels == class_id))
            for class_id in classes
        },
        "rotation": rotation.tolist(),
        "offset": offset.tolist(),
        "centroid_target_lab": {
            str(class_id): COLOR_ANCHORS_LAB[class_id]
            for class_id in classes
        },
        "centroid_achieved_lab_before_gamut_clipping": {
            str(class_id): [
                float(61.0 + 19.0 * achieved[index, 2]),
                float(75.0 * achieved[index, 0]),
                float(70.0 * achieved[index, 1]),
            ]
            for index, class_id in enumerate(classes)
        },
        "orientation_uses_reference_labels": True,
        "point_colors_use_individual_labels": False,
    }
    return oriented, metadata


def _map_pca_to_color(oriented_pcs: np.ndarray) -> np.ndarray:
    """Map continuously oriented PCA scores to the CIELAB display gamut."""
    lab = np.empty_like(oriented_pcs, dtype=np.float32)
    lab[:, 0] = np.clip(61.0 + 19.0 * oriented_pcs[:, 2], 39.0, 83.0)
    lab[:, 1] = np.clip(75.0 * oriented_pcs[:, 0], -80.0, 80.0)
    lab[:, 2] = np.clip(70.0 * oriented_pcs[:, 1], -75.0, 75.0)
    return _lab_to_srgb(lab)


def load_region(region: str) -> dict:
    stem = SCENES[region]["stem"]
    cloud = laspy.read(OUTPUT_ROOT / "pointcloud" / f"{stem}.las")
    xyz = np.column_stack((cloud.x, cloud.y, cloud.z)).astype(np.float64)
    labels = np.asarray(cloud.classification).copy()
    return {"stem": stem, "xyz": xyz, "label": labels,
            "n_points": len(labels)}


def load_feature(region: str, method: str, n_points: int) -> np.ndarray:
    stem = SCENES[region]["stem"]
    path = OUTPUT_ROOT / "features" / method / f"{stem}.safetensors"
    with safe_open(str(path), framework="np") as handle:
        feature = handle.get_tensor("feature").astype(np.float32)
        if "point_index" in handle.keys():
            index = handle.get_tensor("point_index")
            if not np.array_equal(index, np.arange(n_points)):
                raise ValueError(f"{method}/{region}: point_index is not full LAS order")
    if feature.shape != (n_points, METHOD_DIMS[method]):
        raise ValueError(f"{method}/{region}: unexpected shape {feature.shape}")
    if not np.isfinite(feature).all():
        raise ValueError(f"{method}/{region}: non-finite feature")
    return normalize(feature, norm="l2", axis=1, copy=False)


def semantic_colors(labels: np.ndarray) -> np.ndarray:
    colors = np.full(labels.shape[0], OTHER[1], dtype=object)
    for class_id, (_, color) in SEMANTIC.items():
        colors[labels == class_id] = color
    return colors


def sample_ids(count: int, maximum: int, seed: int) -> np.ndarray:
    if count <= maximum:
        return np.arange(count)
    return np.sort(np.random.default_rng(seed).choice(count, maximum, replace=False))


def extract_profile(xyz: np.ndarray, endpoints: tuple, half_width: float) -> dict:
    start, end = np.asarray(endpoints, dtype=np.float64)
    displacement = end - start
    length = float(np.linalg.norm(displacement))
    unit = displacement / length
    relative = xyz[:, :2] - start
    distance = relative @ unit
    cross = relative[:, 0] * -unit[1] + relative[:, 1] * unit[0]
    keep = ((distance >= 0) & (distance <= length)
            & (np.abs(cross) <= half_width))
    if keep.sum() < 300:
        raise ValueError(f"Profile has only {keep.sum()} points")
    return {"distance": distance, "mask": keep, "length": length}


def fit_pca_rgb(regions: dict, method: str) -> tuple[dict, dict]:
    """Fit a display-only PCA for each region, without semantic labels."""
    feature = {
        region: load_feature(region, method, regions[region]["n_points"])
        for region in ("urban", "forest")
    }
    rgb = {}
    fit_metadata = {}
    for region in ("urban", "forest"):
        values = feature[region]
        fit_ids = sample_ids(len(values), PCA_FIT_PER_REGION, RANDOM_SEED)
        branch_metadata = None
        if method == "cvrl":
            # Protected multi-view PCA: preserve three dominant directions
            # from each source before learning the final three display axes.
            branch_scores = []
            branch_metadata = {}
            for branch_name, branch_slice in (
                ("visual", slice(None, 128)),
                ("return", slice(128, None)),
            ):
                branch_components = CVRL_BRANCH_COMPONENTS[region][branch_name]
                branch = normalize(
                    values[:, branch_slice].copy(), norm="l2", axis=1,
                    copy=False,
                )
                branch_pca = PCA(
                    n_components=branch_components, svd_solver="randomized",
                    random_state=RANDOM_SEED,
                ).fit(branch[fit_ids])
                score = np.concatenate([
                    branch_pca.transform(
                        branch[start:start + PCA_TRANSFORM_CHUNK]
                    )
                    for start in range(0, len(branch), PCA_TRANSFORM_CHUNK)
                ], axis=0)
                whitening_scale = np.sqrt(
                    np.maximum(branch_pca.explained_variance_, 1e-12)
                )
                display_gain = CVRL_BRANCH_DISPLAY_GAIN[region][branch_name]
                branch_scores.append((score / whitening_scale) * display_gain)
                branch_metadata[branch_name] = {
                    "retained_components": branch_components,
                    "explained_variance_ratio": (
                        branch_pca.explained_variance_ratio_.tolist()
                    ),
                    "whitening_scale": whitening_scale.tolist(),
                    "display_gain": display_gain,
                }
                del branch, score
            values_for_pca = np.concatenate(branch_scores, axis=1)
            fit_matrix = values_for_pca[fit_ids]
        else:
            values_for_pca = values
            fit_matrix = values[fit_ids]
        pca = PCA(n_components=3, svd_solver="randomized",
                  random_state=RANDOM_SEED)
        pca.fit(fit_matrix)
        projected = np.concatenate([
            pca.transform(values_for_pca[start:start + PCA_TRANSFORM_CHUNK])
            for start in range(0, len(values_for_pca), PCA_TRANSFORM_CHUNK)
        ], axis=0)
        low, high = np.percentile(projected, RGB_CLIP_PERCENTILES, axis=0)
        if np.any(high - low <= 1e-8):
            raise ValueError(f"{method}/{region}: degenerate PCA color range")
        unit_pcs = np.clip(
            (projected - low) / (high - low), 0, 1
        ).astype(np.float32)
        oriented, orientation_metadata = _semantic_orientation(
            _stretch_pca(unit_pcs), regions[region]["label"], fit_ids, region,
        )
        rgb[region] = _map_pca_to_color(oriented)
        detail = {
            "fit_regions": [region],
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "rgb_low": low.tolist(), "rgb_high": high.tolist(),
            "display_orientation": orientation_metadata,
        }
        if method == "cvrl":
            detail["branch_pca"] = branch_metadata
            visual_components = CVRL_BRANCH_COMPONENTS[region]["visual"]
            detail["pc_visual_loading_energy"] = [
                float(np.square(component[:visual_components]).sum())
                for component in pca.components_
            ]
        fit_metadata[region] = detail
        del fit_matrix, projected, values_for_pca
    del feature
    metadata = {
        "method": method, "input_dim": METHOD_DIMS[method],
        "fit_points_per_region": PCA_FIT_PER_REGION,
        "pca_fit_scope": "per-region",
        "rgb_clip_percentiles": list(RGB_CLIP_PERCENTILES),
        "color_mapping": "reference_oriented_pca_to_lab",
        "pca_strategy": (
            "protected region-specific two-stage branch PCA with whitening"
            if method == "cvrl" else "direct PCA"
        ),
        "cvrl_branch_components": (
            CVRL_BRANCH_COMPONENTS if method == "cvrl" else None
        ),
        "cvrl_branch_display_gain": (
            CVRL_BRANCH_DISPLAY_GAIN if method == "cvrl" else None
        ),
        "fits": fit_metadata,
    }
    return rgb, metadata


def cvrl_visual_energy_fraction(regions: dict) -> dict[str, float]:
    """Diagnostic only: quantify branch contribution to raw full-feature norm."""
    output = {}
    for region in ("urban", "forest"):
        stem = regions[region]["stem"]
        path = OUTPUT_ROOT / "features" / "cvrl" / f"{stem}.safetensors"
        with safe_open(str(path), framework="np") as handle:
            feature = handle.get_tensor("feature").astype(np.float32)
        visual_energy = np.square(feature[:, :128]).sum(axis=1)
        return_energy = np.square(feature[:, 128:]).sum(axis=1)
        output[region] = float(np.mean(
            visual_energy / np.maximum(visual_energy + return_energy, 1e-12)
        ))
    return output


def _top_view(ax, data: dict, colors: np.ndarray, profile: dict,
              region: str, row: int) -> None:
    xyz = data["xyz"]
    ids = data["top_ids"]
    point_rgb = np.asarray(mcolors.to_rgba_array(colors[ids]), dtype=np.float32)
    shaded = modulate_rgba_with_pcv(
        point_rgb, data["pcv"], normalize=False,
    )
    image, camera, projection, _radius = render_point_cloud_image(
        xyz[ids, 0], xyz[ids, 1], xyz[ids, 2], shaded,
        camera=data["camera"], width=TOP_RENDER_WIDTH, strength=4.0,
        style="pcv_edl", point_scale=TOP_POINT_SCALE,
        fractional_splat=True, max_point_radius=7, square_frame=True,
    )
    ax.imshow(image, origin="lower", interpolation="nearest")
    ax.set_xlim(-0.5, image.shape[1] - 0.5)
    ax.set_ylim(-0.5, image.shape[0] - 0.5)
    ax.set_box_aspect(1)
    ax.set_xticks([])
    ax.set_yticks([])
    start, end = np.asarray(SCENES[region]["profile"])
    line_x, line_y = to_image_pixels(
        np.asarray([start[0], end[0]]), np.asarray([start[1], end[1]]),
        np.full(2, np.median(xyz[ids, 2])), camera, projection,
    )
    ax.plot(line_x, line_y, "--", lw=1.2,
            color="white", dashes=(3.4, 2), zorder=5)
    ax.plot(line_x, line_y, "--", lw=0.42,
            color="#242A30", dashes=(3.4, 2), zorder=6)
    line_name = "A" if region == "urban" else "B"
    if row == 0:
        ax.text(line_x[0], line_y[0], line_name, ha="right", va="bottom",
                fontsize=7.8, weight="bold", color="#20252A",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.2))
        ax.text(line_x[1], line_y[1], line_name + "′", ha="left", va="top",
                fontsize=7.8, weight="bold", color="#20252A",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.2))
    for spine in ax.spines.values():
        spine.set_visible(False)


def _vertical_profile(ax, data: dict, colors: np.ndarray,
                      profile: dict, region: str, row: int) -> None:
    ids = np.flatnonzero(profile["mask"])
    xyz = data["xyz"]
    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    ax.grid(True, color="#D5D9DE", linewidth=0.48, alpha=0.85)
    ax.scatter(profile["distance"][ids], xyz[ids, 2], c=colors[ids],
               s=PROFILE_POINT_SIZE, lw=0, rasterized=True, zorder=2)
    ax.set_xlim(0, profile["length"])
    ax.set_ylim(*data["profile_zlim"])
    ax.set_box_aspect(1)
    ax.tick_params(labelsize=7.2, length=2.5)
    if row == 3:
        axis_name = "A–A′" if region == "urban" else "B–B′"
        ax.set_xlabel(f"Distance {axis_name} (m)", fontsize=8.5, labelpad=3)
    if region == "urban":
        ax.set_ylabel("Elevation (m)", fontsize=8.5, labelpad=3)


def plot_figure() -> None:
    _set_publication_style()
    plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9,
                         "axes.labelsize": 8.5, "savefig.dpi": 600})
    regions = {name: load_region(name) for name in ("urban", "forest")}
    metadata = {
        "schema": "section44_fig8_pca_color_v5", "random_seed": RANDOM_SEED,
        "pca_fit_scope": "urban and forest fitted independently, separately for each representation",
        "same_point_order": True, "regions": {}, "methods": {},
        "cvrl_pca_strategy": (
            "region-specific protected branch PCA: urban 3V+1R and forest "
            "3V+3R, with a forest-only 1.35x return display gain, "
            "followed by joint PCA to three display dimensions"
        ),
        "per_region_pca": True,
        "color_mapping": "reference_oriented_pca_to_lab",
        "display_orientation_uses_reference_labels": True,
        "display_orientation_type": "weighted rigid Procrustes on class centroids",
        "semantic_anchors_define_only_display_orientation": True,
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
            resolution=PCV_RESOLUTION, point_radius=0,
        )
        profile = extract_profile(xyz, SCENES[name]["profile"],
                                  PROFILE_HALF_WIDTH_M[name])
        data["profile"] = profile
        profile_z = xyz[profile["mask"], 2]
        z_low, z_high = np.quantile(profile_z, [0.002, 0.998])
        data["profile_zlim"] = (float(z_low - 0.8), float(z_high + 0.8))
        labels, counts = np.unique(data["label"][profile["mask"]], return_counts=True)
        metadata["regions"][name] = {
            "stem": data["stem"], "points": data["n_points"],
            "top_view_display_points": len(ids),
            "profile_endpoints_xy": SCENES[name]["profile"],
            "profile_half_width_m": PROFILE_HALF_WIDTH_M[name],
            "profile_points": int(profile["mask"].sum()),
            "profile_class_counts": {
                str(int(label)): int(count) for label, count in zip(labels, counts)
            },
            "profile_zlim_m": data["profile_zlim"],
            "top_renderer": {
                "style": "pcv_edl", "point_scale": TOP_POINT_SCALE,
                "width_px": TOP_RENDER_WIDTH,
                "pcv_directions": PCV_DIRECTIONS,
                "pcv_resolution": PCV_RESOLUTION,
                "camera": "orthographic nadir, +X right, +Y up",
            },
        }

    rgb = {}
    for method in METHOD_DIMS:
        rgb[method], metadata["methods"][method] = fit_pca_rgb(regions, method)
    metadata["cvrl_visual_branch_energy_fraction"] = cvrl_visual_energy_fraction(regions)
    reference_colors = {
        name: semantic_colors(data["label"]) for name, data in regions.items()
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
            colors = (reference_colors[region] if row == 0
                      else rgb[("cvrl" if row == 3 else row_name.lower())][region])
            ax = fig.add_subplot(grid[row + (row > 0), col])
            if row == 0:
                ax.set_title(titles[col], fontsize=8.6, weight="semibold", pad=5)
            if col % 2 == 0:
                _top_view(ax, data, colors, data["profile"], region, row)
            else:
                _vertical_profile(ax, data, colors, data["profile"], region, row)
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
    if EXPORT_PDF:
        fig.savefig(PDF_PATH, facecolor="white")
    plt.close(fig)
    META_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {PNG_PATH}" +
          (f" and {PDF_PATH}" if EXPORT_PDF else ""))


if __name__ == "__main__":
    plot_figure()
