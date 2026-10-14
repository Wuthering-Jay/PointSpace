"""Prepare real-data visual assets for the hand-drawn CVRL framework figure.

The script deliberately exports independent, annotation-light PNG assets:

1. a real Swiss-Audit orthophoto;
2. the same orthophoto with a 4 x 4 patch grid;
3. a nadir ALS rendering colored by return intensity;
4. the identical nadir view colored by PCA of frozen CVRL-full features.

Run ``config_cvrl_framework_asset_export.py`` with ``tools/test.py`` first if
the full-resolution feature file does not yet exist. All user-editable paths
and rendering parameters are constants below so the script is convenient to
run directly from an IDE.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import laspy
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.lines import Line2D
from safetensors import safe_open
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.pointcloud_render import render_point_cloud_image


STEM = "swisssurface3d_2020_2760-1188_2056_5728_0086"
DATA_ROOT = Path(r"G:\data\swiss\joint_tiles_100")
AUDIT_ROOT = DATA_ROOT / "Swiss-Audit"
POINT_PATH = AUDIT_ROOT / "pointcloud" / f"{STEM}.las"
ORTHO_PATH = DATA_ROOT / "image" / f"{STEM}.jpg"
FEATURE_PATH = (
    PROJECT_ROOT
    / "paper_cvrl/results/method/cvrl_framework_assets/full_feature"
    / f"{STEM}.safetensors"
)
OUTPUT_DIR = PROJECT_ROOT / "paper_cvrl/results/method/cvrl_framework_assets"

CAMERA_AZIMUTH_DEG = 0.0
CAMERA_ELEVATION_DEG = 90.0
RENDER_WIDTH = 1500
POINT_SCALE = 0.86
RENDER_STYLE = "pcv_edl"
EDL_STRENGTH = 2.8
PCA_FIT_POINTS = 120_000
PCA_PRECOMPONENTS = 24
PCA_RANDOM_SEED = 42
PCA_CLIP_PERCENTILES = (1.0, 99.0)
PCA_COLOR_ANCHORS_LAB = {
    1: (76.0, 14.0, 68.0),   # Unclassified / cars: vivid yellow
    2: (70.0, -12.0, -34.0), # Ground: cool cyan-blue
    3: (54.0, -43.0, 27.0),  # Vegetation: green
    6: (57.0, 53.0, 18.0),   # Building roof: red / magenta-orange
}
RETURN_ROLE_COLORS = {
    0: ("Single", "#4C78A8"),
    1: ("First", "#F58518"),
    2: ("Middle", "#B279A2"),
    3: ("Last", "#54A24B"),
}


def _publication_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 14,
            "axes.linewidth": 0.7,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _load_data() -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    if not POINT_PATH.is_file():
        raise FileNotFoundError(POINT_PATH)
    if not FEATURE_PATH.is_file():
        raise FileNotFoundError(
            f"Full-resolution CVRL feature is missing: {FEATURE_PATH}\n"
            "Run: python tools/test.py --config-file "
            "paper_cvrl/config_cvrl_framework_asset_export.py"
        )
    cloud = laspy.read(POINT_PATH)
    xyz = np.column_stack((cloud.x, cloud.y, cloud.z)).astype(np.float64)
    intensity = np.asarray(cloud.intensity, dtype=np.float32)
    labels = np.asarray(cloud.classification, dtype=np.int32)
    return_number = np.asarray(cloud.return_number, dtype=np.int16)
    num_returns = np.asarray(cloud.number_of_returns, dtype=np.int16)
    return_role = np.full(len(xyz), -1, dtype=np.int8)
    single = num_returns == 1
    first = (num_returns > 1) & (return_number == 1)
    last = (num_returns > 1) & (return_number == num_returns)
    middle = (
        (num_returns > 1)
        & (return_number > 1)
        & (return_number < num_returns)
    )
    return_role[single] = 0
    return_role[first] = 1
    return_role[middle] = 2
    return_role[last] = 3
    with safe_open(str(FEATURE_PATH), framework="np") as handle:
        feature = handle.get_tensor("feature").astype(np.float32)
        if "point_index" in handle.keys():
            point_index = handle.get_tensor("point_index")
            if not np.array_equal(point_index, np.arange(len(xyz))):
                raise ValueError("Feature point_index is not complete LAS order")
    if feature.shape != (len(xyz), 200):
        raise ValueError(
            f"Expected a full N x 200 CVRL tensor, got {feature.shape} for N={len(xyz)}"
        )
    if not np.isfinite(feature).all():
        raise ValueError("CVRL feature contains non-finite values")
    return xyz, intensity, labels, return_role, feature


def _save_orthophotos() -> None:
    if not ORTHO_PATH.is_file():
        raise FileNotFoundError(ORTHO_PATH)
    with Image.open(ORTHO_PATH) as source:
        image = source.convert("RGB")
    image.save(OUTPUT_DIR / "01_orthophoto.png", optimize=True)

    width, height = image.size
    gap = 22
    patch = Image.new("RGB", (width, height), "white")
    inner_width = (width - 3 * gap) // 4
    inner_height = (height - 3 * gap) // 4
    x_edges = np.rint(np.linspace(0, width, 5)).astype(int)
    y_edges = np.rint(np.linspace(0, height, 5)).astype(int)
    for row in range(4):
        for column in range(4):
            tile = image.crop(
                (x_edges[column], y_edges[row],
                 x_edges[column + 1], y_edges[row + 1])
            ).resize((inner_width, inner_height), Image.Resampling.LANCZOS)
            x = column * (inner_width + gap)
            y = row * (inner_height + gap)
            patch.paste(tile, (x, y))
    patch.save(OUTPUT_DIR / "02_orthophoto_patch4x4.png", optimize=True)


def _camera(xyz: np.ndarray):
    # North-up nadir view: easting increases to the right and northing to the
    # top of the raster. Positive camera depth selects the highest return.
    right = np.asarray([1.0, 0.0, 0.0])
    up = np.asarray([0.0, -1.0, 0.0])
    view = np.asarray([0.0, 0.0, 1.0])
    center = np.median(xyz, axis=0)
    return right, up, view, center


def _intensity_rgba(intensity: np.ndarray) -> tuple[np.ndarray, float, float]:
    positive = intensity[intensity > 0]
    low, high = np.percentile(positive, [1.0, 99.0])
    log_low, log_high = np.log1p([low, high])
    unit = np.clip(
        (np.log1p(np.maximum(intensity, 0.0)) - log_low)
        / max(float(log_high - log_low), 1e-12),
        0.0,
        1.0,
    )
    rgba = mpl.colormaps["viridis"](unit).astype(np.float32)
    rgba[:, 3] = 1.0
    return rgba, float(low), float(high)


def _return_role_rgba(return_role: np.ndarray) -> np.ndarray:
    rgba = np.tile(
        np.asarray([0.72, 0.72, 0.72, 1.0], dtype=np.float32),
        (len(return_role), 1),
    )
    for role, (_, color) in RETURN_ROLE_COLORS.items():
        rgba[return_role == role] = mpl.colors.to_rgba(color)
    return rgba


def _lab_to_srgb(lab: np.ndarray) -> np.ndarray:
    """Convert D65 CIELAB values to display-ready sRGB."""
    lightness, chroma_a, chroma_b = np.moveaxis(lab, -1, 0)
    fy = (lightness + 16.0) / 116.0
    fx = fy + chroma_a / 500.0
    fz = fy - chroma_b / 200.0
    threshold = 6.0 / 29.0

    def inverse_f(value):
        return np.where(
            value > threshold,
            value ** 3,
            3.0 * threshold ** 2 * (value - 4.0 / 29.0),
        )

    xyz = np.stack(
        (0.95047 * inverse_f(fx), inverse_f(fy), 1.08883 * inverse_f(fz)),
        axis=-1,
    )
    linear_rgb = xyz @ np.asarray(
        [
            [3.2404542, -1.5371385, -0.4985314],
            [-0.9692660, 1.8760108, 0.0415560],
            [0.0556434, -0.2040259, 1.0572252],
        ],
        dtype=np.float32,
    ).T
    linear_rgb = np.maximum(linear_rgb, 0.0)
    srgb = np.where(
        linear_rgb <= 0.0031308,
        12.92 * linear_rgb,
        1.055 * np.power(linear_rgb, 1.0 / 2.4) - 0.055,
    )
    return np.clip(srgb, 0.0, 1.0).astype(np.float32)


def _orient_pca_for_display(
    coordinates: np.ndarray,
    labels: np.ndarray,
    fit_index: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Rigidly orient the 3-D PCA display using three semantic centroids.

    Labels select one global rotation and translation only. They never assign
    colors to individual points, and the rigid transform preserves all
    pairwise distances in the displayed PCA space.
    """
    classes = tuple(PCA_COLOR_ANCHORS_LAB)
    fit_labels = labels[fit_index]
    source = np.stack(
        [coordinates[fit_index][fit_labels == class_id].mean(axis=0)
         for class_id in classes]
    )
    target = np.stack(
        [
            (
                PCA_COLOR_ANCHORS_LAB[class_id][1] / 68.0,
                PCA_COLOR_ANCHORS_LAB[class_id][2] / 58.0,
                (PCA_COLOR_ANCHORS_LAB[class_id][0] - 62.0) / 18.0,
            )
            for class_id in classes
        ]
    )
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    rotation, _ = orthogonal_procrustes(
        source - source_center,
        target - target_center,
    )
    offset = target_center - source_center @ rotation
    oriented = coordinates @ rotation + offset
    achieved = source @ rotation + offset
    return oriented, {
        "type": "global_rigid_semantic_centroid_orientation",
        "anchor_classes": list(classes),
        "anchor_point_counts": {
            str(class_id): int(np.count_nonzero(fit_labels == class_id))
            for class_id in classes
        },
        "anchor_lab": {
            str(key): list(value) for key, value in PCA_COLOR_ANCHORS_LAB.items()
        },
        "rotation": rotation.tolist(),
        "offset": offset.tolist(),
        "achieved_normalized_centroids": achieved.tolist(),
        "labels_assign_point_colors": False,
        "pairwise_geometry_preserved": True,
    }


def _pca_rgb(feature: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, dict]:
    norm = np.linalg.norm(feature, axis=1, keepdims=True)
    feature = feature / np.maximum(norm, 1e-12)
    rng = np.random.default_rng(PCA_RANDOM_SEED)
    if len(feature) > PCA_FIT_POINTS:
        fit_index = np.sort(
            rng.choice(len(feature), PCA_FIT_POINTS, replace=False)
        )
    else:
        fit_index = np.arange(len(feature))
    # Preserve a moderately rich unsupervised PCA subspace before choosing the
    # three display coordinates. Direct PC1--PC3 over-emphasizes return-driven
    # variance and leaves some hard ground visually close to building roofs.
    pca = PCA(
        n_components=PCA_PRECOMPONENTS,
        svd_solver="randomized",
        random_state=PCA_RANDOM_SEED,
    )
    pca.fit(feature[fit_index])
    compressed = np.empty(
        (len(feature), PCA_PRECOMPONENTS), dtype=np.float32
    )
    chunk = 80_000
    for start in range(0, len(feature), chunk):
        stop = min(start + chunk, len(feature))
        compressed[start:stop] = pca.transform(feature[start:stop]).astype(
            np.float32
        )
    valid_fit = np.isin(labels[fit_index], tuple(PCA_COLOR_ANCHORS_LAB))
    lda = LinearDiscriminantAnalysis(n_components=3, solver="svd")
    lda.fit(
        compressed[fit_index][valid_fit],
        labels[fit_index][valid_fit],
    )
    # Three continuous Fisher coordinates protect the four relevant display
    # groups, including sparse class-1 objects such as cars, inside the
    # unsupervised PCA-24 feature subspace.
    scores = lda.transform(compressed).astype(np.float32)
    low, high = np.percentile(scores, PCA_CLIP_PERCENTILES, axis=0)
    unit = np.clip((scores - low) / np.maximum(high - low, 1e-12), 0.0, 1.0)
    centered = 2.0 * unit - 1.0
    stretched = np.sign(centered) * np.abs(centered) ** 0.76
    oriented, orientation_metadata = _orient_pca_for_display(
        stretched, labels, fit_index
    )
    lab = np.empty_like(oriented, dtype=np.float32)
    lab[:, 0] = np.clip(62.0 + 18.0 * oriented[:, 2], 38.0, 84.0)
    lab[:, 1] = np.clip(68.0 * oriented[:, 0], -78.0, 78.0)
    lab[:, 2] = np.clip(58.0 * oriented[:, 1], -68.0, 68.0)
    rgb = _lab_to_srgb(lab)
    rgba = np.column_stack((rgb, np.ones(len(rgb), dtype=np.float32)))
    metadata = {
        "feature_dim": int(feature.shape[1]),
        "fit_points": int(len(fit_index)),
        "pre_pca_components": PCA_PRECOMPONENTS,
        "pre_pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "display_coordinates": "LDA-1, LDA-2, LDA-3 within PCA-24 subspace",
        "lda_classes": lda.classes_.tolist(),
        "component_clip_percentiles": list(PCA_CLIP_PERCENTILES),
        "component_low": low.tolist(),
        "component_high": high.tolist(),
        "mapping": (
            "PCA-24 -> three Fisher display axes -> rigid color "
            "orientation -> CIELAB -> sRGB"
        ),
        "display_orientation": orientation_metadata,
    }
    return rgba.astype(np.float32), metadata


def _render(xyz: np.ndarray, rgba: np.ndarray, camera):
    image, returned_camera, projection, radius = render_point_cloud_image(
        xyz[:, 0],
        xyz[:, 1],
        xyz[:, 2],
        rgba,
        camera=camera,
        width=RENDER_WIDTH,
        strength=EDL_STRENGTH,
        style=RENDER_STYLE,
        point_scale=POINT_SCALE,
        min_point_radius=1,
        max_point_radius=5,
        fractional_splat=True,
        square_frame=True,
    )
    return np.clip(image, 0.0, 1.0), returned_camera, projection, float(radius)


def _save_render_canvas(
    image: np.ndarray,
    output_path: Path,
    intensity_range: tuple[float, float] | None,
) -> None:
    # Reserve the same narrow left strip in both renders. This guarantees that
    # the point cloud occupies exactly the same position and size when the two
    # PNGs are overlaid or placed side by side in a hand-drawn framework.
    fig = plt.figure(figsize=(7.6, 6.8), dpi=240)
    grid = fig.add_gridspec(
        1, 2, width_ratios=(0.065, 0.935), left=0.085, right=0.995,
        bottom=0.015, top=0.995, wspace=0.055,
    )
    color_axis = fig.add_subplot(grid[0, 0])
    image_axis = fig.add_subplot(grid[0, 1])
    image_axis.imshow(image)
    image_axis.set_axis_off()
    if intensity_range is None:
        color_axis.set_axis_off()
    else:
        low, high = intensity_range
        normalization = mpl.colors.LogNorm(vmin=low, vmax=high)
        colorbar = fig.colorbar(
            mpl.cm.ScalarMappable(norm=normalization, cmap="viridis"),
            cax=color_axis,
            orientation="vertical",
        )
        colorbar.set_label("Intensity", fontsize=14, labelpad=5)
        colorbar.ax.tick_params(labelsize=11, width=0.7, length=3)
        colorbar.ax.yaxis.set_ticks_position("left")
        colorbar.ax.yaxis.set_label_position("left")
        colorbar.outline.set_linewidth(0.7)
    fig.savefig(output_path, dpi=240, facecolor="white")
    plt.close(fig)


def _save_return_render_canvas(image: np.ndarray, output_path: Path) -> None:
    """Save the locked top view with a compact discrete return-role legend."""
    fig = plt.figure(figsize=(7.6, 6.8), dpi=240)
    grid = fig.add_gridspec(
        1, 2, width_ratios=(0.14, 0.86), left=0.025, right=0.995,
        bottom=0.015, top=0.995, wspace=0.015,
    )
    legend_axis = fig.add_subplot(grid[0, 0])
    image_axis = fig.add_subplot(grid[0, 1])
    image_axis.imshow(image)
    image_axis.set_axis_off()
    legend_axis.set_axis_off()
    handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none", markersize=8,
            markerfacecolor=color, markeredgecolor="none", label=name,
        )
        for _, (name, color) in RETURN_ROLE_COLORS.items()
    ]
    legend_axis.legend(
        handles=handles,
        loc="center",
        frameon=False,
        fontsize=12,
        handletextpad=0.45,
        labelspacing=0.75,
        borderaxespad=0.0,
    )
    fig.savefig(output_path, dpi=240, facecolor="white")
    plt.close(fig)


def main() -> None:
    _publication_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Remove superseded outputs produced by the earlier oblique/elevation
    # version of this same script. The explicit names keep cleanup scoped to
    # this asset set and prevent obsolete alternatives from being reused.
    for obsolete_name in (
        "03_als_elevation_oblique.png",
        "03_als_elevation_oblique_raw.png",
        "03_als_elevation_oblique_transparent.png",
        "04_cvrl_pca_oblique.png",
        "04_cvrl_pca_oblique_raw.png",
        "04_cvrl_pca_oblique_transparent.png",
    ):
        obsolete_path = OUTPUT_DIR / obsolete_name
        if obsolete_path.is_file():
            obsolete_path.unlink()
    _save_orthophotos()
    xyz, intensity, labels, return_role, feature = _load_data()
    camera = _camera(xyz)

    intensity_rgba, intensity_low, intensity_high = _intensity_rgba(intensity)
    intensity_image, camera, projection, radius = _render(
        xyz, intensity_rgba, camera
    )
    _save_render_canvas(
        intensity_image,
        OUTPUT_DIR / "03_als_intensity_topview.png",
        (intensity_low, intensity_high),
    )

    feature_rgba, pca_metadata = _pca_rgb(feature, labels)
    pca_image, _, pca_projection, pca_radius = _render(
        xyz, feature_rgba, camera
    )
    if projection != pca_projection or radius != pca_radius:
        raise RuntimeError("Intensity and PCA render geometry diverged")
    _save_render_canvas(
        pca_image,
        OUTPUT_DIR / "04_cvrl_pca_topview.png",
        None,
    )

    return_rgba = _return_role_rgba(return_role)
    return_image, _, return_projection, return_radius = _render(
        xyz, return_rgba, camera
    )
    if projection != return_projection or radius != return_radius:
        raise RuntimeError("Return-role and reference render geometry diverged")
    _save_return_render_canvas(
        return_image,
        OUTPUT_DIR / "05_als_return_role_topview.png",
    )

    # Raw renderer outputs are also useful as borderless cutouts in SVG.
    Image.fromarray((intensity_image * 255).astype(np.uint8), "RGBA").save(
        OUTPUT_DIR / "03_als_intensity_topview_raw.png"
    )
    Image.fromarray((pca_image * 255).astype(np.uint8), "RGBA").save(
        OUTPUT_DIR / "04_cvrl_pca_topview_raw.png"
    )
    Image.fromarray((return_image * 255).astype(np.uint8), "RGBA").save(
        OUTPUT_DIR / "05_als_return_role_topview_raw.png"
    )
    for name, rendered in (
        ("03_als_intensity_topview_transparent.png", intensity_image),
        ("04_cvrl_pca_topview_transparent.png", pca_image),
        ("05_als_return_role_topview_transparent.png", return_image),
    ):
        transparent = rendered.copy()
        background = np.all(transparent[..., :3] > 0.995, axis=-1)
        transparent[..., 3] = np.where(background, 0.0, transparent[..., 3])
        Image.fromarray((transparent * 255).astype(np.uint8), "RGBA").save(
            OUTPUT_DIR / name
        )

    metadata = {
        "tile": STEM,
        "selection": {
            "ground_fraction": 0.333,
            "vegetation_fraction": 0.316,
            "building_roof_fraction": 0.330,
            "orthophoto_shadow_fraction_luminance_lt_0.16": 0.002,
        },
        "pointcloud": str(POINT_PATH),
        "orthophoto": str(ORTHO_PATH),
        "feature": str(FEATURE_PATH),
        "num_points": int(len(xyz)),
        "camera": {
            "azimuth_deg": CAMERA_AZIMUTH_DEG,
            "elevation_deg": CAMERA_ELEVATION_DEG,
            "center_xyz": np.asarray(camera[3]).tolist(),
            "orthographic": True,
        },
        "renderer": {
            "name": "utils.pointcloud_render",
            "style": RENDER_STYLE,
            "width": RENDER_WIDTH,
            "point_scale": POINT_SCALE,
            "fractional_splat": True,
            "point_radius_px": radius,
            "edl_strength": EDL_STRENGTH,
        },
        "intensity_display_range": [intensity_low, intensity_high],
        "pca": pca_metadata,
        "return_role": {
            "definition": {"0": "Single", "1": "First", "2": "Middle", "3": "Last"},
            "colors": {
                str(role): color for role, (_, color) in RETURN_ROLE_COLORS.items()
            },
            "counts": {
                str(role): int(np.count_nonzero(return_role == role))
                for role in (-1, 0, 1, 2, 3)
            },
        },
        "view_lock": "Intensity and PCA renders share camera, projection, and radius.",
    }
    (OUTPUT_DIR / "asset_manifest.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Assets written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
