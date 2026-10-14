"""Legacy real-data diagnostic for Fig. 3, not the final paper schematic.

Run from the project root in the pointspace environment::

    python paper_cvrl/plot_fig3_vpsd_alignment.py

The orthophoto, ALS positions, correspondence, and DINO patch vectors are
measured data. The 3-pixel image offset is an explanatory perturbation. The
Enc 2/3/4 glyphs are occupied-grid proxies at nominal LitePT scales, because
learned intermediate token coordinates are not stored with this audit tile.
"""

from __future__ import annotations

import json
from pathlib import Path

import laspy
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image
from safetensors import safe_open


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper_cvrl" / "results" / "method"
PNG = OUT / "figure_3_vpsd_real_data_diagnostic.png"
SVG = OUT / "figure_3_vpsd_real_data_diagnostic.svg"
JSON = OUT / "figure_3_vpsd_real_data_diagnostic.json"

STEM = "swisssurface3d_2020_2760-1188_2056_5728_0086"
DATA = Path(r"G:\data\swiss\joint_tiles_100")
POINT_PATH = DATA / "Swiss-Audit" / "pointcloud" / f"{STEM}.las"
IMAGE_PATH = DATA / "image" / f"{STEM}.jpg"
CORR_PATH = DATA / "Swiss-Audit" / "correspondence" / f"{STEM}.safetensors"
DINO_PATH = DATA / "Swiss-Audit" / "dino_feature" / f"{STEM}.safetensors"

PATCH = 16
GRID_WIDTH = 63
POINT_ID = 12571
OFFSET_XY = np.asarray([3, 0], dtype=np.int32)
CONTEXT = (80, 450, 840, 170)  # x, y, width, height in the DOM
ZOOM = (448, 496, 96, 96)

BLACK = "#171C22"
SHARED = "#275F92"
SWITCH = "#D67541"
REMOVED = "#81909A"
BACKGROUND = "#DDE4E8"
GRID = "#D8E1E7"


def _style() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 13,
        "text.color": BLACK,
        "axes.labelcolor": BLACK,
        "xtick.color": BLACK,
        "ytick.color": BLACK,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "svg.fonttype": "none",
    })


def _load():
    for path in (POINT_PATH, IMAGE_PATH, CORR_PATH, DINO_PATH):
        if not path.is_file():
            raise FileNotFoundError(path)
    image = np.asarray(Image.open(IMAGE_PATH).convert("RGB"))
    cloud = laspy.read(POINT_PATH)
    xyz = np.column_stack((cloud.x, cloud.y, cloud.z)).astype(np.float64)
    with safe_open(str(CORR_PATH), framework="np") as handle:
        row_col = handle.get_tensor("pixel_coord").astype(np.int32)
        valid = handle.get_tensor("valid").astype(bool)
        patch_index = handle.get_tensor("patch_index").astype(np.int32)
    with safe_open(str(DINO_PATH), framework="np") as handle:
        teacher = handle.get_tensor("feature").astype(np.float32)
    if len(xyz) != len(row_col):
        raise ValueError("LAS and correspondence point counts differ")
    expected = (row_col[:, 0] // PATCH) * GRID_WIDTH + row_col[:, 1] // PATCH
    if not np.array_equal(expected[valid], patch_index[valid]):
        raise ValueError("Stored row/column coordinates disagree with patch IDs")
    if teacher.shape != (GRID_WIDTH ** 2, 1024):
        raise ValueError(f"Unexpected DINO tensor shape: {teacher.shape}")
    return image, xyz, row_col[:, ::-1].copy(), valid & (patch_index >= 0), patch_index, teacher


def _ids_in_bounds(xy, bounds):
    x0, y0, width, height = bounds
    return ((xy[:, 0] >= x0) & (xy[:, 0] < x0 + width)
            & (xy[:, 1] >= y0) & (xy[:, 1] < y0 + height))


def _patch_id(xy):
    return (xy[:, 1] // PATCH) * GRID_WIDTH + xy[:, 0] // PATCH


def _choose_points(xy, valid, before, after, common):
    # Select a readable subset of measured points: four that leave patch j,
    # six that remain. The complete support set is displayed in panel (b).
    leaving = np.flatnonzero(before & ~after & _ids_in_bounds(xy, ZOOM))
    staying = np.flatnonzero(common & _ids_in_bounds(xy, ZOOM))
    reference = xy[POINT_ID]
    leaving = leaving[np.argsort(np.linalg.norm(xy[leaving] - reference, axis=1))[:4]]
    # Spread retained points across the patch instead of taking a dense knot.
    rng = np.random.default_rng(12)
    staying = rng.choice(staying, min(6, len(staying)), replace=False)
    if len(leaving) < 3 or len(staying) < 4:
        raise RuntimeError("Insufficient real switching/stable points in selected patch")
    return staying.astype(int), leaving.astype(int)


def _grid_support(xyz, ids, origin, scale):
    coordinates = np.floor((xyz[ids] - origin) / scale).astype(np.int64)
    return set(map(tuple, coordinates))


def _example(xyz, xy, valid, patch_index, teacher):
    if not valid[POINT_ID] or not _ids_in_bounds(xy[POINT_ID][None], ZOOM)[0]:
        raise ValueError("Selected example point is no longer valid")
    old_patch = int(patch_index[POINT_ID])
    new_patch = int(_patch_id((xy[POINT_ID] + OFFSET_XY)[None])[0])
    if new_patch != old_patch + 1:
        raise ValueError("Selected point does not cross the intended patch boundary")
    shifted_xy = xy + OFFSET_XY
    before = valid & (patch_index == old_patch)
    after = valid & (_patch_id(shifted_xy) == old_patch)
    common = before & after
    stable_ids, switch_ids = _choose_points(xy, valid, before, after, common)
    a, b = teacher[old_patch], teacher[new_patch]
    distance = 1.0 - float(np.dot(a, b) /
                           max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))
    origin = xyz.min(axis=0)
    hierarchy = {}
    for name, scale in (("Enc 2", 1.0), ("Enc 3", 2.0), ("Enc 4", 4.0)):
        old = _grid_support(xyz, np.flatnonzero(before), origin, scale)
        new = _grid_support(xyz, np.flatnonzero(after), origin, scale)
        hierarchy[name] = {
            "shared": len(old & new),
            "removed": len(old - new),
            "added": len(new - old),
            "old": len(old),
            "new": len(new),
            "nominal_grid_m": scale,
        }
    return {
        "point_xy": xy[POINT_ID],
        "patch_origin": (xy[POINT_ID] // PATCH) * PATCH,
        "old_patch": old_patch,
        "new_patch": new_patch,
        "before": before,
        "after": after,
        "common": common,
        "stable_sample_ids": stable_ids,
        "switch_sample_ids": switch_ids,
        "teacher_cosine_distance": distance,
        "hierarchy": hierarchy,
    }


def _photo(ax, image, bounds):
    x0, y0, width, height = bounds
    ax.imshow(image[y0:y0 + height, x0:x0 + width],
              extent=(x0, x0 + width, y0 + height, y0),
              interpolation="bilinear", zorder=0)
    ax.set_xlim(x0, x0 + width)
    ax.set_ylim(y0 + height, y0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#ABB5BC")
        spine.set_linewidth(0.75)


def _zoom(ax, image):
    _photo(ax, image, ZOOM)
    x0, y0, width, height = ZOOM
    for x in range(x0 + PATCH, x0 + width, PATCH):
        ax.plot([x, x], [y0, y0 + height], color=GRID,
                alpha=0.86, linewidth=0.63, zorder=2)
    for y in range(y0 + PATCH, y0 + height, PATCH):
        ax.plot([x0, x0 + width], [y, y], color=GRID,
                alpha=0.86, linewidth=0.63, zorder=2)


def _target_patches(ax, origin):
    ax.add_patch(Rectangle(origin, PATCH, PATCH,
                           facecolor=SHARED, alpha=0.13, edgecolor="none", zorder=3))
    ax.add_patch(Rectangle(origin + np.asarray([PATCH, 0]), PATCH, PATCH,
                           facecolor=SWITCH, alpha=0.12, edgecolor="none", zorder=3))
    for x in (origin[0], origin[0] + PATCH):
        ax.add_patch(Rectangle((x, origin[1]), PATCH, PATCH,
                               facecolor="none", edgecolor=BLACK,
                               linewidth=0.92, zorder=4))
    ax.text(origin[0] + PATCH / 2, origin[1] - 3.5, r"$t_j$",
            color=BLACK, fontsize=12, ha="center", va="bottom",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.76, pad=0.5),
            zorder=7)
    ax.text(origin[0] + 1.5 * PATCH, origin[1] - 3.5, r"$t_{j+1}$",
            color=BLACK, fontsize=12, ha="center", va="bottom",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.76, pad=0.5),
            zorder=7)


def _pointwise(ax, image, xy, example, shifted):
    _zoom(ax, image)
    _target_patches(ax, example["patch_origin"])
    moved = xy + (OFFSET_XY if shifted else 0)
    stable = example["stable_sample_ids"]
    switched = example["switch_sample_ids"]
    ax.scatter(moved[stable, 0], moved[stable, 1], s=72, c=SHARED,
               edgecolors="white", linewidths=0.9, zorder=6)
    ax.scatter(moved[switched, 0], moved[switched, 1], s=83, c=SWITCH,
               edgecolors="white", linewidths=0.9, zorder=7)


def _patchset(ax, image, xy, example, shifted):
    _zoom(ax, image)
    origin = example["patch_origin"]
    ax.add_patch(Rectangle(origin, PATCH, PATCH,
                           facecolor=SHARED, alpha=0.12, edgecolor="none", zorder=3))
    ax.add_patch(Rectangle(origin, PATCH, PATCH,
                           facecolor="none", edgecolor=BLACK,
                           linewidth=1.35, zorder=4))
    ax.text(origin[0] + PATCH / 2, origin[1] - 4.0, r"$A_j$",
            ha="center", va="bottom", fontsize=13.5, color=BLACK,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.76, pad=0.5),
            zorder=8)
    common = example["common"]
    member = example["after"] if shifted else example["before"]
    ids_shared = np.flatnonzero(common)
    ids_changed = np.flatnonzero(member & ~common)
    coordinates = xy + (OFFSET_XY if shifted else 0)
    ax.scatter(coordinates[ids_shared, 0], coordinates[ids_shared, 1],
               s=40, c=SHARED, edgecolors="white", linewidths=0.45, zorder=6)
    if shifted:
        ax.scatter(coordinates[ids_changed, 0], coordinates[ids_changed, 1],
                   s=43, c=SWITCH, edgecolors="white", linewidths=0.55, zorder=7)
    else:
        ax.scatter(coordinates[ids_changed, 0], coordinates[ids_changed, 1],
                   s=50, facecolors="none", edgecolors=REMOVED,
                   linewidths=1.1, zorder=7)


def _token_lanes(fig, hierarchy):
    ax = fig.add_axes((0.535, 0.170, 0.423, 0.137))
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.25, 2.65)
    ax.axis("off")
    ax.plot([0.52, 0.52], [-0.17, 2.5], color="#D5DCE0", linewidth=0.7)
    names = ("Enc 2", "Enc 3", "Enc 4")
    markers = ("o", "s", "D")
    for row, (name, marker) in enumerate(zip(names, markers)):
        y = 2 - row
        values = hierarchy[name]
        ax.text(0.015, y, name, ha="left", va="center", color=BLACK,
                fontsize=11.1, fontweight="bold")
        for col, old in ((0, True), (1, False)):
            begin = 0.16 + col * 0.52
            count = values["shared"]
            for k in range(count):
                ax.scatter(begin + 0.036 * k, y, s=40, marker=marker,
                           c=SHARED, edgecolors="white", linewidths=0.3, zorder=3)
            changed = values["removed"] if old else values["added"]
            for k in range(changed):
                x = begin + 0.036 * (count + k)
                if old:
                    ax.scatter(x, y, s=42, marker=marker, facecolors="none",
                               edgecolors=REMOVED, linewidths=1.0, zorder=3)
                else:
                    ax.scatter(x, y, s=42, marker=marker, c=SWITCH,
                               edgecolors="white", linewidths=0.3, zorder=3)
    return ax


def _set_key(fig, x0, y):
    fig.text(x0, y, "●", color=SHARED, fontsize=15, ha="right", va="center")
    fig.text(x0 + 0.004, y, "shared", color=BLACK, fontsize=11.2, va="center")
    fig.text(x0 + 0.090, y, "○", color=REMOVED, fontsize=15, ha="right", va="center")
    fig.text(x0 + 0.094, y, "removed", color=BLACK, fontsize=11.2, va="center")
    fig.text(x0 + 0.206, y, "●", color=SWITCH, fontsize=15, ha="right", va="center")
    fig.text(x0 + 0.210, y, "new / switched", color=BLACK,
             fontsize=11.2, va="center")


def _pointwise_key(fig, x0, y):
    fig.text(x0, y, "●", color=SHARED, fontsize=15, ha="right", va="center")
    fig.text(x0 + 0.004, y, "stable target", color=BLACK,
             fontsize=11.2, va="center")
    fig.text(x0 + 0.180, y, "●", color=SWITCH, fontsize=15,
             ha="right", va="center")
    fig.text(x0 + 0.184, y, "target switched", color=BLACK,
             fontsize=11.2, va="center")


def main() -> None:
    _style()
    OUT.mkdir(parents=True, exist_ok=True)
    image, xyz, xy, valid, patch_index, teacher = _load()
    example = _example(xyz, xy, valid, patch_index, teacher)

    fig = plt.figure(figsize=(16.3, 8.9))
    ax_context = fig.add_axes((0.195, 0.755, 0.610, 0.217))
    _photo(ax_context, image, CONTEXT)
    ax_context.add_patch(Rectangle((ZOOM[0], ZOOM[1]), ZOOM[2], ZOOM[3],
                                   facecolor="none", edgecolor=BLACK,
                                   linewidth=1.35, linestyle=(0, (4, 2)), zorder=5))
    fig.text(0.195, 0.982, "Example region", fontsize=13.2,
             fontweight="bold", color=BLACK, ha="left", va="bottom")

    positions = (0.045, 0.268, 0.535, 0.758)
    zoom_axes = [fig.add_axes((x, 0.337, 0.197, 0.327)) for x in positions]
    _pointwise(zoom_axes[0], image, xy, example, shifted=False)
    _pointwise(zoom_axes[1], image, xy, example, shifted=True)
    _patchset(zoom_axes[2], image, xy, example, shifted=False)
    _patchset(zoom_axes[3], image, xy, example, shifted=True)
    for x, name in zip((0.144, 0.367, 0.634, 0.857),
                       ("Aligned", "Offset +3 px", "Aligned", "Offset +3 px")):
        fig.text(x, 0.677, name, ha="center", va="bottom",
                 fontsize=12.6, fontweight="bold", color=BLACK)

    # Both panels use the same perturbation. Colors encode membership changes.
    _pointwise_key(fig, 0.147, 0.716)
    _set_key(fig, 0.634, 0.716)
    fig.text(0.254, 0.276,
             "Small offsets cause discrete target switching near patch boundaries.",
             ha="center", va="center", fontsize=12.2, color=BLACK)
    _token_lanes(fig, example["hierarchy"])
    fig.text(0.745, 0.145,
             "Most supports of $A_j$ are preserved after the same offset.",
             ha="center", va="center", fontsize=12.0, color=BLACK)
    fig.text(0.745, 0.113,
             f"{example['common'].sum()}/{example['before'].sum()} point supports "
             f"retained ({example['common'].sum() / example['before'].sum():.1%}).",
             ha="center", va="center", fontsize=11.7, color=BLACK)
    fig.text(0.254, 0.062,
             "(a) Point-wise targets switch under a small registration offset",
             ha="center", va="center", fontsize=14.0,
             fontweight="bold", color=BLACK)
    fig.text(0.745, 0.062,
             "(b) Patch–token sets remain largely stable",
             ha="center", va="center", fontsize=14.0,
             fontweight="bold", color=BLACK)

    fig.savefig(PNG, dpi=390, bbox_inches="tight", pad_inches=0.035)
    fig.savefig(SVG, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)
    metadata = {
        "source_tile": STEM,
        "orthophoto": str(IMAGE_PATH),
        "pointcloud": str(POINT_PATH),
        "correspondence": str(CORR_PATH),
        "dino_feature": str(DINO_PATH),
        "pixel_coord_storage_order": "row, column; converted to x, y for display",
        "context_xywh_pixels": list(CONTEXT),
        "zoom_xywh_pixels": list(ZOOM),
        "illustrative_offset_xy_pixels": OFFSET_XY.tolist(),
        "selected_real_point_index": POINT_ID,
        "selected_point_pixel_xy": example["point_xy"].tolist(),
        "pointwise_teacher_patch_before_after": [example["old_patch"], example["new_patch"]],
        "pointwise_teacher_cosine_distance": example["teacher_cosine_distance"],
        "pointwise_display_stable_indices": example["stable_sample_ids"].tolist(),
        "pointwise_display_switched_indices": example["switch_sample_ids"].tolist(),
        "patch_set_original_point_count": int(example["before"].sum()),
        "patch_set_after_point_count": int(example["after"].sum()),
        "patch_set_shared_point_count": int(example["common"].sum()),
        "nominal_hierarchy_voxel_support": example["hierarchy"],
        "interpretation_limit": (
            "The illustrative offset and selected patch do not establish "
            "global registration-error robustness. Enc 2/3/4 glyphs are "
            "nominal occupied-grid proxies, not exported learned tokens."
        ),
    }
    JSON.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(PNG)
    print(SVG)


if __name__ == "__main__":
    main()
