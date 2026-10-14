"""Fig. 1: real orthophotos and ALS sections illustrating projection mismatch.

The illustrated patch's surface class is a posterior LAS-label proxy, not a
DINO prediction. The code revalidates the exact Fig. 6 patch correspondence
and q_geo visibility sidecars before drawing either example.

Run from the repository root with the pointspace environment::

    python paper_cvrl/plot_fig1_problem_teaser.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle
import numpy as np
from PIL import Image

from plot_projection_conflict import (
    CASES, CASE_CONFLICT_PATCHES, CASE_SECTION_PIXELS, CLASS_COLORS,
    IMAGE_ROOT, PROFILE_HALF_WIDTH_M, _conflict_patch, _read_case,
    fit_xy_to_pixel_affine,
)
from plot_cvrl_43_results import PALETTE


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper_cvrl" / "results" / "intro"
FIGURE = OUT / "figure_1_visible_surface_bottleneck.png"
AUDIT = OUT / "figure_1_case_audit.json"

IMAGE_CROP_PX = 850                 # 85 m at the 200 m / 2000 px Swiss scale
PROFILE_HALF_LENGTH_M = 42.5
PROFILE_HALF_WIDTH = PROFILE_HALF_WIDTH_M
BLUE = PALETTE["blue"]
RED = PALETTE["vermillion"]
NEUTRAL = "#233442"


def _fit(stem, xyz, pixel):
    rng = np.random.default_rng(2101)
    index = rng.choice(len(xyz), min(len(xyz), 10000), replace=False)
    origin, affine, residual = fit_xy_to_pixel_affine(xyz[index, :2], pixel[index])
    if residual > 2.0:
        raise ValueError(f"{stem}: XYZ-to-pixel median error {residual:.2f} px")
    return origin, affine, residual


def _section_direction(stem, xyz, target_index, origin, affine):
    if stem in CASE_SECTION_PIXELS:
        image_endpoints = np.asarray(CASE_SECTION_PIXELS[stem], dtype=np.float64)
        world_endpoints = origin + (image_endpoints - affine[2]) @ np.linalg.inv(affine[:2])
        vec = np.diff(world_endpoints, axis=0)[0]
        return vec / np.linalg.norm(vec)
    # A profile perpendicular to the local facade trace reveals its height.
    target_xy = xyz[target_index, :2]
    center = np.median(target_xy, axis=0)
    nearby = target_xy[np.linalg.norm(target_xy - center, axis=1) < 8]
    if len(nearby) < 3:
        raise ValueError(f"{stem}: insufficient facade points for direction")
    _, vectors = np.linalg.eigh(np.cov(nearby, rowvar=False))
    tangent = vectors[:, -1]
    return np.array([-tangent[1], tangent[0]])


def _image_panel(ax, stem, image, pixel, patch, origin, affine, center_xy, direction):
    center_row = patch["row"] + patch["size"] / 2
    center_col = patch["col"] + patch["size"] / 2
    half = IMAGE_CROP_PX // 2
    row0 = int(np.clip(center_row - half, 0, image.shape[0] - IMAGE_CROP_PX))
    col0 = int(np.clip(center_col - half, 0, image.shape[1] - IMAGE_CROP_PX))
    tile = image[row0:row0 + IMAGE_CROP_PX, col0:col0 + IMAGE_CROP_PX]
    ax.imshow(tile)

    endpoints = center_xy + np.array([-PROFILE_HALF_LENGTH_M,
                                       PROFILE_HALF_LENGTH_M])[:, None] * direction
    endpoint_pixels = np.column_stack((endpoints - origin, np.ones(2))) @ affine
    ax.plot(endpoint_pixels[:, 1] - col0, endpoint_pixels[:, 0] - row0,
            color="white", linewidth=2.1, linestyle=(0, (4, 3)), zorder=8)
    ax.plot(endpoint_pixels[:, 1] - col0, endpoint_pixels[:, 0] - row0,
            color=NEUTRAL, linewidth=.7, linestyle=(0, (4, 3)), zorder=9)
    ax.text(*(endpoint_pixels[0, ::-1] - [col0, row0]), "A", color="white",
            fontsize=12, weight="bold", zorder=10,
            path_effects=[mpl.patheffects.withStroke(linewidth=2.5, foreground=NEUTRAL)])
    ax.text(*(endpoint_pixels[1, ::-1] - [col0, row0]), "A′", color="white",
            fontsize=12, weight="bold", zorder=10,
            path_effects=[mpl.patheffects.withStroke(linewidth=2.5, foreground=NEUTRAL)])

    x = patch["col"] - col0 - .5
    y = patch["row"] - row0 - .5
    ax.add_patch(Rectangle((x, y), patch["size"], patch["size"],
                           facecolor=BLUE, alpha=.25, edgecolor="none", zorder=11))
    ax.add_patch(Rectangle((x, y), patch["size"], patch["size"],
                           facecolor="none", edgecolor="white", linewidth=1.6,
                           zorder=12))
    for color, width, radius in (("white", 2.7, 24), (RED, 1.8, 24)):
        ax.add_patch(Circle((x + patch["size"] / 2, y + patch["size"] / 2),
                            radius=radius, facecolor="none", edgecolor=color,
                            linewidth=width, zorder=13))
    ax.set_xlim(0, IMAGE_CROP_PX)
    ax.set_ylim(IMAGE_CROP_PX, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#B9C3CB")
        spine.set_linewidth(.8)
    return {"image_crop_row_col": [row0, col0],
            "transect_endpoints_xy": endpoints.tolist()}


def _profile_panel(ax, xyz, classification, region, pixel, patch, center_xy,
                   direction, target_class, surface_class):
    normal = np.array([-direction[1], direction[0]])
    delta = xyz[:, :2] - center_xy
    along = delta @ direction
    across = delta @ normal
    inside = ((np.abs(along) <= PROFILE_HALF_LENGTH_M) &
              (np.abs(across) <= PROFILE_HALF_WIDTH))
    index = np.flatnonzero(inside)
    if len(index) < 50:
        raise ValueError("Too few points in the 0.5 m profile")
    if len(index) > 20000:
        rng = np.random.default_rng(2102)
        index = np.sort(rng.choice(index, 20000, replace=False))

    colors = [CLASS_COLORS.get(int(cls), "#9AA2A9") for cls in classification[index]]
    ax.scatter(along[index], xyz[index, 2], s=5.6, c=colors,
               alpha=.78, linewidths=0, rasterized=True, zorder=2)

    # Only local illustrative points carry visibility outlines. Interior
    # points retain their LAS class fill; red/blue never replace class color.
    same_patch = ((pixel[index, 0] >= patch["row"]) &
                  (pixel[index, 0] < patch["row"] + patch["size"]) &
                  (pixel[index, 1] >= patch["col"]) &
                  (pixel[index, 1] < patch["col"] + patch["size"]))
    visible = index[same_patch & (region[index] == 2) &
                    (classification[index] == surface_class)]
    invisible = np.intersect1d(index, patch["target_index"], assume_unique=True)
    if len(visible) < 3 or len(invisible) < 3:
        raise ValueError(f"Patch section lacks surface/target support: "
                         f"P_vis={len(visible)}, P_inv={len(invisible)}")
    for selected, outline in ((visible, BLUE), (invisible, RED)):
        if len(selected) > 36:
            selected = selected[np.linspace(0, len(selected) - 1, 36, dtype=int)]
        ax.scatter(along[selected], xyz[selected, 2], s=25,
                   facecolors="none", edgecolors=outline, linewidths=.95,
                   rasterized=True, zorder=5)

    interior_count = 0
    if target_class == 2:
        # A real P_inv vegetation point makes the canopy-interior component
        # explicit without conflating it with the ground-target conflict.
        interior = index[same_patch & (region[index] == 0) &
                         (classification[index] == 3)]
        interior_count = int(len(interior))
        if interior_count:
            selected = interior[np.argmin(np.abs(along[interior] + 6))]
            ax.scatter([along[selected]], [xyz[selected, 2]], s=52,
                       facecolors="none", edgecolors=RED, linewidths=1.3,
                       zorder=6)
            ax.annotate("Canopy interior", xy=(along[selected], xyz[selected, 2]),
                        xytext=(9, xyz[selected, 2] + 8),
                        arrowprops={"arrowstyle": "-", "color": NEUTRAL,
                                    "linewidth": .8},
                        fontsize=10, color=NEUTRAL, ha="left", va="center",
                        bbox={"facecolor": "white", "alpha": .86,
                              "edgecolor": "none", "pad": 1.5}, zorder=7)

    z_lo, z_hi = np.percentile(xyz[index, 2], [.3, 99.7])
    ax.set_xlim(-PROFILE_HALF_LENGTH_M, PROFILE_HALF_LENGTH_M)
    ax.set_ylim(z_lo - 2, z_hi + 2)
    ax.set_xlabel("Distance along A–A′ (m)", fontsize=11)
    ax.set_ylabel("Elevation (m)", fontsize=11)
    ax.grid(color="#DFE5E9", linewidth=.65, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#B9C3CB")
        spine.set_linewidth(.8)
    return {"profile_points": int(len(index)),
            "highlight_P_vis_surface": int(len(visible)),
            "highlight_P_inv_target": int(len(invisible)),
            "P_inv_canopy_interior_points": interior_count}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update({"font.family": "Times New Roman", "font.size": 11,
                         "figure.facecolor": "white", "savefig.facecolor": "white"})
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 7.1),
                             gridspec_kw={"width_ratios": [1, 1.58]})
    fig.subplots_adjust(left=.105, right=.985, top=.90, bottom=.17,
                        wspace=.13, hspace=.29)
    axes[0, 0].set_title("Orthophoto: projected image patch", fontsize=14, pad=11)
    axes[0, 1].set_title("ALS: physical vertical structure", fontsize=14, pad=11)
    audit = {"schema": "cvrl_fig1_real_swiss_cases_v1",
             "surface_proxy_rule": "majority LAS class among P_vis points in the patch; not DINO prediction",
             "cases": []}
    labels = (
        ("Canopy over ground", "Visible canopy", "Hidden ground", 3, 2),
        ("Roof over facade", "Visible roof", "Hidden facade", 6, 26),
    )
    for row, ((_, stem, target_class),
              (row_name, visible_name, hidden_name, surface_class, expected_target)) in enumerate(zip(CASES, labels)):
        if target_class != expected_target:
            raise AssertionError("Fig. 1 case ordering differs from Fig. 6")
        xyz, classification, region, pixel = _read_case(stem)
        patch = _conflict_patch(stem, classification, region, pixel, target_class)
        if patch["proxy_class"] != surface_class:
            raise ValueError(f"Unexpected surface class in {stem}")
        image = np.asarray(Image.open(IMAGE_ROOT / f"{stem}.jpg").convert("RGB"))
        origin, affine, residual = _fit(stem, xyz, pixel)
        center_xy = np.median(xyz[patch["target_index"], :2], axis=0)
        direction = _section_direction(stem, xyz, patch["target_index"], origin, affine)
        image_details = _image_panel(axes[row, 0], stem, image, pixel, patch,
                                     origin, affine, center_xy, direction)
        profile_details = _profile_panel(axes[row, 1], xyz, classification, region,
                                          pixel,
                                          patch, center_xy, direction,
                                          target_class, surface_class)
        axes[row, 0].text(-.06, .5, f"({chr(97+row)}) {row_name}",
                          transform=axes[row, 0].transAxes, ha="right", va="center",
                          rotation=90, fontsize=14, weight="bold")
        axes[row, 1].text(.02, .97 if row == 0 else .84,
                          f"Image patch: {visible_name.lower()}\n"
                          f"Projected ALS point: {hidden_name.lower()}",
                          transform=axes[row, 1].transAxes, va="top", ha="left",
                          fontsize=11, color=NEUTRAL,
                          bbox={"facecolor": "white", "alpha": .91,
                                "edgecolor": "none", "pad": 3.5})
        audit["cases"].append({
            "name": row_name, "source_stem": stem, "target_las_class": target_class,
            "surface_proxy_las_class": patch["proxy_class"],
            "patch_id": patch["id"], "surface_proxy_support": patch["support"],
            "surface_proxy_purity": patch["purity"],
            "projected_P_inv_target_count": int(len(patch["target_index"])),
            "affine_median_residual_px": residual,
            "section_center_xy": center_xy.tolist(),
            "section_width_m": PROFILE_HALF_WIDTH * 2,
            **image_details, **profile_details,
        })
        print(row_name, audit["cases"][-1]["profile_points"],
              audit["cases"][-1]["highlight_P_vis_surface"],
              audit["cases"][-1]["highlight_P_inv_target"], flush=True)

    legend = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="none",
               markeredgecolor=BLUE, markeredgewidth=1.5, markersize=8,
               label=r"$P_{vis}$ (outline)"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="none",
               markeredgecolor=RED, markeredgewidth=1.5, markersize=8,
               label=r"$P_{inv}$ (outline)"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=CLASS_COLORS[3],
               markeredgecolor="none", markersize=8, label="Vegetation"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=CLASS_COLORS[2],
               markeredgecolor="none", markersize=8, label="Ground"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=CLASS_COLORS[6],
               markeredgecolor="none", markersize=8, label="Roof"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=CLASS_COLORS[26],
               markeredgecolor="none", markersize=8, label="Facade"),
    ]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(.55, .01),
               ncol=6, frameon=False, fontsize=10.5, columnspacing=1.0,
               handletextpad=.25)
    fig.savefig(FIGURE, dpi=350, bbox_inches="tight", pad_inches=.06)
    plt.close(fig)
    AUDIT.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(FIGURE, flush=True)


if __name__ == "__main__":
    main()
