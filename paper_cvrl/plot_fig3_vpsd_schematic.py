"""Draw Fig. 3 over a real Swiss orthophoto with schematic point/token supports.

The geometry is intentionally illustrative. It must not be interpreted as a
measured registration-robustness result or as actual exported LitePT tokens.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, Rectangle
from PIL import Image, ImageEnhance


OUT = Path(__file__).resolve().parent / "results" / "method"
STEM = "figure_3_vpsd_patch_set_alignment"
IMAGE = Path(r"G:\data\swiss\joint_tiles_100\image\swisssurface3d_2020_2760-1188_2056_5728_0086.jpg")
IMAGE_CROP = (226, 170, 482, 426)
BLUE = "#285B83"
ORANGE = "#CF6E3E"
GRAY = "#83909B"
INK = "#192630"
GRID = "#DEE5E9"
PATCH_BLUE = "#E9F1F7"
PATCH_WARM = "#F8F0EA"
SHIFT = -0.20
PHOTO = None


def style() -> None:
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "mathtext.fontset": "stix",
        "svg.fonttype": "none",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })


def window(ax, *, shifted: bool, pointwise: bool) -> None:
    """Draw the same stationary point coordinates against two patch grids."""
    d = SHIFT if shifted else 0.0
    ax.set_xlim(0.48, 3.30)
    ax.set_ylim(0.22, 3.36)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#BFC9CF")
        spine.set_linewidth(0.85)

    # Real roof–vegetation boundary: image is context, not a point/token target.
    ax.imshow(PHOTO, extent=(0, 4, 0, 4), origin="upper",
              interpolation="bilinear", alpha=0.88, zorder=-1)

    # The image grid and its labels move; the ALS point coordinates never do.
    ax.add_patch(Rectangle((1 + d, 1), 1, 1, facecolor=PATCH_BLUE,
                           edgecolor="none", alpha=0.38, zorder=0))
    if pointwise:
        ax.add_patch(Rectangle((2 + d, 1), 1, 1, facecolor=PATCH_WARM,
                               edgecolor="none", alpha=0.38, zorder=0))
    for x in np.arange(0, 5, dtype=float) + d:
        if ax.get_xlim()[0] < x < ax.get_xlim()[1]:
            ax.axvline(x, color="white", alpha=0.72, linewidth=0.8, zorder=1)
    for y in (1, 2, 3):
        ax.axhline(y, color="white", alpha=0.72, linewidth=0.8, zorder=1)
    ax.plot([2 + d, 2 + d], [0.25, 3.30], color="#657989",
            linewidth=1.15, zorder=2)
    ax.add_patch(Rectangle((1 + d, 1), 1, 1, facecolor="none",
                           edgecolor=BLUE, linewidth=1.25, zorder=3))
    label_bg = dict(facecolor="white", alpha=0.76, edgecolor="none", pad=1.4)
    ax.text(1.5 + d, 2.13, r"Patch $j$", fontsize=15, color=INK,
            ha="center", va="bottom", bbox=label_bg)
    if pointwise:
        ax.text(2.5 + d, 2.13, r"Patch $j+1$", fontsize=15, color=INK,
                ha="center", va="bottom", bbox=label_bg)
    else:
        ax.text(1.5 + d, 0.83, r"$A_j$", fontsize=21,
                fontweight="bold", color="#173F5C", ha="center", va="top",
                bbox=label_bg)
    if shifted:
        ax.annotate("", xy=(1.80, 3.05), xytext=(2.00, 3.05),
                    arrowprops={"arrowstyle": "->", "color": INK,
                                "linewidth": 1.0})
        ax.text(1.90, 3.15, r"$\delta$", ha="center", va="bottom",
                fontsize=16, color=INK, bbox=label_bg)


# Eight fixed ALS locations. Three cross the patch boundary after the grid
# moves left; the rightmost one belongs to j+1 under both alignments.
POINT_XY = np.array([
    [1.28, 1.26], [1.45, 1.68], [1.62, 1.39], [1.70, 1.79],
    [1.89, 1.42], [1.94, 1.68], [1.975, 1.22], [2.60, 1.51],
])


def pointwise(ax, shifted: bool) -> None:
    window(ax, shifted=shifted, pointwise=True)
    boundary = 2 + (SHIFT if shifted else 0.0)
    right = POINT_XY[:, 0] >= boundary
    colors = np.where(right, ORANGE, BLUE)
    ax.scatter(POINT_XY[:, 0], POINT_XY[:, 1], s=75, c=colors,
               edgecolors="white", linewidths=1.1, zorder=5)
    # Thin halos on the three points whose assigned teacher changes.
    if shifted:
        changed = (POINT_XY[:, 0] >= 2 + SHIFT) & (POINT_XY[:, 0] < 2)
        ax.scatter(POINT_XY[changed, 0], POINT_XY[changed, 1], s=150,
                   facecolors="none", edgecolors=ORANGE,
                   linewidths=1.25, zorder=4)
        # These arrows indicate changed teacher assignment, not point motion.
        for y in (1.22, 1.68):
            ax.annotate("", xy=(2.36, y), xytext=(2.045, y),
                        arrowprops={"arrowstyle": "->", "color": ORANGE,
                                    "linewidth": 1.05, "mutation_scale": 8},
                        zorder=6)


# Few broad receptive supports instead of token-as-point glyphs. Their spatial
# extent and count are illustrative; they do not depict measured receptive fields.
TOKEN_FOOTPRINTS = [
    (1.24, 1.35, 1.16, 0.56, -17),
    (1.57, 1.66, 1.34, 0.61, 13),
    (1.38, 1.90, 1.13, 0.45, -8),
]


def patch_set(ax, shifted: bool) -> None:
    window(ax, shifted=shifted, pointwise=False)
    for i, (x, y, width, height, angle) in enumerate(TOKEN_FOOTPRINTS, 1):
        ax.add_patch(Ellipse((x, y), width, height, angle=angle,
                             facecolor=BLUE, alpha=0.15,
                             edgecolor="none", zorder=4))
        ax.add_patch(Ellipse((x, y), width, height, angle=angle,
                             facecolor="none", edgecolor=BLUE,
                             linewidth=1.7, zorder=5))
        ax.text(x, y, rf"$u_{i}$", ha="center", va="center",
                color="#3B5B70", fontsize=12.5, zorder=6,
                bbox=dict(facecolor="white", alpha=0.63,
                          edgecolor="none", pad=0.7))


def legend(fig) -> None:
    y = 0.950
    entries = [
        (0.280, "●", BLUE, r"target $t_j$"),
        (0.468, "●", ORANGE, r"target $t_{j+1}$"),
        (0.682, "○", BLUE, "token support region"),
    ]
    for x, marker, color, label in entries:
        fig.text(x, y, marker, color=color,
                 fontsize=17, ha="right", va="center")
        fig.text(x + 0.004, y, label, color=INK, fontsize=12.7,
                 ha="left", va="center")


def main() -> None:
    global PHOTO
    style()
    OUT.mkdir(parents=True, exist_ok=True)
    with Image.open(IMAGE) as source:
        PHOTO = np.asarray(ImageEnhance.Contrast(
            ImageEnhance.Brightness(source.crop(IMAGE_CROP).convert("RGB"))
            .enhance(1.18)).enhance(0.92))
    fig = plt.figure(figsize=(14.8, 5.15))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    xs = [0.043, 0.282, 0.535, 0.774]
    axes = [fig.add_axes((x, 0.225, 0.181, 0.630)) for x in xs]
    pointwise(axes[0], False)
    pointwise(axes[1], True)
    patch_set(axes[2], False)
    patch_set(axes[3], True)
    for x, label in zip([0.133, 0.372, 0.625, 0.864],
                        ["Aligned", r"Offset $+\delta$",
                         "Aligned", r"Offset $+\delta$"]):
        fig.text(x, 0.844, label, ha="center", va="bottom",
                 color=INK, fontsize=14.0, fontweight="bold")
    fig.text(0.253, 0.181,
             "(a) Point-wise assignment is boundary-sensitive",
             ha="center", va="center", color=INK, fontsize=15.2,
             fontweight="bold")
    fig.text(0.747, 0.181,
             "(b) Patch-token set remains largely stable",
             ha="center", va="center", color=INK, fontsize=15.2,
             fontweight="bold")
    legend(fig)

    png = OUT / f"{STEM}.png"
    svg = OUT / f"{STEM}.svg"
    meta = OUT / f"{STEM}.json"
    fig.savefig(png, dpi=450, bbox_inches="tight", pad_inches=0.035)
    fig.savefig(svg, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)
    meta.write_text(json.dumps({
        "figure_type": "real orthophoto background with illustrative points and token footprints",
        "orthophoto": str(IMAGE),
        "image_crop_xyxy_pixels": list(IMAGE_CROP),
        "offset": "patch grid shifts 0.20 illustrative patch widths to the left; ALS points fixed",
        "pointwise": {"points": len(POINT_XY), "targets_switched": 3},
        "patch_set": {"illustrative_broad_footprints": len(TOKEN_FOOTPRINTS)},
        "caution": "Only the orthophoto is measured data. Point positions, offsets, token footprints, and membership are schematic; broad supports do not guarantee robustness.",
    }, indent=2), encoding="utf-8")
    print(png)
    print(svg)


if __name__ == "__main__":
    main()
