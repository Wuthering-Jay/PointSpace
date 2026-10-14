"""Draw the publication Fig. 2 for the official CVRL-v1m1 LitePT model.

The solid path is the retained LiDAR network. Dashed paths are pretraining
targets only. Fig. 3 and Fig. 4 carry the fine-grained VPSD/RRCL mechanics.

Run from the repository root::

    python paper_cvrl/plot_fig2_cvrl_framework.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper_cvrl" / "results" / "method"
FIGURE = OUT / "figure_2_cvrl_framework.png"

INK = "#263743"
MUTED = "#5B6B76"
RULE = "#B9C5CD"
PALE = "#F5F7F8"
BLUE = "#386FA8"
BLUE_PALE = "#EFF5FA"
ORANGE = "#BE7050"
ORANGE_PALE = "#FBF3EF"
TEAL = "#237C79"
TEAL_PALE = "#EEF7F5"


def rect(ax, x, y, w, h, face="white", edge=RULE, lw=1.15, radius=.07, z=2):
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0.008,rounding_size={radius}",
        facecolor=face, edgecolor=edge, linewidth=lw, zorder=z,
    )
    ax.add_patch(patch)
    return patch


def line(ax, x0, y0, x1, y1, color=INK, lw=1.5, style="solid", z=2):
    ax.plot([x0, x1], [y0, y1], color=color, linewidth=lw,
            linestyle=style, solid_capstyle="round", zorder=z)


def arrow(ax, a, b, color=INK, lw=1.9, dashed=False, ms=12, z=3):
    ax.add_patch(FancyArrowPatch(
        a, b, arrowstyle="-|>", mutation_scale=ms,
        linewidth=lw, color=color,
        linestyle=(0, (4, 3)) if dashed else "solid", zorder=z,
    ))


def text(ax, x, y, value, size=13, color=INK, weight="normal",
         ha="center", va="center", **kwargs):
    return ax.text(x, y, value, ha=ha, va=va, fontsize=size, color=color,
                   fontweight=weight, zorder=5, **kwargs)


def small_box(ax, x, y, w, h, title, detail="", color=RULE,
              face="white", title_size=12.5, detail_size=10.8):
    rect(ax, x, y, w, h, face=face, edge=color, radius=.045, lw=1.0)
    if detail:
        text(ax, x + w / 2, y + h * .67, title, title_size,
             color=color if color != RULE else INK, weight="bold")
        text(ax, x + w / 2, y + h * .27, detail, detail_size)
    else:
        text(ax, x + w / 2, y + h / 2, title, title_size,
             color=color if color != RULE else INK, weight="bold")


def section_title(ax, x, y, value, color, subtitle):
    text(ax, x, y, value, 17.0, color=color, weight="bold", ha="left")
    text(ax, x, y - .33, subtitle, 11.8, color=MUTED, ha="left")


def lidar_icon(ax):
    """A small schematic point set; no scene data are implied."""
    rng = np.random.default_rng(42)
    ground_x = np.linspace(.53, 1.68, 65)
    ground_y = 3.63 + rng.normal(0, .018, len(ground_x))
    ax.scatter(ground_x, ground_y, s=7, c=INK, linewidths=0, zorder=4)
    x = rng.normal(.88, .16, 95)
    y = rng.normal(4.17, .27, 95)
    keep = ((x - .88) / .34) ** 2 + ((y - 4.17) / .52) ** 2 < 1
    ax.scatter(x[keep], y[keep], s=8, c=BLUE, linewidths=0, zorder=4)
    ax.scatter(rng.normal(.89, .025, 19), np.linspace(3.64, 3.96, 19),
               s=6, c=INK, linewidths=0, zorder=4)
    ax.scatter(np.linspace(1.32, 1.64, 28),
               4.12 + rng.normal(0, .015, 28), s=7,
               c=ORANGE, linewidths=0, zorder=4)
    ax.scatter(np.full(18, 1.64) + rng.normal(0, .009, 18),
               np.linspace(3.65, 4.11, 18), s=6,
               c=ORANGE, linewidths=0, zorder=4)


def hierarchy(ax):
    # Level widths suggest spatial pooling; color marks the protected visual
    # anchors while the five channel counts match the Swiss LitePT config.
    levels = [
        ("P0", "36D", 3.95, 1.84, PALE, RULE),
        ("P1", "72D", 4.35, 1.69, PALE, RULE),
        ("P2", "144D", 4.75, 1.54, BLUE_PALE, BLUE),
        ("P3", "252D", 5.15, 1.39, BLUE_PALE, BLUE),
        ("P4", "504D", 5.55, 1.24, BLUE_PALE, BLUE),
    ]
    for name, channels, y, width, face, edge in levels:
        x = 3.74 - width / 2
        rect(ax, x, y, width, .32, face=face, edge=edge,
             radius=.03, lw=1.0)
        text(ax, x + .42, y + .16, name, 11.3, color=INK, weight="bold")
        text(ax, x + width - .37, y + .16, channels, 10.9, color=MUTED)
    for y, color in ((4.28, MUTED), (4.68, MUTED),
                     (5.08, BLUE), (5.48, BLUE)):
        arrow(ax, (3.74, y), (3.74, y + .07), color=color, lw=1.0, ms=7)


def patch_icon(ax, x, y):
    for r in range(3):
        for c in range(3):
            selected = r == 1 and c == 1
            ax.add_patch(Rectangle(
                (x + c * .135, y + r * .135), .119, .119,
                facecolor="#8EB6D8" if selected else "white",
                edgecolor=BLUE if selected else RULE,
                linewidth=1.0 if selected else .6, zorder=4,
            ))
    for px, py in ((x + .17, y + .20), (x + .23, y + .21),
                   (x + .25, y + .27)):
        ax.scatter([px], [py], s=10, color=BLUE, zorder=5)


def downstream_icon(ax):
    rng = np.random.default_rng(8)
    x = np.linspace(13.05, 14.87, 80)
    y = 2.43 + rng.normal(0, .015, len(x))
    ax.scatter(x, y, s=9, c="#B49C7A", linewidths=0, zorder=5)
    tx = rng.normal(13.48, .17, 65)
    ty = rng.normal(2.78, .18, 65)
    ax.scatter(tx, ty, s=10, c="#25895E", linewidths=0, zorder=5)
    ax.scatter(np.linspace(14.17, 14.65, 32),
               2.83 + rng.normal(0, .012, 32), s=10,
               c="#4778B9", linewidths=0, zorder=5)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update({
        "font.family": "Times New Roman", "font.size": 13,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })
    fig, ax = plt.subplots(figsize=(15.6, 8.6))
    fig.subplots_adjust(left=.018, right=.982, top=.985, bottom=.025)
    ax.set_xlim(0, 15.6)
    ax.set_ylim(0, 8.6)
    ax.axis("off")

    # Large structural regions; small steps are shown without separate
    # decorative boxes around every word.
    rect(ax, .20, 2.50, 1.76, 3.67, face="white", edge=RULE, lw=1.2)
    rect(ax, 2.30, 1.69, 2.85, 5.37, face=PALE, edge=INK, lw=1.35)
    rect(ax, 5.48, 4.53, 6.52, 3.63, face=BLUE_PALE, edge=BLUE, lw=1.35)
    rect(ax, 5.48, .47, 6.52, 3.62, face=ORANGE_PALE, edge=ORANGE, lw=1.35)
    rect(ax, 12.37, 1.70, 2.91, 5.36, face=TEAL_PALE,
         edge=TEAL, lw=1.35)

    # Input and one (not two) hierarchy.
    text(ax, 1.08, 5.80, "LiDAR input", 17, weight="bold")
    text(ax, 1.08, 5.43, "XYZ (+ intensity)", 12, color=MUTED)
    lidar_icon(ax)
    text(ax, 1.08, 2.86, "No image / return input", 10.9, color=MUTED)
    section_title(ax, 2.63, 6.69, "Shared LitePT encoder", INK,
                  "one multiscale hierarchy")
    hierarchy(ax)
    text(ax, 3.74, 3.51, "stem / P0 / P1", 12.6, weight="bold")
    text(ax, 3.74, 3.18, "shared gradients · PCGrad", 11.2, color=MUTED)
    line(ax, 2.72, 2.91, 4.76, 2.91, color=RULE, lw=.8)
    text(ax, 3.74, 2.61, "P2–P4: protected anchors", 11.8,
         color=BLUE, weight="bold")
    text(ax, 3.74, 2.20, "RRCL decoder gradients detached", 10.8,
         color=MUTED)
    arrow(ax, (1.98, 4.33), (2.27, 4.33), lw=2.3)
    arrow(ax, (5.17, 5.79), (5.46, 6.22), color=BLUE, lw=2.2)
    arrow(ax, (5.17, 3.02), (5.46, 2.97), color=ORANGE, lw=2.2)

    # VPSD: visible patch-set supervision and persistent visual readout.
    section_title(ax, 5.79, 7.79, "VPSD · visual pathway", BLUE,
                  r"only matched points in $P_{vis}$ receive visual targets")
    text(ax, 6.26, 7.03, "point–patch\nrelations", 11.0, color=BLUE)
    patch_icon(ax, 6.08, 6.28)
    arrow(ax, (6.72, 6.52), (7.06, 6.52), color=BLUE, lw=1.7)
    small_box(ax, 7.09, 6.13, 1.43, .78, "Patch-set", "aggregation",
              color=BLUE, face="white", title_size=12.1, detail_size=11.3)
    arrow(ax, (8.55, 6.52), (8.85, 6.52), color=BLUE, lw=1.7)
    small_box(ax, 8.88, 6.13, 1.37, .78, "Projector", "900 → 1024D",
              color=BLUE, face="white", title_size=12.1, detail_size=11.2)
    arrow(ax, (10.28, 6.52), (10.55, 6.52), color=BLUE, lw=1.7)
    small_box(ax, 10.58, 6.13, 1.08, .78, r"$\mathcal{L}_{V}$", "cosine",
              color=BLUE, face="white", title_size=14, detail_size=11.2)
    small_box(ax, 10.38, 7.04, 1.48, .54, "DOM → DINO", "patch 1024D",
              color=BLUE, face="white", title_size=11.0,
              detail_size=10.2)
    arrow(ax, (11.12, 7.02), (11.12, 6.93), color=BLUE, dashed=True,
          lw=1.3, ms=8)
    text(ax, 9.33, 5.72, "same projector, point-wise readout", 11.1,
         color=MUTED)
    line(ax, 5.82, 5.50, 11.70, 5.50, color="#C8D8E8", lw=.9)
    text(ax, 6.59, 5.10, "P2 / P3 / P4", 12.4, color=BLUE, weight="bold")
    arrow(ax, (7.48, 5.10), (7.73, 5.10), color=BLUE, lw=1.7)
    text(ax, 8.55, 5.10, "projector 1024D", 12.1, color=INK)
    arrow(ax, (9.46, 5.10), (9.68, 5.10), color=BLUE, lw=1.7)
    text(ax, 10.25, 5.10, "fixed PCA", 12.1, color=INK)
    arrow(ax, (10.82, 5.10), (11.00, 5.10), color=BLUE, lw=1.7)
    text(ax, 11.48, 5.10, r"$z^V$ 128D", 13.4,
         color=BLUE, weight="bold")

    # RRCL: decoder feature retained; lightweight pretext heads discarded.
    section_title(ax, 5.79, 3.73, "RRCL · return pathway", ORANGE,
                  "role/context targets only for valid LAS returns")
    small_box(ax, 5.83, 2.66, 2.28, .64, "LitePT decoder", "",
              color=ORANGE, face="white", title_size=13.5)
    arrow(ax, (8.13, 2.98), (9.76, 2.98), color=ORANGE, lw=2.0)
    small_box(ax, 9.79, 2.66, 1.88, .64, r"$z^R$ 72D", "",
              color=ORANGE, face="white", title_size=14)
    text(ax, 6.71, 2.14, "training-only targets", 11.2, color=MUTED)
    small_box(ax, 5.83, 1.22, 1.90, .70, "LAS r, n", "role + KNN histogram",
              color=ORANGE, face="white", title_size=12.7, detail_size=10.9)
    arrow(ax, (7.75, 1.57), (8.10, 1.57), color=ORANGE, dashed=True,
          lw=1.4, ms=9)
    small_box(ax, 8.13, 1.20, 1.52, .74, "Role head", "S / F / M / L",
              color=ORANGE, face="white", title_size=12.1, detail_size=10.8)
    small_box(ax, 10.10, 1.20, 1.56, .74, "Context head", "local proportions",
              color=ORANGE, face="white", title_size=12.1, detail_size=10.4)
    # Independent dashed metadata targets; neither head feeds the other.
    line(ax, 6.77, 1.20, 6.77, 1.04, color=ORANGE, lw=1.25,
         style=(0, (4, 3)))
    line(ax, 6.77, 1.04, 10.88, 1.04, color=ORANGE, lw=1.25,
         style=(0, (4, 3)))
    arrow(ax, (10.88, 1.04), (10.88, 1.18), color=ORANGE,
          dashed=True, lw=1.25, ms=8)
    # Both heads read zR; their predictions are discarded after pretraining.
    arrow(ax, (10.20, 2.64), (8.89, 1.97), color=ORANGE, lw=1.35, ms=8)
    arrow(ax, (10.86, 2.64), (10.88, 1.97), color=ORANGE, lw=1.35, ms=8)
    text(ax, 9.73, .81, r"$\mathcal{L}_{R}=\mathcal{L}_{role}+\lambda_{ctx}\mathcal{L}_{ctx}$",
         12.0, color=ORANGE)

    # Persistent representation and its LiDAR-only downstream use.
    section_title(ax, 12.63, 6.67, "Retained backbone", TEAL,
                  "pretext heads discarded")
    bar_x, bar_y, bar_w = 12.77, 5.50, 2.10
    ax.add_patch(Rectangle((bar_x, bar_y), bar_w * .64, .55,
                           facecolor=BLUE, edgecolor="none", zorder=4))
    ax.add_patch(Rectangle((bar_x + bar_w * .64, bar_y), bar_w * .36, .55,
                           facecolor=ORANGE, edgecolor="none", zorder=4))
    text(ax, bar_x + bar_w * .32, bar_y + .28, "128D", 12.2,
         color="white", weight="bold")
    text(ax, bar_x + bar_w * .82, bar_y + .28, "72D", 12.2,
         color="white", weight="bold")
    text(ax, 13.84, 5.11, r"$z^{CVRL}=z^V\Vert z^R$", 14.1,
         color=TEAL, weight="bold")
    text(ax, 13.84, 4.75, "200D point representation", 12.4,
         color=TEAL)
    arrow(ax, (13.84, 4.50), (13.84, 4.23), color=TEAL, lw=2.0)
    small_box(ax, 12.70, 3.35, 2.27, .77, "LiDAR-only", "downstream fine-tuning",
              color=TEAL, face="white", title_size=13.5,
              detail_size=11.1)
    text(ax, 13.84, 3.06, "semantic segmentation", 11.7,
         color=MUTED)
    downstream_icon(ax)
    text(ax, 13.84, 1.99, "No DOM, DINO or r/n needed", 10.9,
         color=MUTED)
    arrow(ax, (11.99, 5.10), (12.35, 5.58), color=BLUE, lw=2.2)
    arrow(ax, (11.99, 2.98), (12.35, 5.04), color=ORANGE, lw=2.2)

    # Minimal key: target paths are dashed, persistent feature paths solid.
    line(ax, .48, .20, .84, .20, color=INK, lw=2.0)
    text(ax, .94, .20, "retained feature flow", 11.2,
         color=MUTED, ha="left")
    line(ax, 2.61, .20, 2.97, .20, color=ORANGE, lw=1.5,
         style=(0, (4, 3)))
    text(ax, 3.07, .20, "training targets only", 11.2,
         color=MUTED, ha="left")

    fig.savefig(FIGURE, dpi=350, bbox_inches="tight", pad_inches=.035)
    plt.close(fig)
    print(FIGURE)


if __name__ == "__main__":
    main()
