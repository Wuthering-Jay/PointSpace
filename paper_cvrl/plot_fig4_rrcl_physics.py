"""Legacy Fig. 4 diagnostic; the final paper figure uses plot_fig4_rrcl_mechanism.

The point locations, LAS return fields, pulse examples, and K=48
neighbourhoods are read from one Swiss-Audit tile. Only arrows and labels are
explanatory graphics. Run from the repository root::

    python paper_cvrl/plot_fig4_rrcl_physics.py
"""

from __future__ import annotations

import json
from pathlib import Path

import laspy
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper_cvrl" / "results" / "method"
FIGURE = OUT / "figure_4_rrcl_legacy_diagnostic.png"
SVG = OUT / "figure_4_rrcl_legacy_diagnostic.svg"
META = OUT / "figure_4_rrcl_legacy_diagnostic.json"

STEM = "swisssurface3d_2020_2760-1188_2056_5728_0086"
POINT_PATH = Path(
    rf"G:\data\swiss\joint_tiles_100\Swiss-Audit\pointcloud\{STEM}.las"
)
K = 48

INK = "#263743"
MUTED = "#65747D"
S, F, M, L = "#4C78A8", "#F58518", "#B279A2", "#54A24B"
ROLE_COLORS = np.asarray([mpl.colors.to_rgba(c) for c in (S, F, M, L)])
ROLE_NAMES = ("Single", "First", "Middle", "Last")
ROLE_CODES = ("S", "F", "M", "L")


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 12,
            "axes.linewidth": 0.75,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _roles(return_number: np.ndarray, num_returns: np.ndarray) -> np.ndarray:
    role = np.full(return_number.shape, -1, dtype=np.int8)
    single = num_returns == 1
    first = (num_returns > 1) & (return_number == 1)
    last = (num_returns > 1) & (return_number == num_returns)
    middle = (
        (num_returns > 1)
        & (return_number > 1)
        & (return_number < num_returns)
    )
    role[single] = 0
    role[first] = 1
    role[middle] = 2
    role[last] = 3
    return role


def _load():
    if not POINT_PATH.is_file():
        raise FileNotFoundError(POINT_PATH)
    las = laspy.read(POINT_PATH)
    xyz = np.column_stack((las.x, las.y, las.z)).astype(np.float64)
    rn = np.asarray(las.return_number, dtype=np.int16)
    nr = np.asarray(las.number_of_returns, dtype=np.int16)
    gps = np.asarray(las.gps_time, dtype=np.float64)
    labels = np.asarray(las.classification, dtype=np.int16)
    return xyz, rn, nr, gps, labels, _roles(rn, nr)


def _complete_multi_pulse(xyz, rn, nr, gps) -> np.ndarray:
    """Select a complete, vertically extensive real pulse inside tile margins."""
    order = np.argsort(gps, kind="stable")
    gs = gps[order]
    starts = np.r_[0, np.flatnonzero(np.diff(gs)) + 1]
    stops = np.r_[starts[1:], len(order)]
    lo = xyz[:, :2].min(axis=0) + 18.0
    hi = xyz[:, :2].max(axis=0) - 18.0
    best = None
    best_score = -np.inf
    for start, stop in zip(starts, stops):
        idx = order[start:stop]
        # Three-return pulses make F/M/L readable without turning the method
        # figure into a dense pulse catalogue.
        if len(idx) != 3:
            continue
        n = int(nr[idx[0]])
        if n != len(idx) or not np.all(nr[idx] == n):
            continue
        if not np.array_equal(np.sort(rn[idx]), np.arange(1, n + 1)):
            continue
        center = xyz[idx, :2].mean(axis=0)
        if not np.all((center >= lo) & (center <= hi)):
            continue
        span = float(np.ptp(xyz[idx, 2]))
        score = span - 0.18 * n
        if score > best_score:
            best_score, best = score, idx[np.argsort(rn[idx])]
    if best is None:
        raise RuntimeError("No complete multi-return pulse found")
    return best


def _profile(xyz, pulse_idx):
    center = xyz[pulse_idx, :2].mean(axis=0)
    direction = np.asarray([0.34, 0.94], dtype=np.float64)
    direction /= np.linalg.norm(direction)
    normal = np.asarray([-direction[1], direction[0]])
    half_length, half_width = 47.0, 0.85
    relative = xyz[:, :2] - center
    along = relative @ direction
    across = relative @ normal
    mask = (np.abs(along) <= half_length) & (np.abs(across) <= half_width)
    return mask, along + half_length, center, direction, half_length, half_width


def _select_contexts(xyz, role, labels):
    tree = cKDTree(xyz)
    rng = np.random.default_rng(44)
    valid = np.flatnonzero(role >= 0)
    candidates = rng.choice(valid, size=min(9000, len(valid)), replace=False)
    _, query = tree.query(xyz[candidates], k=K + 1, workers=-1)
    query = query[:, 1:]
    neighbor_roles = role[query]
    frac = np.stack([(neighbor_roles == c).mean(axis=1) for c in range(4)], axis=1)
    z_span = np.ptp(xyz[query, 2], axis=1)

    # Labels only select representative examples; they never form RRCL targets.
    layered_ok = (
        (labels[candidates] == 3)
        & ((frac[:, 1] + frac[:, 2]) > 0.50)
        & ((frac[:, 0] + frac[:, 3]) > 0.12)
    )
    layered_score = (
        frac[:, 1] + frac[:, 2]
        + 0.08 * np.clip(z_span, 0, 4)
        + 0.25 * (frac[:, 0] + frac[:, 3])
    )
    layered_score[~layered_ok] = -np.inf
    layered_pos = int(np.argmax(layered_score))
    surface_ok = (
        np.isin(labels[candidates], (2, 6))
        & ((frac[:, 0] + frac[:, 3]) > 0.75)
        & (frac[:, 0] > 0.05) & (frac[:, 3] > 0.05)
    )
    surface_score = (
        frac[:, 0] + frac[:, 3]
        - 0.05 * np.clip(z_span, 0, 4)
        + 0.20 * np.minimum(frac[:, 0], frac[:, 3])
    )
    surface_score[~surface_ok] = -np.inf
    surface_pos = int(np.argmax(surface_score))
    selected = []
    for pos in (layered_pos, surface_pos):
        center = int(candidates[pos])
        neighbors = query[pos]
        counts = np.bincount(role[neighbors], minlength=4)
        selected.append((center, neighbors, counts))
    return selected


def _plot_profile(ax, xyz, role, profile_mask, distance, pulse_idx):
    idx = np.flatnonzero(profile_mask & (role >= 0))
    if len(idx) > 28000:
        rng = np.random.default_rng(7)
        idx = np.sort(rng.choice(idx, 28000, replace=False))
    sizes = np.where(role[idx] == 2, 5.0, 3.4)
    ax.scatter(
        distance[idx], xyz[idx, 2], s=sizes, c=ROLE_COLORS[role[idx]],
        linewidths=0, alpha=0.82, rasterized=True, zorder=2,
    )
    pidx = pulse_idx[np.argsort(xyz[pulse_idx, 2])[::-1]]
    ax.plot(
        distance[pidx], xyz[pidx, 2], color=INK, linewidth=1.0,
        linestyle=(0, (3, 2)), zorder=5,
    )
    ax.scatter(
        distance[pidx], xyz[pidx, 2], s=72,
        c=ROLE_COLORS[role[pidx]], edgecolors="white", linewidths=1.0,
        zorder=7,
    )
    for point in pidx:
        ax.annotate(
            ROLE_CODES[role[point]],
            (distance[point], xyz[point, 2]), xytext=(6, 0),
            textcoords="offset points", ha="left", va="center",
            fontsize=10.5, weight="bold", color=ROLE_COLORS[role[point]],
            zorder=9,
        )
    pulse_center = distance[pidx].mean()
    candidates = idx[(role[idx] == 0) & (np.abs(distance[idx] - pulse_center) < 14)]
    if len(candidates):
        single = candidates[np.argmax(xyz[candidates, 2])]
        ax.scatter(
            distance[single], xyz[single, 2], s=72, c=S,
            edgecolors="white", linewidths=1.0, zorder=7,
        )
        ax.annotate(
            "S", (distance[single], xyz[single, 2]), xytext=(6, 0),
            textcoords="offset points", fontsize=10.5, weight="bold",
            color=S, va="center", zorder=9,
        )
        ax.annotate(
            "single surface response", (distance[single], xyz[single, 2]),
            xytext=(-34, 35), textcoords="offset points", ha="center",
            fontsize=10.0, color=MUTED,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8),
        )
    ax.annotate(
        "one measured pulse", (distance[pidx[0]], xyz[pidx[0], 2]),
        xytext=(28, 34), textcoords="offset points", ha="left",
        fontsize=10.0, color=MUTED,
        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8),
    )
    ax.set_xlabel("Distance along profile (m)", labelpad=4)
    ax.set_ylabel("Elevation (m)", labelpad=4)
    ax.grid(True, color="#E2E6E9", linewidth=0.55, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("(a) Return-role formation", fontsize=15, fontweight="bold", pad=9)
    ax.text(
        0.5, -0.20,
        "Return roles encode laser–scene interaction, not semantic classes.",
        transform=ax.transAxes, ha="center", va="top", fontsize=11.4,
        color=INK, fontweight="bold",
    )


def _project_local(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = points - points[0]
    u = centered[:, 0] - 0.52 * centered[:, 1]
    v = 0.26 * centered[:, 0] + 0.18 * centered[:, 1] + 1.12 * centered[:, 2]
    return u, v


def _plot_neighborhood(ax, xyz, role, example, title):
    center, neighbors, counts = example
    indices = np.r_[center, neighbors]
    u, v = _project_local(xyz[indices])
    for x, y in zip(u[1:], v[1:]):
        ax.plot([u[0], x], [v[0], y], color="#D7DDE1", lw=0.35, zorder=0)
    ax.scatter(
        u[1:], v[1:], s=29, c=ROLE_COLORS[role[neighbors]],
        edgecolors="white", linewidths=0.35, zorder=3,
    )
    ax.scatter(
        u[0], v[0], s=90, facecolor="white", edgecolor=INK,
        linewidth=1.4, zorder=5,
    )
    ax.text(u[0], v[0], "i", ha="center", va="center", fontsize=9.5,
            color=INK, fontweight="bold", zorder=6)
    ax.set_aspect("equal", adjustable="datalim")
    ax.axis("off")
    ax.set_title(title, fontsize=11.4, color=INK, pad=2)
    return counts


def _plot_hist(ax, counts):
    proportions = counts / counts.sum()
    bars = ax.bar(
        np.arange(4), proportions, width=0.68,
        color=[S, F, M, L], edgecolor="none", zorder=3,
    )
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.018,
            f"{count}/{counts.sum()}", ha="center", va="bottom",
            fontsize=8.8, color=INK,
        )
    ax.set_xticks(np.arange(4), ROLE_CODES, fontweight="bold")
    ax.set_ylim(0, max(0.68, float(proportions.max() + 0.12)))
    ax.set_ylabel("Proportion", fontsize=9.5)
    ax.tick_params(axis="y", labelsize=8.5, length=2.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E5E8EA", linewidth=0.5, zorder=0)


def _plot_context(fig, spec, xyz, role, examples):
    outer = spec.subgridspec(
        3, 2, height_ratios=(0.18, 1.0, 1.0), width_ratios=(1.18, 0.92),
        hspace=0.30, wspace=0.18,
    )
    title_ax = fig.add_subplot(outer[0, :])
    title_ax.axis("off")
    title_ax.set_title("(b) Local return-role context", fontsize=15,
                       fontweight="bold", pad=0)
    title_ax.text(
        0.5, 0.02, r"XYZ-KNN, $K=48$; center point excluded",
        ha="center", va="bottom", fontsize=10.7, color=MUTED,
    )
    names = ("Vertically layered neighborhood", "Surface-dominated neighborhood")
    saved = []
    for row, (example, title) in enumerate(zip(examples, names), start=1):
        cloud_ax = fig.add_subplot(outer[row, 0])
        hist_ax = fig.add_subplot(outer[row, 1])
        counts = _plot_neighborhood(cloud_ax, xyz, role, example, title)
        _plot_hist(hist_ax, counts)
        saved.append(counts.tolist())
    return saved


def main() -> None:
    _style()
    OUT.mkdir(parents=True, exist_ok=True)
    xyz, rn, nr, gps, labels, role = _load()
    pulse_idx = _complete_multi_pulse(xyz, rn, nr, gps)
    profile_mask, distance, center, direction, half_length, half_width = _profile(
        xyz, pulse_idx
    )
    examples = _select_contexts(xyz, role, labels)

    fig = plt.figure(figsize=(17.5, 7.7))
    grid = fig.add_gridspec(
        1, 2, width_ratios=(0.46, 0.54), left=0.055, right=0.985,
        top=0.94, bottom=0.14, wspace=0.12,
    )
    profile_ax = fig.add_subplot(grid[0, 0])
    _plot_profile(profile_ax, xyz, role, profile_mask, distance, pulse_idx)
    context_counts = _plot_context(fig, grid[0, 1], xyz, role, examples)

    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=color,
               markeredgecolor="none", markersize=7.5, label=f"{code} — {name}")
        for color, code, name in zip((S, F, M, L), ROLE_CODES, ROLE_NAMES)
    ]
    fig.legend(
        handles=legend, loc="lower center", ncol=4, frameon=False,
        bbox_to_anchor=(0.5, 0.012), fontsize=11.0, handletextpad=0.45,
        columnspacing=1.8,
    )
    fig.savefig(FIGURE, dpi=450, bbox_inches="tight", pad_inches=0.035)
    fig.savefig(SVG, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)

    metadata = {
        "source_tile": STEM,
        "pointcloud": str(POINT_PATH),
        "num_points": int(len(xyz)),
        "role_definition": {
            "S": "number_of_returns == 1",
            "F": "n > 1 and return_number == 1",
            "M": "n > 1 and 1 < return_number < n",
            "L": "n > 1 and return_number == n",
        },
        "profile": {
            "center_xy": center.tolist(),
            "direction_xy": direction.tolist(),
            "length_m": float(2 * half_length),
            "width_m": float(2 * half_width),
            "points": int(profile_mask.sum()),
            "highlighted_pulse_point_indices": pulse_idx.tolist(),
            "highlighted_return_numbers": rn[pulse_idx].tolist(),
        },
        "context": {
            "k": K,
            "center_excluded": True,
            "examples": [
                {
                    "center_point_index": int(example[0]),
                    "center_xyz": xyz[example[0]].tolist(),
                    "counts_S_F_M_L": counts,
                }
                for example, counts in zip(examples, context_counts)
            ],
        },
        "note": (
            "Semantic labels select representative examples for display only; "
            "RRCL targets use LAS return fields only."
        ),
    }
    META.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(FIGURE)
    print(SVG)


if __name__ == "__main__":
    main()
