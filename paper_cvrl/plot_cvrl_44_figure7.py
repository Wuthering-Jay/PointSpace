"""Create the paper-facing Fig. 7 for Section 4.4.

Panel (a) uses *class IoU* inside spatial structure windows, rather than
renaming class recall as IoU. The under-canopy window contains all held-out
audit points whose 1 m XY column has vegetation at least 3 m above; the
exposed-ground window contains all P_vis points in columns without overstory.
Vegetation IoU is evaluated over P_inv/P_vis, while facade/roof IoU uses the
fixed full test set. Panel (b) reuses the preregistered five-seed
representation-surgery results.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_cvrl.analyze_cvrl_44_strata import (
    AUDIT_ROOT, BOOTSTRAP, SEED, geometry_strata, predictions,
)
from paper_cvrl.section43_metrics import mean_iou

RESULT_ROOT = ROOT / "paper_cvrl/results/section_44"
OUTPUT = RESULT_ROOT / "figure_7_return_complementarity.png"
STATS_OUTPUT = RESULT_ROOT / "figure_7_return_complementarity.json"

BLUE = "#356FA5"
RED = "#B9514E"
GRAY = "#9AA3AC"
DARK_GRAY = "#69737D"
LIGHT_GRID = "#E2E7EB"


def class_iou(matrix: np.ndarray, labels: np.ndarray, class_id: int) -> float:
    index = int(np.flatnonzero(labels == class_id)[0])
    tp = float(matrix[index, index])
    union = float(matrix[index].sum() + matrix[:, index].sum() - tp)
    return tp / union if union else float("nan")


def metric(matrix: np.ndarray, labels: np.ndarray, class_id: int | None) -> float:
    return mean_iou(matrix) if class_id is None else class_iou(matrix, labels, class_id)


def paired_gain(y, full, visual, mask, blocks, labels, class_id, seed):
    unique = np.unique(blocks[mask])
    matrices = []
    for prediction in (full, visual):
        matrices.append(np.stack([
            confusion_matrix(y[mask & (blocks == block)],
                             prediction[mask & (blocks == block)], labels=labels)
            for block in unique
        ]))

    def estimate(index):
        return (metric(matrices[0][index].sum(0), labels, class_id)
                - metric(matrices[1][index].sum(0), labels, class_id))

    point = estimate(np.arange(unique.size))
    rng = np.random.default_rng(seed)
    draws = np.asarray([estimate(rng.integers(unique.size, size=unique.size))
                        for _ in range(BOOTSTRAP)])
    draws = draws[np.isfinite(draws)]
    return {"gain": float(point),
            "ci95": [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))],
            "points": int(mask.sum()), "blocks": int(unique.size),
            "bootstrap_draws": draws}


def difference_in_differences(y, full, visual, first, second, blocks, labels,
                              class_id, seed):
    unique = np.unique(blocks[first | second])
    matrices = []
    for mask in (first, second):
        for prediction in (full, visual):
            matrices.append(np.stack([
                confusion_matrix(y[mask & (blocks == block)],
                                 prediction[mask & (blocks == block)], labels=labels)
                for block in unique
            ]))

    def estimate(index):
        values = [metric(matrix[index].sum(0), labels, class_id) for matrix in matrices]
        return (values[0] - values[1]) - (values[2] - values[3])

    point = estimate(np.arange(unique.size))
    rng = np.random.default_rng(seed)
    draws = np.asarray([estimate(rng.integers(unique.size, size=unique.size))
                        for _ in range(BOOTSTRAP)])
    draws = draws[np.isfinite(draws)]
    return {"difference": float(point),
            "ci95": [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))],
            "blocks": int(unique.size)}


def panel_a_statistics():
    models = predictions()
    reference = models["full"]
    y = reference["target"]
    region = reference["region"]
    labels = np.unique(y)
    with (AUDIT_ROOT / "audit_manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        source = {row["stem"]: row["source_block"] for row in csv.DictReader(stream)}
    blocks = np.asarray([source[stem] for stem in reference["stem"]])
    overstory, vegetation_overstory, _, _, _, _ = geometry_strata(reference)
    inv, vis = region == 0, region == 2

    # Spatial windows include all predicted/target classes, so per-class IoU
    # includes both false negatives and false positives.
    all_points = np.ones(y.size, dtype=bool)
    groups = [
        (r"$P_{inv}$", inv, None),
        (r"$P_{vis}$", vis, None),
        ("Under-canopy ground", inv & vegetation_overstory, 2),
        ("Exposed ground", vis & ~overstory, 2),
        ("Vegetation interior", inv, 3),
        ("Visible vegetation", vis, 3),
        ("Building facade", all_points, 26),
        ("Building roof", all_points, 6),
    ]
    full = models["full"]["prediction"]
    visual = models["visual"]["prediction"]
    result = []
    for index, (name, mask, class_id) in enumerate(groups):
        value = paired_gain(y, full, visual, mask, blocks, labels, class_id,
                            SEED + 700 + index)
        value.update(name=name, metric="mIoU" if class_id is None else "class IoU")
        value.pop("bootstrap_draws")
        result.append(value)
    return result


def panel_b_statistics():
    data = json.loads((RESULT_ROOT / "p3_surgery.json").read_text(encoding="utf-8"))["results"]
    output = {}
    for metric, label in (("linear_mIoU", "Frozen linear probe"),
                          ("knn_mIoU", "Cosine kNN")):
        output[label] = {}
        for group, display in (("true", "Correct pair"), ("random", "Random-72D"),
                               ("shuffle", r"Shuffled-$z^R$")):
            values = np.asarray([value[metric]["all"] for key, value in data.items()
                                 if key.startswith(group + "_whole")])
            output[label][display] = {
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                "n": int(values.size),
            }
    return output


def draw(panel_a, panel_b):
    plt.rcParams.update({
        "font.family": "Times New Roman", "font.size": 13.5,
        "axes.labelsize": 14.5, "axes.titlesize": 15.5,
        "xtick.labelsize": 12.5, "ytick.labelsize": 12.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "mathtext.fontset": "stix", "savefig.dpi": 400,
    })
    fig = plt.figure(figsize=(15.8, 7.0))
    grid = GridSpec(1, 2, figure=fig, width_ratios=(1.34, 1.0), wspace=.18)

    ax = fig.add_subplot(grid[0, 0])
    y = np.asarray([0, 1, 3, 4, 6, 7, 9, 10], dtype=float)
    for row, value in zip(y, panel_a):
        point = 100 * value["gain"]
        lo, hi = 100 * np.asarray(value["ci95"])
        ax.errorbar(point, row, xerr=[[point - lo], [hi - point]], fmt="o",
                    color=BLUE, ecolor=BLUE, markersize=7.8,
                    elinewidth=1.8, capsize=3.5, capthick=1.5, zorder=3)
        ax.text(hi + .55, row, f"{point:+.1f}", ha="left", va="center",
                fontsize=11.3, color="#30363B")
    ax.axvline(0, color="#AAB2B9", linestyle="--", linewidth=1.1, zorder=1)
    ax.set_yticks(y, [v["name"] for v in panel_a])
    ax.invert_yaxis()
    ax.set_ylim(10.75, -1.10)
    ax.set_xlabel("IoU gain (percentage points)")
    for label in ax.get_yticklabels():
        label.set_rotation(0)
        label.set_horizontalalignment("right")
        label.set_verticalalignment("center")
    ax.grid(axis="x", color=LIGHT_GRID, linewidth=.8)
    ax.set_axisbelow(True)
    for separator in (2, 5, 8):
        ax.axhline(separator, color=LIGHT_GRID, linewidth=.9)
    xmin, xmax = ax.get_xlim()
    group_labels = (("Regional visibility", -.58), ("Ground structures", 2.42),
                    ("Vegetation structures", 5.42), ("Building structures", 8.42))
    for text, position in group_labels:
        ax.text(xmin, position, text, ha="left", va="center", color="#737D86",
                fontsize=10.7, fontweight="bold")
    ax.set_xlim(xmin, xmax + 2.3)

    right = grid[0, 1].subgridspec(1, 2, wspace=.24)
    readouts = list(panel_b)
    controls = ["Correct pair", "Random-72D", r"Shuffled-$z^R$"]
    colors_ = [BLUE, GRAY, DARK_GRAY]
    axes_b = []
    for panel_index, readout in enumerate(readouts):
        ax_b = fig.add_subplot(right[0, panel_index])
        axes_b.append(ax_b)
        means = [panel_b[readout][control]["mean"] for control in controls]
        errors = [panel_b[readout][control]["sd"] for control in controls]
        x = np.arange(3)
        bars = ax_b.bar(x, means, .72, color=colors_, zorder=3)
        ax_b.errorbar(x[1:], means[1:], yerr=errors[1:], fmt="none",
                      ecolor="#3D444A", elinewidth=1.2, capsize=3, zorder=4)
        for bar, value in zip(bars, means):
            ax_b.text(bar.get_x() + bar.get_width() / 2, value + .008,
                      f"{value:.3f}", ha="center", va="bottom", fontsize=10.8)
        ax_b.set_xticks(x, ["Correct\npair", "Random\n72D", "Shuffled\n$z^R$"])
        ax_b.set_ylabel("mIoU")
        ax_b.set_ylim(0, .60)
        ax_b.set_title(readout, pad=10, fontsize=13.5)
        ax_b.grid(axis="y", color=LIGHT_GRID, linewidth=.8)
        ax_b.set_axisbelow(True)
    fig.subplots_adjust(left=.105, right=.988, top=.94, bottom=.225)
    fig.canvas.draw()
    left_position = ax.get_position()
    right_left = axes_b[0].get_position()
    right_right = axes_b[-1].get_position()
    title_y = .070
    fig.text((left_position.x0 + left_position.x1) / 2, title_y,
             "(a) Conditional IoU gains across structures",
             ha="center", va="bottom", fontfamily="Times New Roman",
             fontsize=15.5, fontweight="bold")
    fig.text((right_left.x0 + right_right.x1) / 2, title_y,
             "(b) Return-pairing control experiments",
             ha="center", va="bottom", fontfamily="Times New Roman",
             fontsize=15.5, fontweight="bold")
    fig.savefig(OUTPUT, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    panel_a = panel_a_statistics()
    panel_b = panel_b_statistics()
    serializable = {"panel_a": panel_a, "panel_b": panel_b,
                    "metric_note": "Class IoU is computed inside spatial windows containing all target/predicted classes."}
    STATS_OUTPUT.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    draw(panel_a, panel_b)
    print(OUTPUT)


if __name__ == "__main__":
    main()
