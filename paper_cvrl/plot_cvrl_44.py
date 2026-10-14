"""Render concise Section 4.4 P1--P4 audit figures from saved JSON results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULT = Path(__file__).resolve().parent / "results/section_44"
BLUE = "#386FA4"
RED = "#B65252"
GRAY = "#788491"
GREEN = "#4C956C"
ORANGE = "#D08C60"


def _setup():
    plt.rcParams.update({
        "font.family": "Times New Roman", "font.size": 13,
        "axes.labelsize": 14, "axes.titlesize": 15,
        "xtick.labelsize": 12, "ytick.labelsize": 12,
        "axes.spines.top": False, "axes.spines.right": False,
        "savefig.dpi": 250,
    })


def plot_p1():
    data = json.loads((RESULT / "p1_intrinsic_ssl.json").read_text(encoding="utf-8"))["models"]
    methods = [("Scratch", "scratch"), ("Normal", "normal"), ("Local shape", "local_shape"),
               ("Role only", "role_only"), ("Context only", "context_only"),
               ("Full RRCL", "full_rrcl")]
    regions = (("All", "all"), ("$P_{inv}$", "P_inv"), ("$P_{vis}$", "P_vis"))
    colors = ("#B8BDC5", GRAY, GREEN, ORANGE, "#8C6BB1", BLUE)
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.3), gridspec_kw={"wspace": .25})
    width = .15
    for ax, readout, title in zip(axes, ("linear", "knn"),
                                  ("Frozen linear probe", "Cosine kNN")):
        x = np.arange(len(regions))
        for j, ((label, key), color) in enumerate(zip(methods, colors)):
            values = [data[key][readout][region]["mIoU"] for _, region in regions]
            ax.bar(x + (j - 2.5) * width, values, width, label=label, color=color)
        ax.set_xticks(x, [label for label, _ in regions])
        ax.set_ylim(0, .36)
        ax.set_ylabel("mIoU")
        ax.set_title(title, loc="left", pad=12)
        ax.grid(axis="y", color="#e8ebef", lw=.8)
        ax.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=6, fontsize=11,
               loc="upper center", bbox_to_anchor=(.5, 1.02))
    fig.savefig(RESULT / "figure_44_p1_intrinsic.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_p2():
    data = json.loads((RESULT / "p2_strata.json").read_text(encoding="utf-8"))["groups"]
    columns = [
        ("Class Recall", [
            ("Under-canopy ground", "under_canopy_ground"),
            ("Exposed ground", "exposed_ground"),
            ("Vegetation interior", "vegetation_interior"),
            ("Visible vegetation", "vegetation_visible"),
        ]),
        ("mIoU", [
            ("Multi-layer columns", "multilayer"),
            ("Single-layer columns", "single_layer"),
            ("All points", "all"),
            ("Invisible region", "P_inv"),
            ("Visible region", "P_vis"),
        ]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.3, 4.7), gridspec_kw={"wspace": .52})
    for ax, (title, items) in zip(axes, columns):
        for i, (label, key) in enumerate(items):
            row = data[key]["differences"]["full_minus_visual"]
            value = 100 * row["difference"]
            lo, hi = 100 * np.asarray(row["ci95"])
            ax.errorbar(value, i, xerr=[[value-lo], [hi-value]], fmt="o",
                        color=RED if key == "under_canopy_ground" else BLUE,
                        markersize=7, capsize=3, lw=1.7)
        ax.axvline(0, color=GRAY, lw=1, ls="--")
        ax.set_yticks(range(len(items)), [name for name, _ in items])
        ax.invert_yaxis()
        ax.set_title(title, loc="left", pad=12)
        ax.set_xlabel("CVRL full − visual (percentage points)")
        ax.grid(axis="x", color="#e8ebef", lw=.8)
        ax.set_axisbelow(True)
    fig.savefig(RESULT / "figure_44_p2_structure.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_p3():
    data = json.loads((RESULT / "p3_surgery.json").read_text(encoding="utf-8"))["results"]
    names = ("Correct pair", "Random 72D", "Shuffled zR")
    groups = ("true", "random", "shuffle")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), gridspec_kw={"wspace": .32})
    for ax, metric, title in zip(axes, ("linear_mIoU", "knn_mIoU"),
                                 ("Frozen linear probe", "Cosine kNN")):
        for i, group in enumerate(groups):
            values = np.array([v[metric]["all"] for k, v in data.items()
                               if k.startswith(group + "_whole")])
            average = float(values.mean())
            error = float(values.std(ddof=1)) if values.size > 1 else 0
            ax.bar(i, average, width=.63, color=BLUE if i == 0 else GRAY,
                   alpha=1 if i == 0 else .75)
            if error:
                ax.errorbar(i, average, yerr=error, fmt="none", ecolor="#333333", capsize=3)
            ax.text(i, average + error + .006, f"{average:.3f}", ha="center", va="bottom", fontsize=12)
        ax.set_xticks(range(3), names, rotation=15, ha="right")
        ax.set_ylim(0, .60 if metric == "linear_mIoU" else .48)
        ax.set_ylabel("mIoU")
        ax.set_title(title, loc="left", pad=12)
        ax.grid(axis="y", color="#e8ebef", lw=.8)
        ax.set_axisbelow(True)
    fig.savefig(RESULT / "figure_44_p3_surgery.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_p4():
    data = json.loads((RESULT / "p4_shortcut.json").read_text(encoding="utf-8"))["groups"]
    groups = [("All", "all"), ("$P_{inv}$", "P_inv"), ("$P_{vis}$", "P_vis"),
              ("Vegetation\ninterior", "vegetation_interior"),
              ("Under-canopy\nground", "under_canopy_ground"),
              ("Multi-layer", "multilayer")]
    y = np.arange(len(groups)); width = .36
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.0), gridspec_kw={"wspace": .16})
    mlp = [data[key]["role"]["mIoU"] for _, key in groups]
    prior = [data[key]["constant_train_prior_baseline"]["role"]["mIoU"] for _, key in groups]
    axes[0].barh(y - width / 2, mlp, width, color=BLUE, label="Geometry MLP")
    axes[0].barh(y + width / 2, prior, width, color=GRAY, alpha=.75, label="Train prior")
    axes[0].set_xlabel("Role mIoU"); axes[0].set_title("Return-role predictability", loc="left")
    mae = [data[key]["context"]["MAE"] for _, key in groups]
    base = [data[key]["constant_train_prior_baseline"]["context_MAE"] for _, key in groups]
    axes[1].barh(y - width / 2, mae, width, color=BLUE, label="Geometry MLP")
    axes[1].barh(y + width / 2, base, width, color=GRAY, alpha=.75, label="Train prior")
    axes[1].set_xlabel("Context MAE (lower is better)")
    axes[1].set_title("Return-context predictability", loc="left")
    for ax in axes:
        ax.set_yticks(y, [name.replace("\n", " ") for name, _ in groups])
        ax.invert_yaxis()
        ax.grid(axis="x", color="#e8ebef", lw=.8); ax.set_axisbelow(True)
    axes[1].tick_params(labelleft=False)
    axes[0].legend(frameon=False, fontsize=11)
    fig.savefig(RESULT / "figure_44_p4_shortcut.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    _setup()
    plot_p1()
    plot_p2()
    plot_p3()
    plot_p4()
