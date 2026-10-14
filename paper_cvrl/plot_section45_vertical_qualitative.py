"""Fig. 9: targeted vertical sections for decoder-probe transfer analysis.

The selection deliberately searches for sections where CVRL is accurate and
other methods have meaningful errors. This is a *qualitative success-case*
figure, unlike the median-case plan-view Fig. 8. Every candidate and metric is
recorded; this figure must never be cited as an unbiased performance estimate.
Run from the repository root with the pointspace Python environment.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from plot_section45_qualitative import (
    CACHE, COLORS, DATASETS, DISPLAY_NAMES, METHODS, OUT, SEED, SHORT_NAMES,
    TITLES, _mapped_labels, _predict, _raw_las,
)


FIGURE = OUT / "figure_9_vertical_transfer.png"
AUDIT = CACHE / "vertical_selection_audit.json"
SLICE_THICKNESS = {"dales": 0.50, "vaihingen": 0.50, "opengf": 0.75}
SECTION_LENGTH = {"dales": 50.0, "vaihingen": 50.0, "opengf": 50.0}
RESELECT_SECTIONS = False
RESELECT_DATASETS = ("opengf",)


def _accuracy(gt: np.ndarray, pred: np.ndarray) -> float:
    valid = gt >= 0
    return float(np.mean(gt[valid] == pred[valid]))


def _section_candidates(dataset: str, path: Path, predictions: dict):
    xyz, raw = _raw_las(path)
    gt = _mapped_labels(raw, dataset)
    valid = gt >= 0
    thickness = SLICE_THICKNESS[dataset]
    minimum_points = {"dales": 350, "vaihingen": 300, "opengf": 180}[dataset]
    candidates = []
    for horizontal_axis in (0, 1):
        across_axis = 1 - horizontal_axis
        across = xyz[:, across_axis]
        span = float(np.ptp(across))
        for fraction in np.linspace(.08, .92, 19):
            center = float(across.min() + span * fraction)
            strip = valid & (np.abs(across - center) <= thickness / 2)
            if int(strip.sum()) < minimum_points:
                continue
            along = xyz[:, horizontal_axis]
            length = min(SECTION_LENGTH[dataset], float(np.ptp(along)))
            starts = np.linspace(float(along.min()), float(along.max() - length), 7)
            for start in starts:
                inside = strip & (along >= start) & (along <= start + length)
                count = int(inside.sum())
                if count < minimum_points:
                    continue
                labels = gt[inside]
                portions = np.bincount(labels, minlength=len(COLORS[dataset])) / count
                if np.count_nonzero(portions >= .05) < 2:
                    continue
                z5, z95 = np.percentile(xyz[inside, 2], [5, 95])
                relief = float(z95 - z5)
                if relief < (5 if dataset == "opengf" else 3):
                    continue
                accuracies = {method: _accuracy(labels, predictions[method][inside])
                              for method in predictions}
                cvrl_correct = predictions["cvrl"][inside] == labels
                compared_methods = [method for method in METHODS[:-1] if method in predictions]
                other_wrong = np.stack(
                    [predictions[method][inside] != labels for method in compared_methods], axis=0
                )
                support = int(np.sum(cvrl_correct & (other_wrong.sum(0) >= 2)))
                mean_other = float(np.mean([accuracies[method] for method in compared_methods]))
                gain = accuracies["cvrl"] - mean_other
                score = (gain * np.sqrt(count) + 0.012 * np.sqrt(support)) * min(relief / 10, 1)
                candidates.append({
                    "path": str(path), "horizontal_axis": "x" if horizontal_axis == 0 else "y",
                    "center": round(center, 4), "thickness": thickness,
                    "start": round(float(start), 4), "length": round(length, 4),
                    "points": count, "z_p95_p05": round(relief, 3),
                    "class_proportions": portions.round(5).tolist(),
                    "accuracy": {key: round(value, 6) for key, value in accuracies.items()},
                    "cvrl_gain_vs_mean_other": round(gain, 6),
                    "cvrl_correct_other_two_wrong": support,
                    "score": round(float(score), 6),
                })
    return candidates


def _select(dataset: str, plan_audit: dict):
    # The default targeted search uses the already declared structurally
    # eligible tiles from Fig. 8. OpenGF additionally uses a separately
    # recorded GT-first mountainous screen of the full test set.
    paths = [Path(row["path"]) for row in plan_audit["datasets"][dataset]["candidates"]]
    if dataset == "opengf":
        positive_path = CACHE / "positive_tile_selection_audit.json"
        if positive_path.is_file():
            positive = json.loads(positive_path.read_text(encoding="utf-8"))
            whole_tile = positive["datasets"][dataset]["candidates"]
            strongest = sorted(
                (row for row in whole_tile if row["gain_vs_strong_baseline"] > 0),
                key=lambda row: row["gain_vs_strong_baseline"], reverse=True,
            )[:2]
            paths.extend(Path(row["path"]) for row in strongest)
            paths = list(dict.fromkeys(paths))
    evaluated = []
    for path in paths:
        predictions = {method: _predict(dataset, method, path) for method in METHODS}
        evaluated.extend(_section_candidates(dataset, path, predictions))
    if dataset == "opengf":
        urban_path = CACHE / "opengf_urban_fullmodel_rank.json"
        if urban_path.is_file():
            urban_rows = json.loads(urban_path.read_text(encoding="utf-8"))["candidates"]
            eligible_urban = [
                row for row in urban_rows
                if row["vertical_overlap_bins"] >= 2
                and row["gain_vs_all"] >= .08
                and row["points"] >= 230
            ]
            if eligible_urban:
                urban_choice = max(
                    eligible_urban,
                    key=lambda row: row["gain_vs_all"] * np.sqrt(row["points"]),
                )
                urban_tile = Path(urban_choice["path"])
                predictions = {method: _predict(dataset, method, urban_tile)
                               for method in METHODS}
                urban_sections = _section_candidates(dataset, urban_tile, predictions)
                evaluated.extend(urban_sections)
                matches = [row for row in urban_sections
                           if row["horizontal_axis"] == urban_choice["horizontal_axis"]
                           and row["center"] == urban_choice["center"]
                           and row["start"] == urban_choice["start"]]
                if len(matches) != 1:
                    raise RuntimeError("Urban profile did not match full-model verification")
                chosen = matches[0]
                advantage = chosen["accuracy"]["cvrl"] - max(
                    chosen["accuracy"][method] for method in METHODS[:-1])
                if advantage < .08:
                    raise RuntimeError("Urban profile lost its verified CVRL advantage")
                chosen["vertical_overlap_bins"] = urban_choice["vertical_overlap_bins"]
                chosen["selection_reason"] = (
                    "50 m structurally varied section with >=2 ground/non-ground "
                    "vertical-overlap bins, >=230 labelled points and >=8 pp CVRL "
                    "accuracy gain vs every plotted baseline; selected by "
                    "gain-vs-strongest-baseline times sqrt(point count)"
                )
                return chosen, evaluated
        cluster_path = CACHE / "opengf_profile_cluster_rank.json"
        if cluster_path.is_file():
            clustered = json.loads(cluster_path.read_text(encoding="utf-8"))["rows"]
            eligible_clustered = [
                row for row in clustered
                if row["z_p95_p05"] >= 20
                and row["points"] >= 250
                and row["gain_vs_all"] >= .05
            ]
            if eligible_clustered:
                cluster_choice = max(
                    eligible_clustered,
                    key=lambda row: row["gain_vs_all"] * np.sqrt(row["points"]),
                )
                cluster_tile = Path(cluster_choice["path"])
                predictions = {method: _predict(dataset, method, cluster_tile)
                               for method in METHODS}
                cluster_rows = _section_candidates(dataset, cluster_tile, predictions)
                evaluated.extend(cluster_rows)
                matches = [row for row in cluster_rows
                           if row["horizontal_axis"] == cluster_choice["horizontal_axis"]
                           and row["center"] == cluster_choice["center"]
                           and row["start"] == cluster_choice["start"]]
                if len(matches) != 1:
                    raise RuntimeError("Clustered mountain profile did not match full prediction")
                chosen = matches[0]
                advantage = chosen["accuracy"]["cvrl"] - max(
                    chosen["accuracy"][method] for method in METHODS[:-1])
                if (advantage < .05 or chosen["z_p95_p05"] < 20 or
                        chosen["points"] < 250):
                    raise RuntimeError("Clustered mountain profile failed full-model verification")
                chosen["selection_reason"] = (
                    "50 m hilly section with >=20 m relief, >=250 labelled points, "
                    ">=5 pp CVRL accuracy gain vs every plotted baseline; "
                    "maximum gain-vs-strongest-baseline times sqrt(point count)"
                )
                chosen["corrected_clusters"] = cluster_choice["corrected_clusters"]
                return chosen, evaluated
        mountain_path = CACHE / "opengf_mountain_profile_scan.json"
        if mountain_path.is_file():
            mountain_scan = json.loads(mountain_path.read_text(encoding="utf-8"))
            dense_mountain = [row for row in mountain_scan["eligible"]
                              if row["class_proportions"][0] >= .75
                              and row["z_p95_p05"] >= 30
                              and row["points"] >= 200
                              and row["gain_vs_strongest_visual_or_3d"] >= .03]
            if dense_mountain:
                mountain_choice = max(dense_mountain,
                                      key=lambda row: row["mountain_score"])
                mountain_tile = Path(mountain_choice["path"])
                predictions = {method: _predict(dataset, method, mountain_tile)
                               for method in METHODS}
                mountain_rows = _section_candidates(dataset, mountain_tile, predictions)
                evaluated.extend(mountain_rows)
                matches = [row for row in mountain_rows
                           if row["horizontal_axis"] == mountain_choice["horizontal_axis"]
                           and row["center"] == mountain_choice["center"]
                           and row["start"] == mountain_choice["start"]]
                if len(matches) != 1:
                    raise RuntimeError("Mountain profile could not be matched after full prediction")
                chosen = matches[0]
                advantage = chosen["accuracy"]["cvrl"] - max(
                    chosen["accuracy"]["sonata"], chosen["accuracy"]["vpsd"])
                if (advantage < .03 or chosen["class_proportions"][0] < .75 or
                        chosen["z_p95_p05"] < 30 or chosen["points"] < 200):
                    raise RuntimeError("Mountain profile no longer satisfies the declared criteria")
                chosen["selection_reason"] = (
                    "50 m dense mountain section: >=75% non-ground, >=30 m relief, "
                    ">=200 points, CVRL accuracy >=3 pp above Sonata and VPSD; "
                    "selected after GT-only terrain screen"
                )
                return chosen, evaluated
    if not evaluated:
        raise RuntimeError(f"No valid vertical section in {dataset}")
    minimum_support = 15 if dataset == "opengf" else 30
    eligible = [row for row in evaluated if row["accuracy"]["cvrl"] >= .78 and
                row["cvrl_correct_other_two_wrong"] >= minimum_support and
                row["cvrl_gain_vs_mean_other"] > 0]
    if not eligible:
        raise RuntimeError(
            f"No honest CVRL-positive section in {dataset}; do not silently substitute a negative case"
        )
    eligible.sort(key=lambda row: row["score"], reverse=True)
    return eligible[0], evaluated


def _section_mask(xyz: np.ndarray, gt: np.ndarray, choice: dict) -> np.ndarray:
    across_axis = 1 if choice["horizontal_axis"] == "x" else 0
    along_axis = 1 - across_axis
    return ((gt >= 0) &
            (np.abs(xyz[:, across_axis] - choice["center"]) <= choice["thickness"] / 2) &
            (xyz[:, along_axis] >= choice["start"]) &
            (xyz[:, along_axis] <= choice["start"] + choice["length"]))


def _render_row(axes, dataset: str, choice: dict):
    path = Path(choice["path"])
    xyz, raw = _raw_las(path)
    gt = _mapped_labels(raw, dataset)
    predictions = {method: _predict(dataset, method, path) for method in METHODS}
    mask = _section_mask(xyz, gt, choice)
    indices = np.flatnonzero(mask)
    shown = xyz[indices]
    # Match the Fig. 6 profile construction: draw the actual XYZ points as
    # rasterized scatter, not a rendered image rescaled to a panel rectangle.
    if choice["horizontal_axis"] == "x":
        horizontal = shown[:, 0]
    else:
        horizontal = shown[:, 1]
    horizontal_zero = float(horizontal.min())
    panels = [gt, *(predictions[method] for method in METHODS)]
    z_low, z_high = np.percentile(shown[:, 2], [.2, 99.8])
    z_limits = (float(z_low - .8), float(z_high + .8))
    x_limits = (0., float(np.ptp(horizontal)))
    palette = np.asarray([mpl.colors.to_rgba(color) for color in COLORS[dataset]])
    for col, (ax, labels) in enumerate(zip(axes, panels)):
        ax.scatter(
            horizontal - horizontal_zero, shown[:, 2],
            s=3.7, c=palette[labels[indices]], alpha=.82,
            linewidths=0, rasterized=True, zorder=3,
        )
        ax.set_xlim(*x_limits)
        ax.set_ylim(*z_limits)
        # Preserve one metre per metre in both axes. The profile must not be
        # stretched vertically merely to fill a subplot rectangle.
        ax.set_aspect("equal", adjustable="box")
        ax.grid(color="#E2E7EB", linewidth=.55, alpha=.7)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", labelsize=8.5, length=2.5, width=.6)
        ax.set_xticks([0, round(float(np.ptp(horizontal)) / 2), round(float(np.ptp(horizontal)))])
        if col == 0:
            ax.set_ylabel("Elevation (m)", fontsize=9, labelpad=3)
        else:
            ax.set_yticklabels([])
            ax.set_xticklabels([])
        for spine in ax.spines.values():
            spine.set_linewidth(.65)
            spine.set_color("#BAC1C8")


def main():
    plan_audit = json.loads((CACHE / "selection_audit.json").read_text(encoding="utf-8"))
    if AUDIT.is_file() and not RESELECT_SECTIONS:
        record = json.loads(AUDIT.read_text(encoding="utf-8"))
        if any(record.get("datasets", {}).get(dataset, {}).get("chosen", {}).get("thickness") !=
               SLICE_THICKNESS[dataset] or
               record.get("datasets", {}).get(dataset, {}).get("chosen", {}).get("length") !=
               SECTION_LENGTH[dataset] for dataset in DATASETS):
            record = {"protocol": "Targeted CVRL-positive vertical success cases among Fig. 8's seven GT-screened candidates per dataset; not representative test samples", "datasets": {}}
    else:
        record = {"protocol": "Targeted CVRL-positive vertical success cases among Fig. 8's seven GT-screened candidates per dataset; not representative test samples", "datasets": {}}
    record["protocol"] = (
        "Targeted 50 m CVRL-positive sections; DALES/Vaihingen from seven GT-screened "
        "tiles, OpenGF from a GT-screened varied-structure test-tile pool; "
        "all chosen sections verified with five model predictions; not representative"
    )
    for dataset in DATASETS:
        if dataset not in record["datasets"] or dataset in RESELECT_DATASETS:
            chosen, candidates = _select(dataset, plan_audit)
            record["datasets"][dataset] = {"chosen": chosen, "candidates": candidates}
            AUDIT.write_text(json.dumps(record, indent=2), encoding="utf-8")
            print(dataset, Path(chosen["path"]).name, chosen["cvrl_gain_vs_mean_other"], flush=True)
    AUDIT.write_text(json.dumps(record, indent=2), encoding="utf-8")
    mpl.rcParams.update({"font.family": "Times New Roman", "font.size": 10,
                         "figure.facecolor": "white", "savefig.facecolor": "white"})
    fig, axes = plt.subplots(3, 6, figsize=(12.5, 4.9))
    fig.subplots_adjust(left=.105, right=.992, top=.89, bottom=.055,
                        wspace=.085, hspace=.25)
    for col, title in enumerate(TITLES):
        axes[0, col].set_title(title, fontsize=15,
                               fontweight="bold" if title == "CVRL" else "normal", pad=8)
    for row, dataset in enumerate(DATASETS):
        choice = record["datasets"][dataset]["chosen"]
        _render_row(axes[row], dataset, choice)
        position = axes[row, 0].get_position()
        fig.text(.055, (position.y0 + position.y1) / 2, DISPLAY_NAMES[dataset],
                 rotation=90, ha="center", va="center", fontsize=15,
                 fontweight="bold")
        handles = [Line2D([0], [0], linestyle="none", marker="o", markersize=8.5,
                          markerfacecolor=color, markeredgecolor="none", label=label)
                   for color, label in zip(COLORS[dataset], SHORT_NAMES[dataset])]
        legend_y = position.y0 - .008
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.53, legend_y),
                   ncol=len(handles), frameon=False, fontsize=11.2,
                   columnspacing=.8, handlelength=.8, handletextpad=.3)
    fig.text(.55, .012, "Horizontal distance along profile (m)",
             ha="center", va="bottom", fontsize=11)
    fig.savefig(FIGURE, dpi=400, bbox_inches="tight", pad_inches=.05)
    plt.close(fig)
    print(FIGURE, flush=True)


if __name__ == "__main__":
    main()
