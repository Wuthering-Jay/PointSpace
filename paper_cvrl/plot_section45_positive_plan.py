"""Additional Fig. 8 using deliberately CVRL-positive test regions.

The original median-case Fig. 8 is never overwritten. Selection uses whole-
tile mIoU for DALES/OpenGF. Vaihingen alone uses a separately audited,
 CVRL-positive 50 m crop. OpenGF's
original seven GT-screened candidates contain no positive full tile, so its
candidate pool is expanded using the same GT-only structural score.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from plot_section45_qualitative import (
    CACHE, COLORS, DATASETS, DISPLAY_NAMES, METHODS, OUT, SCAN_LIMIT, SEED,
    SHORT_NAMES, TITLES, _configuration, _mapped_labels, _miou,
    _predict, _raw_las, _render_row, _scene_score,
)


FIGURE = OUT / "figure_8_qualitative_transfer_cvrl_positive.png"
AUDIT = CACHE / "positive_tile_selection_audit.json"
RESELECT_TILES = False
RESELECT_DATASETS = ()
OPEN_GF_MAX_EVALUATIONS = 60
VAIHINGEN_MAX_EVALUATIONS = 45
LOCAL_SIZE_M = 50.0


def _choose_local_crop(dataset: str, path: Path,
                       predictions: dict[str, np.ndarray],
                       size_m: float = LOCAL_SIZE_M,
                       minimum_coverage: float = .97,
                       minimum_span: float | None = None) -> dict:
    xyz, raw = _raw_las(path)
    gt = _mapped_labels(raw, dataset)
    xy_min = xyz[:, :2].min(axis=0)
    xy_max = xyz[:, :2].max(axis=0)
    starts = [np.linspace(float(xy_min[axis]),
                          float(max(xy_min[axis], xy_max[axis] - size_m)), 5)
              for axis in (0, 1)]
    candidates = []
    for x0 in np.unique(starts[0]):
        for y0 in np.unique(starts[1]):
            bounds = (float(x0), float(x0 + size_m),
                      float(y0), float(y0 + size_m))
            inside = ((xyz[:, 0] >= bounds[0]) & (xyz[:, 0] <= bounds[1]) &
                      (xyz[:, 1] >= bounds[2]) & (xyz[:, 1] <= bounds[3]) &
                      (gt >= 0))
            count = int(inside.sum())
            if count < 2500:
                continue
            portions = np.bincount(gt[inside], minlength=len(COLORS[dataset])) / count
            if dataset == "vaihingen" and np.count_nonzero(portions >= .02) < 3:
                continue
            shown_xy = xyz[inside, :2]
            footprint = np.ptp(shown_xy, axis=0)
            grid = np.clip(np.floor((shown_xy - [x0, y0]) / size_m * 10)
                           .astype(np.int16), 0, 9)
            coverage = len(np.unique(grid[:, 0] * 10 + grid[:, 1])) / 100
            if coverage < minimum_coverage or np.min(footprint) < (
                size_m - 1 if minimum_span is None else minimum_span
            ):
                continue
            scores = {method: _miou(gt[inside], predictions[method][inside],
                                    len(COLORS[dataset]))
                      for method in ("sonata", "vpsd", "cvrl")}
            gain = scores["cvrl"] - max(scores["sonata"], scores["vpsd"])
            candidates.append({"bounds_xy_m": [round(v, 4) for v in bounds],
                               "path": str(path),
                               "points": count,
                               "coverage_10x10": round(coverage, 3),
                               "point_span_xy_m": footprint.round(3).tolist(),
                               "miou": {key: round(value, 6) for key, value in scores.items()},
                               "gain_vs_strong_baseline": round(float(gain), 6)})
    if not candidates:
        raise RuntimeError(f"No 50 m crop with sufficient class support: {path}")
    return max(candidates, key=lambda row: row["gain_vs_strong_baseline"])


def _choose_vaihingen_50m_crop() -> dict:
    path = Path(_configuration("vaihingen", "cvrl").test_data_dir) / \
        "Vaihingen3D_EVAL_WITH_REF_0176.las"
    predictions = {method: _predict("vaihingen", method, path)
                   for method in ("sonata", "vpsd", "cvrl")}
    return _choose_local_crop("vaihingen", path, predictions)


def _evaluate(dataset: str, path: Path, structural: dict | None = None) -> dict:
    xyz, raw = _raw_las(path)
    gt = _mapped_labels(raw, dataset)
    xy = xyz[:, :2]
    lo = xy.min(0)
    span = np.maximum(np.ptp(xy, axis=0), 1e-4)
    aspect = float(span.min() / span.max())
    ij = np.clip(((xy - lo) / span * 10).astype(np.int16), 0, 9)
    coverage = len(np.unique(ij[:, 0] * 10 + ij[:, 1])) / 100
    scores = {}
    for method in ("sonata", "vpsd", "cvrl"):
        scores[method] = round(_miou(gt, _predict(dataset, method, path),
                                     len(COLORS[dataset])), 6)
    gain = scores["cvrl"] - max(scores["sonata"], scores["vpsd"])
    return {"path": str(path), "miou": scores,
            "gain_vs_strong_baseline": round(gain, 6),
            "coverage_10x10": round(coverage, 3),
            "footprint_aspect": round(aspect, 3),
            "structural": structural or {}}


def _expanded_candidates(dataset: str, existing: set[str]) -> list[dict]:
    folder = Path(_configuration(dataset, "cvrl").test_data_dir)
    paths = sorted(folder.glob("*.las")) + sorted(folder.glob("*.laz"))
    rng = np.random.default_rng(SEED + DATASETS.index(dataset))
    chosen = rng.choice(len(paths), size=min(SCAN_LIMIT[dataset], len(paths)),
                        replace=False)
    scored = []
    for index in chosen:
        path = paths[int(index)]
        if str(path) in existing:
            continue
        xyz, raw = _raw_las(path)
        score, stats = _scene_score(dataset, xyz, raw)
        if score > 0:
            scored.append({"path": str(path), "score": round(score, 6), **stats})
    scored.sort(key=lambda row: row["score"], reverse=True)
    return scored


def _select(dataset: str, original_audit: dict) -> dict:
    candidates = original_audit["datasets"][dataset]["candidates"]
    if dataset == "dales":
        evaluated = [_evaluate(dataset, Path(row["path"]), row) for row in candidates]
    else:
        # Include the original seven so the expanded search is auditable.
        evaluated = [_evaluate(dataset, Path(row["path"]), row) for row in candidates]
        existing = {row["path"] for row in candidates}
        pool = _expanded_candidates(dataset, existing)
        limit = OPEN_GF_MAX_EVALUATIONS if dataset == "opengf" else VAIHINGEN_MAX_EVALUATIONS
        for index, candidate in enumerate(pool[:limit], start=1):
            row = _evaluate(dataset, Path(candidate["path"]), candidate)
            evaluated.append(row)
            print(f"{dataset} candidate {index}: {Path(row['path']).name} "
                  f"gain={row['gain_vs_strong_baseline']:+.4f} "
                  f"coverage={row['coverage_10x10']:.2f}", flush=True)
    eligible = [row for row in evaluated if row["gain_vs_strong_baseline"] > 0 and
                (dataset != "vaihingen" or
                 (row["coverage_10x10"] >= .75 and row["footprint_aspect"] >= .80))]
    if not eligible:
        raise RuntimeError(f"No full-tile CVRL mIoU win in {dataset}; no positive figure generated")
    chosen = max(eligible, key=lambda row: row["gain_vs_strong_baseline"])
    return {"chosen": chosen, "candidates": evaluated,
            "selection": "maximum positive whole-tile mIoU gain vs max(Sonata,VPSD)"}


def main():
    original = json.loads((CACHE / "selection_audit.json").read_text(encoding="utf-8"))
    if AUDIT.is_file() and not RESELECT_TILES:
        audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    else:
        audit = {"protocol": "Targeted positive complete tiles; decoder probe; whole-tile mIoU; not representative", "datasets": {}}
    for dataset in DATASETS:
        if dataset not in audit["datasets"] or dataset in RESELECT_DATASETS:
            audit["datasets"][dataset] = _select(dataset, original)
            AUDIT.write_text(json.dumps(audit, indent=2), encoding="utf-8")
        chosen = audit["datasets"][dataset]["chosen"]
        print(dataset, Path(chosen["path"]).name,
              chosen["gain_vs_strong_baseline"], flush=True)

    mpl.rcParams.update({"font.family": "Times New Roman", "font.size": 11,
                         "figure.facecolor": "white", "savefig.facecolor": "white"})
    fig, axes = plt.subplots(3, 6, figsize=(12.5, 7.5))
    fig.subplots_adjust(left=.065, right=.992, top=.925, bottom=.065,
                        wspace=.025, hspace=.24)
    for col, title in enumerate(TITLES):
        axes[0, col].set_title(title, fontsize=15,
                               fontweight="bold" if title == "CVRL" else "normal", pad=9)
    for row, dataset in enumerate(DATASETS):
        crop = None
        if dataset == "vaihingen":
            crop = _choose_vaihingen_50m_crop()
            path = Path(crop["path"])
            audit["datasets"][dataset].pop("crop_100m", None)
            audit["datasets"][dataset]["crop_50m"] = crop
            print(dataset, "50 m crop", crop, flush=True)
        else:
            path = Path(audit["datasets"][dataset]["chosen"]["path"])
            audit["datasets"][dataset].pop("crop_50m", None)
        predictions = {method: _predict(dataset, method, path) for method in METHODS}
        _render_row(axes[row], dataset, path, predictions,
                    crop_bounds=None if crop is None else tuple(crop["bounds_xy_m"]),
                    point_scale=1.22 if dataset == "vaihingen" else None)
        axes[row, 0].text(-.095, .50, DISPLAY_NAMES[dataset],
                          transform=axes[row, 0].transAxes,
                          ha="right", va="center", rotation=90,
                          fontsize=16, fontweight="bold")
        handles = [Line2D([0], [0], linestyle="none", marker="o", markersize=8.5,
                          markerfacecolor=color, markeredgecolor="none", label=label)
                   for color, label in zip(COLORS[dataset], SHORT_NAMES[dataset])]
        legend_y = axes[row, 0].get_position().y0 + .006
        fig.legend(handles=handles, loc="upper center", frameon=False,
                   bbox_to_anchor=(.53, legend_y), ncol=len(handles),
                   fontsize=11.5, columnspacing=.9, handlelength=.8, handletextpad=.3)
    fig.savefig(FIGURE, dpi=400, bbox_inches="tight", pad_inches=.06)
    plt.close(fig)
    AUDIT.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(FIGURE, flush=True)


if __name__ == "__main__":
    main()
