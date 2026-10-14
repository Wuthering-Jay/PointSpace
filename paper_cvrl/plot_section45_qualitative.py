"""Build Fig. 8 from *decoder-probe* test predictions, without hand-picking wins.

Run with the pointspace environment from the repository root::

    python paper_cvrl/plot_section45_qualitative.py

The first run evaluates a small, GT-selected candidate set. Predictions and
the selection audit are cached in ``results/section_45/qualitative_cache``.
Subsequent runs reuse the cache. Set ``REBUILD_PREDICTIONS`` below to refresh.
No LAS predictions are written to the dataset or experiment folders.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import laspy
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pointspace.datasets import build_dataset, collate_fn
from pointspace.models import build_model
from pointspace.utils.config import Config
from utils.pointcloud_render import (
    compute_pcv,
    modulate_rgba_with_pcv,
    render_point_cloud_image,
)


METHODS = ("scratch", "sonata", "d-ditr", "vpsd", "cvrl")
DATASETS = ("dales", "vaihingen", "opengf")
TITLES = ("Ground Truth", "Scratch", "Sonata", "D-DITR", "VPSD", "CVRL")
DISPLAY_NAMES = {"dales": "DALES", "vaihingen": "Vaihingen", "opengf": "OpenGF"}
COLORS = {
    # Keep shared physical classes consistent across datasets. DALES
    # building is matched to Vaihingen facade (not roof), because truck is
    # already red; pole stays charcoal to remain distinct from teal car.
    "dales": ("#B39B79", "#1F8B58", "#0AA4A7", "#B84A3A", "#E2BD2D", "#7F9250", "#3E4C58", "#4B72BE"),
    "vaihingen": ("#E2BD2D", "#75BB52", "#8C9097", "#0AA4A7", "#7F9250", "#CF5D65", "#4B72BE", "#B065A9", "#1F8B58"),
    "opengf": ("#2D9464", "#B39B79"),
}
SHORT_NAMES = {
    "dales": ("Ground", "Vegetation", "Car", "Truck", "Power line", "Fence", "Pole", "Building"),
    "vaihingen": ("Power line", "Low vegetation", "Impervious", "Car", "Fence/hedge", "Roof", "Facade", "Shrub", "Tree"),
    "opengf": ("Non-ground", "Ground"),
}
RAW_IDS = {"dales": tuple(range(1, 9)), "vaihingen": tuple(range(9)), "opengf": (1, 2)}
OUT = ROOT / "paper_cvrl" / "results" / "section_45"
CACHE = OUT / "qualitative_cache"
FIGURE = OUT / "figure_8_qualitative_transfer.png"
REBUILD_PREDICTIONS = False
RESELECT_TILES = False
SEED = 4518
CANDIDATES_PER_DATASET = 7
SCAN_LIMIT = {"dales": 180, "vaihingen": 120, "opengf": 180}
INFER_BATCH_FRAGMENTS = 4
RENDER_POINTS = 115_000
RENDER_WIDTH = 610


def _configuration(dataset: str, method: str) -> Config:
    path = ROOT / "configs" / "downstream" / dataset / f"semseg-{method}-litept-v1m4-{dataset}-dec.py"
    cfg = Config.fromfile(str(path))
    weight = ROOT / "exp" / "downstream" / dataset / f"{method}-dec" / "model" / "model_best.pth"
    if not weight.is_file():
        raise FileNotFoundError(weight)
    cfg.weight = str(weight)
    return cfg


def _raw_las(path: Path):
    las = laspy.read(path)
    xyz = np.column_stack((las.x, las.y, las.z)).astype(np.float32)
    raw = np.asarray(las.classification, dtype=np.int16)
    return xyz, raw


def _mapped_labels(raw: np.ndarray, dataset: str) -> np.ndarray:
    result = np.full(raw.shape, -1, dtype=np.int16)
    for idx, original in enumerate(RAW_IDS[dataset]):
        result[raw == original] = idx
    return result


def _edge_density(xyz: np.ndarray, label: np.ndarray, nclass: int, bins: int = 32) -> float:
    # GT-only, planimetric class-boundary density, not a model outcome.
    xy = xyz[:, :2]
    span = np.maximum(np.ptp(xy, axis=0), 1e-4)
    ij = np.clip(((xy - xy.min(0)) / span * bins).astype(np.int32), 0, bins - 1)
    valid = label >= 0
    counts = np.zeros((bins * bins, nclass), dtype=np.int32)
    np.add.at(counts, (ij[valid, 0] * bins + ij[valid, 1], label[valid]), 1)
    occupied = counts.sum(1).reshape(bins, bins) > 0
    majority = counts.argmax(1).reshape(bins, bins)
    horizontal = occupied[1:] & occupied[:-1] & (majority[1:] != majority[:-1])
    vertical = occupied[:, 1:] & occupied[:, :-1] & (majority[:, 1:] != majority[:, :-1])
    return float((horizontal.sum() + vertical.sum()) / max(occupied.sum(), 1))


def _scene_score(dataset: str, xyz: np.ndarray, raw: np.ndarray) -> tuple[float, dict]:
    label = _mapped_labels(raw, dataset)
    valid = label >= 0
    if valid.sum() < 5000:
        return -1.0, {}
    proportion = np.bincount(label[valid], minlength=len(COLORS[dataset])) / valid.sum()
    boundary = _edge_density(xyz, label, len(COLORS[dataset]))
    relief = float(np.percentile(xyz[:, 2], 95) - np.percentile(xyz[:, 2], 5))
    entropy = float(-(proportion[proportion > 0] * np.log(proportion[proportion > 0])).sum()
                    / np.log(len(proportion)))
    if dataset == "dales":
        required = min(proportion[0] / .10, proportion[1] / .08, proportion[7] / .05, 1.0)
        small = proportion[2:7].sum()
        score = 3 * required + min(small / .006, 1) + 2 * boundary + entropy
    elif dataset == "vaihingen":
        required = min(proportion[2] / .10, proportion[5] / .08,
                       (proportion[1] + proportion[8]) / .10, 1.0)
        small = proportion[3] + proportion[6]
        score = 3 * required + min(small / .008, 1) + 2 * boundary + entropy
    else:
        mix = min(proportion[0] / .30, proportion[1] / .20, 1.0)
        score = 3 * mix + 2 * boundary + min(relief / 20, 1) + entropy
    return float(score), {"proportions": proportion.round(5).tolist(),
                          "boundary_density": round(boundary, 5), "z_p95_p05": round(relief, 3)}


def _candidate_files(dataset: str) -> list[dict]:
    cfg = _configuration(dataset, "cvrl")
    paths = sorted(Path(cfg.test_data_dir).glob("*.las")) + sorted(Path(cfg.test_data_dir).glob("*.laz"))
    rng = np.random.default_rng(SEED + DATASETS.index(dataset))
    chosen = rng.choice(len(paths), size=min(SCAN_LIMIT[dataset], len(paths)), replace=False)
    scored = []
    for i in chosen:
        path = paths[int(i)]
        xyz, raw = _raw_las(path)
        score, stats = _scene_score(dataset, xyz, raw)
        if score > 0:
            scored.append({"path": str(path), "score": round(score, 6), **stats})
    scored.sort(key=lambda item: item["score"], reverse=True)
    # Evenly sample the structurally eligible upper half. This keeps several
    # plausible scenes without making the candidate pool all outliers.
    eligible = scored[:max(CANDIDATES_PER_DATASET, len(scored) // 2)]
    positions = np.linspace(0, len(eligible) - 1, CANDIDATES_PER_DATASET).round().astype(int)
    return [eligible[i] for i in positions]


def _predict(dataset: str, method: str, path: Path) -> np.ndarray:
    output_path = CACHE / dataset / f"{path.stem}__{method}.npz"
    if output_path.is_file() and not REBUILD_PREDICTIONS:
        with np.load(output_path) as archive:
            return archive["prediction"].copy()
    cfg = _configuration(dataset, method)
    ds_cfg = copy.deepcopy(cfg.data.test)
    ds_cfg["data_path"] = None
    ds_cfg["data_list"] = [str(path)]
    ds_cfg["remap_class"] = {raw: i for i, raw in enumerate(RAW_IDS[dataset])}
    sample = build_dataset(ds_cfg)[0]
    fragments = sample["fragment_list"]
    count = len(sample["segment"])
    checkpoint = torch.load(cfg.weight, map_location="cpu", weights_only=False)
    model = build_model(cfg.model).cuda().eval()
    state = {key.removeprefix("module."): value for key, value in checkpoint["state_dict"].items()}
    model.load_state_dict(state, strict=True)
    scores = torch.zeros((count, cfg.data.num_classes), device="cuda", dtype=torch.float32)
    amp_dtype = torch.bfloat16 if cfg.amp_dtype == "bfloat16" else torch.float16
    with torch.inference_mode():
        for start in range(0, len(fragments), INFER_BATCH_FRAGMENTS):
            batch = collate_fn(fragments[start:start + INFER_BATCH_FRAGMENTS])
            batch = {key: value.cuda(non_blocking=True) if isinstance(value, torch.Tensor) else value
                     for key, value in batch.items()}
            with torch.autocast("cuda", dtype=amp_dtype, enabled=bool(cfg.enable_amp)):
                logits = model(batch)["seg_logits"]
            prob = F.softmax(logits.float(), dim=-1)
            offset = 0
            for end in batch["offset"]:
                end = int(end)
                scores.index_add_(0, batch["index"][offset:end].long(), prob[offset:end])
                offset = end
    prediction = scores.argmax(1).cpu().numpy().astype(np.uint8)
    if "origin_segment" in sample:
        prediction = prediction[sample["inverse"]]
    if len(prediction) != count:
        raise AssertionError(f"Prediction length mismatch for {path.name}: {len(prediction)} != {count}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, prediction=prediction)
    del model, scores, checkpoint
    torch.cuda.empty_cache()
    return prediction


def _miou(gt: np.ndarray, prediction: np.ndarray, nclass: int) -> float:
    valid = gt >= 0
    hist = np.bincount(nclass * gt[valid].astype(np.int32) + prediction[valid],
                       minlength=nclass * nclass).reshape(nclass, nclass)
    tp = hist.diagonal()
    union = hist.sum(0) + hist.sum(1) - tp
    present = hist.sum(1) > 0
    return float(np.mean((tp / np.maximum(union, 1))[present]))


def _choose_tile(dataset: str, candidates: list[dict]) -> tuple[dict, list[dict]]:
    evaluated = []
    for candidate in candidates:
        path = Path(candidate["path"])
        _, raw = _raw_las(path)
        gt = _mapped_labels(raw, dataset)
        metrics = {}
        for method in ("sonata", "vpsd", "cvrl"):
            pred = _predict(dataset, method, path)
            metrics[method] = round(_miou(gt, pred, len(COLORS[dataset])), 6)
        gain = metrics["cvrl"] - max(metrics["sonata"], metrics["vpsd"])
        evaluated.append({**candidate, "miou": metrics, "gain_vs_strong_baseline": round(gain, 6)})
        print(f"{dataset} {path.name}: {metrics}, gain={gain:+.4f}", flush=True)
    evaluated.sort(key=lambda row: row["gain_vs_strong_baseline"])
    return evaluated[len(evaluated) // 2], evaluated


def _error_box(xyz: np.ndarray, gt: np.ndarray, prediction: np.ndarray):
    """Return one substantial method-specific error box, or ``None``.

    The multi-scale integral grid finds spatially coherent error mass; hard
    minimum counts/rates prevent a box around a handful of wrong points.
    """
    xy = xyz[:, :2]
    lo = xy.min(0)
    span = np.maximum(np.ptp(xy, axis=0), 1e-4)
    bins = 64
    valid = gt >= 0
    ij = np.clip(((xy - lo) / span * bins).astype(np.int32), 0, bins - 1)
    flat = ij[:, 0] * bins + ij[:, 1]
    counts = np.bincount(flat[valid], minlength=bins * bins).reshape(bins, bins)
    wrong = valid & (prediction != gt)
    errors = np.bincount(flat[wrong], minlength=bins * bins).reshape(bins, bins)
    count_integral = np.pad(counts.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    error_integral = np.pad(errors.cumsum(0).cumsum(1), ((1, 0), (1, 0)))

    def region_sum(integral, x0, y0, x1, y1):
        return int(integral[x1, y1] - integral[x0, y1] -
                   integral[x1, y0] + integral[x0, y0])

    best = (-np.inf, None)
    min_wrong = max(250, int(.002 * valid.sum()))
    for side in (8, 12, 16, 20):
        for x0 in range(0, bins - side + 1, 2):
            for y0 in range(0, bins - side + 1, 2):
                x1, y1 = x0 + side, y0 + side
                npoints = region_sum(count_integral, x0, y0, x1, y1)
                nwrong = region_sum(error_integral, x0, y0, x1, y1)
                if npoints < 800 or nwrong < min_wrong:
                    continue
                error_rate = nwrong / npoints
                if error_rate < .18:
                    continue
                score = error_rate * np.sqrt(nwrong)
                if score > best[0]:
                    box = (lo[0] + span[0] * x0 / bins,
                           lo[1] + span[1] * y0 / bins,
                           lo[0] + span[0] * x1 / bins,
                           lo[1] + span[1] * y1 / bins)
                    best = score, {"xy": [float(value) for value in box],
                                   "points": npoints, "errors": nwrong,
                                   "error_rate": round(error_rate, 6)}
    return best[1]


def _palette_rgba(classes: np.ndarray, dataset: str) -> np.ndarray:
    palette = np.array([mpl.colors.to_rgba(color) for color in COLORS[dataset]], dtype=np.float32)
    rgba = np.tile(np.array([.72, .72, .72, 1], dtype=np.float32), (len(classes), 1))
    valid = (classes >= 0) & (classes < len(palette))
    rgba[valid] = palette[classes[valid]]
    return rgba


def _render_row(ax_row, dataset: str, path: Path, predictions: dict[str, np.ndarray],
                crop_bounds: tuple[float, float, float, float] | None = None,
                point_scale: float | None = None):
    xyz, raw = _raw_las(path)
    gt = _mapped_labels(raw, dataset)
    if any(len(pred) != len(gt) for pred in predictions.values()):
        raise AssertionError(f"Panel prediction length mismatch for {path}")
    if crop_bounds is None:
        candidate_idx = np.arange(len(xyz))
    else:
        x0, x1, y0, y1 = crop_bounds
        candidate_idx = np.flatnonzero(
            (xyz[:, 0] >= x0) & (xyz[:, 0] <= x1) &
            (xyz[:, 1] >= y0) & (xyz[:, 1] <= y1)
        )
        if len(candidate_idx) == 0:
            raise ValueError(f"50 m crop contains no points: {path}")
    rng = np.random.default_rng(SEED + 100 + DATASETS.index(dataset))
    display_idx = np.sort(rng.choice(candidate_idx, min(RENDER_POINTS, len(candidate_idx)),
                                        replace=False))
    shown = xyz[display_idx]
    # PCV is geometry-only, computed once and reused across all six panels.
    pcv = compute_pcv(shown, number_of_directions=128, resolution=768, point_radius=0)
    panels = [gt, *(predictions[method] for method in METHODS)]
    # North-up, exact orthographic nadir: screen x/y are world x/y. In
    # particular, roof height cannot shear a ground/roof boundary in plan view.
    camera = (
        np.array([1., 0., 0.]), np.array([0., 1., 0.]),
        np.array([0., 0., 1.]), np.median(shown, axis=0),
    )
    for col, (ax, labels) in enumerate(zip(ax_row, panels)):
        rgba = _palette_rgba(labels[display_idx], dataset)
        shaded = modulate_rgba_with_pcv(rgba, pcv, normalize=False)
        image, camera, _, _ = render_point_cloud_image(
            shown[:, 0], shown[:, 1], shown[:, 2], shaded,
            camera=camera, width=RENDER_WIDTH, strength=3.0, style="pcv_edl",
            point_scale=(.94 if dataset == "dales" else .86)
            if point_scale is None else point_scale,
            fractional_splat=True, max_point_radius=6,
            square_frame=True,
        )
        ax.imshow(image, origin="lower", interpolation="nearest")
        ax.set_xlim(0, image.shape[1]); ax.set_ylim(0, image.shape[0]); ax.axis("off")


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    selected = {}
    audit_path = CACHE / "selection_audit.json"
    if audit_path.is_file() and not RESELECT_TILES:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    else:
        audit = {"protocol": "Decoder probe; GT-only structural candidates; median CVRL minus max(Sonata,VPSD) candidate gain; error-region diagnostics retained but not drawn", "seed": SEED, "datasets": {}}
    audit["protocol"] = ("Decoder probe; GT-only structural candidates; median CVRL minus "
                         "max(Sonata,VPSD) candidate gain; error-region diagnostics retained but not drawn")
    for dataset in DATASETS:
        if dataset in audit["datasets"]:
            chosen = audit["datasets"][dataset]["chosen"]
            evaluated = audit["datasets"][dataset]["candidates"]
        else:
            candidates = _candidate_files(dataset)
            chosen, evaluated = _choose_tile(dataset, candidates)
        path = Path(chosen["path"])
        predictions = {method: _predict(dataset, method, path) for method in METHODS}
        xyz, raw = _raw_las(path)
        gt = _mapped_labels(raw, dataset)
        boxes = {method: _error_box(xyz, gt, predictions[method]) for method in METHODS}
        selected[dataset] = (path, predictions, boxes)
        audit["datasets"][dataset] = {"chosen": chosen,
                                      "box_protocol": "per_method_multiscale_error_v1",
                                      "method_boxes": boxes,
                                      "candidates": evaluated,
                                      "checkpoints": {method: str(_configuration(dataset, method).weight) for method in METHODS}}
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    mpl.rcParams.update({"font.family": "Times New Roman", "font.size": 11,
                         "figure.facecolor": "white", "savefig.facecolor": "white"})
    # Size this as a true double-column figure; at ~7.2 in print width the
    # 15 pt headings become ~8.6 pt instead of shrinking below 6 pt.
    fig, axes = plt.subplots(3, 6, figsize=(12.5, 7.5))
    fig.subplots_adjust(left=.065, right=.992, top=.925, bottom=.065,
                        wspace=.025, hspace=.24)
    for col, title in enumerate(TITLES):
        axes[0, col].set_title(title, fontsize=15, fontweight="bold" if title == "CVRL" else "normal", pad=9)
    for row, dataset in enumerate(DATASETS):
        path, predictions, boxes = selected[dataset]
        _render_row(axes[row], dataset, path, predictions)
        axes[row, 0].text(-.095, .50, DISPLAY_NAMES[dataset], transform=axes[row, 0].transAxes,
                          ha="right", va="center", rotation=90, fontsize=16, fontweight="bold")
        handles = [Line2D([0], [0], linestyle="none", marker="o", markersize=8.5,
                          markerfacecolor=color, markeredgecolor="none", label=label)
                   for color, label in zip(COLORS[dataset], SHORT_NAMES[dataset])]
        legend_y = axes[row, 0].get_position().y0 + .006
        fig.legend(handles=handles, loc="upper center", frameon=False,
                   bbox_to_anchor=(.53, legend_y), ncol=len(handles),
                   fontsize=11.5, columnspacing=.9, handlelength=.8,
                   handletextpad=.3)
    fig.savefig(FIGURE, dpi=400, bbox_inches="tight", pad_inches=.06)
    plt.close(fig)
    print(f"Saved {FIGURE}", flush=True)


if __name__ == "__main__":
    main()
