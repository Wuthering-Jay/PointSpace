"""Section 4.4 P2: prespecified geometric strata and paired frozen-readout audit.

Configuration is module-level for IDE execution. Source LAS classes are used
only for post-training stratification, never as a pretraining input.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path

import laspy
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_cvrl.section43_metrics import mean_iou

AUDIT_ROOT = Path(r"G:\data\swiss\joint_tiles_100\Swiss-Audit")
OUTPUT = Path(__file__).resolve().parent / "results/section_44/p2_strata.json"
CELL_M = 1.0
MIN_COLUMN_POINTS = 8
OVERSTORY_M = 3.0
MULTILAYER_SPAN_M = 3.0
BOOTSTRAP = 2000
SEED = 42
LOG = logging.getLogger("cvrl_44_p2")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

FILES = {
    "scratch": "p1_scratch_linear.npz",
    "visual": "regional_cvrl_visual_pca128_linear.npz",
    "full": "regional_cvrl_full200_linear.npz",
    "vpsd": "regional_vpsd_pca128_linear.npz",
    "rrcl": "regional_rrcl_independent72_linear.npz",
    "joint_return": "regional_cvrl_return72_linear.npz",
    "normal": "p1_normal_linear.npz",
    "local_shape": "p1_local_shape_linear.npz",
    "role_only": "p1_role_only_linear.npz",
    "context_only": "p1_context_only_linear.npz",
    "full_rrcl": "p1_full_rrcl_linear.npz",
}


def predictions():
    result = {}
    for key, name in FILES.items():
        path = AUDIT_ROOT / "analysis/predictions" / name
        if not path.exists():
            LOG.warning("Optional P1 prediction missing: %s", path.name)
            continue
        with np.load(path) as file:
            result[key] = {k: file[k] for k in file.files}
    reference = result["full"]
    for key, value in result.items():
        for field in ("audit_row_index", "stem", "point_index", "target", "region"):
            if not np.array_equal(reference[field], value[field]):
                raise ValueError(f"{key} mismatched fixed test rows: {field}")
    return result


def metric_bundle(y, prediction, mask, labels):
    matrix = confusion_matrix(y[mask], prediction[mask], labels=labels)
    tp = np.diag(matrix).astype(np.float64)
    union = matrix.sum(1) + matrix.sum(0) - tp
    iou = np.divide(tp, union, out=np.full_like(tp, np.nan), where=union > 0)
    precision, recall_value, f1, support = precision_recall_fscore_support(
        y[mask], prediction[mask], labels=labels, zero_division=0)
    return {
        "OA": float(tp.sum() / max(matrix.sum(), 1)),
        "mIoU": mean_iou(matrix),
        "macro_F1": float(np.mean(f1)),
        "per_class": {
            str(label): {"IoU": None if not np.isfinite(iou[i]) else float(iou[i]),
                         "precision": float(precision[i]), "recall": float(recall_value[i]),
                         "F1": float(f1[i]), "support": int(support[i])}
            for i, label in enumerate(labels)
        },
        "confusion_matrix": matrix.tolist(),
    }


def geometry_strata(reference):
    stem = reference["stem"]
    n = stem.size
    overstory = np.zeros(n, dtype=bool)
    vegetation_overstory = np.zeros(n, dtype=bool)
    multilayer = np.zeros(n, dtype=bool)
    density = np.zeros(n, dtype=np.int32)
    veg_build_boundary = np.zeros(n, dtype=bool)
    valid_return = np.zeros(n, dtype=bool)
    with (AUDIT_ROOT / "audit_manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        filenames = {row["stem"]: row["filename"] for row in csv.DictReader(stream)}
    for count, name in enumerate(np.unique(stem), 1):
        ix = np.flatnonzero(stem == name)
        cloud = laspy.read(AUDIT_ROOT / "pointcloud" / filenames[name])
        if int(np.max(reference["point_index"][ix])) >= len(cloud.points):
            raise ValueError(f"LAS index out of range: {name}")
        xy = np.column_stack((np.asarray(cloud.x), np.asarray(cloud.y)))
        z = np.asarray(cloud.z)
        cls = np.asarray(cloud.classification, dtype=np.int16)
        number = np.asarray(cloud.return_number)
        total = np.asarray(cloud.number_of_returns)
        # Local integer coordinates avoid precision loss in global map coordinates.
        cell = np.floor((xy - xy.min(axis=0)) / CELL_M).astype(np.int32)
        _, inv = np.unique(cell, axis=0, return_inverse=True)
        ncell = int(inv.max()) + 1
        high = np.full(ncell, -np.inf)
        high_vegetation = np.full(ncell, -np.inf)
        low = np.full(ncell, np.inf)
        counts = np.bincount(inv, minlength=ncell)
        np.maximum.at(high, inv, z)
        vegetation_points = cls == 3
        np.maximum.at(high_vegetation, inv[vegetation_points], z[vegetation_points])
        np.minimum.at(low, inv, z)
        # Label mixture is a downstream audit stratum, not a feature or target.
        bits = np.zeros(ncell, dtype=np.uint8)
        np.bitwise_or.at(bits, inv, (cls == 3).astype(np.uint8))
        np.bitwise_or.at(bits, inv, ((cls == 6).astype(np.uint8) << 1))
        point = reference["point_index"][ix]
        valid_return[ix] = (number[point] >= 1) & (total[point] >= 1) & (number[point] <= total[point])
        own = inv[point]
        density[ix] = counts[own]
        overstory[ix] = (high[own] - z[point] >= OVERSTORY_M) & (counts[own] >= MIN_COLUMN_POINTS)
        vegetation_overstory[ix] = ((high_vegetation[own] - z[point] >= OVERSTORY_M)
                                   & (counts[own] >= MIN_COLUMN_POINTS))
        multilayer[ix] = ((high[own] - low[own] >= MULTILAYER_SPAN_M)
                          & (counts[own] >= MIN_COLUMN_POINTS))
        veg_build_boundary[ix] = bits[own] == 3
        if count % 20 == 0 or count == len(np.unique(stem)):
            LOG.info("Geometry strata: %d/%d held-out tiles", count, len(np.unique(stem)))
    return overstory, vegetation_overstory, multilayer, density, veg_build_boundary, valid_return


def recall(y, prediction, mask, class_id):
    selected = mask & (y == class_id)
    if not selected.any():
        return None
    return float(np.mean(prediction[selected] == class_id))


def paired_difference(y, a, b, mask, blocks, class_id=None):
    if class_id is not None:
        mask = mask & (y == class_id)
    if not np.any(mask):
        return None
    unique = np.unique(blocks[mask])
    labels = np.unique(y)
    per_block = []
    for prediction in (a, b):
        per_block.append(np.stack([
            confusion_matrix(y[mask & (blocks == block)], prediction[mask & (blocks == block)], labels=labels)
            for block in unique
        ]))

    def metric(matrix):
        if class_id is not None:
            ix = int(np.flatnonzero(labels == class_id)[0])
            return float(matrix[ix, ix] / matrix[ix].sum()) if matrix[ix].sum() else np.nan
        return mean_iou(matrix)

    point = metric(per_block[0].sum(axis=0)) - metric(per_block[1].sum(axis=0))
    rng = np.random.default_rng(SEED)
    draws = np.empty(BOOTSTRAP)
    for i in range(BOOTSTRAP):
        sample = rng.integers(unique.size, size=unique.size)
        draws[i] = metric(per_block[0][sample].sum(axis=0)) - metric(per_block[1][sample].sum(axis=0))
    finite = draws[np.isfinite(draws)]
    return {"difference": float(point), "ci95": [float(np.quantile(finite, q)) for q in (0.025, 0.975)],
            "points": int(mask.sum()), "blocks": int(unique.size)}


def interaction(y, full, visual, first, second, blocks, class_id=None):
    """Paired block-bootstrap difference-in-differences of full minus visual."""
    labels = np.unique(y)
    unique = np.unique(blocks[first | second])
    matrices = []
    for mask in (first, second):
        if class_id is not None:
            mask = mask & (y == class_id)
        for pred in (full, visual):
            matrices.append(np.stack([
                confusion_matrix(y[mask & (blocks == block)], pred[mask & (blocks == block)], labels=labels)
                for block in unique
            ]))

    def metric(matrix):
        if class_id is None:
            return mean_iou(matrix)
        k = int(np.flatnonzero(labels == class_id)[0])
        return float(matrix[k, k] / matrix[k].sum()) if matrix[k].sum() else np.nan

    def estimate(indices):
        vals = [metric(m[indices].sum(axis=0)) for m in matrices]
        return (vals[0] - vals[1]) - (vals[2] - vals[3])

    point = estimate(np.arange(unique.size))
    rng = np.random.default_rng(SEED + 44)
    draws = np.array([estimate(rng.integers(unique.size, size=unique.size))
                      for _ in range(BOOTSTRAP)])
    finite = draws[np.isfinite(draws)]
    return {"difference_in_differences": float(point),
            "ci95": [float(np.quantile(finite, q)) for q in (0.025, 0.975)],
            "blocks": int(unique.size)}


def run():
    models = predictions()
    ref = models["full"]
    y, region, height = (ref[k] for k in ("target", "region", "height_bin"))
    stem = ref["stem"]
    labels = np.unique(y)
    with (AUDIT_ROOT / "audit_manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        mapping = {r["stem"]: r["source_block"] for r in csv.DictReader(stream)}
    blocks = np.asarray([mapping[s] for s in stem])
    overstory, vegetation_overstory, multilayer, density, boundary, valid_return = geometry_strata(ref)
    inv, amb, vis = (region == k for k in (0, 1, 2))
    # Height bins follow the pre-existing 4.3 audit sidecar. The discrete bin
    # semantics are reported from swiss_audit_visibility.py, not inferred here.
    groups = {
        "all": (np.ones(y.size, bool), None),
        "P_inv": (inv, None), "P_amb": (amb, None), "P_vis": (vis, None),
        "vegetation_interior": ((y == 3) & inv, 3),
        "vegetation_visible": ((y == 3) & vis, 3),
        "under_canopy_ground": ((y == 2) & inv & vegetation_overstory, 2),
        "exposed_ground": ((y == 2) & vis & ~overstory, 2),
        "under_canopy_ground_density16_64": ((y == 2) & inv & vegetation_overstory & (density >= 16) & (density <= 64), 2),
        "exposed_ground_density16_64": ((y == 2) & vis & ~overstory & (density >= 16) & (density <= 64), 2),
        "multilayer": (multilayer, None),
        "single_layer": (~multilayer & (density >= MIN_COLUMN_POINTS), None),
        "multilayer_density16_64": (multilayer & (density >= 16) & (density <= 64), None),
        "single_layer_density16_64": (~multilayer & (density >= 16) & (density <= 64), None),
        "veg_build_boundary": (boundary & np.isin(y, (3, 6)), None),
        "exposed_roof_ground": (vis & ~multilayer & np.isin(y, (2, 6)), None),
    }
    for level in np.unique(height):
        groups[f"vegetation_interior_heightbin_{level}"] = ((y == 3) & inv & (height == level), 3)
    output = {"schema": "cvrl_p2_v1", "geometry": {
        "cell_m": CELL_M, "min_column_points": MIN_COLUMN_POINTS,
        "overstory_m": OVERSTORY_M, "multilayer_span_m": MULTILAYER_SPAN_M,
        "boundary": "same 1 m XY cell contains vegetation and building-roof labels",
        "strata_use": "post-training analysis only"}, "groups": {}}
    for name, (mask, class_id) in groups.items():
        if not mask.any():
            continue
        entry = {"metric": "class_recall" if class_id is not None else "mIoU",
                 "points": int(mask.sum()), "blocks": int(np.unique(blocks[mask]).size),
                 "class_counts": {str(k): int(v) for k, v in zip(*np.unique(y[mask], return_counts=True))},
                 "return_unknown_rate": float(np.mean(~valid_return[mask])),
                 "median_cell_points": float(np.median(density[mask])),
                 "models": {}, "model_metrics": {}, "differences": {}}
        for key, value in models.items():
            prediction = value["prediction"]
            entry["models"][key] = (recall(y, prediction, mask, class_id) if class_id is not None
                                     else mean_iou(confusion_matrix(y[mask], prediction[mask], labels=np.unique(y))))
            entry["model_metrics"][key] = metric_bundle(y, prediction, mask, labels)
        for name_pair, a, b in (("full_minus_visual", "full", "visual"),
                                ("rrcl_minus_vpsd", "rrcl", "vpsd"),
                                ("full_minus_vpsd", "full", "vpsd")):
            entry["differences"][name_pair] = paired_difference(
                y, models[a]["prediction"], models[b]["prediction"], mask, blocks, class_id)
        for baseline in ("scratch", "normal", "local_shape", "role_only", "context_only"):
            if baseline in models and "full_rrcl" in models:
                entry["differences"][f"full_rrcl_minus_{baseline}"] = paired_difference(
                    y, models["full_rrcl"]["prediction"], models[baseline]["prediction"],
                    mask, blocks, class_id)
        output["groups"][name] = entry
        LOG.info("%-32s n=%6d blocks=%2d full-visual=%+.4f", name, entry["points"],
                 entry["blocks"], entry["differences"]["full_minus_visual"]["difference"])
    output["interactions"] = {}
    for label, a, b, class_id in (
        ("under_canopy_vs_exposed_ground", "under_canopy_ground", "exposed_ground", 2),
        ("under_canopy_vs_exposed_ground_density16_64", "under_canopy_ground_density16_64", "exposed_ground_density16_64", 2),
        ("vegetation_interior_vs_visible", "vegetation_interior", "vegetation_visible", 3),
        ("multilayer_vs_single_layer", "multilayer", "single_layer", None),
        ("multilayer_vs_single_layer_density16_64", "multilayer_density16_64", "single_layer_density16_64", None),
    ):
        output["interactions"][label] = interaction(
            y, models["full"]["prediction"], models["visual"]["prediction"],
            groups[a][0], groups[b][0], blocks, class_id)
        for baseline in ("scratch", "normal", "local_shape", "role_only", "context_only"):
            if baseline in models and "full_rrcl" in models:
                output["interactions"][f"full_rrcl_vs_{baseline}:{label}"] = interaction(
                    y, models["full_rrcl"]["prediction"], models[baseline]["prediction"],
                    groups[a][0], groups[b][0], blocks, class_id)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


if __name__ == "__main__":
    run()
