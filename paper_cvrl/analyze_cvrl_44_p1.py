"""Section 4.4 P1 frozen-readout comparison of intrinsic SSL targets.

The script is intentionally IDE-configured.  It evaluates every *complete*
feature directory on exactly the Section 4.3 source-block split and sampled
rows.  Incomplete exports are reported and skipped rather than silently using
different tiles.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_cvrl.analyze_cvrl_43 import (
    _block_split, _build_rows, _load_feature_tile, _manifest, _metric_block,
    _normalize, _stratified_subsample, _tile_slices,
)
from paper_cvrl.section43_metrics import mean_iou

AUDIT_ROOT = Path(r"G:\data\swiss\joint_tiles_100\Swiss-Audit")
OUTPUT = ROOT / "paper_cvrl/results/section_44/p1_intrinsic_ssl.json"
PREDICTION_DIR = AUDIT_ROOT / "analysis/predictions"
FEATURES = {
    "scratch": "features/scratch_litept72",
    "normal": "features/geometry_normal",
    "local_shape": "features/geometry_local_shape",
    "role_only": "features/rrcl_role_only",
    "context_only": "features/rrcl_context_only",
    "full_rrcl": "features/rrcl_independent",
}
MAX_TRAIN = 100_000
MAX_TEST = 100_000
MAX_KNN_TRAIN = 5_000
MAX_KNN_TEST = 5_000
KNN_K = 20
BOOTSTRAP = 2_000
SEED = 42
LOG = logging.getLogger("cvrl_44_p1")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _complete_sources(rows):
    stems = np.unique(rows["stem"])
    complete, status = {}, {}
    for name, relative in FEATURES.items():
        folder = AUDIT_ROOT / relative
        present = {path.stem for path in folder.glob("*.safetensors")} if folder.exists() else set()
        missing = sorted(set(stems) - present)
        status[name] = {
            "folder": str(folder), "expected_tiles": int(stems.size),
            "present_tiles": int(len(set(stems) & present)),
            "complete": not missing, "missing_tiles": missing,
        }
        if not missing:
            complete[name] = relative
        else:
            LOG.warning("Skipping %-14s: %d/%d feature tiles", name,
                        status[name]["present_tiles"], stems.size)
    return complete, status


def _load_selected(rows, folder, selected):
    selected = np.sort(selected)
    output = []
    for _, tile_slice in _tile_slices(rows["stem"]).items():
        chosen = selected[(selected >= tile_slice.start) & (selected < tile_slice.stop)]
        if chosen.size:
            tile = _load_feature_tile(
                AUDIT_ROOT, {"kind": "point", "folder": folder}, rows, tile_slice)
            output.append(tile[chosen - tile_slice.start])
    return np.concatenate(output).astype(np.float32, copy=False)


def _paired_bootstrap(y, first, second, blocks, labels, region):
    result = {}
    rng = np.random.default_rng(SEED + 4401)
    for name, mask in {
        "all": np.ones(y.size, bool), "P_inv": region == 0,
        "P_amb": region == 1, "P_vis": region == 2,
    }.items():
        unique = np.unique(blocks[mask])
        matrices = []
        for prediction in (first, second):
            matrices.append(np.stack([
                confusion_matrix(y[mask & (blocks == block)],
                                 prediction[mask & (blocks == block)], labels=labels)
                for block in unique
            ]))
        point = mean_iou(matrices[0].sum(0)) - mean_iou(matrices[1].sum(0))
        draws = np.empty(BOOTSTRAP)
        for index in range(BOOTSTRAP):
            sample = rng.integers(unique.size, size=unique.size)
            draws[index] = (mean_iou(matrices[0][sample].sum(0)) -
                            mean_iou(matrices[1][sample].sum(0)))
        result[name] = {
            "difference_mIoU": float(point),
            "ci95": [float(np.quantile(draws, q)) for q in (0.025, 0.975)],
            "blocks": int(unique.size), "points": int(mask.sum()),
        }
    return result


def run():
    rows = _build_rows(AUDIT_ROOT, _manifest(AUDIT_ROOT / "audit_manifest.csv"))
    sources, export_status = _complete_sources(rows)
    train_block, test_block = _block_split(rows["source_block"], 0.20)
    valid = rows["label"] >= 0
    train = _stratified_subsample(np.flatnonzero(train_block & valid), rows["label"],
                                  rows["region"], MAX_TRAIN, SEED + 100)
    test = _stratified_subsample(np.flatnonzero(test_block & valid), rows["label"],
                                 rows["region"], MAX_TEST, SEED + 1100)
    knn_train = _stratified_subsample(train, rows["label"], rows["region"],
                                     MAX_KNN_TRAIN, SEED + 2100)
    knn_test = _stratified_subsample(test, rows["label"], rows["region"],
                                    MAX_KNN_TEST, SEED + 3100)
    selected = np.union1d(np.union1d(train, test), np.union1d(knn_train, knn_test))
    positions = {"train": np.searchsorted(selected, train),
                 "test": np.searchsorted(selected, test),
                 "knn_train": np.searchsorted(selected, knn_train),
                 "knn_test": np.searchsorted(selected, knn_test)}
    labels = np.unique(rows["label"][train])
    predictions = {}
    report = {
        "schema": "cvrl_44_p1_v1", "primary_metric": "mIoU",
        "protocol": {
            "split_unit": "original source block", "normalization": "row L2",
            "linear": "class-balanced multinomial logistic regression",
            "knn": f"cosine distance-weighted k={KNN_K}",
            "train_rows": int(train.size), "test_rows": int(test.size),
            "knn_train_rows": int(knn_train.size), "knn_test_rows": int(knn_test.size),
            "heldout_blocks": int(np.unique(rows["source_block"][test]).size),
            "bootstrap": BOOTSTRAP,
        },
        "export_status": export_status, "models": {}, "paired_vs_full_rrcl": {},
    }
    for name, folder in sources.items():
        LOG.info("Evaluating %s", name)
        feature = _normalize(_load_selected(rows, folder, selected))
        linear = LogisticRegression(C=1.0, class_weight="balanced", max_iter=500,
                                    solver="lbfgs", random_state=SEED)
        linear.fit(feature[positions["train"]], rows["label"][train])
        linear_prediction = linear.predict(feature[positions["test"]])
        knn = KNeighborsClassifier(n_neighbors=KNN_K, metric="cosine",
                                   weights="distance", n_jobs=-1)
        knn.fit(feature[positions["knn_train"]], rows["label"][knn_train])
        knn_prediction = knn.predict(feature[positions["knn_test"]])
        predictions[name] = linear_prediction
        report["models"][name] = {
            "feature_dim": int(feature.shape[1]),
            "linear": _metric_block(rows["label"][test], linear_prediction,
                                     rows["region"][test], labels),
            "knn": _metric_block(rows["label"][knn_test], knn_prediction,
                                  rows["region"][knn_test], labels),
        }
        PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            PREDICTION_DIR / f"p1_{name}_linear.npz", audit_row_index=test,
            stem=np.asarray(rows["stem"][test], dtype=str),
            point_index=rows["point_index"][test], target=rows["label"][test],
            prediction=linear_prediction, region=rows["region"][test],
            height_bin=rows["height_bin"][test],
            environment=np.asarray(rows["environment"][test], dtype=str),
        )
        block = report["models"][name]["linear"]
        LOG.info("%-14s mIoU %.4f OA %.4f macro-F1 %.4f", name,
                 block["all"]["mIoU"], block["all"]["OA"],
                 block["all"]["macro_F1"])
    if "full_rrcl" in predictions:
        for name, prediction in predictions.items():
            if name != "full_rrcl":
                report["paired_vs_full_rrcl"][name] = _paired_bootstrap(
                    rows["label"][test], predictions["full_rrcl"], prediction,
                    rows["source_block"][test], labels, rows["region"][test])
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    LOG.info("Saved %s", OUTPUT)
    return report


if __name__ == "__main__":
    run()
