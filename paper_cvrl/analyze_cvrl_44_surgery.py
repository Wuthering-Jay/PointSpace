"""Section 4.4 P3: frozen, paired representation surgery on Swiss-Audit.

Edit the module constants and run in the pointspace environment from the IDE.
No model parameters or source feature files are modified.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.neighbors import KNeighborsClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from paper_cvrl.analyze_cvrl_43 import (
    _block_split, _build_rows, _load_feature_tile, _manifest, _normalize,
    _stratified_subsample, _tile_slices,
)
from paper_cvrl.section43_metrics import mean_iou

AUDIT_ROOT = Path(r"G:\data\swiss\joint_tiles_100\Swiss-Audit")
OUTPUT = Path(__file__).resolve().parent / "results/section_44/p3_surgery.json"
SEEDS = (11, 23, 37, 53, 71)
BOOTSTRAP = 1000
LOG = logging.getLogger("cvrl_44_p3")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def selected_features(root, rows, selected):
    """Load only the fixed probe rows, in global audit-row order."""
    selected = np.sort(selected)
    outputs = {key: [] for key in ("visual", "return", "full")}
    specs = {"visual": "features/cvrl_visual_pca128",
             "return": "features/rrcl", "full": "features/cvrl_full"}
    for stem, block in _tile_slices(rows["stem"]).items():
        chosen = selected[(selected >= block.start) & (selected < block.stop)]
        if not chosen.size:
            continue
        local = chosen - block.start
        for key, folder in specs.items():
            tile = _load_feature_tile(root, {"folder": folder}, rows, block)
            outputs[key].append(tile[local])
    return {key: np.concatenate(value) for key, value in outputs.items()}


def score(y, prediction, region, labels):
    return {
        name: mean_iou(confusion_matrix(y[mask], prediction[mask], labels=labels))
        for name, mask in {
            "all": np.ones(y.size, dtype=bool),
            "P_inv": region == 0, "P_amb": region == 1, "P_vis": region == 2,
        }.items()
    }


def metric_bundle(y, prediction, region, labels):
    result = {}
    masks = {"all": np.ones(y.size, dtype=bool), "P_inv": region == 0,
             "P_amb": region == 1, "P_vis": region == 2}
    for name, mask in masks.items():
        matrix = confusion_matrix(y[mask], prediction[mask], labels=labels)
        tp = np.diag(matrix).astype(np.float64)
        union = matrix.sum(0) + matrix.sum(1) - tp
        iou = np.divide(tp, union, out=np.zeros_like(tp), where=union > 0)
        precision, recall, f1, support = precision_recall_fscore_support(
            y[mask], prediction[mask], labels=labels, zero_division=0)
        result[name] = {
            "OA": float(tp.sum() / max(matrix.sum(), 1)), "mIoU": mean_iou(matrix),
            "macro_F1": float(f1.mean()),
            "per_class": {str(label): {"IoU": float(iou[i]),
                "precision": float(precision[i]), "recall": float(recall[i]),
                "F1": float(f1[i]), "support": int(support[i])}
                for i, label in enumerate(labels)},
            "confusion_matrix": matrix.tolist(),
        }
    return result


def block_ci(y, pred_a, pred_b, blocks, labels, region, seed):
    rng = np.random.default_rng(seed)
    unique = np.unique(blocks)
    per_block = []
    for pred in (pred_a, pred_b):
        per_block.append(np.stack([
            confusion_matrix(y[(blocks == block) & region],
                             pred[(blocks == block) & region], labels=labels)
            for block in unique
        ]))
    differences = np.empty(BOOTSTRAP, dtype=np.float64)
    for i in range(BOOTSTRAP):
        sample = rng.integers(unique.size, size=unique.size)
        differences[i] = (mean_iou(per_block[0][sample].sum(axis=0))
                          - mean_iou(per_block[1][sample].sum(axis=0)))
    return [float(np.quantile(differences, q)) for q in (0.025, 0.975)]


def run():
    rows = _build_rows(AUDIT_ROOT, _manifest(AUDIT_ROOT / "audit_manifest.csv"))
    with np.load(AUDIT_ROOT / "analysis/predictions/regional_cvrl_full200_linear.npz") as file:
        reference = {key: file[key] for key in file.files}
    train_block, test_block = _block_split(rows["source_block"], 0.20)
    train = _stratified_subsample(np.flatnonzero(train_block & (rows["label"] >= 0)),
                                  rows["label"], rows["region"], 100000, 142)
    test = _stratified_subsample(np.flatnonzero(test_block & (rows["label"] >= 0)),
                                 rows["label"], rows["region"], 100000, 1142)
    if not np.array_equal(test, reference["audit_row_index"]):
        raise ValueError("P3 test rows differ from Section 4.3")
    all_index = np.union1d(train, test)
    feature = selected_features(AUDIT_ROOT, rows, all_index)
    lookup = np.searchsorted(all_index, train), np.searchsorted(all_index, test)
    train_pos, test_pos = lookup
    v, r, full = (feature[key] for key in ("visual", "return", "full"))
    concat = np.concatenate((v, r), axis=1)
    component_error = float(np.max(np.abs(full - concat)))
    LOG.info("Full component maximum absolute error: %.8f", component_error)
    if not np.allclose(full, concat, atol=5e-3, rtol=5e-3):
        raise ValueError("CVRL full is not the stored visual/return concatenation")
    labels = np.unique(rows["label"][train])
    ytrain, ytest = rows["label"][train], rows["label"][test]
    region = rows["region"][test]
    blocks = rows["source_block"][test]
    # Marginal random control: shuffle each feature dimension independently.
    # This exactly preserves each zR marginal and never uses labels or geometry.
    report = {"schema": "cvrl_p3_v1", "train_rows": int(train.size),
              "test_rows": int(test.size), "heldout_blocks": int(np.unique(blocks).size),
              "component_error_max": component_error, "seeds": list(SEEDS),
              "results": {}}

    def evaluate(name, x, seed, branch_norm=False):
        if branch_norm:
            x = np.concatenate((_normalize(x[:, :128]), _normalize(x[:, 128:])), axis=1)
        x = _normalize(x).astype(np.float32)
        model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=500,
                                   solver="lbfgs", random_state=42)
        model.fit(x[train_pos], ytrain)
        prediction = model.predict(x[test_pos])
        # Exactly the 4.3 kNN sampling protocol, projected into selected rows.
        knn_train = _stratified_subsample(train, rows["label"], rows["region"], 5000, 2142)
        knn_test = _stratified_subsample(test, rows["label"], rows["region"], 5000, 3142)
        kp_train = np.searchsorted(all_index, knn_train)
        kp_test = np.searchsorted(all_index, knn_test)
        knn = KNeighborsClassifier(n_neighbors=20, metric="cosine", weights="distance", n_jobs=-1)
        knn.fit(x[kp_train], rows["label"][knn_train])
        kp = knn.predict(x[kp_test])
        values = {"linear_mIoU": score(ytest, prediction, region, labels),
                  "linear_metrics": metric_bundle(ytest, prediction, region, labels),
                  "knn_mIoU": score(rows["label"][knn_test], kp, rows["region"][knn_test], labels),
                  "knn_metrics": metric_bundle(rows["label"][knn_test], kp,
                                               rows["region"][knn_test], labels),
                  "full_minus_control_ci95": {
                      key: block_ci(ytest, reference["prediction"], prediction, blocks, labels,
                                    np.ones(test.size, bool) if key == "all" else region == dict(P_inv=0, P_amb=1, P_vis=2)[key], seed + 100)
                      for key in ("all", "P_inv", "P_vis")}}
        LOG.info("%s seed=%d linear All %.4f P_inv %.4f P_vis %.4f",
                 name, seed, *[values["linear_mIoU"][k] for k in ("all", "P_inv", "P_vis")])
        return values

    for mode in ("whole", "branch"):
        norm = mode == "branch"
        report["results"][f"true_{mode}"] = evaluate(f"true_{mode}", full, 42, norm)
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            randomized = np.empty_like(r)
            for j in range(r.shape[1]):
                randomized[:, j] = r[rng.permutation(r.shape[0]), j]
            report["results"][f"random_{mode}_{seed}"] = evaluate(
                f"random_{mode}", np.concatenate((v, randomized), axis=1), seed, norm)
            shuffled = np.empty_like(r)
            for split in (train_pos, test_pos):
                for block in np.unique(rows["source_block"][all_index[split]]):
                    ix = split[rows["source_block"][all_index[split]] == block]
                    shuffled[ix] = r[rng.permutation(ix)]
            report["results"][f"shuffle_{mode}_{seed}"] = evaluate(
                f"shuffle_{mode}", np.concatenate((v, shuffled), axis=1), seed, norm)
            OUTPUT.parent.mkdir(parents=True, exist_ok=True)
            OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    run()
