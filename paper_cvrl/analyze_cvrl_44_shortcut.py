"""Section 4.4 P4: can cheap local geometry predict RRCL targets?

This is an audit, not an SSL transfer baseline.  The MLP sees only intensity
and XYZ-derived local statistics/descriptors.  LAS return fields are read only
after feature construction to form role/context targets.  Module constants are
kept at the top for direct IDE execution, consistent with the other audits.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path

import laspy
import numpy as np
import torch
from scipy.spatial import cKDTree
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_cvrl.analyze_cvrl_43 import (
    _block_split, _build_rows, _manifest, _stratified_subsample,
)
from paper_cvrl.analyze_cvrl_44_strata import (
    AUDIT_ROOT, MIN_COLUMN_POINTS, geometry_strata,
)
from paper_cvrl.section43_metrics import mean_iou

GEOMETRY_ROOT = Path(r"G:\data\swiss\joint_tiles_100\pointcloud_geom_p1")
CACHE = AUDIT_ROOT / "analysis/section_44_p4_shortcut_targets.npz"
OUTPUT = ROOT / "paper_cvrl/results/section_44/p4_shortcut.json"
K = 48
MAX_TRAIN = 100_000
MAX_TEST = 100_000
HIDDEN = 64
EPOCHS = 80
PATIENCE = 10
BATCH_SIZE = 4096
LEARNING_RATE = 2e-3
SEED = 42
REBUILD_CACHE = False
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOG = logging.getLogger("cvrl_44_p4")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def return_role(number, total):
    role = np.full(number.shape, -1, dtype=np.int64)
    role[total == 1] = 0
    role[(total > 1) & (number == 1)] = 1
    role[(total > 1) & (number > 1) & (number < total)] = 2
    role[(total > 1) & (number == total)] = 3
    return role


def _descriptor(cloud, point, neighbor_coord, distance):
    z = np.asarray(cloud.z, dtype=np.float64)[point]
    nz = neighbor_coord[:, :, 2]
    radius = np.maximum(distance[:, -1], 1e-3)
    density = K / ((4.0 / 3.0) * np.pi * radius ** 3)
    names = set(cloud.point_format.dimension_names)
    geometry_names = (
        "geom_linearity", "geom_planarity", "geom_scattering",
        "geom_curvature", "geom_anisotropy", "geom_verticality",
    )
    if not set(geometry_names).issubset(names):
        raise KeyError("Offline geometry LAS lacks the six local descriptors")
    shape = np.column_stack([
        np.asarray(cloud[name], dtype=np.float32)[point] for name in geometry_names
    ])
    intensity = np.log1p(np.asarray(cloud.intensity, dtype=np.float32)[point])[:, None]
    height = np.column_stack((
        z - nz.min(1), nz.max(1) - z, nz.max(1) - nz.min(1),
        nz.std(1), z - np.median(nz, axis=1), np.log1p(density),
    )).astype(np.float32)
    # 13 cheap inputs: log-intensity, six local height/density statistics and
    # six XYZ-PCA descriptors. No q_geo, LAS class, or return field is used.
    return np.concatenate((intensity, height, shape), axis=1).astype(np.float32)


def build_cache(rows, selected, filenames):
    selected = np.sort(selected)
    feature, role_out, context_out, valid_out = [], [], [], []
    position_out = []
    unique_stem = np.unique(rows["stem"][selected])
    for count, stem in enumerate(unique_stem, 1):
        chosen = selected[rows["stem"][selected] == stem]
        cloud = laspy.read(GEOMETRY_ROOT / filenames[stem])
        point = rows["point_index"][chosen].astype(np.int64)
        coord = np.column_stack((cloud.x, cloud.y, cloud.z)).astype(np.float64)
        tree = cKDTree(coord)
        distance, neighbor = tree.query(coord[point], k=K + 1, workers=-1)
        # Remove the actual query index, rather than assuming it is column 0;
        # this remains correct for duplicated zero-distance coordinates.
        keep = neighbor != point[:, None]
        if np.any(keep.sum(1) < K):
            raise ValueError(f"Could not exclude K neighbours for {stem}")
        neighbor = neighbor[keep].reshape(point.size, K)
        distance = distance[keep].reshape(point.size, K)
        number = np.asarray(cloud.return_number, dtype=np.int16)
        total = np.asarray(cloud.number_of_returns, dtype=np.int16)
        role_all = return_role(number, total)
        neighbor_role = role_all[neighbor]
        valid_neighbor = neighbor_role >= 0
        histogram = np.stack([
            np.sum((neighbor_role == c) & valid_neighbor, axis=1) for c in range(4)
        ], axis=1).astype(np.float32)
        denominator = valid_neighbor.sum(1)
        histogram /= np.maximum(denominator[:, None], 1)
        role = role_all[point]
        valid = (role >= 0) & (denominator > 0)
        feature.append(_descriptor(cloud, point, coord[neighbor], distance))
        role_out.append(role)
        context_out.append(histogram)
        valid_out.append(valid)
        position_out.append(chosen)
        if count % 20 == 0 or count == unique_stem.size:
            LOG.info("P4 target cache: %d/%d tiles", count, unique_stem.size)
    position = np.concatenate(position_out)
    order = np.argsort(position)
    result = {
        "audit_row_index": position[order],
        "feature": np.concatenate(feature)[order],
        "role": np.concatenate(role_out)[order],
        "context": np.concatenate(context_out)[order],
        "valid": np.concatenate(valid_out)[order],
    }
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, **result)
    return result


class ShortcutMLP(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.trunk = torch.nn.Sequential(
            torch.nn.Linear(channels, HIDDEN), torch.nn.GELU(),
            torch.nn.Linear(HIDDEN, HIDDEN), torch.nn.GELU(),
        )
        self.role = torch.nn.Linear(HIDDEN, 4)
        self.context = torch.nn.Linear(HIDDEN, 4)

    def forward(self, x):
        x = self.trunk(x)
        return self.role(x), self.context(x)


def classification_metrics(target, prediction):
    labels = np.arange(4)
    matrix = confusion_matrix(target, prediction, labels=labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        target, prediction, labels=labels, zero_division=0)
    tp = np.diag(matrix).astype(np.float64)
    union = matrix.sum(0) + matrix.sum(1) - tp
    iou = np.divide(tp, union, out=np.zeros_like(tp), where=union > 0)
    return {
        "OA": float(tp.sum() / max(matrix.sum(), 1)),
        "mIoU": mean_iou(matrix), "macro_F1": float(f1.mean()),
        "per_role": {str(i): {"IoU": float(iou[i]), "precision": float(precision[i]),
                               "recall": float(recall[i]), "F1": float(f1[i]),
                               "support": int(support[i])} for i in labels},
        "confusion_matrix": matrix.tolist(),
    }


def target_metrics(role, role_prediction, target, probability, mask,
                   majority_role, mean_context):
    mask = mask.astype(bool)
    eps = 1e-8
    kl = np.sum(target[mask] * (np.log(target[mask] + eps) - np.log(probability[mask] + eps)), 1)
    entropy = -np.sum(target[mask] * np.log(target[mask] + eps), 1)
    baseline_role = np.full(mask.sum(), majority_role, dtype=np.int64)
    baseline_context = np.broadcast_to(mean_context, target[mask].shape)
    baseline_kl = np.sum(target[mask] * (
        np.log(target[mask] + eps) - np.log(baseline_context + eps)), 1)
    return {
        "points": int(mask.sum()),
        "role": classification_metrics(role[mask], role_prediction[mask]),
        "context": {
            "MAE": float(np.mean(np.abs(target[mask] - probability[mask]))),
            "MAE_per_role": np.mean(np.abs(target[mask] - probability[mask]), axis=0).tolist(),
            "KL_target_prediction": float(np.mean(kl)),
            "target_entropy": float(np.mean(entropy)),
        },
        "constant_train_prior_baseline": {
            "role": classification_metrics(role[mask], baseline_role),
            "context_MAE": float(np.mean(np.abs(target[mask] - baseline_context))),
            "context_KL": float(np.mean(baseline_kl)),
        },
    }


def run():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    rows = _build_rows(AUDIT_ROOT, _manifest(AUDIT_ROOT / "audit_manifest.csv"))
    train_block, test_block = _block_split(rows["source_block"], 0.20)
    train = _stratified_subsample(np.flatnonzero(train_block & (rows["label"] >= 0)),
                                  rows["label"], rows["region"], MAX_TRAIN, 142)
    test = _stratified_subsample(np.flatnonzero(test_block & (rows["label"] >= 0)),
                                 rows["label"], rows["region"], MAX_TEST, 1142)
    selected = np.union1d(train, test)
    with (AUDIT_ROOT / "audit_manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        filenames = {row["stem"]: row["filename"] for row in csv.DictReader(stream)}
    if CACHE.exists() and not REBUILD_CACHE:
        with np.load(CACHE) as saved:
            data = {key: saved[key] for key in saved.files}
        if not np.array_equal(data["audit_row_index"], selected):
            raise ValueError("Cached P4 rows differ from the frozen split")
    else:
        data = build_cache(rows, selected, filenames)
    train_pos = np.searchsorted(selected, train)
    test_pos = np.searchsorted(selected, test)
    valid_train = data["valid"][train_pos]
    valid_test = data["valid"][test_pos]
    # Validation is source-block-disjoint within the frozen training side.
    train_blocks = rows["source_block"][train]
    unique = np.unique(train_blocks)
    rng = np.random.default_rng(SEED)
    validation_blocks = rng.choice(unique, size=max(1, int(round(0.15 * unique.size))), replace=False)
    validation = valid_train & np.isin(train_blocks, validation_blocks)
    fitting = valid_train & ~np.isin(train_blocks, validation_blocks)
    mean = data["feature"][train_pos[fitting]].mean(0)
    std = data["feature"][train_pos[fitting]].std(0).clip(1e-5)
    x = ((data["feature"] - mean) / std).astype(np.float32)
    model = ShortcutMLP(x.shape[1]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    counts = np.bincount(data["role"][train_pos[fitting]], minlength=4).astype(np.float32)
    weight = np.sqrt(counts.sum() / np.maximum(counts, 1)); weight /= weight.mean()
    weight = torch.tensor(weight, device=DEVICE)
    fit_index = train_pos[fitting]
    validation_index = train_pos[validation]
    best, best_state, stale = np.inf, None, 0
    for epoch in range(EPOCHS):
        model.train(); rng.shuffle(fit_index)
        for start in range(0, fit_index.size, BATCH_SIZE):
            index = fit_index[start:start + BATCH_SIZE]
            xb = torch.from_numpy(x[index]).to(DEVICE)
            yr = torch.from_numpy(data["role"][index]).to(DEVICE)
            yc = torch.from_numpy(data["context"][index]).to(DEVICE)
            role_logits, context_logits = model(xb)
            loss_role = torch.nn.functional.cross_entropy(role_logits, yr, weight=weight)
            loss_context = -(yc * torch.nn.functional.log_softmax(context_logits, -1)).sum(-1).mean()
            loss = loss_role + loss_context
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            iv = validation_index
            rv, cv = model(torch.from_numpy(x[iv]).to(DEVICE))
            validation_loss = (torch.nn.functional.cross_entropy(
                rv, torch.from_numpy(data["role"][iv]).to(DEVICE), weight=weight)
                - (torch.from_numpy(data["context"][iv]).to(DEVICE)
                   * torch.nn.functional.log_softmax(cv, -1)).sum(-1).mean()).item()
        LOG.info("P4 epoch %d validation loss %.5f", epoch + 1, validation_loss)
        if validation_loss < best - 1e-5:
            best, stale = validation_loss, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        role_logits, context_logits = model(torch.from_numpy(x[test_pos]).to(DEVICE))
        role_prediction = role_logits.argmax(1).cpu().numpy()
        probability = context_logits.softmax(-1).cpu().numpy()
    role = data["role"][test_pos]
    context = data["context"][test_pos]
    valid = valid_test
    region = rows["region"][test]
    # Reuse exactly the already-audited P2 masks.
    reference_path = AUDIT_ROOT / "analysis/predictions/p1_full_rrcl_linear.npz"
    with np.load(reference_path) as handle:
        reference = {key: handle[key] for key in handle.files}
    if not np.array_equal(reference["audit_row_index"], test):
        raise ValueError("P4 test rows differ from P1/P2")
    overstory, vegetation_overstory, multilayer, density, _, _ = geometry_strata(reference)
    label = reference["target"]
    groups = {
        "all": np.ones(test.size, bool), "P_inv": region == 0,
        "P_amb": region == 1, "P_vis": region == 2,
        "vegetation_interior": (label == 3) & (region == 0),
        "vegetation_visible": (label == 3) & (region == 2),
        "under_canopy_ground": (label == 2) & (region == 0) & vegetation_overstory,
        "exposed_ground": (label == 2) & (region == 2) & ~overstory,
        "multilayer": multilayer,
        "single_layer": ~multilayer & (density >= MIN_COLUMN_POINTS),
    }
    majority_role = int(np.argmax(counts))
    mean_context = data["context"][fit_index].mean(0)
    report = {
        "schema": "cvrl_p4_shortcut_v1",
        "interpretation": "target predictability only; not a backbone transfer comparison",
        "protocol": {"K": K, "features": ["log_intensity", "six_local_height_density",
                     "six_XYZ_PCA_descriptors"], "train": int(fitting.sum()),
                     "validation": int(validation.sum()), "test": int(valid.sum()),
                     "hidden": HIDDEN, "best_validation_loss": float(best),
                     "epochs_completed": epoch + 1, "device": DEVICE},
        "groups": {name: target_metrics(role, role_prediction, context, probability,
                                         valid & mask, majority_role, mean_context)
                   for name, mask in groups.items() if np.any(valid & mask)},
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    LOG.info("Saved %s; All role mIoU %.4f, context MAE %.4f", OUTPUT,
             report["groups"]["all"]["role"]["mIoU"],
             report["groups"]["all"]["context"]["MAE"])
    return report


if __name__ == "__main__":
    run()
