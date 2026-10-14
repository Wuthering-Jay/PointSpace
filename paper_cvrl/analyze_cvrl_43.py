"""Section 4.3 Swiss-Audit representation and visibility analysis.

The analysis is intentionally offline. Feature extraction is performed by the
normal PointSpace testers, while this script uses one fixed per-tile point
index, source-block splits and identical frozen classifiers for every level.
"""

from __future__ import annotations

import csv
import gc
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
from safetensors import safe_open
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.neighbors import KNeighborsClassifier

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
from paper_cvrl.section43_metrics import mean_iou, per_class_iou


LOGGER = logging.getLogger("cvrl_43")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

REGION_NAMES = {0: "P_inv", 1: "P_amb", 2: "P_vis"}


def _read_tensor(path: Path, key: str) -> np.ndarray:
    with safe_open(str(path), framework="np") as handle:
        if key not in handle.keys():
            raise KeyError(f"{path} has no '{key}' tensor")
        return np.asarray(handle.get_tensor(key))


def _manifest(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _build_rows(audit_root: Path, manifest_rows: Sequence[Mapping]) -> Dict:
    result = {
        "stem": [],
        "point_index": [],
        "label": [],
        "region": [],
        "height_bin": [],
        "environment": [],
        "source_block": [],
        "teacher_valid": [],
        "patch_index": [],
    }
    for row in manifest_rows:
        stem = row["stem"]
        visibility_path = audit_root / "visibility" / f"{stem}.safetensors"
        correspondence_path = audit_root / "correspondence" / f"{stem}.safetensors"
        with safe_open(str(visibility_path), framework="np") as handle:
            index = np.asarray(handle.get_tensor("audit_index"), dtype=np.int64)
            label = np.asarray(handle.get_tensor("classification"), dtype=np.int64)[index]
            region = np.asarray(handle.get_tensor("region"), dtype=np.int64)[index]
            height_bin = np.asarray(handle.get_tensor("height_bin"), dtype=np.int64)[index]
        with safe_open(str(correspondence_path), framework="np") as handle:
            valid = np.asarray(handle.get_tensor("valid"), dtype=bool)[index]
            patch = np.asarray(handle.get_tensor("patch_index"), dtype=np.int64)[index]
            valid &= patch >= 0
        count = index.size
        result["stem"].extend([stem] * count)
        result["point_index"].append(index)
        result["label"].append(label)
        result["region"].append(region)
        result["height_bin"].append(height_bin)
        result["environment"].extend([row["environment"]] * count)
        result["source_block"].extend([row["source_block"]] * count)
        result["teacher_valid"].append(valid)
        result["patch_index"].append(patch)
    for key in ("point_index", "label", "region", "height_bin", "teacher_valid", "patch_index"):
        result[key] = np.concatenate(result[key])
    for key in ("stem", "environment", "source_block"):
        result[key] = np.asarray(result[key], dtype=object)
    return result


def _tile_slices(stems: np.ndarray):
    unique, first, count = np.unique(stems, return_index=True, return_counts=True)
    return {
        stem: slice(int(begin), int(begin + length))
        for stem, begin, length in zip(unique, first, count)
    }


def _load_feature_source(
    audit_root: Path,
    spec: Mapping,
    rows: Dict,
) -> np.ndarray:
    kind = spec.get("kind", "point")
    folder = audit_root / spec["folder"]
    key = spec.get("key", "feature")
    parts = []
    for stem, row_slice in _tile_slices(rows["stem"]).items():
        expected_index = rows["point_index"][row_slice]
        path = folder / f"{stem}.safetensors"
        with safe_open(str(path), framework="np") as handle:
            feature = np.asarray(handle.get_tensor(key))
            if kind == "point":
                if "point_index" in handle.keys():
                    stored_index = np.asarray(
                        handle.get_tensor("point_index"), dtype=np.int64
                    )
                    if not np.array_equal(stored_index, expected_index):
                        lookup = {int(value): index for index, value in enumerate(stored_index)}
                        try:
                            take = np.asarray(
                                [lookup[int(value)] for value in expected_index],
                                dtype=np.int64,
                            )
                        except KeyError as error:
                            raise ValueError(
                                f"{path} does not cover audit point {error.args[0]}"
                            ) from error
                        feature = feature[take]
                elif feature.shape[0] != expected_index.size:
                    raise ValueError(
                        f"{path}: full point feature requires point_index or exact audit rows"
                    )
            elif kind == "patch":
                patch = rows["patch_index"][row_slice]
                safe_patch = np.clip(patch, 0, max(feature.shape[0] - 1, 0))
                feature = feature[safe_patch]
            else:
                raise ValueError("Feature source kind must be 'point' or 'patch'")
        if feature.shape[0] != expected_index.size:
            raise ValueError(f"{path}: feature/audit row mismatch")
        parts.append(feature.astype(np.float32, copy=False))
    return np.concatenate(parts, axis=0)


def _load_feature_tile(audit_root: Path, spec: Mapping, rows: Dict, row_slice):
    """Load one source tile in the common audit-row order."""
    stem = str(rows["stem"][row_slice][0])
    expected_index = rows["point_index"][row_slice]
    path = audit_root / spec["folder"] / f"{stem}.safetensors"
    key = spec.get("key", "feature")
    with safe_open(str(path), framework="np") as handle:
        feature = np.asarray(handle.get_tensor(key))
        if spec.get("kind", "point") == "point":
            if "point_index" in handle.keys():
                stored_index = np.asarray(handle.get_tensor("point_index"), dtype=np.int64)
                if not np.array_equal(stored_index, expected_index):
                    lookup = {int(value): index for index, value in enumerate(stored_index)}
                    feature = feature[
                        np.asarray([lookup[int(value)] for value in expected_index])
                    ]
            elif feature.shape[0] != expected_index.size:
                raise ValueError(f"{path}: point feature/audit row mismatch")
        elif spec.get("kind") == "patch":
            patch = rows["patch_index"][row_slice]
            feature = feature[np.clip(patch, 0, max(feature.shape[0] - 1, 0))]
        else:
            raise ValueError("Feature source kind must be 'point' or 'patch'")
    return feature.astype(np.float32, copy=False)


def _paired_cosine_stream(audit_root, first_spec, second_spec, rows):
    values = []
    for _, row_slice in _tile_slices(rows["stem"]).items():
        mask = rows["teacher_valid"][row_slice]
        if not mask.any():
            continue
        first = _normalize(_load_feature_tile(audit_root, first_spec, rows, row_slice)[mask])
        second = _normalize(_load_feature_tile(audit_root, second_spec, rows, row_slice)[mask])
        if first.shape != second.shape:
            raise ValueError(f"Paired feature shapes differ: {first.shape} vs {second.shape}")
        values.append(np.sum(first * second, axis=1))
    cosine = np.concatenate(values)
    return {
        "points": int(cosine.size),
        "mean": float(cosine.mean()),
        "std": float(cosine.std()),
        "p10": float(np.quantile(cosine, 0.10)),
        "p50": float(np.quantile(cosine, 0.50)),
        "p90": float(np.quantile(cosine, 0.90)),
    }


def _normalize(feature: np.ndarray) -> np.ndarray:
    return feature / np.maximum(np.linalg.norm(feature, axis=1, keepdims=True), 1e-12)


def _block_split(source_block: np.ndarray, test_fraction: float):
    blocks = np.unique(source_block)
    if blocks.size < 2:
        raise ValueError("Audit requires at least two source blocks")
    count = min(max(1, int(round(blocks.size * test_fraction))), blocks.size - 1)
    held_position = np.linspace(0, blocks.size - 1, count).round().astype(int)
    test_blocks = blocks[held_position]
    return ~np.isin(source_block, test_blocks), np.isin(source_block, test_blocks)


def _stratified_subsample(index, label, region, maximum, random_state):
    index = np.asarray(index, dtype=np.int64)
    if index.size <= maximum:
        return index
    rng = np.random.default_rng(random_state)
    strata = label[index].astype(np.int64) * 3 + region[index].astype(np.int64)
    unique, counts = np.unique(strata, return_counts=True)
    allocation = np.maximum(1, np.floor(maximum * counts / counts.sum()).astype(int))
    while allocation.sum() > maximum:
        position = int(np.argmax(allocation))
        if allocation[position] > 1:
            allocation[position] -= 1
        else:
            break
    parts = []
    for value, take in zip(unique, allocation):
        candidates = index[strata == value]
        take = min(int(take), candidates.size)
        parts.append(rng.choice(candidates, size=take, replace=False))
    selected = np.concatenate(parts)
    if selected.size < maximum:
        remaining = np.setdiff1d(index, selected, assume_unique=False)
        take = min(maximum - selected.size, remaining.size)
        selected = np.r_[selected, rng.choice(remaining, size=take, replace=False)]
    return np.sort(selected[:maximum])


def _metric_block(target, prediction, region, labels):
    result = {}
    for name, mask in {
        "all": np.ones(target.size, dtype=bool),
        **{region_name: region == region_id for region_id, region_name in REGION_NAMES.items()},
    }.items():
        if not mask.any():
            result[name] = {"points": 0}
            continue
        y_true = target[mask]
        y_pred = prediction[mask]
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, zero_division=0
        )
        matrix = confusion_matrix(y_true, y_pred, labels=labels)
        class_iou = per_class_iou(matrix)
        result[name] = {
            "points": int(mask.sum()),
            "OA": float(accuracy_score(y_true, y_pred)),
            "macro_F1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "mIoU": mean_iou(matrix),
            "per_class": {
                str(class_id): {
                    "precision": float(precision[index]),
                    "recall": float(recall[index]),
                    "F1": float(f1[index]),
                    "IoU": (float(class_iou[index])
                            if np.isfinite(class_iou[index]) else None),
                    "support": int(support[index]),
                }
                for index, class_id in enumerate(labels)
            },
            "confusion_matrix": matrix.tolist(),
        }
    return result


def _mismatch_summary(
    model_name,
    target,
    prediction,
    region,
    environment,
    height_bin,
):
    mismatch = target != prediction
    rows = {}
    for index in np.flatnonzero(mismatch):
        key = (
            model_name,
            REGION_NAMES[int(region[index])],
            str(environment[index]),
            int(height_bin[index]),
            int(target[index]),
            int(prediction[index]),
        )
        rows[key] = rows.get(key, 0) + 1
    return [
        {
            "model": key[0],
            "region": key[1],
            "environment": key[2],
            "height_bin": key[3],
            "target": key[4],
            "prediction": key[5],
            "count": count,
        }
        for key, count in sorted(rows.items(), key=lambda item: item[1], reverse=True)
    ]


def _fit_probe(
    feature,
    rows,
    mask,
    settings,
    seed_offset=0,
    prediction_path: Path | None = None,
):
    train_block, test_block = _block_split(rows["source_block"], settings["test_fraction"])
    valid = mask & (rows["label"] >= 0)
    train_index = _stratified_subsample(
        np.flatnonzero(valid & train_block),
        rows["label"],
        rows["region"],
        settings["max_train_points"],
        settings["random_state"] + seed_offset,
    )
    test_index = _stratified_subsample(
        np.flatnonzero(valid & test_block),
        rows["label"],
        rows["region"],
        settings["max_test_points"],
        settings["random_state"] + 1000 + seed_offset,
    )
    labels = np.unique(rows["label"][train_index])
    if labels.size < 2:
        raise ValueError("Probe training split contains fewer than two classes")
    train_feature = _normalize(feature[train_index])
    test_feature = _normalize(feature[test_index])
    if not np.isfinite(train_feature).all() or not np.isfinite(test_feature).all():
        raise ValueError("Probe features contain NaN or Inf")
    linear = LogisticRegression(
        C=settings["linear_c"],
        class_weight="balanced",
        max_iter=settings["linear_max_iter"],
        solver="lbfgs",
        random_state=settings["random_state"],
    )
    linear.fit(train_feature, rows["label"][train_index])
    linear_prediction = linear.predict(test_feature)
    if prediction_path is not None:
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            prediction_path,
            audit_row_index=test_index,
            stem=np.asarray(rows["stem"][test_index], dtype=str),
            point_index=rows["point_index"][test_index],
            target=rows["label"][test_index],
            prediction=linear_prediction,
            region=rows["region"][test_index],
            height_bin=rows["height_bin"][test_index],
            environment=np.asarray(rows["environment"][test_index], dtype=str),
        )
    knn_train = _stratified_subsample(
        train_index,
        rows["label"],
        rows["region"],
        settings["max_knn_train_points"],
        settings["random_state"] + 2000 + seed_offset,
    )
    knn_test = _stratified_subsample(
        test_index,
        rows["label"],
        rows["region"],
        settings["max_knn_test_points"],
        settings["random_state"] + 3000 + seed_offset,
    )
    knn = KNeighborsClassifier(
        n_neighbors=min(settings["knn_k"], knn_train.size),
        metric="cosine",
        weights="distance",
        n_jobs=-1,
    )
    knn.fit(_normalize(feature[knn_train]), rows["label"][knn_train])
    knn_prediction = knn.predict(_normalize(feature[knn_test]))
    return {
        "feature_dim": int(feature.shape[1]),
        "train_points": int(train_index.size),
        "test_points": int(test_index.size),
        "linear": _metric_block(
            rows["label"][test_index],
            linear_prediction,
            rows["region"][test_index],
            labels,
        ),
        "knn_train_points": int(knn_train.size),
        "knn_test_points": int(knn_test.size),
        "knn": _metric_block(
            rows["label"][knn_test],
            knn_prediction,
            rows["region"][knn_test],
            labels,
        ),
        "mismatch": _mismatch_summary(
            "linear",
            rows["label"][test_index],
            linear_prediction,
            rows["region"][test_index],
            rows["environment"][test_index],
            rows["height_bin"][test_index],
        ),
    }


def analyze_cvrl_section_43(
    audit_root: str | Path,
    internalization_sources: Mapping[str, Mapping],
    regional_sources: Mapping[str, Mapping],
    output_path: str | Path | None = None,
    test_fraction: float = 0.20,
    max_train_points: int = 100000,
    max_test_points: int = 100000,
    max_knn_train_points: int = 5000,
    max_knn_test_points: int = 5000,
    linear_c: float = 1.0,
    linear_max_iter: int = 500,
    knn_k: int = 20,
    random_state: int = 42,
) -> Dict:
    audit_root = Path(audit_root)
    manifest_rows = _manifest(audit_root / "audit_manifest.csv")
    rows = _build_rows(audit_root, manifest_rows)
    settings = {
        "test_fraction": test_fraction,
        "max_train_points": max_train_points,
        "max_test_points": max_test_points,
        "max_knn_train_points": max_knn_train_points,
        "max_knn_test_points": max_knn_test_points,
        "linear_c": linear_c,
        "linear_max_iter": linear_max_iter,
        "knn_k": knn_k,
        "random_state": random_state,
    }
    report = {
        "schema": "cvrl_section_43_v2",
        "protocol": {
            **settings,
            "split_unit": "original Swiss source block, never individual points",
            "normalization": "row L2",
            "linear_solver": "balanced multinomial LogisticRegression(lbfgs)",
            "regions": REGION_NAMES,
        },
        "audit_tiles": len(manifest_rows),
        "audit_rows": int(rows["label"].size),
        "teacher_visible_rows": int(rows["teacher_valid"].sum()),
        "internalization": {},
        "regional_compensation": {},
        "paired_cosine": {},
    }

    teacher_mask = rows["teacher_valid"]
    for index, (name, spec) in enumerate(internalization_sources.items()):
        LOGGER.info("Loading internalization source %s", name)
        feature = _load_feature_source(audit_root, spec, rows)
        report["internalization"][name] = _fit_probe(
            feature,
            rows,
            teacher_mask,
            settings,
            # All representations must see exactly the same sampled rows.
            seed_offset=0,
            prediction_path=audit_root
            / "analysis"
            / "predictions"
            / f"internalization_{name}_linear.npz",
        )
        del feature
        gc.collect()

    for student_name, teacher_name in (
        ("vpsd_native1024", "dino_native1024"),
        ("cvrl_visual_native1024", "dino_native1024"),
        ("vpsd_pca128", "dino_pca128"),
        ("cvrl_visual_pca128", "dino_pca128"),
    ):
        if student_name in internalization_sources and teacher_name in internalization_sources:
            report["paired_cosine"][f"{student_name}_vs_{teacher_name}"] = (
                _paired_cosine_stream(
                    audit_root,
                    internalization_sources[student_name],
                    internalization_sources[teacher_name],
                    rows,
                )
            )

    for index, (name, spec) in enumerate(regional_sources.items()):
        LOGGER.info("Loading regional source %s", name)
        feature = _load_feature_source(audit_root, spec, rows)
        report["regional_compensation"][name] = _fit_probe(
            feature,
            rows,
            np.ones(rows["label"].shape, dtype=bool),
            settings,
            # Keep train/test samples identical across every representation.
            seed_offset=100,
            prediction_path=audit_root
            / "analysis"
            / "predictions"
            / f"regional_{name}_linear.npz",
        )
        del feature
        gc.collect()

    output_path = (
        audit_root / "analysis" / "section_43_report.json"
        if output_path is None
        else Path(output_path)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    LOGGER.info("Saved section 4.3 report to %s", output_path)
    return report


if __name__ == "__main__":
    root = r"G:\data\swiss\joint_tiles_100\Swiss-Audit"
    # Run after VPSD/CVRL feature export. All point-level exports must use the
    # same visibility/audit_index sidecars generated by swiss_audit_visibility.py.
    analyze_cvrl_section_43(
        audit_root=root,
        internalization_sources={
            "dino_native1024": {"kind": "patch", "folder": "dino_feature"},
            "dino_pca128": {"kind": "patch", "folder": "dino_feature_pca128"},
            "vpsd_native1024": {"kind": "point", "folder": "features/vpsd_native1024"},
            "vpsd_pca128": {"kind": "point", "folder": "features/vpsd_pca128"},
            "cvrl_visual_native1024": {
                "kind": "point",
                "folder": "features/cvrl_visual_native1024",
            },
            "cvrl_visual_pca128": {
                "kind": "point",
                "folder": "features/cvrl_visual_pca128",
            },
        },
        regional_sources={
            "vpsd_pca128": {"kind": "point", "folder": "features/vpsd_pca128"},
            "cvrl_visual_pca128": {
                "kind": "point",
                "folder": "features/cvrl_visual_pca128",
            },
            # These are the return branch and fused output of the joint CVRL
            # checkpoint.  ``cvrl_return72`` is not a separately trained RRCL
            # control and must not be labelled as one in the paper.
            "cvrl_return72": {"kind": "point", "folder": "features/rrcl"},
            "rrcl_independent72": {
                "kind": "point",
                "folder": "features/rrcl_independent",
            },
            "cvrl_full200": {"kind": "point", "folder": "features/cvrl_full"},
            "vpsd_scratch_pca128": {
                "kind": "point",
                "folder": "features/vpsd_scratch_pca128",
            },
        },
        output_path=Path(root) / "analysis" / "section_43_report.json",
    )
