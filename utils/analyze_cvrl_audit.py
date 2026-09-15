"""Fixed frozen-linear/kNN evaluation for CVRL audit features."""

import json
import logging
from pathlib import Path
from typing import Dict

import numpy as np
from safetensors import safe_open
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    jaccard_score,
    precision_score,
    recall_score,
)
from sklearn.neighbors import KNeighborsClassifier


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)


def _load_source(spec):
    path = Path(spec["path"])
    key = spec["key"]
    index_key = spec.get("index_key")
    with safe_open(str(path), framework="np") as handle:
        required = {key, "label", "sample_id"}
        if index_key is not None:
            required.add(index_key)
        missing = required.difference(handle.keys())
        if missing:
            raise KeyError(f"{path} is missing tensors: {sorted(missing)}")
        feature = handle.get_tensor(key).astype(np.float32, copy=False)
        label = handle.get_tensor("label").reshape(-1).astype(np.int64)
        sample_id = handle.get_tensor("sample_id").reshape(-1).astype(np.int64)
        if index_key is not None:
            index = handle.get_tensor(index_key).reshape(-1).astype(np.int64)
            label = label[index]
            sample_id = sample_id[index]
    if not (feature.shape[0] == label.size == sample_id.size):
        raise ValueError(
            f"Feature/label/sample length mismatch in {path}: "
            f"{feature.shape[0]}/{label.size}/{sample_id.size}"
        )
    return feature, label, sample_id


def _normalize(feature):
    norm = np.linalg.norm(feature, axis=1, keepdims=True)
    return feature / np.maximum(norm, 1e-12)


def _fixed_tile_split(sample_id, test_fraction):
    unique = np.unique(sample_id)
    if unique.size < 2:
        raise ValueError("Frozen probe requires at least two audit samples")
    test_count = max(1, int(round(unique.size * test_fraction)))
    test_count = min(test_count, unique.size - 1)
    # IDs are deterministic validation-loader positions. Taking evenly spaced
    # held-out tiles is stable across models and avoids point-level leakage.
    position = np.linspace(0, unique.size - 1, test_count).round().astype(int)
    test_ids = unique[position]
    return ~np.isin(sample_id, test_ids), np.isin(sample_id, test_ids)


def _subsample(index, maximum):
    if index.size <= maximum:
        return index
    position = np.linspace(0, index.size - 1, maximum).round().astype(int)
    return index[position]


def _metrics(target, prediction, focus_classes):
    result = {
        "OA": float(accuracy_score(target, prediction)),
        "macro_F1": float(f1_score(target, prediction, average="macro", zero_division=0)),
    }
    for name, class_id in focus_classes.items():
        binary_target = target == class_id
        binary_prediction = prediction == class_id
        result[name] = {
            "class_id": int(class_id),
            "IoU": float(jaccard_score(binary_target, binary_prediction, zero_division=0)),
            "precision": float(precision_score(binary_target, binary_prediction, zero_division=0)),
            "recall": float(recall_score(binary_target, binary_prediction, zero_division=0)),
            "F1": float(f1_score(binary_target, binary_prediction, zero_division=0)),
            "support": int(binary_target.sum()),
        }
    return result


def analyze_cvrl_audit(
    source_specs: Dict[str, Dict],
    focus_classes=None,
    ignore_labels=(-1,),
    test_fraction=0.2,
    max_train_points=100000,
    linear_c=1.0,
    linear_max_iter=1000,
    knn_k=20,
    random_state=42,
    output_path=None,
):
    """Run the identical frozen protocols over all four diagnostic levels."""
    if focus_classes is None:
        focus_classes = {"building": 6, "road": 11, "bridge": 17}
    report = {
        "protocol": {
            "split": "deterministic tile-level",
            "test_fraction": test_fraction,
            "feature_normalization": "row L2",
            "max_train_points": max_train_points,
            "linear": f"LogisticRegression(C={linear_c}, class_weight=balanced)",
            "knn": f"KNeighborsClassifier(k={knn_k}, metric=cosine, weights=distance)",
            "random_state": random_state,
        },
        "focus_classes": focus_classes,
        "sources": {},
    }
    reference_label = None
    reference_sample_id = None
    for source_name, spec in source_specs.items():
        feature, label, sample_id = _load_source(spec)
        valid = np.isfinite(feature).all(axis=1)
        valid &= ~np.isin(label, np.asarray(ignore_labels))
        feature = _normalize(feature[valid])
        label = label[valid]
        sample_id = sample_id[valid]
        if reference_label is None:
            reference_label = label.copy()
            reference_sample_id = sample_id.copy()
        elif not (
            np.array_equal(label, reference_label)
            and np.array_equal(sample_id, reference_sample_id)
        ):
            raise ValueError(
                f"{source_name}: evaluation rows differ from the first source; "
                "all four levels must use the identical fixed audit subset"
            )
        train_mask, test_mask = _fixed_tile_split(sample_id, test_fraction)
        train_index = _subsample(np.flatnonzero(train_mask), max_train_points)
        test_index = np.flatnonzero(test_mask)
        if np.unique(label[train_index]).size < 2:
            raise ValueError(f"{source_name}: training split has fewer than two classes")

        linear = LogisticRegression(
            C=linear_c,
            max_iter=linear_max_iter,
            class_weight="balanced",
            random_state=random_state,
        )
        linear.fit(feature[train_index], label[train_index])
        linear_prediction = linear.predict(feature[test_index])

        neighbors = min(knn_k, train_index.size)
        knn = KNeighborsClassifier(
            n_neighbors=neighbors,
            metric="cosine",
            weights="distance",
            n_jobs=-1,
        )
        knn.fit(feature[train_index], label[train_index])
        knn_prediction = knn.predict(feature[test_index])
        result = {
            "feature_dim": int(feature.shape[1]),
            "train_points": int(train_index.size),
            "test_points": int(test_index.size),
            "linear": _metrics(label[test_index], linear_prediction, focus_classes),
            "knn": _metrics(label[test_index], knn_prediction, focus_classes),
        }
        report["sources"][source_name] = result
        LOGGER.info(
            "%s %dD | linear OA %.4f macro-F1 %.4f | kNN OA %.4f macro-F1 %.4f",
            source_name,
            feature.shape[1],
            result["linear"]["OA"],
            result["linear"]["macro_F1"],
            result["knn"]["OA"],
            result["knn"]["macro_F1"],
        )
        for class_name in focus_classes:
            LOGGER.info(
                "  %-8s | linear F1 %.4f IoU %.4f | kNN F1 %.4f IoU %.4f",
                class_name,
                result["linear"][class_name]["F1"],
                result["linear"][class_name]["IoU"],
                result["knn"][class_name]["F1"],
                result["knn"][class_name]["IoU"],
            )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        LOGGER.info("Saved protocol report to %s", output_path)
    return report


if __name__ == "__main__":
    # 四级上限诊断。visual-only 与 joint 应使用相同固定审计点集导出的文件。
    joint_audit_path = r"exp\cvrl\path\audit_features\epoch_0010.safetensors"
    visual_only_audit_path = r"exp\cvrl\visual_only\audit_features\epoch_0010.safetensors"
    analyze_cvrl_audit(
        source_specs={
            "teacher_1024": {
                "path": joint_audit_path,
                "key": "teacher_native",
                "index_key": "teacher_point_index",
            },
            "pca_teacher_128": {
                "path": joint_audit_path,
                "key": "teacher_pca",
                "index_key": "teacher_point_index",
            },
            "visual_only_128": {
                "path": visual_only_audit_path,
                "key": "z_visual",
                "index_key": "teacher_point_index",
            },
            "joint_visual_128": {
                "path": joint_audit_path,
                "key": "z_visual",
                "index_key": "teacher_point_index",
            },
        },
        focus_classes={"building": 6, "road": 11, "bridge": 17},
        ignore_labels=(-1,),
        test_fraction=0.2,
        max_train_points=100000,
        linear_c=1.0,
        linear_max_iter=1000,
        knn_k=20,
        random_state=42,
        output_path=r"exp\cvrl\four_level_diagnosis.json",
    )
