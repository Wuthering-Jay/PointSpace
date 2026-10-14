"""LAS semantic training with compact, fixed sparse-label manifests."""

import json
from pathlib import Path

import numpy as np

from .builder import DATASETS
from .las import LasDataset


@DATASETS.register_module()
class SparseLabelLasDataset(LasDataset):
    def __init__(self, *, label_manifest_dir, label_fraction, **kwargs):
        self.label_manifest_dir = Path(label_manifest_dir)
        self.label_fraction = float(label_fraction)
        self._label_summary = json.loads(
            (self.label_manifest_dir / "summary.json").read_text(encoding="utf-8")
        )
        self._fraction_key = str(self.label_fraction)
        if self._fraction_key not in self._label_summary["global_counts"]:
            raise ValueError(f"Fraction {self.label_fraction} absent from label manifest")
        if kwargs.get("test_mode", False):
            raise ValueError("Use ordinary LasDataset for validation and testing")
        transform = kwargs.get("transform") or []
        names = [item.get("type") for item in transform]
        if (
            "ClassFilter" not in names
            or "SparseLabelMask" not in names
            or names.index("SparseLabelMask") != names.index("ClassFilter") + 1
        ):
            raise ValueError("Place SparseLabelMask after ClassFilter in train transforms")
        super().__init__(**kwargs)
        expected = set(self._label_summary["files"])
        actual = {Path(path).name for path in self.data_list}
        if expected != actual:
            raise ValueError("Manifest file list does not match training LAS directory")

    def _scan_classes(self):
        return list(self._label_summary["classes"])

    def _compute_class_weights(self):
        if self.class_weight_mode not in {"sqrt", "auto"}:
            raise ValueError("Sparse labels currently support class_weight='sqrt' only")
        counts = np.asarray(self._label_summary["global_counts"][self._fraction_key], np.float64)
        if np.any(counts == 0):
            raise ValueError(f"No labeled examples for classes {np.flatnonzero(counts == 0).tolist()}")
        weights = 1.0 / np.sqrt(counts / counts.sum())
        self.class_weight = (weights / weights.sum() * len(weights)).astype(np.float32)

    def _compute_sample_weights(self):
        weights = []
        for path in self.data_list:
            counts = self._label_summary["files"][Path(path).name]["counts"][self._fraction_key]
            present = np.asarray(counts) > 0
            # A tile with no retained labels cannot supply a supervised
            # gradient. Keep it in the manifest, but never draw it for train.
            weights.append(float(self.class_weight[present].sum()) if present.any() else 0.0)
        self.sample_weights = np.asarray(weights, dtype=np.float64)
        if self.sample_weights.sum() <= 0:
            raise ValueError("Sparse-label budget contains no trainable tile")
        self.sample_weights *= len(weights) / self.sample_weights.sum()

    def get_data(self, idx):
        data = super().get_data(idx)
        path = Path(self.data_list[idx % len(self.data_list)])
        record = self._label_summary["files"][path.name]
        if path.stat().st_size != record["source_bytes"]:
            raise ValueError(f"LAS source changed since manifest creation: {path}")
        if path.stat().st_mtime_ns != record["source_mtime_ns"]:
            raise ValueError(f"LAS source timestamp changed since manifest creation: {path}")
        n = len(data["segment"])
        if n != record["point_count"]:
            raise ValueError(f"LAS point count changed since manifest creation: {path}")
        with np.load(self.label_manifest_dir / f"{path.stem}.npz") as entry:
            if int(entry["point_count"]) != n:
                raise ValueError(f"Manifest point count mismatch: {path}")
            chosen = entry["index"][entry["rank"] < self.label_fraction]
        keep = np.zeros(n, dtype=np.bool_)
        keep[chosen] = True
        data["sparse_keep"] = keep
        data["index_valid_keys"].append("sparse_keep")
        return data

    def prepare_train_data(self, idx):
        # At 0.01%, a sampled voxel representative can miss every annotated
        # point in a tile. Reject that sample before the criterion sees an
        # all-ignore target (which would make CE return NaN).
        candidate = idx
        for _ in range(8):
            data = self.transform(self.get_data(candidate))
            if np.any(data["segment"] != self.ignore_index):
                return self.post_transform(data)
            if self.sample_weights is not None:
                probability = self.sample_weights / self.sample_weights.sum()
                candidate = int(np.random.choice(len(self.data_list), p=probability))
            else:
                candidate = int(np.random.randint(len(self.data_list)))
        raise RuntimeError("Unable to sample a tile with labeled points after transforms")
