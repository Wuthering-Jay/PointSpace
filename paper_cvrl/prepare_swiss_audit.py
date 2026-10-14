"""Build a deterministic, class-aware Swiss-Audit subset.

The script is intentionally configured in ``__main__`` for direct IDE use.
It never moves or edits source files. Selected LAS, DINO and correspondence
files are copied or hard-linked into ``Swiss-Audit`` and a fixed, globally
fitted PCA model is applied without refitting it on the audit subset.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import laspy
import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file as save_safetensors
from tqdm import tqdm


LOGGER = logging.getLogger("swiss_audit")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

LAS_CLASSES = tuple(range(32))
IMPORTANT_CLASSES = (1, 2, 3, 6, 9, 11, 14, 15, 17, 26)
_HARDLINK_FALLBACK_WARNED = False


def _source_block(stem: str) -> str:
    head, separator, tail = stem.rpartition("_")
    return head if separator and len(tail) == 4 and tail.isdigit() else stem


def _classification_counts(path: Path) -> Dict:
    las = laspy.read(path)
    classification = np.asarray(las.classification, dtype=np.int16)
    counts = np.bincount(classification, minlength=32)[:32].astype(np.int64)
    z = np.asarray(las.z, dtype=np.float64)
    if z.size:
        z_low, z_median, z_high = np.percentile(z, (1, 50, 99))
    else:
        z_low = z_median = z_high = 0.0
    total = max(int(classification.size), 1)
    proportions = counts.astype(np.float64) / total
    nonzero = proportions[proportions > 0]
    entropy = float(-(nonzero * np.log(nonzero)).sum())
    return {
        "stem": path.stem,
        "filename": path.name,
        "source_block": _source_block(path.stem),
        "num_points": int(classification.size),
        "z_p01": float(z_low),
        "z_p50": float(z_median),
        "z_p99": float(z_high),
        "z_span_98": float(z_high - z_low),
        "class_entropy": entropy,
        "class_counts": counts.tolist(),
        "class_proportions": proportions.tolist(),
    }


def _environment(row: Dict) -> str:
    p = np.asarray(row["class_proportions"], dtype=np.float64)
    rare = p[14] + p[15] + p[17] + p[26]
    if rare >= 5e-4:
        return "rare_infrastructure"
    if p[9] >= 0.20:
        return "water"
    if p[6] >= 0.20:
        return "urban"
    if p[3] >= 0.55 and p[6] < 0.05:
        return "dense_vegetation"
    if p[2] >= 0.75 and p[6] < 0.03:
        return "open_ground"
    if p[6] >= 0.03:
        return "suburban"
    return "mixed"


def _priority(row: Dict) -> float:
    p = np.asarray(row["class_proportions"], dtype=np.float64)
    rare_presence = sum(float(p[index] > 0) for index in IMPORTANT_CLASSES)
    return (
        row["class_entropy"]
        + 0.10 * np.log1p(row["num_points"])
        + 0.02 * row["z_span_98"]
        + 0.30 * rare_presence
        + 20.0 * p[17]
        + 5.0 * (p[14] + p[15] + p[26])
    )


def _select_rows(
    rows: List[Dict], target_count: int, max_per_source_block: int
) -> List[Dict]:
    if target_count < 1 or target_count > len(rows):
        raise ValueError("target_count must be within the available file count")
    groups: Dict[str, List[Dict]] = {}
    for row in rows:
        row["environment"] = _environment(row)
        row["priority"] = _priority(row)
        groups.setdefault(row["source_block"], []).append(row)

    selected: Dict[str, Dict] = {}
    block_count = {name: 0 for name in groups}

    def add(row: Dict) -> bool:
        if row["stem"] in selected:
            return False
        block = row["source_block"]
        if block_count[block] >= max_per_source_block:
            return False
        selected[row["stem"]] = row
        block_count[block] += 1
        return True

    # Every original 1 km source block contributes at least one 100 m tile.
    for block in sorted(groups):
        add(max(groups[block], key=lambda item: item["priority"]))

    # Explicitly retain the strongest example of every observed LAS class.
    for class_id in LAS_CLASSES:
        candidates = [
            row for row in rows if row["class_counts"][class_id] > 0
        ]
        if candidates:
            add(max(candidates, key=lambda item: item["class_counts"][class_id]))

    weights = {
        "urban": 0.22,
        "suburban": 0.20,
        "dense_vegetation": 0.20,
        "open_ground": 0.15,
        "water": 0.08,
        "rare_infrastructure": 0.10,
        "mixed": 0.05,
    }
    quota = {
        name: max(1, int(round(target_count * weight)))
        for name, weight in weights.items()
    }
    for environment in weights:
        candidates = sorted(
            (row for row in rows if row["environment"] == environment),
            key=lambda item: item["priority"],
            reverse=True,
        )
        current = sum(
            row["environment"] == environment for row in selected.values()
        )
        for row in candidates:
            if current >= quota[environment] or len(selected) >= target_count:
                break
            current += int(add(row))

    # Fill any quota shortfall by global diversity/rarity priority.
    for row in sorted(rows, key=lambda item: item["priority"], reverse=True):
        if len(selected) >= target_count:
            break
        add(row)
    if len(selected) != target_count:
        raise RuntimeError(
            f"Could select only {len(selected)}/{target_count} files; "
            "increase max_per_source_block"
        )
    return sorted(selected.values(), key=lambda item: item["stem"])


def _link_or_copy(source: Path, target: Path, mode: str) -> None:
    global _HARDLINK_FALLBACK_WARNED
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    if mode == "hardlink":
        try:
            os.link(source, target)
        except OSError:
            if not _HARDLINK_FALLBACK_WARNED:
                LOGGER.warning(
                    "The destination filesystem does not support hard links; "
                    "falling back to physical copies for Swiss-Audit"
                )
                _HARDLINK_FALLBACK_WARNED = True
            shutil.copy2(source, target)
    elif mode == "copy":
        shutil.copy2(source, target)
    else:
        raise ValueError("storage_mode must be 'hardlink' or 'copy'")


def _load_pca_model(path: Path):
    with safe_open(str(path), framework="np") as handle:
        missing = {"mean", "components"}.difference(handle.keys())
        if missing:
            raise KeyError(f"PCA model is missing tensors: {sorted(missing)}")
        mean = np.asarray(handle.get_tensor("mean"), dtype=np.float32)
        components = np.asarray(handle.get_tensor("components"), dtype=np.float32)
        metadata = dict(handle.metadata() or {})
    if components.ndim != 2 or mean.shape != (components.shape[1],):
        raise ValueError("Invalid PCA mean/components shapes")
    if metadata.get("normalize_input", "").lower() != "true":
        raise ValueError("Swiss CVRL PCA model must use normalize_input=True")
    if metadata.get("whiten", "").lower() != "false":
        raise ValueError("Swiss CVRL PCA model must use whiten=False")
    return mean, components, metadata


def apply_fixed_pca_file(
    source: Path,
    target: Path,
    mean: np.ndarray,
    components: np.ndarray,
    output_dtype: str = "float16",
    output_normalize: bool = True,
    output_scale: float | None = None,
    chunk_size: int = 65536,
) -> None:
    """Apply the already fitted CVRL PCA model to one safetensors file."""
    dtype = {"float16": np.float16, "float32": np.float32}[output_dtype]
    output_scale = (
        float(np.sqrt(components.shape[0]))
        if output_scale is None
        else float(output_scale)
    )
    with safe_open(str(source), framework="np") as handle:
        if "feature" not in handle.keys():
            raise KeyError(f"{source} has no 'feature' tensor")
        feature_slice = handle.get_slice("feature")
        rows, channels = feature_slice.get_shape()
        if channels != components.shape[1]:
            raise ValueError(
                f"{source}: feature dim {channels} != PCA input {components.shape[1]}"
            )
        reduced = np.empty((rows, components.shape[0]), dtype=dtype)
        for start in range(0, rows, chunk_size):
            end = min(start + chunk_size, rows)
            value = np.asarray(feature_slice[start:end], dtype=np.float32)
            norm = np.linalg.norm(value, axis=1, keepdims=True)
            value = value / np.maximum(norm, 1e-12)
            value = (value - mean) @ components.T
            if output_normalize:
                norm = np.linalg.norm(value, axis=1, keepdims=True)
                value = value / np.maximum(norm, 1e-12)
                value *= output_scale
            reduced[start:end] = value.astype(dtype, copy=False)
        metadata = dict(handle.metadata() or {})
        extra = {
            key: np.array(handle.get_tensor(key), copy=True, order="C")
            for key in handle.keys()
            if key != "feature"
        }
    metadata.update(
        {
            "audit_transform": "fixed_global_pca",
            "pca_target_dim": str(components.shape[0]),
            "pca_normalize_input": "true",
            "pca_output_normalize": str(output_normalize).lower(),
            "pca_output_scale": str(output_scale),
            "dtype": output_dtype,
        }
    )
    tensors = {"feature": np.ascontiguousarray(reduced), **extra}
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    save_safetensors(tensors, str(temporary), metadata)
    os.replace(temporary, target)


def _write_manifest(path: Path, rows: Sequence[Dict]) -> None:
    fields = [
        "stem",
        "filename",
        "source_block",
        "environment",
        "num_points",
        "z_p01",
        "z_p50",
        "z_p99",
        "z_span_98",
        "class_entropy",
        "priority",
    ] + [f"class_{index}" for index in LAS_CLASSES]
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            record = {key: row[key] for key in fields if key in row}
            record.update(
                {
                    f"class_{index}": row["class_counts"][index]
                    for index in LAS_CLASSES
                }
            )
            writer.writerow(record)


def prepare_swiss_audit(
    source_root: str | Path,
    audit_root: str | Path,
    pca_model_path: str | Path,
    sample_fraction: float = 0.03,
    max_per_source_block: int = 5,
    scan_workers: int = 8,
    storage_mode: str = "hardlink",
    create_pca_teacher: bool = True,
    overwrite_manifest: bool = False,
) -> Dict:
    source_root = Path(source_root)
    audit_root = Path(audit_root)
    pointcloud_root = source_root / "pointcloud"
    files = sorted(pointcloud_root.glob("*.las")) + sorted(
        pointcloud_root.glob("*.laz")
    )
    if not 0.01 <= sample_fraction <= 0.05:
        raise ValueError("sample_fraction must be in [0.01, 0.05]")
    if not files:
        raise FileNotFoundError(f"No LAS/LAZ files in {pointcloud_root}")
    target_count = int(round(len(files) * sample_fraction))
    audit_root.mkdir(parents=True, exist_ok=True)
    scan_cache = audit_root / "all_tile_statistics.json"
    if scan_cache.exists() and not overwrite_manifest:
        rows = json.loads(scan_cache.read_text(encoding="utf-8"))
        LOGGER.info("Loaded cached statistics for %d tiles", len(rows))
    else:
        LOGGER.info("Scanning %d LAS tiles with %d workers", len(files), scan_workers)
        with ThreadPoolExecutor(max_workers=scan_workers) as executor:
            rows = list(
                tqdm(
                    executor.map(_classification_counts, files),
                    total=len(files),
                    desc="Scan Swiss tiles",
                    unit="tile",
                )
            )
        scan_cache.write_text(
            json.dumps(rows, ensure_ascii=False), encoding="utf-8"
        )

    selected = _select_rows(rows, target_count, max_per_source_block)
    for directory in (
        "pointcloud",
        "dino_feature",
        "correspondence",
        "dino_feature_pca128",
        "visibility",
        "features/vpsd_native1024",
        "features/vpsd_pca128",
        "features/cvrl_visual_native1024",
        "features/cvrl_visual_pca128",
        "features/rrcl",
        "features/cvrl_full",
        "analysis",
        "metadata",
    ):
        (audit_root / directory).mkdir(parents=True, exist_ok=True)

    for row in tqdm(selected, desc="Materialize Swiss-Audit", unit="tile"):
        stem = row["stem"]
        point_source = pointcloud_root / row["filename"]
        dino_source = source_root / "dino_feature" / f"{stem}.safetensors"
        corr_source = source_root / "correspondence" / f"{stem}.safetensors"
        for path in (point_source, dino_source, corr_source):
            if not path.is_file():
                raise FileNotFoundError(f"Selected audit triplet is incomplete: {path}")
        _link_or_copy(point_source, audit_root / "pointcloud" / point_source.name, storage_mode)
        _link_or_copy(dino_source, audit_root / "dino_feature" / dino_source.name, storage_mode)
        _link_or_copy(corr_source, audit_root / "correspondence" / corr_source.name, storage_mode)

    pca_model_path = Path(pca_model_path)
    pca_hash = hashlib.sha256(pca_model_path.read_bytes()).hexdigest()
    _link_or_copy(
        pca_model_path,
        audit_root / "metadata" / pca_model_path.name,
        storage_mode,
    )
    if create_pca_teacher:
        mean, components, _ = _load_pca_model(pca_model_path)
        for row in tqdm(selected, desc="Create PCA-128 teacher", unit="tile"):
            source = audit_root / "dino_feature" / f"{row['stem']}.safetensors"
            target = audit_root / "dino_feature_pca128" / source.name
            if not target.exists():
                apply_fixed_pca_file(source, target, mean, components)

    _write_manifest(audit_root / "audit_manifest.csv", selected)
    environments = {}
    class_totals = np.zeros(32, dtype=np.int64)
    for row in selected:
        environments[row["environment"]] = environments.get(row["environment"], 0) + 1
        class_totals += np.asarray(row["class_counts"], dtype=np.int64)
    first_audit_file = audit_root / "pointcloud" / selected[0]["filename"]
    effective_storage_mode = (
        "hardlink"
        if storage_mode == "hardlink" and os.stat(first_audit_file).st_nlink > 1
        else "copy_fallback"
        if storage_mode == "hardlink"
        else storage_mode
    )
    summary = {
        "schema": "pointspace_swiss_audit_v1",
        "source_root": str(source_root),
        "audit_root": str(audit_root),
        "source_tiles": len(files),
        "selected_tiles": len(selected),
        "sample_fraction_requested": sample_fraction,
        "sample_fraction_actual": len(selected) / len(files),
        "source_blocks": len({_source_block(row["stem"]) for row in rows}),
        "selected_source_blocks": len({row["source_block"] for row in selected}),
        "storage_mode_requested": storage_mode,
        "storage_mode_effective": effective_storage_mode,
        "environment_counts": environments,
        "class_point_counts": {
            str(index): int(value)
            for index, value in enumerate(class_totals)
            if value
        },
        "pca_model": str(pca_model_path),
        "pca_model_sha256": pca_hash,
        "pca_output": "row-normalized PCA scores scaled by sqrt(128)",
        "warning": (
            "This audit subset was selected from the already-used Swiss "
            "pretraining corpus. It is an in-distribution mechanism audit, "
            "not an unseen pretraining validation set."
        ),
    }
    (audit_root / "audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    LOGGER.info(
        "Swiss-Audit ready: %d/%d tiles (%.2f%%), %d source blocks",
        len(selected),
        len(files),
        100 * len(selected) / len(files),
        summary["selected_source_blocks"],
    )
    return summary


if __name__ == "__main__":
    prepare_swiss_audit(
        source_root=r"G:\data\swiss\joint_tiles_100",
        audit_root=r"G:\data\swiss\joint_tiles_100\Swiss-Audit",
        pca_model_path=r"G:\data\swiss\joint_tiles_100\dino_feature_pca128\dino_pca_model.safetensors",
        sample_fraction=0.03,
        max_per_source_block=5,
        scan_workers=8,
        storage_mode="hardlink",
        create_pca_teacher=True,
        overwrite_manifest=False,
    )
