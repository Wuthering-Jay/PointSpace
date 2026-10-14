"""Build the frozen Swiss-100 inventory used by Section 4.5 baselines.

This is intentionally configured with module-level constants so it can be run
directly from an IDE.  The normal mode scans every file header and performs a
content/return/correspondence audit on a deterministic subset.  Set
``HASH_ALL_FILES`` to ``True`` only when producing the archival release: the
three aligned folders contain hundreds of GiB and full hashing is I/O bound.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import pyproj  # noqa: F401 - load PROJ DLLs before laspy on Windows
import laspy
import numpy as np
from safetensors import safe_open


DATA_ROOT = Path(r"G:\data\swiss\joint_tiles_100")
OUTPUT_ROOT = Path(__file__).resolve().parent / "results" / "section_45"
SEED = 45
AUDIT_TILE_COUNT = 128
HASH_ALL_FILES = False
HASH_BUFFER_BYTES = 8 * 1024 * 1024


def _files_by_stem(folder: Path, suffixes: tuple[str, ...]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in suffixes:
            if path.stem in result:
                raise ValueError(f"Duplicate stem in {folder}: {path.stem}")
            result[path.stem] = path
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(HASH_BUFFER_BYTES):
            digest.update(block)
    return digest.hexdigest()


def _tensor_schema(path: Path) -> tuple[dict[str, list[int]], dict[str, str]]:
    with safe_open(str(path), framework="np") as handle:
        shapes = {key: list(handle.get_slice(key).get_shape()) for key in handle.keys()}
        metadata = dict(handle.metadata() or {})
    return shapes, metadata


def _point_header(path: Path) -> tuple[int, int, list[str]]:
    with laspy.open(path) as reader:
        header = reader.header
        return (
            int(header.point_count),
            int(header.point_format.id),
            sorted(str(name) for name in header.point_format.dimension_names),
        )


def _audit_selected(point_path: Path, correspondence_path: Path) -> dict:
    cloud = laspy.read(point_path)
    point_count = len(cloud.points)
    dimensions = set(cloud.point_format.dimension_names)
    has_return = {"return_number", "number_of_returns"}.issubset(dimensions)
    if has_return:
        return_number = np.asarray(cloud.return_number, dtype=np.int16)
        number_of_returns = np.asarray(cloud.number_of_returns, dtype=np.int16)
        valid_return = (
            (return_number >= 1)
            & (number_of_returns >= 1)
            & (return_number <= number_of_returns)
        )
        return_valid = int(valid_return.sum())
    else:
        return_valid = 0

    with safe_open(str(correspondence_path), framework="np") as handle:
        valid = np.asarray(handle.get_tensor("valid"), dtype=bool)
        patch_index = np.asarray(handle.get_tensor("patch_index"), dtype=np.int64)
        observability = (
            np.asarray(handle.get_tensor("observability"), dtype=np.float32)
            if "observability" in handle.keys() else None
        )
    valid_mapping = valid & (patch_index >= 0)
    return {
        "points": point_count,
        "return_valid": return_valid,
        "return_valid_ratio": return_valid / max(point_count, 1),
        "correspondence_points": int(valid.size),
        "correspondence_valid": int(valid_mapping.sum()),
        "correspondence_valid_ratio": float(valid_mapping.mean()) if valid.size else 0.0,
        "observability_mean": (
            float(np.mean(observability)) if observability is not None and observability.size
            else None
        ),
    }


def build_manifest() -> dict:
    point_files = _files_by_stem(DATA_ROOT / "pointcloud", (".las", ".laz"))
    dino_files = _files_by_stem(DATA_ROOT / "dino_feature", (".safetensors",))
    correspondence_files = _files_by_stem(
        DATA_ROOT / "correspondence", (".safetensors",)
    )
    stem_sets = {
        "pointcloud": set(point_files),
        "dino_feature": set(dino_files),
        "correspondence": set(correspondence_files),
    }
    shared = set.intersection(*stem_sets.values())
    if any(stems != shared for stems in stem_sets.values()):
        missing = {
            name: sorted(set.union(*stem_sets.values()) - stems)[:20]
            for name, stems in stem_sets.items()
        }
        raise ValueError(f"The three Swiss-100 modalities are not aligned: {missing}")

    stems = sorted(shared)
    rng = random.Random(SEED)
    selected = set(rng.sample(stems, min(AUDIT_TILE_COUNT, len(stems))))
    rows = []
    dino_dims = Counter()
    point_formats = Counter()
    total_points = 0
    sampled_stats = []
    inventory_digest = hashlib.sha256()

    for index, stem in enumerate(stems, 1):
        point_path = point_files[stem]
        dino_path = dino_files[stem]
        correspondence_path = correspondence_files[stem]
        point_count, point_format, dimensions = _point_header(point_path)
        dino_shapes, _ = _tensor_schema(dino_path)
        correspondence_shapes, correspondence_metadata = _tensor_schema(
            correspondence_path
        )
        feature_shape = dino_shapes.get("feature")
        valid_shape = correspondence_shapes.get("valid")
        if not feature_shape or len(feature_shape) != 2:
            raise ValueError(f"{dino_path}: expected 2-D 'feature' tensor")
        if not valid_shape or len(valid_shape) != 1:
            raise ValueError(f"{correspondence_path}: expected 1-D 'valid' tensor")
        aligned = point_count == int(valid_shape[0])
        if not aligned:
            raise ValueError(
                f"{stem}: LAS points={point_count}, correspondence={valid_shape[0]}"
            )

        sizes = (point_path.stat().st_size, dino_path.stat().st_size,
                 correspondence_path.stat().st_size)
        signature = (
            f"{stem}|{sizes}|{point_count}|{feature_shape}|{valid_shape}|"
            f"{correspondence_metadata.get('schema', '')}\n"
        )
        inventory_digest.update(signature.encode("utf-8"))
        total_points += point_count
        dino_dims[int(feature_shape[1])] += 1
        point_formats[point_format] += 1

        hashes = {"point_sha256": "", "dino_sha256": "", "mapping_sha256": ""}
        if HASH_ALL_FILES or stem in selected:
            hashes = {
                "point_sha256": _sha256(point_path),
                "dino_sha256": _sha256(dino_path),
                "mapping_sha256": _sha256(correspondence_path),
            }
        if stem in selected:
            sampled_stats.append(_audit_selected(point_path, correspondence_path))

        rows.append({
            "stem": stem,
            "point_file": point_path.name,
            "point_bytes": sizes[0],
            "point_count": point_count,
            "point_format": point_format,
            "has_intensity": "intensity" in dimensions,
            "has_return_number": "return_number" in dimensions,
            "has_number_of_returns": "number_of_returns" in dimensions,
            "dino_file": dino_path.name,
            "dino_bytes": sizes[1],
            "dino_patch_count": int(feature_shape[0]),
            "dino_dim": int(feature_shape[1]),
            "correspondence_file": correspondence_path.name,
            "correspondence_bytes": sizes[2],
            "correspondence_count": int(valid_shape[0]),
            "correspondence_schema": correspondence_metadata.get("schema", ""),
            "content_audit_sample": stem in selected,
            **hashes,
        })
        if index % 1000 == 0 or index == len(stems):
            print(f"Scanned {index:,}/{len(stems):,} aligned tiles")

    sample_points = sum(item["points"] for item in sampled_stats)
    return_valid = sum(item["return_valid"] for item in sampled_stats)
    mapping_points = sum(item["correspondence_points"] for item in sampled_stats)
    mapping_valid = sum(item["correspondence_valid"] for item in sampled_stats)
    observability = [
        item["observability_mean"] for item in sampled_stats
        if item["observability_mean"] is not None
    ]
    summary = {
        "schema": "cvrl_section45_swiss100_manifest_v1",
        "data_root": str(DATA_ROOT),
        "seed": SEED,
        "tile_count": len(stems),
        "total_raw_points": total_points,
        "dino_dimensions": {str(key): value for key, value in sorted(dino_dims.items())},
        "point_formats": {str(key): value for key, value in sorted(point_formats.items())},
        "inventory_sha256": inventory_digest.hexdigest(),
        "hash_policy": "all files" if HASH_ALL_FILES else "deterministic audit subset",
        "audit_tile_count": len(sampled_stats),
        "audit_point_count": sample_points,
        "return_valid_ratio": return_valid / max(sample_points, 1),
        "correspondence_valid_ratio": mapping_valid / max(mapping_points, 1),
        "mean_tile_observability": float(np.mean(observability)) if observability else None,
        "files": rows,
    }
    return summary


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest()
    json_path = OUTPUT_ROOT / "swiss100_pretrain_manifest.json"
    csv_path = OUTPUT_ROOT / "swiss100_pretrain_manifest.csv"
    stem_path = OUTPUT_ROOT / "swiss100_pretrain_stems.txt"
    json_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    rows = manifest["files"]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    stem_path.write_text("\n".join(row["stem"] for row in rows) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in manifest.items() if key != "files"},
                     indent=2))
    print(json_path)
    print(csv_path)
    print(stem_path)


if __name__ == "__main__":
    main()
