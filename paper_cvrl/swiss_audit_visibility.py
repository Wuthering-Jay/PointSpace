"""Pure-geometry visibility partitions and Swiss-Audit statistics.

Unlike the production visibility used while building image correspondences,
``q_geo`` never reads return number, number of returns, intensity or semantic
class. This prevents RRCL information from leaking into the definition of the
regions used to test RRCL compensation.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import laspy
import numpy as np
from safetensors.numpy import save_file as save_safetensors
from tqdm import tqdm


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from paper_cvrl.q_geo import (
    P_AMB,
    P_INV,
    P_VIS,
    REGION_NAMES,
    compute_q_geo,
    estimate_height_above_ground,
    partition_q_geo,
)


LOGGER = logging.getLogger("swiss_audit_visibility")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def _json_height_bins(height_bins):
    return [
        "inf" if not np.isfinite(value) else float(value)
        for value in height_bins
    ]



def _load_manifest(path: Path) -> Dict[str, Dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return {row["stem"]: row for row in csv.DictReader(stream)}


def _counts_by(values: np.ndarray, region: np.ndarray) -> Dict[str, Dict]:
    result = {}
    for value in np.unique(values):
        mask = values == value
        counts = np.bincount(region[mask], minlength=3)
        total = int(mask.sum())
        result[str(value)] = {
            "points": total,
            **{
                REGION_NAMES[index]: int(counts[index])
                for index in range(3)
            },
            **{
                f"{REGION_NAMES[index]}_ratio": (
                    float(counts[index] / total) if total else 0.0
                )
                for index in range(3)
            },
        }
    return result


def _stratified_audit_index(
    classification: np.ndarray,
    region: np.ndarray,
    maximum: int,
) -> np.ndarray:
    """Deterministically preserve rare class/visibility combinations."""
    if maximum <= 0:
        raise ValueError("audit_points_per_tile must be positive")
    total = classification.size
    if total <= maximum:
        return np.arange(total, dtype=np.int64)
    strata = classification.astype(np.int64) * 3 + region.astype(np.int64)
    unique, counts = np.unique(strata, return_counts=True)
    # Give every non-empty stratum an equal base quota, then fill remaining
    # capacity uniformly from all points not selected by that first pass.
    base = max(1, maximum // max(unique.size, 1))
    parts = []
    for value, count in zip(unique, counts):
        index = np.flatnonzero(strata == value)
        take = min(base, int(count))
        position = np.linspace(0, index.size - 1, take).round().astype(np.int64)
        parts.append(index[position])
    selected = np.unique(np.concatenate(parts))
    if selected.size < maximum:
        remaining = np.setdiff1d(
            np.arange(total, dtype=np.int64), selected, assume_unique=True
        )
        take = min(maximum - selected.size, remaining.size)
        position = np.linspace(0, remaining.size - 1, take).round().astype(np.int64)
        selected = np.sort(np.r_[selected, remaining[position]])
    return selected[:maximum].astype(np.int64, copy=False)


def _process_tile(
    las_path: Path,
    output_path: Path,
    manifest_row: Mapping[str, str],
    invisible_max: float,
    visible_min: float,
    visibility_parameters: Mapping,
    height_bins: Sequence[float],
    audit_points_per_tile: int,
) -> Dict:
    las = laspy.read(las_path)
    points = np.column_stack((las.x, las.y, las.z))
    classification = np.asarray(las.classification, dtype=np.uint8)
    q_geo = compute_q_geo(points, **visibility_parameters)
    region = partition_q_geo(q_geo, invisible_max, visible_min)
    height = estimate_height_above_ground(points, classification)
    height_bin = np.digitize(height, np.asarray(height_bins[1:-1]), right=False).astype(
        np.uint8
    )
    audit_index = _stratified_audit_index(
        classification, region, audit_points_per_tile
    )
    metadata = {
        "schema": "pointspace_q_geo_v1",
        "definition": "pure_geometry_multi_offset_adaptive_splat_zbuffer",
        "return_information_used": "false",
        "invisible_max": str(invisible_max),
        "visible_min": str(visible_min),
        "regions": json.dumps(REGION_NAMES),
        "height_bins": json.dumps(_json_height_bins(height_bins)),
        "source_block": manifest_row.get("source_block", ""),
        "environment": manifest_row.get("environment", "unknown"),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    save_safetensors(
        {
            "q_geo": np.ascontiguousarray(q_geo),
            "region": np.ascontiguousarray(region),
            "height_agl": np.ascontiguousarray(height),
            "height_bin": np.ascontiguousarray(height_bin),
            "classification": np.ascontiguousarray(classification),
            "audit_index": np.ascontiguousarray(audit_index),
        },
        str(temporary),
        metadata,
    )
    os.replace(temporary, output_path)
    region_count = np.bincount(region, minlength=3)
    return {
        "stem": las_path.stem,
        "source_block": manifest_row.get("source_block", ""),
        "environment": manifest_row.get("environment", "unknown"),
        "points": int(points.shape[0]),
        "q_mean": float(q_geo.mean()) if q_geo.size else 0.0,
        "q_p10": float(np.quantile(q_geo, 0.10)) if q_geo.size else 0.0,
        "q_p50": float(np.quantile(q_geo, 0.50)) if q_geo.size else 0.0,
        "q_p90": float(np.quantile(q_geo, 0.90)) if q_geo.size else 0.0,
        "region_count": region_count.tolist(),
        "class_stats": _counts_by(classification, region),
        "height_stats": _counts_by(height_bin, region),
    }


def _merge_nested(rows: Iterable[Dict], key: str) -> Dict[str, Dict]:
    merged: Dict[str, Dict] = {}
    for row in rows:
        for value, stats in row[key].items():
            target = merged.setdefault(
                value, {"points": 0, "P_inv": 0, "P_amb": 0, "P_vis": 0}
            )
            for name in tuple(target):
                target[name] += int(stats[name])
    for stats in merged.values():
        total = max(stats["points"], 1)
        for region_name in REGION_NAMES.values():
            stats[f"{region_name}_ratio"] = stats[region_name] / total
    return merged


def analyze_swiss_audit_visibility(
    audit_root: str | Path,
    invisible_max: float = 0.20,
    visible_min: float = 0.60,
    num_workers: int = 4,
    overwrite: bool = False,
    audit_points_per_tile: int = 4096,
    height_bins=(0.0, 0.5, 2.0, 5.0, 15.0, 30.0, float("inf")),
    **visibility_parameters,
) -> Dict:
    """Compute q_geo sidecars and scene/class/height P_inv summaries."""
    audit_root = Path(audit_root)
    manifest = _load_manifest(audit_root / "audit_manifest.csv")
    las_files = sorted((audit_root / "pointcloud").glob("*.las")) + sorted(
        (audit_root / "pointcloud").glob("*.laz")
    )
    if not las_files:
        raise FileNotFoundError("Swiss-Audit pointcloud directory is empty")

    jobs = []
    cached = []
    for path in las_files:
        output = audit_root / "visibility" / f"{path.stem}.safetensors"
        if output.exists() and not overwrite:
            # Recompute summary from the compact sidecar in a later run is
            # intentionally avoided: overwrite=True guarantees parameters and
            # summaries cannot silently disagree.
            raise FileExistsError(
                f"Visibility output already exists: {output}; set overwrite=True"
            )
        jobs.append((path, output, manifest[path.stem]))

    parameters = {
        "surface_cell_size": visibility_parameters.get("surface_cell_size", "auto"),
        "surface_radius": visibility_parameters.get("surface_radius", "auto"),
        "surface_z_tolerance": visibility_parameters.get(
            "surface_z_tolerance", 0.10
        ),
        "reference_resolution": visibility_parameters.get(
            "reference_resolution", 0.0
        ),
        "surface_oversample": visibility_parameters.get("surface_oversample", 2),
        "surface_num_offsets": visibility_parameters.get("surface_num_offsets", 4),
        "surface_adaptive_radius": visibility_parameters.get(
            "surface_adaptive_radius", True
        ),
        "surface_max_radius_cells": visibility_parameters.get(
            "surface_max_radius_cells", 2
        ),
        "surface_slope_compensation": visibility_parameters.get(
            "surface_slope_compensation", True
        ),
        "surface_max_slope_degrees": visibility_parameters.get(
            "surface_max_slope_degrees", 60.0
        ),
    }
    worker = lambda job: _process_tile(
        *job,
        invisible_max=invisible_max,
        visible_min=visible_min,
        visibility_parameters=parameters,
        height_bins=height_bins,
        audit_points_per_tile=audit_points_per_tile,
    )
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        rows = list(
            tqdm(
                executor.map(worker, jobs),
                total=len(jobs),
                desc="Swiss q_geo",
                unit="tile",
            )
        )

    environment_rows: Dict[str, Dict] = {}
    total_region = np.zeros(3, dtype=np.int64)
    for row in rows:
        count = np.asarray(row["region_count"], dtype=np.int64)
        total_region += count
        target = environment_rows.setdefault(
            row["environment"], {"tiles": 0, "points": 0, "region": np.zeros(3, dtype=np.int64)}
        )
        target["tiles"] += 1
        target["points"] += row["points"]
        target["region"] += count
    environment_stats = {}
    for name, values in environment_rows.items():
        total = max(values["points"], 1)
        environment_stats[name] = {
            "tiles": values["tiles"],
            "points": values["points"],
            **{
                REGION_NAMES[index]: int(values["region"][index])
                for index in range(3)
            },
            **{
                f"{REGION_NAMES[index]}_ratio": float(values["region"][index] / total)
                for index in range(3)
            },
        }
    total_points = int(total_region.sum())
    report = {
        "schema": "pointspace_swiss_audit_visibility_v1",
        "definition": "pure geometry; no return/intensity/class input",
        "thresholds": {
            "P_inv": f"q_geo <= {invisible_max}",
            "P_amb": f"{invisible_max} < q_geo < {visible_min}",
            "P_vis": f"q_geo >= {visible_min}",
        },
        "visibility_parameters": parameters,
        "height_bins_m": _json_height_bins(height_bins),
        "audit_points_per_tile": audit_points_per_tile,
        "tiles": len(rows),
        "points": total_points,
        "region_counts": {
            REGION_NAMES[index]: int(total_region[index]) for index in range(3)
        },
        "region_ratios": {
            REGION_NAMES[index]: float(total_region[index] / max(total_points, 1))
            for index in range(3)
        },
        "environment_stats": environment_stats,
        "class_stats": _merge_nested(rows, "class_stats"),
        "height_stats": _merge_nested(rows, "height_stats"),
        "tile_stats": rows,
    }
    output = audit_root / "analysis" / "q_geo_summary.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info(
        "q_geo ready: %d tiles, %d points; P_inv %.2f%%, P_amb %.2f%%, P_vis %.2f%%",
        len(rows),
        total_points,
        100 * report["region_ratios"]["P_inv"],
        100 * report["region_ratios"]["P_amb"],
        100 * report["region_ratios"]["P_vis"],
    )
    return report


if __name__ == "__main__":
    analyze_swiss_audit_visibility(
        audit_root=r"G:\data\swiss\joint_tiles_100\Swiss-Audit",
        invisible_max=0.20,
        visible_min=0.60,
        num_workers=4,
        overwrite=False,
        audit_points_per_tile=4096,
        surface_cell_size="auto",
        surface_radius="auto",
        surface_z_tolerance=0.10,
        reference_resolution=0.0,
        surface_oversample=2,
        surface_num_offsets=4,
        surface_adaptive_radius=True,
        surface_max_radius_cells=2,
        surface_slope_compensation=True,
        surface_max_slope_degrees=60.0,
    )
