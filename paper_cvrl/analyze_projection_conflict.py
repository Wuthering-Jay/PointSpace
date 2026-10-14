"""Post-hoc patch semantic conflict for geometry-only visibility regions.

The visible-surface class proxy is the LAS-class mode among valid point-image
correspondences in each DINO patch. LAS classes are used ONLY in this audit,
never as a pretraining target or model input. Settings are module-level.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from safetensors import safe_open


AUDIT_ROOT = Path(r"G:\data\swiss\joint_tiles_100\Swiss-Audit")
OUTPUT_JSON = AUDIT_ROOT / "analysis" / "section_43_projection_conflict.json"
OUTPUT_CSV = Path(__file__).resolve().parent / "results/section_43/projection_conflict.csv"
MIN_ANCHOR_POINTS = 5
MIN_ANCHOR_PURITY = 0.70
N_BOOTSTRAP = 2000
RANDOM_SEED = 43
NUM_CLASS_SLOTS = 28
NUM_REGIONS = 3
REGION_NAMES = ("P_inv", "P_amb", "P_vis")


def _read(path, keys):
    with safe_open(str(path), framework="np") as handle:
        return {key: np.asarray(handle.get_tensor(key)) for key in keys}


def _rate(counts, region, class_id=None):
    relevant = counts[:, region] if class_id is None else counts[class_id, region]
    eligible, mismatch = relevant[..., 0].sum(), relevant[..., 1].sum()
    return float(mismatch / eligible) if eligible else None, int(eligible)


def _bootstrap(blocks, region, class_id=None):
    selected = (
        blocks[:, :, region, :].sum(axis=1)
        if class_id is None else blocks[:, class_id, region, :]
    )
    rng = np.random.default_rng(RANDOM_SEED + region + (class_id or 0) * 3)
    sample = rng.integers(0, len(blocks), size=(N_BOOTSTRAP, len(blocks)))
    totals = selected[sample].sum(axis=1)
    rates = np.divide(
        totals[:, 1], totals[:, 0],
        out=np.full(N_BOOTSTRAP, np.nan, dtype=np.float64),
        where=totals[:, 0] > 0,
    )
    valid = rates[np.isfinite(rates)]
    if len(valid) < N_BOOTSTRAP // 2:
        return None
    return [float(np.quantile(valid, 0.025)), float(np.quantile(valid, 0.975))]


def _bootstrap_gap(blocks, class_id=None):
    selected = (
        blocks.sum(axis=1) if class_id is None else blocks[:, class_id]
    )
    rng = np.random.default_rng(RANDOM_SEED + 1000 + (class_id or 0))
    sample = rng.integers(0, len(blocks), size=(N_BOOTSTRAP, len(blocks)))
    totals = selected[sample].sum(axis=1)
    inv = np.divide(
        totals[:, 0, 1], totals[:, 0, 0],
        out=np.full(N_BOOTSTRAP, np.nan), where=totals[:, 0, 0] > 0,
    )
    vis = np.divide(
        totals[:, 2, 1], totals[:, 2, 0],
        out=np.full(N_BOOTSTRAP, np.nan), where=totals[:, 2, 0] > 0,
    )
    gaps = (inv - vis)[np.isfinite(inv - vis)]
    if len(gaps) < N_BOOTSTRAP // 2:
        return None
    return [float(np.quantile(gaps, 0.025)), float(np.quantile(gaps, 0.975))]


def run():
    with (AUDIT_ROOT / "audit_manifest.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as stream:
        manifest = list(csv.DictReader(stream))
    block_counts = defaultdict(
        lambda: np.zeros((NUM_CLASS_SLOTS, NUM_REGIONS, 2), dtype=np.int64)
    )
    tile_summaries = []
    total_audit = 0
    audit_class_region = np.zeros(
        (NUM_CLASS_SLOTS, NUM_REGIONS), dtype=np.int64
    )
    for tile_number, row in enumerate(manifest, 1):
        stem = row["stem"]
        visibility = _read(
            AUDIT_ROOT / "visibility" / f"{stem}.safetensors",
            ("classification", "region", "audit_index"),
        )
        with safe_open(
            str(AUDIT_ROOT / "correspondence" / f"{stem}.safetensors"),
            framework="np",
        ) as handle:
            correspondence = {
                key: np.asarray(handle.get_tensor(key))
                for key in ("patch_index", "pixel_coord", "valid")
            }
            metadata = handle.metadata() or {}
        classification = visibility["classification"].astype(np.int32)
        region = visibility["region"].astype(np.int8)
        audit_index = visibility["audit_index"].astype(np.int64)
        patch = correspondence["patch_index"].astype(np.int32)
        pixel = correspondence["pixel_coord"].astype(np.int32)
        patch_size = int(metadata["patch_size"])
        # Metadata records the feature grid as [width, height], while
        # pixel_coord stores [row, col]. The first grid entry is therefore
        # the row-major stride, including non-square orthophoto crops.
        grid_cols, grid_rows = json.loads(metadata["feature_grid_size"])
        projected_patch = (
            (pixel[:, 0] // patch_size) * grid_cols
            + pixel[:, 1] // patch_size
        )
        if not np.array_equal(projected_patch[patch >= 0], patch[patch >= 0]):
            raise ValueError(f"Pixel/patch mapping mismatch for {stem}")
        valid = correspondence["valid"].astype(bool)
        valid &= (patch >= 0) & (classification >= 0)
        valid &= classification < NUM_CLASS_SLOTS
        patch_count = int(grid_rows * grid_cols)
        if patch_count == 0:
            continue
        flat = patch[valid] * NUM_CLASS_SLOTS + classification[valid]
        patch_class = np.bincount(
            flat, minlength=patch_count * NUM_CLASS_SLOTS
        ).reshape(patch_count, NUM_CLASS_SLOTS)
        anchor_support = patch_class.sum(axis=1)
        anchor_class = patch_class.argmax(axis=1)
        anchor_purity = np.divide(
            patch_class.max(axis=1), anchor_support,
            out=np.zeros(patch_count, dtype=np.float64),
            where=anchor_support > 0,
        )
        eligible_patch = (
            (anchor_support >= MIN_ANCHOR_POINTS)
            & (anchor_purity >= MIN_ANCHOR_PURITY)
        )
        # Hidden points intentionally have patch_index=-1 in the training
        # correspondence. Their pixel_coord is still available for this
        # strictly post-hoc projection audit.
        audit_patch = projected_patch[audit_index]
        audit_class = classification[audit_index]
        audit_region = region[audit_index]
        counted = (
            (audit_class >= 0) & (audit_class < NUM_CLASS_SLOTS)
            & (audit_region >= 0) & (audit_region < NUM_REGIONS)
        )
        np.add.at(
            audit_class_region,
            (audit_class[counted], audit_region[counted]),
            1,
        )
        eligible = (
            (audit_patch >= 0)
            & (audit_patch < patch_count)
            & (audit_class >= 0)
            & (audit_class < NUM_CLASS_SLOTS)
            & (audit_region >= 0)
            & (audit_region < NUM_REGIONS)
        )
        eligible[eligible] &= eligible_patch[audit_patch[eligible]]
        target = audit_class[eligible]
        target_region = audit_region[eligible]
        conflict = target != anchor_class[audit_patch[eligible]]
        counts = block_counts[row["source_block"]]
        np.add.at(counts[..., 0], (target, target_region), 1)
        np.add.at(
            counts[..., 1],
            (target[conflict], target_region[conflict]),
            1,
        )
        total_audit += len(audit_index)
        tile_summaries.append(
            {
                "stem": stem,
                "source_block": row["source_block"],
                "environment": row["environment"],
                "audit_points": len(audit_index),
                "eligible_points": int(eligible.sum()),
                "P_inv_eligible": int(np.sum(target_region == 0)),
                "P_inv_conflict": int(np.sum(conflict & (target_region == 0))),
                "P_inv_ground_eligible": int(
                    np.sum((target == 2) & (target_region == 0))
                ),
                "P_inv_ground_conflict": int(
                    np.sum(conflict & (target == 2) & (target_region == 0))
                ),
                "P_inv_facade_eligible": int(
                    np.sum((target == 26) & (target_region == 0))
                ),
                "P_inv_facade_conflict": int(
                    np.sum(conflict & (target == 26) & (target_region == 0))
                ),
            }
        )
        if tile_number % 100 == 0:
            print(f"Analyzed {tile_number}/{len(manifest)} tiles", flush=True)

    blocks = np.stack(list(block_counts.values()))
    pooled = blocks.sum(axis=0)
    summary = {}
    for class_id in (None, *range(NUM_CLASS_SLOTS)):
        key = "all" if class_id is None else str(class_id)
        summary[key] = {}
        for region, name in enumerate(REGION_NAMES):
            rate, eligible = _rate(pooled, region, class_id)
            if eligible:
                total = (
                    int(audit_class_region[:, region].sum())
                    if class_id is None
                    else int(audit_class_region[class_id, region])
                )
                summary[key][name] = {
                    "eligible_points": eligible,
                    "audit_points": total,
                    "eligible_fraction": float(eligible / total),
                    "conflict_rate": rate,
                    "ci95": _bootstrap(blocks, region, class_id),
                }
        if "P_inv" in summary[key] and "P_vis" in summary[key]:
            summary[key]["P_inv_minus_P_vis"] = {
                "point_estimate": (
                    summary[key]["P_inv"]["conflict_rate"]
                    - summary[key]["P_vis"]["conflict_rate"]
                ),
                "ci95": _bootstrap_gap(blocks, class_id),
            }

    result = {
        "schema": "cvrl_section_43_projection_conflict_v1",
        "definition": (
            "Post-hoc LAS class differs from modal LAS class of valid image-"
            "corresponding points in the same DINO patch"
        ),
        "not_dino_prediction": True,
        "anchor_min_points": MIN_ANCHOR_POINTS,
        "anchor_min_purity": MIN_ANCHOR_PURITY,
        "audit_tiles": len(tile_summaries),
        "audit_points": total_audit,
        "eligible_points": int(pooled[..., 0].sum()),
        "source_blocks": len(blocks),
        "summary": summary,
        "tiles": tile_summaries,
    }
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ("class_id", "region", "eligible_points", "conflict_rate", "ci95_low", "ci95_high")
        )
        for class_id, regions in summary.items():
            for region, metrics in regions.items():
                if region == "P_inv_minus_P_vis":
                    continue
                ci = metrics["ci95"] or (None, None)
                writer.writerow(
                    (class_id, region, metrics["eligible_points"],
                     metrics["conflict_rate"], *ci)
                )
    print(
        f"Saved {OUTPUT_JSON}; eligible {result['eligible_points']:,}/"
        f"{total_audit:,} audit points",
        flush=True,
    )
    return result


if __name__ == "__main__":
    run()
