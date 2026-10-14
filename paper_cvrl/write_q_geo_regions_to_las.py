"""Write Swiss-Audit q_geo partitions into LAS classification.

The source point clouds are never modified.  Each output keeps every original
dimension, copies the original classification into an extra byte dimension,
and uses the visible classification field for easy inspection in LAS viewers:

    1 = P_inv, 2 = P_amb, 3 = P_vis

The continuous score and zero-based region are also retained as ``q_geo`` and
``q_geo_region`` extra dimensions.  Configuration is intentionally module-level
for direct execution from an IDE.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import laspy
import numpy as np
from safetensors import safe_open
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.las_io import write_standard_las


AUDIT_ROOT = Path(r"G:\data\swiss\joint_tiles_100\Swiss-Audit")
INPUT_DIR = AUDIT_ROOT / "pointcloud"
VISIBILITY_DIR = AUDIT_ROOT / "visibility"
OUTPUT_DIR = AUDIT_ROOT / "q_geo_pointcloud"
OUTPUT_EXTENSION = ".laz"
NUM_WORKERS = 4
OVERWRITE = False
MAX_FILES = None

REGION_TO_CLASSIFICATION = np.asarray([1, 2, 3], dtype=np.uint8)
CLASSIFICATION_NAMES = {1: "P_inv", 2: "P_amb", 3: "P_vis"}


def _set_extra_dimension(las, name, type_name, values, description):
    if name not in set(las.point_format.extra_dimension_names):
        las.add_extra_dim(
            laspy.ExtraBytesParams(
                name=name,
                type=type_name,
                description=description,
            )
        )
    las[name] = values


def _convert_one(source_path: Path):
    output_path = OUTPUT_DIR / f"{source_path.stem}{OUTPUT_EXTENSION}"
    if output_path.exists() and not OVERWRITE:
        with laspy.open(output_path) as reader:
            point_count = int(reader.header.point_count)
        return source_path.name, "skipped", point_count
    sidecar_path = VISIBILITY_DIR / f"{source_path.stem}.safetensors"
    if not sidecar_path.is_file():
        raise FileNotFoundError(f"Missing q_geo sidecar: {sidecar_path}")
    las = laspy.read(source_path)
    with safe_open(str(sidecar_path), framework="np") as handle:
        q_geo = np.asarray(handle.get_tensor("q_geo"), dtype=np.float32)
        region = np.asarray(handle.get_tensor("region"), dtype=np.int64)
    if q_geo.shape != (len(las.points),) or region.shape != (len(las.points),):
        raise ValueError(
            f"{source_path.name}: q_geo/region length does not match LAS points"
        )
    if region.size and (region.min() < 0 or region.max() > 2):
        raise ValueError(f"{sidecar_path}: expected region values in [0, 2]")
    original_classification = np.asarray(las.classification, dtype=np.uint8).copy()
    _set_extra_dimension(
        las,
        "original_classification",
        np.uint8,
        original_classification,
        "Original LAS classification",
    )
    _set_extra_dimension(
        las,
        "q_geo",
        np.float32,
        q_geo,
        "Pure geometry visibility [0,1]",
    )
    _set_extra_dimension(
        las,
        "q_geo_region",
        np.uint8,
        region.astype(np.uint8),
        "0=P_inv, 1=P_amb, 2=P_vis",
    )
    las.classification = REGION_TO_CLASSIFICATION[region]
    temporary_path = output_path.with_name(
        f".{output_path.stem}.{os.getpid()}.tmp{OUTPUT_EXTENSION}"
    )
    try:
        write_standard_las(las, temporary_path, warn_on_copc=False)
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return source_path.name, "written", len(las.points)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sources = sorted((*INPUT_DIR.glob("*.las"), *INPUT_DIR.glob("*.laz")))
    if MAX_FILES is not None:
        sources = sources[: int(MAX_FILES)]
    if not sources:
        raise FileNotFoundError(f"No LAS/LAZ files under {INPUT_DIR}")
    written = skipped = points = 0
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(_convert_one, path): path for path in sources}
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Write q_geo LAS", unit="file"
        ):
            _, status, count = future.result()
            written += status == "written"
            skipped += status == "skipped"
            points += count
    summary = {
        "input_dir": str(INPUT_DIR),
        "visibility_dir": str(VISIBILITY_DIR),
        "output_dir": str(OUTPUT_DIR),
        "files": len(sources),
        "written": int(written),
        "skipped": int(skipped),
        "points": int(points),
        "classification_mapping": CLASSIFICATION_NAMES,
        "extra_dimensions": [
            "original_classification",
            "q_geo",
            "q_geo_region",
        ],
    }
    (OUTPUT_DIR / "q_geo_classification.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logging.info(
        "q_geo LAS ready: files=%d, written=%d, skipped=%d, points=%d",
        len(sources), written, skipped, points,
    )


if __name__ == "__main__":
    main()
