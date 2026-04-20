import runpy
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pointspace.writers.las_writer import LASWriter


def test_stage1_validation_pipeline_is_deterministic():
    cfg = runpy.run_path(ROOT / "configs" / "dales" / "ezsp" / "stage1_partition.py")
    val_cfg = cfg["data"]["val"]

    assert val_cfg["loop"] == 1
    assert [t["type"] for t in val_cfg["transform"]] == [
        "ZPercentileCenterShift",
        "SaveNodeIndex",
        "Copy",
        "GridSampling3D",
    ]


def test_las_writer_persists_ezsp_partition_outputs(tmp_path):
    writer = LASWriter(save_dir=str(tmp_path), source_dir=None, compressed=False)
    coord = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    oracle_pred = np.array([1, 2, 3], dtype=np.int32)
    superpoint_level_1 = np.array([0, 0, 1], dtype=np.int32)
    superpoint_level_2 = np.array([0, 0, 0], dtype=np.int32)

    out_path = writer.write(
        "ezsp_partition_scene",
        coord=coord,
        oracle_pred=oracle_pred,
        superpoint_level_1=superpoint_level_1,
        superpoint_level_2=superpoint_level_2,
    )

    import laspy

    las = laspy.read(out_path)
    dim_names = set(las.point_format.dimension_names)

    assert "oracle_pred" in dim_names
    assert "superpoint_level_1" in dim_names
    assert "superpoint_level_2" in dim_names
    assert np.array_equal(np.asarray(las.oracle_pred), oracle_pred)
    assert np.array_equal(np.asarray(las.superpoint_level_1), superpoint_level_1)
    assert np.array_equal(np.asarray(las.superpoint_level_2), superpoint_level_2)
