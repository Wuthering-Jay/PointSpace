"""Prepare pure-geometric visibility sidecars for the two 200 m Fig. 6 cases.

Run directly in the IDE. Source LAS, image, and correspondence are read-only.
The 0.18 m splat radius is a qualitative canopy-gap sensitivity setting; it
does not replace the Swiss-Audit parameters used for Fig. 5 or Fig. 6(a).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import laspy
import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from paper_cvrl.q_geo import partition_q_geo
from pointspace.utils.surface_visibility import compute_surface_visibility

from plot_projection_conflict import (
    CASES, CASE_ROOT, CASE_VISIBILITY_ROOT, CASE_VISIBILITY_SCHEMA,
)


VISIBILITY_PARAMETERS = {
    "surface_cell_size": "auto",
    "surface_radius": 0.18,
    "surface_z_tolerance": 0.10,
    "surface_oversample": 2,
    "surface_num_offsets": 4,
    "surface_adaptive_radius": True,
    "surface_max_radius_cells": 2,
    "surface_slope_compensation": True,
    "surface_max_slope_degrees": 60.0,
    "surface_visibility_threshold": 0.60,
    "echo_decay": 0.0,
    "first_return_bonus": 0.0,
}
INVISIBLE_MAX = 0.20
VISIBLE_MIN = 0.60


def prepare_cases():
    CASE_VISIBILITY_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, stem, target_class in CASES:
        source = CASE_ROOT / "pointcloud" / f"{stem}.las"
        correspondence = CASE_ROOT / "correspondence" / f"{stem}.safetensors"
        output = CASE_VISIBILITY_ROOT / f"{stem}.safetensors"
        cloud = laspy.read(source)
        points = np.column_stack((cloud.x, cloud.y, cloud.z))
        classification = np.asarray(cloud.classification)
        with safe_open(str(correspondence), framework="np") as handle:
            correspondence_count = handle.get_slice("pixel_coord").get_shape()[0]
        if correspondence_count != len(points):
            raise ValueError(f"{stem}: correspondence does not match LAS point count")

        result = compute_surface_visibility(
            points, return_number=None, **VISIBILITY_PARAMETERS
        )
        q_geo = np.ascontiguousarray(result.geometric_observability)
        region = np.ascontiguousarray(
            partition_q_geo(q_geo, INVISIBLE_MAX, VISIBLE_MIN)
        )
        metadata = {
            "schema": CASE_VISIBILITY_SCHEMA,
            "source": str(source),
            "source_bytes": str(source.stat().st_size),
            "definition": "pure_geometry_no_return_or_class_input",
            "scope": "qualitative_200m_cases_only_not_panel_a_statistics",
            "parameters": json.dumps(VISIBILITY_PARAMETERS, sort_keys=True),
            "P_inv_max": str(INVISIBLE_MAX),
            "P_vis_min": str(VISIBLE_MIN),
        }
        temporary = output.with_name(f".{output.name}.tmp")
        save_file({"q_geo": q_geo, "region": region}, str(temporary), metadata)
        os.replace(temporary, output)
        target = classification == target_class
        row = {
            "case": label,
            "stem": stem,
            "points": len(points),
            "target_points": int(target.sum()),
            "target_P_inv": float((region[target] == 0).mean()),
            "target_P_amb": float((region[target] == 1).mean()),
            "target_P_vis": float((region[target] == 2).mean()),
            "sidecar": str(output),
        }
        print(json.dumps(row, ensure_ascii=False))
        rows.append(row)
    return rows


if __name__ == "__main__":
    prepare_cases()
