"""Export full-resolution frozen features for the single Fig. 8 Swiss scene.

Run directly in the pointspace environment. Existing exports are kept unless
OVERWRITE is set. This uses the same checkpoints and test transforms as the
Swiss-100 audit; only the test directory and point subset are changed.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import laspy
import torch
from safetensors.torch import save_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIT_ROOT = Path(r"G:\data\swiss\joint_tiles_100\Swiss-Audit")
SCENE_STEM = "swisssurface3d_2024_2679-1248_2056_5728.copc_0033"
FOREST_STEM = "swisssurface3d_2023_2616-1179_2056_5728_0036"
SCENE_STEMS = (SCENE_STEM, FOREST_STEM)
OUTPUT_ROOT = PROJECT_ROOT / "paper_cvrl/results/section_44/fig8_data"
OVERWRITE = False

EXPORTS = (
    ("vpsd", "configs/vpsd/pretrain-vpsd-v1m1-litept-v1m4-swiss-100.py", "visual"),
    ("rrcl", "configs/rrcl/pretrain-rrcl-v1m1-litept-v1m4-swiss-100.py", "backbone"),
    ("cvrl", "configs/cvrl/audit-cvrl-v1m1-litept-v1m4-swiss-100.py", "full"),
)


def export() -> None:
    scene_dir = OUTPUT_ROOT / "pointcloud"
    scene_dir.mkdir(parents=True, exist_ok=True)
    index_dir = OUTPUT_ROOT / "all_point_index"
    index_dir.mkdir(parents=True, exist_ok=True)
    for scene_stem in SCENE_STEMS:
        source = AUDIT_ROOT / "pointcloud" / f"{scene_stem}.las"
        if not source.exists():
            raise FileNotFoundError(source)
        scene_file = scene_dir / source.name
        if not scene_file.exists():
            shutil.copy2(source, scene_file)
        # The config parser treats `None` as a literal string in --options.
        # Supply an explicit all-point index sidecar instead.
        index_file = index_dir / f"{scene_stem}.safetensors"
        if not index_file.exists():
            point_count = laspy.open(scene_file).header.point_count
            save_file({"audit_index": torch.arange(point_count, dtype=torch.int64)},
                      str(index_file))
        # A per-scene directory avoids accidentally exporting every audit tile.
        single_scene_dir = OUTPUT_ROOT / "input" / scene_stem
        single_scene_dir.mkdir(parents=True, exist_ok=True)
        single_scene_file = single_scene_dir / scene_file.name
        if not single_scene_file.exists():
            shutil.copy2(scene_file, single_scene_file)
        for name, config, feature_source in EXPORTS:
            output_dir = OUTPUT_ROOT / "features" / name
            output_file = output_dir / f"{scene_stem}.safetensors"
            if output_file.exists() and not OVERWRITE:
                print(f"[{name} / {scene_stem}] kept {output_file}", flush=True)
                continue
            output_dir.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                str(PROJECT_ROOT / "tools/test.py"),
                "--config-file", str(PROJECT_ROOT / config),
                "--options",
                "test_only=True",
                f"data.test.data_path={single_scene_dir}",
                f"feature_output_dir={output_dir}",
                f"feature_source={feature_source}",
                f"feature_index_dir={index_dir}",
                f"feature_overwrite={OVERWRITE}",
                "batch_size_test=1",
            ]
            print(f"[{name} / {scene_stem}] exporting full tile", flush=True)
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)
            if not output_file.exists():
                raise RuntimeError(f"[{name}] no output at {output_file}")


if __name__ == "__main__":
    export()
