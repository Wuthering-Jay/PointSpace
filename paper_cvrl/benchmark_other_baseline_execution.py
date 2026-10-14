"""Screen execution-only optimizations for the remaining Section 4.5 baselines.

All scenarios preserve the pretext objective and training data.  The generated
configs retain CacheCleaner so the benchmark follows the long-run training
policy used by the project.  Constants are intentionally editable for IDE use.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pointspace.utils.config import Config


MAX_TRAIN_STEPS = 8
NUM_WORKERS = 2
RESULT_ROOT = PROJECT_ROOT / "paper_cvrl" / "results" / "section_45"
CONFIG_ROOT = RESULT_ROOT / "other_execution_configs"
OUTPUT = RESULT_ROOT / "other_execution_benchmark.json"

SOURCES = {
    "point_m2ae": "configs/baselines/swiss/pretrain-point-m2ae-litept-v1m4-100.py",
    "msc": "configs/baselines/swiss/pretrain-msc-litept-v1m4-100.py",
    "d_ditr": "configs/baselines/swiss/pretrain-d-ditr-litept-v1m4-100.py",
}

SCENARIOS = {
    "point_m2ae_baseline": ("point_m2ae", {}),
    "point_m2ae_no_decoder_checkpoint": (
        "point_m2ae", {"checkpoint_decoder": False}),
    "point_m2ae_no_checkpoints": (
        "point_m2ae", {
            "checkpoint_decoder": False,
            "backbone.checkpoint_mlp": False,
        }),
    "msc_chunk256": ("msc", {"nce_chunk_size": 256}),
    "msc_chunk1024": ("msc", {"nce_chunk_size": 1024}),
    "msc_chunk2048": ("msc", {"nce_chunk_size": 2048}),
    "msc_chunk1024_no_mlp_checkpoint": (
        "msc", {
            "nce_chunk_size": 1024,
            "backbone.checkpoint_mlp": False,
        }),
    "d_ditr_repeated_norm_chunk4096": (
        "d_ditr", {
            "normalize_teacher_once": False,
            "loss_chunk_size": 4096,
            "checkpoint_loss": True,
        }),
    "d_ditr_shared_norm_chunk4096": (
        "d_ditr", {
            "normalize_teacher_once": True,
            "loss_chunk_size": 4096,
            "checkpoint_loss": True,
        }),
    "d_ditr_shared_norm_chunk16384": (
        "d_ditr", {
            "normalize_teacher_once": True,
            "loss_chunk_size": 16384,
            "checkpoint_loss": True,
        }),
    "d_ditr_no_loss_checkpoint": (
        "d_ditr", {
            "normalize_teacher_once": True,
            "loss_chunk_size": 16384,
            "checkpoint_loss": False,
        }),
    "d_ditr_no_checkpoints": (
        "d_ditr", {
            "normalize_teacher_once": True,
            "loss_chunk_size": 16384,
            "checkpoint_loss": False,
            "backbone.checkpoint_mlp": False,
        }),
}


def _set_nested(mapping, dotted_key, value):
    parts = dotted_key.split(".")
    target = mapping
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value


def _write_config(name, method, overrides):
    config = Config.fromfile(str(PROJECT_ROOT / SOURCES[method]))
    for key, value in overrides.items():
        _set_nested(config.model, key, value)
    config.max_train_steps = MAX_TRAIN_STEPS
    config.save_path = f"exp/baselines/other_execution/{name}"
    config.weight = None
    config.resume = False
    config.evaluate = False
    config.test_only = False
    config.num_worker = NUM_WORKERS
    config.persistent_workers = NUM_WORKERS > 0
    config.hooks = [
        dict(type="RuntimeInfoHook"),
        dict(type="ModelHook"),
        dict(type="IterationTimer", warmup_iter=1),
        dict(type="InformationWriter", interval=1),
        dict(type="CacheCleaner", time_multiplier=2.5,
             step_clean_interval=200),
    ]
    for key in (
        "data_root", "train_pointcloud_path", "dino_feature_path",
        "audit_pointcloud_path", "feature_output_dir", "feature_index_dir",
    ):
        if hasattr(config, key):
            setattr(config, key, getattr(config, key).replace("\\", "/"))
    path = CONFIG_ROOT / f"{name}.py"
    config.dump(str(path))
    return path


def main():
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    report = {"max_train_steps": MAX_TRAIN_STEPS, "scenarios": {}}
    for name, (method, overrides) in SCENARIOS.items():
        config_path = _write_config(name, method, overrides)
        print(f"[{name}]", flush=True)
        started = time.perf_counter()
        process = subprocess.run(
            [sys.executable, "tools/train.py", "--config-file", str(config_path)],
            cwd=PROJECT_ROOT,
            env={**os.environ, "PYTHONUTF8": "1"},
            check=False,
        )
        report["scenarios"][name] = {
            "method": method,
            "overrides": overrides,
            "return_code": process.returncode,
            "wall_seconds": time.perf_counter() - started,
            "log": f"exp/baselines/other_execution/{name}/train.log",
        }
        OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
