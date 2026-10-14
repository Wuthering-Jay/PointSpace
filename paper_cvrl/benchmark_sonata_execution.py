"""Benchmark Sonata execution-only optimizations through the real trainer.

The scenarios preserve views, losses and prototypes. They vary the stochastic
correspondence sample count and checkpointed prototype-head chunk size.
Constants are intentionally editable for IDE execution.
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


SOURCE = "configs/baselines/swiss/pretrain-sonata-litept-v1m4-100.py"
MAX_TRAIN_STEPS = 8
NUM_WORKERS = 2
SCENARIOS = {
    "cap8192_chunk256": (8192, 256),
    "cap8192_chunk1024": (8192, 1024),
    "cap4096_chunk1024": (4096, 1024),
    "cap4096_chunk2048": (4096, 2048),
    "cap4096_chunk4096": (4096, 4096),
}
RESULT_ROOT = PROJECT_ROOT / "paper_cvrl" / "results" / "section_45"
CONFIG_ROOT = RESULT_ROOT / "sonata_execution_configs"
OUTPUT = RESULT_ROOT / "sonata_execution_benchmark.json"


def _write_config(name: str, matching_cap: int, chunk_size: int) -> Path:
    config = Config.fromfile(str(PROJECT_ROOT / SOURCE))
    config.model.matching_max_pair = matching_cap
    config.model.head_chunk_size = chunk_size
    config.max_train_steps = MAX_TRAIN_STEPS
    config.save_path = f"exp/baselines/sonata_execution/{name}"
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
    ]
    for key in (
        "data_root", "train_pointcloud_path", "audit_pointcloud_path",
        "feature_output_dir", "feature_index_dir",
    ):
        if hasattr(config, key):
            setattr(config, key, getattr(config, key).replace("\\", "/"))
    path = CONFIG_ROOT / f"{name}.py"
    config.dump(str(path))
    return path


def main() -> None:
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    report = {"max_train_steps": MAX_TRAIN_STEPS, "scenarios": {}}
    for name, (matching_cap, chunk_size) in SCENARIOS.items():
        path = _write_config(name, matching_cap, chunk_size)
        print(f"[{name}]", flush=True)
        started = time.perf_counter()
        process = subprocess.run(
            [sys.executable, "tools/train.py", "--config-file", str(path)],
            cwd=PROJECT_ROOT,
            env={**os.environ, "PYTHONUTF8": "1"},
            check=False,
        )
        report["scenarios"][name] = {
            "matching_max_pair": matching_cap,
            "head_chunk_size": chunk_size,
            "return_code": process.returncode,
            "wall_seconds": time.perf_counter() - started,
            "log": f"exp/baselines/sonata_execution/{name}/train.log",
        }
        OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
