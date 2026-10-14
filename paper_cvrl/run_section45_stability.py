"""Run short, trainer-faithful stability experiments for Section 4.5.

Resolved standalone diagnostic configs are written to the results directory,
then launched in isolated subprocesses so CUDA memory, loggers and EMA state do
not leak between methods.  Change the constants below when a longer diagnostic
is required; no command-line arguments are necessary.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pointspace.utils.config import Config


PYTHON = Path(sys.executable)
MAX_TRAIN_STEPS = 64
NUM_WORKERS = 2
SOURCE_CONFIGS = {
    "point-m2ae": "configs/baselines/swiss/pretrain-point-m2ae-litept-v1m4-100.py",
    "msc": "configs/baselines/swiss/pretrain-msc-litept-v1m4-100.py",
    "sonata": "configs/baselines/swiss/pretrain-sonata-litept-v1m4-100.py",
    "d-ditr": "configs/baselines/swiss/pretrain-d-ditr-litept-v1m4-100.py",
    "concerto": "configs/baselines/swiss/pretrain-concerto-litept-v1m4-100.py",
}
RESULT_ROOT = PROJECT_ROOT / "paper_cvrl" / "results" / "section_45"
CONFIG_ROOT = RESULT_ROOT / "stability_configs"
EXP_ROOT = PROJECT_ROOT / "exp" / "baselines" / "stability"
OUTPUT = RESULT_ROOT / "short_stability_summary.json"


def _diagnostic_config(name: str, source: str) -> Path:
    config = Config.fromfile(str(PROJECT_ROOT / source))
    config.epoch = 20
    config.max_train_steps = MAX_TRAIN_STEPS
    # Keep the dumped Python config portable and avoid Windows backslash
    # escapes (for example ``\b`` in ``baselines``).
    config.save_path = str(
        (EXP_ROOT / name).relative_to(PROJECT_ROOT)
    ).replace("\\", "/")
    config.weight = None
    config.resume = False
    config.evaluate = False
    config.test_only = False
    config.num_worker = NUM_WORKERS
    config.persistent_workers = NUM_WORKERS > 0
    # Config.dump writes Python string literals.  Forward slashes avoid
    # accidental Windows escapes such as ``\f`` and ``\v`` in the resolved
    # diagnostic artifact while remaining valid on Windows.
    for key in (
        "data_root",
        "train_pointcloud_path",
        "dino_feature_path",
        "audit_pointcloud_path",
        "feature_output_dir",
        "feature_index_dir",
    ):
        if hasattr(config, key):
            setattr(config, key, getattr(config, key).replace("\\", "/"))
    config.hooks = [
        dict(type="RuntimeInfoHook"),
        dict(type="ModelHook"),
        dict(type="IterationTimer", warmup_iter=2),
        dict(type="InformationWriter", interval=4),
        dict(type="CacheCleaner", time_multiplier=2.5, step_clean_interval=32),
    ]
    path = CONFIG_ROOT / f"{name}.py"
    config.dump(str(path))
    return path


def _parse_log(path: Path) -> dict:
    losses = []
    records = []
    pattern = re.compile(r"Train:.*?loss:\s*([-+0-9.eE]+)(.*?)(?:Lr:\s*([-+0-9.eE]+))?$")
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.search(line)
        if not match:
            continue
        loss = float(match.group(1))
        losses.append(loss)
        records.append(line)
    if not losses:
        return {"logged_steps": 0, "finite": False, "records": []}
    values = np.asarray(losses, dtype=np.float64)
    x = np.arange(len(values), dtype=np.float64)
    slope = float(np.polyfit(x, values, 1)[0]) if len(values) > 1 else 0.0
    return {
        "logged_steps": len(losses),
        "finite": bool(np.isfinite(values).all()),
        "first_loss": float(values[0]),
        "last_loss": float(values[-1]),
        "minimum_loss": float(values.min()),
        "maximum_loss": float(values.max()),
        "relative_change": float((values[-1] - values[0]) / max(abs(values[0]), 1e-12)),
        "linear_slope_per_log": slope,
        "records": records,
    }


def main() -> None:
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    EXP_ROOT.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "cvrl_section45_short_stability_v1",
        "max_train_steps": MAX_TRAIN_STEPS,
        "optimizer_updates": MAX_TRAIN_STEPS // 4,
        "methods": {},
    }
    for name, source in SOURCE_CONFIGS.items():
        config_path = _diagnostic_config(name, source)
        print(f"[{name}] {config_path}", flush=True)
        start = time.perf_counter()
        process = subprocess.run(
            [str(PYTHON), "tools/train.py", "--config-file", str(config_path)],
            cwd=PROJECT_ROOT,
            env={**os.environ, "PYTHONUTF8": "1"},
            check=False,
        )
        log_path = EXP_ROOT / name / "train.log"
        result = {
            "source_config": source,
            "resolved_config": str(config_path.relative_to(PROJECT_ROOT)),
            "return_code": process.returncode,
            "wall_seconds": time.perf_counter() - start,
            "log": str(log_path.relative_to(PROJECT_ROOT)),
        }
        if log_path.exists():
            result.update(_parse_log(log_path))
        report["methods"][name] = result
        OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
        if process.returncode != 0:
            print(f"{name} failed; continuing so all outcomes are recorded", flush=True)
    print(OUTPUT)


if __name__ == "__main__":
    main()
