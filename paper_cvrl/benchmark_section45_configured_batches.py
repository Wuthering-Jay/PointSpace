"""Validate the actual configured Swiss batch sizes for Section 4.5 methods.

Unlike ``validate_section45_real_batches.py``, this script collates the exact
per-GPU batch size declared by each config.  Sonata and Concerto are also
tested with an 8192 pair cap to decide whether the higher-coverage sensitivity
setting is feasible on the reference RTX 4080.
"""

from __future__ import annotations

import copy
import gc
import json
import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pyproj  # noqa: F401
import numpy as np
import torch

from pointspace.datasets import build_dataset, point_collate_fn
from pointspace.models import build_model
from pointspace.utils.config import Config


SCENARIOS = (
    ("point-m2ae", "configs/baselines/swiss/pretrain-point-m2ae-litept-v1m4-100.py", None),
    ("msc", "configs/baselines/swiss/pretrain-msc-litept-v1m4-100.py", None),
    ("sonata-cap2048", "configs/baselines/swiss/pretrain-sonata-litept-v1m4-100.py", 2048),
    ("sonata-cap8192", "configs/baselines/swiss/pretrain-sonata-litept-v1m4-100.py", 8192),
    ("d-ditr", "configs/baselines/swiss/pretrain-d-ditr-litept-v1m4-100.py", None),
    ("concerto-cap2048", "configs/baselines/swiss/pretrain-concerto-litept-v1m4-100.py", 2048),
    ("concerto-cap8192", "configs/baselines/swiss/pretrain-concerto-litept-v1m4-100.py", 8192),
)
SEED = 45
OUTPUT = (
    Path(__file__).resolve().parent
    / "results" / "section_45" / "configured_batch_validation.json"
)


def _to_cuda(value):
    if torch.is_tensor(value):
        return value.cuda()
    if isinstance(value, dict):
        return {key: _to_cuda(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_to_cuda(item) for item in value)
    if isinstance(value, list):
        return [_to_cuda(item) for item in value]
    return value


def _scalars(output):
    return {
        key: float(value.detach().float().cpu())
        for key, value in output.items()
        if torch.is_tensor(value) and value.numel() == 1
    }


def run_scenario(name: str, config_path: str, cap: int | None) -> dict:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    config = Config.fromfile(config_path)
    dataset = build_dataset(config.data.train)
    accumulation = int(config.gradient_accumulation_steps)
    # Mirror pointspace.engines.defaults.default_setup exactly: the config's
    # batch_size_train denotes the effective per-world batch, while the loader
    # receives the micro-batch after gradient-accumulation division.
    batch_size = max(1, int(config.batch_size_train) // accumulation)
    data_start = time.perf_counter()
    samples = [dataset[index] for index in range(batch_size)]
    batch = point_collate_fn(samples)
    data_seconds = time.perf_counter() - data_start
    tensor_shapes = {
        key: list(value.shape) for key, value in batch.items()
        if torch.is_tensor(value)
    }
    batch = _to_cuda(batch)
    model_config = copy.deepcopy(dict(config.model))
    if cap is not None:
        model_config["matching_max_pair"] = cap
    model = build_model(model_config).cuda().train()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    step_start = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(batch)
    output["loss"].backward()
    torch.cuda.synchronize()
    result = {
        "status": "passed",
        "config": config_path,
        "batch_size": batch_size,
        "effective_batch_size": batch_size * accumulation,
        "gradient_accumulation_steps": accumulation,
        "matching_max_pair": cap,
        "tensor_shapes": tensor_shapes,
        "scalar_outputs": _scalars(output),
        "data_seconds": data_seconds,
        "forward_backward_seconds": time.perf_counter() - step_start,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
    }
    del output, model, batch, samples, dataset
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    report = {
        "schema": "cvrl_section45_configured_batch_validation_v1",
        "device": torch.cuda.get_device_name(),
        "seed": SEED,
        "scenarios": {},
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    for name, config_path, cap in SCENARIOS:
        print(f"[{name}]")
        try:
            result = run_scenario(name, config_path, cap)
        except Exception as error:
            result = {
                "status": "failed",
                "config": config_path,
                "matching_max_pair": cap,
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
            gc.collect()
            torch.cuda.empty_cache()
        report["scenarios"][name] = result
        OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({name: result}, indent=2))
    print(OUTPUT)


if __name__ == "__main__":
    main()
