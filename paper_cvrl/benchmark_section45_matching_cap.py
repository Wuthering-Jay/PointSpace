"""Real-batch matching-cap sensitivity for Sonata and Concerto.

Only ``matching_max_pair`` changes within a method.  The transformed sample,
model initialization and forward RNG are reset to the same seeds for each cap.
This isolates the explicit non-equivalent pair subsampling approximation noted
in the Section 4.5 implementation plan.
"""

from __future__ import annotations

import copy
import gc
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pyproj  # noqa: F401
import numpy as np
import torch

from pointspace.datasets import build_dataset
from pointspace.models import build_model
from pointspace.utils.config import Config


CONFIGS = {
    "sonata": "configs/baselines/swiss/pretrain-sonata-litept-v1m4-100.py",
    "concerto": "configs/baselines/swiss/pretrain-concerto-litept-v1m4-100.py",
}
CAPS = (512, 2048, 8192)
DATA_SEED = 45
MODEL_SEED = 451
FORWARD_SEED = 452
SAMPLE_INDEX = 0
OUTPUT = (
    Path(__file__).resolve().parent
    / "results" / "section_45" / "matching_cap_sensitivity.json"
)


def _to_cuda(value):
    if torch.is_tensor(value):
        return value.cuda()
    if isinstance(value, dict):
        return {key: _to_cuda(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cuda(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cuda(item) for item in value)
    return value


def _scalars(output):
    return {
        key: float(value.detach().float().cpu())
        for key, value in output.items()
        if torch.is_tensor(value) and value.numel() == 1
    }


def run_method(name: str, path: str) -> dict:
    config = Config.fromfile(path)
    torch.manual_seed(DATA_SEED)
    np.random.seed(DATA_SEED)
    dataset = build_dataset(config.data.train)
    cpu_sample = dataset[SAMPLE_INDEX]
    output = {
        "config": path,
        "sample_index": SAMPLE_INDEX,
        "global_view_points": {
            key: int(value.shape[0])
            for key, value in cpu_sample.items()
            if key in ("view1_coord", "view2_coord")
        },
        "caps": {},
    }
    for cap in CAPS:
        model_config = copy.deepcopy(dict(config.model))
        model_config["matching_max_pair"] = cap
        torch.manual_seed(MODEL_SEED)
        model = build_model(model_config).cuda().train()
        sample = _to_cuda(cpu_sample)
        torch.manual_seed(FORWARD_SEED)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = model(sample)
        result["loss"].backward()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        scalars = _scalars(result)
        output["caps"][str(cap)] = {
            "scalar_outputs": scalars,
            "forward_backward_seconds": elapsed,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        }
        print(name, cap, output["caps"][str(cap)])
        del result, sample, model
        gc.collect()
        torch.cuda.empty_cache()
    del cpu_sample, dataset
    return output


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    report = {
        "schema": "cvrl_section45_matching_cap_sensitivity_v1",
        "device": torch.cuda.get_device_name(),
        "caps": list(CAPS),
        "data_seed": DATA_SEED,
        "model_seed": MODEL_SEED,
        "forward_seed": FORWARD_SEED,
        "methods": {},
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    for name, path in CONFIGS.items():
        report["methods"][name] = run_method(name, path)
        OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
