"""Run one real Swiss-100 forward/backward batch for every Section 4.5 baseline.

The script is an engineering acceptance test, not a throughput benchmark:
each method processes one dataset sample with its unmodified official Section
4.5 config and BF16 autocast.  Results are saved so that a synthetic unit test
cannot be mistaken for real-data validation.
"""

from __future__ import annotations

import gc
import json
import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pyproj  # noqa: F401 - initialize PROJ before CUDA/native extensions
import numpy as np
import torch

from pointspace.datasets import build_dataset
from pointspace.models import build_model
from pointspace.utils.config import Config


CONFIGS = {
    "point-m2ae": "configs/baselines/swiss/pretrain-point-m2ae-litept-v1m4-100.py",
    "msc": "configs/baselines/swiss/pretrain-msc-litept-v1m4-100.py",
    "sonata": "configs/baselines/swiss/pretrain-sonata-litept-v1m4-100.py",
    "d-ditr": "configs/baselines/swiss/pretrain-d-ditr-litept-v1m4-100.py",
    "concerto": "configs/baselines/swiss/pretrain-concerto-litept-v1m4-100.py",
}
METHODS_TO_RUN = tuple(CONFIGS)
SAMPLE_INDEX = 0
SEED = 45
OUTPUT = (
    Path(__file__).resolve().parent
    / "results" / "section_45" / "real_batch_validation.json"
)


def _to_cuda(value):
    if torch.is_tensor(value):
        return value.cuda(non_blocking=False)
    if isinstance(value, dict):
        return {key: _to_cuda(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_to_cuda(item) for item in value)
    if isinstance(value, list):
        return [_to_cuda(item) for item in value]
    return value


def _tensor_shapes(sample) -> dict[str, list[int]]:
    return {
        key: list(value.shape)
        for key, value in sample.items()
        if torch.is_tensor(value)
    }


def _scalar_outputs(output) -> dict[str, float]:
    return {
        key: float(value.detach().float().cpu())
        for key, value in output.items()
        if torch.is_tensor(value) and value.numel() == 1
    }


def validate_method(name: str, config_path: str) -> dict:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    config = Config.fromfile(config_path)
    load_start = time.perf_counter()
    dataset = build_dataset(config.data.train)
    sample = dataset[SAMPLE_INDEX]
    data_seconds = time.perf_counter() - load_start
    shapes = _tensor_shapes(sample)
    sample = _to_cuda(sample)
    model = build_model(config.model).cuda().train()
    model_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    step_start = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(sample)
    loss = output["loss"]
    if not torch.isfinite(loss):
        raise FloatingPointError(f"{name}: non-finite loss {float(loss.detach())}")
    loss.backward()
    torch.cuda.synchronize()
    step_seconds = time.perf_counter() - step_start
    gradients = [
        parameter.grad for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not gradients or not any(torch.isfinite(gradient).all() for gradient in gradients):
        raise RuntimeError(f"{name}: no finite trainable gradients")
    result = {
        "status": "passed",
        "config": config_path,
        "sample_index": SAMPLE_INDEX,
        "dataset_size": len(dataset),
        "tensor_shapes": shapes,
        "scalar_outputs": _scalar_outputs(output),
        "data_seconds": data_seconds,
        "forward_backward_seconds": step_seconds,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "model_parameters": model_parameters,
        "trainable_parameters": trainable_parameters,
    }
    del output, loss, gradients, model, sample, dataset
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Section 4.5 real-batch validation requires CUDA")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    results = {
        "schema": "cvrl_section45_real_batch_validation_v1",
        "seed": SEED,
        "sample_index": SAMPLE_INDEX,
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "methods": {},
    }
    for name in METHODS_TO_RUN:
        print(f"[{name}] real Swiss batch")
        try:
            result = validate_method(name, CONFIGS[name])
        except Exception as error:  # preserve all method outcomes in one artifact
            result = {
                "status": "failed",
                "config": CONFIGS[name],
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
            gc.collect()
            torch.cuda.empty_cache()
        results["methods"][name] = result
        OUTPUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(json.dumps({name: result}, indent=2))
    print(OUTPUT)


if __name__ == "__main__":
    main()
