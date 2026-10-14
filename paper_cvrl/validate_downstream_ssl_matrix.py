"""Short real-data forward/backward validation for downstream SSL configs.

This tests every distinct compute path on every downstream dataset.  Methods
sharing the same exported LitePT backbone use the same path; all checkpoint
key mappings are validated separately by the config audit.
"""

import gc
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from pointspace.datasets import build_dataset, point_collate_fn
from pointspace.models import build_model
from pointspace.utils.config import Config


SCENARIOS = (
    ("point-m2ae", "lin"),
    ("point-m2ae", "dec"),
    ("point-m2ae", "ft"),
    ("vpsd", "lin"),
    ("cvrl", "lin"),
    ("cvrl", "ft"),
)
DATASETS = ("dales", "vaihingen", "opengf")
OUTPUT = PROJECT_ROOT / "paper_cvrl" / "results" / "section_45" / "downstream_smoke_validation.json"


def to_cuda(value):
    if torch.is_tensor(value):
        return value.cuda(non_blocking=True)
    if isinstance(value, dict):
        return {key: to_cuda(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_cuda(item) for item in value]
    if isinstance(value, tuple):
        return tuple(to_cuda(item) for item in value)
    return value


def load_pretrained(model, cfg):
    hook = next((dict(item) for item in cfg.hooks if item.type == "CheckpointLoader"), None)
    if hook is None:
        return {"selected": 0, "missing": [], "unexpected": []}
    checkpoint = torch.load(cfg.weight, map_location="cpu", weights_only=False)
    state = checkpoint["state_dict"]
    prefixes = tuple(hook.get("include_prefixes", ()))
    keywords = hook.get("keywords", "")
    replacement = hook.get("replacement", keywords)
    mapped = {}
    for key, value in state.items():
        if not key.startswith("module."):
            key = "module." + key
        unwrapped = key[7:] if key.startswith("module.") else key
        if prefixes and not any(
            key.startswith(prefix) or unwrapped.startswith(prefix)
            for prefix in prefixes
        ):
            continue
        if keywords in key:
            key = key.replace(keywords, replacement, 1)
        if key.startswith("module."):
            key = key[7:]
        mapped[key] = value
    info = model.load_state_dict(mapped, strict=False)
    if info.unexpected_keys:
        raise RuntimeError(f"Unexpected checkpoint keys: {info.unexpected_keys[:8]}")
    return {
        "selected": len(mapped),
        "missing": list(info.missing_keys),
        "unexpected": list(info.unexpected_keys),
    }


def run(dataset_name, method, probe):
    path = PROJECT_ROOT / "configs" / "downstream" / dataset_name / (
        f"semseg-{method}-litept-v1m4-{dataset_name}-{probe}.py"
    )
    cfg = Config.fromfile(str(path))
    # Weight statistics and sampling are orthogonal to this one-step smoke
    # test and would repeatedly scan large datasets.
    train_cfg = dict(cfg.data.train)
    train_cfg["class_weight"] = None
    train_cfg["weighted_sampler"] = False
    train_cfg["loop"] = 1
    dataset = build_dataset(train_cfg)
    batch = point_collate_fn([dataset[0], dataset[min(1, len(dataset) - 1)]])
    num_points = int(batch["coord"].shape[0])

    model = build_model(cfg.model)
    checkpoint = load_pretrained(model, cfg)
    model = model.cuda().train()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-4,
    )
    batch = to_cuda(batch)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(batch)
        loss = output["loss"]
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    result = {
        "status": "passed",
        "config": str(path.relative_to(PROJECT_ROOT)),
        "points": num_points,
        "loss": float(loss.detach().cpu()),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "checkpoint_selected": checkpoint["selected"],
        "checkpoint_missing": len(checkpoint["missing"]),
    }
    del loss, output, optimizer, model, batch, dataset
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    report = {"device": torch.cuda.get_device_name(), "results": {}}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    for dataset_name in DATASETS:
        for method, probe in SCENARIOS:
            key = f"{dataset_name}/{method}-{probe}"
            print(f"[{key}]", flush=True)
            try:
                result = run(dataset_name, method, probe)
            except Exception as error:
                result = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
            report["results"][key] = result
            OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(json.dumps(result, ensure_ascii=False), flush=True)
    print(OUTPUT)


if __name__ == "__main__":
    main()
