"""Find safe maximum FT micro-batches on the three downstream datasets."""

import copy
import gc
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import laspy
import numpy as np
import torch

from pointspace.datasets import build_dataset, point_collate_fn
from pointspace.models import build_model
from pointspace.utils.config import Config
from paper_cvrl.validate_downstream_ssl_matrix import load_pretrained, to_cuda


DATASETS = ("dales", "vaihingen", "opengf")
METHODS = (
    "scratch", "point-m2ae", "msc", "sonata", "d-ditr",
    "concerto", "vpsd", "rrcl", "cvrl",
)
CANDIDATES = (2, 4, 8, 16)
SAFE_RESERVED_GIB = 14.5
OUTPUT = PROJECT_ROOT / "paper_cvrl" / "results" / "section_45" / "downstream_ft_memory.json"


def config_path(dataset_name, method):
    return PROJECT_ROOT / "configs" / "downstream" / dataset_name / (
        f"semseg-{method}-litept-v1m4-{dataset_name}-ft.py"
    )


def build_dense_sample(dataset_name):
    cfg = Config.fromfile(str(config_path(dataset_name, "scratch")))
    train_cfg = dict(cfg.data.train)
    train_cfg["class_weight"] = None
    train_cfg["weighted_sampler"] = False
    train_cfg["loop"] = 1
    dataset = build_dataset(train_cfg)

    header_counts = []
    for index, path in enumerate(dataset.data_list):
        try:
            with laspy.open(path) as reader:
                header_counts.append((int(reader.header.point_count), index))
        except Exception:
            continue
    header_counts.sort(reverse=True)
    transformed = []
    # Large raw files are the likely post-voxel worst cases. Testing 32 avoids
    # choosing a sparse large-area file merely because it has many raw points.
    for rank, (_, index) in enumerate(header_counts[:32]):
        np.random.seed(4500 + rank)
        torch.manual_seed(4500 + rank)
        sample = dataset[index]
        transformed.append((int(sample["coord"].shape[0]), index, sample))
    transformed.sort(key=lambda item: item[0], reverse=True)
    points, index, sample = transformed[0]
    source = dataset.data_list[index]
    del dataset
    gc.collect()
    return sample, {"source": source, "points_per_sample": points}


def clone_sample(sample):
    result = {}
    for key, value in sample.items():
        if torch.is_tensor(value):
            result[key] = value.clone()
        elif isinstance(value, np.ndarray):
            result[key] = value.copy()
        else:
            result[key] = copy.deepcopy(value)
    return result


def run_one(dataset_name, method, micro_batch, dense_sample):
    cfg = Config.fromfile(str(config_path(dataset_name, method)))
    samples = [clone_sample(dense_sample) for _ in range(micro_batch)]
    batch = point_collate_fn(samples)
    points = int(batch["coord"].shape[0])
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
    torch.cuda.synchronize()
    start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(batch)
        loss = output["loss"]
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    result = {
        "status": "passed",
        "micro_batch": micro_batch,
        "points": points,
        "loss": float(loss.detach().cpu()),
        "seconds": time.perf_counter() - start,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "checkpoint_selected": checkpoint["selected"],
    }
    del loss, output, optimizer, model, batch, samples
    gc.collect()
    torch.cuda.empty_cache()
    return result


def guarded_run(dataset_name, method, micro_batch, sample):
    try:
        return run_one(dataset_name, method, micro_batch, sample)
    except torch.cuda.OutOfMemoryError as error:
        gc.collect()
        torch.cuda.empty_cache()
        return {
            "status": "oom",
            "micro_batch": micro_batch,
            "error": str(error).splitlines()[0],
        }
    except Exception as error:
        gc.collect()
        torch.cuda.empty_cache()
        return {
            "status": "failed",
            "micro_batch": micro_batch,
            "error": f"{type(error).__name__}: {error}",
        }


def safe_max(results):
    accepted = [
        value["micro_batch"]
        for value in results
        if value["status"] == "passed"
        and value["peak_reserved_gib"] <= SAFE_RESERVED_GIB
    ]
    return max(accepted, default=1)


def main():
    report = {
        "device": torch.cuda.get_device_name(),
        "device_total_gib": torch.cuda.get_device_properties(0).total_memory / 2**30,
        "safe_reserved_gib": SAFE_RESERVED_GIB,
        "candidates": CANDIDATES,
        "dense_samples": {},
        "search": {},
        "final": {},
        "recommended_micro_batch": {},
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    for dataset_name in DATASETS:
        sample, metadata = build_dense_sample(dataset_name)
        report["dense_samples"][dataset_name] = metadata
        print(f"[{dataset_name}] dense sample: {metadata}", flush=True)

        # Standard LitePT FT and CVRL FT bracket every architecture in this
        # matrix. VPSD/RRCL and the five external SSL methods all use the same
        # downstream LitePT graph; CVRL retains its 200-D dual branch.
        family_max = {}
        for family, method in (("litept", "point-m2ae"), ("cvrl", "cvrl")):
            key = f"{dataset_name}/{family}"
            values = []
            for micro_batch in CANDIDATES:
                value = guarded_run(dataset_name, method, micro_batch, sample)
                values.append(value)
                print(key, json.dumps(value), flush=True)
                OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
                if value["status"] != "passed":
                    break
                # Do not deliberately launch the next doubled batch once the
                # current one already occupies at least 90% of the safety
                # budget.  The next candidate cannot fit and may make WDDM
                # spend minutes paging instead of raising a prompt CUDA OOM.
                if value["peak_reserved_gib"] >= 0.9 * SAFE_RESERVED_GIB:
                    break
            report["search"][key] = values
            family_max[family] = safe_max(values)

        final = {}
        for method in METHODS:
            family = "cvrl" if method == "cvrl" else "litept"
            micro_batch = family_max[family]
            value = guarded_run(dataset_name, method, micro_batch, sample)
            final[method] = value
            report["recommended_micro_batch"][f"{dataset_name}/{method}"] = (
                micro_batch if value["status"] == "passed" and
                value["peak_reserved_gib"] <= SAFE_RESERVED_GIB else
                max(1, micro_batch // 2)
            )
            print(f"{dataset_name}/{method}", json.dumps(value), flush=True)
            OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
        report["final"][dataset_name] = final
        del sample
        gc.collect()

    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
