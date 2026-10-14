"""Refine FT micro-batch limits between the power-of-two brackets.

Run after :mod:`benchmark_downstream_ft_memory`.  The base benchmark finds a
safe bracket cheaply; this script tests every integer inside that bracket and
updates the shared JSON report with the actual largest value below the reserved
memory safety limit.
"""

import gc
import json

import torch

from benchmark_downstream_ft_memory import (
    DATASETS,
    METHODS,
    OUTPUT,
    SAFE_RESERVED_GIB,
    build_dense_sample,
    guarded_run,
)


# Values strictly between the safe and unsafe power-of-two measurements.
RANGES = {
    "dales": {"litept": range(9, 16), "cvrl": range(5, 8)},
    "vaihingen": {"litept": range(9, 16), "cvrl": range(5, 8)},
    "opengf": {"litept": range(9, 16), "cvrl": range(9, 16)},
}


def main():
    # On Windows/WDDM an oversized CUDA allocation can spill into shared GPU
    # memory and stall for minutes instead of raising OOM.  Enforce the same
    # 14.5-GiB budget in the CUDA allocator so the first unsafe integer fails
    # promptly and reproducibly.
    total_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
    torch.cuda.set_per_process_memory_fraction(
        min(1.0, SAFE_RESERVED_GIB / total_gib), device=0
    )
    report = json.loads(OUTPUT.read_text(encoding="utf-8"))
    report["refinement"] = {}
    report["recommendation_policy"] = (
        "largest integer micro-batch with peak reserved CUDA memory <= "
        f"{SAFE_RESERVED_GIB} GiB on the selected dense sample"
    )
    for dataset_name in DATASETS:
        sample, metadata = build_dense_sample(dataset_name)
        assert metadata["source"] == report["dense_samples"][dataset_name]["source"]
        family_max = {}
        for family, method in (("litept", "point-m2ae"), ("cvrl", "cvrl")):
            key = f"{dataset_name}/{family}"
            values = []
            # Start with the safe power-of-two result from the base search.
            base_values = report["search"][key]
            accepted = [
                value["micro_batch"] for value in base_values
                if value["status"] == "passed"
                and value["peak_reserved_gib"] <= SAFE_RESERVED_GIB
            ]
            best = max(accepted)
            for micro_batch in RANGES[dataset_name][family]:
                value = guarded_run(dataset_name, method, micro_batch, sample)
                values.append(value)
                print(key, json.dumps(value), flush=True)
                if (
                    value["status"] != "passed"
                    or value["peak_reserved_gib"] > SAFE_RESERVED_GIB
                ):
                    break
                best = micro_batch
            report["refinement"][key] = values
            family_max[family] = best

        for method in METHODS:
            family = "cvrl" if method == "cvrl" else "litept"
            report["recommended_micro_batch"][f"{dataset_name}/{method}"] = (
                family_max[family]
            )
        del sample
        gc.collect()
        OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(OUTPUT)


if __name__ == "__main__":
    main()
