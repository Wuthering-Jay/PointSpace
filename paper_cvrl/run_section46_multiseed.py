"""Prepare and sequentially run the Section 4.6 stability experiment.

The sparse-label manifest remains fixed at ``sparse_labels_seed42``.  The
three experiment seeds therefore measure optimization/data-pipeline
variability rather than confounding it with a new annotation draw.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.generate_downstream_ssl_configs import render


SEEDS = (42, 3407, 5291)
METHODS = ("scratch", "point-m2ae")
BUDGETS = (
    ("0p01pct", 0.0001),
    ("0p1pct", 0.001),
    ("1pct", 0.01),
    ("10pct", 0.1),
    ("100pct", None),
)
CONFIG_ROOT = ROOT / "configs/label_efficiency/dales/multiseed"
EXP_ROOT = "exp/label_efficiency_multiseed/dales"


def is_complete(save_path: Path) -> bool:
    log_path = save_path / "train.log"
    if not log_path.exists():
        return False
    return "Test result:" in log_path.read_text(encoding="utf-8", errors="replace")


def prepare_configs() -> list[dict]:
    jobs = []
    for seed in SEEDS:
        output_dir = CONFIG_ROOT / f"seed{seed}"
        output_dir.mkdir(parents=True, exist_ok=True)
        save_root = f"{EXP_ROOT}/seed{seed}"
        for budget_name, fraction in BUDGETS:
            for method in METHODS:
                filename = (
                    f"semseg-{method}-litept-v1m4-dales-ft-"
                    f"{budget_name}-seed{seed}.py"
                )
                path = output_dir / filename
                text = render(
                    "dales",
                    method,
                    "ft",
                    label_fraction=fraction,
                    save_path_root=save_root,
                    experiment_seed=seed,
                    label_efficiency_full=fraction is None,
                )
                path.write_text(text, encoding="utf-8")
                fraction_key = "1" if fraction is None else f"{fraction:g}"
                jobs.append({
                    "seed": seed,
                    "method": method,
                    "budget": budget_name,
                    "label_fraction": fraction,
                    "config": str(path.relative_to(ROOT)).replace("\\", "/"),
                    "save_path": f"{save_root}/{fraction_key}/{method}-ft",
                })
    manifest = CONFIG_ROOT / "jobs.json"
    manifest.write_text(json.dumps(jobs, indent=2), encoding="utf-8")
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Scratch and Point-M2AE DALES label-efficiency repeats."
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Generate and validate the 30 standalone configs without training.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run jobs even if their train.log already contains Test result.",
    )
    args = parser.parse_args()
    jobs = prepare_configs()
    print(f"Prepared {len(jobs)} configs under {CONFIG_ROOT}", flush=True)
    if args.prepare_only:
        return
    for index, job in enumerate(jobs, 1):
        save_path = ROOT / job["save_path"]
        if not args.force and is_complete(save_path):
            print(
                f"[{index:02d}/{len(jobs)}] skip completed: "
                f"seed={job['seed']} {job['budget']} {job['method']}",
                flush=True,
            )
            continue
        print(
            f"[{index:02d}/{len(jobs)}] run: seed={job['seed']} "
            f"{job['budget']} {job['method']}",
            flush=True,
        )
        command = [
            sys.executable,
            str(ROOT / "tools/train.py"),
            "--config-file",
            str(ROOT / job["config"]),
        ]
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
