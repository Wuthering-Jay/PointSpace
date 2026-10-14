"""Run all Section 4.5 Swiss pretraining baselines sequentially.

Completed checkpoints (including runs completed by EarlyStoppingHook) are
skipped. Partial checkpoints are resumed through each standalone config.
Constants are intentionally editable for direct IDE execution.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pointspace.utils.config import Config


CONFIGS = (
    "configs/baselines/swiss/pretrain-point-m2ae-litept-v1m4-100.py",
    "configs/baselines/swiss/pretrain-msc-litept-v1m4-100.py",
    "configs/baselines/swiss/pretrain-d-ditr-litept-v1m4-100.py",
    "configs/baselines/swiss/pretrain-sonata-litept-v1m4-100.py",
    "configs/baselines/swiss/pretrain-concerto-litept-v1m4-100.py",
)
SKIP_COMPLETED = True
FAIL_FAST = True
REPORT_PATH = (
    PROJECT_ROOT / "paper_cvrl" / "results" / "section_45"
    / "pretraining_run_status.json"
)


def _checkpoint_status(config):
    checkpoint_path = PROJECT_ROOT / config.weight
    if not checkpoint_path.is_file():
        return "missing", None
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        epoch = int(checkpoint.get("epoch", 0))
    except Exception as error:
        return f"unreadable: {type(error).__name__}: {error}", None
    if epoch >= int(config.epoch):
        return "completed", epoch

    state_path = checkpoint_path.parent / "early_stopping.json"
    if state_path.is_file():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if (
                int(state.get("epoch", -1)) == epoch
                and int(state.get("bad_epochs", 0))
                >= int(state.get("patience", 3))
            ):
                return "early-stopped", epoch
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
    return "partial", epoch


def main():
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = {"configs": {}, "started_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    for relative in CONFIGS:
        path = PROJECT_ROOT / relative
        config = Config.fromfile(str(path))
        status, epoch = _checkpoint_status(config)
        record = {
            "checkpoint_status": status,
            "checkpoint_epoch": epoch,
            "target_epoch": int(config.epoch),
        }
        report["configs"][relative] = record
        if SKIP_COMPLETED and status in {"completed", "early-stopped"}:
            record["action"] = "skipped"
            print(
                f"[skip] {relative}: {status} at epoch {epoch}", flush=True
            )
            REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
            continue

        action = "resume" if status == "partial" else "train"
        record["action"] = action
        print(f"[{action}] {relative}", flush=True)
        started = time.perf_counter()
        process = subprocess.run(
            [sys.executable, "tools/train.py", "--config-file", str(path)],
            cwd=PROJECT_ROOT,
            check=False,
        )
        record["return_code"] = process.returncode
        record["wall_seconds"] = time.perf_counter() - started
        REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
        if process.returncode != 0 and FAIL_FAST:
            raise SystemExit(process.returncode)
    report["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
