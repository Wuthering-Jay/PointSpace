"""Create and verify an architecture-matched random VPSD/CVRL checkpoint.

For the Section 4.3 visual-internalization test, use ``TARGET = "vpsd"`` so
the control has the same architecture and 128-D visual output as VPSD.  The
``cvrl`` target is retained for the separate full-method architecture control.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pointspace.models import build_model
from pointspace.utils.config import Config


TARGET = "vpsd"  # "vpsd" for Section 4.3; "cvrl" for full-method control
TARGETS = {
    "vpsd": {
        "config": "configs/vpsd/pretrain-vpsd-v1m1-litept-v1m4-swiss-100.py",
        "output": "paper_cvrl/tmp/vpsd_scratch_swiss100_seed42.pth",
        "feature_source": "visual",
        "feature_dim": 128,
    },
    "cvrl": {
        "config": "configs/cvrl/pretrain-cvrl-v1m1-litept-v1m4-swiss-100.py",
        "output": "paper_cvrl/tmp/cvrl_scratch_swiss100_seed42.pth",
        "feature_source": "full",
        "feature_dim": 200,
    },
}
SEED = 42
OVERWRITE = False


def _seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    if TARGET not in TARGETS:
        raise ValueError(f"TARGET must be one of {tuple(TARGETS)}, got {TARGET!r}")
    target = TARGETS[TARGET]
    config_path = PROJECT_ROOT / target["config"]
    output_path = PROJECT_ROOT / target["output"]
    if output_path.exists() and not OVERWRITE:
        print(f"Using existing Scratch checkpoint: {output_path}")
    else:
        _seed_everything(SEED)
        cfg = Config.fromfile(str(config_path))
        model = build_model(cfg.model)
        state_dict = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": 0,
                "state_dict": state_dict,
                "id2class": None,
                "meta": {
                    "scratch": True,
                    "seed": SEED,
                    "target": TARGET,
                    "source_config": str(config_path.relative_to(PROJECT_ROOT)),
                    "feature_source": target["feature_source"],
                    "feature_dim": target["feature_dim"],
                },
            },
            output_path,
        )
        del model

    # Strict CPU reload catches missing/unexpected parameters before a costly
    # GPU feature-export job is started.
    cfg = Config.fromfile(str(config_path))
    checkpoint = torch.load(output_path, map_location="cpu", weights_only=False)
    verify_model = build_model(cfg.model)
    incompatible = verify_model.load_state_dict(checkpoint["state_dict"], strict=True)
    parameter_count = sum(parameter.numel() for parameter in verify_model.parameters())
    finite = all(
        not tensor.is_floating_point() or torch.isfinite(tensor).all().item()
        for tensor in checkpoint["state_dict"].values()
    )
    result = {
        "checkpoint": str(output_path),
        "target": TARGET,
        "seed": SEED,
        "parameter_count": parameter_count,
        "state_tensor_count": len(checkpoint["state_dict"]),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "all_finite": finite,
        "feature_source": target["feature_source"],
        "feature_dim": target["feature_dim"],
    }
    result_path = output_path.with_suffix(".json")
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
