"""Small GPU forward/backward check for all four P1 configs.

Uses one Swiss-Audit source tile, not the full Swiss training set. This smoke
test never updates weights or saves a checkpoint. Run in the pointspace env.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from pointspace.datasets import build_dataset
from pointspace.models import build_model
from pointspace.utils.config import Config

CONFIG_DIR = ROOT / "configs/rrcl/p1_swiss100"
AUDIT_ROOT = Path(r"G:\data\swiss\joint_tiles_100\Swiss-Audit")
POINT_MAX = 2048


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("P1 smoke test requires CUDA for LitePT sparse operators")
    for filename in (
        "geometry_normal.py", "geometry_local_shape.py",
        "rrcl_role_only.py", "rrcl_context_only.py",
        "pretrain-rrcl-v1m1-litept-v1m4-swiss-100.py",
    ):
        config_path = (
            CONFIG_DIR.parent / filename
            if filename.startswith("pretrain-") else CONFIG_DIR / filename
        )
        cfg = Config.fromfile(str(config_path))
        train_cfg = cfg.data.train
        train_cfg.data_path = str(
            AUDIT_ROOT / ("geometry_p1_smoke" if cfg.model.type == "GeometrySSL" else "pointcloud")
        )
        if cfg.model.type == "RRCL":
            train_cfg.class_weight = [1.0] * 4
        for transform in train_cfg.transform:
            if transform["type"] == "SphereCrop":
                transform["point_max"] = POINT_MAX
        dataset = build_dataset(train_cfg)
        sample = {key: (value.cuda(non_blocking=True) if isinstance(value, torch.Tensor) else value)
                  for key, value in dataset[0].items()}
        model = build_model(cfg.model).cuda().train()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(sample)
        loss = output["loss"]
        if not torch.isfinite(loss):
            raise ValueError(f"{filename}: non-finite loss")
        loss.backward()
        backbone_grad = next(p.grad for p in model.backbone.parameters() if p.grad is not None)
        if not torch.isfinite(backbone_grad).all():
            raise ValueError(f"{filename}: non-finite backbone gradient")
        print(filename, "points", sample["coord"].shape[0], "loss", float(loss.detach()))
        del model, dataset, sample
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
