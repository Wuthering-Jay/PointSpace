"""Recalibrate LitePT BatchNorm buffers on clean Swiss-100 point clouds.

This utility never updates learnable parameters.  It runs the retained
LitePT backbone with gradients disabled, keeps stochastic layers in eval
mode, and puts only BatchNorm modules in train mode.  Per-point activation
statistics are accumulated across a deterministic random subset and written
to a checkpoint copy; the source checkpoint is never modified.

The defaults at the bottom are intentionally IDE-friendly.  Command-line
arguments are also provided for reproducible batch execution.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, Mapping

# On Windows, import pyproj before torch/spconv.  Their bundled DLL search
# paths can otherwise make pyproj resolve an incompatible native dependency.
import pyproj  # noqa: F401
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pointspace.models  # noqa: F401,E402 - populate model registry
from pointspace.datasets import build_dataset, point_collate_fn  # noqa: E402
from pointspace.models import build_model  # noqa: E402
from utils.generate_downstream_ssl_configs import litept_backbone  # noqa: E402


LOGGER = logging.getLogger("swiss_bn_calibration")

CHECKPOINTS = OrderedDict(
    [
        (
            "scratch",
            (
                PROJECT_ROOT
                / "exp/rrcl/swiss/scratch-litept-v1m4-100/model/random_seed42.pth",
                "backbone.",
            ),
        ),
        (
            "point-m2ae",
            (
                PROJECT_ROOT
                / "exp/baselines/swiss/pretrain-point-m2ae-litept-v1m4-100/model/backbone_last.pth",
                "backbone.",
            ),
        ),
        (
            "msc",
            (
                PROJECT_ROOT
                / "exp/baselines/swiss/pretrain-msc-litept-v1m4-100/model/backbone_last.pth",
                "backbone.",
            ),
        ),
        (
            "sonata",
            (
                PROJECT_ROOT
                / "exp/baselines/swiss/pretrain-sonata-litept-v1m4-100/model/backbone_last.pth",
                "backbone.",
            ),
        ),
        (
            "d-ditr",
            (
                PROJECT_ROOT
                / "exp/baselines/swiss/pretrain-d-ditr-litept-v1m4-100/model/backbone_last.pth",
                "backbone.",
            ),
        ),
        (
            "concerto",
            (
                PROJECT_ROOT
                / "exp/baselines/swiss/pretrain-concerto-litept-v1m4-100/model/backbone_last.pth",
                "backbone.",
            ),
        ),
        (
            "vpsd",
            (
                PROJECT_ROOT
                / "exp/vpsd/swiss/pretrain-v1m1-litept-v1m4-100/model/model_last.pth",
                "backbone.",
            ),
        ),
        (
            "rrcl",
            (
                PROJECT_ROOT
                / "exp/rrcl/swiss/pretrain-v1m1-litept-v1m4-100/model/backbone_last.pth",
                "backbone.",
            ),
        ),
        (
            "cvrl",
            (
                PROJECT_ROOT
                / "exp/cvrl/swiss/pretrain-v1m1-litept-v1m4-100-full/model/model_last.pth",
                "backbone.return_backbone.",
            ),
        ),
    ]
)


def _output_path(source: Path) -> Path:
    return source.with_name(f"{source.stem}_bncal_swiss{source.suffix}")


def _state_dict(checkpoint: Mapping) -> Mapping[str, torch.Tensor]:
    state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state, Mapping):
        raise TypeError("Checkpoint state_dict must be a mapping")
    return state


def _extract_backbone_state(
    state: Mapping[str, torch.Tensor], prefix: str
) -> Dict[str, torch.Tensor]:
    extracted = {
        key[len(prefix) :]: value
        for key, value in state.items()
        if key.startswith(prefix)
    }
    if not extracted:
        raise KeyError(f"No checkpoint keys start with {prefix!r}")
    return extracted


def _build_clean_dataset(data_path: str, grid_size: float, point_max: int):
    return build_dataset(
        dict(
            type="LasDataset",
            split="train",
            data_path=data_path,
            test_mode=False,
            loop=1,
            transform=[
                dict(type="ZPercentileCenterShift", percentile=2.0),
                dict(type="RobustLogIntensity"),
                dict(
                    type="GridSample",
                    grid_size=grid_size,
                    hash_type="fnv",
                    mode="train",
                    return_grid_coord=True,
                ),
                dict(type="SphereCrop", point_max=point_max, mode="center"),
            ],
            post_transform=[
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord"),
                    feat_keys=("coord", "intensity"),
                ),
            ],
        )
    )


def _move_batch(batch, device):
    return {
        key: value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


class BatchNormMoments:
    """Point-weighted activation moments collected by BN pre-hooks."""

    def __init__(self, model: nn.Module):
        self.modules = {
            name: module
            for name, module in model.named_modules()
            if isinstance(module, nn.modules.batchnorm._BatchNorm)
        }
        self.sums = {}
        self.square_sums = {}
        self.counts = {name: 0 for name in self.modules}
        self.calls = {name: 0 for name in self.modules}
        self.handles = [
            module.register_forward_pre_hook(self._hook(name))
            for name, module in self.modules.items()
        ]

    def _hook(self, name):
        def collect(_module, inputs):
            value = inputs[0].detach().float()
            if value.ndim < 2:
                raise ValueError(
                    f"BatchNorm input for {name!r} has invalid shape "
                    f"{tuple(value.shape)}"
                )
            reduce_dims = tuple(i for i in range(value.ndim) if i != 1)
            batch_sum = value.sum(dim=reduce_dims)
            batch_square_sum = value.square().sum(dim=reduce_dims)
            count = value.numel() // value.shape[1]
            if name not in self.sums:
                self.sums[name] = batch_sum
                self.square_sums[name] = batch_square_sum
            else:
                self.sums[name].add_(batch_sum)
                self.square_sums[name].add_(batch_square_sum)
            self.counts[name] += int(count)
            self.calls[name] += 1

        return collect

    def finalize(self):
        stats = {}
        for name, module in self.modules.items():
            count = self.counts[name]
            if count < 2:
                raise RuntimeError(
                    f"BatchNorm module {name!r} collected only {count} values"
                )
            mean = self.sums[name] / count
            centered_sum = self.square_sums[name] - count * mean.square()
            variance = (centered_sum / (count - 1)).clamp_min_(0.0)
            module.running_mean.copy_(mean.to(module.running_mean.dtype))
            module.running_var.copy_(variance.to(module.running_var.dtype))
            module.num_batches_tracked.fill_(self.calls[name])
            stats[name] = {
                "count": count,
                "mean_abs": float(mean.abs().mean().cpu()),
                "variance_mean": float(variance.mean().cpu()),
            }
        return stats

    def close(self):
        for handle in self.handles:
            handle.remove()


def _set_bn_calibration_mode(model: nn.Module):
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.train()
            module.reset_running_stats()


def _copy_calibrated_buffers(
    checkpoint,
    model: nn.Module,
    prefix: str,
):
    state = _state_dict(checkpoint)
    model_state = model.state_dict()
    changed = []
    for local_key, value in model_state.items():
        if not (
            local_key.endswith("running_mean")
            or local_key.endswith("running_var")
            or local_key.endswith("num_batches_tracked")
        ):
            continue
        checkpoint_key = prefix + local_key
        if checkpoint_key not in state:
            continue
        state[checkpoint_key] = value.detach().cpu().clone()
        changed.append(checkpoint_key)
    if not changed:
        raise RuntimeError(f"No BatchNorm buffers matched prefix {prefix!r}")
    return changed


def _verify_copy(source_checkpoint, output_checkpoint, changed_keys):
    source_state = _state_dict(source_checkpoint)
    output_state = _state_dict(output_checkpoint)
    if source_state.keys() != output_state.keys():
        raise AssertionError("Calibration changed checkpoint key structure")
    allowed = set(changed_keys)
    actual_changes = []
    for key in source_state:
        left, right = source_state[key], output_state[key]
        if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
            if not torch.equal(left.cpu(), right.cpu()):
                actual_changes.append(key)
        elif left != right:
            actual_changes.append(key)
    illegal = sorted(set(actual_changes).difference(allowed))
    if illegal:
        raise AssertionError(
            "Non-BatchNorm checkpoint values changed: " + ", ".join(illegal[:8])
        )
    return actual_changes


@torch.inference_mode()
def calibrate_method(
    name: str,
    source: Path,
    prefix: str,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
    amp_dtype: torch.dtype,
):
    LOGGER.info("Calibrating %s from %s", name, source)
    source_checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    source_state = _state_dict(source_checkpoint)
    model = build_model(litept_backbone(freeze_encoder=False)).to(device)
    incompatible = model.load_state_dict(
        _extract_backbone_state(source_state, prefix), strict=False
    )
    allowed_missing = tuple(key for key in incompatible.missing_keys if key.startswith("dec."))
    unexpected = tuple(incompatible.unexpected_keys)
    if unexpected:
        raise RuntimeError(f"Unexpected LitePT keys for {name}: {unexpected[:8]}")
    if len(allowed_missing) != len(incompatible.missing_keys):
        raise RuntimeError(
            f"Non-decoder LitePT keys missing for {name}: "
            f"{tuple(k for k in incompatible.missing_keys if not k.startswith('dec.'))[:8]}"
        )

    _set_bn_calibration_mode(model)
    moments = BatchNormMoments(model)
    processed_batches = 0
    processed_points = 0
    autocast_enabled = device.type == "cuda"
    try:
        for batch in tqdm(loader, total=max_batches, desc=f"BN {name}", unit="batch"):
            batch = _move_batch(batch, device)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=autocast_enabled,
            ):
                output = model(batch)
            feature = output.feat if hasattr(output, "feat") else output
            if not torch.isfinite(feature).all():
                raise FloatingPointError(f"Non-finite output while calibrating {name}")
            processed_points += int(batch["coord"].shape[0])
            processed_batches += 1
            if processed_batches >= max_batches:
                break
        statistics = moments.finalize()
    finally:
        moments.close()

    changed_keys = _copy_calibrated_buffers(source_checkpoint, model, prefix)
    output = _output_path(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(source_checkpoint, output)
    saved_checkpoint = torch.load(output, map_location="cpu", weights_only=False)
    actual_changes = _verify_copy(
        torch.load(source, map_location="cpu", weights_only=False),
        saved_checkpoint,
        changed_keys,
    )
    report = {
        "source": str(source),
        "output": str(output),
        "prefix": prefix,
        "batches": processed_batches,
        "points": processed_points,
        "batch_norm_modules": len(statistics),
        "matched_bn_buffers": len(changed_keys),
        "changed_bn_buffers": len(actual_changes),
        "missing_decoder_keys": len(allowed_missing),
        "statistics": statistics,
    }
    LOGGER.info(
        "%s: %d batches, %d points, %d BN modules -> %s",
        name,
        processed_batches,
        processed_points,
        len(statistics),
        output,
    )
    del model, source_checkpoint, saved_checkpoint
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return report


def calibrate_all(
    methods: Iterable[str],
    data_path: str,
    max_samples: int = 512,
    batch_size: int = 2,
    num_workers: int = 2,
    grid_size: float = 0.25,
    point_max: int = 220000,
    seed: int = 42,
    amp_dtype: str = "bfloat16",
):
    selected = list(methods)
    unknown = sorted(set(selected).difference(CHECKPOINTS))
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[amp_dtype]
    dataset = _build_clean_dataset(data_path, grid_size, point_max)
    sample_count = min(int(max_samples), len(dataset))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:sample_count].tolist()
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        collate_fn=point_collate_fn,
    )
    max_batches = (sample_count + batch_size - 1) // batch_size
    reports = {}
    for method in selected:
        source, prefix = CHECKPOINTS[method]
        if not source.exists():
            raise FileNotFoundError(source)
        reports[method] = calibrate_method(
            method, source, prefix, loader, device, max_batches, dtype
        )
    report_path = PROJECT_ROOT / "exp/bn_calibration_swiss100/calibration_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    # A method-subset run (for example adding a calibrated scratch baseline)
    # must not erase the audit records produced for the other checkpoints.
    if report_path.exists():
        previous = json.loads(report_path.read_text(encoding="utf-8"))
        previous.update(reports)
        reports = previous
    report_path.write_text(
        json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    LOGGER.info("Calibration report: %s", report_path)
    return reports


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", choices=tuple(CHECKPOINTS), default=list(CHECKPOINTS))
    parser.add_argument("--data-path", default=r"G:\data\swiss\joint_tiles_100\pointcloud")
    parser.add_argument("--max-samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--grid-size", type=float, default=0.25)
    parser.add_argument("--point-max", type=int, default=220000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp-dtype", choices=("bfloat16", "float16"), default="bfloat16")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    args = _parse_args()
    calibrate_all(
        methods=args.methods,
        data_path=args.data_path,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        grid_size=args.grid_size,
        point_max=args.point_max,
        seed=args.seed,
        amp_dtype=args.amp_dtype,
    )
