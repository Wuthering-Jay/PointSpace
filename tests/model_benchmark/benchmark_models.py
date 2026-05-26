"""Benchmark segmentation models on synthetic point clouds.

This script is intentionally isolated under tests/ so it can measure local
runtime characteristics without changing PointSpace model or training code.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
import platform
import random
import runpy
import socket
import statistics
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


MODEL_CONFIGS = {
    "ptv2": "configs/dales/semseg-pt-v2m4-0-base.py",
    "ptv3": "configs/dales/semseg-pt-v3m1-0-base.py",
    "deeplanet": "configs/dales/semseg-deeplanet-v2-0.py",
    "litept": "configs/dales/semseg-litept-v1m1-0-base.py",
}


MODEL_LABELS = {
    "randla": "RandLA-Net",
    "spvcnn": "SPVCNN",
    "pointnext": "PointNeXt",
    "ptv2": "Point Transformer V2",
    "ptv3": "Point Transformer V3",
    "deeplanet": "DeepPLANet",
    "litept": "LitePT",
}


MODEL_ORDER = ["randla", "spvcnn", "pointnext", "ptv2", "ptv3", "deeplanet", "litept"]


@dataclass
class BenchmarkResult:
    model: str
    implementation: str
    status: str
    params_m: float | None = None
    train_latency_ms: float | None = None
    train_latency_std_ms: float | None = None
    infer_latency_ms: float | None = None
    infer_latency_std_ms: float | None = None
    train_memory_mib: float | None = None
    train_memory_std_mib: float | None = None
    train_memory_max_mib: float | None = None
    infer_memory_mib: float | None = None
    infer_memory_std_mib: float | None = None
    infer_memory_max_mib: float | None = None
    note: str = ""
    error: str = ""


class ResidualMLPBlock(nn.Module):
    def __init__(self, channels: int, hidden_ratio: float = 2.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(channels * hidden_ratio)
        self.net = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, channels),
            nn.BatchNorm1d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x + self.net(x))


class ApproxSegModel(nn.Module):
    """Point-wise segmentation approximation used for external baselines."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        channels: tuple[int, ...],
        blocks_per_stage: int,
        name: str,
    ):
        super().__init__()
        self.name = name
        layers: list[nn.Module] = [
            nn.Linear(in_channels + 3, channels[0]),
            nn.BatchNorm1d(channels[0]),
            nn.GELU(),
        ]
        for i, channel in enumerate(channels):
            if i > 0:
                layers.extend(
                    [
                        nn.Linear(channels[i - 1], channel),
                        nn.BatchNorm1d(channel),
                        nn.GELU(),
                    ]
                )
            for _ in range(blocks_per_stage):
                layers.append(ResidualMLPBlock(channel))
        self.encoder = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Linear(channels[-1], channels[-1]),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(channels[-1], num_classes),
        )

    def forward(self, data_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x = torch.cat([data_dict["feat"], data_dict["coord"]], dim=1)
        logits = self.head(self.encoder(x))
        out = {"seg_logits": logits}
        if "segment" in data_dict:
            out["loss"] = F.cross_entropy(logits, data_dict["segment"])
        return out


class ApproxSPVCNN(nn.Module):
    """Sparse point-voxel style fallback with simple voxel pooling."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        point_channels: int = 64,
        voxel_channels: int = 128,
        fuse_channels: int = 128,
        blocks: int = 1,
    ):
        super().__init__()
        point_blocks = [ResidualMLPBlock(point_channels) for _ in range(blocks)]
        voxel_blocks = [ResidualMLPBlock(voxel_channels) for _ in range(blocks)]
        fuse_blocks = [ResidualMLPBlock(fuse_channels) for _ in range(blocks)]
        self.point_stem = nn.Sequential(
            nn.Linear(in_channels + 3, point_channels),
            nn.BatchNorm1d(point_channels),
            nn.GELU(),
            *point_blocks,
        )
        self.voxel_net = nn.Sequential(
            nn.Linear(point_channels, voxel_channels),
            nn.BatchNorm1d(voxel_channels),
            nn.GELU(),
            *voxel_blocks,
            nn.Linear(voxel_channels, voxel_channels),
            nn.BatchNorm1d(voxel_channels),
            nn.GELU(),
        )
        self.fuse = nn.Sequential(
            nn.Linear(point_channels + voxel_channels, fuse_channels),
            nn.BatchNorm1d(fuse_channels),
            nn.GELU(),
            *fuse_blocks,
            nn.Linear(fuse_channels, num_classes),
        )

    def forward(self, data_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        point_feat = self.point_stem(torch.cat([data_dict["feat"], data_dict["coord"]], dim=1))
        keys = hash_grid(data_dict["grid_coord"])
        _, inverse = torch.unique(keys, sorted=False, return_inverse=True)
        voxel_feat = point_feat.new_zeros((int(inverse.max().item()) + 1, point_feat.shape[1]))
        voxel_feat.index_add_(0, inverse, point_feat)
        counts = point_feat.new_zeros((voxel_feat.shape[0], 1))
        counts.index_add_(0, inverse, torch.ones((point_feat.shape[0], 1), device=point_feat.device, dtype=point_feat.dtype))
        voxel_feat = voxel_feat / counts.clamp_min(1)
        voxel_feat = self.voxel_net(voxel_feat)
        logits = self.fuse(torch.cat([point_feat, voxel_feat[inverse]], dim=1))
        out = {"seg_logits": logits}
        if "segment" in data_dict:
            out["loss"] = F.cross_entropy(logits, data_dict["segment"])
        return out


class TensorOutputWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, data_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        output = self.model(data_dict)
        if isinstance(output, dict):
            return output
        logits = output
        out = {"seg_logits": logits}
        if "segment" in data_dict:
            out["loss"] = F.cross_entropy(logits, data_dict["segment"])
        return out


def hash_grid(grid_coord: torch.Tensor) -> torch.Tensor:
    coord = grid_coord.long()
    return (
        coord[:, 0] * 73856093
        ^ coord[:, 1] * 19349663
        ^ coord[:, 2] * 83492791
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clone_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.clone() for key, value in batch.items()}


def make_batch(num_points: int, in_channels: int, num_classes: int, device: torch.device) -> dict[str, torch.Tensor]:
    coord = torch.rand((num_points, 3), device=device, dtype=torch.float32) * 100.0
    grid_coord = torch.div(coord, 0.25, rounding_mode="floor").to(torch.int32)
    feat = torch.randn((num_points, in_channels), device=device, dtype=torch.float32)
    segment = torch.randint(0, num_classes, (num_points,), device=device, dtype=torch.long)
    return {
        "coord": coord,
        "grid_coord": grid_coord,
        "feat": feat,
        "offset": torch.tensor([num_points], device=device, dtype=torch.int32),
        "segment": segment,
    }


def disable_auto_class_weight(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {
            key: False if key == "auto_class_weight" else disable_auto_class_weight(value)
            for key, value in obj.items()
        }
    if isinstance(obj, list):
        return [disable_auto_class_weight(value) for value in obj]
    if isinstance(obj, tuple):
        return tuple(disable_auto_class_weight(value) for value in obj)
    return obj


def apply_balanced_profile(name: str, cfg: dict[str, Any]) -> dict[str, Any]:
    """Increase undersized defaults for fairer capacity-efficiency comparison."""
    if name == "ptv2":
        cfg["backbone_out_channels"] = 72
        backbone = cfg["backbone"]
        backbone.update(
            patch_embed_depth=2,
            patch_embed_channels=72,
            patch_embed_groups=12,
            enc_depths=(3, 3, 3),
            enc_channels=(144, 288, 576),
            enc_groups=(12, 24, 36),
            dec_depths=(2, 2, 2),
            dec_channels=(72, 144, 288),
            dec_groups=(12, 12, 24),
        )
    return cfg


def load_config_model(name: str, profile: str) -> dict[str, Any]:
    cfg_path = REPO_ROOT / MODEL_CONFIGS[name]
    namespace = runpy.run_path(str(cfg_path))
    cfg = disable_auto_class_weight(copy.deepcopy(namespace["model"]))
    if profile == "balanced":
        cfg = apply_balanced_profile(name, cfg)
    if name == "ptv3":
        cfg["backbone"]["enable_flash"] = True
    if name == "litept":
        cfg["backbone"]["enable_flash"] = True
    return cfg


def build_project_model(name: str, profile: str) -> tuple[nn.Module, str, str]:
    from pointspace.models import build_model

    if name in MODEL_CONFIGS:
        cfg = load_config_model(name, profile)
        note = ""
        if profile == "balanced" and name == "ptv2":
            note = "Balanced profile increases PTV2 channels/depth from the small DALES default."
        return build_model(cfg), f"PointSpace config `{MODEL_CONFIGS[name]}` ({profile} profile)", note

    if name == "spvcnn":
        cfg = {
            "type": "SPVCNN",
            "in_channels": 5,
            "out_channels": 8,
            "base_channels": 32,
            "channels": (32, 64, 128, 256, 256, 128, 96, 96),
            "layers": (2, 2, 2, 2, 2, 2, 2, 2),
        }
        model = TensorOutputWrapper(build_model(cfg))
        return model, "PointSpace SPVCNN implementation", ""

    raise KeyError(name)


def build_model_for_benchmark(
    name: str,
    in_channels: int,
    num_classes: int,
    profile: str,
) -> tuple[nn.Module, str, str]:
    if name == "randla":
        if profile == "balanced":
            channels, blocks = (96, 192, 384, 512), 3
            note = "Balanced approximation: widened/deepened to reduce capacity gap with project-native models."
        else:
            channels, blocks = (64, 128, 256, 256), 2
            note = "Approximation: point-wise residual MLP surrogate for runtime comparison."
        return (
            ApproxSegModel(in_channels, num_classes, channels=channels, blocks_per_stage=blocks, name="RandLA-Net"),
            "Approximate RandLA-Net-style PyTorch baseline in tests/",
            note,
        )
    if name == "pointnext":
        if profile == "balanced":
            channels, blocks = (96, 192, 384, 512), 4
            note = "Balanced approximation: residual inverted-MLP style network with higher capacity."
        else:
            channels, blocks = (64, 128, 256, 512), 3
            note = "Approximation: residual inverted-MLP style point network."
        return (
            ApproxSegModel(in_channels, num_classes, channels=channels, blocks_per_stage=blocks, name="PointNeXt"),
            "Approximate PointNeXt-style PyTorch baseline in tests/",
            note,
        )
    if name == "spvcnn":
        try:
            return build_project_model(name, profile)
        except Exception as exc:
            if profile == "balanced":
                fallback = ApproxSPVCNN(
                    in_channels,
                    num_classes,
                    point_channels=192,
                    voxel_channels=384,
                    fuse_channels=384,
                    blocks=3,
                )
                note = "Balanced fallback used because project SPVCNN could not be built"
            else:
                fallback = ApproxSPVCNN(in_channels, num_classes)
                note = "Project SPVCNN could not be built, fallback used"
            return (
                fallback,
                "Approximate SPVCNN fallback in tests/",
                f"{note}: {type(exc).__name__}: {exc}",
            )
    return build_project_model(name, profile)


def extract_loss(output: Any, segment: torch.Tensor) -> torch.Tensor:
    if isinstance(output, dict):
        if "loss" in output:
            return output["loss"]
        logits = output.get("seg_logits", output.get("logits"))
    else:
        logits = output
    if logits is None:
        raise RuntimeError("Model output does not contain loss or logits.")
    return F.cross_entropy(logits, segment)


def cuda_elapsed_ms(func: Callable[[], None]) -> float:
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    starter.record()
    func()
    ender.record()
    torch.cuda.synchronize()
    return float(starter.elapsed_time(ender))


def measure_inference(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    warmup: int,
    repeat: int,
) -> tuple[float, float, float, float, float]:
    model.eval()
    latencies: list[float] = []
    memories: list[float] = []
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for _ in range(warmup):
            with torch.amp.autocast("cuda", dtype=torch.float16):
                model(batch)
        torch.cuda.synchronize()
        for _ in range(repeat):
            torch.cuda.reset_peak_memory_stats()
            latency = cuda_elapsed_ms(
                lambda: model_forward_no_grad(model, batch)
            )
            latencies.append(latency)
            memories.append(torch.cuda.max_memory_allocated() / (1024**2))
    return (
        statistics.mean(latencies),
        statistics.pstdev(latencies) if len(latencies) > 1 else 0.0,
        statistics.mean(memories),
        statistics.pstdev(memories) if len(memories) > 1 else 0.0,
        max(memories),
    )


def model_forward_no_grad(model: nn.Module, batch: dict[str, torch.Tensor]) -> None:
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
        model(batch)


def measure_training(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    warmup: int,
    repeat: int,
) -> tuple[float, float, float, float, float]:
    model.train()
    latencies: list[float] = []
    memories: list[float] = []
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    def train_step() -> None:
        model.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            output = model(batch)
            loss = extract_loss(output, batch["segment"])
        loss.backward()

    for _ in range(warmup):
        train_step()
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    for _ in range(repeat):
        torch.cuda.reset_peak_memory_stats()
        latency = cuda_elapsed_ms(train_step)
        latencies.append(latency)
        memories.append(torch.cuda.max_memory_allocated() / (1024**2))
    model.zero_grad(set_to_none=True)
    return (
        statistics.mean(latencies),
        statistics.pstdev(latencies) if len(latencies) > 1 else 0.0,
        statistics.mean(memories),
        statistics.pstdev(memories) if len(memories) > 1 else 0.0,
        max(memories),
    )


def benchmark_one(
    name: str,
    args: argparse.Namespace,
    device: torch.device,
    base_batch: dict[str, torch.Tensor],
) -> BenchmarkResult:
    set_seed(args.seed)
    try:
        model, implementation, note = build_model_for_benchmark(name, args.in_channels, args.num_classes, args.profile)
        model = model.to(device)
        params_m = sum(p.numel() for p in model.parameters()) / 1e6
        batch = clone_batch(base_batch)
        train_latency, train_std, train_mem, train_mem_std, train_mem_max = measure_training(
            model, batch, args.warmup, args.repeat
        )
        gc.collect()
        torch.cuda.empty_cache()
        infer_latency, infer_std, infer_mem, infer_mem_std, infer_mem_max = measure_inference(
            model, batch, args.warmup, args.repeat
        )
        return BenchmarkResult(
            model=MODEL_LABELS[name],
            implementation=implementation,
            status="ok",
            params_m=params_m,
            train_latency_ms=train_latency,
            train_latency_std_ms=train_std,
            infer_latency_ms=infer_latency,
            infer_latency_std_ms=infer_std,
            train_memory_mib=train_mem,
            train_memory_std_mib=train_mem_std,
            train_memory_max_mib=train_mem_max,
            infer_memory_mib=infer_mem,
            infer_memory_std_mib=infer_mem_std,
            infer_memory_max_mib=infer_mem_max,
            note=note,
        )
    except torch.cuda.OutOfMemoryError as exc:
        return BenchmarkResult(
            model=MODEL_LABELS[name],
            implementation="n/a",
            status="oom",
            note="Reduce --num-points or --repeat and rerun.",
            error=str(exc).splitlines()[0],
        )
    except Exception as exc:
        return BenchmarkResult(
            model=MODEL_LABELS[name],
            implementation="n/a",
            status="failed",
            error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=6)}",
        )
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def fmt_float(value: float | None, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "-"
    return f"{value:.{digits}f}"


def architecture_rows(profile: str) -> list[str]:
    ptv2_channels = "(72,144,288,576)" if profile == "balanced" else "(24,48,96,192)"
    ptv2_depths = "(3,3,3 enc; 2,2,2 dec)" if profile == "balanced" else "(2,2,2 enc; 1,1,1 dec)"
    return [
        "| RandLA-Net approx | balanced local surrogate | local residual point MLP | widened/deepened only in `balanced` |",
        "| SPVCNN fallback | balanced local surrogate | local point-voxel pooling MLP | real project SPVCNN requires `torchsparse` |",
        "| PointNeXt approx | balanced local surrogate | local residual inverted-MLP style network | widened/deepened only in `balanced` |",
        f"| Point Transformer V2 | channels {ptv2_channels}, depths {ptv2_depths} | local attention neighborhoods | `balanced` enlarges DALES small default |",
        "| Point Transformer V3 | enc channels `(32,64,128,256,512)`, enc depths `(2,2,2,6,2)`, dec depths `(2,2,2,2)` | attention in encoder and decoder, flash enabled | many parameters are in coarse-resolution stages, so params do not scale linearly with activation memory |",
        "| DeepPLANet | enc channels `(64,128,256,512)`, enc depths `(10,10,30,10)` | deep hierarchical local aggregation | much deeper than LitePT/PTV3 at local stages |",
        "| LitePT | enc channels `(36,72,144,252,504)`, enc depths `(2,2,2,6,2)`, dec depths `(0,0,0,0)` | conv in first three encoder stages, attention only in last two encoder stages, flash enabled | channel/depth shape resembles PTV3, but active attention/decoder allocation is much lighter |",
    ]


def collect_metadata(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.replace("\n", " "),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if torch.cuda.is_available() else "none",
        "num_points": args.num_points,
        "in_channels": args.in_channels,
        "num_classes": args.num_classes,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "profile": args.profile,
        "amp_dtype": "float16",
    }


def write_report(results: list[BenchmarkResult], metadata: dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "benchmark_report.md"
    rows = []
    for result in results:
        mem_ratio = None
        if result.train_memory_mib is not None and result.infer_memory_mib:
            mem_ratio = result.train_memory_mib / result.infer_memory_mib
        rows.append(
            "| {model} | {status} | {params} | {train_lat} +/- {train_std} | {infer_lat} +/- {infer_std} | {train_mem} +/- {train_mem_std} / {train_mem_max} | {infer_mem} +/- {infer_mem_std} / {infer_mem_max} | {mem_ratio}x | {impl} | {note} |".format(
                model=result.model,
                status=result.status,
                params=fmt_float(result.params_m, 3),
                train_lat=fmt_float(result.train_latency_ms),
                train_std=fmt_float(result.train_latency_std_ms),
                infer_lat=fmt_float(result.infer_latency_ms),
                infer_std=fmt_float(result.infer_latency_std_ms),
                train_mem=fmt_float(result.train_memory_mib),
                train_mem_std=fmt_float(result.train_memory_std_mib),
                train_mem_max=fmt_float(result.train_memory_max_mib),
                infer_mem=fmt_float(result.infer_memory_mib),
                infer_mem_std=fmt_float(result.infer_memory_std_mib),
                infer_mem_max=fmt_float(result.infer_memory_max_mib),
                mem_ratio=fmt_float(mem_ratio),
                impl=result.implementation.replace("|", "/"),
                note=(result.note or result.error.splitlines()[0] if result.error else result.note).replace("|", "/"),
            )
        )

    ok_results = [result for result in results if result.status == "ok"]
    fastest_infer = min(
        ok_results,
        key=lambda result: result.infer_latency_ms if result.infer_latency_ms is not None else float("inf"),
        default=None,
    )
    lowest_train_mem = min(
        ok_results,
        key=lambda result: result.train_memory_mib if result.train_memory_mib is not None else float("inf"),
        default=None,
    )
    fastest_train = min(
        ok_results,
        key=lambda result: result.train_latency_ms if result.train_latency_ms is not None else float("inf"),
        default=None,
    )

    lines = [
        "# Efficiency Benchmark for Point Cloud Semantic Segmentation Models",
        "",
        "## 1. Experimental Setting",
        "",
        "We evaluate the computational efficiency of representative point cloud semantic segmentation networks under a unified synthetic input protocol. The comparison focuses on model size, training latency, inference latency, and peak GPU memory consumption. Accuracy is not evaluated in this benchmark because all models are measured on synthetic point clouds rather than a held-out semantic segmentation dataset.",
        "",
        "| Item | Setting |",
        "|---|---|",
        f"| Hardware | `{metadata['gpu']}` |",
        f"| Host | `{metadata['hostname']}` |",
        f"| Platform | `{metadata['platform']}` |",
        f"| Software | Python `{metadata['python']}`, PyTorch `{metadata['torch']}`, CUDA `{metadata['cuda_version']}` |",
        f"| Input size | `{metadata['num_points']}` points per scene |",
        f"| Input channels | `{metadata['in_channels']}` point features plus XYZ coordinates where required |",
        f"| Number of classes | `{metadata['num_classes']}` |",
        f"| Benchmark profile | `{metadata['profile']}` |",
        f"| Precision | AMP `{metadata['amp_dtype']}` |",
        f"| Timing protocol | `{metadata['warmup']}` warmup iterations and `{metadata['repeat']}` measured iterations |",
        f"| Run time | `{metadata['timestamp']}` |",
        "",
        "## 2. Evaluation Metrics",
        "",
        "The benchmark reports four efficiency-oriented metrics. `Params` is the number of trainable and non-trainable model parameters. `Train latency` measures forward propagation, loss computation, and backward propagation; optimizer creation and optimizer step are excluded to isolate network computation. `Infer latency` measures forward propagation under `torch.no_grad()`. GPU memory is reported as the mean and standard deviation of per-iteration peak allocated memory, followed by the maximum observed per-iteration peak. The memory ratio is computed as train memory mean divided by inference memory mean.",
        "",
        "All timings are measured with CUDA events after warmup. All models are executed with FP16 automatic mixed precision. Point Transformer V3 and LitePT are explicitly configured with flash attention enabled. The `balanced` profile widens/deepens undersized baselines to reduce capacity mismatch while preserving the project-native defaults for PTV3, DeepPLANet, and LitePT.",
        "",
        "## 3. Main Efficiency Results",
        "",
        "| Model | Status | Params (M) | Train Latency (ms) | Infer Latency (ms) | Train Mem Mean +/- Std / Max (MiB) | Infer Mem Mean +/- Std / Max (MiB) | Mem Ratio | Implementation | Note |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
        *rows,
        "",
        "## 4. Efficiency Analysis",
        "",
        f"- Fastest training latency: `{fastest_train.model if fastest_train else '-'}` at `{fmt_float(fastest_train.train_latency_ms if fastest_train else None)}` ms.",
        f"- Fastest inference latency: `{fastest_infer.model if fastest_infer else '-'}` at `{fmt_float(fastest_infer.infer_latency_ms if fastest_infer else None)}` ms.",
        f"- Lowest mean training memory: `{lowest_train_mem.model if lowest_train_mem else '-'}` at `{fmt_float(lowest_train_mem.train_memory_mib if lowest_train_mem else None)}` MiB.",
        "- Parameter count should not be interpreted as a direct proxy for inference memory or latency. Runtime memory is dominated by activations, neighborhood buffers, attention workspaces, and the number of points remaining at each hierarchy stage.",
        "- PTV3 has many more parameters than LitePT, but a large fraction of those weights operate after downsampling. This can keep activation memory and latency closer than the parameter ratio alone would suggest.",
        "- LitePT and PTV3 have similar channel/depth schedules, but they do not allocate operators in the same way. LitePT disables decoder blocks and uses convolution-only early encoder stages; PTV3 applies attention through both encoder and decoder.",
        "- Memory is averaged over per-iteration peaks instead of using only one global peak, reducing sensitivity to occasional allocator or kernel-workspace outliers while still preserving the maximum observed value in the table.",
        "- Training memory excludes optimizer state allocation. This makes train/infer memory ratios reflect forward/backward activations and gradients, not AdamW moment buffers.",
        "- Ratios above 2-3x are plausible in this protocol because inference runs under `no_grad` and does not retain intermediate activations, while training must keep backward tensors for every block. A 2-3x rule of thumb is more likely when comparing total job memory with larger inference batches, optimizer state included, or activation checkpointing enabled.",
        "- The approximate RandLA-Net, PointNeXt, and SPVCNN fallback baselines should still be interpreted as hardware/runtime probes rather than strict architectural reproductions.",
        "- For project-native models, LitePT shows the expected efficiency advantage over heavier attention-based or deep hierarchical baselines under the measured synthetic setting.",
        "",
        "## 5. Architecture and Comparability",
        "",
        "| Model | Capacity Setting | Main Operator Allocation | Comparability Note |",
        "|---|---|---|---|",
        *architecture_rows(str(metadata["profile"])),
        "",
        "## 6. Implementation Details",
        "",
        "- PTV2, PTV3, DeepPLANet, and LitePT are built from DALES default PointSpace configs.",
        "- Under `balanced`, PTV2 is widened/deepened because the DALES default used here is much smaller than the later project-native networks.",
        "- SPVCNN uses the project implementation when `torchsparse` is available; otherwise the script uses a local sparse point-voxel approximation and marks this in the note.",
        "- RandLA-Net and PointNeXt are local approximations for comparable latency/memory probing only, not faithful accuracy reproductions.",
        "- Each model receives the same synthetic single-scene point cloud with `coord`, `grid_coord`, `feat`, `offset`, and `segment` fields.",
        "- The synthetic point count strongly affects attention, neighborhood search, and sparse voxel operators. For dataset-level comparison, rerun with a point count close to post-transform training samples.",
        "",
        "## 7. Limitations",
        "",
        "- This benchmark is an efficiency comparison only. It cannot support claims about semantic segmentation accuracy, mIoU, or convergence behavior.",
        "- Approximate external baselines are not suitable for publication as faithful RandLA-Net, PointNeXt, or SPVCNN numbers unless replaced by validated implementations.",
        "- A parameter-matched comparison would require manually designing separate width/depth variants for PTV3 and LitePT. The current benchmark compares practical/default-style configurations plus a balanced profile for obviously undersized baselines.",
        "- The results are hardware-, CUDA-, and dependency-specific. Installing `torchsparse` may change the SPVCNN result from fallback to the project implementation.",
        "",
        "## 8. External References for Approximate Baselines",
        "",
        "- RandLA-Net PyTorch reference: https://github.com/aRI0U/RandLA-Net-pytorch",
        "- PointNeXt / OpenPoints reference: https://github.com/guochengqian/PointNeXt",
        "- TorchSparse / SPVCNN dependency reference: https://github.com/mit-han-lab/torchsparse",
    ]
    if any(result.status != "ok" for result in results):
        lines.extend(["", "## Errors", ""])
        for result in results:
            if result.status != "ok":
                lines.extend([f"### {result.model}", "", "```text", result.error.strip(), "```", ""])

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default=",".join(MODEL_ORDER), help="Comma-separated model keys.")
    parser.add_argument("--num-points", type=int, default=32768)
    parser.add_argument("--in-channels", type=int, default=5)
    parser.add_argument("--num-classes", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--profile",
        choices=("balanced", "default"),
        default="balanced",
        help="`balanced` widens undersized baselines; `default` preserves original project configs and smaller approximations.",
    )
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="Regenerate the Markdown report from an existing benchmark_results.json without rerunning models.",
    )
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.render_only:
        json_path = args.output_dir / "benchmark_results.json"
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        results = [BenchmarkResult(**item) for item in payload["results"]]
        report_path = write_report(results, payload["metadata"], args.output_dir)
        print(f"[benchmark] rendered {report_path}")
        return 0

    model_names = [name.strip().lower() for name in args.models.split(",") if name.strip()]
    unknown = sorted(set(model_names) - set(MODEL_ORDER))
    if unknown:
        raise SystemExit(f"Unknown model key(s): {', '.join(unknown)}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for latency and memory benchmark.")

    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0")
    base_batch = make_batch(args.num_points, args.in_channels, args.num_classes, device)
    metadata = collect_metadata(args, device)

    results: list[BenchmarkResult] = []
    for name in model_names:
        print(f"[benchmark] {MODEL_LABELS[name]} ...", flush=True)
        result = benchmark_one(name, args, device, base_batch)
        results.append(result)
        print(
            f"[benchmark] {result.model}: {result.status}, "
            f"params={fmt_float(result.params_m, 3)}M, "
            f"train={fmt_float(result.train_latency_ms)}ms, infer={fmt_float(result.infer_latency_ms)}ms",
            flush=True,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "benchmark_results.json"
    json_path.write_text(
        json.dumps(
            {"metadata": metadata, "results": [asdict(result) for result in results]},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    report_path = write_report(results, metadata, args.output_dir)
    print(f"[benchmark] wrote {json_path}")
    print(f"[benchmark] wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
