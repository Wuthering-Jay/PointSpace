"""流式审计 HPSD 训练数据在目标层级的正射可视监督覆盖。

该工具按训练配置执行真实 transform 和 encoder pooling，只保存计数、直方图与
有界优先级样本，不保留全部逐点/逐 token 特征。它不会运行 HPSD projector，
也不会修改 checkpoint 或数据文件。
"""

import argparse
import json
import random
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from pointspace.datasets import build_dataset, point_collate_fn
from pointspace.models import build_model
from pointspace.models.backbone.hpsd.hpsd_v1m1 import build_token_patch_edges
from pointspace.models.backbone.vrsr.ops import (
    aggregate_patch_teacher_to_tokens,
    compute_token_visibility,
)
from pointspace.models.utils.misc import offset2batch
from pointspace.utils.config import Config
from pointspace.utils.logger import get_root_logger


class PriorityReservoir:
    """用随机优先级维护固定容量的无偏流式样本。"""

    def __init__(self, capacity, seed=42):
        self.capacity = int(capacity)
        self.rng = np.random.default_rng(seed)
        self.values = np.empty(0, dtype=np.float32)
        self.priority = np.empty(0, dtype=np.float64)

    def update(self, values):
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        if values.size == 0 or self.capacity <= 0:
            return
        priority = self.rng.random(values.size)
        values = np.concatenate((self.values, values))
        priority = np.concatenate((self.priority, priority))
        if values.size > self.capacity:
            keep = np.argpartition(priority, -self.capacity)[-self.capacity :]
            values = values[keep]
            priority = priority[keep]
        self.values = values
        self.priority = priority

    def quantiles(self):
        if self.values.size == 0:
            return {}
        probability = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)
        value = np.quantile(self.values, probability)
        return {f"p{int(p * 100):02d}": float(v) for p, v in zip(probability, value)}


class VisibilityAuditor:
    """按真实训练数据流统计点级和 HPSD token 级可视性。"""

    def __init__(
        self,
        config_file,
        output_dir,
        device="cuda",
        batch_size=1,
        num_workers=0,
        max_samples=None,
        reservoir_size=200000,
        source_q=0.6,
        source_purity=None,
        min_source_points=4,
        min_source_patches=1,
        seed=42,
    ):
        self.config_file = Path(config_file)
        self.output_dir = Path(output_dir)
        self.device = torch.device(
            device if device != "cuda" or torch.cuda.is_available() else "cpu"
        )
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.max_samples = None if max_samples is None else int(max_samples)
        self.source_q = float(source_q)
        self.source_purity = None if source_purity is None else float(source_purity)
        self.min_source_points = int(min_source_points)
        self.min_source_patches = int(min_source_patches)
        self.seed = int(seed)
        self.logger = get_root_logger()
        self.visibility_sample = PriorityReservoir(reservoir_size, seed)
        self.purity_sample = PriorityReservoir(reservoir_size, seed + 1)
        self.patch_count_sample = PriorityReservoir(reservoir_size, seed + 2)
        self.point_support_sample = PriorityReservoir(reservoir_size, seed + 3)

    def _build(self):
        cfg = Config.fromfile(str(self.config_file))
        cfg.data.train.loop = 1
        dataset = build_dataset(cfg.data.train)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.device.type == "cuda",
            collate_fn=partial(point_collate_fn, mix_prob=0.0),
            generator=torch.Generator().manual_seed(self.seed),
        )
        model = build_model(cfg.model).to(self.device).eval()
        if not isinstance(model.distill_level, int):
            raise TypeError("Configured model does not expose an integer distill_level")
        return cfg, dataset, loader, model

    @staticmethod
    def _to_device(batch, device):
        return {
            key: value.to(device, non_blocking=True)
            if isinstance(value, torch.Tensor)
            else value
            for key, value in batch.items()
        }

    def process(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        cfg, dataset, loader, model = self._build()
        sample_limit = len(dataset)
        if self.max_samples is not None:
            sample_limit = min(sample_limit, self.max_samples)

        total = dict(
            samples=0,
            points=0,
            valid_points=0,
            tokens=0,
            fully_invisible_tokens=0,
            mixed_tokens=0,
            fully_visible_tokens=0,
            source_tokens=0,
            samples_with_source=0,
            samples_with_target=0,
            samples_with_source_and_target=0,
            teacher_tokens=0,
            token_patch_edges=0,
        )
        visibility_hist = np.zeros(20, dtype=np.int64)
        purity_hist = np.zeros(20, dtype=np.int64)
        height_token_count = np.zeros(10, dtype=np.int64)
        height_visibility_sum = np.zeros(10, dtype=np.float64)
        height_invisible_count = np.zeros(10, dtype=np.int64)

        progress = tqdm(total=sample_limit, desc="Auditing HPSD visibility", unit="tile")
        with torch.inference_mode():
            for batch in loader:
                remaining = sample_limit - total["samples"]
                if remaining <= 0:
                    break
                batch = self._to_device(batch, self.device)
                # 只运行 encoder hierarchy；不创建 concat/projector 的训练激活。
                _, hierarchy = model.backbone(batch, return_hierarchy=True)
                level = hierarchy[model.distill_level]
                num_tokens = int(level.point.feat.shape[0])
                visibility, valid_count, point_count = compute_token_visibility(
                    level.input_to_level,
                    batch["dino_valid"],
                    num_tokens,
                )
                edges = build_token_patch_edges(
                    level.input_to_level,
                    batch["dino_patch_index"],
                    batch["dino_valid"],
                    num_tokens=num_tokens,
                    num_patches=batch["dino_feature"].shape[0],
                    validate_mapping=False,
                )
                teacher = torch.nn.functional.normalize(
                    batch["dino_feature"].float(), dim=-1, eps=1e-12
                )
                stats = aggregate_patch_teacher_to_tokens(
                    teacher, edges, edge_weight=model.edge_weight
                )
                source_keep = visibility[stats.token] >= self.source_q
                source_keep &= stats.point_count >= self.min_source_points
                source_keep &= stats.patch_count >= self.min_source_patches
                if self.source_purity is not None:
                    source_keep &= stats.purity >= self.source_purity
                source_token = stats.token[source_keep]

                input_batch = offset2batch(batch["offset"].long())
                token_batch = level.point.batch.long()
                batch_samples = int(batch["offset"].shape[0])
                if batch_samples > remaining:
                    # max_samples 只在最后一个 batch 截断统计；通常 batch_size=1。
                    batch_samples = remaining
                for sample in range(batch_samples):
                    point_mask = input_batch == sample
                    token_mask = token_batch == sample
                    point_num = int(point_mask.sum())
                    token_num = int(token_mask.sum())
                    valid_num = int(batch["dino_valid"][point_mask].sum())
                    q = visibility[token_mask]
                    target_num = int((q == 0).sum())
                    source_num = int((token_batch[source_token] == sample).sum())

                    # 在每个 tile 内按 robust z 范围归一化后统计十个高度层，
                    # 避免不同绝对高程的 tile 在全局分桶时彼此错位。
                    token_z = level.point.coord[token_mask, 2].float()
                    if token_z.numel() > 1:
                        z_low, z_high = torch.quantile(
                            token_z, token_z.new_tensor([0.05, 0.95])
                        )
                        z_scale = (z_high - z_low).clamp_min(1e-6)
                        height_bin = torch.floor(
                            ((token_z - z_low) / z_scale).clamp(0, 0.999999) * 10
                        ).long()
                    else:
                        height_bin = torch.zeros_like(token_z, dtype=torch.long)
                    bin_count = torch.bincount(height_bin, minlength=10)
                    bin_q_sum = torch.bincount(
                        height_bin, weights=q.float(), minlength=10
                    )
                    bin_invisible = torch.bincount(
                        height_bin, weights=(q == 0).float(), minlength=10
                    )
                    height_token_count += bin_count.cpu().numpy()
                    height_visibility_sum += bin_q_sum.cpu().numpy()
                    height_invisible_count += bin_invisible.long().cpu().numpy()

                    total["samples"] += 1
                    total["points"] += point_num
                    total["valid_points"] += valid_num
                    total["tokens"] += token_num
                    total["fully_invisible_tokens"] += target_num
                    total["mixed_tokens"] += int(((q > 0) & (q < 1)).sum())
                    total["fully_visible_tokens"] += int((q == 1).sum())
                    total["source_tokens"] += source_num
                    total["samples_with_source"] += int(source_num > 0)
                    total["samples_with_target"] += int(target_num > 0)
                    total["samples_with_source_and_target"] += int(
                        source_num > 0 and target_num > 0
                    )
                    progress.update(1)

                total["teacher_tokens"] += int(stats.token.numel())
                total["token_patch_edges"] += edges.num_edges
                q_cpu = visibility.detach().cpu().numpy()
                visibility_hist += np.histogram(q_cpu, bins=20, range=(0.0, 1.0))[0]
                self.visibility_sample.update(q_cpu)
                if stats.purity.numel() > 0:
                    purity_cpu = stats.purity.detach().cpu().numpy()
                    purity_hist += np.histogram(
                        purity_cpu, bins=20, range=(-1.0, 1.0)
                    )[0]
                    self.purity_sample.update(purity_cpu)
                    self.patch_count_sample.update(stats.patch_count.cpu().numpy())
                    self.point_support_sample.update(stats.point_count.cpu().numpy())
                if total["samples"] >= sample_limit:
                    break
        progress.close()

        denominator = max(total["samples"], 1)
        point_denominator = max(total["points"], 1)
        token_denominator = max(total["tokens"], 1)
        report = dict(
            config=str(self.config_file.resolve()),
            device=str(self.device),
            thresholds=dict(
                source_q=self.source_q,
                source_purity=self.source_purity,
                min_source_points=self.min_source_points,
                min_source_patches=self.min_source_patches,
            ),
            counts=total,
            ratios=dict(
                point_valid=total["valid_points"] / point_denominator,
                token_fully_invisible=total["fully_invisible_tokens"] / token_denominator,
                token_mixed=total["mixed_tokens"] / token_denominator,
                token_fully_visible=total["fully_visible_tokens"] / token_denominator,
                sample_with_source=total["samples_with_source"] / denominator,
                sample_with_target=total["samples_with_target"] / denominator,
                sample_with_source_and_target=total[
                    "samples_with_source_and_target"
                ]
                / denominator,
            ),
            quantiles=dict(
                visibility=self.visibility_sample.quantiles(),
                teacher_purity=self.purity_sample.quantiles(),
                teacher_patch_count=self.patch_count_sample.quantiles(),
                teacher_point_support=self.point_support_sample.quantiles(),
            ),
            histograms=dict(
                visibility=dict(range=[0.0, 1.0], bins=visibility_hist.tolist()),
                teacher_purity=dict(range=[-1.0, 1.0], bins=purity_hist.tolist()),
            ),
            height_profile=[
                dict(
                    normalized_z_bin=[index / 10, (index + 1) / 10],
                    token_count=int(height_token_count[index]),
                    mean_visibility=float(
                        height_visibility_sum[index]
                        / max(height_token_count[index], 1)
                    ),
                    fully_invisible_ratio=float(
                        height_invisible_count[index]
                        / max(height_token_count[index], 1)
                    ),
                )
                for index in range(10)
            ],
        )
        json_path = self.output_dir / "visibility_audit.json"
        json_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        markdown_path = self.output_dir / "visibility_audit.md"
        markdown_path.write_text(self._to_markdown(report), encoding="utf-8")
        self.logger.info(f"Visibility audit saved to {self.output_dir}")
        return report

    @staticmethod
    def _to_markdown(report):
        count = report["counts"]
        ratio = report["ratios"]
        threshold = report["thresholds"]
        lines = [
            "# HPSD 可视监督覆盖审计",
            "",
            f"- 样本数：{count['samples']}",
            f"- 点级 valid 比例：{ratio['point_valid']:.4f}",
            f"- level token 数：{count['tokens']}",
            f"- fully-invisible token 比例：{ratio['token_fully_invisible']:.4f}",
            f"- mixed token 比例：{ratio['token_mixed']:.4f}",
            f"- 同时具有 source/target 的样本比例：{ratio['sample_with_source_and_target']:.4f}",
            "",
            "## Source 条件",
            "",
            f"`q >= {threshold['source_q']}`，最小支持点数 "
            f"`{threshold['min_source_points']}`，最小 patch 数 "
            f"`{threshold['min_source_patches']}`，purity 阈值 "
            f"`{threshold['source_purity']}`。",
            "",
            "## 分位数",
            "",
            "```json",
            json.dumps(report["quantiles"], ensure_ascii=False, indent=2),
            "```",
            "",
            "## 归一化高度层",
            "",
            "| z 区间 | token 数 | 平均可视率 | fully-invisible 比例 |",
            "| --- | ---: | ---: | ---: |",
        ]
        for item in report["height_profile"]:
            low, high = item["normalized_z_bin"]
            lines.append(
                f"| {low:.1f}-{high:.1f} | {item['token_count']} | "
                f"{item['mean_visibility']:.4f} | "
                f"{item['fully_invisible_ratio']:.4f} |"
            )
        lines.append("")
        return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(description="Audit HPSD visibility coverage")
    parser.add_argument(
        "--config-file",
        default=r"configs\hpsd\pretrain-hpsd-litept-v1m4-hubei.py",
    )
    parser.add_argument("--output-dir", default=r"exp\hpsd_visibility_audit")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--reservoir-size", type=int, default=200000)
    parser.add_argument("--source-q", type=float, default=0.6)
    parser.add_argument("--source-purity", type=float, default=None)
    parser.add_argument("--min-source-points", type=int, default=4)
    parser.add_argument("--min-source-patches", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    VisibilityAuditor(
        config_file=args.config_file,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
        reservoir_size=args.reservoir_size,
        source_q=args.source_q,
        source_purity=args.source_purity,
        min_source_points=args.min_source_points,
        min_source_patches=args.min_source_patches,
        seed=args.seed,
    ).process()
