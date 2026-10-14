import json
from pathlib import Path

import pytest
import torch

from pointspace.datasets import build_dataset, point_collate_fn
from pointspace.models import build_model
from pointspace.models.losses.misc import CrossEntropyLoss
from pointspace.utils.config import Config
from utils.create_sparse_label_manifests import COLLAPSE_FRACTIONS


CONFIG_DIR = Path("configs/label_efficiency/dales/scratch_collapse")
MANIFEST = Path(r"E:\data\DALES\dales_las\tile\sparse_labels_seed42\summary.json")


def test_five_scratch_collapse_configs_match_ft_protocol():
    paths = sorted(CONFIG_DIR.glob("*.py"))
    assert len(paths) == 5
    seen = set()
    save_paths = set()
    for path in paths:
        cfg = Config.fromfile(str(path))
        seen.add(cfg.data.train.label_fraction)
        save_paths.add(cfg.save_path)
        assert cfg.pretrain_method == "scratch" and cfg.probe_mode == "ft"
        assert cfg.weight is None and cfg.resume is False
        assert cfg.epoch == 20 and cfg.data.train.loop == 3
        assert cfg.scheduler.total_steps == 20
        assert cfg.batch_size_train == 24
        assert cfg.gradient_accumulation_steps == 2
        assert cfg.writer is None
        assert cfg.data.train.type == "SparseLabelLasDataset"
        assert cfg.data.val.type == cfg.data.test.type == "LasDataset"
        assert cfg.save_path.startswith("exp/label_efficiency/dales/scratch_collapse/")
        assert cfg.data.train.label_manifest_dir == str(MANIFEST.parent)
        hooks = [hook.type for hook in cfg.hooks]
        assert hooks.index("SemSegEvaluator") < hooks.index("CheckpointSaver")
        assert hooks.index("CheckpointSaver") < hooks.index("EarlyStoppingHook")
        early = cfg.hooks[hooks.index("EarlyStoppingHook")]
        assert early.monitor == "mIoU" and early.mode == "max"
        assert early.min_delta_percent == 0.25 and early.patience == 3
        assert "PreciseEvaluator" in hooks
    assert seen == set(COLLAPSE_FRACTIONS)
    assert len(save_paths) == 5


def test_real_collapse_manifest_is_nested_and_covers_classes():
    if not MANIFEST.exists():
        pytest.skip("DALES manifest unavailable")
    summary = json.loads(MANIFEST.read_text(encoding="utf-8"))
    counts = [summary["global_counts"][str(f)] for f in COLLAPSE_FRACTIONS]
    assert all(all(n > 0 for n in row) for row in counts)
    assert all(sum(a) < sum(b) for a, b in zip(counts, counts[1:]))
    assert len(summary["files"]) == 2900


def test_zero_label_tile_can_be_collated_with_labeled_tile():
    if not MANIFEST.exists():
        pytest.skip("DALES manifest unavailable")
    cfg = Config.fromfile(str(CONFIG_DIR / "semseg-scratch-litept-v1m4-dales-ft-0p01pct.py"))
    dataset = build_dataset(cfg.data.train)
    zero_tiles = [index for index, path in enumerate(dataset.data_list)
                  if sum(dataset._label_summary["files"][Path(path).name]
                         ["counts"]["0.0001"]) == 0]
    assert len(zero_tiles) == 1
    assert dataset.sample_weights[zero_tiles[0]] == 0.0
    zero = dataset.get_data(zero_tiles[0])
    zero = dataset.post_transform(dataset.transform(zero))
    labeled = dataset[0]
    assert not (zero["segment"] >= 0).any()
    assert (labeled["segment"] >= 0).any()
    assert (dataset[zero_tiles[0]]["segment"] >= 0).any()
    batch = point_collate_fn([zero, labeled])
    logits = torch.randn(len(batch["segment"]), cfg.data.num_classes)
    assert torch.isfinite(CrossEntropyLoss(ignore_index=-1)(logits, batch["segment"]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_lowest_budget_scratch_backward():
    if not MANIFEST.exists():
        pytest.skip("DALES manifest unavailable")
    cfg = Config.fromfile(str(CONFIG_DIR / "semseg-scratch-litept-v1m4-dales-ft-0p01pct.py"))
    dataset = build_dataset(cfg.data.train)
    sample = dataset[0]
    assert (sample["segment"] >= 0).any()
    model = build_model(cfg.model).cuda().train()
    model.criteria.set_class_weight(dataset.class_weight)
    sample = {key: value.cuda() if torch.is_tensor(value) else value
              for key, value in sample.items()}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(sample)["loss"]
    assert torch.isfinite(loss)
    loss.backward()
