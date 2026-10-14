import json
from pathlib import Path

import laspy
import numpy as np
import pytest
import torch

from pointspace.datasets import build_dataset
from pointspace.datasets.sparse_las import SparseLabelLasDataset
from pointspace.datasets.transform import ClassFilter, SparseLabelMask
from pointspace.models.losses.misc import CrossEntropyLoss
from pointspace.models import build_model
from pointspace.utils.config import Config
from utils.create_sparse_label_manifests import create_manifests
from utils.generate_dales_label_efficiency_configs import MAIN_CURVE_FRACTIONS


@pytest.fixture
def sparse_fixture(tmp_path):
    source = tmp_path / "train"
    source.mkdir()
    for number in range(2):
        n = 100_000
        las = laspy.LasData(laspy.LasHeader(point_format=3, version="1.2"))
        las.x = np.arange(n) * 0.01
        las.y = np.full(n, number)
        las.z = np.zeros(n)
        las.classification = np.tile(np.arange(1, 9, dtype=np.uint8), n // 8)
        las.write(source / f"tile_{number}.las")
    output = tmp_path / "masks"
    summary = create_manifests(source, output, seed=42)
    return source, output, summary


def test_manifest_is_nested_deterministic_and_compact(sparse_fixture):
    source, output, summary = sparse_fixture
    assert summary["total_points"] == 200_000
    assert sum(summary["global_counts"]["0.001"]) < sum(summary["global_counts"]["0.01"])
    for las_path in source.glob("*.las"):
        record = summary["files"][las_path.name]
        with np.load(output / f"{las_path.stem}.npz") as entry:
            index, rank = entry["index"], entry["rank"]
            assert len(index) < record["point_count"] * 0.11
            assert len(np.unique(index)) == len(index)
            assert np.all(rank < 0.1)
            assert set(index[rank < 0.001]).issubset(set(index[rank < 0.01]))
    first = json.loads((output / "summary.json").read_text())
    create_manifests(source, output, seed=42)
    assert first == json.loads((output / "summary.json").read_text())


def test_sparse_dataset_keeps_geometry_and_hides_labels(sparse_fixture):
    source, output, summary = sparse_fixture
    dataset = SparseLabelLasDataset(
        data_path=str(source), label_manifest_dir=str(output), label_fraction=0.001,
        required_class=list(range(1, 9)), remap_class=True,
        class_weight="sqrt", weighted_sampler=True,
        transform=[dict(type="ClassFilter", keep_classes=list(range(8))),
                   dict(type="SparseLabelMask")],
    )
    assert len(dataset.sample_weights) == 2
    assert np.isfinite(dataset.class_weight).all()
    raw = dataset.get_data(0)
    assert len(raw["segment"]) == summary["files"]["tile_0.las"]["point_count"]
    sparse = dataset.prepare_train_data(0)
    assert len(sparse["coord"]) == len(raw["coord"])
    assert int((sparse["segment"] >= 0).sum()) == sum(
        summary["files"]["tile_0.las"]["counts"]["0.001"]
    )
    assert "sparse_keep" not in sparse


def test_sparse_dataset_windows_worker_loading(sparse_fixture):
    source, output, _ = sparse_fixture
    dataset = SparseLabelLasDataset(
        data_path=str(source), label_manifest_dir=str(output), label_fraction=0.001,
        required_class=list(range(1, 9)), remap_class=True,
        transform=[dict(type="ClassFilter", keep_classes=list(range(8))),
                   dict(type="SparseLabelMask")],
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=2)
    sample = next(iter(loader))
    assert int((sample["segment"] >= 0).sum()) > 0


def test_manifest_rejects_modified_source(sparse_fixture):
    source, output, _ = sparse_fixture
    dataset = SparseLabelLasDataset(
        data_path=str(source), label_manifest_dir=str(output), label_fraction=0.01,
        required_class=list(range(1, 9)), remap_class=True,
        transform=[dict(type="ClassFilter", keep_classes=list(range(8))),
                   dict(type="SparseLabelMask")],
    )
    dataset._label_summary["files"]["tile_0.las"]["source_bytes"] += 1
    with pytest.raises(ValueError, match="LAS source changed"):
        dataset.get_data(0)


def test_real_dales_all_sparse_budgets_train_sample():
    manifest = Path(r"E:\data\DALES\dales_las\tile\sparse_labels_seed42\summary.json")
    if not manifest.exists():
        pytest.skip("DALES manifest not available on this machine")
    tags = {f"{fraction * 100:g}pct".replace(".", "p")
            for fraction in MAIN_CURVE_FRACTIONS}
    configs = sorted(path for path in Path("configs/label_efficiency/dales").glob("semseg-cvrl-*.py")
                     if path.stem.rsplit("-ft-", 1)[-1] in tags)
    assert len(configs) == 4
    for config_path in configs:
        cfg = Config.fromfile(str(config_path))
        dataset = build_dataset(cfg.data.train)
        sample = dataset[0]
        labels = sample["segment"].long()
        valid = labels >= 0
        assert valid.any()
        assert valid.sum() < len(labels) * 0.12
        assert len(sample["coord"]) == len(labels)
        assert "sparse_keep" not in sample
        assert torch.isfinite(
            CrossEntropyLoss(ignore_index=-1)(
                torch.randn(len(labels), cfg.data.num_classes), labels
            )
        )


def test_all_sparse_ft_configs_match_full_protocol():
    config_dir = Path("configs/label_efficiency/dales")
    tags = {f"{fraction * 100:g}pct".replace(".", "p")
            for fraction in MAIN_CURVE_FRACTIONS}
    paths = sorted(path for path in config_dir.glob("*.py")
                   if path.stem.rsplit("-ft-", 1)[-1] in tags)
    assert len(paths) == 36
    fractions = set()
    for path in paths:
        cfg = Config.fromfile(str(path))
        fractions.add(cfg.data.train.label_fraction)
        assert cfg.probe_mode == "ft"
        assert cfg.epoch == 20 and cfg.data.train.loop == 3
        assert cfg.scheduler.total_steps == 20
        assert cfg.data.val.type == cfg.data.test.type == "LasDataset"
        assert cfg.data.train.type == "SparseLabelLasDataset"
        assert cfg.data.train.weighted_sampler is True
        early = [(i, hook) for i, hook in enumerate(cfg.hooks)
                 if hook.type == "EarlyStoppingHook"]
        assert len(early) == 1
        early_index, early_hook = early[0]
        assert early_hook.monitor == "mIoU"
        assert early_hook.mode == "max"
        assert early_hook.source == "comm_info"
        assert early_hook.patience == 3
        assert early_hook.min_delta_percent == 0.25
        assert early_hook.warmup_epochs == 2
        assert next(i for i, hook in enumerate(cfg.hooks)
                    if hook.type == "SemSegEvaluator") < early_index
        assert next(i for i, hook in enumerate(cfg.hooks)
                    if hook.type == "CheckpointSaver") < early_index
        assert [t.type for t in cfg.data.train.transform[:2]] == [
            "ClassFilter", "SparseLabelMask"
        ]
        assert cfg.data.train.data_path == r"E:\data\DALES\dales_las\tile\train"
        assert cfg.data.val.data_path == r"E:\data\DALES\dales_las\tile\test"
    assert fractions == set(MAIN_CURVE_FRACTIONS)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_real_sparse_ft_single_backward():
    manifest = Path(r"E:\data\DALES\dales_las\tile\sparse_labels_seed42\summary.json")
    if not manifest.exists():
        pytest.skip("DALES manifest not available on this machine")
    cfg = Config.fromfile(
        "configs/label_efficiency/dales/semseg-scratch-litept-v1m4-dales-ft-0p1pct.py"
    )
    sample = build_dataset(cfg.data.train)[0]
    model = build_model(cfg.model).cuda().train()
    model.criteria.set_class_weight(
        build_dataset(cfg.data.train).class_weight
    )
    sample = {key: value.cuda() if torch.is_tensor(value) else value
              for key, value in sample.items()}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(sample)["loss"]
    assert torch.isfinite(loss)
    loss.backward()
