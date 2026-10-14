from pathlib import Path

from pointspace.utils.config import Config
from utils.generate_downstream_ssl_configs import DATASETS, METHODS, PROBES


ROOT = Path(__file__).resolve().parents[1]


def _path(root, dataset, method, probe):
    return (
        ROOT
        / "configs"
        / root
        / dataset
        / f"semseg-{method}-litept-v1m4-{dataset}-{probe}.py"
    )


def _litept(cfg):
    backbone = cfg.model.backbone
    if backbone.type == "CVRL-v1m1-backbone":
        return backbone.return_backbone
    if backbone.type == "VPSD-v1m1-backbone":
        return backbone.backbone
    return backbone


def test_formal_complete_transfer_protocol():
    for dataset in DATASETS:
        for method in METHODS:
            for probe in PROBES:
                cfg = Config.fromfile(str(_path("downstream", dataset, method, probe)))
                assert cfg.epoch == (20 if dataset == "opengf" else 10)
                assert cfg.data.train.loop == (3 if dataset == "opengf" else 5)
                assert any(hook.type == "PreciseEvaluator" for hook in cfg.hooks)
                assert _litept(cfg).drop_path == 0.0
                assert not any(
                    hook.type in {
                        "GradualUnfreezeHook",
                        "FreezeBatchNormStatsHook",
                        "FreezeStochasticModulesHook",
                    }
                    for hook in cfg.hooks
                )
                precise = [
                    (index, hook)
                    for index, hook in enumerate(cfg.hooks)
                    if hook.type == "PreciseBatchNormHook"
                ]
                evaluator_index = next(
                    i for i, hook in enumerate(cfg.hooks)
                    if hook.type == "SemSegEvaluator"
                )
                saver_index = next(
                    i for i, hook in enumerate(cfg.hooks)
                    if hook.type == "CheckpointSaver"
                )
                assert evaluator_index < saver_index
                early = [
                    (index, hook)
                    for index, hook in enumerate(cfg.hooks)
                    if hook.type == "EarlyStoppingHook"
                ]
                if dataset == "dales":
                    assert early == []
                else:
                    assert len(early) == 1
                    early_index, early_hook = early[0]
                    assert saver_index < early_index
                    assert early_hook.monitor == "mIoU"
                    assert early_hook.mode == "max"
                    assert early_hook.source == "comm_info"
                    assert early_hook.patience == 3
                    assert early_hook.min_delta_percent == 0.25
                if probe == "lin":
                    assert precise == []
                else:
                    assert len(precise) == 1
                    index, hook = precise[0]
                    assert index < evaluator_index
                    assert hook.scope == ("decoder" if probe == "dec" else "all")
                    assert hook.num_batches == 64
                if method == "scratch" and probe == "ft":
                    assert cfg.weight is None
                else:
                    assert "_bncal_swiss.pth" in cfg.weight
                if probe == "ft":
                    assert cfg.optimizer.lr == 1e-3
                    assert list(cfg.param_dicts) == []
