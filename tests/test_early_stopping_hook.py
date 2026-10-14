import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from pointspace.engines.hooks.default import HookBase
from pointspace.engines.hooks.early_stopping import EarlyStoppingHook
from pointspace.engines.train import TrainerBase
from pointspace.utils.config import Config
from pointspace.utils.events import EventStorage


class _Trainer:
    def __init__(self, root, resume=False, start_epoch=0):
        self.cfg = SimpleNamespace(save_path=str(root), resume=resume)
        self.start_epoch = start_epoch
        self.epoch = start_epoch
        self.comm_info = {}
        self.storage = EventStorage()
        self.logger = logging.getLogger(f"early-stop-test-{id(self)}")
        self.stop_training_requested = False
        self.stop_training_reason = None

    def request_stop(self, reason):
        self.stop_training_requested = True
        self.stop_training_reason = reason


def _put_epoch(trainer, hook, epoch, name, value):
    trainer.epoch = epoch
    trainer.storage.put_scalar(name, value)
    hook.after_epoch()
    trainer.storage.reset_histories()


def test_relative_percent_delta_and_patience(tmp_path):
    trainer = _Trainer(tmp_path)
    hook = EarlyStoppingHook(
        monitor="loss", mode="min", patience=3,
        min_delta_percent=0.1,
    )
    hook.trainer = trainer
    _put_epoch(trainer, hook, 0, "loss", 100.0)
    assert hook.best == 100.0
    _put_epoch(trainer, hook, 1, "loss", 99.95)  # only 0.05%
    assert hook.bad_epochs == 1
    _put_epoch(trainer, hook, 2, "loss", 99.8)   # 0.2%, significant
    assert hook.best == 99.8
    assert hook.bad_epochs == 0
    for epoch, value in ((3, 99.75), (4, 99.76), (5, 100.0)):
        _put_epoch(trainer, hook, epoch, "loss", value)
    assert trainer.stop_training_requested
    assert hook.bad_epochs == 3


def test_accuracy_and_declared_metric_modes(tmp_path):
    trainer = _Trainer(tmp_path)
    hook = EarlyStoppingHook(
        monitor="mIoU", mode="auto", patience=3,
        min_delta_percent=0.1, source="comm_info",
    )
    hook.trainer = trainer
    trainer.comm_info.update(
        current_metric_name="mIoU", current_metric_mode="max",
        current_metric_value=0.5,
    )
    hook.after_epoch()
    trainer.epoch = 1
    trainer.comm_info["current_metric_value"] = 0.5004
    hook.after_epoch()
    assert hook.bad_epochs == 1
    trainer.epoch = 2
    trainer.comm_info["current_metric_value"] = 0.501
    hook.after_epoch()
    assert hook.best == pytest.approx(0.501)
    assert hook.bad_epochs == 0

    trainer.comm_info.update(
        current_metric_name="neg_MAE", current_metric_value=-0.2,
        current_metric_mode="max",
    )
    negative = EarlyStoppingHook(monitor="neg_MAE", mode="auto")
    negative.trainer = trainer
    assert negative._resolved_mode("neg_MAE") == "max"


def test_warmup_and_resume_state(tmp_path):
    trainer = _Trainer(tmp_path)
    hook = EarlyStoppingHook(
        monitor="loss", mode="min", patience=1,
        min_delta_percent=1.0, warmup_epochs=2,
    )
    hook.trainer = trainer
    _put_epoch(trainer, hook, 0, "loss", 1.0)
    _put_epoch(trainer, hook, 1, "loss", 1.1)
    assert not trainer.stop_training_requested
    assert hook.bad_epochs == 0
    _put_epoch(trainer, hook, 2, "loss", 1.1)
    assert trainer.stop_training_requested

    state_path = Path(tmp_path) / "model" / "early_stopping.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["epoch"] == 3
    resumed_trainer = _Trainer(tmp_path, resume=True, start_epoch=3)
    resumed = EarlyStoppingHook(
        monitor="loss", mode="min", patience=1,
        min_delta_percent=1.0, warmup_epochs=2,
    )
    resumed.trainer = resumed_trainer
    resumed.before_train()
    assert resumed.best == hook.best
    assert resumed.bad_epochs == hook.bad_epochs


def test_missing_metric_is_strict_by_default(tmp_path):
    trainer = _Trainer(tmp_path)
    hook = EarlyStoppingHook(monitor="does_not_exist")
    hook.trainer = trainer
    with pytest.raises(KeyError, match="could not find metric"):
        hook.after_epoch()


def test_all_section45_ssl_configs_enable_relative_early_stopping():
    configs = (
        "configs/baselines/swiss/pretrain-point-m2ae-litept-v1m4-100.py",
        "configs/baselines/swiss/pretrain-msc-litept-v1m4-100.py",
        "configs/baselines/swiss/pretrain-d-ditr-litept-v1m4-100.py",
        "configs/baselines/swiss/pretrain-sonata-litept-v1m4-100.py",
        "configs/baselines/swiss/pretrain-concerto-litept-v1m4-100.py",
    )
    for path in configs:
        config = Config.fromfile(path)
        hooks = [dict(item) for item in config.hooks]
        selected = [item for item in hooks if item["type"] == "EarlyStoppingHook"]
        assert len(selected) == 1
        early = selected[0]
        assert early["monitor"] == "loss"
        assert early["mode"] == "min"
        assert early["patience"] == 3
        assert early["min_delta_percent"] == pytest.approx(0.1)
        assert early["warmup_epochs"] == 2
        assert next(i for i, h in enumerate(hooks) if h["type"] == "CheckpointSaver") \
            < next(i for i, h in enumerate(hooks) if h["type"] == "EarlyStoppingHook")


def test_trainer_loop_stops_cleanly_and_runs_after_train():
    events = []

    class _Writer:
        def close(self):
            events.append("writer_closed")

    class _MiniTrainer(TrainerBase):
        def __init__(self):
            super().__init__()
            self.max_epoch = 5
            self.writer = _Writer()
            self.loop_events = events

        def before_epoch(self):
            self.data_iterator = enumerate([None])
            super().before_epoch()

        def run_step(self):
            events.append("step")

    trainer = _MiniTrainer()
    trainer.register_hooks([dict(type="EarlyStoppingLoopTestHook")])
    trainer.train()
    assert events == ["step", "after_epoch", "after_train", "writer_closed"]
    assert trainer.epoch == 0


# Register after the class definition so this test-only hook is discoverable by
# the same registry path used in production.
from pointspace.engines.hooks.builder import HOOKS


@HOOKS.register_module("EarlyStoppingLoopTestHook")
class _EarlyStoppingLoopTestHook(HookBase):
    def after_epoch(self):
        # The event list is injected by the test through the trainer.
        self.trainer.loop_events.append("after_epoch")
        self.trainer.request_stop("unit test")

    def after_train(self):
        self.trainer.loop_events.append("after_train")
