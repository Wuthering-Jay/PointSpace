"""Metric-aware, resume-safe early stopping."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

from pointspace.utils import comm

from .builder import HOOKS
from .default import HookBase


@HOOKS.register_module()
class EarlyStoppingHook(HookBase):
    """Stop after an epoch when a scalar has not improved sufficiently.

    Improvement is defined only in relative percentage terms so the same hook
    works for losses, accuracies and arbitrary metrics with different scales.
    For example, ``min_delta_percent=0.1`` requires a value to beat the best
    value by more than 0.1% of ``abs(best)``.

    The hook must be placed after the evaluator that produces a validation
    metric. For a training metric such as ``loss``, place it after
    ``InformationWriter`` and preferably after ``CheckpointSaver`` so the
    stopping epoch is saved before the request is acted upon.
    """

    def __init__(
        self,
        monitor="loss",
        mode="auto",
        patience=3,
        min_delta_percent=0.1,
        warmup_epochs=0,
        source="auto",
        strict=True,
        state_filename="early_stopping.json",
    ):
        if mode not in {"auto", "min", "max"}:
            raise ValueError("EarlyStoppingHook mode must be auto, min or max")
        if source not in {"auto", "storage", "comm_info"}:
            raise ValueError(
                "EarlyStoppingHook source must be auto, storage or comm_info"
            )
        if int(patience) < 1:
            raise ValueError("EarlyStoppingHook patience must be at least 1")
        if float(min_delta_percent) < 0:
            raise ValueError("min_delta_percent must be non-negative")
        if int(warmup_epochs) < 0:
            raise ValueError("warmup_epochs must be non-negative")
        self.monitor = str(monitor)
        self.mode = mode
        self.patience = int(patience)
        self.min_delta_percent = float(min_delta_percent)
        self.warmup_epochs = int(warmup_epochs)
        self.source = source
        self.strict = bool(strict)
        self.state_filename = str(state_filename)
        self.best = None
        self.bad_epochs = 0

    def _resolved_mode(self, metric_name):
        if self.mode != "auto":
            return self.mode
        lower = metric_name.lower()
        declared_mode = self.trainer.comm_info.get("current_metric_mode")
        if (
            self.trainer.comm_info.get("current_metric_name") == metric_name
            and declared_mode in {"min", "max"}
        ):
            return declared_mode
        if lower.startswith(("neg_", "negative_")):
            return "max"
        minimize_tokens = ("loss", "error", "mae", "mse", "rmse", "distance")
        return "min" if any(token in lower for token in minimize_tokens) else "max"

    def _storage_candidates(self):
        candidates = [self.monitor]
        if self.monitor.startswith("train/"):
            candidates.append(self.monitor[len("train/"):])
        if self.monitor.startswith("val/"):
            candidates.append("val_" + self.monitor[len("val/"):])
        candidates.append(self.monitor.replace("/", "_"))
        return tuple(dict.fromkeys(candidates))

    def _read_metric(self):
        current_name = self.trainer.comm_info.get("current_metric_name")
        current_value = self.trainer.comm_info.get("current_metric_value")
        use_current = (
            self.monitor == "current_metric"
            or (current_name is not None and self.monitor == current_name)
        )
        if self.source in {"auto", "comm_info"} and use_current:
            if current_value is not None:
                return str(current_name or self.monitor), float(current_value)

        if self.source in {"auto", "storage"}:
            histories = self.trainer.storage.histories()
            for name in self._storage_candidates():
                if name in histories and histories[name].count:
                    return name, float(histories[name].avg)

        if self.source in {"auto", "comm_info"}:
            value = self.trainer.comm_info.get(self.monitor)
            if value is not None:
                return self.monitor, float(value)

        available = sorted(self.trainer.storage.histories().keys())
        message = (
            f"EarlyStoppingHook could not find metric '{self.monitor}'. "
            f"Available storage metrics: {available}; current metric: "
            f"{current_name!r}"
        )
        if self.strict:
            raise KeyError(message)
        self.trainer.logger.warning(message + "; skipping this epoch")
        return None, None

    def _state_path(self):
        return Path(self.trainer.cfg.save_path) / "model" / self.state_filename

    def before_train(self):
        if not getattr(self.trainer.cfg, "resume", False):
            return
        path = self._state_path()
        if not path.is_file():
            return
        state = json.loads(path.read_text(encoding="utf-8"))
        state_epoch = int(state.get("epoch", -1))
        if state_epoch != int(self.trainer.start_epoch):
            self.trainer.logger.warning(
                "Ignoring stale early-stopping state at epoch %d; checkpoint "
                "resumes from epoch %d",
                state_epoch,
                self.trainer.start_epoch,
            )
            return
        if state.get("monitor") != self.monitor:
            self.trainer.logger.warning(
                "Ignoring early-stopping state for monitor %r (configured %r)",
                state.get("monitor"),
                self.monitor,
            )
            return
        self.best = state.get("best")
        self.bad_epochs = int(state.get("bad_epochs", 0))
        self.trainer.logger.info(
            "Resumed early stopping: monitor=%s best=%s bad_epochs=%d/%d",
            self.monitor,
            self.best,
            self.bad_epochs,
            self.patience,
        )

    def _save_state(self, metric_name, mode):
        if not comm.is_main_process():
            return
        path = self._state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "epoch": self.trainer.epoch + 1,
                    "monitor": self.monitor,
                    "resolved_metric": metric_name,
                    "mode": mode,
                    "best": self.best,
                    "bad_epochs": self.bad_epochs,
                    "patience": self.patience,
                    "min_delta_percent": self.min_delta_percent,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        os.replace(temporary, path)

    def after_epoch(self):
        metric_name, local_value = self._read_metric()
        if local_value is None:
            return
        values = [float(value) for value in comm.all_gather(local_value)]
        value = sum(values) / len(values)
        if not math.isfinite(value):
            improved = False
        elif self.best is None:
            improved = True
        else:
            relative_delta = abs(float(self.best)) * self.min_delta_percent / 100.0
            mode = self._resolved_mode(metric_name)
            improved = (
                value < float(self.best) - relative_delta
                if mode == "min"
                else value > float(self.best) + relative_delta
            )

        mode = self._resolved_mode(metric_name)
        epoch = self.trainer.epoch + 1
        if improved:
            self.best = value
            self.bad_epochs = 0
            status = "improved"
        elif epoch <= self.warmup_epochs:
            self.bad_epochs = 0
            status = "warm-up"
        else:
            self.bad_epochs += 1
            status = "no significant improvement"

        self._save_state(metric_name, mode)
        self.trainer.logger.info(
            "Early stopping: %s=%.6g best=%.6g mode=%s delta=%.4g%% "
            "status=%s patience=%d/%d",
            metric_name,
            value,
            float("nan") if self.best is None else float(self.best),
            mode,
            self.min_delta_percent,
            status,
            self.bad_epochs,
            self.patience,
        )
        if epoch > self.warmup_epochs and self.bad_epochs >= self.patience:
            self.trainer.request_stop(
                f"early stopping on {metric_name}: no >"
                f"{self.min_delta_percent:g}% improvement for "
                f"{self.bad_epochs} epochs"
            )
