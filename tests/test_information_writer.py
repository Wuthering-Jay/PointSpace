"""
Tests for InformationWriter hook — interval parameter.

Validates:
1. Default interval=1  → logs every step
2. interval=N         → logs only at steps that are multiples of N
3. Storage scalars    → accumulated every step regardless of interval
4. iter_info reset    → comm_info["iter_info"] cleared on skipped steps
5. TensorBoard writes → gated by interval
6. wandb writes       → gated by interval
7. after_epoch        → always runs (unaffected by interval)
8. before_step        → always builds iter_info prefix
9. Edge: interval=1   → identical behaviour to old code
10. Edge: step == interval exactly → logged at that step
"""

import logging
import unittest
from unittest.mock import MagicMock, patch, PropertyMock


from pointspace.engines.hooks.misc import InformationWriter


# ──────────────────────────────────────────────────────────────────────────────
# Helper factory
# ──────────────────────────────────────────────────────────────────────────────

def _make_trainer(loss_val=0.5, lr=0.001, num_classes=1):
    """Return a minimal mock trainer that satisfies InformationWriter needs."""
    trainer = MagicMock()
    trainer.logger = logging.getLogger("test_iw")
    trainer.logger.setLevel(logging.DEBUG)

    # epoch / iteration bookkeeping
    trainer.epoch = 0
    trainer.max_epoch = 10
    trainer.comm_info = {"iter": 0, "iter_info": ""}

    # optimizer
    trainer.optimizer.state_dict.return_value = {
        "param_groups": [{"lr": lr}]
    }

    # model output
    loss_tensor = MagicMock()
    loss_tensor.item.return_value = loss_val
    trainer.comm_info["model_output_dict"] = {"loss": loss_tensor}

    # storage: history().val returns the last value put
    hist = MagicMock()
    hist.val = loss_val
    trainer.storage.history.return_value = hist

    # train_loader length
    trainer.train_loader.__len__ = MagicMock(return_value=100)

    # writer / wandb off by default
    trainer.writer = None
    trainer.cfg.enable_wandb = False

    return trainer


def _attach(hook, trainer):
    """Bind hook to trainer and simulate before_train."""
    hook.trainer = trainer
    trainer.comm_info["iter_info"] = ""
    trainer.start_epoch = 0
    hook.before_train()


# ──────────────────────────────────────────────────────────────────────────────
# 1. Default interval=1 logs every step
# ──────────────────────────────────────────────────────────────────────────────

class TestIntervalDefault(unittest.TestCase):

    def setUp(self):
        self.hook = InformationWriter()           # interval=1
        self.trainer = _make_trainer()
        _attach(self.hook, self.trainer)

    def test_default_interval_is_one(self):
        self.assertEqual(self.hook.interval, 1)

    def test_logs_every_step(self):
        with patch.object(self.trainer.logger, "info") as mock_log:
            for step in range(1, 6):
                self.trainer.comm_info["iter"] = step - 1
                self.hook.before_step()
                self.hook.after_step()
            self.assertEqual(mock_log.call_count, 5)   # logged 5 / 5 steps


# ──────────────────────────────────────────────────────────────────────────────
# 2. interval=5 logs only at multiples of 5
# ──────────────────────────────────────────────────────────────────────────────

class TestIntervalFive(unittest.TestCase):

    def setUp(self):
        self.hook = InformationWriter(interval=5)
        self.trainer = _make_trainer()
        _attach(self.hook, self.trainer)

    def test_interval_stored(self):
        self.assertEqual(self.hook.interval, 5)

    def test_logs_only_at_multiples(self):
        with patch.object(self.trainer.logger, "info") as mock_log:
            for step in range(1, 11):            # steps 1..10
                self.trainer.comm_info["iter"] = step - 1
                self.hook.before_step()
                self.hook.after_step()
            # steps 5 and 10 → 2 calls
            self.assertEqual(mock_log.call_count, 2)

    def test_skipped_steps_do_not_log(self):
        with patch.object(self.trainer.logger, "info") as mock_log:
            # run only step 1 (not a multiple of 5)
            self.trainer.comm_info["iter"] = 0
            self.hook.before_step()
            self.hook.after_step()
            mock_log.assert_not_called()

    def test_log_at_step_five(self):
        with patch.object(self.trainer.logger, "info") as mock_log:
            for step in range(1, 6):
                self.trainer.comm_info["iter"] = step - 1
                self.hook.before_step()
                self.hook.after_step()
            mock_log.assert_called_once()


# ──────────────────────────────────────────────────────────────────────────────
# 3. Storage scalars accumulated every step
# ──────────────────────────────────────────────────────────────────────────────

class TestStorageAlwaysAccumulated(unittest.TestCase):

    def setUp(self):
        self.hook = InformationWriter(interval=10)
        self.trainer = _make_trainer()
        _attach(self.hook, self.trainer)

    def test_put_scalar_called_every_step(self):
        for step in range(1, 6):
            self.trainer.comm_info["iter"] = step - 1
            self.hook.before_step()
            self.hook.after_step()
        # storage.put_scalar should have been called 5 times (once per step)
        self.assertEqual(self.trainer.storage.put_scalar.call_count, 5)

    def test_put_scalar_value(self):
        self.trainer.comm_info["iter"] = 0
        self.hook.before_step()
        self.hook.after_step()
        self.trainer.storage.put_scalar.assert_called_once_with("loss", 0.5)


# ──────────────────────────────────────────────────────────────────────────────
# 4. iter_info reset on skipped steps
# ──────────────────────────────────────────────────────────────────────────────

class TestIterInfoReset(unittest.TestCase):

    def setUp(self):
        self.hook = InformationWriter(interval=5)
        self.trainer = _make_trainer()
        _attach(self.hook, self.trainer)

    def test_iter_info_cleared_on_skip(self):
        """iter_info must be reset to '' even on non-logged steps."""
        self.trainer.comm_info["iter"] = 0   # step 1, not a multiple of 5
        self.hook.before_step()
        # before_step sets a non-empty prefix
        self.assertNotEqual(self.trainer.comm_info["iter_info"], "")
        self.hook.after_step()
        # after_step must have reset it
        self.assertEqual(self.trainer.comm_info["iter_info"], "")

    def test_iter_info_cleared_after_log(self):
        """iter_info also reset when step IS logged."""
        for step in range(1, 6):
            self.trainer.comm_info["iter"] = step - 1
            self.hook.before_step()
            self.hook.after_step()
        self.assertEqual(self.trainer.comm_info["iter_info"], "")


# ──────────────────────────────────────────────────────────────────────────────
# 5. TensorBoard writes gated by interval
# ──────────────────────────────────────────────────────────────────────────────

class TestTensorBoardInterval(unittest.TestCase):

    def setUp(self):
        self.hook = InformationWriter(interval=5)
        self.trainer = _make_trainer()
        self.trainer.writer = MagicMock()        # enable TensorBoard
        self.trainer.cfg.enable_wandb = False
        _attach(self.hook, self.trainer)

    def test_no_tb_write_on_skipped_step(self):
        self.trainer.comm_info["iter"] = 0      # step 1, interval=5 → skip
        self.hook.before_step()
        self.hook.after_step()
        self.trainer.writer.add_scalar.assert_not_called()

    def test_tb_write_on_interval_step(self):
        for step in range(1, 6):
            self.trainer.comm_info["iter"] = step - 1
            self.hook.before_step()
            self.hook.after_step()
        # lr + loss = 2 add_scalar calls at step 5
        self.assertEqual(self.trainer.writer.add_scalar.call_count, 2)

    def test_tb_writes_10_steps_interval5(self):
        for step in range(1, 11):
            self.trainer.comm_info["iter"] = step - 1
            self.hook.before_step()
            self.hook.after_step()
        # steps 5, 10 → 2 × (lr + loss) = 4 add_scalar calls
        self.assertEqual(self.trainer.writer.add_scalar.call_count, 4)


# ──────────────────────────────────────────────────────────────────────────────
# 6. wandb writes gated by interval
# ──────────────────────────────────────────────────────────────────────────────

class TestWandbInterval(unittest.TestCase):

    def setUp(self):
        self.wandb_patcher = patch("pointspace.engines.hooks.misc.wandb")
        self.mock_wandb = self.wandb_patcher.start()
        self.mock_wandb.run.step = 0
        self.hook = InformationWriter(interval=5)
        self.trainer = _make_trainer()
        self.trainer.writer = MagicMock()
        self.trainer.cfg.enable_wandb = True
        _attach(self.hook, self.trainer)

    def tearDown(self):
        self.wandb_patcher.stop()

    def test_no_wandb_on_skipped_step(self):
        self.trainer.comm_info["iter"] = 0   # step 1 → skip
        self.hook.before_step()
        self.hook.after_step()
        self.mock_wandb.log.assert_not_called()

    def test_wandb_on_interval_step(self):
        for step in range(1, 6):
            self.trainer.comm_info["iter"] = step - 1
            self.hook.before_step()
            self.hook.after_step()
        # 2 wandb.log calls at step 5 (lr + loss)
        self.assertEqual(self.mock_wandb.log.call_count, 2)


# ──────────────────────────────────────────────────────────────────────────────
# 7. after_epoch always logs (unaffected by interval)
# ──────────────────────────────────────────────────────────────────────────────

class TestAfterEpochUnaffected(unittest.TestCase):

    def setUp(self):
        self.hook = InformationWriter(interval=100)  # very large interval
        self.trainer = _make_trainer()
        _attach(self.hook, self.trainer)

        # storage history for epoch-level keys
        hist = MagicMock()
        hist.avg = 0.42
        self.trainer.storage.history.return_value = hist
        self.trainer.storage._history = {"loss": hist}

        # val meter
        self.trainer.comm_info["metric_info"] = "mIoU: 0.80"

    def test_after_epoch_logs_regardless_of_interval(self):
        with patch.object(self.trainer.logger, "info") as mock_log:
            self.hook.after_epoch()
            mock_log.assert_called()

    def test_after_epoch_writes_tb(self):
        self.trainer.writer = MagicMock()
        self.trainer.cfg.enable_wandb = False
        # populate model_output_keys by running one step first
        self.hook.model_output_keys = ["loss"]
        self.hook.after_epoch()
        self.trainer.writer.add_scalar.assert_called()


# ──────────────────────────────────────────────────────────────────────────────
# 8. before_step always builds iter_info prefix
# ──────────────────────────────────────────────────────────────────────────────

class TestBeforeStepPrefix(unittest.TestCase):

    def test_prefix_built_even_on_skip(self):
        hook = InformationWriter(interval=10)
        trainer = _make_trainer()
        _attach(hook, trainer)

        trainer.comm_info["iter"] = 0   # step 1 — will be skipped
        hook.before_step()
        # iter_info must contain the "Train: [...]" prefix
        self.assertIn("Train:", trainer.comm_info["iter_info"])

    def test_curr_iter_increments_every_step(self):
        hook = InformationWriter(interval=10)
        trainer = _make_trainer()
        _attach(hook, trainer)

        for step in range(1, 4):
            trainer.comm_info["iter"] = step - 1
            hook.before_step()
            hook.after_step()
        self.assertEqual(hook.curr_iter, 3)


# ──────────────────────────────────────────────────────────────────────────────
# 9. interval=1 is identical to original every-step behaviour
# ──────────────────────────────────────────────────────────────────────────────

class TestIntervalOneEquality(unittest.TestCase):

    def test_all_steps_logged_with_interval_one(self):
        hook = InformationWriter(interval=1)
        trainer = _make_trainer()
        _attach(hook, trainer)
        with patch.object(trainer.logger, "info") as mock_log:
            for step in range(1, 11):
                trainer.comm_info["iter"] = step - 1
                hook.before_step()
                hook.after_step()
            self.assertEqual(mock_log.call_count, 10)


# ──────────────────────────────────────────────────────────────────────────────
# 10. Log message content includes expected keys
# ──────────────────────────────────────────────────────────────────────────────

class TestLogMessageContent(unittest.TestCase):

    def test_log_contains_loss_and_lr(self):
        hook = InformationWriter(interval=1)
        trainer = _make_trainer(loss_val=0.123, lr=0.0025)
        _attach(hook, trainer)
        with patch.object(trainer.logger, "info") as mock_log:
            trainer.comm_info["iter"] = 0
            hook.before_step()
            hook.after_step()
            logged_msg = mock_log.call_args[0][0]
        self.assertIn("loss", logged_msg)
        self.assertIn("Lr", logged_msg)

    def test_log_contains_epoch_info(self):
        hook = InformationWriter(interval=1)
        trainer = _make_trainer()
        _attach(hook, trainer)
        with patch.object(trainer.logger, "info") as mock_log:
            trainer.epoch = 2
            trainer.comm_info["iter"] = 0
            hook.before_step()
            hook.after_step()
            logged_msg = mock_log.call_args[0][0]
        self.assertIn("Train:", logged_msg)


if __name__ == "__main__":
    unittest.main()
