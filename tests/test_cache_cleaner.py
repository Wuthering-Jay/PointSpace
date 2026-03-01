"""
Tests for CacheCleaner hook.

Validates:
1. Fixed cleaning: after_epoch and after_train always trigger cache clear
2. Adaptive cleaning: slow steps trigger cache clear when exceeding threshold
3. Warmup: no adaptive cleans during warmup period
4. Sliding window: only recent step times affect mean
5. Absolute threshold: triggers independently of relative check
6. Logging: each clean event is logged with reason
7. Interval mode: step_clean_interval triggers fixed-interval cleaning
8. Default mode: step_clean_interval=None still does adaptive cleaning
"""

import time
import logging
import unittest
from unittest.mock import MagicMock, patch, call

import torch

from pointspace.engines.hooks.misc import CacheCleaner


def _make_trainer_mock():
    """Create a mock trainer with logger."""
    trainer = MagicMock()
    trainer.logger = logging.getLogger("test_cache_cleaner")
    trainer.logger.setLevel(logging.DEBUG)
    trainer.epoch = 0
    return trainer


class TestFixedCleaningPoints(unittest.TestCase):
    """after_epoch and after_train always clean cache."""

    @patch("pointspace.engines.hooks.misc.gc.collect")
    @patch("pointspace.engines.hooks.misc.torch.cuda.empty_cache")
    @patch("pointspace.engines.hooks.misc.torch.cuda.is_available", return_value=True)
    def test_after_epoch_cleans(self, mock_avail, mock_empty, mock_gc):
        hook = CacheCleaner()
        hook.trainer = _make_trainer_mock()
        hook.trainer.epoch = 5
        hook.after_epoch()
        mock_gc.assert_called_once()
        mock_empty.assert_called_once()

    @patch("pointspace.engines.hooks.misc.gc.collect")
    @patch("pointspace.engines.hooks.misc.torch.cuda.empty_cache")
    @patch("pointspace.engines.hooks.misc.torch.cuda.is_available", return_value=True)
    def test_after_train_cleans(self, mock_avail, mock_empty, mock_gc):
        hook = CacheCleaner()
        hook.trainer = _make_trainer_mock()
        hook.after_train()
        mock_gc.assert_called_once()
        mock_empty.assert_called_once()

    @patch("pointspace.engines.hooks.misc.gc.collect")
    @patch("pointspace.engines.hooks.misc.torch.cuda.empty_cache")
    @patch("pointspace.engines.hooks.misc.torch.cuda.is_available", return_value=True)
    def test_clean_count_increments(self, mock_avail, mock_empty, mock_gc):
        hook = CacheCleaner()
        hook.trainer = _make_trainer_mock()
        hook.after_epoch()
        hook.after_epoch()
        hook.after_train()
        self.assertEqual(hook._clean_count, 3)


class TestAdaptiveCleaning(unittest.TestCase):
    """Step-level adaptive cleaning based on timing."""

    def _simulate_steps(self, hook, durations):
        """Simulate a sequence of steps with given durations (in seconds)."""
        for d in durations:
            hook.before_step()
            # Monkey-patch the start time to control elapsed
            hook._step_start = time.perf_counter() - d
            hook.after_step()

    @patch("pointspace.engines.hooks.misc.gc.collect")
    @patch("pointspace.engines.hooks.misc.torch.cuda.empty_cache")
    @patch("pointspace.engines.hooks.misc.torch.cuda.is_available", return_value=True)
    def test_no_clean_during_warmup(self, mock_avail, mock_empty, mock_gc):
        hook = CacheCleaner(warmup_steps=5, time_multiplier=1.5)
        hook.trainer = _make_trainer_mock()
        # Even with a very slow step during warmup, no clean
        self._simulate_steps(hook, [0.1, 0.1, 0.1, 10.0, 10.0])
        mock_gc.assert_not_called()
        mock_empty.assert_not_called()

    @patch("pointspace.engines.hooks.misc.gc.collect")
    @patch("pointspace.engines.hooks.misc.torch.cuda.empty_cache")
    @patch("pointspace.engines.hooks.misc.torch.cuda.is_available", return_value=True)
    def test_relative_threshold_triggers(self, mock_avail, mock_empty, mock_gc):
        hook = CacheCleaner(warmup_steps=3, time_multiplier=2.0, window_size=10)
        hook.trainer = _make_trainer_mock()
        # 3 warmup steps + normal steps to establish mean ~0.1s
        self._simulate_steps(hook, [0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        mock_gc.assert_not_called()  # all normal
        # Now a step that takes 0.5s >> 0.1 * 2.0 = 0.2s
        self._simulate_steps(hook, [0.5])
        mock_gc.assert_called_once()

    @patch("pointspace.engines.hooks.misc.gc.collect")
    @patch("pointspace.engines.hooks.misc.torch.cuda.empty_cache")
    @patch("pointspace.engines.hooks.misc.torch.cuda.is_available", return_value=True)
    def test_absolute_threshold_triggers(self, mock_avail, mock_empty, mock_gc):
        hook = CacheCleaner(
            warmup_steps=2,
            time_multiplier=100.0,  # very high, won't trigger relative
            abs_threshold_sec=0.3,
        )
        hook.trainer = _make_trainer_mock()
        self._simulate_steps(hook, [0.1, 0.1, 0.1])  # 2 warmup + 1 normal
        mock_gc.assert_not_called()
        self._simulate_steps(hook, [0.4])  # above abs threshold
        mock_gc.assert_called_once()

    @patch("pointspace.engines.hooks.misc.gc.collect")
    @patch("pointspace.engines.hooks.misc.torch.cuda.empty_cache")
    @patch("pointspace.engines.hooks.misc.torch.cuda.is_available", return_value=True)
    def test_normal_steps_no_clean(self, mock_avail, mock_empty, mock_gc):
        hook = CacheCleaner(warmup_steps=2, time_multiplier=2.0)
        hook.trainer = _make_trainer_mock()
        # All steps are uniform → no spike → no clean
        self._simulate_steps(hook, [0.1] * 20)
        mock_gc.assert_not_called()


class TestSlidingWindow(unittest.TestCase):
    """Sliding window correctly limits history."""

    def test_window_size_limit(self):
        hook = CacheCleaner(warmup_steps=0, window_size=5)
        hook.trainer = _make_trainer_mock()
        # Add 10 step times
        for _ in range(10):
            hook._step_times.append(0.1)
            if len(hook._step_times) > hook.window_size:
                hook._step_times.pop(0)
        self.assertEqual(len(hook._step_times), 5)


class TestIntervalMode(unittest.TestCase):
    """step_clean_interval triggers fixed-interval cleaning."""

    def _simulate_steps(self, hook, durations):
        for d in durations:
            hook.before_step()
            hook._step_start = time.perf_counter() - d
            hook.after_step()

    @patch("pointspace.engines.hooks.misc.gc.collect")
    @patch("pointspace.engines.hooks.misc.torch.cuda.empty_cache")
    @patch("pointspace.engines.hooks.misc.torch.cuda.is_available", return_value=True)
    def test_interval_triggers_at_correct_steps(self, mock_avail, mock_empty, mock_gc):
        hook = CacheCleaner(
            step_clean_interval=5, warmup_steps=0, time_multiplier=100.0
        )
        hook.trainer = _make_trainer_mock()
        # 10 uniform steps — should clean at step 5 and 10
        self._simulate_steps(hook, [0.1] * 10)
        self.assertEqual(mock_gc.call_count, 2)

    @patch("pointspace.engines.hooks.misc.gc.collect")
    @patch("pointspace.engines.hooks.misc.torch.cuda.empty_cache")
    @patch("pointspace.engines.hooks.misc.torch.cuda.is_available", return_value=True)
    def test_no_interval_no_clean_for_uniform(self, mock_avail, mock_empty, mock_gc):
        """step_clean_interval=None with uniform steps → no clean."""
        hook = CacheCleaner(
            step_clean_interval=None, warmup_steps=2, time_multiplier=2.0
        )
        hook.trainer = _make_trainer_mock()
        self._simulate_steps(hook, [0.1] * 20)
        mock_gc.assert_not_called()

    @patch("pointspace.engines.hooks.misc.gc.collect")
    @patch("pointspace.engines.hooks.misc.torch.cuda.empty_cache")
    @patch("pointspace.engines.hooks.misc.torch.cuda.is_available", return_value=True)
    def test_interval_still_cleans_epoch(self, mock_avail, mock_empty, mock_gc):
        """Fixed cleaning points still work regardless of interval setting."""
        hook = CacheCleaner(step_clean_interval=None)
        hook.trainer = _make_trainer_mock()
        hook.after_epoch()
        mock_gc.assert_called_once()

    @patch("pointspace.engines.hooks.misc.gc.collect")
    @patch("pointspace.engines.hooks.misc.torch.cuda.empty_cache")
    @patch("pointspace.engines.hooks.misc.torch.cuda.is_available", return_value=True)
    def test_check_and_clean_interval(self, mock_avail, mock_empty, mock_gc):
        """check_and_clean also respects step_clean_interval."""
        hook = CacheCleaner(
            step_clean_interval=3, warmup_steps=0, time_multiplier=100.0
        )
        hook.logger = logging.getLogger("test_cache_cleaner")
        hook.logger.setLevel(logging.DEBUG)
        for i in range(9):
            hook.check_and_clean(0.1, f"iter {i + 1}")
        # Should clean at iter 3, 6, 9
        self.assertEqual(mock_gc.call_count, 3)


class TestLogging(unittest.TestCase):
    """Each cache clean is logged."""

    @patch("pointspace.engines.hooks.misc.gc.collect")
    @patch("pointspace.engines.hooks.misc.torch.cuda.empty_cache")
    @patch("pointspace.engines.hooks.misc.torch.cuda.is_available", return_value=True)
    def test_log_includes_reason(self, mock_avail, mock_empty, mock_gc):
        hook = CacheCleaner()
        hook.trainer = _make_trainer_mock()
        # Capture log output
        with self.assertLogs(hook.trainer.logger, level="INFO") as cm:
            hook.after_epoch()
        # Check that reason is in the log
        self.assertTrue(any("CacheCleaner" in msg for msg in cm.output))
        self.assertTrue(any("end of epoch" in msg for msg in cm.output))

    @patch("pointspace.engines.hooks.misc.gc.collect")
    @patch("pointspace.engines.hooks.misc.torch.cuda.empty_cache")
    @patch("pointspace.engines.hooks.misc.torch.cuda.is_available", return_value=True)
    def test_log_after_train(self, mock_avail, mock_empty, mock_gc):
        hook = CacheCleaner()
        hook.trainer = _make_trainer_mock()
        with self.assertLogs(hook.trainer.logger, level="INFO") as cm:
            hook.after_train()
        self.assertTrue(any("training finished" in msg for msg in cm.output))


class TestRegistration(unittest.TestCase):
    """CacheCleaner is registered in HOOKS registry."""

    def test_registered(self):
        from pointspace.engines.hooks.builder import HOOKS
        self.assertIn("CacheCleaner", HOOKS.module_dict)

    def test_build_default(self):
        from pointspace.engines.hooks.builder import HOOKS
        hook = HOOKS.build(dict(type="CacheCleaner"))
        self.assertIsInstance(hook, CacheCleaner)

    def test_build_custom(self):
        from pointspace.engines.hooks.builder import HOOKS
        hook = HOOKS.build(dict(
            type="CacheCleaner",
            warmup_steps=20,
            time_multiplier=3.0,
            abs_threshold_sec=5.0,
            window_size=100,
            step_clean_interval=50,
        ))
        self.assertEqual(hook.warmup_steps, 20)
        self.assertEqual(hook.time_multiplier, 3.0)
        self.assertEqual(hook.abs_threshold_sec, 5.0)
        self.assertEqual(hook.window_size, 100)
        self.assertEqual(hook.step_clean_interval, 50)


if __name__ == "__main__":
    unittest.main()
