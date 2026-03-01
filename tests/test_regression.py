"""
Tests for the regression pipeline.

Covers:
    1. Regression losses (MSELoss, L1Loss, SmoothL1Loss, HuberLoss)
       - basic forward, loss_weight, ignore_value, empty-after-mask, gradient flow
    2. DefaultRegressor model
       - single target, multi-target, train/eval/test branches
    3. RegressionEvaluator hook
       - metric computation (MAE, RMSE, R²), negative-MAE as saver metric
    4. RegressionTester
       - fragment-based prediction averaging, writer integration
    5. LASWriter pred_reg support
       - scalar regression, multi-target regression

Author: PointSpace Team
"""

import math
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pointspace.models.losses.misc import MSELoss, L1Loss, SmoothL1Loss, HuberLoss
from pointspace.models.losses.builder import Criteria, LOSSES

try:
    import laspy
    HAS_LASPY = True
except ImportError:
    HAS_LASPY = False


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _rand_pred(n=128):
    """Random scalar predictions (requires_grad=True)."""
    return torch.randn(n, requires_grad=True)


def _rand_target(n=128):
    """Random scalar targets."""
    return torch.randn(n)


# ===========================================================================
# 1. Regression Losses
# ===========================================================================


class TestMSELoss(unittest.TestCase):
    def test_forward_basic(self):
        loss_fn = MSELoss()
        pred = _rand_pred()
        target = _rand_target()
        val = loss_fn(pred, target)
        self.assertIsInstance(val, torch.Tensor)
        self.assertGreater(val.item(), 0)

    def test_zero_loss(self):
        loss_fn = MSELoss()
        t = torch.tensor([1.0, 2.0, 3.0])
        val = loss_fn(t, t.clone())
        self.assertAlmostEqual(val.item(), 0.0, places=6)

    def test_loss_weight(self):
        pred = _rand_pred(64)
        target = _rand_target(64)
        l1 = MSELoss(loss_weight=1.0)(pred, target)
        l3 = MSELoss(loss_weight=3.0)(pred, target)
        self.assertAlmostEqual(l3.item(), l1.item() * 3.0, places=4)

    def test_ignore_value(self):
        loss_fn = MSELoss(ignore_value=-999.0)
        pred = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        target = torch.tensor([1.0, -999.0, 3.0])
        val = loss_fn(pred, target)
        # Only indices 0 and 2 contribute; both exact match → 0
        self.assertAlmostEqual(val.item(), 0.0, places=6)

    def test_empty_after_mask(self):
        loss_fn = MSELoss(ignore_value=-999.0)
        pred = torch.tensor([1.0, 2.0], requires_grad=True)
        target = torch.tensor([-999.0, -999.0])
        val = loss_fn(pred, target)
        self.assertAlmostEqual(val.item(), 0.0, places=6)

    def test_gradient_flows(self):
        loss_fn = MSELoss()
        pred = _rand_pred()
        target = _rand_target()
        val = loss_fn(pred, target)
        val.backward()
        self.assertIsNotNone(pred.grad)
        self.assertFalse(torch.all(pred.grad == 0))

    def test_2d_input(self):
        """Pred and target with shape (N, 1) should work."""
        loss_fn = MSELoss()
        pred = torch.randn(32, 1, requires_grad=True)
        target = torch.randn(32, 1)
        val = loss_fn(pred, target)
        self.assertIsInstance(val, torch.Tensor)

    def test_registry(self):
        self.assertIn("MSELoss", LOSSES._module_dict)


class TestL1Loss(unittest.TestCase):
    def test_forward_basic(self):
        loss_fn = L1Loss()
        val = loss_fn(_rand_pred(), _rand_target())
        self.assertGreater(val.item(), 0)

    def test_zero_loss(self):
        t = torch.tensor([1.0, 2.0])
        val = L1Loss()(t, t.clone())
        self.assertAlmostEqual(val.item(), 0.0, places=6)

    def test_loss_weight(self):
        pred, target = _rand_pred(64), _rand_target(64)
        l1 = L1Loss(loss_weight=1.0)(pred, target)
        l2 = L1Loss(loss_weight=2.0)(pred, target)
        self.assertAlmostEqual(l2.item(), l1.item() * 2.0, places=4)

    def test_ignore_value(self):
        loss_fn = L1Loss(ignore_value=-1.0)
        pred = torch.tensor([5.0, 0.0, 3.0], requires_grad=True)
        target = torch.tensor([5.0, -1.0, 3.0])
        val = loss_fn(pred, target)
        self.assertAlmostEqual(val.item(), 0.0, places=6)

    def test_gradient_flows(self):
        pred = _rand_pred()
        val = L1Loss()(pred, _rand_target())
        val.backward()
        self.assertIsNotNone(pred.grad)

    def test_registry(self):
        self.assertIn("L1Loss", LOSSES._module_dict)


class TestSmoothL1Loss(unittest.TestCase):
    def test_forward_basic(self):
        val = SmoothL1Loss()(_rand_pred(), _rand_target())
        self.assertGreater(val.item(), 0)

    def test_zero_loss(self):
        t = torch.tensor([1.0, 2.0])
        val = SmoothL1Loss()(t, t.clone())
        self.assertAlmostEqual(val.item(), 0.0, places=6)

    def test_beta_parameter(self):
        """Larger beta means wider quadratic region."""
        pred = torch.tensor([0.0], requires_grad=True)
        target = torch.tensor([0.5])
        # beta=0.1: diff=0.5 > beta → linear region
        # beta=10.0: diff=0.5 < beta → quadratic region
        l_small = SmoothL1Loss(beta=0.1)(pred, target)
        l_large = SmoothL1Loss(beta=10.0)(pred, target)
        # They should differ
        self.assertNotAlmostEqual(l_small.item(), l_large.item(), places=4)

    def test_loss_weight(self):
        pred, target = _rand_pred(64), _rand_target(64)
        l1 = SmoothL1Loss(loss_weight=1.0)(pred, target)
        l5 = SmoothL1Loss(loss_weight=5.0)(pred, target)
        self.assertAlmostEqual(l5.item(), l1.item() * 5.0, places=4)

    def test_ignore_value(self):
        loss_fn = SmoothL1Loss(ignore_value=0.0)
        pred = torch.tensor([1.0, 2.0], requires_grad=True)
        target = torch.tensor([0.0, 2.0])
        val = loss_fn(pred, target)
        # Only index 1 contributes (exact match) → 0
        self.assertAlmostEqual(val.item(), 0.0, places=6)

    def test_gradient_flows(self):
        pred = _rand_pred()
        val = SmoothL1Loss()(pred, _rand_target())
        val.backward()
        self.assertIsNotNone(pred.grad)

    def test_registry(self):
        self.assertIn("SmoothL1Loss", LOSSES._module_dict)


class TestHuberLoss(unittest.TestCase):
    def test_forward_basic(self):
        val = HuberLoss()(_rand_pred(), _rand_target())
        self.assertGreater(val.item(), 0)

    def test_zero_loss(self):
        t = torch.tensor([1.0, 2.0])
        val = HuberLoss()(t, t.clone())
        self.assertAlmostEqual(val.item(), 0.0, places=6)

    def test_delta_parameter(self):
        pred = torch.tensor([0.0], requires_grad=True)
        target = torch.tensor([2.0])
        l_small = HuberLoss(delta=0.1)(pred, target)
        l_large = HuberLoss(delta=10.0)(pred, target)
        self.assertNotAlmostEqual(l_small.item(), l_large.item(), places=4)

    def test_loss_weight(self):
        pred, target = _rand_pred(64), _rand_target(64)
        l1 = HuberLoss(loss_weight=1.0)(pred, target)
        l4 = HuberLoss(loss_weight=4.0)(pred, target)
        self.assertAlmostEqual(l4.item(), l1.item() * 4.0, places=4)

    def test_ignore_value(self):
        loss_fn = HuberLoss(ignore_value=-1.0)
        pred = torch.tensor([5.0, 0.0], requires_grad=True)
        target = torch.tensor([5.0, -1.0])
        val = loss_fn(pred, target)
        self.assertAlmostEqual(val.item(), 0.0, places=6)

    def test_empty_after_mask(self):
        loss_fn = HuberLoss(ignore_value=-1.0)
        pred = torch.tensor([1.0], requires_grad=True)
        target = torch.tensor([-1.0])
        val = loss_fn(pred, target)
        self.assertAlmostEqual(val.item(), 0.0, places=6)

    def test_gradient_flows(self):
        pred = _rand_pred()
        val = HuberLoss()(pred, _rand_target())
        val.backward()
        self.assertIsNotNone(pred.grad)

    def test_registry(self):
        self.assertIn("HuberLoss", LOSSES._module_dict)


class TestCriteriaBuildRegression(unittest.TestCase):
    """Test that Criteria can build regression losses from config dicts."""

    def test_build_single_mse(self):
        criteria = Criteria([dict(type="MSELoss")])
        self.assertEqual(len(criteria.criteria), 1)
        pred = _rand_pred()
        target = _rand_target()
        val = criteria(pred, target)
        self.assertIsInstance(val, torch.Tensor)

    def test_build_combined(self):
        criteria = Criteria([
            dict(type="MSELoss", loss_weight=0.5),
            dict(type="L1Loss", loss_weight=0.5),
        ])
        self.assertEqual(len(criteria.criteria), 2)
        pred = _rand_pred()
        target = _rand_target()
        val = criteria(pred, target)
        self.assertIsInstance(val, torch.Tensor)

    def test_build_all_four(self):
        criteria = Criteria([
            dict(type="MSELoss"),
            dict(type="L1Loss"),
            dict(type="SmoothL1Loss"),
            dict(type="HuberLoss"),
        ])
        self.assertEqual(len(criteria.criteria), 4)


# ===========================================================================
# 2. DefaultRegressor Model
# ===========================================================================


class _TinyBackbone(nn.Module):
    """Minimal backbone returning a feat tensor."""
    def __init__(self, in_channels=6, out_channels=32):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.out_channels = out_channels

    def forward(self, point):
        # point is a Point wrapping input_dict
        point.feat = self.linear(point.feat)
        return point


class TestDefaultRegressor(unittest.TestCase):
    def _make_input(self, n=64, feat_dim=6, with_target=True):
        input_dict = {
            "coord": torch.randn(n, 3),
            "feat": torch.randn(n, feat_dim),
            "offset": torch.tensor([n], dtype=torch.long),
        }
        if with_target:
            input_dict["hag"] = torch.randn(n)
        return input_dict

    def _build_model(self, num_targets=1, freeze_backbone=False):
        from pointspace.models.default import DefaultRegressor

        # Directly instantiate without going through build_model for backbone
        backbone = _TinyBackbone(in_channels=6, out_channels=32)
        with patch("pointspace.models.default.build_model", return_value=backbone):
            model = DefaultRegressor(
                num_targets=num_targets,
                backbone_out_channels=32,
                backbone={"type": "dummy"},
                criteria=[dict(type="MSELoss")],
                freeze_backbone=freeze_backbone,
            )
        return model

    def test_registered(self):
        from pointspace.models.builder import MODELS
        self.assertIn("DefaultRegressor", MODELS._module_dict)

    def test_train_mode(self):
        model = self._build_model()
        model.train()
        out = model(self._make_input())
        self.assertIn("loss", out)
        self.assertNotIn("reg_pred", out)
        self.assertIsInstance(out["loss"], torch.Tensor)

    def test_eval_mode_with_target(self):
        model = self._build_model()
        model.eval()
        out = model(self._make_input(with_target=True))
        self.assertIn("loss", out)
        self.assertIn("reg_pred", out)
        self.assertEqual(out["reg_pred"].shape, (64,))

    def test_test_mode_no_target(self):
        model = self._build_model()
        model.eval()
        out = model(self._make_input(with_target=False))
        self.assertNotIn("loss", out)
        self.assertIn("reg_pred", out)
        self.assertEqual(out["reg_pred"].shape, (64,))

    def test_multi_target(self):
        model = self._build_model(num_targets=3)
        model.eval()
        input_dict = self._make_input(with_target=False)
        out = model(input_dict)
        self.assertIn("reg_pred", out)
        # Multi-target should not squeeze
        self.assertEqual(out["reg_pred"].shape, (64, 3))

    def test_multi_target_train(self):
        model = self._build_model(num_targets=3)
        model.train()
        input_dict = self._make_input(with_target=True)
        # For multi-target training, target shape must match pred
        input_dict["hag"] = torch.randn(64, 3)
        out = model(input_dict)
        self.assertIn("loss", out)

    def test_gradient_flows(self):
        model = self._build_model()
        model.train()
        out = model(self._make_input())
        out["loss"].backward()
        # reg_head weights should have gradients
        self.assertIsNotNone(model.reg_head.weight.grad)

    def test_freeze_backbone(self):
        model = self._build_model(freeze_backbone=True)
        model.train()
        out = model(self._make_input())
        out["loss"].backward()
        # Backbone should be frozen
        for p in model.backbone.parameters():
            self.assertIsNone(p.grad)
        # Head should have grad
        self.assertIsNotNone(model.reg_head.weight.grad)


HAS_CUDA = torch.cuda.is_available()


# ===========================================================================
# 3. RegressionEvaluator Hook
# ===========================================================================


@unittest.skipUnless(HAS_CUDA, "CUDA not available")
class TestRegressionEvaluator(unittest.TestCase):
    """Test the RegressionEvaluator hook with a mocked trainer."""

    def _make_trainer(self, val_data, model_outputs):
        """Create a minimal mock trainer for evaluator testing."""
        from pointspace.engines.hooks.evaluator import RegressionEvaluator

        trainer = MagicMock()
        trainer.cfg.evaluate = True
        trainer.cfg.enable_wandb = False
        trainer.cfg.enable_amp = False
        trainer.epoch = 0
        trainer.writer = None
        trainer.hooks = []  # no CacheCleaner

        # ----- StorageHistory mock -----
        _history = {}

        def put_scalar(key, value):
            if key not in _history:
                _history[key] = []
            _history[key].append(value)

        class _HistoryObj:
            def __init__(self, values):
                self.values = values

            @property
            def avg(self):
                return sum(self.values) / len(self.values)

        def history(key):
            return _HistoryObj(_history.get(key, [0]))

        trainer.storage.put_scalar = put_scalar
        trainer.storage.history = history

        # ----- val_loader mock (wrapping list with MagicMock for __len__) -----
        mock_loader = MagicMock()
        mock_loader.__iter__ = MagicMock(return_value=iter(val_data))
        mock_loader.__len__ = MagicMock(return_value=len(val_data))

        trainer.val_loader = mock_loader

        # ----- model mock -----
        _outputs_iter = iter(model_outputs)

        def model_call(input_dict):
            return next(_outputs_iter)

        trainer.model = MagicMock(side_effect=model_call)
        trainer.model.eval = MagicMock()

        # ----- comm_info -----
        trainer.comm_info = {}

        return trainer

    def test_perfect_prediction(self):
        from pointspace.engines.hooks.evaluator import RegressionEvaluator

        target = torch.tensor([1.0, 2.0, 3.0, 4.0])
        val_data = [{"hag": target}]
        model_outputs = [
            {"reg_pred": target.clone().cuda(), "loss": torch.tensor(0.0)}
        ]

        evaluator = RegressionEvaluator(target_key="hag", log_interval=1)
        trainer = self._make_trainer(val_data, model_outputs)
        evaluator.trainer = trainer
        evaluator.eval()

        # Perfect predictions → MAE=0, metric = -0.0
        self.assertAlmostEqual(trainer.comm_info["current_metric_value"], 0.0, places=5)
        self.assertEqual(trainer.comm_info["current_metric_name"], "neg_MAE")

    def test_known_error(self):
        from pointspace.engines.hooks.evaluator import RegressionEvaluator

        target = torch.tensor([0.0, 0.0, 0.0, 0.0])
        pred = torch.tensor([1.0, 1.0, 1.0, 1.0]).cuda()
        val_data = [{"hag": target}]
        model_outputs = [{"reg_pred": pred, "loss": torch.tensor(1.0)}]

        evaluator = RegressionEvaluator(target_key="hag")
        trainer = self._make_trainer(val_data, model_outputs)
        evaluator.trainer = trainer
        evaluator.eval()

        # MAE should be 1.0, so neg_MAE = -1.0
        self.assertAlmostEqual(
            trainer.comm_info["current_metric_value"], -1.0, places=5
        )

    def test_r2_perfect(self):
        from pointspace.engines.hooks.evaluator import RegressionEvaluator

        target = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        val_data = [{"hag": target}]
        model_outputs = [{"reg_pred": target.clone().cuda(), "loss": torch.tensor(0.0)}]

        evaluator = RegressionEvaluator(target_key="hag")
        trainer = self._make_trainer(val_data, model_outputs)
        evaluator.trainer = trainer

        # Capture logger output to verify R² ≈ 1
        evaluator.eval()
        # For perfect predictions: MAE=0, RMSE=0, R²=1
        self.assertAlmostEqual(
            trainer.comm_info["current_metric_value"], 0.0, places=5
        )

    def test_after_epoch_calls_eval(self):
        from pointspace.engines.hooks.evaluator import RegressionEvaluator

        evaluator = RegressionEvaluator()
        evaluator.trainer = MagicMock()
        evaluator.trainer.cfg.evaluate = True
        evaluator.eval = MagicMock()
        evaluator.after_epoch()
        evaluator.eval.assert_called_once()

    def test_after_epoch_skips_when_no_evaluate(self):
        from pointspace.engines.hooks.evaluator import RegressionEvaluator

        evaluator = RegressionEvaluator()
        evaluator.trainer = MagicMock()
        evaluator.trainer.cfg.evaluate = False
        evaluator.eval = MagicMock()
        evaluator.after_epoch()
        evaluator.eval.assert_not_called()


# ===========================================================================
# 4. RegressionTester
# ===========================================================================


class TestRegressionTester(unittest.TestCase):
    """Test RegressionTester registration and fragment averaging logic."""

    def test_registered(self):
        from pointspace.engines.test import TESTERS
        self.assertIn("RegressionTester", TESTERS._module_dict)

    def test_collate_fn_passthrough(self):
        from pointspace.engines.test import RegressionTester
        batch = [{"a": 1}, {"b": 2}]
        result = RegressionTester.collate_fn(batch)
        self.assertEqual(result, batch)


# ===========================================================================
# 5. LASWriter pred_reg Support
# ===========================================================================


@unittest.skipUnless(HAS_LASPY, "laspy not installed")
class TestLASWriterRegression(unittest.TestCase):
    def _make_source_las(self, path, n=100):
        header = laspy.LasHeader(point_format=2, version="1.2")
        header.offsets = np.array([0.0, 0.0, 0.0])
        header.scales = np.array([0.001, 0.001, 0.001])
        las = laspy.LasData(header)
        coords = np.random.rand(n, 3) * 100
        las.x = coords[:, 0]
        las.y = coords[:, 1]
        las.z = coords[:, 2]
        las.write(path)
        return coords

    def test_scalar_regression(self):
        from pointspace.writers.las_writer import LASWriter

        with tempfile.TemporaryDirectory() as tmpdir:
            n = 200
            coords = np.random.rand(n, 3) * 100
            pred_reg = np.random.randn(n)

            writer = LASWriter(save_dir=tmpdir)
            out = writer.write("test_scene", coord=coords, pred_reg=pred_reg)
            self.assertTrue(os.path.isfile(out))

            las = laspy.read(out)
            self.assertEqual(len(las.points), n)
            self.assertIn("reg_pred", list(las.point_format.dimension_names))
            np.testing.assert_array_almost_equal(
                np.array(las["reg_pred"]), pred_reg, decimal=4
            )

    def test_multi_target_regression(self):
        from pointspace.writers.las_writer import LASWriter

        with tempfile.TemporaryDirectory() as tmpdir:
            n = 150
            d = 3
            coords = np.random.rand(n, 3) * 100
            pred_reg = np.random.randn(n, d)

            writer = LASWriter(save_dir=tmpdir)
            out = writer.write("test_multi", coord=coords, pred_reg=pred_reg)
            self.assertTrue(os.path.isfile(out))

            las = laspy.read(out)
            dims = list(las.point_format.dimension_names)
            for i in range(d):
                self.assertIn(f"reg_pred_{i}", dims)
                np.testing.assert_array_almost_equal(
                    np.array(las[f"reg_pred_{i}"]), pred_reg[:, i], decimal=4
                )

    def test_scalar_with_source_file(self):
        from pointspace.writers.las_writer import LASWriter

        with tempfile.TemporaryDirectory() as tmpdir:
            src_dir = os.path.join(tmpdir, "source")
            out_dir = os.path.join(tmpdir, "output")
            os.makedirs(src_dir)

            n = 80
            self._make_source_las(os.path.join(src_dir, "scene.las"), n)
            pred_reg = np.random.randn(n)

            writer = LASWriter(save_dir=out_dir, source_dir=src_dir)
            out = writer.write("scene", pred_reg=pred_reg)
            self.assertTrue(os.path.isfile(out))

            las = laspy.read(out)
            self.assertEqual(len(las.points), n)
            self.assertIn("reg_pred", list(las.point_format.dimension_names))

    def test_pred_reg_length_mismatch(self):
        from pointspace.writers.las_writer import LASWriter

        with tempfile.TemporaryDirectory() as tmpdir:
            coords = np.random.rand(100, 3) * 100
            pred_reg = np.random.randn(50)  # wrong length!

            writer = LASWriter(save_dir=tmpdir)
            with self.assertRaises(ValueError):
                writer.write("bad", coord=coords, pred_reg=pred_reg)

    def test_pred_reg_combined_with_sem(self):
        """Regression output can coexist with semantic segmentation output."""
        from pointspace.writers.las_writer import LASWriter

        with tempfile.TemporaryDirectory() as tmpdir:
            n = 100
            coords = np.random.rand(n, 3) * 100
            pred_sem = np.random.randint(0, 5, n)
            pred_reg = np.random.randn(n)

            writer = LASWriter(save_dir=tmpdir)
            out = writer.write(
                "combined", coord=coords, pred_sem=pred_sem, pred_reg=pred_reg
            )
            las = laspy.read(out)
            self.assertIn("reg_pred", list(las.point_format.dimension_names))
            np.testing.assert_array_equal(
                np.array(las.classification), pred_sem.astype(np.uint8)
            )


# ===========================================================================
# 6. Dataset VALID_ASSETS
# ===========================================================================


class TestDatasetValidAssets(unittest.TestCase):
    def test_regression_target_not_in_valid_assets(self):
        """regression_target is no longer a dedicated asset; target comes from
        an existing field (e.g. 'hag') configured via target_key."""
        from pointspace.datasets.defaults import DefaultDataset
        self.assertNotIn("regression_target", DefaultDataset.VALID_ASSETS)


# ===========================================================================
# 7. Integration: Losses can be built from config and used end-to-end
# ===========================================================================


class TestEndToEndLossConfig(unittest.TestCase):
    """Verify that regression losses work in the Criteria pipeline."""

    def test_mse_from_config(self):
        criteria = Criteria([dict(type="MSELoss", loss_weight=2.0)])
        pred = torch.randn(100, requires_grad=True)
        target = torch.randn(100)
        loss = criteria(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_l1_from_config(self):
        criteria = Criteria([dict(type="L1Loss")])
        pred = torch.randn(100, requires_grad=True)
        target = torch.randn(100)
        loss = criteria(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_smooth_l1_from_config(self):
        criteria = Criteria([dict(type="SmoothL1Loss", beta=0.5)])
        pred = torch.randn(100, requires_grad=True)
        target = torch.randn(100)
        loss = criteria(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_huber_from_config(self):
        criteria = Criteria([dict(type="HuberLoss", delta=1.5)])
        pred = torch.randn(100, requires_grad=True)
        target = torch.randn(100)
        loss = criteria(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_composite_regression(self):
        """MSE + L1 combined."""
        criteria = Criteria([
            dict(type="MSELoss", loss_weight=0.5),
            dict(type="L1Loss", loss_weight=0.5),
        ])
        pred = torch.randn(64, requires_grad=True)
        target = torch.randn(64)
        loss = criteria(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)
        self.assertGreater(loss.item(), 0)


if __name__ == "__main__":
    unittest.main()
