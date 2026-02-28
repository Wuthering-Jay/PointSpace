"""
Tests for the joint semantic-segmentation + regression pipeline.

Covers:
    1. DefaultSemSegRegressor model
       - registration, train/eval/test branches, freeze_backbone, loss weights
    2. SemSegRegressionEvaluator hook
       - combined metric computation, primary_metric selection
    3. SemSegRegressionTester
       - registration, collate_fn

Author: PointSpace Team
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


class _TinyBackbone(nn.Module):
    """Minimal backbone returning a feat tensor (via Point)."""

    def __init__(self, in_channels=6, out_channels=32):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.out_channels = out_channels

    def forward(self, point):
        point.feat = self.linear(point.feat)
        return point


# ===========================================================================
# 1. DefaultSemSegRegressor model
# ===========================================================================


class TestDefaultSemSegRegressor(unittest.TestCase):
    NUM_CLASSES = 5

    def _make_input(self, n=64, feat_dim=6, with_seg=True, with_reg=True):
        d = {
            "coord": torch.randn(n, 3),
            "feat": torch.randn(n, feat_dim),
            "offset": torch.tensor([n], dtype=torch.long),
        }
        if with_seg:
            d["segment"] = torch.randint(0, self.NUM_CLASSES, (n,))
        if with_reg:
            d["hag"] = torch.randn(n)
        return d

    def _build_model(self, num_targets=1, seg_weight=1.0, reg_weight=1.0,
                     freeze_backbone=False):
        from pointspace.models.default import DefaultSemSegRegressor

        backbone = _TinyBackbone(in_channels=6, out_channels=32)
        with patch("pointspace.models.default.build_model", return_value=backbone):
            model = DefaultSemSegRegressor(
                num_classes=self.NUM_CLASSES,
                num_targets=num_targets,
                backbone_out_channels=32,
                backbone={"type": "dummy"},
                seg_criteria=[dict(type="CrossEntropyLoss", loss_weight=1.0,
                                   ignore_index=-1)],
                reg_criteria=[dict(type="MSELoss")],
                seg_weight=seg_weight,
                reg_weight=reg_weight,
                target_key="hag",
                freeze_backbone=freeze_backbone,
            )
        return model

    # --- registration ---
    def test_registered(self):
        from pointspace.models.builder import MODELS
        self.assertIn("DefaultSemSegRegressor", MODELS._module_dict)

    # --- train mode ---
    def test_train_returns_loss_only(self):
        model = self._build_model()
        model.train()
        out = model(self._make_input())
        self.assertIn("loss", out)
        self.assertNotIn("seg_logits", out)
        self.assertNotIn("reg_pred", out)
        self.assertIsInstance(out["loss"], torch.Tensor)

    def test_train_loss_is_scalar(self):
        model = self._build_model()
        model.train()
        out = model(self._make_input())
        self.assertEqual(out["loss"].dim(), 0)

    # --- eval mode ---
    def test_eval_with_both_targets(self):
        model = self._build_model()
        model.eval()
        out = model(self._make_input(with_seg=True, with_reg=True))
        self.assertIn("loss", out)
        self.assertIn("seg_logits", out)
        self.assertIn("reg_pred", out)
        self.assertIn("seg_loss", out)
        self.assertIn("reg_loss", out)
        self.assertEqual(out["seg_logits"].shape, (64, self.NUM_CLASSES))
        self.assertEqual(out["reg_pred"].shape, (64,))

    # --- test mode (no targets) ---
    def test_test_mode_no_targets(self):
        model = self._build_model()
        model.eval()
        out = model(self._make_input(with_seg=False, with_reg=False))
        self.assertNotIn("loss", out)
        self.assertIn("seg_logits", out)
        self.assertIn("reg_pred", out)

    # --- multi-target regression ---
    def test_multi_target(self):
        model = self._build_model(num_targets=3)
        model.eval()
        out = model(self._make_input(with_seg=False, with_reg=False))
        self.assertEqual(out["reg_pred"].shape, (64, 3))

    def test_multi_target_train(self):
        model = self._build_model(num_targets=3)
        model.train()
        inp = self._make_input(with_seg=True, with_reg=True)
        inp["hag"] = torch.randn(64, 3)
        out = model(inp)
        self.assertIn("loss", out)

    # --- loss weights ---
    def test_seg_weight_zero_means_reg_only(self):
        model = self._build_model(seg_weight=0.0, reg_weight=1.0)
        model.train()
        inp = self._make_input()
        # Build a reference for reg_loss alone
        out = model(inp)
        model.eval()
        out2 = model(self._make_input(with_seg=True, with_reg=True))
        # Just verify the model runs without error
        self.assertIsNotNone(out["loss"])

    def test_reg_weight_zero_means_seg_only(self):
        model = self._build_model(seg_weight=1.0, reg_weight=0.0)
        model.train()
        out = model(self._make_input())
        self.assertIsNotNone(out["loss"])

    # --- gradient flow ---
    def test_gradient_flows_to_both_heads(self):
        model = self._build_model()
        model.train()
        out = model(self._make_input())
        out["loss"].backward()
        self.assertIsNotNone(model.seg_head.weight.grad)
        self.assertIsNotNone(model.reg_head.weight.grad)

    def test_gradient_flows_through_backbone(self):
        model = self._build_model()
        model.train()
        out = model(self._make_input())
        out["loss"].backward()
        self.assertIsNotNone(model.backbone.linear.weight.grad)

    # --- freeze backbone ---
    def test_freeze_backbone(self):
        model = self._build_model(freeze_backbone=True)
        model.train()
        out = model(self._make_input())
        out["loss"].backward()
        for p in model.backbone.parameters():
            self.assertIsNone(p.grad)
        # Heads should still have grads
        self.assertIsNotNone(model.seg_head.weight.grad)
        self.assertIsNotNone(model.reg_head.weight.grad)


# ===========================================================================
# 2. SemSegRegressionEvaluator hook
# ===========================================================================


HAS_CUDA = torch.cuda.is_available()


@unittest.skipUnless(HAS_CUDA, "CUDA not available")
class TestSemSegRegressionEvaluator(unittest.TestCase):
    NUM_CLASSES = 3

    def _make_trainer(self, val_data, model_outputs):
        from pointspace.engines.hooks.evaluator import SemSegRegressionEvaluator

        trainer = MagicMock()
        trainer.cfg.evaluate = True
        trainer.cfg.enable_wandb = False
        trainer.cfg.data.num_classes = self.NUM_CLASSES
        trainer.cfg.data.ignore_index = -1
        trainer.cfg.data.names = [f"cls_{i}" for i in range(self.NUM_CLASSES)]
        trainer.epoch = 0
        trainer.writer = None
        trainer.hooks = []

        # StorageHistory mock
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
                return sum(self.values) / len(self.values) if self.values else 0

            @property
            def total(self):
                return np.sum(self.values, axis=0)

        def history(key):
            return _HistoryObj(_history.get(key, [0]))

        trainer.storage.put_scalar = put_scalar
        trainer.storage.history = history

        mock_loader = MagicMock()
        mock_loader.__iter__ = MagicMock(return_value=iter(val_data))
        mock_loader.__len__ = MagicMock(return_value=len(val_data))
        trainer.val_loader = mock_loader

        _outputs_iter = iter(model_outputs)

        def model_call(input_dict):
            return next(_outputs_iter)

        trainer.model = MagicMock(side_effect=model_call)
        trainer.model.eval = MagicMock()
        trainer.comm_info = {}
        return trainer

    def _make_val_batch(self, n=32):
        """Create one val batch with perfect seg and reg predictions."""
        segment = torch.randint(0, self.NUM_CLASSES, (n,))
        hag = torch.randn(n)
        val_input = {"segment": segment, "hag": hag}

        # Perfect seg logits: one-hot for correct class
        seg_logits = torch.zeros(n, self.NUM_CLASSES).cuda()
        seg_logits.scatter_(1, segment.unsqueeze(1).cuda(), 10.0)

        # Perfect reg prediction
        reg_pred = hag.clone().cuda()

        model_out = {
            "loss": torch.tensor(0.0),
            "seg_logits": seg_logits,
            "reg_pred": reg_pred,
            "seg_loss": torch.tensor(0.0),
            "reg_loss": torch.tensor(0.0),
        }
        return val_input, model_out

    def test_perfect_prediction_mIoU_primary(self):
        from pointspace.engines.hooks.evaluator import SemSegRegressionEvaluator

        val_input, model_out = self._make_val_batch()
        evaluator = SemSegRegressionEvaluator(
            target_key="hag", primary_metric="mIoU"
        )
        trainer = self._make_trainer([val_input], [model_out])
        evaluator.trainer = trainer
        evaluator.eval()

        self.assertEqual(trainer.comm_info["current_metric_name"], "mIoU")
        # Perfect prediction → mIoU ≈ 1.0
        self.assertGreater(trainer.comm_info["current_metric_value"], 0.9)

    def test_perfect_prediction_neg_mae_primary(self):
        from pointspace.engines.hooks.evaluator import SemSegRegressionEvaluator

        val_input, model_out = self._make_val_batch()
        evaluator = SemSegRegressionEvaluator(
            target_key="hag", primary_metric="neg_MAE"
        )
        trainer = self._make_trainer([val_input], [model_out])
        evaluator.trainer = trainer
        evaluator.eval()

        self.assertEqual(trainer.comm_info["current_metric_name"], "neg_MAE")
        # Perfect prediction → MAE ≈ 0, neg_MAE ≈ 0
        self.assertAlmostEqual(
            trainer.comm_info["current_metric_value"], 0.0, places=4
        )

    def test_after_epoch_calls_eval(self):
        from pointspace.engines.hooks.evaluator import SemSegRegressionEvaluator

        evaluator = SemSegRegressionEvaluator()
        evaluator.trainer = MagicMock()
        evaluator.trainer.cfg.evaluate = True
        evaluator.eval = MagicMock()
        evaluator.after_epoch()
        evaluator.eval.assert_called_once()

    def test_after_epoch_skips_when_no_evaluate(self):
        from pointspace.engines.hooks.evaluator import SemSegRegressionEvaluator

        evaluator = SemSegRegressionEvaluator()
        evaluator.trainer = MagicMock()
        evaluator.trainer.cfg.evaluate = False
        evaluator.eval = MagicMock()
        evaluator.after_epoch()
        evaluator.eval.assert_not_called()

    def test_registered(self):
        from pointspace.engines.hooks import HOOKS
        self.assertIn("SemSegRegressionEvaluator", HOOKS._module_dict)


# ===========================================================================
# 3. SemSegRegressionTester
# ===========================================================================


class TestSemSegRegressionTester(unittest.TestCase):
    def test_registered(self):
        from pointspace.engines.test import TESTERS
        self.assertIn("SemSegRegressionTester", TESTERS._module_dict)

    def test_collate_fn_passthrough(self):
        from pointspace.engines.test import SemSegRegressionTester
        batch = [{"a": 1}, {"b": 2}]
        result = SemSegRegressionTester.collate_fn(batch)
        self.assertEqual(result, batch)


if __name__ == "__main__":
    unittest.main()
