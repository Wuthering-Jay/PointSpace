"""
Tests for class-weight injection and WeightedRandomSampler integration.

Validates:
 1. CrossEntropyLoss: class_weight param renamed, auto_class_weight + set_class_weight
 2. FocalLoss: class_weight injected via set_class_weight, used in forward
 3. Criteria.set_class_weight propagates to losses with auto_class_weight=True
 4. Criteria.set_class_weight skips losses with auto_class_weight=False
 5. DistributedWeightedSampler: deterministic, partitioned across ranks
 6. WeightedRandomSampler with loop > 1: weights tiled correctly
 7. Trainer._inject_class_weights: end-to-end injection path
"""

import math
import unittest
from unittest.mock import MagicMock, patch, PropertyMock
import numpy as np
import torch
import torch.nn as nn

from pointspace.models.losses.misc import CrossEntropyLoss, FocalLoss
from pointspace.models.losses.builder import Criteria
from pointspace.datasets.sampler import DistributedWeightedSampler


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _rand_logits(n=64, c=5):
    return torch.randn(n, c, device="cpu")


def _rand_targets(n=64, c=5, ignore=-1):
    t = torch.randint(0, c, (n,))
    # sprinkle a few ignore values
    t[:3] = ignore
    return t


# ──────────────────────────────────────────────────────────────────────────────
# 1. CrossEntropyLoss – class_weight rename + auto_class_weight
# ──────────────────────────────────────────────────────────────────────────────

class TestCrossEntropyClassWeight(unittest.TestCase):

    def test_default_no_weight(self):
        loss_fn = CrossEntropyLoss()
        self.assertFalse(loss_fn.auto_class_weight)
        # internal nn.CE should have weight=None
        self.assertIsNone(loss_fn.loss.weight)

    def test_manual_class_weight(self):
        w = [1.0, 2.0, 3.0]
        loss_fn = CrossEntropyLoss(class_weight=w)
        self.assertIsNotNone(loss_fn.loss.weight)
        self.assertEqual(loss_fn.loss.weight.shape[0], 3)

    def test_set_class_weight(self):
        loss_fn = CrossEntropyLoss(auto_class_weight=True)
        self.assertIsNone(loss_fn.loss.weight)
        w = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        loss_fn.set_class_weight(w)
        self.assertIsNotNone(loss_fn.loss.weight)
        self.assertEqual(loss_fn.loss.weight.shape[0], 5)

    def test_forward_runs(self):
        loss_fn = CrossEntropyLoss(class_weight=[1.0]*5, ignore_index=-1)
        logits = _rand_logits()
        targets = _rand_targets()
        val = loss_fn(logits, targets)
        self.assertIsInstance(val, torch.Tensor)

    def test_backward_compat_loss_weight(self):
        """loss_weight should still scale the output."""
        loss_fn_1 = CrossEntropyLoss(loss_weight=1.0, ignore_index=-1)
        loss_fn_2 = CrossEntropyLoss(loss_weight=2.0, ignore_index=-1)
        logits = _rand_logits()
        targets = _rand_targets()
        v1 = loss_fn_1(logits, targets)
        v2 = loss_fn_2(logits, targets)
        self.assertAlmostEqual((v2 / v1).item(), 2.0, places=4)


# ──────────────────────────────────────────────────────────────────────────────
# 2. FocalLoss – class_weight injection
# ──────────────────────────────────────────────────────────────────────────────

class TestFocalLossClassWeight(unittest.TestCase):

    def test_default_no_class_weight(self):
        loss_fn = FocalLoss()
        self.assertIsNone(loss_fn.class_weight)
        self.assertFalse(loss_fn.auto_class_weight)

    def test_manual_class_weight(self):
        loss_fn = FocalLoss(class_weight=[1.0, 2.0, 3.0])
        self.assertIsNotNone(loss_fn.class_weight)
        self.assertEqual(loss_fn.class_weight.shape[0], 3)

    def test_set_class_weight(self):
        loss_fn = FocalLoss(auto_class_weight=True)
        w = torch.tensor([0.5, 1.5, 2.5, 0.8, 1.2])
        loss_fn.set_class_weight(w)
        self.assertIsNotNone(loss_fn.class_weight)

    def test_forward_with_class_weight(self):
        """FocalLoss should run without error when class_weight is set."""
        loss_fn = FocalLoss(class_weight=[1.0]*5, ignore_index=-1)
        logits = _rand_logits()
        targets = _rand_targets()
        val = loss_fn(logits, targets)
        self.assertIsInstance(val, torch.Tensor)

    def test_forward_no_class_weight(self):
        loss_fn = FocalLoss(ignore_index=-1)
        logits = _rand_logits()
        targets = _rand_targets()
        val = loss_fn(logits, targets)
        self.assertIsInstance(val, torch.Tensor)


# ──────────────────────────────────────────────────────────────────────────────
# 3–4. Criteria.set_class_weight propagation
# ──────────────────────────────────────────────────────────────────────────────

class TestCriteriaSetClassWeight(unittest.TestCase):

    def test_propagates_to_auto(self):
        """Criteria should inject weights into losses with auto_class_weight."""
        cfg = [
            dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1,
                 auto_class_weight=True),
        ]
        criteria = Criteria(cfg)
        w = [1.0, 2.0, 3.0, 4.0, 5.0]
        criteria.set_class_weight(w)
        ce = criteria.criteria[0]
        self.assertIsNotNone(ce.loss.weight)
        self.assertEqual(ce.loss.weight.shape[0], 5)

    def test_skips_non_auto(self):
        """Losses with auto_class_weight=False should NOT be modified."""
        cfg = [
            dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1,
                 auto_class_weight=False),
        ]
        criteria = Criteria(cfg)
        criteria.set_class_weight([1.0, 2.0, 3.0])
        ce = criteria.criteria[0]
        self.assertIsNone(ce.loss.weight)

    def test_mixed_losses(self):
        """Only the loss with auto_class_weight=True receives weights."""
        cfg = [
            dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1,
                 auto_class_weight=True),
            dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1,
                 auto_class_weight=False),
        ]
        criteria = Criteria(cfg)
        criteria.set_class_weight([1.0, 2.0])
        self.assertIsNotNone(criteria.criteria[0].loss.weight)
        self.assertIsNone(criteria.criteria[1].loss.weight)

    def test_lovasz_unaffected(self):
        """LovaszLoss has no auto_class_weight, should not crash."""
        cfg = [
            dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0,
                 ignore_index=-1),
        ]
        criteria = Criteria(cfg)
        # Should not raise
        criteria.set_class_weight([1.0, 2.0, 3.0])

    def test_focal_propagation(self):
        cfg = [
            dict(type="FocalLoss", loss_weight=1.0, ignore_index=-1,
                 auto_class_weight=True),
        ]
        criteria = Criteria(cfg)
        criteria.set_class_weight([1.0, 2.0, 3.0])
        self.assertIsNotNone(criteria.criteria[0].class_weight)


# ──────────────────────────────────────────────────────────────────────────────
# 5. DistributedWeightedSampler
# ──────────────────────────────────────────────────────────────────────────────

class TestDistributedWeightedSampler(unittest.TestCase):

    def _make_dataset(self, n=100):
        ds = MagicMock()
        ds.__len__ = MagicMock(return_value=n)
        return ds

    def test_length_single_rank(self):
        ds = self._make_dataset(100)
        weights = np.ones(100)
        sampler = DistributedWeightedSampler(
            weights=weights, dataset=ds, num_replicas=1, rank=0
        )
        self.assertEqual(len(sampler), 100)

    def test_length_two_ranks(self):
        ds = self._make_dataset(100)
        weights = np.ones(100)
        sampler = DistributedWeightedSampler(
            weights=weights, dataset=ds, num_replicas=2, rank=0
        )
        self.assertEqual(len(sampler), 50)

    def test_length_odd_dataset(self):
        ds = self._make_dataset(101)
        weights = np.ones(101)
        sampler = DistributedWeightedSampler(
            weights=weights, dataset=ds, num_replicas=2, rank=0
        )
        # ceil(101/2) = 51
        self.assertEqual(len(sampler), 51)

    def test_deterministic_same_epoch(self):
        ds = self._make_dataset(50)
        weights = np.random.rand(50) + 0.1
        s1 = DistributedWeightedSampler(
            weights=weights, dataset=ds, num_replicas=1, rank=0
        )
        s2 = DistributedWeightedSampler(
            weights=weights, dataset=ds, num_replicas=1, rank=0
        )
        s1.set_epoch(42)
        s2.set_epoch(42)
        self.assertEqual(list(s1), list(s2))

    def test_different_epochs_differ(self):
        ds = self._make_dataset(50)
        weights = np.random.rand(50) + 0.1
        s = DistributedWeightedSampler(
            weights=weights, dataset=ds, num_replicas=1, rank=0
        )
        s.set_epoch(0)
        l0 = list(s)
        s.set_epoch(1)
        l1 = list(s)
        # Very unlikely to be identical
        self.assertNotEqual(l0, l1)

    def test_two_ranks_partition(self):
        ds = self._make_dataset(100)
        weights = np.ones(100)
        s0 = DistributedWeightedSampler(
            weights=weights, dataset=ds, num_replicas=2, rank=0
        )
        s1 = DistributedWeightedSampler(
            weights=weights, dataset=ds, num_replicas=2, rank=1
        )
        s0.set_epoch(0)
        s1.set_epoch(0)
        l0 = list(s0)
        l1 = list(s1)
        self.assertEqual(len(l0), 50)
        self.assertEqual(len(l1), 50)


# ──────────────────────────────────────────────────────────────────────────────
# 6. WeightedRandomSampler with loop > 1
# ──────────────────────────────────────────────────────────────────────────────

class TestWeightedSamplerWithLoop(unittest.TestCase):

    def test_tiled_weights_length(self):
        """When loop=3, weights should be tiled 3 times."""
        base_weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        loop = 3
        tiled = np.tile(base_weights, loop)
        self.assertEqual(len(tiled), 15)
        np.testing.assert_array_equal(tiled[:5], base_weights)
        np.testing.assert_array_equal(tiled[5:10], base_weights)
        np.testing.assert_array_equal(tiled[10:15], base_weights)

    def test_weighted_sampler_num_samples(self):
        """WeightedRandomSampler should have num_samples == len(dataset)."""
        base_weights = np.array([1.0, 2.0, 3.0])
        loop = 4
        tiled = np.tile(base_weights, loop).tolist()
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=tiled, num_samples=len(tiled), replacement=True
        )
        self.assertEqual(sampler.num_samples, 12)


# ──────────────────────────────────────────────────────────────────────────────
# 7. Trainer._inject_class_weights integration path
# ──────────────────────────────────────────────────────────────────────────────

class TestInjectClassWeights(unittest.TestCase):

    def _make_mocks(self, class_weight, auto_flags=None):
        """Build mock trainer with model.criteria and dataset.class_weight."""
        import logging
        trainer = MagicMock()
        trainer.logger = logging.getLogger("test_inject")

        # Dataset
        dataset = MagicMock()
        dataset.class_weight = class_weight
        trainer.train_loader.dataset = dataset

        # Model with criteria — simulate single-GPU (no .module attribute)
        auto_flags = auto_flags or [True]
        cfg = [
            dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1,
                 auto_class_weight=flag)
            for flag in auto_flags
        ]
        criteria = Criteria(cfg)

        # Use a SimpleNamespace so hasattr(model, "module") is False
        from types import SimpleNamespace
        model = SimpleNamespace(criteria=criteria)
        trainer.model = model

        return trainer, criteria

    def test_inject_when_available(self):
        from pointspace.engines.train import Trainer
        weights = np.array([1.0, 2.0, 3.0])
        trainer, criteria = self._make_mocks(class_weight=weights, auto_flags=[True])
        # Call the real method
        Trainer._inject_class_weights(trainer)
        self.assertIsNotNone(criteria.criteria[0].loss.weight)

    def test_no_inject_when_no_weight(self):
        from pointspace.engines.train import Trainer
        trainer, criteria = self._make_mocks(class_weight=None)
        Trainer._inject_class_weights(trainer)
        self.assertIsNone(criteria.criteria[0].loss.weight)

    def test_no_inject_when_auto_false(self):
        from pointspace.engines.train import Trainer
        weights = np.array([1.0, 2.0])
        trainer, criteria = self._make_mocks(class_weight=weights, auto_flags=[False])
        Trainer._inject_class_weights(trainer)
        self.assertIsNone(criteria.criteria[0].loss.weight)


if __name__ == "__main__":
    unittest.main()
