"""
Tests for batched fragment inference in SemSegTester.

Validates that:
1. Fragment batching partitions correctly (e.g., 15 frags / bs=4 → [4,4,4,3])
2. Prediction accumulation is numerically identical between bs=1 and bs>1
3. Logging outputs per-batch cumulative fragment counts (e.g., 4/15, 8/15, 12/15, 15/15)

Author: PointSpace Team
"""

import math
import logging
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Utility: reproduce the batching logic extracted from SemSegTester.test()
# ---------------------------------------------------------------------------


def compute_fragment_batches(num_fragments: int, fragment_batch_size: int):
    """Return a list of (s_i, e_i) index pairs for each batch of fragments."""
    batch_num = int(math.ceil(num_fragments / fragment_batch_size))
    batches = []
    for i in range(batch_num):
        s_i = i * fragment_batch_size
        e_i = min((i + 1) * fragment_batch_size, num_fragments)
        batches.append((s_i, e_i))
    return batches


def compute_batch_sizes(num_fragments: int, fragment_batch_size: int):
    """Return a list of actual batch sizes (e.g., [4, 4, 4, 3] for 15/4)."""
    batches = compute_fragment_batches(num_fragments, fragment_batch_size)
    return [e - s for s, e in batches]


# ---------------------------------------------------------------------------
# Simulate the prediction accumulation logic from SemSegTester
# ---------------------------------------------------------------------------


def make_fake_fragment(num_points_total: int, num_points_frag: int, num_classes: int):
    """
    Create a fake fragment dict mimicking the structure expected by collate_fn.
    - 'coord': (num_points_frag, 3) random coords
    - 'index': indices into the full point cloud
    - 'offset': cumulative point count (for a single fragment, just [num_points_frag])
    """
    indices = np.sort(
        np.random.choice(num_points_total, size=num_points_frag, replace=False)
    )
    return dict(
        coord=torch.randn(num_points_frag, 3),
        index=torch.from_numpy(indices).long(),
        offset=torch.tensor([num_points_frag]).int(),
    )


def fake_collate_fn(fragment_list):
    """
    Minimal collate_fn that concatenates tensors and accumulates offsets,
    similar to pointspace.datasets.utils.collate_fn for Mapping inputs.
    """
    keys = fragment_list[0].keys()
    result = {}
    for key in keys:
        values = [d[key] for d in fragment_list]
        if key == "offset":
            # offset: diff -> concat -> cumsum (reproduce collate_fn behaviour)
            diffs = [v.diff(prepend=torch.tensor([0])) for v in values]
            result[key] = torch.cumsum(torch.cat(diffs), dim=0)
        else:
            result[key] = torch.cat(values)
    return result


def fake_model_forward(input_dict, num_classes):
    """
    Return a fake seg_logits tensor with deterministic values based on coord.
    Shape: (total_points_in_batch, num_classes).
    We use coord.sum(-1) as a seed so the output is reproducible.
    """
    n = input_dict["coord"].shape[0]
    # Simple deterministic logits: each class score = coord.sum(-1) + class_idx
    logits = input_dict["coord"].sum(-1, keepdim=True) + torch.arange(
        num_classes
    ).float().unsqueeze(0)
    return {"seg_logits": logits}


def run_fragment_inference(
    fragment_list, num_points_total, num_classes, fragment_batch_size
):
    """
    Reproduce the core fragment inference loop from SemSegTester.test()
    using the specified fragment_batch_size.
    """
    pred = torch.zeros(num_points_total, num_classes)
    num_fragments = len(fragment_list)
    batch_num = int(math.ceil(num_fragments / fragment_batch_size))
    log_entries = []

    for i in range(batch_num):
        s_i = i * fragment_batch_size
        e_i = min((i + 1) * fragment_batch_size, num_fragments)
        input_dict = fake_collate_fn(fragment_list[s_i:e_i])
        idx_part = input_dict["index"]

        pred_part = fake_model_forward(input_dict, num_classes)["seg_logits"]
        pred_part = F.softmax(pred_part, -1)

        bs = 0
        for be in input_dict["offset"]:
            pred[idx_part[bs:be], :] += pred_part[bs:be]
            bs = be

        log_entries.append(f"{e_i}/{num_fragments}")

    return pred, log_entries


# ===========================================================================
# Test Cases
# ===========================================================================


class TestFragmentBatchPartitioning(unittest.TestCase):
    """Test that fragments are partitioned into correct batch sizes."""

    def test_exact_division(self):
        """12 fragments / bs=4 → [4, 4, 4]"""
        sizes = compute_batch_sizes(12, 4)
        self.assertEqual(sizes, [4, 4, 4])

    def test_remainder(self):
        """15 fragments / bs=4 → [4, 4, 4, 3]"""
        sizes = compute_batch_sizes(15, 4)
        self.assertEqual(sizes, [4, 4, 4, 3])

    def test_single_fragment(self):
        """1 fragment / bs=4 → [1]"""
        sizes = compute_batch_sizes(1, 4)
        self.assertEqual(sizes, [1])

    def test_bs_one(self):
        """5 fragments / bs=1 → [1, 1, 1, 1, 1] (original behaviour)"""
        sizes = compute_batch_sizes(5, 1)
        self.assertEqual(sizes, [1, 1, 1, 1, 1])

    def test_bs_larger_than_fragments(self):
        """3 fragments / bs=8 → [3]"""
        sizes = compute_batch_sizes(3, 8)
        self.assertEqual(sizes, [3])

    def test_bs_equals_fragments(self):
        """4 fragments / bs=4 → [4]"""
        sizes = compute_batch_sizes(4, 4)
        self.assertEqual(sizes, [4])

    def test_all_fragments_covered(self):
        """Ensure all fragment indices are covered exactly once."""
        for n in [1, 5, 7, 15, 16, 100]:
            for bs in [1, 2, 3, 4, 8, 16, 64]:
                batches = compute_fragment_batches(n, bs)
                # No gaps
                for j in range(1, len(batches)):
                    self.assertEqual(batches[j][0], batches[j - 1][1])
                # Starts at 0, ends at n
                self.assertEqual(batches[0][0], 0)
                self.assertEqual(batches[-1][1], n)
                # Total count
                total = sum(e - s for s, e in batches)
                self.assertEqual(total, n)


class TestPredictionAccumulation(unittest.TestCase):
    """
    Verify that batched fragment inference (bs>1) produces numerically
    identical predictions as the original single-fragment inference (bs=1).
    """

    def _make_fragments(self, num_fragments, num_points_total, points_per_frag, num_classes):
        torch.manual_seed(42)
        np.random.seed(42)
        fragments = []
        for _ in range(num_fragments):
            fragments.append(
                make_fake_fragment(num_points_total, points_per_frag, num_classes)
            )
        return fragments

    def test_bs1_vs_bs4(self):
        """Predictions with bs=1 must equal predictions with bs=4."""
        num_points_total = 1000
        num_classes = 20
        points_per_frag = 200
        num_fragments = 15

        fragments = self._make_fragments(
            num_fragments, num_points_total, points_per_frag, num_classes
        )

        pred_bs1, _ = run_fragment_inference(
            fragments, num_points_total, num_classes, fragment_batch_size=1
        )
        pred_bs4, _ = run_fragment_inference(
            fragments, num_points_total, num_classes, fragment_batch_size=4
        )

        self.assertTrue(
            torch.allclose(pred_bs1, pred_bs4, atol=1e-6),
            f"Max diff: {(pred_bs1 - pred_bs4).abs().max().item()}"
        )

    def test_bs1_vs_bs_all(self):
        """Predictions with bs=1 must equal predictions with bs=N (all at once)."""
        num_points_total = 500
        num_classes = 10
        points_per_frag = 100
        num_fragments = 7

        fragments = self._make_fragments(
            num_fragments, num_points_total, points_per_frag, num_classes
        )

        pred_bs1, _ = run_fragment_inference(
            fragments, num_points_total, num_classes, fragment_batch_size=1
        )
        pred_all, _ = run_fragment_inference(
            fragments, num_points_total, num_classes, fragment_batch_size=num_fragments
        )

        self.assertTrue(
            torch.allclose(pred_bs1, pred_all, atol=1e-6),
            f"Max diff: {(pred_bs1 - pred_all).abs().max().item()}"
        )

    def test_various_batch_sizes(self):
        """All batch sizes produce identical results."""
        num_points_total = 800
        num_classes = 5
        points_per_frag = 150
        num_fragments = 13

        fragments = self._make_fragments(
            num_fragments, num_points_total, points_per_frag, num_classes
        )

        pred_ref, _ = run_fragment_inference(
            fragments, num_points_total, num_classes, fragment_batch_size=1
        )

        for bs in [2, 3, 4, 5, 7, 13, 20]:
            pred, _ = run_fragment_inference(
                fragments, num_points_total, num_classes, fragment_batch_size=bs
            )
            self.assertTrue(
                torch.allclose(pred_ref, pred, atol=1e-6),
                f"bs={bs}: max diff = {(pred_ref - pred).abs().max().item()}"
            )


class TestLoggingFormat(unittest.TestCase):
    """
    Verify that log entries show cumulative fragment counts per batch
    (e.g., 4/15, 8/15, 12/15, 15/15 for 15 fragments / bs=4).
    """

    def test_log_entries_bs4_15frags(self):
        """15 fragments / bs=4 → log entries: 4/15, 8/15, 12/15, 15/15"""
        num_points_total = 500
        num_classes = 5
        points_per_frag = 50
        num_fragments = 15

        torch.manual_seed(0)
        np.random.seed(0)
        fragments = [
            make_fake_fragment(num_points_total, points_per_frag, num_classes)
            for _ in range(num_fragments)
        ]

        _, log_entries = run_fragment_inference(
            fragments, num_points_total, num_classes, fragment_batch_size=4
        )

        self.assertEqual(log_entries, ["4/15", "8/15", "12/15", "15/15"])

    def test_log_entries_bs1(self):
        """5 fragments / bs=1 → log entries: 1/5, 2/5, 3/5, 4/5, 5/5"""
        num_points_total = 200
        num_classes = 3
        points_per_frag = 50
        num_fragments = 5

        torch.manual_seed(0)
        np.random.seed(0)
        fragments = [
            make_fake_fragment(num_points_total, points_per_frag, num_classes)
            for _ in range(num_fragments)
        ]

        _, log_entries = run_fragment_inference(
            fragments, num_points_total, num_classes, fragment_batch_size=1
        )

        self.assertEqual(log_entries, ["1/5", "2/5", "3/5", "4/5", "5/5"])

    def test_log_entries_bs_larger_than_frags(self):
        """3 fragments / bs=8 → log entries: 3/3"""
        num_points_total = 200
        num_classes = 3
        points_per_frag = 50
        num_fragments = 3

        torch.manual_seed(0)
        np.random.seed(0)
        fragments = [
            make_fake_fragment(num_points_total, points_per_frag, num_classes)
            for _ in range(num_fragments)
        ]

        _, log_entries = run_fragment_inference(
            fragments, num_points_total, num_classes, fragment_batch_size=8
        )

        self.assertEqual(log_entries, ["3/3"])

    def test_log_entries_exact_division(self):
        """12 fragments / bs=4 → log entries: 4/12, 8/12, 12/12"""
        num_points_total = 500
        num_classes = 5
        points_per_frag = 50
        num_fragments = 12

        torch.manual_seed(0)
        np.random.seed(0)
        fragments = [
            make_fake_fragment(num_points_total, points_per_frag, num_classes)
            for _ in range(num_fragments)
        ]

        _, log_entries = run_fragment_inference(
            fragments, num_points_total, num_classes, fragment_batch_size=4
        )

        self.assertEqual(log_entries, ["4/12", "8/12", "12/12"])


class TestGetAttrFallback(unittest.TestCase):
    """
    Verify that fragment_batch_size defaults to 1 when not in config,
    preserving backward compatibility.
    """

    def test_default_value(self):
        cfg = MagicMock(spec=[])  # empty spec, no attributes
        result = getattr(cfg, "fragment_batch_size", 1)
        self.assertEqual(result, 1)

    def test_custom_value(self):
        cfg = MagicMock()
        cfg.fragment_batch_size = 4
        result = getattr(cfg, "fragment_batch_size", 1)
        self.assertEqual(result, 4)


if __name__ == "__main__":
    unittest.main()
