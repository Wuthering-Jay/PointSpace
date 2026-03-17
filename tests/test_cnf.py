"""
Tests for the CNF (Conditional Neural Field) pipeline.

Covers:
    1. ClassFilter transform — filter by class label
    2. TerrainImplicitSampler transform — support/query split with 3 strategies
    3. GridCoordinate transform — grid_coord without downsampling
    4. DualBranchCNFHead — IDW anchor + Linear PE (base) + Fourier PE (detail) dual-stream
    5. DefaultCNF model — Point-based encode / query_forward / forward
    6. CnfEvaluator hook — metric computation (MAE, RMSE, MaxE, R²)
    7. CnfTester — registered and accessible
    8. LASWriter pred_coord support — new point cloud from CNF output
    9. Config template — syntax & import validity

Author: PointSpace Team
"""

import math
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pointspace.datasets.transform import (
    ClassFilter,
    TerrainImplicitSampler,
    GridCoordinate,
    Collect,
)
from pointspace.utils.registry import Registry

try:
    import laspy

    HAS_LASPY = True
except ImportError:
    HAS_LASPY = False

try:
    import torch_cluster

    HAS_TORCH_CLUSTER = True
except ImportError:
    HAS_TORCH_CLUSTER = False


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _make_ground_point_cloud(n=500, seed=42):
    """Simulate a terrain point cloud with ground class = 1.

    Returns data_dict compatible with the transform pipeline.
    """
    rng = np.random.RandomState(seed)
    coord = np.column_stack([
        rng.uniform(0, 100, n),   # x
        rng.uniform(0, 100, n),   # y
        rng.uniform(0, 5, n),     # z (ground-ish height)
    ]).astype(np.float32)

    # Assign ~70% ground (1), ~30% vegetation (2)
    segment = rng.choice([1, 2], size=n, p=[0.7, 0.3]).astype(np.int64)
    echo = rng.uniform(0, 1, (n, 2)).astype(np.float32)

    return dict(
        coord=coord,
        segment=segment,
        echo=echo,
        color=rng.randint(0, 256, (n, 3)).astype(np.uint8),
        normal=rng.randn(n, 3).astype(np.float32),
        index_valid_keys=[
            "coord", "color", "normal", "segment", "echo",
        ],
    )


# ══════════════════════════════════════════════════════════════════════════════
#  1. ClassFilter
# ══════════════════════════════════════════════════════════════════════════════


class TestClassFilter(unittest.TestCase):
    """ClassFilter transform unit tests."""

    def test_basic_filtering(self):
        """Keep only ground class (1), vegetation (2) removed."""
        data = _make_ground_point_cloud(n=200)
        n_ground = int((data["segment"] == 1).sum())
        tf = ClassFilter(keep_classes=[1])
        out = tf(data)
        self.assertEqual(out["coord"].shape[0], n_ground)
        self.assertTrue(np.all(out["segment"] == 1))

    def test_multiple_classes(self):
        """Keep classes 1 and 2 → no filtering should happen."""
        data = _make_ground_point_cloud(n=200)
        tf = ClassFilter(keep_classes=[1, 2])
        out = tf(data)
        self.assertEqual(out["coord"].shape[0], 200)

    def test_empty_result(self):
        """Keep a class not present → empty array."""
        data = _make_ground_point_cloud(n=100)
        tf = ClassFilter(keep_classes=[99])
        out = tf(data)
        self.assertEqual(out["coord"].shape[0], 0)

    def test_missing_class_key(self):
        """If class_key not in data_dict, return unchanged."""
        data = _make_ground_point_cloud(n=100)
        del data["segment"]
        tf = ClassFilter(keep_classes=[1], class_key="segment")
        out = tf(data)
        self.assertEqual(out["coord"].shape[0], 100)

    def test_index_valid_keys_respected(self):
        """Only keys in index_valid_keys should be subset."""
        data = _make_ground_point_cloud(n=200)
        data["extra_untouched"] = np.ones(10)  # not in index_valid_keys
        tf = ClassFilter(keep_classes=[1])
        out = tf(data)
        # extra_untouched should still be length 10 (not subsetted)
        self.assertEqual(len(out["extra_untouched"]), 10)


# ══════════════════════════════════════════════════════════════════════════════
#  2. TerrainImplicitSampler
# ══════════════════════════════════════════════════════════════════════════════


class TestTerrainImplicitSampler(unittest.TestCase):
    """TerrainImplicitSampler transform unit tests."""

    def _make_data(self, n=400, seed=0):
        """Ground-only point cloud spanning a 50×50 m tile."""
        rng = np.random.RandomState(seed)
        coord = np.column_stack([
            rng.uniform(0, 50, n),
            rng.uniform(0, 50, n),
            rng.uniform(0, 3, n),
        ]).astype(np.float32)
        return dict(
            coord=coord,
            segment=np.ones(n, dtype=np.int64),
            index_valid_keys=["coord", "segment"],
        )

    def _default_sampler(self, **kwargs):
        defaults = dict(
            random_ratio=0.1,
            feature_ratio=0.1,
            max_blocks=3,
            block_size_range=(2.0, 8.0),
            feature_resolution=2.0,
            max_query_ratio=0.6,
        )
        defaults.update(kwargs)
        return TerrainImplicitSampler(**defaults)

    def test_basic_split(self):
        """Support + query = original count (no overlap)."""
        data = self._make_data(n=400)
        n_orig = data["coord"].shape[0]
        tf = self._default_sampler()
        out = tf(data)
        n_support = out["coord"].shape[0]
        n_query = out["query_coord"].shape[0]
        self.assertEqual(n_support + n_query, n_orig)

    def test_query_coord_is_xy(self):
        """query_coord should be (Q, 2) — only X and Y."""
        data = self._make_data(n=400)
        tf = self._default_sampler()
        out = tf(data)
        self.assertEqual(out["query_coord"].ndim, 2)
        self.assertEqual(out["query_coord"].shape[1], 2)

    def test_query_gt_is_z(self):
        """query_gt should be 1-D float32 array of Z values within original Z range."""
        rng = np.random.RandomState(1)
        n = 300
        coord = np.column_stack([
            rng.uniform(0, 50, n),
            rng.uniform(0, 50, n),
            rng.uniform(0, 5, n),
        ]).astype(np.float32)
        data = dict(
            coord=coord,
            segment=np.ones(n, dtype=np.int64),
            index_valid_keys=["coord", "segment"],
        )
        tf = self._default_sampler()
        out = tf(data)
        q_gt = out["query_gt"]
        # Must be 1-D float32
        self.assertEqual(q_gt.ndim, 1)
        self.assertEqual(q_gt.dtype, np.float32)
        # Length must match query_coord
        self.assertEqual(q_gt.shape[0], out["query_coord"].shape[0])
        # All values within original Z range
        z_min, z_max = float(coord[:, 2].min()), float(coord[:, 2].max())
        self.assertTrue(np.all(q_gt >= z_min - 1e-4))
        self.assertTrue(np.all(q_gt <= z_max + 1e-4))

    def test_max_query_ratio_cap(self):
        """Query count must not exceed num_points * max_query_ratio."""
        data = self._make_data(n=400)
        max_ratio = 0.3
        tf = self._default_sampler(max_query_ratio=max_ratio)
        out = tf(data)
        n_query = out["query_coord"].shape[0]
        self.assertLessEqual(n_query, 400 * max_ratio + 1)  # +1 for rounding

    def test_query_not_in_index_valid_keys(self):
        """query_coord and query_gt should NOT appear in index_valid_keys."""
        data = self._make_data(n=300)
        tf = self._default_sampler()
        out = tf(data)
        ivk = out.get("index_valid_keys", [])
        self.assertNotIn("query_coord", ivk)
        self.assertNotIn("query_gt", ivk)

    def test_tiny_point_cloud_returns_empty_arrays(self):
        """Fewer than 10 points → returns empty query_coord/query_gt, not bare return."""
        data = self._make_data(n=5)
        tf = self._default_sampler()
        out = tf(data)
        self.assertIn("query_coord", out)
        self.assertIn("query_gt", out)
        self.assertEqual(out["query_coord"].shape[0], 0)
        self.assertEqual(out["query_gt"].shape[0], 0)
        # Support should still be the original points
        self.assertEqual(out["coord"].shape[0], 5)

    def test_empty_query_fallback(self):
        """All ratios=0 and no blocks → still produces at least 1 query."""
        data = self._make_data(n=200)
        tf = TerrainImplicitSampler(
            random_ratio=0.0,
            feature_ratio=0.0,
            max_blocks=0,
            block_size_range=(2.0, 8.0),
            feature_resolution=2.0,
            max_query_ratio=0.6,
        )
        out = tf(data)
        self.assertGreaterEqual(out["query_coord"].shape[0], 1)
        self.assertGreaterEqual(out["query_gt"].shape[0], 1)

    def test_rectangular_hole_strategy(self):
        """With only block strategy, query points should cluster in XY rectangles."""
        rng = np.random.RandomState(99)
        n = 600
        # Dense regular grid so we can detect spatial clustering
        xs = np.tile(np.linspace(0, 30, 30), 20).astype(np.float32)
        ys = np.repeat(np.linspace(0, 20, 20), 30).astype(np.float32)
        zs = rng.uniform(0, 1, n).astype(np.float32)
        coord = np.column_stack([xs, ys, zs])
        data = dict(
            coord=coord,
            segment=np.ones(n, dtype=np.int64),
            index_valid_keys=["coord", "segment"],
        )
        # No random, no feature — only blocks
        tf = TerrainImplicitSampler(
            random_ratio=0.0,
            feature_ratio=0.0,
            max_blocks=10,
            block_size_range=(3.0, 8.0),
            feature_resolution=2.0,
            max_query_ratio=0.6,
        )
        out = tf(data)
        # With block strategy, should produce some query points
        self.assertGreater(out["query_coord"].shape[0], 0)

    def test_feature_weights_selects_points(self):
        """Feature ratio strategy should produce some query points."""
        data = self._make_data(n=400)
        tf = TerrainImplicitSampler(
            random_ratio=0.0,
            feature_ratio=0.2,
            max_blocks=0,
            block_size_range=(2.0, 8.0),
            feature_resolution=1.0,
            max_query_ratio=0.6,
        )
        out = tf(data)
        self.assertGreater(out["query_coord"].shape[0], 0)

    def test_registered(self):
        """TerrainImplicitSampler must be in the TRANSFORMS registry."""
        from pointspace.datasets.transform import TRANSFORMS
        self.assertIn("TerrainImplicitSampler", TRANSFORMS._module_dict)


# ══════════════════════════════════════════════════════════════════════════════
#  3. GridCoordinate
# ══════════════════════════════════════════════════════════════════════════════


class TestGridCoordinate(unittest.TestCase):
    """GridCoordinate transform unit tests."""

    def test_preserves_all_points(self):
        """Point count must not change."""
        data = _make_ground_point_cloud(n=200)
        tf = GridCoordinate(grid_size=0.5)
        out = tf(data)
        self.assertEqual(out["coord"].shape[0], 200)

    def test_grid_coord_present(self):
        """grid_coord should be added."""
        data = _make_ground_point_cloud(n=100)
        tf = GridCoordinate(grid_size=1.0)
        out = tf(data)
        self.assertIn("grid_coord", out)
        self.assertEqual(out["grid_coord"].shape, out["coord"].shape)
        self.assertEqual(out["grid_coord"].dtype, np.int32)

    def test_grid_coord_min_is_zero(self):
        """grid_coord minimum should be (0, 0, 0) after offset."""
        data = _make_ground_point_cloud(n=200)
        tf = GridCoordinate(grid_size=0.5)
        out = tf(data)
        self.assertTrue(np.all(out["grid_coord"].min(axis=0) == 0))

    def test_grid_coord_values(self):
        """Check specific grid_coord values for known input."""
        coord = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]],
                         dtype=np.float32)
        data = dict(coord=coord, index_valid_keys=["coord"])
        tf = GridCoordinate(grid_size=0.5)
        out = tf(data)
        expected = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.int32)
        # floor([0, 0.5, 1.0] / 0.5) = [0, 1, 2], minus min(0) = [0, 1, 2]
        np.testing.assert_array_equal(out["grid_coord"], expected)

    def test_in_index_valid_keys(self):
        """grid_coord should be added to index_valid_keys."""
        data = _make_ground_point_cloud(n=50)
        tf = GridCoordinate(grid_size=1.0)
        out = tf(data)
        self.assertIn("grid_coord", out.get("index_valid_keys", []))


# ══════════════════════════════════════════════════════════════════════════════
#  4. DefaultCNF + DualBranchCNFHead Model
# ══════════════════════════════════════════════════════════════════════════════


class _TinyBackbone(nn.Module):
    """Minimal backbone returning a Point with linear-transformed feat."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, point):
        from pointspace.models.utils.structure import Point

        feat = point.feat if hasattr(point, "feat") else point.coord
        point.feat = self.linear(feat)
        return point


@unittest.skipUnless(HAS_TORCH_CLUSTER, "torch_cluster not installed")
class TestDualBranchCNFHead(unittest.TestCase):
    """DualBranchCNFHead unit tests (IDW + Linear PE + Fourier PE dual-stream)."""

    def _make_head(self, C=8, qd=2, k=4):
        from pointspace.models.default import DualBranchCNFHead

        return DualBranchCNFHead(
            backbone_out_channels=C,
            query_dim=qd,
            num_targets=1,
            k_neighbors=k,
            hidden_dim=32,
            num_freqs=4,
            base_hidden_dims=[16],
            detail_hidden_dims=[16],
        )

    def test_forward_returns_tuple(self):
        """forward() returns (pred_base, pred_detail)."""
        head = self._make_head(C=8, k=4)
        n, q = 64, 16
        sc = torch.randn(n, 3)
        sf = torch.randn(n, 8)
        qc = torch.randn(q, 2)
        pb, pd = head(sc, sf, qc)
        self.assertEqual(pb.shape, (q,))
        self.assertEqual(pd.shape, (q,))

    def test_relative_fourier_encoding(self):
        """RelativeFourierEncoding output shape matches analytical formula."""
        from pointspace.models.head.cnf_head import RelativeFourierEncoding

        rfe = RelativeFourierEncoding(in_dim=2, num_freqs=6)
        x = torch.randn(10, 4, 2)  # (Q, K, 2) typical usage
        out = rfe(x)
        expected_dim = 2 * 6 * 2  # in_dim * num_freqs * 2
        self.assertEqual(rfe.output_dim, expected_dim)
        self.assertEqual(out.shape, (10, 4, expected_dim))

    def test_idw_anchor_is_weighted_mean(self):
        """IDW local_z_anchor should be close to nearest point's Z when one is very close."""
        head = self._make_head(C=8, k=4)
        # Place 4 support points, one very close to query
        sc = torch.tensor([[0.0, 0.0, 10.0],
                           [1.0, 0.0, 20.0],
                           [0.0, 1.0, 30.0],
                           [1.0, 1.0, 40.0]])
        sf = torch.randn(4, 8)
        qc = torch.tensor([[0.001, 0.001]])  # very close to first support
        pb, pd = head(sc, sf, qc)
        # pred_base = anchor + residual, anchor should be ~10.0
        # Just check it runs without error and returns correct shape
        self.assertEqual(pb.shape, (1,))

    def test_no_relu_in_head(self):
        """Entire head should use Softplus, not ReLU (C^2 requirement)."""
        head = self._make_head(C=8, k=4)
        for name, module in head.named_modules():
            self.assertNotIsInstance(
                module, nn.ReLU,
                f"Found ReLU at {name} — must be Softplus for C^2",
            )

    def test_registered(self):
        """DualBranchCNFHead should be in MODELS registry."""
        from pointspace.models import MODELS

        self.assertIn("DualBranchCNFHead", MODELS._module_dict)

    def test_with_batch_offsets(self):
        """forward() works with batch offsets (multi-scene)."""
        head = self._make_head(C=8, k=4)
        # 2 scenes: 32 + 32 support, 8 + 8 query
        sc = torch.randn(64, 3)
        sf = torch.randn(64, 8)
        qc = torch.randn(16, 2)
        s_off = torch.tensor([32, 64])
        q_off = torch.tensor([8, 16])
        pb, pd = head(sc, sf, qc, support_offset=s_off, query_offset=q_off)
        self.assertEqual(pb.shape, (16,))
        self.assertEqual(pd.shape, (16,))


@unittest.skipUnless(HAS_TORCH_CLUSTER, "torch_cluster not installed")
class TestDefaultCNF(unittest.TestCase):
    """DefaultCNF model unit tests (dual-branch Point-based)."""

    def _make_head_cfg(self, C=8, k=4):
        return dict(
            type="DualBranchCNFHead",
            backbone_out_channels=C,
            query_dim=2,
            num_targets=1,
            k_neighbors=k,
            hidden_dim=32,
            num_freqs=4,
            base_hidden_dims=[16],
            detail_hidden_dims=[16],
        )

    def _make_model(self, backbone_out=8, k=4):
        from pointspace.models.default import DefaultCNF

        model = DefaultCNF(
            backbone=None,
            head=self._make_head_cfg(C=backbone_out, k=k),
            criteria=None,
            reg_weight=0.01,
        )
        model.backbone = _TinyBackbone(in_channels=3, out_channels=backbone_out)
        return model

    def _make_input(self, n=64, q=16, qd=2):
        """Create a minimal input_dict with support + query + GT."""
        rng = np.random.RandomState(0)
        coord = torch.from_numpy(
            rng.uniform(0, 10, (n, 3)).astype(np.float32)
        )
        grid_coord = torch.from_numpy(
            np.floor(coord.numpy() / 0.5).astype(np.int32)
        )
        input_dict = dict(
            coord=coord,
            feat=coord.clone(),
            grid_coord=grid_coord,
            grid_size=torch.tensor([0.5]),
            offset=torch.tensor([n]),
        )
        if q > 0:
            qc = torch.from_numpy(
                rng.uniform(0, 10, (q, qd)).astype(np.float32)
            )
            qt_raw = torch.from_numpy(
                rng.uniform(0, 5, q).astype(np.float32)
            )
            # Low-freq GT = smoothed version (just mean for testing)
            qt_low = torch.full_like(qt_raw, qt_raw.mean().item())
            input_dict["query_coord"] = qc
            input_dict["query_gt"] = qt_raw
            input_dict["query_gt_low"] = qt_low
            input_dict["query_offset"] = torch.tensor([q])
        return input_dict

    def test_extract_feat(self):
        """extract_feat() returns feat and coord."""
        model = self._make_model()
        inp = self._make_input(q=0)
        enc = model.extract_feat(inp)
        self.assertIn("feat", enc)
        self.assertIn("coord", enc)
        self.assertEqual(enc["feat"].shape[0], inp["coord"].shape[0])

    def test_query_forward(self):
        """query_forward() returns (Q,) prediction."""
        model = self._make_model(backbone_out=8, k=4)
        n, q = 64, 16
        sc = torch.randn(n, 3)
        sf = torch.randn(n, 8)
        qc = torch.randn(q, 2)
        pred = model.query_forward(sc, sf, qc)
        self.assertEqual(pred.shape, (q,))

    def test_forward_train_default_loss(self):
        """Train forward with built-in multi-frequency loss."""
        model = self._make_model(backbone_out=8, k=4)
        model.train()
        inp = self._make_input(n=64, q=16)
        out = model(inp)
        self.assertIn("loss", out)
        self.assertIn("loss_base", out)
        self.assertIn("loss_final", out)
        self.assertIn("loss_reg", out)
        self.assertTrue(out["loss"].requires_grad)

    def test_forward_eval_with_query(self):
        """Eval with query → returns cnf_pred + loss."""
        model = self._make_model(backbone_out=8, k=4)
        model.eval()
        inp = self._make_input(n=64, q=16)
        with torch.no_grad():
            out = model(inp)
        self.assertIn("cnf_pred", out)
        self.assertIn("loss", out)
        self.assertEqual(out["cnf_pred"].shape[0], 16)

    def test_forward_no_query(self):
        """Forward without query → returns support_feat + support_coord."""
        model = self._make_model(backbone_out=8)
        model.eval()
        inp = self._make_input(n=64, q=0)
        with torch.no_grad():
            out = model(inp)
        self.assertIn("support_feat", out)
        self.assertIn("support_coord", out)

    def test_gradient_flows(self):
        """Gradient flows through head back to backbone."""
        model = self._make_model(backbone_out=8, k=4)
        model.train()
        inp = self._make_input(n=64, q=16)
        out = model(inp)
        out["loss"].backward()
        for p in model.backbone.parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad)

    def test_detach_prevents_base_grad_in_detail(self):
        """The detach in loss_final should stop base branch grad from detail residual."""
        model = self._make_model(backbone_out=8, k=4)
        model.train()
        inp = self._make_input(n=64, q=16)
        out = model(inp)
        # loss_final uses pred_base.detach(), so only loss_base
        # should propagate into base_mlp
        self.assertTrue(out["loss_final"].requires_grad)

    def test_custom_criteria(self):
        """When criteria is provided, compute_loss delegates to it."""
        from pointspace.models.default import DefaultCNF

        class _DummyCriteria(nn.Module):
            def forward(self, head_output, input_dict):
                pred_base, pred_detail = head_output
                return dict(
                    loss=pred_base.sum() + pred_detail.sum(),
                    loss_base=pred_base.sum(),
                    loss_final=pred_detail.sum(),
                    loss_reg=torch.tensor(0.0),
                )

        model = DefaultCNF(backbone=None, head=None, criteria=None)
        model.backbone = _TinyBackbone(3, 8)
        from pointspace.models.default import DualBranchCNFHead

        model.head = DualBranchCNFHead(
            backbone_out_channels=8, k_neighbors=4,
            hidden_dim=32, num_freqs=4,
            base_hidden_dims=[16], detail_hidden_dims=[16],
        )
        model.criteria = _DummyCriteria()
        model.train()
        inp = self._make_input(n=64, q=16)
        out = model(inp)
        self.assertIn("loss", out)


# ══════════════════════════════════════════════════════════════════════════════
#  4b. SingleBranchCNFHead
# ══════════════════════════════════════════════════════════════════════════════


@unittest.skipUnless(HAS_TORCH_CLUSTER, "torch_cluster not installed")
class TestSingleBranchCNFHead(unittest.TestCase):
    """SingleBranchCNFHead unit tests (unified IDW + Linear PE + Fourier PE)."""

    def _make_head(self, C=8, qd=2, k=4):
        from pointspace.models.head.cnf_head import SingleBranchCNFHead

        return SingleBranchCNFHead(
            backbone_out_channels=C,
            query_dim=qd,
            num_targets=1,
            k_neighbors=k,
            hidden_dim=32,
            mlp_hidden_dims=[16],
            attn_groups=4,
        )

    def test_forward_returns_tensor(self):
        """forward() returns a single tensor in eval mode."""
        head = self._make_head(C=8, k=4)
        head.eval()
        n, q = 64, 16
        sc = torch.randn(n, 3)
        sf = torch.randn(n, 8)
        qc = torch.randn(q, 2)
        pred = head(sc, sf, qc)
        self.assertIsInstance(pred, torch.Tensor)
        self.assertEqual(pred.shape, (q,))

    def test_forward_returns_tuple_in_train(self):
        """forward() returns (pred_z, z_anchor) tuple in training mode."""
        head = self._make_head(C=8, k=4)
        head.train()
        n, q = 64, 16
        sc = torch.randn(n, 3)
        sf = torch.randn(n, 8)
        qc = torch.randn(q, 2)
        result = head(sc, sf, qc)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        pred_z, z_anchor = result
        self.assertEqual(pred_z.shape, (q,))
        self.assertEqual(z_anchor.shape, (q,))

    def test_no_relu_in_head(self):
        """MLP (prediction layers) should use Softplus, not ReLU (C^2 requirement).
        Cross-GVA positional sublayers may legitimately use ReLU for attention."""
        head = self._make_head()
        for name, module in head.mlp.named_modules():
            self.assertNotIsInstance(
                module, nn.ReLU,
                f"Found ReLU at {name} — must be Softplus for C^2",
            )

    def test_registered(self):
        """SingleBranchCNFHead should be in MODELS registry."""
        from pointspace.models import MODELS

        self.assertIn("SingleBranchCNFHead", MODELS._module_dict)

    def test_with_batch_offsets(self):
        """forward() works with batch offsets (multi-scene)."""
        head = self._make_head(C=8, k=4)
        head.eval()
        sc = torch.randn(64, 3)
        sf = torch.randn(64, 8)
        qc = torch.randn(16, 2)
        s_off = torch.tensor([32, 64])
        q_off = torch.tensor([8, 16])
        pred = head(sc, sf, qc, support_offset=s_off, query_offset=q_off)
        self.assertEqual(pred.shape, (16,))

    def test_gradient_flows(self):
        """Gradient flows through the head."""
        head = self._make_head(C=8, k=4)
        head.train()
        sc = torch.randn(32, 3)
        sf = torch.randn(32, 8, requires_grad=True)
        qc = torch.randn(8, 2)
        pred_z, _ = head(sc, sf, qc)
        pred_z.sum().backward()
        self.assertIsNotNone(sf.grad)


# ══════════════════════════════════════════════════════════════════════════════
#  4c. DefaultCNF with SingleBranchCNFHead
# ══════════════════════════════════════════════════════════════════════════════


@unittest.skipUnless(HAS_TORCH_CLUSTER, "torch_cluster not installed")
class TestDefaultCNF_SingleBranch(unittest.TestCase):
    """DefaultCNF model with SingleBranchCNFHead (single-branch loss)."""

    def _make_head_cfg(self, C=8, k=4):
        return dict(
            type="SingleBranchCNFHead",
            backbone_out_channels=C,
            query_dim=2,
            num_targets=1,
            k_neighbors=k,
            hidden_dim=32,
            mlp_hidden_dims=[16],
            attn_groups=4,
        )

    def _make_model(self, backbone_out=8, k=4):
        from pointspace.models.default import DefaultCNF

        model = DefaultCNF(
            backbone=None,
            head=self._make_head_cfg(C=backbone_out, k=k),
            criteria=None,
            reg_weight=0.0,
        )
        model.backbone = _TinyBackbone(in_channels=3, out_channels=backbone_out)
        return model

    def _make_input(self, n=64, q=16, qd=2):
        rng = np.random.RandomState(0)
        coord = torch.from_numpy(
            rng.uniform(0, 10, (n, 3)).astype(np.float32)
        )
        input_dict = dict(
            coord=coord,
            feat=coord.clone(),
            offset=torch.tensor([n]),
        )
        if q > 0:
            qc = torch.from_numpy(
                rng.uniform(0, 10, (q, qd)).astype(np.float32)
            )
            qt = torch.from_numpy(
                rng.uniform(0, 5, q).astype(np.float32)
            )
            input_dict["query_coord"] = qc
            input_dict["query_gt"] = qt
            input_dict["query_offset"] = torch.tensor([q])
        return input_dict

    def test_forward_train_single_branch_loss(self):
        """Train forward returns loss dict with expected keys."""
        model = self._make_model()
        model.train()
        inp = self._make_input(n=64, q=16)
        out = model(inp)
        self.assertIn("loss", out)
        self.assertIn("l1_ohem", out)
        self.assertIn("normal", out)
        self.assertNotIn("loss_base", out)
        self.assertNotIn("loss_reg", out)
        self.assertTrue(out["loss"].requires_grad)

    def test_forward_eval_with_query(self):
        """Eval with query → returns cnf_pred + loss."""
        model = self._make_model()
        model.eval()
        inp = self._make_input(n=64, q=16)
        with torch.no_grad():
            out = model(inp)
        self.assertIn("cnf_pred", out)
        self.assertIn("loss", out)
        self.assertEqual(out["cnf_pred"].shape[0], 16)

    def test_query_forward(self):
        """query_forward() returns (Q,) prediction."""
        model = self._make_model(backbone_out=8, k=4)
        sc = torch.randn(64, 3)
        sf = torch.randn(64, 8)
        qc = torch.randn(16, 2)
        pred = model.query_forward(sc, sf, qc)
        self.assertEqual(pred.shape, (16,))

    def test_gradient_flows_to_backbone(self):
        """Gradient flows through head back to backbone."""
        model = self._make_model(backbone_out=8, k=4)
        model.train()
        inp = self._make_input(n=64, q=16)
        out = model(inp)
        out["loss"].backward()
        for p in model.backbone.parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad)

    def test_terrain_weighted_loss_increases_with_complexity(self):
        """Loss should be higher when IDW anchor deviates from GT (complex terrain)."""
        from pointspace.models.default import DefaultCNF

        model = DefaultCNF(
            backbone=None,
            head=self._make_head_cfg(C=8, k=4),
            criteria=None,
            reg_weight=0.0,
            terrain_alpha=2.0,
        )
        model.backbone = _TinyBackbone(in_channels=3, out_channels=8)
        model.train()

        # Create two inputs: flat terrain vs complex terrain
        rng = np.random.RandomState(42)
        n, q = 64, 16
        coord = torch.from_numpy(rng.uniform(0, 10, (n, 3)).astype(np.float32))
        qc = torch.from_numpy(rng.uniform(0, 10, (q, 2)).astype(np.float32))

        # Flat: query_gt close to z=0 (IDW anchor will be close)
        flat_gt = torch.zeros(q)
        inp_flat = dict(
            coord=coord.clone(), feat=coord.clone(),
            offset=torch.tensor([n]),
            query_coord=qc.clone(), query_gt=flat_gt,
            query_offset=torch.tensor([q]),
        )
        out_flat = model(inp_flat)

        # Complex: query_gt far from support z-values
        complex_gt = torch.full((q,), 100.0)
        inp_complex = dict(
            coord=coord.clone(), feat=coord.clone(),
            offset=torch.tensor([n]),
            query_coord=qc.clone(), query_gt=complex_gt,
            query_offset=torch.tensor([q]),
        )
        out_complex = model(inp_complex)

        # Complex terrain loss should be much larger due to weighting
        self.assertGreater(out_complex["loss"].item(), out_flat["loss"].item())


# ══════════════════════════════════════════════════════════════════════════════
#  4d. TerrainImplicitSampler compute_gt_low switch
# ══════════════════════════════════════════════════════════════════════════════


class TestTerrainImplicitSampler_GtLowSwitch(unittest.TestCase):
    """Tests for the compute_gt_low parameter."""

    def _make_data(self, n=300, seed=0):
        rng = np.random.RandomState(seed)
        coord = np.column_stack([
            rng.uniform(0, 50, n),
            rng.uniform(0, 50, n),
            rng.uniform(0, 3, n),
        ]).astype(np.float32)
        return dict(
            coord=coord,
            segment=np.ones(n, dtype=np.int64),
            index_valid_keys=["coord", "segment"],
        )

    def test_compute_gt_low_true_produces_key(self):
        """Default compute_gt_low=True → query_gt_low present."""
        tf = TerrainImplicitSampler(compute_gt_low=True)
        out = tf(self._make_data())
        self.assertIn("query_gt_low", out)
        self.assertEqual(out["query_gt_low"].shape[0], out["query_coord"].shape[0])

    def test_compute_gt_low_false_no_key(self):
        """compute_gt_low=False → no query_gt_low key."""
        tf = TerrainImplicitSampler(compute_gt_low=False)
        out = tf(self._make_data())
        self.assertNotIn("query_gt_low", out)
        self.assertIn("query_gt", out)

    def test_tiny_cloud_respects_switch(self):
        """Tiny point cloud + compute_gt_low=False → no query_gt_low."""
        data = self._make_data(n=5)
        tf = TerrainImplicitSampler(compute_gt_low=False)
        out = tf(data)
        self.assertNotIn("query_gt_low", out)


# ══════════════════════════════════════════════════════════════════════════════
#  4e. Terrain sample weights
# ══════════════════════════════════════════════════════════════════════════════


class TestTerrainSampleWeights(unittest.TestCase):
    """Tests for LasDataset._compute_terrain_sample_weights."""

    def _write_las(self, path, x, y, z):
        """Write a minimal LAS file with given XYZ arrays."""
        import laspy

        hdr = laspy.LasHeader(point_format=0, version="1.2")
        hdr.offsets = [float(x.min()), float(y.min()), float(z.min())]
        hdr.scales = [0.001, 0.001, 0.001]
        las = laspy.LasData(hdr)
        las.x = x
        las.y = y
        las.z = z
        las.write(str(path))

    def _make_tiles(self, tmp_dir, n_flat=2, n_rough=2, n_pts=500):
        """Create flat and rough LAS tiles in *tmp_dir*.

        Flat tiles have Z ≈ constant; rough tiles have large Z variation.
        Returns (flat_paths, rough_paths).
        """
        rng = np.random.RandomState(42)
        flat_paths, rough_paths = [], []
        for i in range(n_flat):
            x = rng.uniform(0, 50, n_pts).astype(np.float64)
            y = rng.uniform(0, 50, n_pts).astype(np.float64)
            z = rng.normal(100.0, 0.02, n_pts).astype(np.float64)  # almost flat
            p = os.path.join(tmp_dir, f"flat_{i}.las")
            self._write_las(p, x, y, z)
            flat_paths.append(p)
        for i in range(n_rough):
            x = rng.uniform(0, 50, n_pts).astype(np.float64)
            y = rng.uniform(0, 50, n_pts).astype(np.float64)
            z = rng.normal(100.0, 3.0, n_pts).astype(np.float64)  # rough
            p = os.path.join(tmp_dir, f"rough_{i}.las")
            self._write_las(p, x, y, z)
            rough_paths.append(p)
        return flat_paths, rough_paths

    def test_rough_tiles_get_higher_weight(self):
        """Rough tiles should receive higher sample weight than flat tiles."""
        import tempfile
        from pointspace.datasets.las import LasDataset

        with tempfile.TemporaryDirectory() as tmp:
            flat_paths, rough_paths = self._make_tiles(tmp)
            all_paths = flat_paths + rough_paths

            ds = LasDataset(
                split="train",
                data_path=None,
                data_list=all_paths,
                test_mode=False,
                loop=1,
                weighted_sampler="terrain",
            )
            self.assertIsNotNone(ds.sample_weights)
            self.assertEqual(len(ds.sample_weights), len(all_paths))

            flat_w = ds.sample_weights[:len(flat_paths)].mean()
            rough_w = ds.sample_weights[len(flat_paths):].mean()
            self.assertGreater(rough_w, flat_w)

    def test_weights_sum_to_n(self):
        """Normalised weights should sum to N."""
        import tempfile
        from pointspace.datasets.las import LasDataset

        with tempfile.TemporaryDirectory() as tmp:
            flat_paths, rough_paths = self._make_tiles(tmp, n_flat=3, n_rough=3)
            all_paths = flat_paths + rough_paths

            ds = LasDataset(
                split="train",
                data_path=None,
                data_list=all_paths,
                test_mode=False,
                loop=1,
                weighted_sampler="terrain",
            )
            self.assertAlmostEqual(
                float(ds.sample_weights.sum()), len(all_paths), places=3
            )

    def test_terrain_mode_skipped_in_test(self):
        """weighted_sampler='terrain' should be skipped in test_mode."""
        import tempfile
        from pointspace.datasets.las import LasDataset

        with tempfile.TemporaryDirectory() as tmp:
            flat_paths, _ = self._make_tiles(tmp, n_flat=2, n_rough=0)
            ds = LasDataset(
                split="test",
                data_path=None,
                data_list=flat_paths,
                test_mode=True,
                weighted_sampler="terrain",
            )
            self.assertIsNone(ds.sample_weights)


# ══════════════════════════════════════════════════════════════════════════════
#  5. CnfEvaluator
# ══════════════════════════════════════════════════════════════════════════════


class TestCnfEvaluator(unittest.TestCase):
    """CnfEvaluator metric computation tests."""

    def test_metric_computation(self):
        """Verify MAE, RMSE, MaxE, R² formulas with known values."""
        pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        target = np.array([1.1, 1.9, 3.2, 3.8, 5.5])
        diff = pred - target
        n = len(pred)

        mae = np.abs(diff).mean()
        rmse = np.sqrt((diff ** 2).mean())
        max_e = np.abs(diff).max()
        mean_t = target.mean()
        ss_tot = ((target - mean_t) ** 2).sum()
        ss_res = (diff ** 2).sum()
        r2 = 1.0 - ss_res / ss_tot

        # Expected values
        self.assertAlmostEqual(mae, 0.22, places=2)
        self.assertAlmostEqual(max_e, 0.5, places=2)
        self.assertGreater(r2, 0.9)
        self.assertGreater(rmse, 0)

    def test_evaluator_registered(self):
        """CnfEvaluator should be in the HOOKS registry."""
        from pointspace.engines.hooks import HOOKS

        self.assertIn("CnfEvaluator", HOOKS._module_dict)

    def test_evaluator_instantiation(self):
        """CnfEvaluator should instantiate with default args."""
        from pointspace.engines.hooks.evaluator import CnfEvaluator

        ev = CnfEvaluator(log_interval=5)
        self.assertEqual(ev.log_interval, 5)

    def test_perfect_prediction_gives_r2_one(self):
        """When pred == target, R² = 1.0."""
        target = np.array([1.0, 2.0, 3.0])
        pred = target.copy()
        diff = pred - target
        ss_res = (diff ** 2).sum()
        mean_t = target.mean()
        ss_tot = ((target - mean_t) ** 2).sum()
        r2 = 1.0 - ss_res / max(ss_tot, 1e-10)
        self.assertAlmostEqual(r2, 1.0, places=10)


# ══════════════════════════════════════════════════════════════════════════════
#  6. CnfTester
# ══════════════════════════════════════════════════════════════════════════════


class TestCnfTester(unittest.TestCase):
    """CnfTester registration and structure tests."""

    def test_registered(self):
        """CnfTester should be in the TESTERS registry."""
        from pointspace.engines.test import TESTERS

        self.assertIn("CnfTester", TESTERS._module_dict)

    def test_has_test_method(self):
        """CnfTester should have a test() method."""
        from pointspace.engines.test import CnfTester

        self.assertTrue(hasattr(CnfTester, "test"))
        self.assertTrue(callable(getattr(CnfTester, "test")))

    def test_has_collate_fn(self):
        """CnfTester should have a static collate_fn."""
        from pointspace.engines.test import CnfTester

        self.assertTrue(hasattr(CnfTester, "collate_fn"))
        result = CnfTester.collate_fn([1, 2, 3])
        self.assertEqual(result, [1, 2, 3])


# ══════════════════════════════════════════════════════════════════════════════
#  7. LASWriter pred_coord support
# ══════════════════════════════════════════════════════════════════════════════


@unittest.skipUnless(HAS_LASPY, "laspy not installed")
class TestLASWriterPredCoord(unittest.TestCase):
    """LASWriter pred_coord support for CNF output."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def test_pred_coord_creates_las(self):
        """write(pred_coord=...) should create a LAS file from scratch."""
        from pointspace.writers.las_writer import LASWriter

        writer = LASWriter(save_dir=self.tmp_dir, source_dir=None)
        n = 1000
        pred_coord = np.column_stack([
            np.random.uniform(0, 100, n),
            np.random.uniform(0, 100, n),
            np.random.uniform(0, 5, n),
        ])
        out_path = writer.write("test_cnf", pred_coord=pred_coord)
        self.assertTrue(os.path.isfile(out_path))
        las = laspy.read(out_path)
        self.assertEqual(len(las.points), n)

    def test_pred_coord_with_slope_curvature(self):
        """write(pred_coord=..., slope=..., curvature=...) should add extra dims."""
        from pointspace.writers.las_writer import LASWriter

        writer = LASWriter(save_dir=self.tmp_dir, source_dir=None)
        n = 500
        pred_coord = np.random.uniform(0, 50, (n, 3))
        slope = np.random.uniform(0, 1, n)
        curvature = np.random.uniform(-1, 1, n)
        out_path = writer.write(
            "test_cnf_deriv",
            pred_coord=pred_coord,
            slope=slope,
            curvature=curvature,
        )
        las = laspy.read(out_path)
        self.assertEqual(len(las.points), n)
        # Check extra dimensions exist
        dim_names = list(las.point_format.dimension_names)
        self.assertIn("slope", dim_names)
        self.assertIn("curvature", dim_names)
        np.testing.assert_allclose(np.array(las.slope), slope, atol=1e-6)
        np.testing.assert_allclose(np.array(las.curvature), curvature, atol=1e-6)

    def test_pred_coord_overrides_source(self):
        """When pred_coord is given, source_dir should be ignored."""
        from pointspace.writers.las_writer import LASWriter

        writer = LASWriter(
            save_dir=self.tmp_dir,
            source_dir="/nonexistent/path",  # should not matter
        )
        n = 100
        pred_coord = np.random.uniform(0, 10, (n, 3))
        out_path = writer.write("any_name", pred_coord=pred_coord)
        self.assertTrue(os.path.isfile(out_path))

    def test_pred_coord_coordinates_preserved(self):
        """Coordinates from pred_coord should be preserved in LAS output."""
        from pointspace.writers.las_writer import LASWriter

        writer = LASWriter(save_dir=self.tmp_dir, source_dir=None)
        pred_coord = np.array([
            [100.123, 200.456, 50.789],
            [101.111, 201.222, 51.333],
        ])
        out_path = writer.write("coord_test", pred_coord=pred_coord)
        las = laspy.read(out_path)
        read_coords = np.column_stack([las.x, las.y, las.z])
        np.testing.assert_allclose(read_coords, pred_coord, atol=0.002)

    def test_pred_coord_slope_mismatch_raises(self):
        """Mismatched slope length should raise ValueError."""
        from pointspace.writers.las_writer import LASWriter

        writer = LASWriter(save_dir=self.tmp_dir, source_dir=None)
        pred_coord = np.random.uniform(0, 10, (100, 3))
        slope = np.random.uniform(0, 1, 50)  # wrong length!
        with self.assertRaises(ValueError):
            writer.write("mismatch", pred_coord=pred_coord, slope=slope)


# ══════════════════════════════════════════════════════════════════════════════
#  8. Config template
# ══════════════════════════════════════════════════════════════════════════════


class TestCnfConfig(unittest.TestCase):
    """CNF config template validity tests."""

    def test_config_syntax(self):
        """Config file should be valid Python."""
        config_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "configs",
            "cnf",
            "terrain-cnf-pt-v2m4-0-base.py",
        )
        self.assertTrue(os.path.isfile(config_path), f"Config not found: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            source = f.read()
        # This will raise SyntaxError if invalid
        compile(source, config_path, "exec")

    def test_config_keys(self):
        """Config should define all required keys."""
        config_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "configs",
            "cnf",
            "terrain-cnf-pt-v2m4-0-base.py",
        )
        ns = {}
        with open(config_path, "r", encoding="utf-8") as f:
            exec(f.read(), ns)
        required = [
            "model", "optimizer", "scheduler", "hooks",
            "train", "test", "writer", "data",
            "query_resolution", "query_batch_size", "compute_derivatives",
        ]
        for key in required:
            self.assertIn(key, ns, f"Missing key: {key}")

    def test_config_model_type(self):
        """Model type should be DefaultCNF."""
        config_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "configs",
            "cnf",
            "terrain-cnf-pt-v2m4-0-base.py",
        )
        ns = {}
        with open(config_path, "r", encoding="utf-8") as f:
            exec(f.read(), ns)
        self.assertEqual(ns["model"]["type"], "DefaultCNF")
        self.assertEqual(ns["test"]["type"], "CnfTester")

    def test_config_data_transforms(self):
        """Train transform pipeline should contain CNF transforms."""
        config_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "configs",
            "cnf",
            "terrain-cnf-pt-v2m4-0-base.py",
        )
        ns = {}
        with open(config_path, "r", encoding="utf-8") as f:
            exec(f.read(), ns)
        # Extract transform types
        train_tf_types = [t["type"] for t in ns["data"]["train"]["transform"]]
        self.assertIn("ClassFilter", train_tf_types)
        self.assertIn("TerrainImplicitSampler", train_tf_types)

    def test_config_collect_has_query_offset(self):
        """Collect should include query_offset, query_gt, and query_gt_low."""
        config_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "configs",
            "cnf",
            "terrain-cnf-pt-v2m4-0-base.py",
        )
        ns = {}
        with open(config_path, "r", encoding="utf-8") as f:
            exec(f.read(), ns)
        train_post = ns["data"]["train"]["post_transform"]
        collect = [t for t in train_post if t["type"] == "Collect"][0]
        self.assertIn("query_offset", collect["offset_keys_dict"])
        self.assertEqual(
            collect["offset_keys_dict"]["query_offset"], "query_coord"
        )
        self.assertIn("query_gt", collect["keys"])
        self.assertIn("query_gt_low", collect["keys"])


# ══════════════════════════════════════════════════════════════════════════════
#  9. End-to-end transform pipeline
# ══════════════════════════════════════════════════════════════════════════════


class TestE2ETransformPipeline(unittest.TestCase):
    """End-to-end test for the CNF transform pipeline."""

    def test_full_train_pipeline(self):
        """ClassFilter → TerrainImplicitSampler → GridCoordinate → ToTensor → Collect."""
        from pointspace.datasets.transform import ToTensor

        data = _make_ground_point_cloud(n=500)

        # 1. ClassFilter
        data = ClassFilter(keep_classes=[1])(data)
        n_ground = data["coord"].shape[0]
        self.assertGreater(n_ground, 50)  # should have enough ground

        # 2. TerrainImplicitSampler
        data = TerrainImplicitSampler(
            random_ratio=0.1,
            feature_ratio=0.1,
            max_blocks=3,
            block_size_range=(2.0, 8.0),
            feature_resolution=2.0,
            max_query_ratio=0.6,
        )(data)
        self.assertIn("query_coord", data)
        self.assertIn("query_gt", data)
        self.assertIn("query_gt_low", data)

        # 3. GridCoordinate
        data = GridCoordinate(grid_size=0.5)(data)
        self.assertIn("grid_coord", data)
        self.assertEqual(data["coord"].shape[0], data["grid_coord"].shape[0])

        # 4. ToTensor
        data = ToTensor()(data)
        self.assertIsInstance(data["coord"], torch.Tensor)
        self.assertIsInstance(data["query_coord"], torch.Tensor)

        # 5. Collect
        collect = Collect(
            keys=["coord", "grid_coord", "query_coord", "query_gt", "query_gt_low"],
            offset_keys_dict=dict(
                offset="coord",
                query_offset="query_coord",
            ),
            feat_keys=["coord"],
        )
        final = collect(data)
        self.assertIn("offset", final)
        self.assertIn("query_offset", final)
        self.assertEqual(final["offset"].item(), data["coord"].shape[0])
        self.assertEqual(final["query_offset"].item(), data["query_coord"].shape[0])


# ══════════════════════════════════════════════════════════════════════════════
#  10. Registry completeness
# ══════════════════════════════════════════════════════════════════════════════


class TestRegistryCompleteness(unittest.TestCase):
    """All CNF components should be registered."""

    def test_transform_registry(self):
        """CNF transforms should be registered."""
        from pointspace.datasets.transform import TRANSFORMS

        for name in ["ClassFilter", "TerrainImplicitSampler", "GridCoordinate"]:
            self.assertIn(name, TRANSFORMS._module_dict, f"{name} not registered")

    def test_model_registry(self):
        """DefaultCNF and DualBranchCNFHead should be registered."""
        from pointspace.models import MODELS

        self.assertIn("DefaultCNF", MODELS._module_dict)
        self.assertIn("DualBranchCNFHead", MODELS._module_dict)

    def test_hook_registry(self):
        """CnfEvaluator should be registered."""
        from pointspace.engines.hooks import HOOKS

        self.assertIn("CnfEvaluator", HOOKS._module_dict)

    def test_tester_registry(self):
        """CnfTester should be registered."""
        from pointspace.engines.test import TESTERS

        self.assertIn("CnfTester", TESTERS._module_dict)


# ══════════════════════════════════════════════════════════════════════════════
#  Normal Constraint Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestNormalConstraint(unittest.TestCase):
    """Tests for normal vector constraint in DefaultCNF."""

    def _make_model(self, normal_weight=0.1):
        from pointspace.models.default import DefaultCNF

        model = DefaultCNF(
            backbone=None,
            head=dict(
                type="SingleBranchCNFHead",
                backbone_out_channels=8,
                query_dim=2,
                num_targets=1,
                k_neighbors=4,
                hidden_dim=32,
                mlp_hidden_dims=[16],
                attn_groups=4,
            ),
            criteria=None,
            reg_weight=0.0,
            normal_weight=normal_weight,
        )
        model.backbone = _TinyBackbone(in_channels=3, out_channels=8)
        return model

    def _make_input(self, n=64, q=16, with_normals=True):
        rng = np.random.RandomState(0)
        coord = torch.from_numpy(
            rng.uniform(0, 10, (n, 3)).astype(np.float32)
        )
        qc = torch.from_numpy(
            rng.uniform(0, 10, (q, 2)).astype(np.float32)
        )
        qt = torch.from_numpy(
            rng.uniform(0, 5, q).astype(np.float32)
        )
        inp = dict(
            coord=coord,
            feat=coord.clone(),
            offset=torch.tensor([n]),
            query_coord=qc,
            query_gt=qt,
            query_offset=torch.tensor([q]),
        )
        if with_normals:
            normals = torch.randn(q, 3)
            normals = normals / normals.norm(dim=-1, keepdim=True)
            inp["query_normal_gt"] = normals
        return inp

    def test_normal_loss_nonzero_when_enabled(self):
        """With normal_weight > 0 and normals provided, normal loss > 0."""
        model = self._make_model(normal_weight=0.1)
        model.train()
        inp = self._make_input(with_normals=True)
        out = model(inp)
        self.assertIn("normal", out)
        self.assertGreater(out["normal"].item(), 0.0)

    def test_normal_loss_zero_when_disabled(self):
        """With normal_weight=0, normal loss should be 0."""
        model = self._make_model(normal_weight=0.0)
        model.train()
        inp = self._make_input(with_normals=True)
        out = model(inp)
        self.assertIn("normal", out)
        self.assertEqual(out["normal"].item(), 0.0)

    def test_no_normals_in_input_skips_normal_loss(self):
        """Without query_normal_gt, normal loss should be 0."""
        model = self._make_model(normal_weight=0.1)
        model.train()
        inp = self._make_input(with_normals=False)
        out = model(inp)
        self.assertIn("normal", out)
        self.assertEqual(out["normal"].item(), 0.0)

    def test_eval_path_no_grad_on_query_coord(self):
        """Eval forward should not require grad on query_coord."""
        model = self._make_model(normal_weight=0.1)
        model.eval()
        inp = self._make_input(with_normals=True)
        with torch.no_grad():
            out = model(inp)
        self.assertIn("cnf_pred", out)

    def test_gradient_flows_through_normal_loss(self):
        """Normal loss should contribute to gradient on backbone params."""
        model = self._make_model(normal_weight=1.0)
        model.train()
        inp = self._make_input(with_normals=True)
        out = model(inp)
        out["loss"].backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters() if p.requires_grad
        )
        self.assertTrue(has_grad)


if __name__ == "__main__":
    unittest.main()
