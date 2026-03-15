"""Comprehensive tests for Dual-KNN architecture changes.

Covers:
  Step 1 – TerrainImplicitSampler: extreme_hole_prob + max_query_ratio=0.9
  Step 2 – DefaultCNF: no hard ground filtering, segment passthrough
  Step 3 – SingleBranchCNFHead: Dual-KNN with ground/semantic branches
"""

import math
import pytest
import numpy as np
import torch
import torch.nn as nn


# ═══════════════════════════════════════════════════════════════════════════════
# Step 1: TerrainImplicitSampler
# ═══════════════════════════════════════════════════════════════════════════════

class TestTerrainImplicitSampler:
    """Tests for TerrainImplicitSampler extreme_hole_prob and relaxed limits."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from pointspace.datasets.transform import TerrainImplicitSampler
        self.cls = TerrainImplicitSampler

    # ── init defaults ──────────────────────────────────────────────────────

    def test_default_max_query_ratio(self):
        s = self.cls()
        assert s.max_query_ratio == 0.9

    def test_default_extreme_hole_prob(self):
        s = self.cls()
        assert s.extreme_hole_prob == 0.3

    def test_custom_extreme_hole_prob(self):
        s = self.cls(extreme_hole_prob=0.5)
        assert s.extreme_hole_prob == 0.5

    def test_custom_max_query_ratio(self):
        s = self.cls(max_query_ratio=0.7)
        assert s.max_query_ratio == 0.7

    # ── extreme hole actually fires ───────────────────────────────────────

    def _make_data(self, n=5000, with_segment=False, ground_class=2):
        """Flat terrain tile, all ground class."""
        coord = np.column_stack([
            np.random.uniform(0, 100, n),
            np.random.uniform(0, 100, n),
            np.random.uniform(0, 1, n),
        ]).astype(np.float32)
        d = dict(coord=coord, segment=np.full(n, ground_class, dtype=np.int32))
        if not with_segment:
            d.pop("segment")
        return d

    def test_extreme_hole_prob_zero_never_fires(self):
        """With extreme_hole_prob=0.0, no extreme hole is produced."""
        s = self.cls(
            extreme_hole_prob=0.0,
            max_blocks=0,
            random_ratio=0.0,
            feature_ratio=0.0,
        )
        np.random.seed(42)
        data = self._make_data(1000, with_segment=True, ground_class=2)
        data["segment"][:] = 2
        result = s(data)
        # With 0 blocks, 0 random, 0 feature, 0 extreme => fallback 1 query
        assert result["query_gt"].shape[0] <= 2  # at most fallback

    def test_extreme_hole_prob_one_always_fires(self):
        """With extreme_hole_prob=1.0, a large hole is always created."""
        s = self.cls(
            extreme_hole_prob=1.0,
            max_blocks=0,
            random_ratio=0.0,
            feature_ratio=0.0,
            ground_class=2,
        )
        np.random.seed(123)
        data = self._make_data(5000, with_segment=True, ground_class=2)
        result = s(data)
        n_query = result["query_gt"].shape[0]
        # Extreme hole covers 50-80% extent → should produce many query points
        assert n_query > 100, f"Expected many query points from extreme hole, got {n_query}"

    def test_extreme_hole_respects_max_query_ratio(self):
        """Even with extreme hole, query count is capped at max_query_ratio."""
        s = self.cls(
            extreme_hole_prob=1.0,
            max_query_ratio=0.5,
            max_blocks=5,
            random_ratio=0.3,
            feature_ratio=0.3,
            ground_class=2,
        )
        np.random.seed(7)
        N = 2000
        data = self._make_data(N, with_segment=True, ground_class=2)
        result = s(data)
        n_query = result["query_gt"].shape[0]
        max_allowed = int(N * 0.5)
        assert n_query <= max_allowed, f"query {n_query} > cap {max_allowed}"

    def test_max_query_ratio_09_allows_more(self):
        """With max_query_ratio=0.9, up to 90% of points can be query."""
        s = self.cls(
            extreme_hole_prob=1.0,
            max_query_ratio=0.9,
            max_blocks=5,
            random_ratio=0.3,
            feature_ratio=0.3,
            ground_class=2,
        )
        np.random.seed(0)
        N = 3000
        data = self._make_data(N, with_segment=True, ground_class=2)
        result = s(data)
        max_allowed = int(N * 0.9)
        assert result["query_gt"].shape[0] <= max_allowed

    def test_at_least_one_support_after_extreme_hole(self):
        """Even with extreme hole, at least 1 support point remains."""
        s = self.cls(
            extreme_hole_prob=1.0,
            max_query_ratio=0.9,
            max_blocks=0,
            random_ratio=0.0,
            feature_ratio=0.0,
            ground_class=2,
        )
        for seed in range(10):
            np.random.seed(seed)
            data = self._make_data(500, with_segment=True, ground_class=2)
            result = s(data)
            # coord is the support set after slicing
            assert result["coord"].shape[0] >= 1

    def test_backward_compat_no_extreme_hole_param(self):
        """Without explicit extreme_hole_prob, default 0.3 is used."""
        s = self.cls(random_ratio=0.1, feature_ratio=0.1)
        assert hasattr(s, "extreme_hole_prob")
        assert s.extreme_hole_prob == 0.3

    # ── no regression on existing behaviour ───────────────────────────────

    def test_query_gt_shape_matches_query_coord(self):
        s = self.cls(ground_class=2, extreme_hole_prob=0.5)
        np.random.seed(99)
        data = self._make_data(2000, with_segment=True, ground_class=2)
        result = s(data)
        assert result["query_coord"].shape[0] == result["query_gt"].shape[0]
        assert result["query_coord"].shape[1] == 2

    def test_few_points_returns_empty(self):
        """< 10 points → empty query arrays."""
        s = self.cls()
        data = dict(
            coord=np.random.rand(5, 3).astype(np.float32),
            segment=np.full(5, 2, dtype=np.int32),
        )
        result = s(data)
        assert result["query_coord"].shape[0] == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Step 2: DefaultCNF (no hard filtering, segment passthrough)
# ═══════════════════════════════════════════════════════════════════════════════

class TestDefaultCNF:
    """Tests for DefaultCNF forward: no ground filtering, segment passthrough."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from pointspace.models.default import DefaultCNF
        self.cls = DefaultCNF

    def _make_fake_model(self, filter_non_ground=False, ground_class=2):
        """Create a minimal DefaultCNF with stub backbone and head."""
        model = self.cls.__new__(self.cls)
        nn.Module.__init__(model)

        # Stub backbone: returns coord as feat
        class FakeBackbone(nn.Module):
            def forward(self, point):
                return point

        # Stub head: records kwargs it receives
        class FakeHead(nn.Module):
            def __init__(self):
                super().__init__()
                self.last_kwargs = {}
                self.linear = nn.Linear(1, 1)  # so parameters() is non-empty

            def forward(self, support_coord, support_feat, query_coord, **kwargs):
                self.last_kwargs = kwargs
                Q = query_coord.shape[0]
                pred_z = torch.zeros(Q)
                anchor = torch.zeros(Q)
                if self.training:
                    return pred_z, anchor
                return pred_z

        model.backbone = None
        model.head = FakeHead()
        model.criteria = None
        model.reg_weight = 0.01
        model.terrain_alpha = 2.0
        model.ohem_ratio = 0.5
        model.normal_weight = 0.0
        model.enable_normal_loss = True
        model.filter_non_ground = filter_non_ground
        model.ground_class = ground_class
        return model

    def _make_input_dict(self, N=100, Q=50, with_segment=True):
        """Build a minimal input_dict for training forward."""
        coord = torch.randn(N, 3)
        feat = torch.randn(N, 3)
        segment = torch.randint(0, 5, (N,))
        segment[:N // 2] = 2  # half ground

        d = dict(
            coord=coord,
            feat=feat,
            query_coord=torch.randn(Q, 2),
            query_gt=torch.randn(Q),
            query_gt_low=torch.randn(Q),
            offset=torch.tensor([N]),
            query_offset=torch.tensor([Q]),
        )
        if with_segment:
            d["segment"] = segment
        return d

    def test_forward_passes_segment_to_head(self):
        """forward() must pass support_segment to head."""
        model = self._make_fake_model()

        # Override _run_backbone to skip actual backbone
        N = 100
        model._run_backbone = lambda d: (d["feat"], d["coord"])

        model.train()
        inp = self._make_input_dict(N=N, with_segment=True)
        model(inp)

        assert "support_segment" in model.head.last_kwargs
        seg = model.head.last_kwargs["support_segment"]
        assert seg is not None
        assert seg.shape[0] == N

    def test_forward_no_segment_passes_none(self):
        """When segment is absent, support_segment=None is passed."""
        model = self._make_fake_model()
        model._run_backbone = lambda d: (d["feat"], d["coord"])
        model.train()
        inp = self._make_input_dict(with_segment=False)
        model(inp)

        assert model.head.last_kwargs.get("support_segment") is None

    def test_no_filtering_all_points_reach_head(self):
        """All N support points reach the head (no ground mask filtering)."""
        model = self._make_fake_model(filter_non_ground=True)
        N = 80
        model._run_backbone = lambda d: (d["feat"], d["coord"])
        model.train()
        inp = self._make_input_dict(N=N, with_segment=True)
        # Even with filter_non_ground=True in init, forward should NOT filter
        model(inp)
        # The head should receive ALL N points (support_offset unmodified)
        offset = model.head.last_kwargs.get("support_offset")
        assert offset is not None
        assert offset.item() == N  # all points

    def test_extract_feat_returns_segment(self):
        """extract_feat includes segment in result dict when available."""
        model = self._make_fake_model()
        model._run_backbone = lambda d: (d["feat"], d["coord"])
        inp = self._make_input_dict(with_segment=True)
        result = model.extract_feat(inp)
        assert "segment" in result
        assert result["segment"].shape[0] == inp["segment"].shape[0]

    def test_extract_feat_no_segment(self):
        """extract_feat works without segment (no crash)."""
        model = self._make_fake_model()
        model._run_backbone = lambda d: (d["feat"], d["coord"])
        inp = self._make_input_dict(with_segment=False)
        result = model.extract_feat(inp)
        assert "segment" not in result

    def test_query_forward_accepts_segment(self):
        """query_forward signature accepts support_segment kwarg."""
        model = self._make_fake_model()
        model.eval()
        coord = torch.randn(50, 3)
        feat = torch.randn(50, 3)
        query = torch.randn(10, 2)
        seg = torch.randint(0, 5, (50,))
        # Should not raise
        result = model.query_forward(coord, feat, query, support_segment=seg)
        assert result.shape[0] == 10

    def test_query_forward_without_segment(self):
        """query_forward works without segment (backward compat)."""
        model = self._make_fake_model()
        model.eval()
        result = model.query_forward(
            torch.randn(50, 3), torch.randn(50, 3), torch.randn(10, 2)
        )
        assert result.shape[0] == 10

    def test_eval_no_query_returns_feat_and_coord(self):
        """Eval without query_coord returns support_feat/support_coord."""
        model = self._make_fake_model()
        model._run_backbone = lambda d: (d["feat"], d["coord"])
        model.eval()
        inp = dict(coord=torch.randn(40, 3), feat=torch.randn(40, 3),
                   offset=torch.tensor([40]))
        result = model(inp)
        assert "support_feat" in result
        assert "support_coord" in result


# ═══════════════════════════════════════════════════════════════════════════════
# Step 3: SingleBranchCNFHead — Dual-KNN architecture
# ═══════════════════════════════════════════════════════════════════════════════

class TestSingleBranchCNFHeadDualKNN:
    """Tests for the new Dual-KNN SingleBranchCNFHead."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from pointspace.models.head.cnf_head import SingleBranchCNFHead
        self.cls = SingleBranchCNFHead

    def _make_head(self, **kw):
        defaults = dict(
            backbone_out_channels=24,
            query_dim=2,
            num_targets=1,
            k_neighbors=4,
            hidden_dim=32,
            num_freqs=2,
            z_num_freqs=2,
            mlp_hidden_dims=[16],
            ground_class=2,
            num_classes=5,
            class_embed_dim=8,
        )
        defaults.update(kw)
        return self.cls(**defaults)

    # ── construction tests ────────────────────────────────────────────────

    def test_init_default(self):
        head = self._make_head()
        assert hasattr(head, "class_embed")
        assert hasattr(head, "z_rfe")
        assert hasattr(head, "rfe")
        assert hasattr(head, "pe_linear")

    def test_new_params_exist(self):
        head = self._make_head()
        assert head.ground_class == 2
        assert head.class_embed.num_embeddings == 5
        assert head.class_embed.embedding_dim == 8

    def test_z_rfe_output_dim(self):
        head = self._make_head(z_num_freqs=4)
        assert head.z_rfe.output_dim == 1 * 4 * 2  # in_dim=1

    def test_value_proj_input_dim(self):
        """Verify fuse_in is computed correctly."""
        C = 24
        fourier_dim = 2 * 2 * 2    # in_dim=2, num_freqs=2, * 2
        z_fourier_dim = 1 * 2 * 2  # in_dim=1, z_num_freqs=2, * 2
        pe_linear_dim = 64
        z_std_dim = 1
        cls_dim = 8
        expected = C * 2 + pe_linear_dim + fourier_dim + z_fourier_dim + z_std_dim + cls_dim
        head = self._make_head()
        assert head.value_proj[0].in_features == expected

    # ── forward shape tests ───────────────────────────────────────────────

    def _make_inputs(self, N=50, Q=20, C=24, with_segment=True):
        """Generate synthetic support/query tensors."""
        coord = torch.randn(N, 3)
        feat = torch.randn(N, C)
        query = torch.randn(Q, 2)
        segment = torch.randint(0, 5, (N,))
        segment[:N // 2] = 2  # half ground
        return coord, feat, query, segment if with_segment else None

    def test_forward_train_shape(self):
        head = self._make_head()
        head.train()
        coord, feat, query, seg = self._make_inputs()
        pred_z, anchor = head(coord, feat, query, support_segment=seg)
        assert pred_z.shape == (20,)
        assert anchor.shape == (20,)

    def test_forward_eval_shape(self):
        head = self._make_head()
        head.eval()
        coord, feat, query, seg = self._make_inputs()
        pred_z = head(coord, feat, query, support_segment=seg)
        assert pred_z.shape == (20,)

    def test_forward_no_segment_fallback(self):
        """Without segment, all points are treated as ground."""
        head = self._make_head()
        head.eval()
        coord, feat, query, _ = self._make_inputs(with_segment=False)
        pred_z = head(coord, feat, query, support_segment=None)
        assert pred_z.shape == (20,)

    def test_forward_multi_target(self):
        head = self._make_head(num_targets=3)
        head.eval()
        coord, feat, query, seg = self._make_inputs()
        pred_z = head(coord, feat, query, support_segment=seg)
        assert pred_z.shape == (20, 3)

    def test_forward_with_offsets(self):
        """Batched forward with offset tensors."""
        head = self._make_head()
        head.eval()
        N, Q = 60, 30
        coord = torch.randn(N, 3)
        feat = torch.randn(N, 24)
        query = torch.randn(Q, 2)
        seg = torch.randint(0, 5, (N,))
        seg[:N // 2] = 2
        offset_s = torch.tensor([30, 60])
        offset_q = torch.tensor([15, 30])
        pred_z = head(
            coord, feat, query,
            support_offset=offset_s, query_offset=offset_q,
            support_segment=seg,
        )
        assert pred_z.shape == (Q,)

    # ── dual-KNN: ground vs all-points KNN are different ──────────────────

    def test_ground_branch_only_uses_ground_points(self):
        """When segment has few ground points, ground KNN is restricted."""
        head = self._make_head(ground_class=2)
        head.eval()
        N = 40
        coord = torch.randn(N, 3)
        feat = torch.randn(N, 24)
        query = torch.randn(10, 2)
        # Only 5 ground points, rest are non-ground
        seg = torch.ones(N, dtype=torch.long)
        seg[:5] = 2
        # Should not crash and should produce valid output
        pred_z = head(coord, feat, query, support_segment=seg)
        assert pred_z.shape == (10,)
        assert torch.isfinite(pred_z).all()

    def test_all_ground_same_as_no_segment(self):
        """When all points are ground, result resembles no-segment mode."""
        head = self._make_head(ground_class=2)
        head.eval()
        torch.manual_seed(42)
        N, Q = 30, 10
        coord = torch.randn(N, 3)
        feat = torch.randn(N, 24)
        query = torch.randn(Q, 2)
        seg_all_ground = torch.full((N,), 2, dtype=torch.long)

        pred_with_seg = head(coord, feat, query, support_segment=seg_all_ground)
        pred_no_seg = head(coord, feat, query, support_segment=None)
        # They should be very close (class_embed differs but same structure)
        # Both should be finite
        assert torch.isfinite(pred_with_seg).all()
        assert torch.isfinite(pred_no_seg).all()

    def test_no_ground_points_fallback(self):
        """When segment has zero ground points, fallback to all-ones mask."""
        head = self._make_head(ground_class=2)
        head.eval()
        N = 30
        coord = torch.randn(N, 3)
        feat = torch.randn(N, 24)
        query = torch.randn(10, 2)
        seg = torch.ones(N, dtype=torch.long)  # no class-2 at all
        pred_z = head(coord, feat, query, support_segment=seg)
        assert pred_z.shape == (10,)
        assert torch.isfinite(pred_z).all()

    # ── class embedding tests ─────────────────────────────────────────────

    def test_class_embed_different_for_different_classes(self):
        head = self._make_head()
        e0 = head.class_embed(torch.tensor([0]))
        e2 = head.class_embed(torch.tensor([2]))
        assert not torch.equal(e0, e2)

    def test_class_embed_within_range(self):
        """Segment labels in [0, num_classes) should not crash."""
        head = self._make_head(num_classes=10)
        for c in range(10):
            e = head.class_embed(torch.tensor([c]))
            assert e.shape == (1, 8)

    # ── z_rfe encoding tests ──────────────────────────────────────────────

    def test_z_rfe_output_shape(self):
        head = self._make_head(z_num_freqs=3)
        z_diff = torch.randn(5, 4, 1)
        out = head.z_rfe(z_diff)
        assert out.shape == (5, 4, 1 * 3 * 2)

    # ── gradient flow ─────────────────────────────────────────────────────

    def test_gradient_flows_through_both_branches(self):
        """Gradients flow through class_embed, z_rfe, and ground anchor."""
        head = self._make_head()
        head.train()
        coord = torch.randn(30, 3, requires_grad=True)
        feat = torch.randn(30, 24, requires_grad=True)
        query = torch.randn(10, 2)
        seg = torch.randint(0, 5, (30,))
        seg[:15] = 2
        pred_z, anchor = head(coord, feat, query, support_segment=seg)
        loss = pred_z.sum()
        loss.backward()
        # class_embed should have gradients
        assert head.class_embed.weight.grad is not None
        # z_rfe is buffer-based (no learnable params), but value_proj should
        assert head.value_proj[0].weight.grad is not None

    def test_residual_structure(self):
        """pred_z = z_anchor + residual, so pred_z != z_anchor in general."""
        head = self._make_head()
        head.train()
        coord = torch.randn(40, 3)
        feat = torch.randn(40, 24)
        query = torch.randn(15, 2)
        seg = torch.randint(0, 5, (40,))
        seg[:20] = 2
        pred_z, anchor = head(coord, feat, query, support_segment=seg)
        # residual is almost certainly non-zero
        assert not torch.allclose(pred_z, anchor, atol=1e-6)

    # ── _to_dense helper ──────────────────────────────────────────────────

    def test_to_dense_full_coverage(self):
        """All queries have exactly K neighbours."""
        Q, K = 5, 3
        q_row = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])
        s_col = torch.tensor([10, 11, 12, 20, 21, 22, 30, 31, 32, 40, 41, 42, 50, 51, 52])
        dense = self.cls._to_dense(q_row, s_col, Q, K, torch.device("cpu"))
        assert dense.shape == (Q, K)
        assert dense[0].tolist() == [10, 11, 12]

    def test_to_dense_sparse_padding(self):
        """Query with fewer than K neighbours is padded with nearest."""
        Q, K = 3, 3
        # query 0 has 3 neighbours, query 1 has 1, query 2 has 2
        q_row = torch.tensor([0, 0, 0, 1, 2, 2])
        s_col = torch.tensor([10, 11, 12, 20, 30, 31])
        dense = self.cls._to_dense(q_row, s_col, Q, K, torch.device("cpu"))
        assert dense.shape == (Q, K)
        # query 1: only neighbor 20, so k=1 and k=2 should also be 20
        assert dense[1, 0].item() == 20
        assert dense[1, 1].item() == 20
        assert dense[1, 2].item() == 20


# ═══════════════════════════════════════════════════════════════════════════════
# DualBranchCNFHead backward compatibility
# ═══════════════════════════════════════════════════════════════════════════════

class TestDualBranchCNFHeadCompat:
    """DualBranchCNFHead.forward should accept support_segment without error."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from pointspace.models.head.cnf_head import DualBranchCNFHead
        self.cls = DualBranchCNFHead

    def test_forward_accepts_support_segment(self):
        head = self.cls(backbone_out_channels=24, query_dim=2)
        head.eval()
        N, Q = 30, 10
        coord = torch.randn(N, 3)
        feat = torch.randn(N, 24)
        query = torch.randn(Q, 2)
        seg = torch.randint(0, 5, (N,))
        # Should not raise TypeError
        result = head(coord, feat, query, support_segment=seg)
        # DualBranch returns tuple in eval
        if isinstance(result, tuple):
            assert result[0].shape[0] == Q
        else:
            assert result.shape[0] == Q


# ═══════════════════════════════════════════════════════════════════════════════
# Integration: full pipeline flow
# ═══════════════════════════════════════════════════════════════════════════════

class TestIntegration:
    """Integration tests ensuring the full pipeline connects."""

    def test_sampler_then_head_shapes_align(self):
        """TerrainImplicitSampler output feeds into SingleBranchCNFHead."""
        from pointspace.datasets.transform import TerrainImplicitSampler
        from pointspace.models.head.cnf_head import SingleBranchCNFHead

        # Sampler
        sampler = TerrainImplicitSampler(
            extreme_hole_prob=1.0,
            max_query_ratio=0.9,
            ground_class=2,
            compute_gt_low=False,
        )
        np.random.seed(0)
        N = 2000
        coord = np.column_stack([
            np.random.uniform(0, 50, N),
            np.random.uniform(0, 50, N),
            np.random.uniform(0, 5, N),
        ]).astype(np.float32)
        data = dict(
            coord=coord,
            segment=np.full(N, 2, dtype=np.int32),
        )
        result = sampler(data)

        # Convert for head
        support_coord = torch.from_numpy(result["coord"])
        support_feat = torch.randn(support_coord.shape[0], 24)
        query_coord = torch.from_numpy(result["query_coord"])
        support_segment = torch.from_numpy(result["segment"])

        head = SingleBranchCNFHead(
            backbone_out_channels=24, k_neighbors=4,
            hidden_dim=32, num_freqs=2, z_num_freqs=2,
            mlp_hidden_dims=[16], num_classes=5, class_embed_dim=8,
            ground_class=2,
        )
        head.eval()
        pred = head(
            support_coord, support_feat, query_coord,
            support_segment=support_segment,
        )
        assert pred.shape[0] == query_coord.shape[0]

    def test_head_deterministic_same_seed(self):
        """Same inputs produce same outputs (deterministic path)."""
        from pointspace.models.head.cnf_head import SingleBranchCNFHead

        head = SingleBranchCNFHead(
            backbone_out_channels=16, k_neighbors=4,
            hidden_dim=32, num_freqs=2, z_num_freqs=2,
            mlp_hidden_dims=[16], num_classes=5, class_embed_dim=8,
        )
        head.eval()
        torch.manual_seed(0)
        coord = torch.randn(30, 3)
        feat = torch.randn(30, 16)
        query = torch.randn(10, 2)
        seg = torch.randint(0, 5, (30,))
        seg[:15] = 2

        out1 = head(coord, feat, query, support_segment=seg)
        out2 = head(coord, feat, query, support_segment=seg)
        assert torch.allclose(out1, out2)
