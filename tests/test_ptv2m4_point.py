"""
Tests for PointTransformerV2 (PT-v2m4) with Point-based input/output.

Tests verify:
1. Model accepts Point input and returns Point
2. Model accepts data_dict and returns Point (backward compat)
3. Output Point has correct feat shape (N, dec_channels[0])
4. Model inherits from PointModule
5. Forward/backward pass correctness
6. Integration with DefaultSegmentorV2 (config-level)
7. No seg_head inside backbone

Requirements: CUDA, pointops, torch_geometric, torch_scatter
"""

import pytest
import torch
import torch.nn as nn

# Skip entire module if CUDA is not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for pointops"
)


def _lazy_imports():
    """Lazy import to avoid import errors when CUDA is unavailable."""
    from pointspace.models.utils.structure import Point
    from pointspace.models.modules import PointModule
    from pointspace.models.builder import MODELS

    return Point, PointModule, MODELS


def _build_small_model(in_channels=6, device="cuda"):
    """Build a small PTv2m5 model for testing (minimal config)."""
    _, _, MODELS = _lazy_imports()
    cfg = dict(
        type="PT-v2m4",
        in_channels=in_channels,
        patch_embed_depth=1,
        patch_embed_channels=32,
        patch_embed_groups=4,
        patch_embed_neighbours=4,
        enc_depths=(1, 1),
        enc_channels=(64, 128),
        enc_groups=(8, 16),
        enc_neighbours=(8, 8),
        dec_depths=(1, 1),
        dec_channels=(32, 64),
        dec_groups=(4, 8),
        dec_neighbours=(8, 8),
        grid_sizes=(0.06, 0.15),
        attn_qkv_bias=True,
        pe_multiplier=False,
        pe_bias=True,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        enable_checkpoint=False,
        unpool_backend="map",
    )
    from pointspace.models.builder import build_model

    model = build_model(cfg).to(device)
    return model


def _make_data(n_points=200, in_channels=6, batch_size=2, device="cuda"):
    """Create synthetic point cloud data."""
    Point, _, _ = _lazy_imports()

    coord = torch.randn(n_points, 3, device=device)
    feat = torch.randn(n_points, in_channels, device=device)
    # Split points roughly equally across batches
    pts_per_batch = n_points // batch_size
    offset = torch.tensor(
        [pts_per_batch * (i + 1) for i in range(batch_size)],
        dtype=torch.int32,
        device=device,
    )
    offset[-1] = n_points  # ensure last offset covers all points

    return coord, feat, offset


# ============================================================
# Test: Model registration
# ============================================================
class TestRegistration:
    def test_registered_as_ptv2m5(self):
        _, _, MODELS = _lazy_imports()
        assert "PT-v2m4" in MODELS.module_dict

    def test_inherits_point_module(self):
        _, PointModule, _ = _lazy_imports()
        model = _build_small_model()
        assert isinstance(model, PointModule)

    def test_no_seg_head(self):
        """Backbone should NOT contain seg_head (it belongs in segmentor)."""
        model = _build_small_model()
        assert not hasattr(model, "seg_head"), "seg_head should not be in backbone"

    def test_no_num_classes(self):
        """Backbone should NOT have num_classes attribute."""
        model = _build_small_model()
        assert not hasattr(model, "num_classes"), "num_classes belongs in segmentor"


# ============================================================
# Test: Forward with Point input
# ============================================================
class TestForwardPoint:
    def test_point_input_returns_point(self):
        Point, _, _ = _lazy_imports()
        model = _build_small_model()
        model.eval()
        coord, feat, offset = _make_data()
        point = Point(coord=coord, feat=feat, offset=offset)
        with torch.no_grad():
            out = model(point)
        assert isinstance(out, Point), f"Expected Point, got {type(out)}"

    def test_output_feat_shape(self):
        """Output feat should be [N, dec_channels[0]]."""
        Point, _, _ = _lazy_imports()
        model = _build_small_model()
        model.eval()
        n_points = 200
        coord, feat, offset = _make_data(n_points=n_points)
        point = Point(coord=coord, feat=feat, offset=offset)
        with torch.no_grad():
            out = model(point)
        # dec_channels[0] = 32 in our small config
        assert out.feat.shape == (n_points, 32), f"Expected (200, 32), got {out.feat.shape}"

    def test_output_preserves_coord(self):
        """Output Point should still have the original coord."""
        Point, _, _ = _lazy_imports()
        model = _build_small_model()
        model.eval()
        coord, feat, offset = _make_data()
        point = Point(coord=coord, feat=feat, offset=offset)
        with torch.no_grad():
            out = model(point)
        assert torch.equal(out.coord, coord)

    def test_output_preserves_offset(self):
        Point, _, _ = _lazy_imports()
        model = _build_small_model()
        model.eval()
        coord, feat, offset = _make_data()
        point = Point(coord=coord, feat=feat, offset=offset)
        with torch.no_grad():
            out = model(point)
        assert torch.equal(out.offset, offset)


# ============================================================
# Test: Forward with dict input (backward compat)
# ============================================================
class TestForwardDict:
    def test_dict_input_returns_point(self):
        Point, _, _ = _lazy_imports()
        model = _build_small_model()
        model.eval()
        coord, feat, offset = _make_data()
        data_dict = dict(coord=coord, feat=feat, offset=offset)
        with torch.no_grad():
            out = model(data_dict)
        assert isinstance(out, Point)

    def test_dict_input_feat_shape(self):
        Point, _, _ = _lazy_imports()
        model = _build_small_model()
        model.eval()
        n_points = 200
        coord, feat, offset = _make_data(n_points=n_points)
        data_dict = dict(coord=coord, feat=feat, offset=offset)
        with torch.no_grad():
            out = model(data_dict)
        assert out.feat.shape == (n_points, 32)


# ============================================================
# Test: Backward pass
# ============================================================
class TestBackward:
    def test_backward_pass(self):
        """forward + backward should not crash."""
        Point, _, _ = _lazy_imports()
        model = _build_small_model()
        model.train()
        coord, feat, offset = _make_data()
        point = Point(coord=coord, feat=feat, offset=offset)
        out = model(point)
        loss = out.feat.sum()
        loss.backward()
        # Check that at least some parameters have gradients
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        assert has_grad, "No parameter received gradients"


# ============================================================
# Test: Batch size variations
# ============================================================
class TestBatchSizes:
    def test_single_batch(self):
        Point, _, _ = _lazy_imports()
        model = _build_small_model()
        model.eval()
        coord, feat, offset = _make_data(n_points=100, batch_size=1)
        point = Point(coord=coord, feat=feat, offset=offset)
        with torch.no_grad():
            out = model(point)
        assert out.feat.shape[0] == 100

    def test_multi_batch(self):
        Point, _, _ = _lazy_imports()
        model = _build_small_model()
        model.eval()
        coord, feat, offset = _make_data(n_points=300, batch_size=3)
        point = Point(coord=coord, feat=feat, offset=offset)
        with torch.no_grad():
            out = model(point)
        assert out.feat.shape[0] == 300


# ============================================================
# Test: DefaultSegmentorV2 integration
# ============================================================
class TestSegmentorV2Integration:
    def test_segmentor_v2_forward(self):
        """Full pipeline: DefaultSegmentorV2 wraps PTv2m5 backbone."""
        Point, _, MODELS = _lazy_imports()
        from pointspace.models.builder import build_model

        num_classes = 20
        cfg = dict(
            type="DefaultSegmentorV2",
            num_classes=num_classes,
            backbone_out_channels=32,  # dec_channels[0] of our small model
            backbone=dict(
                type="PT-v2m4",
                in_channels=6,
                patch_embed_depth=1,
                patch_embed_channels=32,
                patch_embed_groups=4,
                patch_embed_neighbours=4,
                enc_depths=(1, 1),
                enc_channels=(64, 128),
                enc_groups=(8, 16),
                enc_neighbours=(8, 8),
                dec_depths=(1, 1),
                dec_channels=(32, 64),
                dec_groups=(4, 8),
                dec_neighbours=(8, 8),
                grid_sizes=(0.06, 0.15),
                attn_qkv_bias=True,
                pe_multiplier=False,
                pe_bias=True,
                attn_drop_rate=0.0,
                drop_path_rate=0.0,
                enable_checkpoint=False,
                unpool_backend="map",
            ),
            criteria=[dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1)],
        )
        segmentor = build_model(cfg).cuda()
        segmentor.eval()

        n_points = 200
        coord, feat, offset = _make_data(n_points=n_points, batch_size=2)
        data_dict = dict(coord=coord, feat=feat, offset=offset)
        with torch.no_grad():
            result = segmentor(data_dict)
        assert "seg_logits" in result
        assert result["seg_logits"].shape == (n_points, num_classes)

    def test_segmentor_v2_train(self):
        """Training mode: DefaultSegmentorV2 returns loss."""
        Point, _, MODELS = _lazy_imports()
        from pointspace.models.builder import build_model

        num_classes = 20
        cfg = dict(
            type="DefaultSegmentorV2",
            num_classes=num_classes,
            backbone_out_channels=32,
            backbone=dict(
                type="PT-v2m4",
                in_channels=6,
                patch_embed_depth=1,
                patch_embed_channels=32,
                patch_embed_groups=4,
                patch_embed_neighbours=4,
                enc_depths=(1, 1),
                enc_channels=(64, 128),
                enc_groups=(8, 16),
                enc_neighbours=(8, 8),
                dec_depths=(1, 1),
                dec_channels=(32, 64),
                dec_groups=(4, 8),
                dec_neighbours=(8, 8),
                grid_sizes=(0.06, 0.15),
                attn_qkv_bias=True,
                pe_multiplier=False,
                pe_bias=True,
                attn_drop_rate=0.0,
                drop_path_rate=0.0,
                enable_checkpoint=False,
                unpool_backend="map",
            ),
            criteria=[dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1)],
        )
        segmentor = build_model(cfg).cuda()
        segmentor.train()

        n_points = 200
        coord, feat, offset = _make_data(n_points=n_points, batch_size=2)
        segment = torch.randint(0, num_classes, (n_points,), device="cuda")
        data_dict = dict(coord=coord, feat=feat, offset=offset, segment=segment)
        result = segmentor(data_dict)
        assert "loss" in result
        assert result["loss"].requires_grad
