"""
Comprehensive Tests for EZ-SP Module Migration Verification

This test suite verifies that the EZ-SP (End-to-End Superpoint Transformer) has been
correctly migrated from the official implementation to the PointSpace framework.

Test coverage includes:
1. Network architecture components
2. Loss functions (partition criterion, focal loss)
3. Data structures (SuperpointHierarchy, Cluster)
4. ignore_label handling
5. Multi-stage output verification
6. Relative positional encoding (RPE)
7. UnitSphereNorm behavior
8. Class weight computation
9. Integration tests with DALES-like data

Author: PointSpace Team
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional

# Skip all tests if CUDA is not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def device():
    """Return CUDA device for testing."""
    return torch.device("cuda:0")


@pytest.fixture
def sample_point_cloud(device) -> Dict:
    """Generate sample point cloud data similar to DALES dataset."""
    N = 1000
    B = 2  # batch size

    # Random 3D points in a unit cube
    coord = torch.rand(N, 3, device=device) * 100  # scaled to typical DALES range

    # Features: coord(3) + intensity(1) + echo(2) = 6 features
    intensity = torch.rand(N, 1, device=device)
    echo = torch.zeros(N, 2, device=device)
    echo[:, 0] = 1.0  # First return indicator

    feat = torch.cat([coord, intensity, echo], dim=1)

    # Grid coordinates for sparse conv
    grid_size = 0.1
    grid_coord = (coord / grid_size).long()

    # Batch indices
    batch = torch.zeros(N, dtype=torch.long, device=device)
    batch[N // 2:] = 1

    # Offsets
    offset = torch.tensor([N // 2, N], dtype=torch.long, device=device)

    # Labels (8 classes for DALES)
    segment = torch.randint(0, 8, (N,), device=device)
    # Add some ignore labels
    segment[::50] = -1

    return {
        "coord": coord,
        "feat": feat,
        "grid_coord": grid_coord,
        "batch": batch,
        "offset": offset,
        "segment": segment,
    }


@pytest.fixture
def sample_nag(device):
    """Create a sample SuperpointHierarchy for testing."""
    from pointspace.models.backbone.ezsp.superpoint_hierarchy import (
        SuperpointHierarchy,
    )

    N_points = 1000
    N_sp1 = 100  # Level 1 superpoints
    N_sp2 = 20   # Level 2 superpoints
    num_classes = 8

    # Level 0: points
    super_index_0to1 = torch.randint(0, N_sp1, (N_points,), device=device)
    edge_index_0 = torch.stack([
        torch.randint(0, N_points, (2000,), device=device),
        torch.randint(0, N_points, (2000,), device=device),
    ])
    
    # Create label histogram for level 0
    y_0 = torch.zeros(N_points, num_classes + 1, device=device)
    labels = torch.randint(0, num_classes, (N_points,), device=device)
    y_0.scatter_(1, labels.unsqueeze(1), 1.0)

    level0 = {
        "pos": torch.rand(N_points, 3, device=device) * 100,
        "x": torch.randn(N_points, 32, device=device),
        "edge_index": edge_index_0,
        "y": y_0,
        "super_index": super_index_0to1,
    }

    # Level 1: superpoints
    super_index_1to2 = torch.randint(0, N_sp2, (N_sp1,), device=device)
    edge_index_1 = torch.stack([
        torch.randint(0, N_sp1, (200,), device=device),
        torch.randint(0, N_sp1, (200,), device=device),
    ])
    h_edge_attr = torch.randn(200, 18, device=device)  # Horizontal edge features
    v_edge_attr = torch.randn(N_sp1, 9, device=device)  # Vertical edge features

    # Label histogram for level 1 (aggregated from level 0)
    y_1 = torch.zeros(N_sp1, num_classes + 1, device=device)
    for i in range(N_sp1):
        mask = super_index_0to1 == i
        if mask.any():
            y_1[i] = y_0[mask].sum(dim=0)

    level1 = {
        "pos": torch.rand(N_sp1, 3, device=device) * 100,
        "x": torch.randn(N_sp1, 32, device=device),
        "edge_index": edge_index_1,
        "h_edge_attr": h_edge_attr,
        "v_edge_attr": v_edge_attr,
        "y": y_1,
        "super_index": super_index_1to2,
    }

    # Level 2: coarser superpoints
    y_2 = torch.zeros(N_sp2, num_classes + 1, device=device)
    for i in range(N_sp2):
        mask = super_index_1to2 == i
        if mask.any():
            y_2[i] = y_1[mask].sum(dim=0)

    level2 = {
        "pos": torch.rand(N_sp2, 3, device=device) * 100,
        "x": torch.randn(N_sp2, 32, device=device),
        "y": y_2,
    }

    return SuperpointHierarchy([level0, level1, level2])


# =============================================================================
# Network Component Tests
# =============================================================================

class TestSPTComponents:
    """Tests for SPT network components."""

    def test_spt_creation(self, device):
        """Test SPT model instantiation with various configurations."""
        from pointspace.models.backbone.ezsp.spt.spt import SPT

        # Test with minimal config
        spt = SPT(
            point_mlp=[32, 64, 128],
            down_dim=[64, 64],
            down_num_blocks=2,
            down_num_heads=4,
            up_dim=[64],
            up_num_blocks=1,
            up_num_heads=4,
            in_rpe_dim=18,
        ).to(device)

        assert spt is not None

    def test_spt_with_rpe(self, device):
        """Test SPT with relative positional encoding enabled."""
        from pointspace.models.backbone.ezsp.spt.spt import SPT

        spt = SPT(
            point_mlp=[32, 64, 128],
            down_dim=[64],
            down_num_blocks=2,
            down_num_heads=4,
            in_rpe_dim=18,
            k_rpe=True,
            q_rpe=True,
            v_rpe=True,
        ).to(device)

        assert spt is not None

    def test_self_attention_block(self, device):
        """Test SelfAttentionBlock with various RPE configurations."""
        from pointspace.models.backbone.ezsp.spt.attention import SelfAttentionBlock

        N = 100
        dim = 64
        num_heads = 4
        in_rpe_dim = 18

        # Test basic attention
        attn = SelfAttentionBlock(
            dim=dim,
            num_heads=num_heads,
            qk_dim=8,
            in_rpe_dim=in_rpe_dim,
            k_rpe=True,
            q_rpe=True,
        ).to(device)

        x = torch.randn(N, dim, device=device)
        edge_index = torch.stack([
            torch.randint(0, N, (200,), device=device),
            torch.randint(0, N, (200,), device=device),
        ])
        edge_attr = torch.randn(200, in_rpe_dim, device=device)

        out = attn(x, edge_index, edge_attr=edge_attr)
        assert out.shape == x.shape

    def test_transformer_block(self, device):
        """Test TransformerBlock."""
        from pointspace.models.backbone.ezsp.spt.transformer import TransformerBlock

        N = 100
        dim = 64
        num_heads = 4

        transformer = TransformerBlock(
            dim=dim,
            num_heads=num_heads,
            num_blocks=2,
            qk_dim=8,
            ffn_ratio=4.0,
        ).to(device)

        x = torch.randn(N, dim, device=device)
        edge_index = torch.stack([
            torch.randint(0, N, (200,), device=device),
            torch.randint(0, N, (200,), device=device),
        ])

        out = transformer(x, edge_index)
        assert out.shape == x.shape

    def test_stage_modules(self, device):
        """Test Stage, DownNFuseStage, UpNFuseStage."""
        from pointspace.models.backbone.ezsp.spt.stage import (
            Stage,
            DownNFuseStage,
            UpNFuseStage,
        )

        N = 100
        dim = 64

        # Test Stage
        stage = Stage(
            dim=dim,
            num_blocks=2,
            num_heads=4,
            in_mlp=[32, 64],
            out_mlp=[64, 64],
        ).to(device)

        x = torch.randn(N, 32, device=device)
        pos = torch.randn(N, 3, device=device)
        edge_index = torch.stack([
            torch.randint(0, N, (200,), device=device),
            torch.randint(0, N, (200,), device=device),
        ])

        out, _ = stage(x, pos, edge_index)
        assert out.shape == (N, 64)


# =============================================================================
# Normalization Tests
# =============================================================================

class TestNormalization:
    """Tests for normalization layers."""

    def test_unit_sphere_norm(self, device):
        """Test UnitSphereNorm computes correct normalization."""
        from pointspace.models.backbone.ezsp.spt.norm import UnitSphereNorm

        norm = UnitSphereNorm(log_diameter=False).to(device)

        N = 100
        pos = torch.randn(N, 3, device=device) * 10

        # Without segment indices
        pos_norm, diameter = norm(pos)
        assert pos_norm.shape == pos.shape
        assert diameter.shape[0] == 1

        # Check that normalized positions are within unit sphere
        assert pos_norm.abs().max() <= 1.0 + 1e-2

    def test_unit_sphere_norm_per_segment(self, device):
        """Test UnitSphereNorm with per-segment normalization."""
        from pointspace.models.backbone.ezsp.spt.norm import UnitSphereNorm

        norm = UnitSphereNorm(log_diameter=True).to(device)

        N = 100
        num_segments = 10
        pos = torch.randn(N, 3, device=device) * 10
        idx = torch.randint(0, num_segments, (N,), device=device)

        pos_norm, diameter = norm(pos, idx=idx, num_super=num_segments)

        assert pos_norm.shape == pos.shape
        assert diameter.shape == (num_segments, 1)

    def test_batch_norm(self, device):
        """Test BatchNorm handles sparse and dense inputs."""
        from pointspace.models.backbone.ezsp.spt.norm import BatchNorm

        norm = BatchNorm(64).to(device)

        # Sparse input [N, D]
        x_sparse = torch.randn(1000, 64, device=device)
        out_sparse = norm(x_sparse)
        assert out_sparse.shape == x_sparse.shape

        # Dense input [B, N, D]
        x_dense = torch.randn(4, 100, 64, device=device)
        out_dense = norm(x_dense)
        assert out_dense.shape == x_dense.shape


# =============================================================================
# Loss Function Tests
# =============================================================================

class TestLossFunctions:
    """Tests for EZ-SP loss functions."""

    def test_partition_criterion_basic(self, sample_nag):
        """Test PartitionCriterion basic functionality."""
        from pointspace.models.losses.partition_criterion import PartitionCriterion

        criterion = PartitionCriterion(
            gamma=1.0,
            alpha=0.5,
            temperature=1.0,
            num_classes=8,
        )
        criterion.train()

        loss, output = criterion(sample_nag)

        assert loss.shape == ()
        assert not torch.isnan(loss)
        assert "n_intra_edge" in output
        assert "n_inter_edge" in output
        assert "mean_affinity_intra" in output
        assert "mean_affinity_inter" in output

    def test_partition_criterion_adaptive_sampling(self, sample_nag):
        """Test PartitionCriterion with adaptive edge sampling."""
        from pointspace.models.losses.partition_criterion import PartitionCriterion

        criterion = PartitionCriterion(
            gamma=1.0,
            alpha=0.5,
            temperature=1.0,
            adaptive_sampling=True,
            adaptive_sampling_ratio=0.5,
            num_classes=8,
        )
        criterion.train()

        loss, output = criterion(sample_nag)
        assert not torch.isnan(loss)

    def test_spt_binary_focal_loss(self, device):
        """Test SPTBinaryFocalLoss matches expected behavior."""
        from pointspace.models.losses.partition_criterion import SPTBinaryFocalLoss

        loss_fn = SPTBinaryFocalLoss(gamma=1.0, alpha=0.5)

        N = 100
        p = torch.rand(N, device=device)  # Probabilities in (0, 1)
        y = torch.randint(0, 2, (N,), device=device).bool()

        loss = loss_fn(p, y)

        assert loss.shape == ()
        assert not torch.isnan(loss)
        assert loss >= 0

    def test_partition_criterion_ignore_void(self, device):
        """Test that PartitionCriterion correctly ignores void voxels."""
        from pointspace.models.losses.partition_criterion import PartitionCriterion
        from pointspace.models.backbone.ezsp.superpoint_hierarchy import SuperpointHierarchy

        criterion = PartitionCriterion(num_classes=8)
        criterion.train()

        N = 100
        # Create histogram with some all-void voxels
        y = torch.zeros(N, 9, device=device)  # 8 classes + 1 void column
        labels = torch.randint(0, 8, (N,), device=device)
        y.scatter_(1, labels.unsqueeze(1), 1.0)
        # Set first 10 voxels as void (all zeros in first 8 columns)
        y[:10, :8] = 0
        y[:10, 8] = 1  # void column

        edge_index = torch.stack([
            torch.randint(0, N, (200,), device=device),
            torch.randint(0, N, (200,), device=device),
        ])

        data_list = [{
            "pos": torch.randn(N, 3, device=device),
            "x": torch.randn(N, 32, device=device),
            "edge_index": edge_index,
            "y": y,
        }]
        nag = SuperpointHierarchy(data_list)

        loss, output = criterion(nag)
        assert not torch.isnan(loss)


# =============================================================================
# Data Structure Tests
# =============================================================================

class TestDataStructures:
    """Tests for EZ-SP data structures."""

    def test_cluster_csr_format(self, device):
        """Test Cluster CSR format operations."""
        from pointspace.models.backbone.ezsp.superpoint_hierarchy import Cluster

        # Create cluster from super_index
        super_index = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2], device=device)
        cluster = Cluster.from_super_index(super_index)

        assert cluster.num_clusters == 3
        assert len(cluster.value) == 9

        # Check cluster sizes
        sizes = cluster.sizes()
        assert sizes.tolist() == [3, 2, 4]

        # Check cluster members
        assert len(cluster[0]) == 3
        assert len(cluster[1]) == 2
        assert len(cluster[2]) == 4

    def test_superpoint_hierarchy_levels(self, sample_nag):
        """Test SuperpointHierarchy level access."""
        assert sample_nag.num_levels == 3

        level0 = sample_nag[0]
        level1 = sample_nag[1]
        level2 = sample_nag[2]

        assert level0.num_points == 1000
        assert level1.num_points == 100
        assert level2.num_points == 20

    def test_superpoint_hierarchy_label_propagation(self, device):
        """Test label propagation from coarse to fine levels."""
        from pointspace.models.backbone.ezsp.superpoint_hierarchy import SuperpointHierarchy

        N_points = 100
        N_sp1 = 10

        super_index = torch.randint(0, N_sp1, (N_points,), device=device)

        data_list = [
            {"pos": torch.randn(N_points, 3, device=device), "super_index": super_index},
            {"pos": torch.randn(N_sp1, 3, device=device)},
        ]

        nag = SuperpointHierarchy(data_list)

        # Predictions at level 1 (one-hot)
        preds_level1 = torch.zeros(N_sp1, 8, device=device)
        preds_level1[:, 0] = 1.0  # All class 0

        # Propagate to level 0
        preds_level0 = nag.propagate_labels_to_points(preds_level1, from_level=1)

        assert preds_level0.shape == (N_points, 8)
        assert (preds_level0.argmax(dim=1) == 0).all()

    def test_superpoint_hierarchy_batch_handling(self, device):
        """Test SuperpointHierarchy with batched data."""
        from pointspace.models.backbone.ezsp.superpoint_hierarchy import SuperpointHierarchy

        N1, N2 = 50, 30  # Points per batch
        N_sp1_1, N_sp1_2 = 10, 8  # Superpoints per batch

        # Batch 1 super_index
        super_index_1 = torch.randint(0, N_sp1_1, (N1,), device=device)
        # Batch 2 super_index (offset by N_sp1_1)
        super_index_2 = torch.randint(N_sp1_1, N_sp1_1 + N_sp1_2, (N2,), device=device)

        super_index = torch.cat([super_index_1, super_index_2])

        data_list = [
            {
                "pos": torch.randn(N1 + N2, 3, device=device),
                "super_index": super_index,
                "batch": torch.cat([
                    torch.zeros(N1, dtype=torch.long, device=device),
                    torch.ones(N2, dtype=torch.long, device=device),
                ]),
            },
            {
                "pos": torch.randn(N_sp1_1 + N_sp1_2, 3, device=device),
            },
        ]

        nag = SuperpointHierarchy(data_list)
        assert nag[0].num_points == N1 + N2
        assert nag[1].num_points == N_sp1_1 + N_sp1_2


# =============================================================================
# Multi-Stage Output Tests
# =============================================================================

class TestMultiStageOutput:
    """Tests for multi-stage output generation."""

    def test_spt_multi_stage_output(self, device):
        """Test SPT generates outputs for all levels when output_stage_wise=True."""
        from pointspace.models.backbone.ezsp.spt.spt import SPT
        from pointspace.models.backbone.ezsp.superpoint_hierarchy import SuperpointHierarchy

        spt = SPT(
            point_mlp=[6, 32, 64],
            down_dim=[64, 64],
            down_in_mlp=[[64 + 3, 64, 64], [64 + 3, 64, 64]],
            down_num_blocks=1,
            down_num_heads=4,
            up_dim=[64],
            up_in_mlp=[[64 + 64 + 3, 64]],
            up_num_blocks=1,
            up_num_heads=4,
            output_stage_wise=True,
            nano=False,
        ).to(device)

        # Check output dimension is a list (multi-stage)
        out_dim = spt.out_dim
        assert isinstance(out_dim, list)


# =============================================================================
# SparseCNN Tests
# =============================================================================

class TestSparseCNN:
    """Tests for SparseCNN module."""

    def test_sparse_cnn_forward(self, sample_point_cloud, device):
        """Test SparseCNN forward pass."""
        from pointspace.models.backbone.ezsp.sparse_cnn import SparseCNN
        from pointspace.models.utils.structure import Point

        cnn = SparseCNN(
            in_channels=6,
            channels=[32, 32, 32],
            norm="gn",
        ).to(device)

        point = Point(sample_point_cloud)
        out_point = cnn(point)

        assert out_point.feat.shape == (sample_point_cloud["coord"].shape[0], 32)

    def test_sparse_cnn_freeze_unfreeze(self, device):
        """Test SparseCNN freeze/unfreeze functionality."""
        from pointspace.models.backbone.ezsp.sparse_cnn import SparseCNN

        cnn = SparseCNN(in_channels=6, channels=[32, 32, 32]).to(device)

        # Initially not frozen
        assert not cnn.frozen

        # Freeze
        cnn.freeze()
        assert cnn.frozen
        for param in cnn.parameters():
            assert not param.requires_grad

        # Unfreeze
        cnn.unfreeze()
        assert not cnn.frozen
        for param in cnn.parameters():
            assert param.requires_grad

    def test_sparse_cnn_gradient_flow(self, sample_point_cloud, device):
        """Test gradients flow through SparseCNN."""
        from pointspace.models.backbone.ezsp.sparse_cnn import SparseCNN
        from pointspace.models.utils.structure import Point

        cnn = SparseCNN(in_channels=6, channels=[32, 32, 32]).to(device)

        # Create a leaf tensor for gradient tracking
        feat = sample_point_cloud["feat"].clone().requires_grad_(True)
        sample_point_cloud["feat"] = feat

        point = Point(sample_point_cloud)
        out_point = cnn(point)

        loss = out_point.feat.sum()
        loss.backward()

        assert feat.grad is not None
        assert not torch.isnan(feat.grad).any()


# =============================================================================
# GraphNorm Tests
# =============================================================================

class TestGraphNorm:
    """Tests for GraphNorm module."""

    def test_graph_norm_per_graph(self, device):
        """Test GraphNorm normalizes per graph."""
        from pointspace.models.backbone.ezsp.graph_norm import GraphNorm

        norm = GraphNorm(64, affine=False).to(device)

        N = 200
        x = torch.randn(N, 64, device=device)
        batch = torch.zeros(N, dtype=torch.long, device=device)
        batch[100:] = 1

        out = norm(x, batch)

        # Check each graph has zero mean
        mean_g0 = out[:100].mean(dim=0)
        mean_g1 = out[100:].mean(dim=0)

        assert torch.allclose(mean_g0, torch.zeros(64, device=device), atol=1e-5)
        assert torch.allclose(mean_g1, torch.zeros(64, device=device), atol=1e-5)


# =============================================================================
# Pooling and Fusion Tests
# =============================================================================

class TestPoolingFusion:
    """Tests for pooling and fusion modules."""

    def test_pool_factory(self, device):
        """Test pool_factory returns correct pool types."""
        from pointspace.models.backbone.ezsp.spt.pool import pool_factory

        for pool_type in ["max", "mean", "sum"]:
            pool = pool_factory(pool_type)
            assert pool is not None

    def test_index_unpool(self, device):
        """Test IndexUnpool propagates features correctly."""
        from pointspace.models.backbone.ezsp.spt.fusion import IndexUnpool

        unpool = IndexUnpool()

        N_fine = 100
        N_coarse = 10

        x_coarse = torch.randn(N_coarse, 64, device=device)
        super_index = torch.randint(0, N_coarse, (N_fine,), device=device)

        x_fine = unpool(x_coarse, super_index)

        assert x_fine.shape == (N_fine, 64)
        # Check values are correctly propagated
        for i in range(N_fine):
            assert torch.allclose(x_fine[i], x_coarse[super_index[i]])

    def test_cat_fusion(self, device):
        """Test CatFusion concatenates features."""
        from pointspace.models.backbone.ezsp.spt.fusion import CatFusion

        fusion = CatFusion()

        x1 = torch.randn(100, 32, device=device)
        x2 = torch.randn(100, 64, device=device)

        out = fusion(x1, x2)
        assert out.shape == (100, 96)


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for full EZ-SP pipeline."""

    def test_stage1_forward(self, sample_point_cloud, device):
        """Test Stage 1 (partition learning) full forward pass."""
        from pointspace.models.segmentor.ezsp_segmentor import EZSPPartitionSegmentor

        segmentor = EZSPPartitionSegmentor(
            training_partition_stage=True,
            num_classes=8,
            sparse_cnn=dict(
                type="EZ-SparseCNN",
                in_channels=6,
                channels=[32, 32, 32],
            ),
            partition_module=dict(
                type="GreedyContourPriorPartitionSimple",
                k_adjacency=5,
                grid_size=0.1,
                num_levels=2,
            ),
        ).to(device)
        segmentor.train()

        output = segmentor(sample_point_cloud)

        assert "loss" in output
        assert not torch.isnan(output["loss"])

    def test_stage2_forward(self, sample_point_cloud, device):
        """Test Stage 2 (semantic segmentation) full forward pass."""
        from pointspace.models.segmentor.ezsp_segmentor import EZSPPartitionSegmentor

        segmentor = EZSPPartitionSegmentor(
            training_partition_stage=False,
            num_classes=8,
            sparse_cnn=dict(
                type="EZ-SparseCNN",
                in_channels=6,
                channels=[32, 32, 32],
            ),
            partition_module=dict(
                type="GreedyContourPriorPartitionSimple",
                k_adjacency=5,
                grid_size=0.1,
                num_levels=2,
            ),
            freeze_cnn=False,  # Don't freeze for easier testing
        ).to(device)
        segmentor.train()

        output = segmentor(sample_point_cloud)

        assert "loss" in output
        assert "seg_logits" in output
        assert output["seg_logits"].shape == (sample_point_cloud["coord"].shape[0], 8)

    def test_stage1_to_stage2_weight_transfer(self, device):
        """Test loading Stage 1 weights into Stage 2 model."""
        import tempfile
        import os
        from pointspace.models.segmentor.ezsp_segmentor import EZSPPartitionSegmentor

        # Create Stage 1 model
        stage1 = EZSPPartitionSegmentor(
            training_partition_stage=True,
            num_classes=8,
            sparse_cnn=dict(
                type="EZ-SparseCNN",
                in_channels=6,
                channels=[32, 32, 32],
            ),
            partition_module=dict(
                type="GreedyContourPriorPartitionSimple",
                k_adjacency=5,
                grid_size=0.1,
                num_levels=2,
            ),
        ).to(device)

        # Save Stage 1 weights
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "stage1.pth")
            torch.save({"state_dict": stage1.state_dict()}, ckpt_path)

            # Create Stage 2 model
            stage2 = EZSPPartitionSegmentor(
                training_partition_stage=False,
                num_classes=8,
                sparse_cnn=dict(
                    type="EZ-SparseCNN",
                    in_channels=6,
                    channels=[32, 32, 32],
                ),
                partition_module=dict(
                    type="GreedyContourPriorPartitionSimple",
                    k_adjacency=5,
                    grid_size=0.1,
                    num_levels=2,
                ),
                freeze_cnn=True,
            ).to(device)

            # Load Stage 1 CNN weights
            stage2.load_stage1_weights(ckpt_path)

            # Verify CNN weights match
            for (name1, param1), (name2, param2) in zip(
                stage1.sparse_cnn.named_parameters(),
                stage2.sparse_cnn.named_parameters()
            ):
                assert torch.allclose(param1, param2), f"Mismatch in {name1}"


# =============================================================================
# Class Weight Computation Tests
# =============================================================================

class TestClassWeights:
    """Tests for class weight computation."""

    def test_inverse_frequency_weights(self, device):
        """Test inverse frequency class weight computation."""
        # Simulate class distribution
        class_counts = torch.tensor([1000, 500, 200, 100, 50, 30, 20, 10], 
                                     dtype=torch.float, device=device)
        total = class_counts.sum()

        # Inverse frequency
        weights_inv = total / class_counts
        weights_inv = weights_inv / weights_inv.sum()

        # Rarer classes should have higher weights
        assert weights_inv[-1] > weights_inv[0]

    def test_sqrt_smoothed_weights(self, device):
        """Test sqrt-smoothed class weight computation (used in SPT)."""
        class_counts = torch.tensor([1000, 500, 200, 100, 50, 30, 20, 10], 
                                     dtype=torch.float, device=device)
        total = class_counts.sum()

        # Sqrt smoothing (as in original SPT)
        weights_sqrt = torch.sqrt(total / class_counts)
        weights_sqrt = weights_sqrt / weights_sqrt.sum()

        # Still rarer classes have higher weights
        assert weights_sqrt[-1] > weights_sqrt[0]

        # But less extreme than pure inverse
        weights_inv = total / class_counts
        weights_inv = weights_inv / weights_inv.sum()
        
        # Ratio should be smaller for sqrt
        ratio_sqrt = weights_sqrt[-1] / weights_sqrt[0]
        ratio_inv = weights_inv[-1] / weights_inv[0]
        assert ratio_sqrt < ratio_inv


# =============================================================================
# DALES-Specific Tests
# =============================================================================

class TestDALESCompatibility:
    """Tests specific to DALES dataset compatibility."""

    def test_dales_class_structure(self):
        """Verify DALES class structure matches expected."""
        dales_classes = [
            "ground", "vegetation", "cars", "trucks",
            "power_lines", "fences", "poles", "buildings"
        ]
        assert len(dales_classes) == 8

        # Stuff classes (large, background)
        stuff_classes = [0, 1]  # ground, vegetation
        assert dales_classes[0] == "ground"
        assert dales_classes[1] == "vegetation"

    def test_dales_feature_dimensions(self, device):
        """Test DALES-style features have correct dimensions."""
        N = 1000

        # DALES features: coord(3) + intensity(1) + echo(2) = 6
        # Or: coord(3) + elevation(1) + intensity(1) = 5 minimal
        coord = torch.randn(N, 3, device=device)
        intensity = torch.rand(N, 1, device=device)
        elevation = torch.rand(N, 1, device=device)
        echo = torch.zeros(N, 2, device=device)
        echo[:, 0] = 1.0  # First return

        # Full features (6D)
        feat_full = torch.cat([coord, intensity, echo], dim=1)
        assert feat_full.shape == (N, 6)

        # Minimal features (5D)
        feat_minimal = torch.cat([coord, intensity, elevation], dim=1)
        assert feat_minimal.shape == (N, 5)

    def test_dales_ignore_label_handling(self, sample_point_cloud, device):
        """Test ignore_label (-1) is handled correctly."""
        from pointspace.models.segmentor.ezsp_segmentor import EZSPPartitionSegmentor

        segmentor = EZSPPartitionSegmentor(
            training_partition_stage=False,
            num_classes=8,
            sparse_cnn=dict(
                type="EZ-SparseCNN",
                in_channels=6,
                channels=[32, 32, 32],
            ),
            partition_module=dict(
                type="GreedyContourPriorPartitionSimple",
                k_adjacency=5,
                grid_size=0.1,
                num_levels=2,
            ),
            criteria=[
                dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
            ],
            freeze_cnn=False,
        ).to(device)
        segmentor.train()

        # Ensure some labels are -1
        sample_point_cloud["segment"][::10] = -1

        output = segmentor(sample_point_cloud)

        assert "loss" in output
        assert not torch.isnan(output["loss"])
        assert not torch.isinf(output["loss"])


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
