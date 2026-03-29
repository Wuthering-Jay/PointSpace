"""
Unit Tests for EZ-SP Module

Tests for:
- GraphNorm
- SparseCNN
- SuperpointHierarchy / Cluster
- GreedyContourPriorPartition
- PartitionCriterion
- EZSPPartitionSegmentor

Author: PointSpace Team
"""

import pytest
import torch
import torch.nn as nn

# Skip all tests if CUDA is not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


class TestGraphNorm:
    """Tests for GraphNorm layer"""

    def test_forward_shape(self):
        """Test output shape matches input shape"""
        from pointspace.models.backbone.ezsp.graph_norm import GraphNorm

        norm = GraphNorm(64)
        x = torch.randn(1000, 64)
        batch = torch.zeros(1000, dtype=torch.long)
        batch[500:] = 1  # Two graphs

        out = norm(x, batch)
        assert out.shape == x.shape

    def test_per_graph_normalization(self):
        """Test that normalization is done per-graph"""
        from pointspace.models.backbone.ezsp.graph_norm import GraphNorm

        norm = GraphNorm(64, affine=False)
        x = torch.randn(100, 64)
        batch = torch.zeros(100, dtype=torch.long)
        batch[50:] = 1

        out = norm(x, batch)

        # Check mean is approximately 0 for each graph
        mean_g0 = out[:50].mean(dim=0)
        mean_g1 = out[50:].mean(dim=0)

        assert torch.allclose(mean_g0, torch.zeros(64), atol=1e-5)
        assert torch.allclose(mean_g1, torch.zeros(64), atol=1e-5)

    def test_gradient_flow(self):
        """Test gradients flow through GraphNorm"""
        from pointspace.models.backbone.ezsp.graph_norm import GraphNorm

        norm = GraphNorm(64)
        x = torch.randn(100, 64, requires_grad=True)
        batch = torch.zeros(100, dtype=torch.long)

        out = norm(x, batch)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_affine_parameters(self):
        """Test learnable affine parameters"""
        from pointspace.models.backbone.ezsp.graph_norm import GraphNorm

        norm_affine = GraphNorm(64, affine=True)
        norm_no_affine = GraphNorm(64, affine=False)

        assert norm_affine.weight is not None
        assert norm_affine.bias is not None
        assert norm_no_affine.weight is None
        assert norm_no_affine.bias is None


class TestCluster:
    """Tests for Cluster data structure"""

    def test_from_super_index(self):
        """Test Cluster creation from super_index"""
        from pointspace.models.backbone.ezsp.superpoint_hierarchy import Cluster

        # 10 points, 3 clusters
        super_index = torch.tensor([0, 1, 0, 2, 1, 0, 2, 1, 0, 2])
        cluster = Cluster.from_super_index(super_index)

        assert cluster.num_clusters == 3
        assert len(cluster.value) == 10

    def test_cluster_membership(self):
        """Test getting cluster members"""
        from pointspace.models.backbone.ezsp.superpoint_hierarchy import Cluster

        super_index = torch.tensor([0, 0, 0, 1, 1, 2])
        cluster = Cluster.from_super_index(super_index)

        # Cluster 0 should have 3 members
        members_0 = cluster[0]
        assert len(members_0) == 3

        # Cluster 1 should have 2 members
        members_1 = cluster[1]
        assert len(members_1) == 2

        # Cluster 2 should have 1 member
        members_2 = cluster[2]
        assert len(members_2) == 1

    def test_cluster_sizes(self):
        """Test cluster sizes computation"""
        from pointspace.models.backbone.ezsp.superpoint_hierarchy import Cluster

        super_index = torch.tensor([0, 0, 0, 1, 1, 2])
        cluster = Cluster.from_super_index(super_index)

        sizes = cluster.sizes()
        assert sizes.tolist() == [3, 2, 1]


class TestSuperpointHierarchy:
    """Tests for SuperpointHierarchy"""

    def test_creation(self):
        """Test hierarchy creation"""
        from pointspace.models.backbone.ezsp.superpoint_hierarchy import (
            SuperpointHierarchy,
        )

        data_list = [
            {"pos": torch.randn(100, 3), "x": torch.randn(100, 32)},
            {"pos": torch.randn(20, 3), "x": torch.randn(20, 32)},
        ]

        nag = SuperpointHierarchy(data_list)
        assert nag.num_levels == 2
        assert nag[0].num_points == 100
        assert nag[1].num_points == 20

    def test_level_ratios(self):
        """Test compression ratio computation"""
        from pointspace.models.backbone.ezsp.superpoint_hierarchy import (
            SuperpointHierarchy,
        )

        data_list = [
            {"pos": torch.randn(100, 3)},
            {"pos": torch.randn(20, 3)},
            {"pos": torch.randn(5, 3)},
        ]

        nag = SuperpointHierarchy(data_list)
        ratios = nag.get_level_ratios()

        assert len(ratios) == 2
        assert ratios[0] == 5.0  # 100/20
        assert ratios[1] == 4.0  # 20/5

    def test_label_propagation(self):
        """Test label propagation from coarse to fine"""
        from pointspace.models.backbone.ezsp.superpoint_hierarchy import (
            SuperpointHierarchy,
        )

        # Level 0: 10 points, Level 1: 3 superpoints
        super_index = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])

        data_list = [
            {"pos": torch.randn(10, 3), "super_index": super_index},
            {"pos": torch.randn(3, 3)},
        ]

        nag = SuperpointHierarchy(data_list)

        # Predictions at level 1
        level1_preds = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).float()

        # Propagate to level 0
        point_preds = nag.propagate_labels_to_points(level1_preds, from_level=1)

        assert point_preds.shape == (10, 3)
        # First 3 points should have class 0
        assert (point_preds[:3].argmax(dim=1) == 0).all()


class TestSparseCNN:
    """Tests for SparseCNN"""

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA required for spconv"
    )
    def test_forward_shape(self):
        """Test output shape"""
        from pointspace.models.backbone.ezsp.sparse_cnn import SparseCNN
        from pointspace.models.utils.structure import Point

        cnn = SparseCNN(in_channels=6, channels=[32, 32, 32]).cuda()

        # Create test data
        N = 1000
        point = Point(
            coord=torch.randn(N, 3).cuda(),
            feat=torch.randn(N, 6).cuda(),
            grid_coord=torch.randint(0, 100, (N, 3)).cuda(),
            batch=torch.zeros(N, dtype=torch.long).cuda(),
            offset=torch.tensor([N], dtype=torch.long).cuda(),
        )

        out_point = cnn(point)

        assert out_point.feat.shape == (N, 32)

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA required for spconv"
    )
    def test_gradient_flow(self):
        """Test gradients flow through SparseCNN"""
        from pointspace.models.backbone.ezsp.sparse_cnn import SparseCNN
        from pointspace.models.utils.structure import Point

        cnn = SparseCNN(in_channels=6, channels=[32, 32, 32]).cuda()

        N = 100
        # Create tensor directly on CUDA to be a leaf tensor
        feat = torch.randn(N, 6, device="cuda", requires_grad=True)
        point = Point(
            coord=torch.randn(N, 3, device="cuda"),
            feat=feat,
            grid_coord=torch.randint(0, 50, (N, 3), device="cuda"),
            batch=torch.zeros(N, dtype=torch.long, device="cuda"),
            offset=torch.tensor([N], dtype=torch.long, device="cuda"),
        )

        out_point = cnn(point)
        loss = out_point.feat.sum()
        loss.backward()

        # Check gradients exist and are valid
        assert feat.grad is not None
        assert not torch.isnan(feat.grad).any()


class TestPartitionModule:
    """Tests for GreedyContourPriorPartition"""

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA required"
    )
    def test_simple_partition(self):
        """Test simple grid-based partition"""
        from pointspace.models.backbone.ezsp.graph_partition import (
            GreedyContourPriorPartitionSimple,
        )

        partition = GreedyContourPriorPartitionSimple(
            k_adjacency=5, grid_size=0.1, num_levels=2
        )

        N = 100
        pos = torch.randn(N, 3).cuda()
        x = torch.randn(N, 32).cuda()
        offset = torch.tensor([N], dtype=torch.long).cuda()

        nag = partition(pos, x, offset)

        assert nag.num_levels == 3  # Level 0 + 2 partition levels
        assert nag[0].num_points == N


class TestPartitionCriterion:
    """Tests for PartitionCriterion loss"""

    def test_forward(self):
        """Test loss computation"""
        from pointspace.models.losses.partition_criterion import PartitionCriterion
        from pointspace.models.backbone.ezsp.superpoint_hierarchy import (
            SuperpointHierarchy,
        )

        criterion = PartitionCriterion(num_classes=3)

        # Create mock NAG
        N = 100
        edge_index = torch.stack(
            [
                torch.randint(0, N, (200,)),
                torch.randint(0, N, (200,)),
            ]
        )
        y = torch.zeros(N, 3)
        labels = torch.randint(0, 3, (N,))
        y.scatter_(1, labels.unsqueeze(1), 1)

        data_list = [
            {
                "pos": torch.randn(N, 3),
                "x": torch.randn(N, 32),
                "edge_index": edge_index,
                "y": y,
            }
        ]
        nag = SuperpointHierarchy(data_list)

        loss, output = criterion(nag)

        assert loss.shape == ()
        assert "n_intra_edge" in output
        assert "n_inter_edge" in output

    def test_adaptive_sampling(self):
        """Test adaptive sampling for class imbalance"""
        from pointspace.models.losses.partition_criterion import PartitionCriterion
        from pointspace.models.backbone.ezsp.superpoint_hierarchy import (
            SuperpointHierarchy,
        )

        criterion = PartitionCriterion(
            num_classes=2, adaptive_sampling=True, adaptive_sampling_ratio=0.5
        )
        criterion.train()

        # Create imbalanced edge labels
        N = 100
        # All same class -> all intra edges
        labels = torch.zeros(N, dtype=torch.long)
        labels[90:] = 1  # Only 10% different class

        edge_index = torch.stack(
            [
                torch.randint(0, N, (200,)),
                torch.randint(0, N, (200,)),
            ]
        )
        y = torch.zeros(N, 2)
        y.scatter_(1, labels.unsqueeze(1), 1)

        data_list = [
            {
                "pos": torch.randn(N, 3),
                "x": torch.randn(N, 32),
                "edge_index": edge_index,
                "y": y,
            }
        ]
        nag = SuperpointHierarchy(data_list)

        loss, output = criterion(nag)

        # Should still compute loss without error
        assert not torch.isnan(loss)


class TestEZSPSegmentor:
    """Tests for EZSPPartitionSegmentor"""

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA required"
    )
    def test_stage1_forward(self):
        """Test Stage 1 (partition learning) forward"""
        from pointspace.models.segmentor.ezsp_segmentor import EZSPPartitionSegmentor

        segmentor = EZSPPartitionSegmentor(
            training_partition_stage=True,
            num_classes=13,
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
        ).cuda()
        segmentor.train()

        N = 100
        input_dict = {
            "coord": torch.randn(N, 3).cuda(),
            "feat": torch.randn(N, 6).cuda(),
            "grid_coord": torch.randint(0, 50, (N, 3)).cuda(),
            "offset": torch.tensor([N], dtype=torch.long).cuda(),
            "segment": torch.randint(0, 13, (N,)).cuda(),
        }

        output = segmentor(input_dict)

        assert "loss" in output

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA required"
    )
    def test_stage2_forward(self):
        """Test Stage 2 (semantic segmentation) forward"""
        from pointspace.models.segmentor.ezsp_segmentor import EZSPPartitionSegmentor

        segmentor = EZSPPartitionSegmentor(
            training_partition_stage=False,
            num_classes=13,
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
            freeze_cnn=False,
        ).cuda()
        segmentor.train()

        N = 100
        input_dict = {
            "coord": torch.randn(N, 3).cuda(),
            "feat": torch.randn(N, 6).cuda(),
            "grid_coord": torch.randint(0, 50, (N, 3)).cuda(),
            "offset": torch.tensor([N], dtype=torch.long).cuda(),
            "segment": torch.randint(0, 13, (N,)).cuda(),
        }

        output = segmentor(input_dict)

        assert "loss" in output
        assert "seg_logits" in output
        assert output["seg_logits"].shape == (N, 13)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
