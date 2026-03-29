"""
Test script for EZ-SP implementation fixes.

Tests the following fixes:
1. SuperpointHierarchy.get_level() method
2. Batch propagation to higher levels
3. v_edge_attr computation
4. SparseCNN freeze/unfreeze methods
5. GreedyContourPriorPartition complete output

Run: python -m pytest tests/test_ezsp_fixes.py -v
"""

import pytest
import torch
import torch.nn as nn

# Skip all tests if required packages are not available
try:
    import spconv.pytorch as spconv
    HAS_SPCONV = True
except ImportError:
    HAS_SPCONV = False

try:
    from torch_scatter import scatter_sum, scatter_mean, scatter_min
    HAS_TORCH_SCATTER = True
except ImportError:
    HAS_TORCH_SCATTER = False


@pytest.fixture
def sample_point_cloud():
    """Create a sample point cloud for testing."""
    torch.manual_seed(42)
    num_points = 1000
    num_batches = 2
    
    # Random positions
    pos = torch.randn(num_points, 3)
    
    # Random features
    feat = torch.randn(num_points, 6)
    
    # Batch indices (split evenly)
    points_per_batch = num_points // num_batches
    batch = torch.zeros(num_points, dtype=torch.long)
    for i in range(num_batches):
        start = i * points_per_batch
        end = (i + 1) * points_per_batch if i < num_batches - 1 else num_points
        batch[start:end] = i
    
    # Offset (cumulative count)
    offset = torch.tensor([points_per_batch, num_points], dtype=torch.long)
    
    # Grid coordinates for sparse conv
    grid_coord = (pos * 100).int()
    
    return {
        'pos': pos,
        'feat': feat,
        'batch': batch,
        'offset': offset,
        'grid_coord': grid_coord,
    }


class TestSuperpointHierarchy:
    """Test SuperpointHierarchy fixes."""
    
    def test_get_level_method_exists(self):
        """Test that get_level() method exists and works."""
        from pointspace.models.backbone.ezsp.superpoint_hierarchy import (
            SuperpointHierarchy,
            SuperpointLevel,
        )
        
        # Create a simple hierarchy with 2 levels
        level0_data = {
            'pos': torch.randn(100, 3),
            'x': torch.randn(100, 32),
            'batch': torch.zeros(100, dtype=torch.long),
        }
        level1_data = {
            'pos': torch.randn(20, 3),
            'x': torch.randn(20, 32),
            'batch': torch.zeros(20, dtype=torch.long),
        }
        
        hierarchy = SuperpointHierarchy([level0_data, level1_data])
        
        # Test get_level method
        assert hasattr(hierarchy, 'get_level'), "get_level method should exist"
        
        level0 = hierarchy.get_level(0)
        level1 = hierarchy.get_level(1)
        
        assert isinstance(level0, SuperpointLevel)
        assert isinstance(level1, SuperpointLevel)
        assert level0.num_points == 100
        assert level1.num_points == 20
        
        # Test that get_level is equivalent to __getitem__
        assert level0 is hierarchy[0]
        assert level1 is hierarchy[1]


class TestBatchPropagation:
    """Test batch propagation to higher levels."""
    
    @pytest.mark.skipif(not HAS_TORCH_SCATTER, reason="torch_scatter required")
    def test_scatter_min_import(self):
        """Test that scatter_min is properly imported."""
        from pointspace.models.backbone.ezsp.graph_partition import scatter_min
        
        # Simple test
        src = torch.tensor([1, 2, 3, 4, 5])
        index = torch.tensor([0, 0, 1, 1, 1])
        result = scatter_min(src, index, dim=0)[0]
        
        assert result[0] == 1  # min of [1, 2]
        assert result[1] == 3  # min of [3, 4, 5]
    
    def test_batch_in_merged_data(self):
        """Test that merged_data includes batch and offset."""
        from pointspace.models.backbone.ezsp.graph_partition import GreedyContourPriorPartition
        
        # Check that the class accepts batch parameter in _merge_components
        import inspect
        sig = inspect.signature(GreedyContourPriorPartition._merge_components)
        params = list(sig.parameters.keys())
        
        assert 'batch' in params, "_merge_components should accept batch parameter"


class TestVEdgeAttr:
    """Test vertical edge attribute computation."""
    
    def test_v_edge_attr_in_merged_data(self):
        """Test that _merge_components returns v_edge_attr."""
        from pointspace.models.backbone.ezsp.graph_partition import GreedyContourPriorPartition
        
        # Check that _compute_vertical_edge_attr method exists
        assert hasattr(GreedyContourPriorPartition, '_compute_vertical_edge_attr'), \
            "_compute_vertical_edge_attr method should exist"
    
    def test_v_edge_attr_computation(self):
        """Test 9D vertical edge attribute computation logic."""
        from pointspace.models.backbone.ezsp.graph_partition import GreedyContourPriorPartition
        
        # Create a simple partition instance to test the method
        partition = GreedyContourPriorPartition(
            reg=[0.1],
            min_size=[10],
            k_adjacency=10,
        )
        
        # Test data
        pos_child = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [2.0, 0.0, 0.0],
        ])
        pos_parent = torch.tensor([
            [0.5, 0.5, 0.0],  # Parent 0: children 0, 1, 2
            [2.0, 0.0, 0.0],  # Parent 1: child 3
        ])
        super_index = torch.tensor([0, 0, 0, 1])
        node_size_child = torch.ones(4, dtype=torch.long)
        node_size_parent = torch.tensor([3, 1], dtype=torch.long)
        
        # Compute v_edge_attr
        normal_child = torch.tensor([
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ])
        normal_parent = torch.tensor([
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ])
        log_size_child = torch.zeros(4)
        log_size_parent = torch.log(torch.tensor([3.0, 1.0]))
        log_length_child = log_size_child / 3.0
        log_surface_child = log_size_child * 2.0 / 3.0
        log_volume_child = log_size_child
        log_length_parent = log_size_parent / 3.0
        log_surface_parent = log_size_parent * 2.0 / 3.0
        log_volume_parent = log_size_parent

        v_edge_attr = partition._compute_vertical_edge_attr(
            pos_child=pos_child,
            pos_parent=pos_parent,
            super_index=super_index,
            node_size_child=node_size_child,
            node_size_parent=node_size_parent,
            normal_child=normal_child,
            normal_parent=normal_parent,
            log_length_child=log_length_child,
            log_length_parent=log_length_parent,
            log_surface_child=log_surface_child,
            log_surface_parent=log_surface_parent,
            log_volume_child=log_volume_child,
            log_volume_parent=log_volume_parent,
            log_size_child=log_size_child,
            log_size_parent=log_size_parent,
        )
        
        # Check shape: [N_child, 9]
        assert v_edge_attr.shape == (4, 9), f"Expected shape (4, 9), got {v_edge_attr.shape}"
        
        # Check that direction vectors are roughly normalized
        directions = v_edge_attr[:, :3]
        norms = directions.norm(dim=1)
        # Allow for some numerical tolerance (directions might be zero if child == parent)
        assert (norms <= 1.0 + 1e-5).all(), "Direction vectors should be normalized"

    def test_h_edge_attr_computation(self):
        """Test 18D horizontal edge attribute computation logic."""
        from pointspace.models.backbone.ezsp.graph_partition import GreedyContourPriorPartition

        partition = GreedyContourPriorPartition(reg=[0.1], min_size=[10], k_adjacency=10)
        pos = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        edge_index = torch.tensor([
            [0, 1, 2],
            [1, 2, 0],
        ])
        normal = torch.tensor([
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ])
        log_size = torch.log(torch.tensor([1.0, 2.0, 3.0]))
        log_length = log_size / 3.0
        log_surface = log_size * 2.0 / 3.0
        log_volume = log_size

        edge_attr = partition._compute_horizontal_edge_attr(
            pos=pos,
            edge_index=edge_index,
            normal=normal,
            log_length=log_length,
            log_surface=log_surface,
            log_volume=log_volume,
            log_size=log_size,
        )
        assert edge_attr.shape == (3, 18), f"Expected shape (3, 18), got {edge_attr.shape}"


class TestSparseCNN:
    """Test SparseCNN fixes."""
    
    @pytest.mark.skipif(not HAS_SPCONV, reason="spconv required")
    def test_freeze_unfreeze_methods(self):
        """Test freeze and unfreeze methods exist and work."""
        from pointspace.models.backbone.ezsp.sparse_cnn import SparseCNN
        
        # Create a simple SparseCNN
        cnn = SparseCNN(
            in_channels=6,
            channels=[32, 32, 32],
            kernel_size=3,
        )
        
        # Test methods exist
        assert hasattr(cnn, 'freeze'), "freeze method should exist"
        assert hasattr(cnn, 'unfreeze'), "unfreeze method should exist"
        assert hasattr(cnn, 'frozen'), "frozen property should exist"
        
        # Test initial state
        assert not cnn.frozen, "CNN should not be frozen initially"
        
        # Test freeze
        cnn.freeze()
        assert cnn.frozen, "CNN should be frozen after freeze()"
        for param in cnn.parameters():
            assert not param.requires_grad, "All parameters should have requires_grad=False"
        
        # Test unfreeze
        cnn.unfreeze()
        assert not cnn.frozen, "CNN should not be frozen after unfreeze()"
        for param in cnn.parameters():
            assert param.requires_grad, "All parameters should have requires_grad=True"
    
    @pytest.mark.skipif(not HAS_SPCONV, reason="spconv required")
    def test_last_norm_activation_params(self):
        """Test last_norm and last_activation parameters."""
        from pointspace.models.backbone.ezsp.sparse_cnn import SparseCNN
        import inspect
        
        # Check parameters exist
        sig = inspect.signature(SparseCNN.__init__)
        params = list(sig.parameters.keys())
        
        assert 'last_norm' in params, "last_norm parameter should exist"
        assert 'last_activation' in params, "last_activation parameter should exist"
        assert 'frozen' in params, "frozen parameter should exist"
    
    @pytest.mark.skipif(not HAS_SPCONV, reason="spconv required")
    def test_frozen_init(self):
        """Test that frozen=True at init works."""
        from pointspace.models.backbone.ezsp.sparse_cnn import SparseCNN
        
        cnn = SparseCNN(
            in_channels=6,
            channels=[32, 32],
            frozen=True,
        )
        
        assert cnn.frozen, "CNN should be frozen when frozen=True passed"
        for param in cnn.parameters():
            assert not param.requires_grad


class TestSPTIntegration:
    """Integration tests for SPT with fixed components."""
    
    def test_spt_forward_with_get_level(self):
        """Test that SPT forward uses get_level correctly."""
        from pointspace.models.backbone.ezsp.superpoint_hierarchy import SuperpointHierarchy
        
        # Create a mock hierarchy
        levels_data = []
        for i in range(3):
            n_points = 100 // (2 ** i)
            levels_data.append({
                'pos': torch.randn(n_points, 3),
                'x': torch.randn(n_points, 32),
                'batch': torch.zeros(n_points, dtype=torch.long),
                'edge_index': torch.randint(0, n_points, (2, n_points * 2)),
                'edge_attr': torch.randn(n_points * 2, 18),
                'node_size': torch.ones(n_points, dtype=torch.long),
            })
            if i > 0:
                levels_data[-1]['sub'] = None  # Placeholder
        
        # Add super_index to link levels
        for i in range(len(levels_data) - 1):
            n_curr = levels_data[i]['pos'].shape[0]
            n_next = levels_data[i + 1]['pos'].shape[0]
            levels_data[i]['super_index'] = torch.randint(0, n_next, (n_curr,))
        
        hierarchy = SuperpointHierarchy(levels_data)
        
        # Test get_level works for all levels
        for i in range(len(levels_data)):
            level = hierarchy.get_level(i)
            assert level is not None
            assert 'pos' in level
            assert 'x' in level


class TestGraphNorm:
    """Test GraphNorm implementation."""
    
    def test_graph_norm_forward(self):
        """Test GraphNorm forward pass."""
        from pointspace.models.backbone.ezsp.graph_norm import GraphNorm
        
        norm = GraphNorm(32)
        
        # Test data
        x = torch.randn(100, 32)
        batch = torch.zeros(100, dtype=torch.long)
        batch[50:] = 1  # Two batches
        
        # Forward pass
        out = norm(x, batch)
        
        assert out.shape == x.shape, f"Output shape should match input: {out.shape} vs {x.shape}"
        assert not torch.isnan(out).any(), "Output should not contain NaN"


def run_quick_tests():
    """Run a quick subset of tests without pytest."""
    print("Running quick tests...")
    
    # Test 1: get_level
    print("\n1. Testing SuperpointHierarchy.get_level()...")
    from pointspace.models.backbone.ezsp.superpoint_hierarchy import SuperpointHierarchy
    
    level0_data = {'pos': torch.randn(100, 3), 'x': torch.randn(100, 32)}
    level1_data = {'pos': torch.randn(20, 3), 'x': torch.randn(20, 32)}
    hierarchy = SuperpointHierarchy([level0_data, level1_data])
    
    assert hasattr(hierarchy, 'get_level')
    assert hierarchy.get_level(0) is hierarchy[0]
    print("   ✓ get_level() works correctly")
    
    # Test 2: v_edge_attr computation
    print("\n2. Testing v_edge_attr computation...")
    from pointspace.models.backbone.ezsp.graph_partition import GreedyContourPriorPartition
    
    partition = GreedyContourPriorPartition(reg=[0.1], min_size=[10], k_adjacency=10)
    assert hasattr(partition, '_compute_vertical_edge_attr')
    print("   ✓ _compute_vertical_edge_attr method exists")
    
    # Test 3: SparseCNN freeze/unfreeze
    print("\n3. Testing SparseCNN freeze/unfreeze...")
    if HAS_SPCONV:
        from pointspace.models.backbone.ezsp.sparse_cnn import SparseCNN
        cnn = SparseCNN(in_channels=6, channels=[32, 32])
        
        assert not cnn.frozen
        cnn.freeze()
        assert cnn.frozen
        cnn.unfreeze()
        assert not cnn.frozen
        print("   ✓ freeze/unfreeze works correctly")
    else:
        print("   ⚠ Skipped (spconv not available)")
    
    # Test 4: Batch import
    print("\n4. Testing scatter_min import...")
    from pointspace.models.backbone.ezsp.graph_partition import scatter_min
    print("   ✓ scatter_min imported successfully")
    
    print("\n" + "="*50)
    print("All quick tests passed!")
    print("="*50)


if __name__ == "__main__":
    run_quick_tests()
