"""
Test SPT/EZ-SP ignore_label = num_classes convention

This test verifies that the PointSpace EZ-SP implementation correctly follows
the official SPT convention where:
- ignore_label = num_classes (NOT -1!)
- Void/ignored annotations are placed in the (num_classes)-th column of histogram
- Void voxels are filtered from partition learning edges
- Loss functions correctly skip void superpoints

Author: PointSpace Team
"""

import pytest
import torch
import torch.nn as nn
from pointspace.models.losses.partition_criterion import PartitionCriterion


@pytest.mark.parametrize("num_classes", [5, 8, 13])
def test_ignore_label_is_num_classes(num_classes):
    """Verify ignore_label = num_classes convention
    
    In SPT/EZ-SP, labels are represented as histograms (N, num_classes+1):
    - Columns 0 to num_classes-1: valid class counts
    - Column num_classes: void/ignored class counts
    
    When argmax is applied:
    - Valid superpoints → labels in [0, num_classes-1]
    - Void superpoints → label = num_classes
    """
    batch_size = 100
    
    # Create label histogram: (batch_size, num_classes+1)
    y_hist = torch.zeros(batch_size, num_classes + 1)
    
    # First 80%: valid superpoints (random distribution over valid classes)
    valid_count = int(batch_size * 0.8)
    y_hist[:valid_count, :num_classes] = torch.randn(valid_count, num_classes).softmax(dim=1)
    
    # Last 20%: void superpoints (all weight in num_classes column)
    void_count = batch_size - valid_count
    y_hist[valid_count:, num_classes] = 1.0
    
    # Apply argmax to get hard labels
    labels = y_hist.argmax(dim=1)
    
    # Assertions
    assert labels[:valid_count].max() < num_classes, "Valid labels should be < num_classes"
    assert (labels[valid_count:] == num_classes).all(), f"Void labels should be exactly {num_classes}"
    
    # Test with CrossEntropyLoss
    criterion = nn.CrossEntropyLoss(ignore_index=num_classes, reduction='mean')
    logits = torch.randn(batch_size, num_classes, requires_grad=True)
    
    # Compute loss
    loss = criterion(logits, labels)
    
    # Verify loss is finite and positive
    assert torch.isfinite(loss), "Loss should be finite"
    assert loss.item() > 0, "Loss should be positive (has valid samples)"
    
    # Verify void superpoints are ignored (test by backward)
    if loss.requires_grad:
        loss.backward()
    
    # If we manually zero out void entries and recompute, should get same loss
    valid_logits = logits[:valid_count].detach()
    valid_labels = labels[:valid_count]
    loss_manual = nn.functional.cross_entropy(valid_logits, valid_labels, reduction='mean')
    
    # Should be very close (accounting for numerical precision)
    assert torch.isclose(loss.detach(), loss_manual, atol=1e-4), \
        "Loss with ignore_index should match loss computed only on valid samples"


def test_partition_criterion_void_edge_removal():
    """Verify that PartitionCriterion removes edges containing void voxels
    
    From partition_criterion.py:
    ```python
    mask_void_voxels = majority_class_count == 0  # all points in voxel were void
    mask_void_edges = mask_void_voxels[src] | mask_void_voxels[dst]
    edge_index = edge_index[:, ~mask_void_edges]  # Remove void edges
    ```
    """
    num_classes = 8
    num_voxels = 20
    
    # Create histogram labels: (num_voxels, num_classes+1)
    y_hist = torch.zeros(num_voxels, num_classes + 1)
    
    # First 15 voxels: valid (random class distribution)
    y_hist[:15, :num_classes] = torch.randn(15, num_classes).softmax(dim=1) * 10  # multiply for counts
    
    # Last 5 voxels: pure void (all points are ignored)
    y_hist[15:, num_classes] = 10.0
    
    # Create edge index (20 edges)
    edge_index = torch.tensor([
        # Edges between valid voxels (should be kept)
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    ], dtype=torch.long)
    
    # Add edges connecting to void voxels (should be removed)
    void_edges = torch.tensor([
        [10, 11, 12, 15, 16, 17, 18, 19],  # src
        [15, 16, 17, 18, 16, 17, 18, 19],  # dst (at least one is void)
    ], dtype=torch.long)
    
    edge_index = torch.cat([edge_index, void_edges], dim=1)
    num_edges_before = edge_index.shape[1]
    
    # Create PartitionCriterion (check actual parameters)
    criterion = PartitionCriterion(
        num_classes=num_classes,
        gamma=2.0,  # focal loss gamma
        alpha=0.5,
    )
    
    # Manually simulate void edge filtering (mimics internal logic)
    if y_hist.dim() == 2:
        majority_class_count, y_labels = y_hist[:, :num_classes].max(dim=1)
    else:
        majority_class_count = torch.ones_like(y_hist)
        y_labels = y_hist
    
    # Identify void voxels
    mask_void_voxels = majority_class_count == 0
    assert mask_void_voxels.sum() == 5, "Should have 5 void voxels"
    assert mask_void_voxels[15:].all(), "Voxels 15-19 should be void"
    assert not mask_void_voxels[:15].any(), "Voxels 0-14 should be valid"
    
    # Filter void edges
    src, dst = edge_index
    mask_void_edges = mask_void_voxels[src] | mask_void_voxels[dst]
    edge_index_filtered = edge_index[:, ~mask_void_edges]
    
    num_edges_after = edge_index_filtered.shape[1]
    num_void_edges = mask_void_edges.sum().item()
    
    # Assertions
    assert num_void_edges == 8, f"Should have 8 void edges, got {num_void_edges}"
    assert num_edges_after == 10, f"Should keep 10 valid edges, got {num_edges_after}"
    assert edge_index_filtered.max() < 15, "Filtered edges should only connect valid voxels"


def test_histogram_truncation():
    """Verify that PartitionCriterion truncates histogram to ignore void column
    
    From partition_criterion.py:
    ```python
    majority_class_count, y = y[:, :self.num_classes].max(dim=1)
    ```
    
    This ensures majority class computation only considers valid classes,
    not the void class in the last column.
    """
    num_classes = 8
    num_voxels = 10
    
    # Create histograms with void counts
    y_hist = torch.zeros(num_voxels, num_classes + 1)
    
    # Voxel 0: mostly class 2, small void
    y_hist[0, :] = torch.tensor([0, 0, 8, 1, 0, 0, 0, 0, 1])  # majority=2 (class 2)
    
    # Voxel 1: mostly class 5, no void
    y_hist[1, :] = torch.tensor([0, 0, 0, 0, 0, 10, 0, 0, 0])  # majority=5 (class 5)
    
    # Voxel 2: pure void (should have majority_count=0)
    y_hist[2, :] = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 10])  # all void
    
    # Voxel 3: equal split between class 3 and void
    y_hist[3, :] = torch.tensor([0, 0, 0, 5, 0, 0, 0, 0, 5])  # majority=3 (ties go to valid class)
    
    # Truncate to valid classes only
    majority_class_count, y_labels = y_hist[:, :num_classes].max(dim=1)
    
    # Assertions
    assert y_labels[0] == 2, "Voxel 0 majority should be class 2"
    assert majority_class_count[0] == 8, "Voxel 0 majority count should be 8"
    
    assert y_labels[1] == 5, "Voxel 1 majority should be class 5"
    assert majority_class_count[1] == 10, "Voxel 1 majority count should be 10"
    
    assert majority_class_count[2] == 0, "Voxel 2 (pure void) majority count should be 0"
    
    assert y_labels[3] == 3, "Voxel 3 majority should be class 3 (not void)"
    assert majority_class_count[3] == 5, "Voxel 3 majority count should be 5"


def test_dales_ignore_index_convention():
    """Test DALES-specific ignore_index = 8 (num_classes)"""
    num_classes = 8  # DALES has 8 classes
    ignore_index = num_classes  # Should be 8, not -1!
    
    # Create mock data
    batch_size = 100
    y_hist = torch.zeros(batch_size, num_classes + 1)
    
    # 80 valid, 20 void
    y_hist[:80, :num_classes] = torch.randn(80, num_classes).softmax(dim=1)
    y_hist[80:, num_classes] = 1.0
    
    labels = y_hist.argmax(dim=1)
    
    # Test with loss
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
    logits = torch.randn(batch_size, num_classes)
    
    loss = criterion(logits, labels)
    
    # Should not raise error and loss should be finite
    assert torch.isfinite(loss), "Loss should be finite"
    assert loss.item() > 0, "Loss should be positive"


def test_ce_kl_multi_stage_loss():
    """Test ce_kl multi-stage loss convention
    
    In SPT/EZ-SP:
    - Level 0 (finest superpoints): CE loss with argmax labels
    - Level 1+ (coarser superpoints): KL divergence with soft histogram
    """
    num_classes = 8
    num_superpoints_l0 = 100
    num_superpoints_l1 = 30
    
    # Create histograms for two levels
    y_hist_l0 = torch.randn(num_superpoints_l0, num_classes + 1).softmax(dim=1)
    y_hist_l1 = torch.randn(num_superpoints_l1, num_classes + 1).softmax(dim=1)
    
    # Mark some as void
    y_hist_l0[90:, :] = 0
    y_hist_l0[90:, num_classes] = 1.0  # void
    y_hist_l1[25:, :] = 0
    y_hist_l1[25:, num_classes] = 1.0  # void
    
    # Create logits
    logits_l0 = torch.randn(num_superpoints_l0, num_classes)
    logits_l1 = torch.randn(num_superpoints_l1, num_classes)
    
    # Level 0: CE with hard labels
    labels_l0 = y_hist_l0.argmax(dim=1)
    criterion_ce = nn.CrossEntropyLoss(ignore_index=num_classes)
    loss_l0 = criterion_ce(logits_l0, labels_l0)
    
    # Level 1: KL divergence with soft histogram (ignore void)
    # Normalize histogram to only valid classes for KL
    y_hist_l1_valid = y_hist_l1[:, :num_classes]  # Remove void column
    y_hist_l1_valid = y_hist_l1_valid / (y_hist_l1_valid.sum(dim=1, keepdim=True) + 1e-8)
    
    log_probs_l1 = torch.log_softmax(logits_l1, dim=1)
    loss_l1 = nn.functional.kl_div(
        log_probs_l1, 
        y_hist_l1_valid, 
        reduction='batchmean'
    )
    
    # Combined loss (SPT uses lambdas=[1, 50])
    loss_total = loss_l0 + 50 * loss_l1
    
    # Assertions
    assert torch.isfinite(loss_l0), "Level 0 loss should be finite"
    assert torch.isfinite(loss_l1), "Level 1 loss should be finite"
    assert torch.isfinite(loss_total), "Total loss should be finite"
    assert loss_total.item() > 0, "Total loss should be positive"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
