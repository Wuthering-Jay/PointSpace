"""
End-to-End Test for EZ-SP Partition Training

This script tests the entire EZ-SP partition learning pipeline:
1. Model instantiation from config
2. Forward pass with training data
3. Loss computation and backward pass
4. Validation forward pass with partition computation
5. Integration with PointSpace training framework
"""

import torch
import sys
sys.path.insert(0, 'e:/code/python/PointSpace')

from addict import Dict
from pointspace.models.builder import MODELS
from pointspace.models.losses.builder import LOSSES


def create_dummy_batch(batch_size=2, points_per_sample=500, num_classes=8):
    """Create a dummy batch mimicking DALES data."""
    total_points = batch_size * points_per_sample

    # Create overlapping clusters to ensure inter-class edges
    coords = []
    segments = []

    points_per_class = points_per_sample // num_classes
    for b in range(batch_size):
        for c in range(num_classes):
            # Each class gets a slightly offset cluster
            class_coords = torch.randn(points_per_class, 3) * 2 + torch.tensor([c * 2.0, 0, 0])
            coords.append(class_coords)
            segments.append(torch.full((points_per_class,), c, dtype=torch.long))

    coord = torch.cat(coords)
    segment = torch.cat(segments)

    # Shuffle to mix classes
    perm = torch.randperm(total_points)
    coord = coord[perm]
    segment = segment[perm]

    # Create batch data
    batch = Dict(
        feat=torch.randn(total_points, 5),  # 5 channels: xyz + echo (+ optionally normals)
        coord=coord,
        grid_coord=torch.randint(0, 100, (total_points, 3)),
        offset=torch.tensor([points_per_sample * i for i in range(1, batch_size + 1)]),
        segment=segment,
    )

    return batch


def test_model_instantiation():
    """Test 1: Model can be instantiated from config."""
    print("=" * 60)
    print("Test 1: Model Instantiation")
    print("=" * 60)

    model_cfg = dict(
        type="EZSPPartitionSegmentor",
        backbone=dict(
            type="TinySparseCNN",
            in_channels=5,
            channels=[32, 32, 32],
            kernel_sizes=[7, 3, 3],
        ),
        partition_criteria=[
            dict(
                type="EZSPContrastiveLoss",
                affinity_temperature=1.0,
                focal_gamma=1.0,
                adaptive_sampling_ratio=0.7,
                num_classes=8,
                k=8,
                loss_weight=1.0,
                train_only=True,
            ),
        ],
        partition_cfg=dict(
            reg=0.02,
            min_size=30,
            k=8,
            edge_weight_mode="unit",
            verbose=False,
        ),
        compute_partition_on_val=True,
    )

    model = MODELS.build(model_cfg)
    model = model.cuda()

    print(f"✓ Model instantiated: {model.__class__.__name__}")
    print(f"  - Backbone: {model.backbone.__class__.__name__}")
    print(f"  - Partition criteria: {len(model.partition_criteria)} loss(es)")
    print(f"  - Validation partition: {model.partition is not None}")

    return model


def test_training_forward(model):
    """Test 2: Training forward pass with loss computation."""
    print("\n" + "=" * 60)
    print("Test 2: Training Forward Pass")
    print("=" * 60)

    model.train()

    # Create dummy batch
    batch = create_dummy_batch(batch_size=2, points_per_sample=500, num_classes=8)
    batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    print(f"Input batch:")
    print(f"  - Total points: {batch['feat'].shape[0]}")
    print(f"  - Batch size: {len(batch['offset'])}")
    print(f"  - Unique classes: {batch['segment'].unique().tolist()}")

    # Forward pass
    output = model(batch)

    print(f"\nOutput:")
    print(f"  - Keys: {list(output.keys())}")
    print(f"  - Loss: {output['loss'].item():.4f}")
    print(f"  - l_partition: {output['l_partition'].item():.4f}")
    print(f"  - feat shape: {output['feat'].shape}")

    assert 'loss' in output, "Missing 'loss' in output"
    assert 'feat' in output, "Missing 'feat' in output"
    assert output['loss'].requires_grad, "Loss doesn't require grad"

    print("✓ Training forward pass successful")

    return output


def test_backward_pass(output):
    """Test 3: Backward pass and gradient computation."""
    print("\n" + "=" * 60)
    print("Test 3: Backward Pass")
    print("=" * 60)

    loss = output['loss']

    # Backward
    loss.backward()

    print(f"Loss value: {loss.item():.4f}")
    print(f"✓ Backward pass completed without errors")


def test_validation_forward(model):
    """Test 4: Validation forward pass with partition computation."""
    print("\n" + "=" * 60)
    print("Test 4: Validation Forward Pass")
    print("=" * 60)

    model.eval()

    # Create new batch for validation
    batch = create_dummy_batch(batch_size=2, points_per_sample=500, num_classes=8)
    batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    with torch.no_grad():
        output = model(batch)

    print(f"Output:")
    print(f"  - Keys: {list(output.keys())}")
    print(f"  - feat shape: {output['feat'].shape}")

    if 'super_index' in output:
        print(f"  - super_index shape: {output['super_index'].shape}")
        print(f"  - num_superpoints: {output['num_superpoints']}")
        print(f"  - Compression ratio: {output['feat'].shape[0] / output['num_superpoints']:.1f}x")
        print("✓ Partition computation successful")
    else:
        print("  - No partition computed (optional)")

    print("✓ Validation forward pass successful")


def test_loss_only():
    """Test 5: Standalone contrastive loss."""
    print("\n" + "=" * 60)
    print("Test 5: Standalone Contrastive Loss")
    print("=" * 60)

    loss_cfg = dict(
        type="EZSPContrastiveLoss",
        affinity_temperature=1.0,
        focal_gamma=1.0,
        adaptive_sampling_ratio=0.7,
        num_classes=8,
        k=8,
        loss_weight=1.0,
    )

    loss_fn = LOSSES.build(loss_cfg)
    loss_fn = loss_fn.cuda()
    loss_fn.train()

    # Create test data
    N = 1000
    feat = torch.randn(N, 32, requires_grad=True).cuda()

    # Overlapping clusters
    coord = torch.cat([
        torch.randn(250, 3).cuda() * 1 + torch.tensor([[0, 0, 0]]).cuda(),
        torch.randn(250, 3).cuda() * 1 + torch.tensor([[3, 0, 0]]).cuda(),
        torch.randn(250, 3).cuda() * 1 + torch.tensor([[0, 3, 0]]).cuda(),
        torch.randn(250, 3).cuda() * 1 + torch.tensor([[3, 3, 0]]).cuda(),
    ])

    segment = torch.cat([
        torch.zeros(250, dtype=torch.long).cuda(),
        torch.ones(250, dtype=torch.long).cuda(),
        torch.full((250,), 2, dtype=torch.long).cuda(),
        torch.full((250,), 3, dtype=torch.long).cuda(),
    ])

    offset = torch.tensor([N]).cuda()

    # Compute loss
    loss = loss_fn(feat=feat, pos=coord, segment=segment, offset=offset)

    print(f"Loss value: {loss.item():.4f}")
    print(f"Loss requires grad: {loss.requires_grad}")

    # Backward
    loss.backward()
    print(f"Gradient norm: {feat.grad.norm().item():.4f}")

    print("✓ Contrastive loss works correctly")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("EZ-SP END-TO-END TESTING")
    print("=" * 60)

    try:
        # Test 1: Model instantiation
        model = test_model_instantiation()

        # Test 2: Training forward
        output = test_training_forward(model)

        # Test 3: Backward pass
        test_backward_pass(output)

        # Test 4: Validation forward
        test_validation_forward(model)

        # Test 5: Loss only
        test_loss_only()

        # Summary
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        print("\nEZ-SP partition learning pipeline is ready for training!")
        print("\nNext steps:")
        print("1. Prepare DALES dataset with required preprocessing")
        print("2. Run training: python tools/train.py configs/dales/ezsp-partition-0.py")
        print("3. Monitor partition quality during training")
        print("4. Use learned features for semantic segmentation (Phase 2)")

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"TEST FAILED: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
