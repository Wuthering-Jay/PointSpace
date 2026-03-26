# EZ-SP for PointSpace

GPU-accelerated superpoint segmentation adapted from the Superpoint Transformer project.

## Overview

EZ-SP (Easy Superpoint) provides 72× faster superpoint learning compared to PTv3 by using:
- **GPU-based clustering**: `torch-graph-components` for graph partitioning
- **Lightweight CNN**: TinySparseCNN (32→32→32 channels)
- **Contrastive learning**: Same-class points attract, different-class repel
- **Forward-pass graphs**: KNN graph built on GPU during forward pass

## Installation

```bash
# Install required dependency
pip install torch-graph-components
```

## Components

### 1. TinySparseCNN
Lightweight sparse CNN for feature extraction (32 channels, 3 conv blocks).

```python
from pointspace.models.ezsp import TinySparseCNN

backbone = TinySparseCNN(
    in_channels=9,
    channels=[32, 32, 32],
    kernel_sizes=[7, 3, 3],
)
```

### 2. GPUGreedyPartition
GPU-accelerated graph clustering using contour prior energy.

```python
from pointspace.models.ezsp import GPUGreedyPartition

partition = GPUGreedyPartition(
    reg=0.02,          # Regularization (controls coarseness)
    min_size=30,       # Minimum superpoint size
    k=8,               # KNN neighbors
)

result = partition(feat=features, pos=positions, batch=batch_indices)
# Returns: super_index, num_superpoints, super_feat, super_pos, etc.
```

### 3. EZSPContrastiveLoss
Contrastive loss for training partition features.

```python
from pointspace.models.ezsp import EZSPContrastiveLoss

loss_fn = EZSPContrastiveLoss(
    affinity_temperature=1.0,
    focal_gamma=1.0,
    adaptive_sampling_ratio=0.7,
    num_classes=8,
    k=8,
)

loss = loss_fn(feat=features, pos=positions, segment=labels, offset=offsets)
```

### 4. EZSPPartitionSegmentor
Complete training module combining all components.

```python
from pointspace.models.ezsp import build_ezsp_partition_model

model = build_ezsp_partition_model(
    in_channels=9,
    cnn_channels=[32, 32, 32],
    partition_min_size=30,
    num_classes=8,
)
```

## Training Workflow

### Phase 1: Partition Learning

Train contrastive features for superpoint clustering:

```bash
# Train on DALES dataset
python tools/train.py configs/dales/ezsp-partition-0.py
```

**What happens:**
- TinySparseCNN extracts 32-dim features from points
- KNN graph built on GPU (k=8 neighbors)
- Contrastive loss: same-class edges have high affinity, cross-class edges low
- Validation: GPU greedy partition creates superpoints

**Configuration:** `configs/dales/ezsp-partition-0.py`
- Model: `EZSPPartitionSegmentor`
- Loss: `EZSPContrastiveLoss`
- Epochs: 200 (partition learning needs more iterations)
- LR: 5e-4 with cosine annealing

### Phase 2: Semantic Segmentation (Future)

Use learned features with full SPT model for semantic segmentation.

## Configuration Example

```python
# DALES EZ-SP Partition Config
model = dict(
    type="EZSPPartitionSegmentor",
    backbone=dict(
        type="TinySparseCNN",
        in_channels=5,  # coord (3) + echo (1) + normalized (1)
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
        ),
    ],
    partition_cfg=dict(
        reg=0.02,
        min_size=30,
        k=8,
    ),
)
```

## Key Parameters

### Partition Parameters
- `reg`: Regularization strength (higher = coarser partitions). Default: 0.02
- `min_size`: Minimum superpoint size. Default: 30
- `k`: KNN neighbors. Default: 8

### Loss Parameters
- `affinity_temperature`: Controls feature similarity sensitivity. Default: 1.0
- `focal_gamma`: Focal loss focusing parameter. Default: 1.0
- `adaptive_sampling_ratio`: Balance inter/intra class edges. Default: 0.7

### CNN Parameters
- `channels`: Channel sizes for each conv block. Default: [32, 32, 32]
- `kernel_sizes`: Kernel sizes for each conv block. Default: [7, 3, 3]

## Utilities

### Format Conversion (offset ↔ ptr)

PointSpace uses offset format, NAG/SPT uses CSR ptr format:

```python
from pointspace.models.ezsp import offset_to_ptr, ptr_to_offset

# PointSpace: offset = [5, 8, 12] (no leading zero)
# NAG/SPT:    ptr    = [0, 5, 8, 12] (with leading zero)

ptr = offset_to_ptr(offset)      # [5, 8, 12] -> [0, 5, 8, 12]
offset = ptr_to_offset(ptr)      # [0, 5, 8, 12] -> [5, 8, 12]

# Other utilities
from pointspace.models.ezsp import (
    sizes_to_ptr,        # [5, 3, 4] -> [0, 5, 8, 12]
    ptr_to_sizes,        # [0, 5, 8, 12] -> [5, 3, 4]
    batch_to_ptr,        # batch indices -> ptr
    ptr_to_batch,        # ptr -> batch indices
)
```

## Testing

Run end-to-end test:

```bash
python test_ezsp_e2e.py
```

Tests:
1. Model instantiation from config
2. Training forward pass with loss
3. Backward pass and gradient computation
4. Validation with partition computation
5. Standalone contrastive loss

## Architecture Highlights

**Why GPU-based?**
- Cut-Pursuit (CPU) is slow, limits scalability
- `torch-graph-components` provides CUDA kernels for graph ops
- 72× speedup compared to PTv3

**Why lightweight CNN?**
- Partition only needs discriminative features, not semantic
- 32 channels sufficient for graph clustering
- Keeps training fast and memory efficient

**Why contrastive loss?**
- Direct supervision on graph edges
- No need for edge classification (0/1)
- Adaptive sampling balances inter/intra class edges

## File Structure

```
pointspace/models/ezsp/
├── __init__.py              # Module exports
├── utils.py                 # offset ↔ ptr conversion
├── tiny_sparse_cnn.py       # Lightweight feature CNN
├── partition.py             # GPU greedy partition
├── loss.py                  # Contrastive loss
└── segmentor.py             # Training wrapper

configs/dales/
└── ezsp-partition-0.py      # DALES partition config

test_ezsp_e2e.py             # End-to-end test
```

## Reference

- Paper: [EZ-SP: Fast Superpoint Learning](https://arxiv.org/abs/2402.04991)
- Original: https://github.com/drprojects/superpoint_transformer
- Dependency: `torch-graph-components` (pip installable)

## Notes

- Phase 1 (this implementation): Partition feature learning
- Phase 2 (future): Full SPT with learned features for semantic segmentation
- Dataset: Requires point clouds with semantic labels
- Memory: TinySparseCNN is lightweight, can handle large scenes

## Troubleshooting

**No inter-class edges found:**
- Check if classes are well-separated in space
- Increase KNN k or use larger r_max
- Verify adaptive_sampling_ratio not too high

**Loss is zero:**
- Ensure training mode is enabled
- Check if segment labels are provided
- Verify num_classes matches dataset

**CUDA out of memory:**
- Reduce batch_size or points per scene
- Enable sharding in partition config
- Use gradient_accumulation_steps
