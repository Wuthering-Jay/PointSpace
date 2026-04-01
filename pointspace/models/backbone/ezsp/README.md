# EZ-SP (End-to-End Superpoint Transformer) for PointSpace

This module provides a complete implementation of the EZ-SP (Superpoint Transformer) 
architecture, adapted from the official implementation for the PointSpace framework.

## ⚠️ CRITICAL: ignore_label Convention

**SPT/EZ-SP uses `ignore_index = num_classes` (NOT -1!)**

This is a fundamental architectural decision that affects all training stages:

### Histogram Label Representation

Labels use **(N, C+1)** histogram format:
- Columns 0 to C-1: valid class point counts
- Column C: void/ignored point counts

```python
# Example: DALES with 8 classes
num_classes = 8
ignore_index = 8  # Must be num_classes!

# Superpoint histogram (10 points total):
# - 7 "ground" (class 0), 2 "vegetation" (class 1), 1 void
y_hist = [7, 2, 0, 0, 0, 0, 0, 0, 1]  # Shape: (9,)
argmax(y_hist) = 0  # Dominant class: ground

# Pure void superpoint:
y_hist_void = [0, 0, 0, 0, 0, 0, 0, 0, 10]
argmax(y_hist_void) = 8  # Will be ignored by loss
```

### Stage 1: Void Edge Filtering

Partition learning **removes all edges touching void voxels**:
- Prevents CNN from learning meaningless boundaries
- Isolates feature space from unlabeled regions

### Stage 2: Multi-Stage Loss

- **ce**: CrossEntropy (hard labels via argmax)
- **kl**: KL divergence (soft histogram)
- **ce_kl** (default): CE on Level-1, KL on Level-2+

See `files/ignore_label_correction.md` for detailed explanation.

## Reference

- **Paper**: "Superpoint Transformer for 3D Scene Instance Segmentation" (Robert et al., 2023)
- **Original Code**: https://github.com/drprojects/superpoint_transformer
- **ArXiv**: https://arxiv.org/abs/2306.08045

## Architecture Overview

```
Point Cloud
    │
    ▼
┌──────────────┐
│  SparseCNN   │  ← 3D稀疏卷积特征提取
│  (3 layers)  │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   Greedy     │  ← 基于贪心算法的超点分割
│  Partition   │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Superpoint   │  ← 多层级超点图表示
│  Hierarchy   │
│   (NAG)      │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│     SPT      │  ← UNet结构的Transformer
│ Transformer  │
└──────┬───────┘
       │
       ▼
   Semantic
   Labels
```

## Two-Stage Training

### Stage 1: Partition Learning
- 训练目标: 学习边界对齐的点特征
- 损失函数: PartitionCriterion (边分类损失)
- 输出: 预训练的SparseCNN权重

### Stage 2: Semantic Segmentation  
- 训练目标: 超点图上的语义分割
- 损失函数: CrossEntropy + Lovasz
- 输入: 使用Stage 1预训练的CNN

## Module Structure

```
pointspace/models/backbone/ezsp/
├── sparse_cnn.py           # SparseCNN (3D稀疏卷积)
├── graph_norm.py           # GraphNorm层
├── graph_partition.py      # 贪心分割算法
├── superpoint_hierarchy.py # 超点层次结构 (NAG)
├── ezsp_transformer.py     # SPT Transformer包装器
├── spt/                    # SPT核心组件
│   ├── spt.py             # SPT主网络
│   ├── stage.py           # Stage, DownNFuseStage, UpNFuseStage
│   ├── attention.py       # SelfAttentionBlock (带RPE)
│   ├── transformer.py     # TransformerBlock
│   ├── mlp.py             # MLP模块
│   ├── norm.py            # 归一化层 (BatchNorm, UnitSphereNorm等)
│   ├── pool.py            # 池化操作 (max, mean, sum)
│   ├── fusion.py          # 特征融合 (CatFusion, IndexUnpool)
│   ├── dropout.py         # DropPath等
│   └── utils.py           # 工具函数
└── __init__.py
```

## Key Components

### 1. SuperpointHierarchy (NAG)

分层点云表示:

```python
from pointspace.models.backbone.ezsp.superpoint_hierarchy import SuperpointHierarchy

# 创建层次结构
nag = SuperpointHierarchy([level0_data, level1_data, level2_data])

# 访问层级
level0 = nag[0]  # 原始点云
level1 = nag[1]  # 第一层超点
level2 = nag[2]  # 第二层超点

# 标签传播
point_preds = nag.propagate_labels_to_points(sp_preds, from_level=1)
```

### 2. SparseCNN

3D稀疏卷积特征提取:

```python
from pointspace.models.backbone.ezsp.sparse_cnn import SparseCNN
from pointspace.models.utils.structure import Point

cnn = SparseCNN(
    in_channels=6,      # coord(3) + features
    channels=[32, 32, 32],
    norm="gn",          # GraphNorm
)

point = Point(input_dict)
output_point = cnn(point)  # feat: [N, 32]
```

### 3. PartitionCriterion

边分类损失:

```python
from pointspace.models.losses.partition_criterion import PartitionCriterion

criterion = PartitionCriterion(
    gamma=1.0,           # Focal loss参数
    alpha=0.5,           # 类平衡权重
    temperature=1.0,     # 亲和度温度
    adaptive_sampling=True,
    adaptive_sampling_ratio=0.9,
    num_classes=8,
)

loss, output = criterion(nag)
```

### 4. EZSPPartitionSegmentor

完整分割器:

```python
from pointspace.models.segmentor.ezsp_segmentor import EZSPPartitionSegmentor

# Stage 1
segmentor_s1 = EZSPPartitionSegmentor(
    training_partition_stage=True,
    num_classes=8,
    sparse_cnn=dict(...),
    partition_module=dict(...),
)

# Stage 2
segmentor_s2 = EZSPPartitionSegmentor(
    training_partition_stage=False,
    num_classes=8,
    sparse_cnn=dict(...),
    partition_module=dict(...),
    transformer=dict(...),
    freeze_cnn=True,
)

# 加载Stage 1权重
segmentor_s2.load_stage1_weights("stage1_checkpoint.pth")
```

## Configuration Example (DALES)

```python
# configs/dales/semseg-ezsp-v1-0.py

model = dict(
    type="EZSPPartitionSegmentor",
    training_partition_stage=False,
    num_classes=8,
    sparse_cnn=dict(
        type="EZ-SparseCNN",
        in_channels=5,
        channels=[32, 32, 32],
        norm="gn",
    ),
    partition_module=dict(
        type="GreedyContourPriorPartitionSimple",
        k_adjacency=10,
        grid_size=0.1,
        num_levels=2,
    ),
    transformer=dict(
        type="EZSPTransformerSimple",
        num_classes=8,
        in_channels=32,
        hidden_dim=64,
        num_heads=16,
        num_blocks=3,
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=0.5),
    ],
)
```

## Key Differences from Official Implementation

| Aspect | Official (PyTorch Lightning) | PointSpace |
|--------|------------------------------|------------|
| Framework | PyTorch Lightning | Custom training loop |
| Config | Hydra YAML | Python dict |
| Data Format | PLY | LAS/LAZ (via laspy) |
| NAG | Custom Data class | SuperpointHierarchy |
| Partitioning | Cut-Pursuit (C++) | Greedy (Python/CUDA) |

## Testing

Run tests with:

```bash
# 基础测试
python -m pytest tests/test_ezsp/test_ezsp.py -v

# 综合测试
python -m pytest tests/test_ezsp/test_ezsp_comprehensive.py -v

# 快速验证
python tests/test_ezsp_fixes.py
```

## Training Commands

```bash
# Stage 1: Partition Learning
python tools/train.py --config-file configs/dales/semseg-ezsp-v1-0.py

# Stage 2: Set training_partition_stage=False in config, then:
python tools/train.py --config-file configs/dales/semseg-ezsp-v1-0.py
```

## Citation

```bibtex
@article{robert2023superpoint,
  title={Superpoint Transformer for 3D Scene Instance Segmentation},
  author={Robert, Damien and Raguet, Hugo and Landrieu, Loic},
  journal={arXiv preprint arXiv:2306.08045},
  year={2023}
}
```

## License

This implementation follows the same license as the original Superpoint Transformer project.
