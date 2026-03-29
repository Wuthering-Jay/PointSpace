# SPT (Superpoint Transformer) Integration for EZ-SP

## 概述

本文档描述了如何在 PointSpace 的 EZ-SP 架构中使用 SPT (Superpoint Transformer) 网络进行语义分割。

## 架构组件

### 核心模块

所有 SPT 组件位于 `pointspace/models/backbone/ezsp/spt/`:

```
spt/
├── __init__.py          # 模块导出
├── mlp.py              # MLP, FFN, Classifier
├── norm.py             # BatchNorm, UnitSphereNorm, GroupNorm
├── dropout.py          # DropPath (随机深度)
├── utils.py            # 初始化和工具函数
├── attention.py        # SelfAttentionBlock (多头自注意力 + RPE)
├── transformer.py      # TransformerBlock (Pre-norm 残差)
├── pool.py             # 池化操作 (Max, Mean, Attentive)
├── fusion.py           # 特征融合和上采样
├── stage.py            # Stage, DownNFuseStage, UpNFuseStage
└── spt.py              # SPT 主网络 (UNet-like)
```

### 包装器

`ezsp_transformer.py` 提供两个模型：

1. **EZSPTransformer** - 完整的 SPT 网络，支持自定义配置
2. **EZSPTransformerSimple** - 简化版本，用于快速实验

## EZ-SP 两阶段训练流程

### Stage 1: 分区学习 (Partition Learning)

训练 CNN 学习适合超点分区的点特征：

```
输入 → SparseCNN → 点嵌入 → GreedyPartition → PartitionCriterion
```

**配置文件**: `configs/ezsp/ezsp_stage1_*.py`

**训练命令**:
```bash
python tools/train.py --config-file configs/ezsp/ezsp_stage1_scannet.py
```

### Stage 2: 语义分割 (Semantic Segmentation)

使用预训练的 CNN 和 SPT Transformer 进行语义分割：

```
输入 → 预训练 SparseCNN → 点嵌入 → GreedyPartition → SPT Transformer → 语义标签
```

**配置文件**: 
- `configs/ezsp/ezsp_stage2_spt.py` - 完整 SPT (3 down + 2 up stages)
- `configs/ezsp/ezsp_stage2_simple.py` - 简化版 (1 down + 1 up stages)

**训练命令**:
```bash
# 加载 Stage 1 权重
python tools/train.py \
    --config-file configs/ezsp/ezsp_stage2_spt.py \
    --load-from exp/ezsp_stage1/model_best.pth
```

## SPT 网络配置

### 完整配置示例

```python
transformer=dict(
    type="EZSPTransformer",
    num_classes=13,
    in_channels=32,  # 与 SparseCNN 输出匹配
    
    # PointStage (Level-0 处理)
    nano=False,  # 使用 PointStage
    point_mlp=[32, 64],
    point_drop=0.1,
    
    # 下采样阶段 (编码器)
    down_dim=[64, 128, 256],  # 3 个下采样阶段
    down_num_heads=[4, 8, 8],
    down_num_blocks=[2, 2, 2],
    down_ffn_ratio=4,
    
    # 上采样阶段 (解码器)
    up_dim=[128, 64],  # 2 个上采样阶段
    up_num_heads=[8, 4],
    up_num_blocks=[2, 2],
    
    # 特征设置
    use_pos=True,  # 使用归一化位置
    pool="max",    # 最大池化
    fusion="cat",  # 拼接融合
)
```

### 简化配置示例

```python
transformer=dict(
    type="EZSPTransformerSimple",
    num_classes=13,
    in_channels=32,
    hidden_dim=64,
    num_heads=4,
    num_blocks=2,
    use_pos=True,
)
```

## SPT 网络结构

### UNet-like 架构

```
Level 0 (Points)    ────────► PointStage ────────┐
                                                   │
Level 1 (SP)        ────────► DownStage_1 ───────┼──► UpStage_1 ──► 输出
                                   │              │        ▲
Level 2 (SP)        ────────► DownStage_2 ───────┼────────┘
                                   │              │
Level 3 (SP)        ────────► DownStage_3 ───────┘
```

### 关键特性

1. **多尺度处理**: 在不同分辨率的超点层级上操作
2. **图注意力**: 使用相对位置编码 (RPE) 的自注意力机制
3. **跳跃连接**: UNet 风格的编码器-解码器结构
4. **特征融合**: 支持拼接、加法等融合方式
5. **池化策略**: 支持 Max、Mean、Attentive 等池化

## 相对位置编码 (RPE)

SPT 支持从边特征和节点特征差异计算 RPE：

```python
# 从边特征计算 RPE
k_rpe=True,   # Key 的 RPE
q_rpe=True,   # Query 的 RPE
v_rpe=False,  # Value 的 RPE

# 从节点特征差异计算 RPE
k_delta_rpe=False,  # Key 的 delta RPE
q_delta_rpe=False,  # Query 的 delta RPE
```

### RPE 维度约定（与官方 SPT/EZ-SP 对齐）

- `edge_attr`（水平边）用于 Transformer Self-Attention 的 RPE，维度固定为 **18**：
  `mean_off(3) + std_off(3) + mean_dist(1) + angle_source(1) + angle_target(1) + centroid_dir(3) + centroid_dist(1) + normal_angle(1) + log_length(1) + log_surface(1) + log_volume(1) + log_size(1)`

- `v_edge_attr`（垂直边，child->parent）用于 AttentivePool 的 RPE，维度固定为 **9**：
  `centroid_dir(3) + centroid_dist(1) + normal_angle(1) + log_length(1) + log_surface(1) + log_volume(1) + log_size(1)`

- SPT 参数建议：
  - `in_rpe_dim=18`：用于 Self-Attention（水平边）
  - AttentivePool 的 `in_rpe_dim=9`：用于垂直边

注意：官方 EZ-SP 配置中常见 `v_edge_hf=[]`，即默认不启用垂直边 RPE；PointSpace 中仍提供完整 9 维 `v_edge_attr` 以保证与 SPT 能力对齐并支持后续实验。

## 注意事项

### 1. 内存使用

- 完整 SPT 网络内存占用较大
- 对于大场景，考虑：
  - 减少 `down_num_blocks` 和 `up_num_blocks`
  - 降低 `down_dim` 维度
  - 使用简化配置 `EZSPTransformerSimple`

### 2. 分区参数

`partition_module` 的 `min_size` 参数影响超点数量：
- 较小的 `min_size` → 更多超点 → 更高分辨率 → 更大内存
- 较大的 `min_size` → 更少超点 → 更低分辨率 → 更小内存

推荐设置: `min_size=[5, 30, 90]`

### 3. 特征维度匹配

确保维度匹配：
- `transformer.in_channels` = `sparse_cnn` 输出通道数
- `point_mlp[-1]` = `down_dim[0]`
- 融合时: `up_in_mlp` 需要处理拼接后的维度

### 4. Stage 1 权重加载

Stage 2 训练前必须加载 Stage 1 预训练的 CNN 权重：

```python
# 在配置文件中设置
load_from = "exp/ezsp_stage1/model_best.pth"

# 或使用命令行参数
--load-from exp/ezsp_stage1/model_best.pth
```

## 性能优化建议

### 训练速度

1. **减少 Transformer blocks**: `down_num_blocks=[1, 1, 1]`
2. **使用简单配置**: `EZSPTransformerSimple`
3. **减少注意力头**: `down_num_heads=[2, 4, 4]`

### 精度提升

1. **增加 Transformer blocks**: `down_num_blocks=[3, 3, 3]`
2. **使用 RPE**: `k_rpe=True, q_rpe=True`
3. **更大的特征维度**: `down_dim=[128, 256, 512]`
4. **数据增强**: 旋转、缩放、弹性变形

## 调试技巧

### 1. 检查超点层级

```python
# 在 forward 中打印超点信息
for i in range(nag.num_levels):
    level = nag[i]
    print(f"Level {i}: {level['pos'].shape[0]} nodes")
```

### 2. 可视化注意力

```python
# 在 SelfAttentionBlock 中保存注意力权重
self.attn_weights = attn  # [E, H]
```

### 3. 检查特征维度

```python
# 在 Stage forward 中打印特征形状
print(f"Input: {x.shape}, Output: {x.shape}")
```

## 参考

- [Superpoint Transformer Paper](https://arxiv.org/abs/2306.08045)
- [Official SPT Repository](https://github.com/drprojects/superpoint_transformer)
- PointSpace EZ-SP 实现: `pointspace/models/backbone/ezsp/`

## 版本历史

- v1.0 (2024-03): 初始版本，完整 SPT 组件迁移
- v1.1 (2024-03): 添加 EZSPTransformer 和 EZSPTransformerSimple 包装器
- v1.2 (2024-03): 完善配置文件和文档
