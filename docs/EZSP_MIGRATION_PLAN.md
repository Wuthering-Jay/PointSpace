# EZ-SP 迁移到 PointSpace 完整计划

> 版本: 1.0
> 日期: 2026-03-28
> 状态: 规划阶段

---

## 一、项目概述

### 1.1 目标

将 Superpoint Transformer 团队的 EZ-SP（Easy Superpoints）架构迁移到 PointSpace 框架，形成 PointSpace 风格的可学习端到端超点分割网络。

### 1.2 EZ-SP 核心创新

| 特性 | 传统 SPT | EZ-SP |
|------|----------|-------|
| 分区算法 | Cut-Pursuit (CPU) | Greedy Contour Prior (GPU) |
| 特征来源 | 手工特征 | CNN 学习特征 |
| 训练方式 | 预处理分区 | 端到端两阶段 |
| 速度 | 慢 (CPU瓶颈) | 快 (全GPU) |

### 1.3 两阶段训练流程

```
┌─────────────────────────────────────────────────────────────────┐
│                     第一阶段: 分区学习                            │
├─────────────────────────────────────────────────────────────────┤
│  Input (coord + handcraft features)                             │
│      ↓                                                          │
│  SparseCNN (3层稀疏卷积, spconv)                                 │
│      ↓                                                          │
│  点嵌入 (32维特征)                                               │
│      ↓                                                          │
│  PartitionCriterion (边分类对比损失)                             │
│      - 同类点对: 高亲和力 (target=1)                             │
│      - 异类点对: 低亲和力 (target=0)                             │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     第二阶段: 语义分割                            │
├─────────────────────────────────────────────────────────────────┤
│  预训练 SparseCNN (冻结或微调)                                   │
│      ↓                                                          │
│  点嵌入 (32维特征)                                               │
│      ↓                                                          │
│  GreedyContourPriorPartition (GPU分区)                          │
│      ↓                                                          │
│  SuperpointHierarchy (多层级超点图)                              │
│      ↓                                                          │
│  Superpoint Transformer                                         │
│      ↓                                                          │
│  语义标签                                                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 二、架构设计

### 2.1 关键设计决策

#### ⚠️ 核心约束: 数据流时序

```
❌ 错误设计 (时间悖论):
   DataLoader (CPU) → GreedyPartition(需要特征!) → Model (GPU)
                            ↑
                       特征还没提取! 死锁!

✅ 正确设计:
   DataLoader (CPU) → 原始数据 → Model.forward (GPU)
                                       ↓
                                 SparseCNN → 特征
                                       ↓
                                 GreedyPartition (nn.Module)
                                       ↓
                                 NAG → Transformer
```

**结论**: `GreedyContourPriorPartition` 必须是 `nn.Module`，在 `model.forward()` 中执行，不能是 `Transform`。

### 2.2 目录结构

```
pointspace/
├── models/
│   ├── backbone/
│   │   └── ezsp/                           # EZ-SP 核心模块
│   │       ├── __init__.py
│   │       ├── sparse_cnn.py               # [M1] SparseCNN (spconv版)
│   │       ├── graph_norm.py               # [M2] GraphNorm
│   │       ├── graph_partition.py          # [M3] GreedyContourPriorPartition
│   │       ├── superpoint_hierarchy.py     # [M4] NAG等效结构
│   │       └── spt_transformer.py          # [M5] Transformer stages
│   │
│   ├── losses/
│   │   ├── binary_focal.py                 # [L1] BinaryFocalLoss
│   │   └── partition_criterion.py          # [L2] PartitionCriterion
│   │
│   └── segmentor/
│       └── ezsp_segmentor.py               # [S1] EZSPPartitionSegmentor
│
├── datasets/
│   └── (无新增, KNN图构建已移至GPU端)
│
├── tests/
│   └── test_ezsp/                          # 测试套件
│       ├── __init__.py
│       ├── test_sparse_cnn.py
│       ├── test_graph_norm.py
│       ├── test_graph_partition.py
│       ├── test_superpoint_hierarchy.py
│       ├── test_partition_criterion.py
│       └── test_full_pipeline.py
│
└── configs/
    └── ezsp/                               # 配置文件
        ├── _base_/
        │   └── ezsp_base.py
        ├── partition/                      # 第一阶段
        │   └── ezsp_partition_s3dis.py
        └── semantic/                       # 第二阶段
            └── ezsp_semseg_s3dis.py
```

### 2.3 模块依赖图

```
                    ┌─────────────────┐
                    │   外部依赖       │
                    │ torch-graph-    │
                    │ components      │
                    └────────┬────────┘
                             │
    ┌────────────────────────┼────────────────────────┐
    │                        │                        │
    ▼                        ▼                        ▼
┌───────┐              ┌───────────┐            ┌───────────┐
│  L1   │              │    M2     │            │    M4     │
│Binary │              │ GraphNorm │            │Superpoint │
│Focal  │              │           │            │Hierarchy  │
└───┬───┘              └─────┬─────┘            └─────┬─────┘
    │                        │                        │
    ▼                        ▼                        │
┌───────┐              ┌───────────┐                  │
│  L2   │              │    M1     │                  │
│Partit.│              │SparseCNN  │                  │
│Criter.│              │           │                  │
└───┬───┘              └─────┬─────┘                  │
    │                        │                        │
    │                        ▼                        │
    │                  ┌───────────┐                  │
    │                  │    M3     │◄─────────────────┘
    │                  │GreedyPart │
    │                  │           │
    │                  └─────┬─────┘
    │                        │
    │    ┌───────────────────┼───────────────────┐
    │    │                   │                   │
    │    │                   ▼                   ▼
    │    │             ┌───────────┐       ┌───────────┐
    │    │             │    M5     │       │    S1     │
    │    │             │   SPT     │◄──────│  EZSP     │◄──┐
    │    │             │Transformer│       │Segmentor  │   │
    │    │             └───────────┘       └───────────┘   │
    │    │                                       ▲         │
    │    │  (KNN图构建已内置于M3)                  │         │
    └────┴───────────────────────────────────────┘         │
                                                           │
                                              configs ─────┘
```

---

## 三、模块详细设计

### 3.1 [M1] SparseCNN - 稀疏卷积特征提取器

**文件**: `pointspace/models/backbone/ezsp/sparse_cnn.py`

**功能**: 使用 spconv 提取点云特征嵌入

**原始实现** (torchsparse):
```python
# 原始 SPT 实现
class SparseCNN(nn.ModuleList):
    def __init__(self, cnn=[dim_hf, 32, 32, 32], kernel_size=3, ...):
        for i in range(1, len(cnn)):
            block = ConvBlock(cnn[i-1], cnn[i], kernel_size, ...)
            self.append(block)
```

**PointSpace 实现**:
```python
@MODELS.register_module("EZ-SparseCNN")
class SparseCNN(PointModule):
    """
    EZ-SP 稀疏CNN特征提取器 (spconv版)

    架构: input_dim → [32 → 32 → 32]
    每个块: SubMConv3d → GraphNorm → ReLU → (可选残差)

    参数:
        in_channels: int - 输入通道数 (手工特征维度)
        channels: List[int] - 各层通道数, 默认 [32, 32, 32]
        kernel_size: int - 卷积核大小, 默认 3
        dilation: int - 膨胀率, 默认 1
        norm: str - 归一化类型 'gn' (GraphNorm) | 'bn' (BatchNorm)
        activation: str - 激活函数 'relu' | 'leakyrelu'
        residual: bool - 块内残差连接
        global_residual: bool - 全局残差 (输入直接加到输出)

    输入:
        point: Point 对象, 需包含:
            - coord: [N, 3] 坐标
            - feat: [N, C_in] 手工特征
            - grid_coord: [N, 3] 体素化坐标
            - batch: [N] batch索引
            - offset: [B] 累积点数

    输出:
        point: Point 对象, feat 更新为 [N, channels[-1]] 的CNN嵌入
    """

    def __init__(
        self,
        in_channels: int,
        channels: List[int] = [32, 32, 32],
        kernel_size: int = 3,
        dilation: int = 1,
        norm: str = "gn",
        activation: str = "relu",
        residual: bool = True,
        global_residual: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.global_residual = global_residual

        # 输入投影 (如果通道数不匹配)
        if in_channels != channels[0]:
            self.input_proj = nn.Linear(in_channels, channels[0])
        else:
            self.input_proj = nn.Identity()

        # 构建卷积块
        self.blocks = nn.ModuleList()
        prev_ch = channels[0]
        for ch in channels:
            self.blocks.append(
                ConvBlock(
                    in_channels=prev_ch,
                    out_channels=ch,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    norm=norm,
                    activation=activation,
                    residual=residual,
                )
            )
            prev_ch = ch

        # 全局残差投影
        if global_residual and in_channels != channels[-1]:
            self.global_proj = nn.Linear(in_channels, channels[-1])
        elif global_residual:
            self.global_proj = nn.Identity()

    def forward(self, point: Point) -> Point:
        # 保存原始特征 (全局残差用)
        feat_input = point.feat

        # 输入投影
        point.feat = self.input_proj(point.feat)

        # 构建 SparseConvTensor
        point.sparsify()

        # 逐块卷积
        for block in self.blocks:
            point = block(point)

        # 全局残差
        if self.global_residual:
            point.feat = point.feat + self.global_proj(feat_input)

        return point


class ConvBlock(PointModule):
    """单个稀疏卷积块"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        norm: str = "gn",
        activation: str = "relu",
        residual: bool = True,
    ):
        super().__init__()
        self.residual = residual and (in_channels == out_channels)

        # 稀疏卷积
        self.conv = spconv.SubMConv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=dilation * (kernel_size // 2),
            dilation=dilation,
            bias=False,
        )

        # 归一化
        if norm == "gn":
            self.norm = GraphNorm(out_channels)
        else:
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)

        # 激活
        if activation == "relu":
            self.act = nn.ReLU(inplace=True)
        elif activation == "leakyrelu":
            self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, point: Point) -> Point:
        identity = point.sparse_conv_feat.features if self.residual else None

        # 稀疏卷积
        point.sparse_conv_feat = self.conv(point.sparse_conv_feat)
        feat = point.sparse_conv_feat.features

        # 归一化 (GraphNorm需要batch索引)
        if isinstance(self.norm, GraphNorm):
            feat = self.norm(feat, point.batch)
        else:
            feat = self.norm(feat)

        # 激活
        feat = self.act(feat)

        # 残差
        if self.residual:
            feat = feat + identity

        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(feat)
        point.feat = feat
        return point
```

**torchsparse → spconv 关键映射**:

| torchsparse | spconv | 说明 |
|-------------|--------|------|
| `SparseTensor(coords, feats)` | `SparseConvTensor(features, indices, spatial_shape, batch_size)` | 张量构造 |
| `spnn.Conv3d` | `spconv.SubMConv3d` | 子流形卷积 |
| `x.F` | `x.features` | 特征提取 |
| coords: `[N, 4]` = `[batch, x, y, z]` | indices: `[N, 4]` = `[batch, z, y, x]` | **坐标顺序不同!** |
| `x.C` | `x.indices` | 坐标/索引 |

---

### 3.2 [M2] GraphNorm - 图归一化

**文件**: `pointspace/models/backbone/ezsp/graph_norm.py`

**功能**: 按图/batch 独立归一化，适应不同超点大小

**为什么不能用 BatchNorm**:
- 点云中不同超点/簇的点数差异巨大
- BatchNorm 会破坏局部特征分布
- GraphNorm 对每个图独立计算统计量

```python
@MODELS.register_module()
class GraphNorm(nn.Module):
    """
    图归一化层

    对每个图 (batch) 独立计算均值和方差进行归一化

    参数:
        num_features: int - 特征维度
        eps: float - 数值稳定性
        affine: bool - 是否使用可学习参数
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        affine: bool = True,
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, x: Tensor, batch: Tensor) -> Tensor:
        """
        参数:
            x: [N, C] 点特征
            batch: [N] batch索引 (0, 0, ..., 1, 1, ..., B-1)

        返回:
            x_norm: [N, C] 归一化后的特征
        """
        from torch_scatter import scatter_mean

        # 按 batch 计算均值: [B, C]
        mean = scatter_mean(x, batch, dim=0)
        x_centered = x - mean[batch]

        # 按 batch 计算方差: [B, C]
        var = scatter_mean(x_centered ** 2, batch, dim=0)
        std = (var + self.eps).sqrt()
        x_norm = x_centered / std[batch]

        # 可学习参数
        if self.affine:
            x_norm = x_norm * self.weight + self.bias

        return x_norm

    def extra_repr(self) -> str:
        return f'{self.num_features}, eps={self.eps}, affine={self.affine}'
```

---

### 3.3 [M3] GreedyContourPriorPartition - GPU贪婪分区

**文件**: `pointspace/models/backbone/ezsp/graph_partition.py`

**功能**: 基于CNN特征的GPU端贪婪超点分区

**关键**: 这是 `nn.Module`，在 `forward()` 中执行，不是 Transform!

```python
@MODELS.register_module()
class GreedyContourPriorPartition(nn.Module):
    """
    基于轮廓先验的贪婪组件合并分区模块

    核心思想:
        1. 使用 CNN 学习的特征计算边权重 (特征相似度)
        2. 基于能量函数贪婪合并相邻组件
        3. 生成多层级超点图

    能量函数:
        E = Σ_i ||X_i - μ_i||² + reg * Σ_(i,j)∈E w_ij * [μ_i ≠ μ_j]

        - 第一项: 组件内部一致性
        - 第二项: 组件边界平滑性 (轮廓先验)

    参数:
        reg: float | List[float] - 正则化强度, 典型值 2e-2
            越大 → 分区越粗 → 超点越少越大
        min_size: int | List[int] - 各层级最小超点大小, 典型值 [5, 30, 90]
        k_adjacency: int - KNN邻居数 (GPU端构建邻接图)
        spatial_weight: float | None - 空间坐标权重
            None: 纯特征分区 (EZ-SP默认)
            float: x ← [x, spatial_weight * pos]
        edge_weight_mode: str - 边权重计算模式
            'unit': 1 (无权重)
            'exp_neg_latent_distance': exp(-||x_i - x_j|| / d_0)
            'affinity_latent_distance': 亲和力形式
        d_0: float | None - 参考距离, None则自动计算为均值
        w_adjacency: float - 孤立节点新建边的权重
        max_iterations: int - 最大合并迭代次数, -1表示无限制

    输入:
        pos: [N, 3] - 点坐标
        x: [N, C] - CNN提取的点嵌入 (关键!)
        offset: [B] - 累积点数 (用于GPU KNN隔离batch)
        y: [N, num_classes] | None - 可选GT标签直方图

    输出:
        SuperpointHierarchy - 多层级超点图结构

    注意:
        邻接图在forward()中通过GPU KNN动态构建，不需要预先传入edge_index
    """

    _EDGE_WEIGHT_MODES = [
        'unit',
        'inverse_distance',
        'exp_neg_distance',
        'exp_neg_latent_distance',
        'affinity_latent_distance',
    ]

    def __init__(
        self,
        reg: Union[float, List[float]] = 2e-2,
        min_size: Union[int, List[int]] = [5, 30, 90],
        spatial_weight: Optional[float] = None,
        edge_weight_mode: str = 'unit',
        d_0: Optional[float] = None,
        k_adjacency: int = 5,
        w_adjacency: float = 0.0,
        max_iterations: int = -1,
        edge_reduce: str = 'add',
    ):
        super().__init__()

        # 参数标准化为列表
        if isinstance(min_size, list):
            num_levels = len(min_size)
        elif isinstance(reg, list):
            num_levels = len(reg)
        else:
            num_levels = 1

        self.reg = reg if isinstance(reg, list) else [reg] * num_levels
        self.min_size = min_size if isinstance(min_size, list) else [min_size] * num_levels

        assert len(self.reg) == len(self.min_size), \
            f"reg ({len(self.reg)}) 和 min_size ({len(self.min_size)}) 长度必须相同"

        self.spatial_weight = spatial_weight
        self.edge_weight_mode = edge_weight_mode
        self.d_0 = d_0
        self.k_adjacency = k_adjacency
        self.w_adjacency = w_adjacency
        self.max_iterations = max_iterations
        self.edge_reduce = edge_reduce

        assert edge_weight_mode in self._EDGE_WEIGHT_MODES, \
            f"无效的 edge_weight_mode: {edge_weight_mode}, 可选: {self._EDGE_WEIGHT_MODES}"

    def forward(
        self,
        pos: Tensor,
        x: Tensor,
        offset: Tensor,
        y: Optional[Tensor] = None,
    ) -> 'SuperpointHierarchy':
        """
        执行层级分区

        数据流:
            1. GPU KNN 构建邻接图 (使用 pointops, 自动隔离batch)
            2. Level 0 → Level 1 → ... → Level L

            每个Level:
                1. 计算边权重 (基于特征距离)
                2. 可选: 拼接空间坐标到特征
                3. 调用 torch-graph-components 进行贪婪合并
                4. 构建下一层级数据
        """
        device = pos.device
        num_points = pos.shape[0]

        # ========== GPU KNN 构建邻接图 ==========
        from libs.pointops.functions import knn_query
        neighbor_idx, neighbor_dist = knn_query(self.k_adjacency, pos, offset)
        edge_index = self._neighbor_idx_to_edge_index(neighbor_idx)

        # 初始化 Level 0 数据
        data = {
            'pos': pos,
            'x': x,
            'edge_index': edge_index,
            'batch': batch,
            'offset': offset,
            'node_size': torch.ones(num_points, device=device, dtype=torch.long),
            'super_index': None,  # 指向上一层的索引
        }

        # 处理标签 (转为直方图格式)
        if y is not None:
            if y.dim() == 1:
                # 单标签 → 直方图
                num_classes = y.max().item() + 1
                y_hist = torch.zeros(num_points, num_classes, device=device)
                valid_mask = y >= 0
                y_hist[valid_mask] = F.one_hot(y[valid_mask], num_classes).float()
            else:
                y_hist = y
            data['y'] = y_hist

        data_list = [data]

        # 层级分区
        for level, (reg, min_size) in enumerate(zip(self.reg, self.min_size)):
            d = data_list[level]

            # 1. 计算边权重
            edge_attr = self._compute_edge_weights(d['x'], d['edge_index'])
            d['edge_attr'] = edge_attr

            # 2. 可选: 拼接空间坐标
            x_partition = d['x']
            if self.spatial_weight is not None and self.spatial_weight > 0:
                x_partition = torch.cat([
                    x_partition,
                    d['pos'] * self.spatial_weight
                ], dim=1)

            # 3. 调用组件合并
            super_index, merged_data = self._merge_components(
                x=x_partition,
                pos=d['pos'],
                node_size=d['node_size'],
                edge_index=d['edge_index'],
                edge_attr=edge_attr,
                reg=reg,
                min_size=min_size,
                y=d.get('y'),
            )

            # 4. 更新当前层 super_index
            d['super_index'] = super_index
            data_list[level] = d

            # 5. 添加新层级
            data_list.append(merged_data)

        return SuperpointHierarchy(data_list)

    def _compute_edge_weights(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """
        基于特征计算边权重

        参数:
            x: [N, C] 点特征
            edge_index: [2, E] 边索引

        返回:
            edge_attr: [E] 边权重
        """
        src, dst = edge_index[0], edge_index[1]

        if self.edge_weight_mode == 'unit':
            return torch.ones(edge_index.shape[1], device=x.device)

        # 计算特征距离
        latent_dist = (x[src] - x[dst]).norm(dim=1)
        d_0 = self.d_0 if self.d_0 is not None else latent_dist.mean()

        if self.edge_weight_mode == 'exp_neg_latent_distance':
            return torch.exp(-latent_dist / d_0)
        elif self.edge_weight_mode == 'affinity_latent_distance':
            d_neg_exp = torch.exp(-latent_dist / d_0)
            eps = 1e-6
            return d_neg_exp / (1 - d_neg_exp + eps)
        elif self.edge_weight_mode == 'inverse_distance':
            return 1 / (1 + latent_dist / d_0)
        else:
            raise ValueError(f"未知的 edge_weight_mode: {self.edge_weight_mode}")

    def _merge_components(
        self,
        x: Tensor,
        pos: Tensor,
        node_size: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        reg: float,
        min_size: int,
        y: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict]:
        """
        调用 torch-graph-components 进行组件合并

        返回:
            super_index: [N] 每个点所属的超点ID
            merged_data: dict 新层级的数据
        """
        from torch_graph_components import merge_components_by_contour_prior
        from torch_scatter import scatter_sum, scatter_mean

        device = x.device
        num_nodes = x.shape[0]

        # 初始化: 每个点是一个组件
        # 调用核心合并算法
        super_index, X_merged, S_merged, E_merged, W_merged = \
            merge_components_by_contour_prior(
                X=x,
                S=node_size.float(),
                E=edge_index,
                W=edge_attr * reg,
                min_size=min_size,
                k=self.k_adjacency,
                w_adjacency=self.w_adjacency,
                max_iterations=self.max_iterations,
                edge_reduce=self.edge_reduce,
            )

        # 计算新层级的位置 (加权平均)
        P_merged = scatter_mean(
            pos * node_size.unsqueeze(-1).float(),
            super_index,
            dim=0,
        ) / S_merged.unsqueeze(-1).clamp(min=1)

        # 构建 Cluster 对象 (子点索引)
        from pointspace.models.backbone.ezsp.superpoint_hierarchy import Cluster
        sub = Cluster.from_super_index(super_index, num_nodes)

        # 聚合标签
        y_merged = None
        if y is not None:
            y_merged = scatter_sum(y, super_index, dim=0)

        merged_data = {
            'pos': P_merged,
            'x': X_merged,
            'node_size': S_merged.long(),
            'edge_index': E_merged,
            'edge_attr': W_merged,
            'sub': sub,
            'super_index': None,
            'y': y_merged,
        }

        return super_index, merged_data

    def _neighbor_idx_to_edge_index(self, neighbor_idx: Tensor) -> Tensor:
        """
        将 KNN 邻居索引转换为 edge_index 格式

        参数:
            neighbor_idx: [N, K] 每个点的K个邻居索引

        返回:
            edge_index: [2, N*K] 边索引 (去除无效边 -1)
        """
        N, K = neighbor_idx.shape
        device = neighbor_idx.device

        # 构建源节点索引 [N, K]
        src = torch.arange(N, device=device).unsqueeze(1).expand(N, K)

        # 展平
        src = src.reshape(-1)  # [N*K]
        dst = neighbor_idx.reshape(-1)  # [N*K]

        # 过滤无效边 (neighbor_idx == -1 表示无效)
        valid_mask = dst >= 0
        src = src[valid_mask]
        dst = dst[valid_mask]

        # 组合为 edge_index
        edge_index = torch.stack([src, dst], dim=0)

        return edge_index
```

---

### 3.4 [M4] SuperpointHierarchy - 层级超点图结构

**文件**: `pointspace/models/backbone/ezsp/superpoint_hierarchy.py`

**功能**: NAG (Nested Attributed Graph) 的 PointSpace 等效实现

```python
class Cluster:
    """
    CSR格式的簇成员索引

    存储每个超点包含的子点索引，使用CSR格式高效存储

    属性:
        pointer: [num_clusters + 1] 每个簇的起始位置
        value: [total_points] 所有子点索引

    示例:
        超点0包含点 [0, 2, 5]
        超点1包含点 [1, 3, 4]
        pointer = [0, 3, 6]
        value = [0, 2, 5, 1, 3, 4]
    """

    def __init__(self, pointer: Tensor, value: Tensor):
        self.pointer = pointer
        self.value = value

    @classmethod
    def from_super_index(cls, super_index: Tensor, num_points: int) -> 'Cluster':
        """从 super_index 构建 Cluster"""
        device = super_index.device
        num_clusters = super_index.max().item() + 1

        # 计算每个簇的大小
        sizes = torch.zeros(num_clusters, dtype=torch.long, device=device)
        sizes.scatter_add_(0, super_index, torch.ones_like(super_index))

        # 构建 pointer
        pointer = torch.zeros(num_clusters + 1, dtype=torch.long, device=device)
        pointer[1:] = sizes.cumsum(0)

        # 构建 value (排序后的点索引)
        sorted_indices = super_index.argsort()
        value = sorted_indices

        return cls(pointer, value)

    def __getitem__(self, idx: int) -> Tensor:
        """获取第 idx 个簇的成员"""
        start = self.pointer[idx].item()
        end = self.pointer[idx + 1].item()
        return self.value[start:end]

    @property
    def num_clusters(self) -> int:
        return len(self.pointer) - 1

    def to(self, device) -> 'Cluster':
        return Cluster(self.pointer.to(device), self.value.to(device))


class SuperpointLevel(dict):
    """
    单个层级的超点数据

    属性:
        pos: [N_l, 3] 超点位置
        x: [N_l, C] 超点特征
        node_size: [N_l] 超点包含的点数
        edge_index: [2, E_l] 超点图边
        edge_attr: [E_l] 边属性
        super_index: [N_l] 指向上一层的映射 (如果有)
        sub: Cluster 子点索引
        y: [N_l, num_classes] 标签直方图 (可选)
    """

    @property
    def num_points(self) -> int:
        return self['pos'].shape[0]

    @property
    def num_edges(self) -> int:
        return self['edge_index'].shape[1] if 'edge_index' in self else 0


class SuperpointHierarchy:
    """
    层级超点图结构 (NAG等效)

    存储多层级的超点图数据，从原始点 (Level 0) 到最粗粒度超点 (Level L)

    属性:
        levels: List[SuperpointLevel] - 各层级数据
        num_levels: int - 层级数量

    层级关系:
        Level 0: 原始点
        Level 1: 第一层超点 (由 Level 0 点聚合)
        Level 2: 第二层超点 (由 Level 1 超点聚合)
        ...
        Level L: 最粗粒度超点

    索引映射:
        super_index[l]: Level l 的每个元素所属的 Level l+1 超点
        sub[l]: Level l 的每个超点包含的 Level l-1 元素
    """

    def __init__(self, data_list: List[Dict]):
        self.levels = [SuperpointLevel(d) for d in data_list]

    @property
    def num_levels(self) -> int:
        return len(self.levels)

    def __getitem__(self, idx: int) -> SuperpointLevel:
        return self.levels[idx]

    def __len__(self) -> int:
        return self.num_levels

    @property
    def device(self):
        return self.levels[0]['pos'].device

    def to(self, device) -> 'SuperpointHierarchy':
        """移动到指定设备"""
        new_levels = []
        for level in self.levels:
            new_level = {}
            for k, v in level.items():
                if isinstance(v, Tensor):
                    new_level[k] = v.to(device)
                elif isinstance(v, Cluster):
                    new_level[k] = v.to(device)
                else:
                    new_level[k] = v
            new_levels.append(new_level)
        return SuperpointHierarchy(new_levels)

    def get_level_ratios(self) -> List[float]:
        """计算各层级的压缩比"""
        ratios = []
        for i in range(1, self.num_levels):
            ratio = self.levels[i-1].num_points / max(self.levels[i].num_points, 1)
            ratios.append(ratio)
        return ratios

    def propagate_labels_to_points(self, level_preds: Tensor, level: int = -1) -> Tensor:
        """
        将超点预测传播回原始点

        参数:
            level_preds: [N_l, C] level层的预测
            level: int 预测所在层级, -1表示最后一层

        返回:
            point_preds: [N_0, C] 原始点的预测
        """
        if level < 0:
            level = self.num_levels + level

        preds = level_preds

        # 从 level 逐层向下传播
        for l in range(level, 0, -1):
            super_index = self.levels[l-1]['super_index']
            preds = preds[super_index]

        return preds
```

---

### 3.5 [L1] BinaryFocalLoss - 二分类Focal Loss

**文件**: `pointspace/models/losses/binary_focal.py`

```python
@LOSSES.register_module()
class BinaryFocalLoss(nn.Module):
    """
    二分类 Focal Loss

    L = -α * (1-p)^γ * log(p)        (正样本)
    L = -(1-α) * p^γ * log(1-p)      (负样本)

    参数:
        gamma: float - 聚焦参数, 默认 2.0
            越大 → 对难样本关注越多
        alpha: float - 正样本权重, 默认 0.25
        reduction: str - 'mean' | 'sum' | 'none'
        loss_weight: float - 损失权重
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.25,
        reduction: str = 'mean',
        loss_weight: float = 1.0,
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        参数:
            pred: [N] 预测logits
            target: [N] 二分类标签 (0或1)

        返回:
            loss: scalar
        """
        pred_prob = torch.sigmoid(pred)

        # 计算 focal weight
        p_t = pred_prob * target + (1 - pred_prob) * (1 - target)
        focal_weight = (1 - p_t) ** self.gamma

        # 计算 alpha weight
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)

        # BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(
            pred, target.float(), reduction='none'
        )

        # Focal loss
        loss = alpha_t * focal_weight * bce_loss

        if self.reduction == 'mean':
            return loss.mean() * self.loss_weight
        elif self.reduction == 'sum':
            return loss.sum() * self.loss_weight
        else:
            return loss * self.loss_weight
```

---

### 3.6 [L2] PartitionCriterion - 分区损失

**文件**: `pointspace/models/losses/partition_criterion.py`

```python
@LOSSES.register_module()
class PartitionCriterion(nn.Module):
    """
    分区学习的边分类损失

    核心思想:
        将分区问题转化为边分类问题
        - INTER_EDGE (跨类边): target=0, 应该分离
        - INTRA_EDGE (同类边): target=1, 应该聚合

    亲和力计算:
        affinity = exp(-||X_i - X_j|| / temperature)

    损失函数:
        loss = BinaryFocalLoss(affinity, target)

    参数:
        loss_function: dict - 损失函数配置 (BinaryFocalLoss)
        temperature: float - 亲和力温度参数
        adaptive_sampling: bool - 是否自适应采样
        adaptive_sampling_ratio: float - 采样比例
        num_classes: int - 语义类别数
        loss_weight: float - 损失权重
    """

    def __init__(
        self,
        loss_function: dict = None,
        temperature: float = 1.0,
        adaptive_sampling: bool = True,
        adaptive_sampling_ratio: float = 0.9,
        num_classes: int = 13,
        loss_weight: float = 1.0,
    ):
        super().__init__()

        # 损失函数
        if loss_function is None:
            loss_function = dict(type="BinaryFocalLoss", gamma=1.0)
        self.loss_fn = build_criteria(loss_function)

        self.temperature = temperature
        self.adaptive_sampling = adaptive_sampling
        self.adaptive_sampling_ratio = adaptive_sampling_ratio
        self.num_classes = num_classes
        self.loss_weight = loss_weight

    def forward(self, nag: 'SuperpointHierarchy') -> Tuple[Tensor, Dict]:
        """
        参数:
            nag: SuperpointHierarchy 对象
                需要 level[0] 包含:
                - x: [N, C] 点特征
                - edge_index: [2, E] 边索引
                - y: [N, num_classes] 标签直方图

        返回:
            loss: Tensor
            output: dict 包含统计信息
        """
        level0 = nag[0]
        x = level0['x']
        edge_index = level0['edge_index']
        y = level0['y']  # [N, num_classes] 标签直方图

        src, dst = edge_index[0], edge_index[1]

        # 计算边的目标标签
        # 同类边: target=1, 异类边: target=0
        y_src = y[src].argmax(dim=1)
        y_dst = y[dst].argmax(dim=1)
        edge_target = (y_src == y_dst).float()

        # 统计边类型
        n_intra = (edge_target == 1).sum()
        n_inter = (edge_target == 0).sum()

        # 自适应采样 (处理类别不平衡)
        if self.adaptive_sampling and self.training:
            edge_index, edge_target, sample_mask = self._adaptive_sample(
                edge_index, edge_target, n_intra, n_inter
            )
            src, dst = edge_index[0], edge_index[1]

        # 计算亲和力
        feat_dist = (x[src] - x[dst]).norm(dim=1)
        affinity = torch.exp(-feat_dist / self.temperature)

        # 计算损失
        loss = self.loss_fn(affinity, edge_target) * self.loss_weight

        output = {
            'loss': loss,
            'n_intra_edge': n_intra,
            'n_inter_edge': n_inter,
            'mean_affinity_intra': affinity[edge_target == 1].mean() if n_intra > 0 else 0,
            'mean_affinity_inter': affinity[edge_target == 0].mean() if n_inter > 0 else 0,
        }

        return loss, output

    def _adaptive_sample(
        self,
        edge_index: Tensor,
        edge_target: Tensor,
        n_intra: int,
        n_inter: int,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        自适应采样以平衡类别

        策略: 多数类欠采样到少数类的 1/adaptive_sampling_ratio
        """
        device = edge_index.device

        intra_mask = edge_target == 1
        inter_mask = edge_target == 0

        # 计算采样数量
        n_minority = min(n_intra, n_inter)
        n_sample = int(n_minority / self.adaptive_sampling_ratio)

        # 采样
        if n_intra > n_sample:
            intra_indices = intra_mask.nonzero().squeeze(-1)
            keep_intra = intra_indices[torch.randperm(len(intra_indices))[:n_sample]]
            intra_mask = torch.zeros_like(intra_mask)
            intra_mask[keep_intra] = True

        if n_inter > n_sample:
            inter_indices = inter_mask.nonzero().squeeze(-1)
            keep_inter = inter_indices[torch.randperm(len(inter_indices))[:n_sample]]
            inter_mask = torch.zeros_like(inter_mask)
            inter_mask[keep_inter] = True

        sample_mask = intra_mask | inter_mask

        return edge_index[:, sample_mask], edge_target[sample_mask], sample_mask
```

---

### 3.7 [S1] EZSPPartitionSegmentor - 两阶段分割器

**文件**: `pointspace/models/segmentor/ezsp_segmentor.py`

```python
@MODELS.register_module()
class EZSPPartitionSegmentor(nn.Module):
    """
    EZ-SP 两阶段分割器

    第一阶段 (training_partition_stage=True):
        Input → SparseCNN → 点嵌入 → GreedyPartition → PartitionCriterion
        目标: 学习好的点特征使分区边界对齐语义边界

    第二阶段 (training_partition_stage=False):
        Input → 预训练SparseCNN → 点嵌入 → GreedyPartition → Transformer → 语义标签
        目标: 在超点图上进行语义分割

    参数:
        training_partition_stage: bool - 当前训练阶段
        num_classes: int - 语义类别数
        sparse_cnn: dict - SparseCNN配置
        partition_module: dict - GreedyContourPriorPartition配置
        partition_criterion: dict - PartitionCriterion配置 (第一阶段)
        transformer: dict - Transformer配置 (第二阶段)
        criteria: dict - 语义分割损失配置 (第二阶段)
        freeze_cnn: bool - 第二阶段是否冻结CNN
    """

    def __init__(
        self,
        training_partition_stage: bool = True,
        num_classes: int = 13,
        sparse_cnn: dict = None,
        partition_module: dict = None,
        partition_criterion: dict = None,
        transformer: dict = None,
        criteria: dict = None,
        freeze_cnn: bool = True,
    ):
        super().__init__()

        self.training_partition_stage = training_partition_stage
        self.num_classes = num_classes
        self.freeze_cnn = freeze_cnn

        # SparseCNN (两阶段都需要)
        self.sparse_cnn = build_model(sparse_cnn)

        # 分区模块 (两阶段都需要)
        self.partition_module = build_model(partition_module)

        if training_partition_stage:
            # 第一阶段: 分区损失
            self.partition_criterion = build_criteria(partition_criterion)
        else:
            # 第二阶段: Transformer + 语义损失
            self.transformer = build_model(transformer)
            self.criteria = build_criteria(criteria)

            # 冻结CNN
            if freeze_cnn:
                for param in self.sparse_cnn.parameters():
                    param.requires_grad = False

    def forward(self, input_dict: Dict) -> Dict:
        """
        前向传播

        数据流:
            1. input_dict 包含原始 coord, feat, edge_index
            2. SparseCNN 提取点嵌入
            3. GreedyPartition 动态分区
            4. 根据阶段计算损失
        """
        point = Point(input_dict)

        # ========== Step 1: SparseCNN 特征提取 ==========
        point = self.sparse_cnn(point)
        # point.feat: [N, 32] CNN嵌入

        # ========== Step 2: 动态分区 ==========
        # GPU KNN + 贪婪分区 (邻接图在partition_module内部构建)
        nag = self.partition_module(
            pos=point.coord,
            x=point.feat,
            offset=point.offset,
            y=input_dict.get('segment'),
        )

        # ========== Step 3: 根据阶段处理 ==========
        if self.training_partition_stage:
            return self._forward_partition_stage(nag, input_dict)
        else:
            return self._forward_semantic_stage(nag, input_dict)

    def _forward_partition_stage(self, nag: 'SuperpointHierarchy', input_dict: Dict) -> Dict:
        """第一阶段: 分区学习"""
        if self.training:
            loss, partition_output = self.partition_criterion(nag)
            return {
                'loss': loss,
                'n_inter_edge': partition_output['n_inter_edge'],
                'n_intra_edge': partition_output['n_intra_edge'],
            }
        else:
            # 验证时计算 oracle mIoU
            return self._compute_partition_metrics(nag, input_dict)

    def _forward_semantic_stage(self, nag: 'SuperpointHierarchy', input_dict: Dict) -> Dict:
        """第二阶段: 语义分割"""
        # Transformer 在 NAG 上进行分割
        seg_logits = self.transformer(nag)

        # 传播回原始点
        if seg_logits.shape[0] != input_dict['coord'].shape[0]:
            seg_logits = nag.propagate_labels_to_points(seg_logits)

        if self.training:
            loss = self.criteria(seg_logits, input_dict['segment'])
            return {'loss': loss, 'seg_logits': seg_logits}
        elif 'segment' in input_dict:
            loss = self.criteria(seg_logits, input_dict['segment'])
            return {'loss': loss, 'seg_logits': seg_logits}
        else:
            return {'seg_logits': seg_logits}

    def _compute_partition_metrics(self, nag: 'SuperpointHierarchy', input_dict: Dict) -> Dict:
        """计算分区质量指标 (Oracle mIoU)"""
        # Oracle: 每个超点取多数标签
        level1 = nag[1]
        if 'y' not in level1 or level1['y'] is None:
            return {'nag': nag}

        y_hist = level1['y'][:, :self.num_classes]
        y_oracle = y_hist.argmax(dim=1)

        # 传播回原始点
        super_index = nag[0]['super_index']
        y_pred = y_oracle[super_index]
        y_true = input_dict['segment']

        return {
            'nag': nag,
            'y_pred': y_pred,
            'y_true': y_true,
        }
```

---

### ~~3.8 [T1] AdjacencyGraph - 邻接图构建~~ (已删除)

> ⚠️ **设计变更**: 邻接图构建已从 Transform 移至 `GreedyContourPriorPartition.forward()` 中
>
> **原因**:
> 1. **跨Batch幽灵边问题**: 在Transform阶段，coord是展平的，包含整个Batch的点。如果不传入batch向量，knn_graph会产生跨样本的错误边连接
> 2. **性能优化**: CPU端KNN是瓶颈，使用PointSpace已有的CUDA KNN算子 (`libs/pointops`) 可大幅提速
>
> **新方案**: 在 `GreedyContourPriorPartition.forward()` 开头使用GPU KNN:
> ```python
> from libs.pointops.functions import knn_query
>
> def forward(self, pos, x, batch, offset, ...):
>     # GPU KNN (自动通过offset隔离batch)
>     neighbor_idx, neighbor_dist = knn_query(self.k_adjacency, pos, offset)
>     # 转换为 edge_index [2, E] 格式
>     edge_index = self._neighbor_idx_to_edge_index(neighbor_idx)
>     ...
> ```

---

## 四、配置文件设计

### 4.1 基础配置

**文件**: `configs/ezsp/_base_/ezsp_base.py`

```python
# EZ-SP 基础配置

# SparseCNN 默认配置
sparse_cnn = dict(
    type="EZ-SparseCNN",
    in_channels=6,  # RGB + 法向量
    channels=[32, 32, 32],
    kernel_size=3,
    norm="gn",
    activation="relu",
    residual=True,
    global_residual=False,
)

# 分区模块默认配置
partition_module = dict(
    type="GreedyContourPriorPartition",
    reg=2e-2,
    min_size=[5, 30, 90],
    spatial_weight=None,
    edge_weight_mode="unit",
    k_adjacency=5,
)

# 分区损失默认配置
partition_criterion = dict(
    type="PartitionCriterion",
    loss_function=dict(type="BinaryFocalLoss", gamma=1.0),
    temperature=1.0,
    adaptive_sampling=True,
    adaptive_sampling_ratio=0.9,
)
```

### 4.2 第一阶段配置 (分区训练)

**文件**: `configs/ezsp/partition/ezsp_partition_s3dis.py`

```python
_base_ = [
    "../../_base_/default_runtime.py",
    "../_base_/ezsp_base.py",
]

# 数据集
data_root = "data/s3dis"
num_classes = 13

# 模型
model = dict(
    type="EZSPPartitionSegmentor",
    training_partition_stage=True,
    num_classes=num_classes,
    sparse_cnn=dict(
        type="EZ-SparseCNN",
        in_channels=6,
        channels=[32, 32, 32],
    ),
    partition_module=dict(
        type="GreedyContourPriorPartition",
        reg=2e-2,
        min_size=[5, 30, 90],
    ),
    partition_criterion=dict(
        type="PartitionCriterion",
        num_classes=num_classes,
    ),
)

# 数据
data = dict(
    train=dict(
        type="S3DISDataset",
        split="train",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="GridSample", grid_size=0.04, ...),
            # 注意: 不需要 AdjacencyGraph, KNN图在GPU端动态构建
            dict(type="ToTensor"),
            dict(type="Collect", keys=("coord", "grid_coord", "segment"), ...),
        ],
    ),
    val=dict(...),
)

# 优化器
optimizer = dict(type="AdamW", lr=0.002, weight_decay=0.005)
scheduler = dict(type="OneCycleLR", max_lr=0.002, pct_start=0.05)

# 训练
epoch = 100
eval_epoch = 10
```

### 4.3 第二阶段配置 (语义分割)

**文件**: `configs/ezsp/semantic/ezsp_semseg_s3dis.py`

```python
_base_ = [
    "../../_base_/default_runtime.py",
    "../_base_/ezsp_base.py",
]

# 加载第一阶段模型
resume_from = "exp/ezsp_partition_s3dis/best.pth"
load_keys = ["sparse_cnn"]  # 只加载CNN权重

# 模型
model = dict(
    type="EZSPPartitionSegmentor",
    training_partition_stage=False,
    num_classes=13,
    freeze_cnn=True,
    sparse_cnn=dict(...),
    partition_module=dict(...),
    transformer=dict(
        type="SPTransformer",  # 需要实现
        ...
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0),
    ],
)
```

---

## 五、实现计划

### 5.1 依赖管理

**新增依赖**:
```
torch-graph-components  # 核心组件合并算法
```

**现有依赖** (已安装):
```
spconv                  # 稀疏卷积
torch-scatter           # 散射操作
torch-cluster           # KNN图构建
einops                  # 张量操作
```

### 5.2 实现阶段

#### 阶段一: 基础模块 (预计3天)

| 任务 | 文件 | 优先级 | 依赖 |
|------|------|--------|------|
| GraphNorm | `graph_norm.py` | P0 | 无 |
| BinaryFocalLoss | `binary_focal.py` | P0 | 无 |
| SparseCNN | `sparse_cnn.py` | P0 | GraphNorm |
| 单元测试 | `test_*.py` | P0 | 上述模块 |

#### 阶段二: 分区模块 (预计4天)

| 任务 | 文件 | 优先级 | 依赖 |
|------|------|--------|------|
| Cluster | `superpoint_hierarchy.py` | P0 | 无 |
| SuperpointHierarchy | `superpoint_hierarchy.py` | P0 | Cluster |
| GreedyContourPriorPartition | `graph_partition.py` | P0 | SuperpointHierarchy, torch-graph-components, pointops |
| PartitionCriterion | `partition_criterion.py` | P0 | BinaryFocalLoss |
| 单元测试 | `test_*.py` | P0 | 上述模块 |

#### 阶段三: Segmentor集成 (预计2天)

| 任务 | 文件 | 优先级 | 依赖 |
|------|------|--------|------|
| EZSPPartitionSegmentor | `ezsp_segmentor.py` | P0 | 所有上述模块 |
| 配置文件 | `configs/ezsp/` | P1 | Segmentor |
| 端到端测试 | `test_full_pipeline.py` | P0 | Segmentor |

#### 阶段四: Transformer (预计5天)

| 任务 | 文件 | 优先级 | 依赖 |
|------|------|--------|------|
| SPTransformer Encoder | `spt_transformer.py` | P0 | SuperpointHierarchy |
| SPTransformer Decoder | `spt_transformer.py` | P0 | Encoder |
| 第二阶段测试 | `test_*.py` | P0 | Transformer |

### 5.3 测试计划

#### 单元测试

```
tests/test_ezsp/
├── test_graph_norm.py
│   - test_forward_shape
│   - test_per_graph_normalization
│   - test_gradient_flow
│   - test_vs_batchnorm_difference
│
├── test_sparse_cnn.py
│   - test_forward_shape
│   - test_spconv_indices_format
│   - test_residual_connection
│   - test_with_point_object
│
├── test_graph_partition.py
│   - test_forward_with_cnn_features
│   - test_hierarchical_levels
│   - test_edge_weight_modes
│   - test_min_size_constraint
│   - test_isolated_nodes
│
├── test_superpoint_hierarchy.py
│   - test_cluster_from_super_index
│   - test_level_propagation
│   - test_device_transfer
│
├── test_partition_criterion.py
│   - test_edge_classification
│   - test_adaptive_sampling
│   - test_focal_loss_gradient
│
└── test_full_pipeline.py
    - test_stage1_training_step
    - test_stage2_training_step
    - test_inference
    - test_config_loading
```

#### 集成测试

```python
def test_stage1_dataflow():
    """验证: DataLoader → CNN → Partition 数据流正确"""
    pass

def test_stage2_dataflow():
    """验证: DataLoader → CNN → Partition → Transformer 数据流正确"""
    pass

def test_end_to_end_s3dis():
    """S3DIS 数据集端到端测试"""
    pass
```

---

## 六、验收标准

### 6.1 功能验收

- [ ] SparseCNN 输出形状正确 `[N, 32]`
- [ ] GraphNorm 按batch独立归一化
- [ ] GreedyPartition 生成多层级超点图
- [ ] PartitionCriterion 计算边分类损失
- [ ] 第一阶段训练收敛，损失下降
- [ ] 第二阶段训练收敛，mIoU提升

### 6.2 性能验收

- [ ] SparseCNN 推理速度与原 torchsparse 版本相当 (±20%)
- [ ] 分区模块 GPU 利用率 > 80%
- [ ] 内存占用合理 (单卡可训练)

### 6.3 代码质量

- [ ] 所有新代码通过现有测试框架
- [ ] 类型注解完整
- [ ] docstring 规范
- [ ] 遵循 PointSpace 代码风格

---

## 七、风险与缓解

| 风险 | 可能性 | 影响 | 缓解措施 |
|------|--------|------|----------|
| torch-graph-components 安装失败 | 中 | 高 | 提前测试; 准备纯Python备用实现 |
| spconv 坐标格式错误 | 中 | 中 | 详细单元测试; 参考现有 SpUNet |
| 分区算法OOM | 低 | 中 | 实现分批处理; 调整 min_size |
| Transformer 迁移复杂度高 | 高 | 中 | 先完成分区阶段; Transformer 可单独迭代 |

---

## 附录

### A. 参考资料

1. EZ-SP 论文: `reference_code/superpoint_transformer/2512.00385v2.pdf`
2. SPT 官方代码: `reference_code/superpoint_transformer/src/`
3. PointSpace SpUNet: `pointspace/models/sparse_unet/`
4. torch-graph-components: https://github.com/yourrepo/torch-graph-components

### B. 关键文件映射

| SPT 原始文件 | PointSpace 目标文件 |
|--------------|---------------------|
| `src/nn/sparse.py` | `models/backbone/ezsp/sparse_cnn.py` |
| `src/transforms/partition.py` | `models/backbone/ezsp/graph_partition.py` |
| `src/models/semantic.py` | `models/segmentor/ezsp_segmentor.py` |
| `src/loss/partition.py` | `models/losses/partition_criterion.py` |
| `src/data/nag.py` | `models/backbone/ezsp/superpoint_hierarchy.py` |
