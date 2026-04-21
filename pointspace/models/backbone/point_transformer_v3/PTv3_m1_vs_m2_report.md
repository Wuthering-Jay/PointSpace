# Point Transformer V3 m1 vs m2 对比报告

本文对 `pointspace/models/backbone/point_transformer_v3` 下的两种实现进行对比：

- `point_transformer_v3m1_base.py`
- `point_transformer_v3m2_sonata.py`

它们都属于 Point Transformer V3 家族，整体骨架一致，但在输入预处理、分层下采样、归一化策略、解码方式和训练相关能力上有明显差异。

## 1. 共同点

两者都遵循类似的 backbone 主线：

1. 将输入 `data_dict` 转成 `Point`
2. 做点云序列化或网格化相关预处理
3. 先做 `Embedding`
4. 经过多 stage 编码器
5. 如果启用 decoder，再逐级上采样恢复点级特征
6. 输出仍然是 `Point`，特征写回 `point.feat`

共同的核心 block 也很接近，都是：

- `cpe`：上下文位置编码卷积
- `self-attention`：基于 serialized patch 的注意力
- `mlp`
- residual + `DropPath`

也就是说，两者的“注意力 block”思想基本一致，主要区别在于外围的数据组织方式。

## 2. PT-v3m1 处理流程

对应实现：[`point_transformer_v3m1_base.py`](./point_transformer_v3m1_base.py)

### 2.1 输入与序列化

`forward()` 的流程是：

1. `point = Point(data_dict)`
2. `point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)`
3. `point.sparsify()`

这里的关键是 `serialization()`：

- 使用 `grid_coord` / `coord + grid_size`
- 生成 `serialized_code`
- 生成 `serialized_order`
- 生成 `serialized_inverse`
- 让后续 attention 在“序列 patch”上运行

`point.sparsify()` 则把 `Point` 变成 `spconv.SparseConvTensor` 相关的稀疏结构，供 `spconv.SubMConv3d` 使用。

### 2.2 Embedding

`Embedding` 使用：

- `spconv.SubMConv3d(in_channels -> embed_channels, kernel_size=5)`
- 可选 `norm_layer`
- 可选 `act_layer`

这说明 m1 的 stem 不只是线性映射，而是带稀疏卷积的局部空间建模。

### 2.3 Encoder

`self.enc` 是一个 `PointSequential`，按 stage 堆叠。

每个 stage 的结构是：

- 第 0 stage：直接进入若干 `Block`
- 后续 stage：先做 `SerializedPooling`
- 再做若干 `Block`

`SerializedPooling` 的特点：

- 基于 `serialized_code` 和位移右移 `>> pooling_depth`
- 支持 `stride = 2^k`
- 用 `torch_scatter.segment_csr` 做 reduce
- 记录 `pooling_inverse` 和 `pooling_parent`
- 下采样后重新 `serialization()` 和 `sparsify()`

### 2.4 Decoder

如果 `enc_mode=False`，m1 会启用 decoder：

- `SerializedUnpooling`
- 若干 `Block`

`SerializedUnpooling` 的做法是：

- 取出 `pooling_parent`
- 用 `pooling_inverse` 把子层特征回填到父层
- `parent.feat = parent.feat + point.feat[inverse]`

### 2.5 归一化与训练策略

m1 最突出的特点是它支持 `PDNorm`：

- `pdnorm_bn`
- `pdnorm_ln`
- `pdnorm_decouple`
- `pdnorm_adaptive`
- `pdnorm_affine`
- `pdnorm_conditions`

这说明 m1 更偏向“跨数据集/跨条件”的规范化适配。

另外 m1 还支持：

- `enable_flash`
- `enable_rpe`
- `upcast_attention`
- `upcast_softmax`
- `pre_norm`
- `enc_mode`

但没有 m2 那种显式的 `LayerScale` 和 `freeze_encoder` 逻辑。

## 3. PT-v3m2 处理流程

对应实现：[`point_transformer_v3m2_sonata.py`](./point_transformer_v3m2_sonata.py)

### 3.1 输入与序列化

`forward()` 的流程是：

1. `point = Point(data_dict)`
2. `point = self.embedding(point)`
3. `point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)`
4. `point.sparsify()`
5. `point = self.enc(point)`
6. 如启用 decoder，则 `point = self.dec(point)`

和 m1 一样，m2 也依赖 `serialization()` 和 `sparsify()`，但其周边的 pooling/unpooling 逻辑已经从“serialized code 变换”转向了更直接的 `grid_coord` 操作。

### 3.2 Embedding

m2 的 `Embedding` 更轻：

- `nn.Linear(in_channels, embed_channels)`
- 可选 `norm_layer`
- 可选 `act_layer`
- 支持 `mask_token`

这说明 m2 在输入 stem 上更接近纯 token 化，而不是引入稀疏卷积 stem。

### 3.3 Encoder

m2 的 encoder 结构和 m1 类似，也是 stage 化堆叠：

- 第 0 stage：直接若干 `Block`
- 后续 stage：先 `GridPooling`
- 再若干 `Block`

`GridPooling` 的特点：

- 优先使用 `grid_coord`
- 若没有，则由 `coord` 和 `grid_size` 推导
- 通过 `torch.div(grid_coord, stride)` 进行下采样
- 用位运算把 batch 信息编码进坐标
- 记录 `pooling_inverse`、`pooling_parent`、`idx_ptr`

相比 m1，m2 的下采样更直接，不依赖 serialized code 的位移语义。

### 3.4 Decoder

m2 的 decoder 使用：

- `GridUnpooling`
- 若干 `Block`

`GridUnpooling` 的逻辑是：

- 取出 `pooling_parent`
- 用 `pooling_inverse` 做索引回填
- `parent.feat = parent.feat + self.proj(point).feat[inverse]`

它和 m1 的 `SerializedUnpooling` 目标一致，但实现路径更简单，依赖的是 grid pooling 的 parent/inverse 关系。

### 3.5 额外能力

m2 相比 m1 增加了几个很明显的工程特性：

- `LayerScale`
- `mask_token`
- `freeze_encoder`
- `_init_weights`

其中：

- `LayerScale` 用于对残差分支做可学习缩放，常用于稳定深层残差训练
- `mask_token` 支持 masked modeling 风格输入
- `freeze_encoder` 允许冻结 embedding 和 encoder
- `_init_weights` 对 `Linear` 和 `SubMConv3d` 做显式初始化

## 4. 核心差异对比

| 维度 | PT-v3m1 | PT-v3m2 |
|---|---|---|
| Stem / Embedding | `spconv.SubMConv3d` + norm + act | `nn.Linear` + norm + act |
| 序列化前处理 | `Point.serialization()` + `Point.sparsify()` | 同样保留 |
| 下采样模块 | `SerializedPooling` | `GridPooling` |
| 上采样模块 | `SerializedUnpooling` | `GridUnpooling` |
| 下采样依据 | serialized code 位移和 patch 结构 | grid coordinate + stride |
| 归一化策略 | 支持 `PDNorm`，更偏多条件/多域 | 主要是 `LayerNorm` |
| 残差稳定性 | 普通 `DropPath` | `DropPath + LayerScale` |
| 训练增强能力 | 偏基础 backbone | 支持 `mask_token`、`freeze_encoder` |
| 初始化 | 以框架默认行为为主 | 显式 `_init_weights` |
| 工程定位 | 更接近原始 V3 设计与多域适配 | 更偏简化、工程化、易控制 |

## 5. 结构层面的理解

### 5.1 m1 更像“serialized sparse transformer”

m1 的主线可以理解为：

- 点云先序列化
- 再在 serialized patch 上做 attention
- 用 serialized pooling/unpooling 维护多尺度结构
- 同时引入 `spconv` 语义，使局部空间建模更强

它更强调“序列化表示”和“稀疏卷积表示”的联合。

### 5.2 m2 更像“grid-based point transformer”

m2 的主线可以理解为：

- 点云进入 embedding 后
- 通过 grid pooling 做分层
- attention 仍然依赖 serialized patch
- 但跨层结构更直接地由 grid 关系维护

它更强调工程简洁性和训练可控性。

## 6. 实际使用建议

- 如果你更关注原始 V3 思路、多条件归一化、与现有预训练/域适配策略对齐，优先看 `PT-v3m1`
- 如果你更关注简化结构、显式初始化、可冻结 encoder、mask token 这类工程能力，优先看 `PT-v3m2`
- 如果要做新 backbone 变体，m2 更容易改参数，m1 更适合做“研究版基线”

## 7. 结论

这两个 PT-V3 网络的共同骨架非常接近，都是“serialized attention + hierarchical pooling/unpooling”的 point transformer。

真正的差异主要集中在：

1. 输入 stem 的实现方式
2. 层级下采样/上采样的组织方式
3. 归一化与训练稳定性机制
4. 是否提供额外训练控制能力

如果把它们抽象成一句话：

- `PT-v3m1` 更偏“原生 V3 + 多域适配”
- `PT-v3m2` 更偏“工程化 V3 + 结构简化 + 训练可控”

