# 面向机载激光雷达的层级化 DINO Patch 蒸馏设计报告

## 1. 文档状态

- 状态：设计完成，尚未实现
- 目标工程：PointSpace
- 目标数据集：`LasImageDataset`
- 目标三维骨干：PTV3、LitePT
- 二维教师：预计算 DINOv3 ViT-L/16 patch 特征
- 推荐方法名称：Hierarchical Patch-Set Distillation（HPSD）

本文档给出一套将预计算 DINO 特征蒸馏到机载激光雷达三维网络中的完整工程方案。方案保留 DiTR Distillation 的核心训练边界：二维教师只在无标签预训练阶段出现，预训练结束后丢弃跨模态投影头，下游训练和推理只保留三维 backbone。同时吸收 Concerto/UTONIA 在中间层级开展二维—三维对齐的优点，并重点解决 coarse 3D token 可能覆盖多个 DINO patch、逐点输出 1024 维特征显存开销过大、正射影像只观察表面而点云包含遮挡点等问题。

## 2. 设计目标与非目标

### 2.1 设计目标

1. 让 PTV3 和 LitePT 使用同一套蒸馏 wrapper、数据字段和损失实现。
2. 在与 DINO patch 物理尺度接近的三维层级上进行监督，而不是固定在输入点层。
3. 完整保留一个 3D token 对应多个 DINO patch、一个 patch 对应多个 3D token 的关系。
4. 避免为每个输入点生成 1024 维预测，显著降低蒸馏分支的激活显存。
5. 使用现有 `LasImageDataset` 中预计算的 Safetensors 特征，不在训练过程中运行 DINO。
6. 蒸馏完成后可以无歧义地迁移 `backbone.*` 权重到普通语义分割模型。
7. 正确处理无影像覆盖、正射不可见点、空监督样本、重叠 tile 和不等长 batch。

### 2.2 非目标

第一版不复制 Concerto/UTONIA 的完整 EMA 3D teacher、prototype、Sinkhorn、global/local view 和 mask prediction 系统。HPSD 首先作为独立的 DINO-to-LiDAR distiller，用最少变量验证跨模态蒸馏本身。后续可把 HPSD loss 作为 UTONIA 的附加目标，但不应将两者在第一版中强耦合。

第一版也不实现推理期图像注入。它只对应 DiTR 的 Distillation 路线，不对应 Injection 路线。

## 3. 现有方法评估

### 3.1 DiTR Distillation

DiTR 的 `DefaultDistiller` 在线提取 DINO 特征，把每个可见点映射到一个 DINO patch，然后对最终逐点三维特征执行 `Linear(C3D, C2D)`，使用逐点余弦损失监督。预训练结束后丢弃投影头，将 backbone checkpoint 加载到普通分割模型。

这一设计的优点是结构简单、教师和 student 边界清楚、下游推理完全不依赖图像。主要问题是监督单位和尺度不匹配：DINO patch 覆盖一块二维区域，而最终逐点特征对应的空间支持更小；同一 patch 内大量点收到完全相同的目标，形成重复监督；输出 `[N, 1024]` 的 student prediction 会产生较大激活和梯度显存。

此外，DiTR 的通用多相机代码按点选择 patch。在单张正射影像中不存在视角选择问题，但植被下地面、立面和内部回波仍可能被错误地视为与正射 patch 对应。

### 3.2 Concerto/UTONIA 二维分支

Concerto/UTONIA 允许通过 `enc2d_upcast_level` 控制在哪个三维层级与二维特征对齐，并利用 pooling trace 将 correspondence 池化到目标层级。随后，它把共享同一 DINO patch 的三维特征通过 `scatter_mean` 聚合，再计算 patch 级余弦损失。

这个方向比逐点蒸馏更合理，但现有 `pool_corr` 会在三维 pooling 时平均 patch row/col。若一个 coarse token 覆盖多个 patch，平均坐标最终只对应一个 patch，完整的 patch 集合及其覆盖比例会丢失。平均值还可能落入原本没有点对应的 patch。

### 3.3 HPSD 的取舍

HPSD 保留 DiTR 的单 student、蒸馏头可丢弃和纯三维推理，保留 Concerto 的层级监督和 patch 级聚合，但不平均 correspondence 坐标。它显式构建 token-patch 稀疏二部图，并以 DINO patch 为监督单位。

| 项目 | DiTR | Concerto/UTONIA | HPSD |
| --- | --- | --- | --- |
| 三维监督位置 | 最终逐点输出 | 可配置中间层 | 统一 `distill_level` |
| 监督单位 | 点 | patch | patch |
| 多 patch 关系 | 不适用 | correspondence 坐标平均 | 完整稀疏边 |
| Student 高维输出 | `[N,1024]` | 聚合后输出 | `[E,d]` 或 `[U,d]` |
| DINO 计算 | 在线 | 在线 | 预计算 Safetensors |
| 三维 teacher | 无 | EMA/offline teacher | 默认无 |
| 推理图像依赖 | 无 | 迁移后无 | 无 |

## 4. 当前数据契约

`LasImageDataset` 在单样本上提供以下关键字段，其中 `N` 是当前点数，`P` 是图像 patch 数，`C2D` 是 DINO 通道数：

| 字段 | 形状 | 类型 | 含义 |
| --- | --- | --- | --- |
| `coord` | `[N,3]` | float32 | 三维坐标 |
| `feat` | `[N,Cin]` | float32 | 网络输入点特征 |
| `offset` | `[B]`（合批后） | int64 | 点云累计点数 |
| `dino_feature` | `[P,C2D]` | float16/float32 | 行优先展平的 DINO patch 特征 |
| `dino_patch_index` | `[N]` | int64 | 点对应的样本内/合批后全局 patch 索引，无效为 -1 |
| `dino_pixel_coord` | `[N,2]` | int64 | 原图像素 `(row,col)` |
| `dino_valid` | `[N]` | bool | 影像覆盖和表面可见性是否合法 |
| `dino_offset` | `[B]`（合批后） | int64 | 每个样本 patch 数的累计边界 |
| `dino_feature_size` | `[B,2]` | int64 | patch 网格 `(Hf,Wf)` |
| `dino_patch_size` | `[B]` | int64 | patch 对应的像素边长 |
| `dino_original_size` | `[B,2]` | int64 | 原图尺寸 |

`point_collate_fn` 已经在合批时为有效 `dino_patch_index` 加上前序样本的 patch 数，因此模型看到的是可以直接索引合并后 `dino_feature` 的全局 patch index。所有 DINO 点级字段都在 `index_valid_keys` 中，GridSample、Crop 和点丢弃操作会同步索引，不需要逐点二次校验。

模型入口必须检查以下不变量：

1. `len(dino_patch_index) == len(coord)`；
2. `dino_valid=False` 的点应当对应 `patch_index=-1`；
3. 所有有效 patch index 满足 `0 <= index < len(dino_feature)`；
4. 每个点引用的 patch 必须位于其所属样本的 `dino_offset` 区间内；
5. 没有有效 patch 的样本允许存在，并且不得产生 NaN。

## 5. 尺度诊断与默认层级

对 `E:\data\湖北\joint_tiles` 前 10 个真实 tile 的点—像素关系进行线性拟合，得到正射影像地面分辨率约为 `0.1 m/pixel`。当前 DINO patch size 为 16，因此单个 DINO patch 对应约 `1.6 m × 1.6 m` 的地面范围。

以输入 `grid_size=0.5 m`、后续 stride 为 `(2,2,2,2)` 近似统计，各层 token-patch 关系如下：

| 近似三维尺度 | 每 token patch 数中位数 / P90 | 每 patch token 数中位数 / P90 | 平均 token 数/tile |
| ---: | ---: | ---: | ---: |
| 0.5 m | 1 / 1 | 3 / 7 | 45,210 |
| 1 m | 1 / 2 | 2 / 4 | 26,123 |
| 2 m | 2 / 4 | 2 / 3 | 11,406 |
| 4 m | 4 / 9 | 1 / 2 | 3,725 |
| 8 m | 15 / 27 | 1 / 2 | 979 |

因此建议把约 2 米的 encoder level 2 设为默认蒸馏层级，并把约 1 米的 level 1 作为首要对照。level 3 和 level 4 跨越 patch 过多，不适合作为默认值。这里的尺度是 voxel/pooling 尺度，不是 Transformer 的完整感受野，因此最终选择必须通过 level 1/2 消融确定。

配置应使用绝对含义清晰的 `distill_level`，不要在新模块中继续使用相对 bottleneck 的 `enc2d_upcast_level`：

- `distill_level=0`：输入编码层；
- `distill_level=1`：第一次 encoder pooling 后；
- `distill_level=2`：第二次 encoder pooling 后；
- 数值越大，空间层级越粗。

## 6. 总体架构

建议新增一个顶层模型 `DinoPatchDistiller`：

```text
DinoPatchDistiller
├── backbone                         PTV3 或 LitePT
├── hierarchy_adapter                统一提取 encoder level 和细到粗映射
├── token_patch_relation_builder     构建稀疏 token-patch 二部图
├── student_projector                C3D -> d
├── edge_decoder（可选）             对同一 token 的不同 patch 做关系调制
├── frozen_teacher_projector         C2D -> d
└── patch_distillation_loss          样本平衡的余弦损失
```

训练数据流：

```text
点云 ──> PTV3/LitePT encoder hierarchy ──> 选取 level L 的 token 特征
  │                                             │
  └── patch_index/valid ──> 组合 pooling map ──> token-patch 稀疏边
                                                │
DINO Safetensors ──> 冻结投影 ──> teacher patch │
                                                ▼
                                      3D patch prediction
                                                │
                                                ▼
                                         cosine loss
```

模型在 eval 或特征提取模式下可以返回选定层级或最终 backbone 特征，但下游语义分割不应继续使用 distiller wrapper，而应加载同结构普通 backbone。

## 7. Backbone 层级接口

### 7.1 统一接口

PTV3 与 LitePT 应增加可选参数，而默认 forward 行为保持不变：

```python
point = backbone(input_dict, return_hierarchy=False)

point, hierarchy = backbone(input_dict, return_hierarchy=True)
```

建议的 hierarchy 元素：

```python
HierarchyLevel(
    point=point_level,
    input_to_level=input_to_level,  # [N], 输入点 -> 当前 token
    level=level,
    stride=stride_from_input,
)
```

至少需要：

- `point.feat: [M_l,C_l]`
- `point.coord: [M_l,3]`
- `point.batch` 或 `point.offset`
- `input_to_level: [N]`

### 7.2 映射组成

若第 `l` 次 pooling 的 `pooling_inverse_l` 表示 level `l-1` 到 level `l` 的映射，则：

```python
input_to_level = arange(N)
for inverse in pooling_inverses[:level]:
    input_to_level = inverse[input_to_level]
```

第一版可以通过现有 `pooling_parent/pooling_inverse` trace 组装 hierarchy。长期应由 encoder forward 显式保存各层，避免依赖 unpooling 后的链式对象状态。PTV3 和 LitePT 必须共享同一个 `HierarchyLevel` 数据结构，跨模态 wrapper 不应包含特定 backbone 的模块名称判断。

### 7.3 兼容要求

- `return_hierarchy=False` 时 checkpoint key、forward 返回类型和现有分割配置必须完全不变；
- `traceable=True` 是蒸馏配置的必要条件；
- 若 backbone 不提供指定 level，应在构建阶段报错，不应静默退化到最终点层；
- 每层 `input_to_level.max() < len(level.point.feat)`；
- 映射不得跨 batch 样本。

## 8. Token-Patch 稀疏关系

### 8.1 定义

对于输入点 `i`：

- `token_i = input_to_level[i]`
- `patch_i = dino_patch_index[i]`
- `valid_i = dino_valid[i] and patch_i >= 0`

每个有效点生成一条原始关系 `(token_i, patch_i)`。对相同二元组去重并计数，得到：

```python
edge_token:       [E]
edge_patch:       [E]
edge_point_count: [E]
```

高效实现可以构造 int64 key：

```python
edge_key = token_id * num_total_patches + patch_id
unique_key, point_to_edge, edge_count = torch.unique(
    edge_key, return_inverse=True, return_counts=True
)
edge_token = unique_key // num_total_patches
edge_patch = unique_key % num_total_patches
```

必须在构造 key 前完成范围和 batch 边界断言。`num_total_patches` 来自 `dino_feature.shape[0]`。

### 8.2 边统计

建议同时计算：

- `edge_point_count`：该 token-patch 交集内点数；
- `edge_token_fraction`：该 patch 占 token 有效点的比例；
- `edge_patch_fraction`：该 token 占 patch 有效点的比例；
- `edge_mean_pixel`：交集内点的平均像素坐标；
- `edge_relative_pixel`：平均像素相对 patch 中心的位置。

边权默认使用：

```python
edge_weight = sqrt(edge_point_count)
```

原始点数权重会偏向点密集和多回波区域；完全 uniform 又会让单个噪点边与稳定覆盖边等权。平方根计数是稳妥的第一版折中。

## 9. Patch-Centric 三维预测

### 9.1 基础版本

为确保目标层之后的更深 encoder stage 同样获得蒸馏梯度，先将目标层到
bottleneck 的特征沿 pooling inverse 非破坏性地 up-cast 到目标层并拼接。
token 的空间支持仍由目标层决定，但特征包含完整深层语义。随后投影到蒸馏空间：

```python
fused_feat = concat(level_l, upcast(level_l+1), ..., upcast(bottleneck))
token_embed = student_projector(fused_feat)  # [M,d]
edge_embed = token_embed[edge_token]          # [E,d]
```

再以 patch 为目标聚合：

```python
patch_pred = scatter_sum(
    edge_embed * edge_weight[:, None], edge_patch
) / scatter_sum(edge_weight, edge_patch)[:, None]
```

最终只选择被至少一条有效边引用的 patch 参与监督。没有点覆盖的黑色或边缘 patch 不参与 loss。

这种 patch-centric 方向优于先为每个 token 平均 DINO target，因为 DINO teacher 的原生单位就是 patch；保持 teacher 不被跨 patch 平均，更有利于保留边界和小目标语义。

### 9.2 Relation-Aware Edge Decoder

基础版本中，一个 token 对所有相邻 patch 输出相同 embedding。增强版本可加入轻量边调制：

```python
edge_context = concat(
    relative_row,
    relative_col,
    log1p(edge_point_count),
    edge_token_fraction,
    edge_patch_fraction,
)
edge_embed = token_embed[edge_token] + edge_mlp(edge_context)
```

其中相对像素坐标除以 patch size，归一化到大致 `[-0.5,0.5]`。这使同一个 coarse token 可以针对不同 patch 生成不同的低维响应，但不会产生 `[E,1024]` 的高维开销。

第一版应默认 `edge_decoder.enable=False`，验证基础 all-edge 方法后再启用，便于明确收益来源。

## 10. Teacher 维度与显存方案

### 10.1 问题

DiTR 为每点生成 `C2D=1024` 维预测。仅一百万点的 fp16 prediction 就约占 1.9 GiB，尚未包含线性层输入、反向梯度和优化器状态。HPSD 虽然将监督单位降到 patch，但如果每个有效 patch仍生成 1024 维 student 激活，batch 较大时仍不理想。

### 10.2 第一版推荐方案

第一版保留 DINO 原生 `C2D=1024`，不压缩 teacher：

```python
student_projector: fused_C3D -> 1024
teacher_projector: Identity
```

HPSD 只为目标层实际使用的 patch 生成 `[U,1024]` prediction，不生成
逐点 `[N,1024]` prediction。应先通过真实显存测试判断是否确有进一步降维需求。

如果原生 1024 维在目标 batch size 下仍是瓶颈，再使用训练集 DINO patch
样本拟合 PCA：

1. 从训练集均匀采样 patch，避免单个大 tile 主导；
2. 只采样被有效点引用的 patch；
3. 拟合 mean 和 PCA components；
4. 保存为 Safetensors；
5. 训练时作为 buffer 加载，始终冻结；
6. teacher projection 在 `torch.no_grad()` 中执行。

PCA 路径中也不能让 teacher projector 与 student 一起无约束训练，否则 teacher 空间可能退化，损失失去固定目标。

PCA-512/PCA-256 是条件性优化，不属于第一版默认路径；1024 维原空间余弦是默认基线和精度参照。

### 10.3 CompactDinoPatches

建议增加一个最终数据变换，在所有逐点 crop/grid/drop 之后、Collect 之前执行：

```python
used = unique(dino_patch_index[dino_valid])
dino_feature = dino_feature[used]
dino_source_patch_index = used
dino_patch_index = remap_to_compact_index(dino_patch_index, used)
dino_offset = len(used)
```

这样只把仍被当前点集引用的 DINO patch 传输到 GPU。`dino_source_patch_index` 是样本级 patch 数组，用于恢复原始 `(patch_row,patch_col)`：

```python
patch_row = source_index // feature_width
patch_col = source_index % feature_width
```

该变换必须支持空 used 集合，并与 `point_collate_fn` 的全局 patch offset 调整兼容。

## 11. 损失函数

### 11.1 主损失

teacher 和 student embedding 均在 float32 中 L2 normalize：

```python
student = F.normalize(patch_pred.float(), dim=-1)
teacher = F.normalize(teacher_projected.float(), dim=-1)
loss_patch = 1.0 - (student * teacher).sum(-1)
```

应先按样本对有效 patch 求均值，再对 batch 求均值：

```python
loss = mean_over_batch(mean_over_valid_patches(loss_patch))
```

不能直接对全 batch 所有 patch 求均值，否则有效覆盖面积大、点密度高的 tile 会获得更大训练权重。

### 11.2 Patch 可靠性

第一版推荐：

- `min_patch_points=1`；
- patch 之间 uniform；
- patch 内 token 聚合使用 `sqrt_count` 边权；
- 无有效 patch 的样本从 batch mean 中跳过；
- 若整个 batch 无有效 patch，返回与 student 参数相连的零损失。

后续可比较 `min_patch_points=2/4` 和基于覆盖点数的 capped reliability weight，但不应在第一版同时加入过多启发式规则。

### 11.3 可选辅助损失

可以为每个 token 构造其覆盖 patch teacher 特征的加权平均，增加小权重 token loss。它可能改善训练稳定性，但也会平滑多 patch 语义，因此建议默认关闭或令权重不超过 0.1。

## 12. 正射可见性

正射影像主要表达最高可见表面，而机载点云同时包含树冠、树冠内部、地面、屋顶、立面和多次回波。若同一 patch 同时监督树冠和被遮挡地面，teacher target 在物理上是错误的。

正式蒸馏数据建议使用：

```python
surface_only_valid=True
surface_cell_size="auto"
surface_radius="auto"
surface_z_tolerance=0.15
```

HPSD 只使用 `dino_valid=True` 的点构边。无直接 DINO 监督的内部点仍会通过共享 backbone、稀疏卷积和 self-attention 获得间接梯度。

应保留 `surface_only_valid=False/True` 消融，以量化可见性过滤的实际价值。

## 13. 数据增强兼容性

由于 patch index 作为点属性随点同步索引，以下变换天然兼容：

- GridSample；
- Crop/SphereCrop；
- RandomDropout；
- 点坐标旋转、缩放、平移；
- 坐标 jitter 和 elastic distortion；
- 只改变点特征的归一化与增强。

几何增强后点的空间位置改变，但 teacher patch 仍代表该点在原始观测中的视觉语义，这与跨视图自监督的基本假设一致。

需要重点测试：

- 会复制点的 transform 是否同步复制所有 DINO 点级字段；
- Mix3D 是否同步合并 `dino_feature/dino_offset` 并全局化 patch index；
- CompactDinoPatches 必须放在所有会删除点的 transform 之后；
- 不允许仅重排 `dino_feature` 而不更新 patch index。

## 14. 训练与 checkpoint 迁移

### 14.1 预训练

`DinoPatchDistiller` checkpoint 建议包含：

```text
backbone.*
student_projector.*
edge_decoder.*              可选
teacher_projector.*         buffer 或冻结参数
```

DINO patch 特征是输入数据，不属于 checkpoint。

### 14.2 下游微调

下游使用普通 `DefaultSegmentorV2 + PTV3/LitePT + seg_head`。加载预训练 checkpoint 时只迁移 `backbone.*`：

- `student_projector.*` 丢弃；
- `edge_decoder.*` 丢弃；
- `teacher_projector.*` 丢弃；
- `seg_head.*` 随机初始化。

建议提供显式的 checkpoint include-prefix 配置，而不完全依赖 `strict=False` 的隐式 missing/unexpected key 行为。日志中应打印 backbone 实际加载参数比例，防止命名不一致导致看似成功、实际未加载。

### 14.3 纯三维推理

微调和推理数据集改回 `LasDataset`。模型输入不再需要任何 `dino_*` 字段，显存和时延与普通三维分割网络一致。

## 15. 推荐配置草案

```python
model = dict(
    type="DinoPatchDistiller",
    backbone=dict(
        type="PT-v3m1",  # 或 LitePT-v1m1/LitePT-v1m3
        in_channels=5,
        traceable=True,
        # ...
    ),
    distill_level=2,
    level_channels=(64, 64, 128, 256, 512),
    teacher_dim=1024,
    distill_dim=1024,
    relation=dict(
        type="TokenPatchBipartite",
        edge_weight="sqrt_count",
        min_patch_points=1,
        preserve_all_patches=True,
    ),
    student_projector=dict(
        type="Linear",
        out_channels=1024,
    ),
    edge_decoder=dict(
        enable=False,
        hidden_channels=128,
        use_relative_pixel=True,
        use_count=True,
        use_coverage=True,
    ),
    teacher_projector=dict(type="Identity"),
    criteria=dict(
        type="PatchCosineLoss",
        sample_balanced=True,
        fp32_normalize=True,
        loss_weight=1.0,
    ),
)
```

推荐数据配置顺序：

```python
transform = [
    # 点级增强
    ...,
    dict(type="GridSample", grid_size=0.5, mode="train"),
    # 其他删除/裁剪点的变换
    ...,
    dict(type="CompactDinoPatches"),
    dict(type="ToTensor"),
    dict(type="Collect", keys=(...)),
]
```

## 16. 复杂度与显存分析

定义：

- `N`：输入点数；
- `M`：目标层 token 数；
- `E`：unique token-patch 边数；
- `U`：有点覆盖的有效 patch 数；
- `C2D=1024`；
- `d=256`。

DiTR student prediction 激活规模为：

```text
O(N * C2D)
```

HPSD 的主要额外激活为：

```text
token embedding: O(M * d)
edge embedding:  O(E * d)    仅启用 edge decoder 时
patch prediction: O(U * d)
teacher input: O(U * C2D)    无梯度，可使用 fp16
```

在当前数据的 2 米近似层级上，每 tile 平均约 11,406 个 token、12,520 个有效 patch，一个 token 的 patch 数中位数为 2。与十万到数十万输入点的 `[N,1024]` 输出相比，student 侧激活显著降低。

构边中的 `torch.unique` 是主要离散操作。第一版应优先保证正确性；若 profiler 表明它成为瓶颈，再考虑排序编码、CUDA scatter 或提前缓存部分静态关系。由于数据增强和 GridSample 会改变保留点集合，不建议一开始缓存目标层关系。

## 17. 风险与对应措施

### 17.1 目标层过粗

表现：每 token 跨越大量 patch，同一个 token 收到冲突 target。

措施：记录每 batch 的 patches-per-token 分布；默认 level 2；对比 level 1；超过阈值时报警而不是自动丢弃。

### 17.2 目标层过细

表现：每 patch 对应大量 token，监督接近重复逐点回归，计算量上升。

措施：比较 tokens-per-patch 和吞吐量；使用 patch-centric 聚合保证最终每 patch 只有一个 loss。

### 17.3 隐藏点污染

表现：树冠和地面共享同一 teacher target，垂直结构特征被错误拉近。

措施：默认使用 surface-only correspondence；记录有效点比例和每 patch 垂直 token 数。

### 17.4 PCA 损失语义

表现：256 维空间无法保持全部 DINO 结构。

措施：比较 PCA-256、PCA-512 和原始 1024；PCA 使用被点覆盖 patch 的全训练集样本拟合。

### 17.5 Dataset compaction 与 index 错位

表现：patch index 指向错误 teacher 行，loss 正常但监督完全错误。

措施：为 CompactDinoPatches 编写逐值单元测试；保存 source index；测试合批前后 gather 结果一致。

### 17.6 Backbone hierarchy 漂移

表现：PTV3 和 LitePT 对 level 编号理解不一致，配置不可迁移。

措施：统一 `HierarchyLevel`；构建时验证 stage 数和 channel；测试每层 token 数严格非增并且映射范围合法。

## 18. 实现计划

### 阶段 A：统一 hierarchy

1. 定义公共 `HierarchyLevel/PointHierarchy` 数据结构；
2. 为目标 PTV3 添加 `return_hierarchy`；
3. 为目标 LitePT 添加相同接口；
4. 保证默认 forward 和现有 checkpoint 完全兼容；
5. 用合成点云验证每层映射。

验收标准：PTV3/LitePT 均能返回相同结构，`input_to_level` 与逐层 inverse 组合完全一致，现有分割 forward 数值不变。

### 阶段 B：基础 HPSD

1. 实现 token-patch relation builder；
2. 实现 patch-centric `sqrt_count` 聚合；
3. 实现原始 1024 维 cosine 基线；
4. 实现空 patch、无效点和 batch 平衡；
5. 实现 distiller checkpoint 输出。

验收标准：合成数据上手算结果一致；PTV3/LitePT 均能完成 forward/backward；下游可以只加载 backbone。

### 阶段 C：显存优化

1. 实现 PCA 拟合工具与 FrozenPCA；
2. 默认切换到 256 维蒸馏；
3. 实现 CompactDinoPatches；
4. profiler 对比逐点 DiTR baseline。

验收标准：同 batch 下峰值显存显著低于逐点 1024 维方案，且 patch gather 与 compaction 前逐值一致。

### 阶段 D：关系增强

1. 实现 edge relative pixel statistics；
2. 实现可选 edge decoder；
3. 增加多 patch 统计日志；
4. 完成 level 和 edge 方法消融。

### 阶段 E：UTONIA 联合训练

在独立 HPSD 收益确认后，将 `patch_distillation_loss` 作为 UTONIA 的附加分支。DINO loss 作用于 student 的指定层，3D EMA teacher 继续服务 mask/unmask 目标，两者共享 student backbone，但不共享 teacher head。

## 19. 测试计划

### 19.1 单元测试

1. 手工构造点→token→patch 小图，验证 unique edge、count、fraction；
2. 一个 token 对多个 patch；
3. 一个 patch 对多个 token；
4. 重复回波点对应同一 patch；
5. 全部点无效；
6. batch 中一个样本有 patch、另一个没有；
7. CompactDinoPatches 的 source/remap 可逆性；
8. PCA projector 冻结且无梯度；
9. AMP 下 cosine 使用 float32；
10. PTV3/LitePT hierarchy mapping 一致性。

### 19.2 集成测试

1. `LasImageDataset -> transform -> point_collate_fn -> PTV3 -> loss -> backward`；
2. 同样流程替换 LitePT；
3. `num_workers=0/2/8`；
4. 单卡和 DDP；
5. gradient accumulation；
6. checkpoint 保存、恢复和只迁移 backbone；
7. 无 DINO 字段的普通分割推理。

### 19.3 真实数据测试

在 `joint_tiles` 上记录：

- 每层 token 数；
- valid 点比例；
- used patch 数；
- unique edge 数；
- patches-per-token 中位数/P90/P99；
- tokens-per-patch 中位数/P90/P99；
- 空监督样本数；
- loss 是否有限；
- 峰值 GPU 显存；
- data time、forward time、backward time；
- 每秒处理点数和 patch 数。

## 20. 消融实验矩阵

### 20.1 Correspondence 处理

1. DiTR 风格逐点 1024 维；
2. coarse token 最近 patch；
3. coarse token 的 DINO target 均值；
4. HPSD all-edge patch-centric；
5. HPSD + relation-aware edge decoder。

### 20.2 层级

- level 0；
- level 1；
- level 2；
- level 3。

### 20.3 Teacher 空间

- DINO-1024；
- PCA-512；
- PCA-256；
- random orthogonal-256。

### 20.4 航空点云因素

- surface-only on/off；
- edge uniform/count/sqrt-count；
- `min_patch_points=1/2/4`；
- 纯 HPSD 与 HPSD+UTONIA。

### 20.5 下游评价

- 全量标注微调 mIoU；
- 1%、5%、10% 标注量微调；
- linear probing；
- backbone PCA 可视化；
- 峰值显存和训练吞吐量。

## 21. 日志与可观测性

每个训练 epoch 建议记录：

```text
distill/loss
distill/valid_samples
distill/valid_points_ratio
distill/used_patches
distill/edges
distill/patches_per_token_p50
distill/patches_per_token_p90
distill/tokens_per_patch_p50
distill/tokens_per_patch_p90
distill/teacher_norm
distill/student_norm
```

这些统计不应逐 iteration 全量同步到 CPU。可以每若干 step 抽样，DDP 下对标量做 reduce。若多 patch 分布随增强显著变化，这些日志可以快速区分层级选择问题和优化器问题。

## 22. 第一版最终建议

第一版实现应严格控制范围，推荐固定为：

```text
LasImageDataset
+ surface_only_valid correspondence
+ CompactDinoPatches
+ PTV3/LitePT unified hierarchy
+ distill_level=2
+ complete token-patch edges
+ sqrt-count patch-centric pooling
+ native DINO-1024 cosine target
+ sample-balanced cosine loss
+ single 3D student
```

暂不启用 PCA、edge decoder、token auxiliary loss、EMA 3D teacher、prototype 和 Sinkhorn。首先建立 DiTR-style point baseline 与 HPSD level 1/2 基线，确认尺度对齐和显存收益。只有原生 1024 维在目标 batch size 下显存确实不足时才进入 PCA 阶段。若 all-edge 方法优于最近 patch/平均 target，再加入 edge decoder；若纯 HPSD 已稳定提升下游性能，再与 UTONIA 3D 自监督目标联合。

这一顺序能够让每个设计决策都有独立证据，避免最终系统虽然复杂但无法判断收益来自何处。
