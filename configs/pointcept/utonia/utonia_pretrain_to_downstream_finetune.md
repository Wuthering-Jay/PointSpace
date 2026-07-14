# Utonia 预训练权重加载与下游任务微调方式

本文说明 Utonia 预训练完成后，如何把预训练权重加载到下游任务，以及 `lin`、`dec`、`ft` 等配置分别代表什么微调方式。

相关配置示例：

- `semseg-utonia-v1m1-0a-scannet-lin.py`
- `semseg-utonia-v1m1-0b-scannet-dec.py`
- `semseg-utonia-v1m1-0c-scannet-ft.py`
- `semseg-utonia-v1m1-0d-scannet-nocolor-lin.py`

## 总体思路

Utonia 预训练阶段使用的是 `Utonia-v1m1` 这个预训练封装模型，里面包含：

- `student.backbone`
- `teacher.backbone`
- `mask_head`
- `unmask_head`
- `enc2d_model`
- 若干自监督 loss 相关模块

下游任务阶段通常不再使用整个 `Utonia-v1m1` 预训练壳，而是只取其中的 **student backbone**，放进具体任务模型里。例如语义分割使用：

```python
model = dict(
    type="DefaultSegmentorV2",
    backbone=dict(type="PT-v3m3", ...),
    ...
)
```

也就是说，下游任务模型结构大致是：

```text
预训练 student.backbone 权重
        ↓
下游任务 backbone
        ↓
seg_head / decoder / task head
        ↓
具体任务 loss
```

## 预训练权重如何加载

下游配置中的关键 hook 是：

```python
hooks = [
    dict(
        type="CheckpointLoader",
        keywords="module.student.backbone",
        replacement="module.backbone",
    ),
    ...
]
```

预训练 checkpoint 中的权重 key 通常类似：

```text
module.student.backbone.embedding.stem.linear.weight
module.student.backbone.enc.enc0.block0.attn.qkv.weight
...
```

而下游 `DefaultSegmentorV2` 期望的 key 是：

```text
module.backbone.embedding.stem.linear.weight
module.backbone.enc.enc0.block0.attn.qkv.weight
...
```

所以 `CheckpointLoader` 会做 key 替换：

```text
module.student.backbone  ->  module.backbone
```

这样就能把 Utonia 预训练阶段的 student backbone 加载到下游语义分割模型的 backbone 中。

## 加载时哪些模块有权重

会从预训练 checkpoint 加载：

- backbone 的 embedding
- encoder blocks
- 如果下游模型中存在同名 decoder，且 checkpoint 中也有对应权重，则可能加载；但 Utonia 预训练通常是 `enc_mode=True`，没有训练 decoder

通常不会从预训练 checkpoint 加载：

- `seg_head`
- 新建 decoder
- 任务特定 head
- loss 模块

这些模块会随机初始化，并在下游任务训练中学习。

## 三种主要微调方式

### 1. lin：线性探测

示例：

```text
semseg-utonia-v1m1-0a-scannet-lin.py
semseg-utonia-v1m1-0d-scannet-nocolor-lin.py
```

核心配置：

```python
model = dict(
    type="DefaultSegmentorV2",
    backbone_out_channels=1386,
    backbone=dict(
        enc_mode=True,
        freeze_encoder=False,
        ...
    ),
    freeze_backbone=True,
)
```

含义：

- 只使用 encoder，不构建 decoder。
- 加载预训练 backbone。
- 冻结整个 backbone。
- 只训练最后的 `seg_head`。

`backbone_out_channels=1386` 来自多层 encoder 特征拼接：

```text
54 + 108 + 216 + 432 + 576 = 1386
```

`lin` 主要用于评估预训练表征质量。因为 backbone 不更新，所以它回答的是：

```text
预训练好的 3D 表征，在不微调 backbone 的情况下，线性分类头能做到什么程度？
```

优点：

- 训练稳定。
- 成本低。
- 能较纯粹地评估预训练特征。

缺点：

- 下游任务适配能力有限。
- 性能通常低于全量微调。

### 2. dec：冻结 encoder，训练 decoder 和 head

示例：

```text
semseg-utonia-v1m1-0b-scannet-dec.py
```

核心配置：

```python
model = dict(
    type="DefaultSegmentorV2",
    backbone_out_channels=54,
    backbone=dict(
        enc_mode=False,
        freeze_encoder=True,
        dec_depths=(2, 2, 2, 2),
        dec_channels=(54, 108, 216, 432),
        ...
    ),
    freeze_backbone=False,
)
```

含义：

- 使用 encoder-decoder 结构。
- encoder 加载预训练权重并冻结。
- decoder 是下游任务中新建的，参与训练。
- `seg_head` 参与训练。

这类方式可以理解为：

```text
固定预训练 encoder，当作通用特征提取器；
训练一个 decoder，把低分辨率/多层特征恢复到点级密集预测空间。
```

优点：

- 比 `lin` 有更强的任务适配能力。
- 比 `ft` 更不容易破坏预训练 encoder。
- 适合数据量不是特别大、但需要 dense prediction 的任务。

缺点：

- encoder 不动，领域差异很大时适配能力仍然受限。
- decoder 从零训练，需要一定训练时间。

### 3. ft：全量微调

示例：

```text
semseg-utonia-v1m1-0c-scannet-ft.py
```

核心配置：

```python
model = dict(
    type="DefaultSegmentorV2",
    backbone_out_channels=54,
    backbone=dict(
        enc_mode=False,
        freeze_encoder=False,
        ...
    ),
    freeze_backbone=False,
)
```

含义：

- 使用 encoder-decoder 结构。
- encoder 加载预训练权重。
- decoder 随机初始化。
- encoder、decoder、`seg_head` 全部参与训练。

这就是完整 fine-tuning。

优点：

- 下游适配能力最强。
- 通常能取得最高性能。

缺点：

- 训练成本更高。
- 对学习率更敏感。
- 小数据集上可能过拟合或破坏预训练特征。

通常 `ft` 会使用更小的学习率。例如：

```python
optimizer = dict(type="AdamW", lr=0.001, weight_decay=0.01)
param_dicts = [dict(keyword="block", lr=0.0001)]
```

这里 backbone blocks 使用更低学习率，避免过快破坏预训练权重。

## nocolor 配置是什么意思

示例：

```text
semseg-utonia-v1m1-0d-scannet-nocolor-lin.py
```

它不是新的微调范式，而是输入模态消融实验。

关键 transform：

```python
dict(type="RandomDropColor", drop_ratio=1.0, drop_application_ratio=1.0)
```

含义是把所有点的颜色全部置零。

但是 `Collect` 中仍然保留：

```python
feat_keys=("coord", "color", "normal")
```

这样输入通道数仍然是：

```text
coord 3 + color 3 + normal 3 = 9
```

好处是：

- `backbone.in_channels=9` 不变；
- 第一层 embedding 权重形状不变；
- 可以直接加载 Utonia 官方预训练权重；
- 但模型实际看不到颜色信息。

这类配置用于回答：

```text
如果不给颜色，只靠几何和法线，预训练特征表现如何？
```

## 适配新任务时需要改什么

通常需要改：

```python
num_classes
data.names
dataset_type
data_root
split
criteria
feat_keys
backbone.in_channels
backbone_out_channels
```

### 类别数

例如 ScanNet 是：

```python
num_classes=20
```

如果你的任务是 8 类，就要改成：

```python
num_classes=8
```

并同步修改：

```python
data = dict(
    num_classes=8,
    names=[...],
)
```

### 输入特征通道

Utonia 预训练常见输入是：

```python
feat_keys=("coord", "color", "normal")
```

对应：

```text
coord 3 + color 3 + normal 3 = 9
```

因此 backbone 配置为：

```python
in_channels=9
```

如果你换成 LAS 数据，例如：

```python
feat_keys=("coord", "echo", "intensity")
```

那么通道数变成：

```text
coord 3 + echo 2 + intensity 1 = 6
```

此时需要：

```python
in_channels=6
```

但这会带来一个重要问题：预训练权重中的第一层 embedding 是按 9 通道训练的，形状和 6 通道模型不匹配。

## 输入通道不一致时怎么办

有三种常见处理方式。

### 方式 A：保持 9 通道，缺失模态填 0

例如没有 color，就仍然保留 color 字段，但全部置零：

```python
dict(type="RandomDropColor", drop_ratio=1.0, drop_application_ratio=1.0)
feat_keys=("coord", "color", "normal")
in_channels=9
```

优点：

- 最容易直接加载官方权重。
- 不需要改 checkpoint。
- 适合做模态消融。

缺点：

- 输入中存在无信息通道。
- 如果你的真实任务特征和预训练模态差异很大，表达不一定最优。

### 方式 B：修改 in_channels，跳过第一层 embedding 权重

例如：

```python
feat_keys=("coord", "echo", "intensity")
in_channels=6
```

这种情况下需要让 checkpoint 加载时跳过 shape 不匹配的 embedding 权重。

优点：

- 输入特征定义更干净。
- 更适合真实业务模态。

缺点：

- 第一层随机初始化。
- 需要确认加载逻辑能跳过 shape mismatch。

### 方式 C：手动迁移 embedding 权重

例如从 9 通道权重中保留 coord 部分，新增 echo/intensity 通道随机初始化或用已有通道均值初始化。

优点：

- 能最大限度复用已有权重。

缺点：

- 需要写权重转换脚本。
- 如果通道语义差异大，手动映射未必合理。

## 如何选择微调方式

建议顺序：

1. 先跑 `lin`

确认预训练权重能正确加载，数据管线正常，标签和评价无问题。

2. 再跑 `dec`

如果任务是语义分割、实例分割、点级回归等 dense prediction，`dec` 通常比 `lin` 更合理。

3. 最后跑 `ft`

当数据量足够、任务与预训练域差异较大，或者你追求最高指标时，使用全量微调。

4. 做输入模态消融

例如：

- full：`coord + color + normal`
- nocolor：`coord + 0 + normal`
- no-normal：`coord + color + 0`
- geometry only：`coord + 0 + 0`

## 快速对照表

| 配置后缀 | backbone 结构 | encoder 是否训练 | decoder 是否训练 | head 是否训练 | 用途 |
| --- | --- | --- | --- | --- | --- |
| `lin` | encoder only | 否 | 无 | 是 | 线性探测，评估预训练特征 |
| `dec` | encoder-decoder | 否 | 是 | 是 | 冻结预训练 encoder，训练任务 decoder |
| `ft` | encoder-decoder | 是 | 是 | 是 | 全量微调，追求最高性能 |
| `nocolor-lin` | encoder only | 否 | 无 | 是 | 无颜色输入的线性探测 |

## 最关键的几条

1. 预训练 checkpoint 中最有用的是 `student.backbone`。
2. 下游加载靠 `CheckpointLoader` 把 `module.student.backbone` 替换成 `module.backbone`。
3. `lin` 冻结 backbone，只训线性头。
4. `dec` 冻结 encoder，训练 decoder 和 head。
5. `ft` 训练 encoder、decoder 和 head。
6. 如果改变输入特征通道数，要特别处理第一层 embedding 权重。
