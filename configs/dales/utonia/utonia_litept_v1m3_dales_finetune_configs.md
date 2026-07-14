# DALES Utonia LitePT-v1m3 微调配置说明

本文说明 `configs/dales/utonia` 下 4 个 DALES 语义分割微调配置：

- `semseg-utonia-litept-v1m3-dales-base.py`
- `semseg-utonia-litept-v1m3-dales-lin.py`
- `semseg-utonia-litept-v1m3-dales-dec.py`
- `semseg-utonia-litept-v1m3-dales-ft.py`

它们都用于把 Utonia 自监督预训练得到的 LitePT-v1m3 backbone 权重迁移到 DALES 语义分割任务。

## 共同设定

4 个配置都面向 DALES LAS 点云语义分割，输入特征为：

```text
coord + echo
```

其中 `coord` 是 3 维坐标，`echo` 在当前数据读取逻辑中为 2 维回波相关特征，因此：

```python
in_channels = 5
feature_keys = ["coord", "echo"]
```

类别数为 8：

```text
ground, vegetation, cars, trucks, power lines, fences, poles, buildings
```

预训练权重默认从这里加载：

```python
weight = "exp/dales/utonia/pretrain-litept-v1m3-dales-xyz-echo/model/model_last.pth"
```

加载时使用：

```python
dict(
    type="CheckpointLoader",
    keywords="module.student.backbone",
    replacement="module.backbone",
)
```

原因是 Utonia 自监督预训练 checkpoint 中的 backbone 位于：

```text
module.student.backbone.*
```

而下游 `DefaultSegmentorV2` 中 backbone 位于：

```text
module.backbone.*
```

所以需要把 key 前缀替换后再加载。

## 文件关系

`semseg-utonia-litept-v1m3-dales-base.py` 是本地公共配置，同时也可以直接作为 full fine-tuning 配置运行。

另外 3 个配置继承它：

```python
_base_ = ["./semseg-utonia-litept-v1m3-dales-base.py"]
```

这样数据路径、类别、数据增强、损失函数、hook、权重加载规则只需要维护一份。`lin/dec/ft` 只覆盖自己关心的微调方式差异。

## 四种配置对比

| 配置 | 主要用途 | `enc_mode` | `freeze_encoder` | `freeze_backbone` | `backbone_out_channels` |
|---|---|---:|---:|---:|---:|
| `base` | 默认全量微调，也作为公共配置 | `False` | `False` | `False` | `72` |
| `lin` | 线性评估 encoder 表征 | `True` | `False` | `True` | `1008` |
| `dec` | 冻结 encoder，只训练 decoder/head | `False` | `True` | `False` | `72` |
| `ft` | 明确命名的全量微调实验 | `False` | `False` | `False` | `72` |

## `base`：公共配置与默认全量微调

文件：

```text
configs/dales/utonia/semseg-utonia-litept-v1m3-dales-base.py
```

它定义了完整的下游语义分割流程：

- 数据路径：train / val / test / pred
- DALES 类别与 remap 规则
- LitePT-v1m3 backbone 结构
- `DefaultSegmentorV2`
- CrossEntropy + Lovasz 损失
- 预训练权重加载规则
- 训练、验证、测试 transform
- evaluator、checkpoint、writer 等 hook

它本身的微调方式是 full fine-tuning：

```python
enc_mode = False
freeze_encoder = False
freeze_backbone = False
backbone_out_channels = 72
```

含义是：

- 使用 LitePT 的 encoder + decoder 结构；
- encoder、decoder、分割 head 都参与训练；
- 输出给分割 head 的是 decoder 恢复后的较高分辨率特征；
- `backbone_out_channels=72` 对应 decoder 输出通道。

这个配置适合数据量较充分、希望模型充分适配 DALES 语义分割任务的场景。

## `lin`：线性评估

文件：

```text
configs/dales/utonia/semseg-utonia-litept-v1m3-dales-lin.py
```

关键覆盖项：

```python
model = dict(
    backbone_out_channels=1008,
    backbone=dict(
        enc_mode=True,
        freeze_encoder=False,
    ),
    freeze_backbone=True,
)
```

`freeze_backbone=True` 表示整个 backbone 冻结，只训练 `DefaultSegmentorV2` 的分割头。这里的核心目标不是追求最高精度，而是评估自监督预训练得到的 encoder 表征质量。

`enc_mode=True` 表示 LitePT 以 encoder-only 方式输出特征，不走 decoder。此时输出特征是多层 encoder 特征回填并拼接后的结果，因此通道数是：

```text
36 + 72 + 144 + 252 + 504 = 1008
```

所以需要：

```python
backbone_out_channels = 1008
```

`lin` 适合回答这个问题：

```text
预训练 backbone 冻住不动，仅靠一个轻量分割头，特征本身有多好？
```

如果 `lin` 效果明显好，说明预训练学到了较强的可迁移点云表征。

## `dec`：冻结 encoder，训练 decoder/head

文件：

```text
configs/dales/utonia/semseg-utonia-litept-v1m3-dales-dec.py
```

关键覆盖项：

```python
model = dict(
    backbone_out_channels=72,
    backbone=dict(
        enc_mode=False,
        freeze_encoder=True,
    ),
    freeze_backbone=False,
)
```

它与 `lin` 的区别是：

- `lin` 冻结整个 backbone，只训练分割 head；
- `dec` 只冻结 encoder，但允许 decoder 和分割 head 训练。

`enc_mode=False` 表示模型会使用 decoder，把 encoder 的多尺度特征逐级恢复到更细分辨率。decoder 输出通道为 72，因此：

```python
backbone_out_channels = 72
```

`dec` 适合评估：

```text
固定预训练 encoder，只学习任务相关的解码器，能达到什么效果？
```

它比 `lin` 更有适配能力，因为 decoder 可以学习 DALES 语义分割所需的空间恢复和类别边界信息；但它仍然保护 encoder 不被下游监督信号改动。

## `ft`：全量微调

文件：

```text
configs/dales/utonia/semseg-utonia-litept-v1m3-dales-ft.py
```

当前它只覆盖：

```python
save_path = "exp/dales/utonia/semseg-litept-v1m3-ft"
```

其余设置继承 `base`，因此训练方式与 `base` 一致：

```python
enc_mode = False
freeze_encoder = False
freeze_backbone = False
backbone_out_channels = 72
```

也就是说，`ft` 是一个命名更明确的 full fine-tuning 实验入口。保留它的意义是让实验目录和对比表更清晰：

```text
lin: 线性评估
dec: 冻结 encoder 训练 decoder
ft : 全量微调
```

如果不需要区分实验命名，直接运行 `base` 与运行 `ft` 在训练策略上等价。

## 推荐实验顺序

建议按以下顺序做对比：

1. `lin`

   先看预训练 encoder 表征本身是否有效。它训练成本低，能快速暴露预训练是否学到了有用特征。

2. `dec`

   冻结 encoder，只训练 decoder/head。这个实验能判断：固定预训练特征后，仅靠下游解码器适配，性能能提升多少。

3. `ft`

   最后做全量微调。它通常上限最高，但也最容易受学习率、数据量、类别不平衡和增强策略影响。

## 学习率差异

公共配置中：

```python
optimizer = dict(type="AdamW", lr=1e-3, weight_decay=2e-3)
param_dicts = [dict(keyword="block", lr=1e-4)]
```

`lin` 和 `dec` 覆盖为：

```python
optimizer = dict(type="AdamW", lr=2e-3, weight_decay=2e-3)
param_dicts = [dict(keyword="block", lr=2e-4)]
```

原因是：

- `lin` 主要训练新初始化的分割头，可以使用稍大学习率；
- `dec` 主要训练 decoder/head，也可以比 full fine-tuning 更激进；
- `ft/base` 会更新 encoder，学习率更保守一些，避免破坏预训练表征。

## 与 Utonia 预训练的关系

预训练配置中当前使用：

```python
up_cast_level = 0
head_in_channels = 504
```

这表示自监督损失主要作用在最深层、最语义化的 encoder token 上。

下游 `lin` 使用：

```python
enc_mode = True
backbone_out_channels = 1008
```

这是因为线性评估阶段可以取 encoder 多层拼接特征：

```text
36 + 72 + 144 + 252 + 504 = 1008
```

即使预训练时 `up_cast_level=0`，下游仍然可以在提特征时使用更细层级的多层拼接特征。但需要注意：浅层部分不是被 Utonia 自监督 head 直接优化的，它们更多是通过 backbone 端到端训练间接受到约束。

## 选择建议

- 想快速检查预训练是否有效：用 `lin`。
- 想保守迁移，避免 encoder 被小数据集过拟合：用 `dec`。
- 想追求最终语义分割精度：用 `ft` 或直接用 `base`。
- 想维护公共数据/模型设置：改 `base`。
- 想只改变某种微调策略：改对应的 `lin/dec/ft`。

