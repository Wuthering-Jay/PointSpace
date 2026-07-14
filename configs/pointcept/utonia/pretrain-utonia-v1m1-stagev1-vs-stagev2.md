# pretrain-utonia-v1m1 Stage v1 与 Stage v2 配置差异

对比文件：

- `configs/pointcept/utonia/pretrain-utonia-v1m1-0-base_stagev1.py`
- `configs/pointcept/utonia/pretrain-utonia-v1m1-0-base_stagev2.py`

## 总览

Stage v2 可以看作是在 Stage v1 的模型、优化器、学习率调度和基础增强策略上，扩展训练数据规模与数据类型的版本。两者的 Utonia 模型结构、backbone、loss 权重、epoch、batch size、学习率、weight decay、scheduler、hooks 基本一致；主要差异集中在训练器选择、数据采样策略、新增 transform 和训练数据集列表。

## 主要差异

| 模块 | Stage v1 | Stage v2 | 影响 |
| --- | --- | --- | --- |
| 训练器 | 未显式指定 `train` | 新增 `train = dict(type="PartialSampledTrainer")` | Stage v2 使用部分采样训练器，配合后面的 `sampled_dataset_*` 设置控制数据采样。 |
| 模型结构 | `Utonia-v1m1`，PT-v3m3 backbone | 完全一致 | 模型容量、head、teacher/student、mask 配置不变。 |
| 优化器与 scheduler | AdamW + OneCycleLR | 完全一致 | 训练超参保持一致，差异主要来自数据和 trainer。 |
| 基础 transform | `outdoor_transform`、`obj_transform`、`indoor_transform` | 保留上述 transform | Stage v2 继承 Stage v1 的主要增强流水线。 |
| 新增 transform | 无 | 新增 `obj_realscale_withbg_transform`、`obj_realscale_nobg_transform`、`hk_transform` | 支持真实尺度物体、有/无背景物体数据，以及 HK 地图数据。 |
| 数据集数量 | 4 个 | 15 个 | Stage v2 的预训练数据覆盖 outdoor、object、indoor、mapping 等更多域。 |
| 数据采样控制 | 无 | 新增 `sampled_dataset_index=4`、`sampled_dataset_limit=90000` | Stage v2 对某个数据集索引启用采样上限，避免单一数据源过度主导。 |
| hooks | CheckpointLoader、ModelHook、WeightDecaySchedular、IterationTimer、InformationWriter、CheckpointSaver | 完全一致 | 运行时 hook 行为一致。 |

## 保持一致的部分

以下配置在两个文件中一致：

- 基础运行时：`_base_ = ["../_base_/default_runtime.py"]`
- 图像输入与 patch 参数：`crop_h=518`、`crop_w=518`、`patch_size=14`
- 训练规模相关参数：`batch_size=256`、`num_worker=1024`
- AMP 与梯度设置：`enable_amp=True`、`amp_dtype="bfloat16"`、`clip_grad=1.0`
- 模型主体：`type="Utonia-v1m1"`
- 2D image encoder 权重：`dinov2_vitg14_reg` / `facebook/dinov2-with-registers-giant`
- 3D backbone：`PT-v3m3`
- backbone 深度与通道数：
  - `enc_depths=(3, 3, 3, 12, 3)`
  - `enc_channels=(54, 108, 216, 432, 576)`
  - `enc_num_head=(3, 6, 12, 24, 32)`
- self-supervised 训练相关参数：
  - `num_global_view=2`
  - `num_local_view=4`
  - mask size / mask ratio / teacher temp warmup 设置一致
  - loss weight 一致
  - momentum schedule 一致
- 训练周期与优化：
  - `epoch=100`
  - `base_lr=0.004`
  - `lr_decay=0.9`
  - `base_wd=0.04`
  - `final_wd=0.2`
  - `optimizer = AdamW`
  - `scheduler = OneCycleLR`
- hook 列表一致。

## 训练器差异

Stage v2 新增：

```python
train = dict(type="PartialSampledTrainer")
```

Stage v1 没有显式设置 `train`，因此会使用默认 runtime 或框架默认 trainer。Stage v2 明确使用 `PartialSampledTrainer`，并在 `data` 中加入：

```python
sampled_dataset_index = 4
sampled_dataset_limit = 90000
```

这说明 Stage v2 针对拼接数据集中的某个数据集做了部分采样限制。按 Stage v2 的数据集列表顺序，索引 `4` 对应第 5 个数据集 `Cap3DImagePointDataset`。因此该配置很可能用于限制 Cap3D 的采样数量为 `90000`，防止其样本量过大影响多数据源训练平衡。

## Transform 差异

### Stage v1 已有 transform

Stage v1 定义了 3 类 transform：

| transform | 主要用途 |
| --- | --- |
| `outdoor_transform` | Waymo 等室外自动驾驶点云图像数据。 |
| `obj_transform` | PartNet 等归一化物体数据。 |
| `indoor_transform` | ScanNet、Structured3D 等室内场景数据。 |

Stage v2 保留了这三类 transform，主体内容基本一致。

### Stage v2 新增 `obj_realscale_withbg_transform`

该 transform 用于真实尺度、带背景的物体/多视角数据，Stage v2 中用于 GraspNet：

```python
transform=obj_realscale_withbg_transform
```

关键特征：

- 没有 `NormalizeCoord`，保留更接近真实尺度的坐标。
- 初始 `RandomScale` 使用 `[3.6, 4.4]`，尺度放大明显。
- `MultiViewGenerator` 中：
  - `global_view_scale=(0.4, 1.0)`
  - `local_view_scale=(0.1, 0.4)`
- 全局和局部 transform 都包含：
  - `CenterShift`
  - `RandomShift`
  - `RandomScale`
  - 三轴 `RandomRotate`
  - `RandomFlip`
  - `RandomJitter`
  - `ElasticDistortion`

### Stage v2 新增 `obj_realscale_nobg_transform`

该 transform 用于真实尺度、无背景的物体数据，Stage v2 中用于 ScanObjectNN：

```python
transform=obj_realscale_nobg_transform
```

关键特征：

- 初始 `RandomScale` 使用 `[0.9, 1.1]`。
- `MultiViewGenerator` 中：
  - `global_view_scale=(0.8, 1.0)`
  - `local_view_scale=(0.6, 0.8)`
- 相比 `obj_realscale_withbg_transform`，视图裁剪比例更大，更偏向保留完整物体。

### Stage v2 新增 `hk_transform`

该 transform 用于 HK 地图数据，Stage v2 中用于：

```python
type="HKDataset"
data_root="data/hk_3d_maps_N"
transform=hk_transform
```

关键特征：

- 先执行 `CenterShift`。
- 使用固定缩放：

```python
dict(type="RandomScale", scale=[0.01, 0.01])
```

- 随后再进行 `[0.9, 1.1]` 的随机缩放。
- `MultiViewGenerator` 中：

```python
global_view_scale=(0.4, 0.1)
local_view_scale=(0.1, 0.4)
```

注意：`global_view_scale=(0.4, 0.1)` 的上下界顺序看起来与常见写法相反。若该参数期望 `(min, max)`，这里可能需要额外确认实现是否支持这种顺序。

## 数据集差异

### Stage v1 数据集

Stage v1 训练数据由 4 个数据集拼接：

| 顺序 | 数据集 | 类型 | split | transform |
| --- | --- | --- | --- | --- |
| 0 | Waymo | `WaymoImagePointDataset` | `training`, `validation` | `outdoor_transform` |
| 1 | PartNet | `PartNetDataDataset` | `train` | `obj_transform` |
| 2 | ScanNet | `DefaultImagePointDataset` | `train`, `val`, `test` | `indoor_transform` |
| 3 | Structured3D | `DefaultImagePointDataset` | `train`, `val`, `test` | `indoor_transform` |

### Stage v2 数据集

Stage v2 训练数据扩展为 15 个数据集：

| 顺序 | 数据集 | 类型 | split | transform |
| --- | --- | --- | --- | --- |
| 0 | HK | `HKDataset` | `train` | `hk_transform` |
| 1 | NuScenes | `NuScenesImagePointDataset` | `train`, `val`, `test` | `outdoor_transform` |
| 2 | SemanticKITTI | `SemanticKITTIImagePointDataset` | `train`, `val`, `test` | `outdoor_transform` |
| 3 | Waymo | `WaymoImagePointDataset` | `training`, `validation` | `outdoor_transform` |
| 4 | Cap3D | `Cap3DImagePointDataset` | `train` | `obj_transform` |
| 5 | PartNet | `PartNetDataDataset` | `train` | `obj_transform` |
| 6 | GraspNet | `DefaultMultiViewImagePointDataset` | `train`, `val`, `test` | `obj_realscale_withbg_transform` |
| 7 | ScanObjectNN | `ScanObjectNNRawDataset` | `train` | `obj_realscale_nobg_transform` |
| 8 | ArkitScenes | `DefaultImagePointDataset` | `Training`, `Validation` | `indoor_transform` |
| 9 | ScanNet | `DefaultImagePointDataset` | `train`, `val`, `test` | `indoor_transform` |
| 10 | ScanNet++ | `DefaultImagePointDataset` | `train`, `val`, `test` | `indoor_transform` |
| 11 | S3DIS | `DefaultImagePointDataset` | `Area_1` - `Area_6` | `indoor_transform` |
| 12 | HM3D | `DefaultImagePointDataset` | `train`, `val` | `indoor_transform` |
| 13 | Structured3D | `DefaultImagePointDataset` | `train`, `val`, `test` | `indoor_transform` |
| 14 | RE10K | `DefaultImagePointDataset` | `train`, `test` | `indoor_transform` |

### 数据覆盖变化

Stage v2 相比 Stage v1 新增的数据源包括：

- 室外/自动驾驶：`NuScenes`、`SemanticKITTI`
- 地图数据：`HK`
- 物体数据：`Cap3D`、`GraspNet`、`ScanObjectNN`
- 室内数据：`ArkitScenes`、`ScanNet++`、`S3DIS`、`HM3D`、`RE10K`

Stage v1 中已有且 Stage v2 继续保留的数据源：

- `Waymo`
- `PartNet`
- `ScanNet`
- `Structured3D`

## 细节级差异

### 注释变更

Stage v1 的 `obj_transform` 中有一行被注释掉的 `ZShift`：

```python
# dict(type="ZShift", apply_center=True),
```

Stage v2 删除了这条注释。该变化不影响实际执行。

Stage v1 的 `indoor_transform` 中有一行注释：

```python
# view_keys=("coord", "origin_coord", "color", "normal"),
```

Stage v2 删除了该注释。该变化同样不影响实际执行。

### Waymo 注释变化

Stage v1 中 Waymo 注释为：

```python
# Waymo 10 Hz
```

Stage v2 中改为：

```python
# Waymo
```

实际 Waymo 配置仍为：

```python
sweeps=3
sweep_gap=1
```

## 结论

Stage v1 是一个较精简的多源预训练配置，覆盖 Waymo、PartNet、ScanNet 和 Structured3D 四类代表性数据。

Stage v2 是一个更大规模、更混合域的预训练配置：模型和优化超参没有变化，但引入了 `PartialSampledTrainer`、数据采样限制、3 个新增 transform，以及 11 个额外数据源。它更适合进行跨室外、室内、物体、地图、多视角数据的统一预训练。

如果只关心模型结构或学习率策略，两个配置几乎没有区别；如果关心训练数据分布和预训练覆盖域，Stage v2 与 Stage v1 差异非常大。
