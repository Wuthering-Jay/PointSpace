# EZ-SP Stage1 代码级分析报告

## 1. 结论先行

EZ-SP 的 `stage1` 是一个**独立训练的点特征学习阶段**，目标不是直接输出语义类别，而是学习一个适合后续超点分区的低维嵌入空间。

更准确地说：

1. 输入是经过体素化后的 `Data`，在 `partition` 配置里默认用 `partition_hf` 作为 CNN 输入特征。  
   见 [configs/datamodule/partition/default_ezsp.yaml:25](e:/code/python/PointSpace/reference_code/superpoint_transformer/configs/datamodule/partition/default_ezsp.yaml:25), [configs/datamodule/partition/default_ezsp.yaml:47](e:/code/python/PointSpace/reference_code/superpoint_transformer/configs/datamodule/partition/default_ezsp.yaml:47)
2. 主体网络是一个带 sparse CNN 的 `PointStage`，输出的是每个体素/点的嵌入向量 `x`。  
   见 [src/nn/stage.py:574](e:/code/python/PointSpace/reference_code/superpoint_transformer/src/nn/stage.py:574)
3. 训练目标是**边级别的二分类亲和力学习**：同类边亲和力高，异类边亲和力低。  
   见 [src/loss/partition_criterion.py:13](e:/code/python/PointSpace/reference_code/superpoint_transformer/src/loss/partition_criterion.py:13), [src/loss/partition_criterion.py:75](e:/code/python/PointSpace/reference_code/superpoint_transformer/src/loss/partition_criterion.py:75)
4. 训练得到的 checkpoint 不直接拿来做语义分类，而是被后续 `PretrainedCNN` 和 stage2 的第一阶段加载复用。  
   见 [src/transforms/point.py:630](e:/code/python/PointSpace/reference_code/superpoint_transformer/src/transforms/point.py:630), [src/models/semantic.py:255](e:/code/python/PointSpace/reference_code/superpoint_transformer/src/models/semantic.py:255)

---

## 2. Stage1 在工程中的位置

Stage1 不是一个单独的 `LightningModule` 类，而是通过：

- `PartitionAndSemanticModule(training_partition_stage=True)` 进入分区训练模式  
  [src/models/semantic.py:1400](e:/code/python/PointSpace/reference_code/superpoint_transformer/src/models/semantic.py:1400), [src/models/semantic.py:1453](e:/code/python/PointSpace/reference_code/superpoint_transformer/src/models/semantic.py:1453)
- 配置文件 `configs/model/partition/default_ezsp.yaml` 组织网络、分区器和 partition loss  
  [configs/model/partition/default_ezsp.yaml:1](e:/code/python/PointSpace/reference_code/superpoint_transformer/configs/model/partition/default_ezsp.yaml:1)

对应训练命令：

```bash
python src/train.py experiment=partition/<dataset>_ezsp
```

见 [README.md:464](e:/code/python/PointSpace/reference_code/superpoint_transformer/README.md:464)

---

## 3. Stage1 的训练目标

## 3.1 目标直觉

README 对 stage1 的描述是：

- 训练一个小型模型，学习适合分区的点特征
- 在语义边界处做对比式学习

见 [README.md:448](e:/code/python/PointSpace/reference_code/superpoint_transformer/README.md:448)

代码中实际实现是：

- 用嵌入后的点特征 `x` 计算边亲和力 `exp(-||x_i - x_j|| / T)`
- 用 ground-truth 语义标签判断边是 `inter-edge` 还是 `intra-edge`
- 通过二值损失优化嵌入空间

见 [src/loss/partition_criterion.py:20](e:/code/python/PointSpace/reference_code/superpoint_transformer/src/loss/partition_criterion.py:20), [src/loss/partition_criterion.py:120](e:/code/python/PointSpace/reference_code/superpoint_transformer/src/loss/partition_criterion.py:120), [src/loss/partition_criterion.py:240](e:/code/python/PointSpace/reference_code/superpoint_transformer/src/loss/partition_criterion.py:240)

## 3.2 代码里的目标形式

`PartitionCriterion` 的关键逻辑：

```python
target_affinity = (y[edge_index[0]] == y[edge_index[1]]).int()
predicted_affinity = exp(-distance / temperature)
loss = BinaryFocalLoss(predicted_affinity, target_affinity.bool())
```

对应代码：

- `target_affinity` 由相邻节点是否同类定义  
  [src/loss/partition_criterion.py:120](e:/code/python/PointSpace/reference_code/superpoint_transformer/src/loss/partition_criterion.py:120)
- `predicted_affinity` 由特征距离映射得到  
  [src/loss/partition_criterion.py:240](e:/code/python/PointSpace/reference_code/superpoint_transformer/src/loss/partition_criterion.py:240)

---

## 4. Stage1 网络结构

## 4.1 顶层网络

Stage1 使用的是 `SPT` 的特化配置：

- `spt-0.yaml`：没有 down/up 层，只保留 `PointStage`
  [configs/model/semantic/spt-0.yaml:1](e:/code/python/PointSpace/reference_code/superpoint_transformer/configs/model/semantic/spt-0.yaml:1)
- `partition/default_ezsp.yaml` 再叠加 `_point_cnn.yaml`，把 `PointStage` 改成 sparse CNN 版本  
  [configs/model/partition/default_ezsp.yaml:1](e:/code/python/PointSpace/reference_code/superpoint_transformer/configs/model/partition/default_ezsp.yaml:1), [configs/model/partition/_point_cnn.yaml:1](e:/code/python/PointSpace/reference_code/superpoint_transformer/configs/model/partition/_point_cnn.yaml:1)

## 4.2 `PointStage`

`PointStage` 是 stage1 的核心：

- 如果 `cnn_blocks=True`，先用 sparse CNN 处理输入特征
- 再决定是否接一个 MLP
- 最后输出用于分区的点嵌入 `x`

见 [src/nn/stage.py:574](e:/code/python/PointSpace/reference_code/superpoint_transformer/src/nn/stage.py:574)

### 输入

- `x`: `[N, C_in]`
- `coords`: `[N, 3]`，仅 sparse CNN 需要
- `batch`: `[N]`
- `pos`, `super_index`, `edge_index`, `edge_attr`

见 [src/nn/stage.py:724](e:/code/python/PointSpace/reference_code/superpoint_transformer/src/nn/stage.py:724)

这里可以更精确地区分一下每个输入的作用：

- `x` 是当前层节点特征
- `coords` 是 sparse CNN 的量化坐标
- `pos` 是几何位置，用于 `UnitSphereNorm`
- `super_index` 是当前节点到父节点的映射
- `edge_index` / `edge_attr` 是当前层图结构及其边特征

对 stage1 来说，`coords` 只有在 `cnn_blocks=True` 时才会被 sparse CNN 使用；`pos` 和 `super_index` 虽然会被传入，但在默认 EZ-SP 配置里并不会进入特征拼接路径。

### 输出

- `x_out`: `[N, C_out]`
- `diameter_parent`

见 [src/nn/stage.py:806](e:/code/python/PointSpace/reference_code/superpoint_transformer/src/nn/stage.py:806)

`diameter_parent` 是由 `UnitSphereNorm` 估计出来的“父层节点直径”。  
它的逻辑是：

- 如果传了 `super_index`，就按父节点分组计算每个父节点的尺度
- 如果没有 `super_index`，就退化成对整批节点的全局尺度

对应实现见 [src/nn/stage.py:183](e:/code/python/PointSpace/reference_code/superpoint_transformer/src/nn/stage.py:183), [src/nn/stage.py:216](e:/code/python/PointSpace/reference_code/superpoint_transformer/src/nn/stage.py:216)

在 stage1 的默认配置里：

- `use_pos: False`
- `use_diameter_parent: False`

所以这些量虽然被计算，但不会拼进输入特征里。  
见 [configs/model/partition/default_ezsp.yaml:47](e:/code/python/PointSpace/reference_code/superpoint_transformer/configs/model/partition/default_ezsp.yaml:47)

## 4.3 Stage1 的 sparse CNN 配置

stage1 的 `_point_cnn.yaml` 指定：

- `point_cnn_blocks: True`
- `point_mlp: null`
- `point_mlp_on_cnn_feats: False`

见 [configs/model/partition/_point_cnn.yaml:13](e:/code/python/PointSpace/reference_code/superpoint_transformer/configs/model/partition/_point_cnn.yaml:13)

这意味着 stage1 的第一阶段更接近：

```text
input features -> sparse CNN -> point embedding
```

而不是完整的“CNN + MLP 并行”语义版结构。

---

## 5. Stage1 的预处理链路

Stage1 的预处理与 stage2 很像，但有一个关键不同点：

- `point_hf` 会被设置为 `partition_hf`
- 即，CNN 学的是专门为了分区服务的输入特征

见 [configs/datamodule/partition/default_ezsp.yaml:25](e:/code/python/PointSpace/reference_code/superpoint_transformer/configs/datamodule/partition/default_ezsp.yaml:25)

## 5.1 输入数据的实际形态

stage1 在 `pre_transform` 中先做：

1. `GridSampling3D`：原始点云 -> 体素 `Data`
2. `PointFeatures` / `GroundElevation`：补充点特征
3. `KNN + AdjacencyGraph`：建立分区训练用的邻接图
4. `RemoveKeys`：去掉暂不需要的原始边属性

见 [configs/datamodule/partition/default_ezsp.yaml:31](e:/code/python/PointSpace/reference_code/superpoint_transformer/configs/datamodule/partition/default_ezsp.yaml:31)

这里最需要澄清的是层级：

- `GridSampling3D` 之后，`Data` 的节点已经是体素/栅格聚合节点
- `PointFeatures`、`GroundElevation`、`KNN`、`AdjacencyGraph` 都是在这些体素节点上继续补特征、找邻域和建边
- sparse CNN 读取的 `coords` 也是这些量化后的节点坐标，而不是原始 raw points

因此 stage1 的“点嵌入”更准确地说是**体素节点嵌入**；真正的超点是在后面的分区 transform 里由这些体素节点进一步合并出来的。

### `GridSampling3D` 的意义

`GridSampling3D` 会把原始点聚成体素，并把标签聚合成 histogram：

- `Data.y` 不是单标签，而是 `(num_voxels, num_classes+1)` 的直方图形式
- 最后一列是 void 类

见 [src/transforms/sampling.py:86](e:/code/python/PointSpace/reference_code/superpoint_transformer/src/transforms/sampling.py:86)

这也解释了为什么后续 `PartitionCriterion` 不是对原始点做监督，而是先把体素节点的 label histogram 压成多数类，再做边级二值亲和力学习。

### `PointFeatures` 和 `GroundElevation` 具体在算什么

`PointFeatures` 支持的键包括：

- `rgb`, `hsv`, `lab`
- `density`
- `linearity`, `planarity`, `scattering`, `verticality`
- `normal`, `length`, `surface`, `volume`, `curvature`

见 [src/transforms/point.py:18](e:/code/python/PointSpace/reference_code/superpoint_transformer/src/transforms/point.py:18)

它们都是当前 `Data` 节点级特征，不是超点级特征。由于此时已经经历 `GridSampling3D`，所以默认是在体素节点上计算和保存的。

`GroundElevation` 则额外写入 `data.elevation`。它会先通过 `z_threshold`、`verticality_threshold`、`xy_grid` 过滤掉尽可能多的非地面节点，再用 `ransac` / `knn` / `mlp` 拟合地面表面，最后得到每个节点相对地面的高度差。  
见 [src/transforms/point.py:148](e:/code/python/PointSpace/reference_code/superpoint_transformer/src/transforms/point.py:148)

## 5.2 `on_device_train_transform`

stage1 的 on-device 训练增强主要包括：

- `SampleRadiusSubgraphs`：限制子图规模
- `NAGJitterKey`、`DropoutColumns`、`DropoutRows`：增强点/边特征
- `ColorDrop` / `ColorAutoContrast`：颜色增强
- `KNN + AdjacencyGraph`：构建训练边

见 [configs/datamodule/partition/default_ezsp.yaml:52](e:/code/python/PointSpace/reference_code/superpoint_transformer/configs/datamodule/partition/default_ezsp.yaml:52)

这里的“耗时”问题，代码其实是有明确工程处理的：

1. `pre_transform` 一开始就有 `DataTo(device='cuda')`，所以这条链路默认是在 GPU 上做
2. `on_device_train_transform` 本身也是 GPU 侧的增强和建图，不是把重操作丢回 CPU

见 [configs/datamodule/partition/default_ezsp.yaml:31](e:/code/python/PointSpace/reference_code/superpoint_transformer/configs/datamodule/partition/default_ezsp.yaml:31)

另外，分区合并的“官方快速实现”不是老式纯 CPU cut-pursuit，而是借助 `torch_graph_components` 的图组件操作：

- `wcc_by_max_propagation`
- `merge_components_by_contour_prior`
- `component_graph`

见 [src/utils/components.py:2](e:/code/python/PointSpace/reference_code/superpoint_transformer/src/utils/components.py:2)

所以 EZ-SP 的快速性更多来自：

- sparse CNN 先在体素节点上快速抽特征
- 图构建和组件合并走 GPU/高效图实现
- 训练目标直接是边级亲和力，而不是昂贵的逐点复杂结构推理

---

## 6. Stage1 的前向与损失

## 6.1 分区训练模式的前向

`PartitionAndSemanticModule.forward()` 在 `training_partition_stage=True` 时：

1. 把 `self.net.point_hf` 加入 `sample.x`
2. 调用 `self.net.forward_first_stage(...)`
3. 把输出嵌入 `x` 写回 `sample.x`
4. 必要时调用 `self.partition(sample)` 生成硬分区
5. 返回 `PartitionOutput`

见 [src/models/semantic.py:1512](e:/code/python/PointSpace/reference_code/superpoint_transformer/src/models/semantic.py:1512)

### 代码片段

```python
sample.add_keys_to(keys=self.net.point_hf, to='x', delete_after=not self.net.store_features)
x, diameter = self.net.forward_first_stage(...)
sample.x = x
nag = self.partition(sample) if needed else None
return PartitionOutput(y=sample.y, x=sample.x, edge_index=sample.edge_index, partition=...)
```

对应代码：

- `add_keys_to` 和 `forward_first_stage`  
  [src/models/semantic.py:1513](e:/code/python/PointSpace/reference_code/superpoint_transformer/src/models/semantic.py:1513)
- `PartitionOutput` 构造  
  [src/models/semantic.py:1539](e:/code/python/PointSpace/reference_code/superpoint_transformer/src/models/semantic.py:1539)

## 6.2 `PartitionOutput` 的格式

- `y`: 节点标签 histogram，形状 `[N, num_classes+1]`
- `x`: 节点嵌入，形状 `[N, D]`
- `edge_index`: 图边 `[2, E]`
- `partition`: 可选，硬分区得到的 `NAG`

见 [src/utils/output_partition.py:10](e:/code/python/PointSpace/reference_code/superpoint_transformer/src/utils/output_partition.py:10)

## 6.3 损失函数细节

`PartitionCriterion` 会：

1. 从 `y` 中取多数类作为节点标签
2. 删除自环和纯 void 节点边
3. 计算边亲和力目标 `target_affinity`
4. 用 `exp(-||x_i-x_j||/T)` 得到预测亲和力
5. 用 `BinaryFocalLoss` 回传梯度

见 [src/loss/partition_criterion.py:92](e:/code/python/PointSpace/reference_code/superpoint_transformer/src/loss/partition_criterion.py:92)

这基本可以直接回答“损失是不是边上的二值化损失”这个问题：**是的，默认就是边级二值亲和力损失**。

更细一点说：

- 正样本边 = 两端节点属于同类，`target_affinity=1`
- 负样本边 = 两端节点属于不同类，`target_affinity=0`
- 预测值不是分类 logits，而是特征距离经过 `exp(-d/T)` 映射得到的亲和力
- `BinaryFocalLoss` 用来缓解正负边极不平衡以及难样本占比低的问题

默认 stage1 路径里没有再额外叠加别的损失项，所以这条训练线的主损失就是这个边级 focal loss。

### 关键输入输出

输入：

- `partition_output.y`: `[N, C+1]`
- `partition_output.x`: `[N, D]`
- `partition_output.edge_index`: `[2, E]`

输出：

- `loss`: 标量
- `partition_output.n_inter_edge`: 统计值

见 [src/loss/partition_criterion.py:62](e:/code/python/PointSpace/reference_code/superpoint_transformer/src/loss/partition_criterion.py:62)

## 6.4 为什么训练时经常不显式算 partition

stage1 的 `partition_during_training` 默认是 `False`。  
这意味着训练步主要优化嵌入和边亲和力，不一定每个 batch 都构造硬分区。

见 [configs/model/partition/default_ezsp.yaml:20](e:/code/python/PointSpace/reference_code/superpoint_transformer/configs/model/partition/default_ezsp.yaml:20), [src/models/semantic.py:1527](e:/code/python/PointSpace/reference_code/superpoint_transformer/src/models/semantic.py:1527)

验证时会算 partition，用来评估 partition purity、`n_sp`、`points_per_superpoint` 等指标。  
见 [src/models/semantic.py:1633](e:/code/python/PointSpace/reference_code/superpoint_transformer/src/models/semantic.py:1633)

### 不平衡是怎么处理的

你提到的 `inter-edge` 远少于 `intra-edge`，代码里是有专门处理的：

1. `BinaryFocalLoss(gamma=1)` 会降低大量容易样本对训练的支配
2. `adaptive_sampling_ratio: 0.9` 会做边样本自适应采样，尽量把训练分布拉向更平衡
3. 如果某个 batch 里没有有效边，或者没有 `inter-edge`，`fake_edge_classification_loss` 会返回一个可反传的 0 loss，避免训练中断
4. 如果整个 epoch 都没有 inter-edge，`on_train_epoch_end` 会直接报错，防止训练实际上退化成“只看同类边”

见 [src/loss/partition_criterion.py:92](e:/code/python/PointSpace/reference_code/superpoint_transformer/src/loss/partition_criterion.py:92), [src/loss/partition_criterion.py:131](e:/code/python/PointSpace/reference_code/superpoint_transformer/src/loss/partition_criterion.py:131), [src/loss/partition_criterion.py:165](e:/code/python/PointSpace/reference_code/superpoint_transformer/src/loss/partition_criterion.py:165), [src/models/semantic.py:1594](e:/code/python/PointSpace/reference_code/superpoint_transformer/src/models/semantic.py:1594), [src/models/semantic.py:1598](e:/code/python/PointSpace/reference_code/superpoint_transformer/src/models/semantic.py:1598)

---

## 7. Stage1 的分区器本体

stage1 的硬分区用的是 `GreedyContourPriorPartition`：

```python
partition:
  _target_: src.transforms.partition.GreedyContourPriorPartition
```

见 [configs/model/partition/default_ezsp.yaml:25](e:/code/python/PointSpace/reference_code/superpoint_transformer/configs/model/partition/default_ezsp.yaml:25)

它的输入是：

- `Data.pos`
- `Data.x`
- `Data.edge_index`

见 [src/transforms/partition.py:391](e:/code/python/PointSpace/reference_code/superpoint_transformer/src/transforms/partition.py:391)

它的输出是：

- 一个 `NAG`

见 [src/transforms/partition.py:517](e:/code/python/PointSpace/reference_code/superpoint_transformer/src/transforms/partition.py:517)

### 代码层含义

它先：

1. 根据 `edge_weight_mode` 计算边权
2. 必要时把位置 `pos` 拼接进 `x`
3. 调用 `merge_components_by_contour_prior_on_data(...)`
4. 逐层构造 `NAG`

见 [src/transforms/partition.py:524](e:/code/python/PointSpace/reference_code/superpoint_transformer/src/transforms/partition.py:524)

---

## 8. Stage1 的训练输出与保存

`PartitionAndSemanticModule` 在 stage1 模式下会：

- 删除语义 head
- 记录分区损失与 `n_inter_edge`
- 在验证/测试时记录 partition purity 指标

见 [src/models/semantic.py:1482](e:/code/python/PointSpace/reference_code/superpoint_transformer/src/models/semantic.py:1482), [src/models/semantic.py:1570](e:/code/python/PointSpace/reference_code/superpoint_transformer/src/models/semantic.py:1570)

此外，checkpoint 会保存版本信息：

- `__version__`
- `commit_hash`

见 [src/models/semantic.py:1336](e:/code/python/PointSpace/reference_code/superpoint_transformer/src/models/semantic.py:1336)

---

## 9. Stage1 与 Stage2 的权重是怎么“连起来”的

这是你问的重点，代码上不是“共享一个训练图”，而是**checkpoint 复用**。

## 9.1 连接点一：预处理时复用 stage1 权重

在 stage2 的数据预处理里，`PretrainedCNN` 会：

1. 读取 `datamodule.pretrained_cnn_ckpt_path`
2. 构造一个与 stage1 兼容的 `PointStage`
3. 从 checkpoint 中只提取 `net.first_stage.*` 的参数
4. 用这些参数计算 `data.x`

见 [configs/datamodule/semantic/default_ezsp.yaml:67](e:/code/python/PointSpace/reference_code/superpoint_transformer/configs/datamodule/semantic/default_ezsp.yaml:67), [src/transforms/point.py:687](e:/code/python/PointSpace/reference_code/superpoint_transformer/src/transforms/point.py:687), [src/transforms/point.py:720](e:/code/python/PointSpace/reference_code/superpoint_transformer/src/transforms/point.py:720)

### 关键代码

```python
first_stage_keys = [k for k in checkpoint['state_dict'].keys() if 'first_stage' in k]
ckpt_dict = {k.replace('net.first_stage.', ''): checkpoint['state_dict'][k] for k in first_stage_keys}
first_stage.load_state_dict(model_dict)
```

见 [src/transforms/point.py:721](e:/code/python/PointSpace/reference_code/superpoint_transformer/src/transforms/point.py:721)

这说明：

- stage1 checkpoint 中真正被 stage2 预处理复用的是 `first_stage` 这部分
- 不是整个 `PartitionAndSemanticModule`

## 9.2 连接点二：stage2 语义网络初始化时复用同一份权重

stage2 语义训练时，如果不从头训练 CNN，则：

1. 读取 `pretrained_cnn_ckpt_path`
2. 调 `PretrainedCNN.load_checkpoint(...)`
3. 把同一份 `first_stage` 权重加载到语义模型的 `self.net.first_stage`

见 [src/models/semantic.py:261](e:/code/python/PointSpace/reference_code/superpoint_transformer/src/models/semantic.py:261), [src/models/semantic.py:279](e:/code/python/PointSpace/reference_code/superpoint_transformer/src/models/semantic.py:279)

### 代码片段

```python
self.net.first_stage = PretrainedCNN.load_checkpoint(
    self.net.first_stage,
    ckpt_path,
    self.device,
    verbose=False)
```

见 [src/models/semantic.py:279](e:/code/python/PointSpace/reference_code/superpoint_transformer/src/models/semantic.py:279)

## 9.3 stage2 默认是冻结还是可训练

默认配置里：

- `train_cnn_from_scratch: False`
- `freeze_cnn: False`

见 [configs/model/semantic/_point_cnn.yaml:5](e:/code/python/PointSpace/reference_code/superpoint_transformer/configs/model/semantic/_point_cnn.yaml:5)

这意味着 stage2 的 CNN：

- 先用 stage1 checkpoint 初始化
- 默认情况下**不冻结**
- 仍然可以参与 stage2 的联合微调

如果你显式设 `freeze_cnn=True`，代码会冻结 `cnn_blocks`。  
见 [src/models/semantic.py:285](e:/code/python/PointSpace/reference_code/superpoint_transformer/src/models/semantic.py:285)

---

## 10. 训练和推理时的权重关系总结

可以把它理解成下面三种角色：

1. `stage1` 训练出的 checkpoint = “分区嵌入器”的来源
2. `PretrainedCNN` = “离线算分区输入特征”的使用者
3. stage2 的 `self.net.first_stage` = “语义网络第一阶段初始化”的使用者

它们的关系是：

```text
stage1 checkpoint
   -> PretrainedCNN (预处理生成 partition_hf embeddings)
   -> stage2 first_stage 初始化 (可继续微调或冻结)
```

不是：

- 不共享 optimizer state
- 不共享训练图
- 不在一个 step 中端到端联合更新

---

## 11. 输入到底是点、体素还是超点

stage1 的答案要分层说：

1. 原始数据源是点云（raw points）
2. 进入 stage1 训练/预处理前先经过 `GridSampling3D`，所以直接送入网络和分区器的是**体素节点**
3. 分区完成后构造的是 `NAG`，它的 `level-1+` 是**超点**

所以更准确的说法是：

**stage1 的网络输入是体素化后的点集/图原子层，训练目标是学出适合超点分区的嵌入；真正的超点是在后面的 partition transform 里构造出来的。**

---

## 12. 一份最小的关键代码索引

### 12.1 Stage1 训练开关

```yaml
# configs/model/partition/default_ezsp.yaml
training_partition_stage: True
```

### 12.2 分区损失

```python
# src/loss/partition_criterion.py
target_affinity = (y[edge_index[0]] == y[edge_index[1]]).int()
predicted_affinity = torch.exp(-distances / self.affinity_temperature)
loss = self.loss_function(predicted_affinity, target_affinity.bool())
```

### 12.3 Stage1 checkpoint 复用到 stage2 预处理

```python
# src/transforms/point.py
ckpt_dict = {
    k.replace('net.first_stage.', ""): checkpoint['state_dict'][k]
    for k in first_stage_keys}
```

### 12.4 Stage1 checkpoint 复用到 stage2 网络初始化

```python
# src/models/semantic.py
self.net.first_stage = PretrainedCNN.load_checkpoint(...)
```
