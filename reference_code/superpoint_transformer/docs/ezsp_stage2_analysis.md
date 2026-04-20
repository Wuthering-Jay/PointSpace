# EZ-SP Stage2 代码级分析报告

## 1. 报告范围与结论先行

本报告聚焦 **EZ-SP 的第二阶段（semantic stage）**，即：

- 配置上 `training_partition_stage: False` 的训练/推理流程  
  见 [configs/experiment/semantic/default_ezsp.yaml:8](e:/code/python/PointSpace/reference_code/superpoint_transformer/configs/experiment/semantic/default_ezsp.yaml:8)
- 模型类为 `PartitionAndSemanticModule`，但在该模式下走 `SemanticSegmentationModule` 语义分支  
  见 [src/models/semantic.py:1400](e:/code/python/PointSpace/reference_code/superpoint_transformer/src/models/semantic.py:1400), [src/models/semantic.py:1545](e:/code/python/PointSpace/reference_code/superpoint_transformer/src/models/semantic.py:1545)

一句话结论：

1. **输入网络的直接对象是 `NAG`（层级分区图）**，不是原始点云张量。  
2. `NAG` 的 **level-0 是体素点（voxel）**，`level-1+` 是超点层级。  
3. Stage2 的主体是 **UNet-like 的 Superpoint Transformer**，在超点图上做 attention（用 `edge_index/edge_attr`），并通过层级关系做 down/up。  

---

## 2. Stage2 全流程（从原始点到最终语义）

```text
原始点云(raw points)
  -> GridSampling3D 体素化 (P0)
  -> PretrainedCNN 生成分区嵌入
  -> GreedyContourPriorPartition 生成层级超点 NAG (P1/P2/P3...)
  -> SegmentFeatures + RadiusHorizontalGraph 构建超点特征与图
  -> (训练时) on_device_* 采样/增强/边特征补全
  -> SPT(UNet-like, graph transformer) 输出 level-1(superpoint) logits
  -> 语义损失(CE/KL系列, 基于 y_hist)
  -> 推理时可映射到 voxel/full-res 点
```

核心证据：

- pre-transform 链路（含 `GridSampling3D`, `PretrainedCNN`, `GreedyContourPriorPartition`）  
  [configs/datamodule/semantic/default_ezsp.yaml:32](e:/code/python/PointSpace/reference_code/superpoint_transformer/configs/datamodule/semantic/default_ezsp.yaml:32)
- `SPT` 明确是 “UNet-like architecture processing NAG”  
  [src/models/components/spt.py:15](e:/code/python/PointSpace/reference_code/superpoint_transformer/src/models/components/spt.py:15)
- 推理输出分发到 voxel/full-res  
  [src/utils/output_semantic.py:114](e:/code/python/PointSpace/reference_code/superpoint_transformer/src/utils/output_semantic.py:114), [src/utils/output_semantic.py:139](e:/code/python/PointSpace/reference_code/superpoint_transformer/src/utils/output_semantic.py:139)

---

## 3. 数据结构与输入格式

## 3.1 `NAG` 与层级语义

- `NAG` 是层级 `Data` 列表，`Data.super_index` 表示 `P_i -> P_{i+1}` 映射，`Data.sub` 表示反向聚合关系。  
  [docs/data_structures.md:24](e:/code/python/PointSpace/reference_code/superpoint_transformer/docs/data_structures.md:24), [docs/data_structures.md:41](e:/code/python/PointSpace/reference_code/superpoint_transformer/docs/data_structures.md:41)
- `NAG` 使用绝对层级索引，`start_i_level` 指明起始层。  
  [docs/data_structures.md:46](e:/code/python/PointSpace/reference_code/superpoint_transformer/docs/data_structures.md:46)

## 3.2 点 / 体素 / 超点在 Stage2 中的角色

- `GridSampling3D`：把 3D 点聚为体素节点（P0）。  
  [src/transforms/sampling.py:86](e:/code/python/PointSpace/reference_code/superpoint_transformer/src/transforms/sampling.py:86)
- README 说明：训练主要监督 P1 超点，推理可回到 P0 体素和 full-res。  
  [README.md:543](e:/code/python/PointSpace/reference_code/superpoint_transformer/README.md:543)

因此：

1. **原始输入文件是点云**。  
2. **网络前的计算对象先变成体素（P0）并构建超点层级（P1+）**。  
3. **语义主干直接处理的是超点层级图（NAG）**。  

---

## 4. 输入网络前的预处理组件（Stage2）

以下来自 `configs/datamodule/semantic/default_ezsp.yaml`，按执行顺序给出关键 I/O。

## 4.1 `pre_transform`（离线/预处理阶段）

入口：单个 `Data`（原始点）  
出口：`NAG`（已分层、已构图）

关键步骤：

1. `GridSampling3D`：点 -> 体素聚合（保留标签直方图等）  
   [configs/datamodule/semantic/default_ezsp.yaml:39](e:/code/python/PointSpace/reference_code/superpoint_transformer/configs/datamodule/semantic/default_ezsp.yaml:39), [src/transforms/sampling.py:86](e:/code/python/PointSpace/reference_code/superpoint_transformer/src/transforms/sampling.py:86)
2. `PretrainedCNN`：用 stage1 checkpoint 计算分区嵌入，写入 `data.x`  
   [configs/datamodule/semantic/default_ezsp.yaml:67](e:/code/python/PointSpace/reference_code/superpoint_transformer/configs/datamodule/semantic/default_ezsp.yaml:67), [src/transforms/point.py:630](e:/code/python/PointSpace/reference_code/superpoint_transformer/src/transforms/point.py:630)
3. `GreedyContourPriorPartition`：基于 `x + edge_index (+pos)` 生成层级超点 `NAG`  
   [configs/datamodule/semantic/default_ezsp.yaml:90](e:/code/python/PointSpace/reference_code/superpoint_transformer/configs/datamodule/semantic/default_ezsp.yaml:90), [src/transforms/partition.py:383](e:/code/python/PointSpace/reference_code/superpoint_transformer/src/transforms/partition.py:383)
4. `SegmentFeatures`：计算超点 handcrafted 特征  
   [configs/datamodule/semantic/default_ezsp.yaml:107](e:/code/python/PointSpace/reference_code/superpoint_transformer/configs/datamodule/semantic/default_ezsp.yaml:107)
5. `RadiusHorizontalGraph`：构建 level-1+ 的超点水平图边  
   [configs/datamodule/semantic/default_ezsp.yaml:115](e:/code/python/PointSpace/reference_code/superpoint_transformer/configs/datamodule/semantic/default_ezsp.yaml:115)

示例代码片段（配置）：

```yaml
# configs/datamodule/semantic/default_ezsp.yaml
- transform: GridSampling3D
- transform: PretrainedCNN
- transform: GreedyContourPriorPartition
- transform: SegmentFeatures
- transform: RadiusHorizontalGraph
```

## 4.2 `on_device_train_transform`（在线训练增强）

入口：`NAG`  
出口：采样/增强后的 `NAG`

关键影响：

1. `SampleRadiusSubgraphs(i_level=1)`、`SampleSegments`：对子图和超点采样，控制训练显存与随机性。  
   [configs/datamodule/semantic/default_ezsp.yaml:152](e:/code/python/PointSpace/reference_code/superpoint_transformer/configs/datamodule/semantic/default_ezsp.yaml:152)
2. `OnTheFlyHorizontalEdgeFeatures` + `OnTheFlyVerticalEdgeFeatures`：在线构造 `edge_attr`/`v_edge_attr`。  
   [configs/datamodule/semantic/default_ezsp.yaml:196](e:/code/python/PointSpace/reference_code/superpoint_transformer/configs/datamodule/semantic/default_ezsp.yaml:196)
3. `NAGAddSelfLoops`：给水平图加自环。  
   [configs/datamodule/semantic/default_ezsp.yaml:289](e:/code/python/PointSpace/reference_code/superpoint_transformer/configs/datamodule/semantic/default_ezsp.yaml:289)

## 4.3 预训练 CNN 与分区参数（数据集实例）

S3DIS EZ-SP 语义配置示例：

- `pretrained_cnn_dim_without_in_dim: [32,32,32]`
- `contour_prior_min_size: [5,30,90]`（3级分区）
- `partition_hf: ['rgb']`

见 [configs/datamodule/semantic/s3dis_ezsp.yaml:10](e:/code/python/PointSpace/reference_code/superpoint_transformer/configs/datamodule/semantic/s3dis_ezsp.yaml:10)

---

## 5. Stage2 网络结构（SPT）与模块 I/O

## 5.1 顶层模型关系

- 语义模型是 `PartitionAndSemanticModule`，stage2 时 `training_partition_stage=False`。  
  [configs/experiment/semantic/default_ezsp.yaml:8](e:/code/python/PointSpace/reference_code/superpoint_transformer/configs/experiment/semantic/default_ezsp.yaml:8)
- 网络主体 `net` 是 `src.models.components.spt.SPT`。  
  [configs/model/semantic/spt.yaml:10](e:/code/python/PointSpace/reference_code/superpoint_transformer/configs/model/semantic/spt.yaml:10)
- `spt-3.yaml` 指定 3-level 下采样与 2-level 上采样的 UNet 深度。  
  [configs/model/semantic/spt-3.yaml:6](e:/code/python/PointSpace/reference_code/superpoint_transformer/configs/model/semantic/spt-3.yaml:6)

## 5.2 `SPT.forward(nag)` 的输入输出

输入：

- `nag: NAG`，其中  
  - level-0：体素节点（包含 `pos`, `coords`, `x` 等）  
  - level-1+：超点节点和超点图边（`edge_index`, `edge_attr`, `v_edge_attr`）

证据：`SPT.forward` 内部按 level0 和 `'1+'` 分别 `add_keys_to`。  
[src/models/components/spt.py:770](e:/code/python/PointSpace/reference_code/superpoint_transformer/src/models/components/spt.py:770)

输出：

- 默认：一个张量 `x`（通常对应 level-1 节点特征）  
- `output_stage_wise=True` 时：多层 list 输出

见 [src/models/components/spt.py:875](e:/code/python/PointSpace/reference_code/superpoint_transformer/src/models/components/spt.py:875)

## 5.3 子模块与 I/O 细节

### A. `PointStage`（level-0）

输入（典型）：

- `x`: `[N0, C0]`（点/体素特征）
- `coords`: `[N0, 3]`（若启用 sparse CNN）
- `batch`: `[N0]`
- `pos`, `super_index`, `edge_index`, `edge_attr`（供注入/归一化）

输出：

- `x0_out`: `[N0, C0']`
- `diameter_parent`: `[N1, 1]` 或相应形态

见 [src/nn/stage.py:724](e:/code/python/PointSpace/reference_code/superpoint_transformer/src/nn/stage.py:724)

### B. `DownNFuseStage`（i -> i+1）

输入：

- `x_child`（下层节点特征）
- `x_parent`（上层节点 handcrafted /已有特征）
- `pool_index`（`super_index`，child 到 parent 映射）
- `edge_index/edge_attr`（当前层水平图）
- `v_edge_attr`（垂直边特征）

输出：

- 上层更新后的 `x_parent_out`

见 [src/nn/stage.py:413](e:/code/python/PointSpace/reference_code/superpoint_transformer/src/nn/stage.py:413)

### C. `UpNFuseStage`（i+1 -> i）

输入：

- `x_parent`（高层特征）
- `x_child`（低层 skip）
- `unpool_index`（通常是 `super_index`）
- `edge_index/edge_attr`

输出：

- 低层恢复后的 `x_child_out`

见 [src/nn/stage.py:545](e:/code/python/PointSpace/reference_code/superpoint_transformer/src/nn/stage.py:545)

### D. `Stage/TransformerBlock`（图注意力核心）

`Stage` 内部把 `edge_index/edge_attr` 送入 `TransformerBlock`：  
[src/nn/stage.py:277](e:/code/python/PointSpace/reference_code/superpoint_transformer/src/nn/stage.py:277)

`TransformerBlock` 明确以图边作为自注意力邻接：

- `edge_index`: `[2, E]`
- `edge_attr`: `[E, F]`（相对位置编码等）

见 [src/nn/transformer.py:195](e:/code/python/PointSpace/reference_code/superpoint_transformer/src/nn/transformer.py:195)

---

## 6. 语义头与输出格式

`SemanticSegmentationModule.forward`：

```python
x = self.net(nag)
logits = self.head(x)  # 或多阶段 head 列表
output = SemanticSegmentationOutput(logits)
```

见 [src/models/semantic.py:291](e:/code/python/PointSpace/reference_code/superpoint_transformer/src/models/semantic.py:291)

输出对象 `SemanticSegmentationOutput`：

- 单阶段：`logits` 形状 `[N1, num_classes]`
- 多阶段：`logits` 是 list，元素分别对应不同层
- `semantic_pred()`：`argmax(logits)`，默认是 level-1 超点预测

见 [src/utils/output_semantic.py:17](e:/code/python/PointSpace/reference_code/superpoint_transformer/src/utils/output_semantic.py:17), [src/utils/output_semantic.py:76](e:/code/python/PointSpace/reference_code/superpoint_transformer/src/utils/output_semantic.py:76)

---

## 7. Stage2 损失函数（代码分支级）

入口：`model_step(batch: NAG)`  
见 [src/models/semantic.py:378](e:/code/python/PointSpace/reference_code/superpoint_transformer/src/models/semantic.py:378)

## 7.1 标签格式：`y_hist`

- 监督不是单标签，而是每个超点的标签直方图（包含 void 列）  
- `get_target()` 负责从 `nag` 提取 `y_hist`（单阶段 `nag[1].y`，多阶段 list）

见 [src/models/semantic.py:618](e:/code/python/PointSpace/reference_code/superpoint_transformer/src/models/semantic.py:618)

## 7.2 支持的损失类型

单阶段（`multi_stage_loss=False`）：

1. `ce`: `criterion(logits, y_hist.argmax(dim=1))`
2. `wce`: 先把直方图折叠到 dominant 类再算 `loss_with_target_histogram`
3. `kl`: `loss_with_target_histogram(criterion, logits, y_hist)`

见 [src/models/semantic.py:460](e:/code/python/PointSpace/reference_code/superpoint_transformer/src/models/semantic.py:460)

多阶段（`multi_stage_loss=True`）：

1. `ce`
2. `wce`
3. `ce_kl`（常用：第1级 CE，其余级 KL-hist）
4. `wce_kl`
5. `kl`

见 [src/models/semantic.py:397](e:/code/python/PointSpace/reference_code/superpoint_transformer/src/models/semantic.py:397)

配置侧：

- `default_ezsp` 开启 `multi_stage_loss_lambdas: [1,25,100]`  
  [configs/experiment/semantic/default_ezsp.yaml:11](e:/code/python/PointSpace/reference_code/superpoint_transformer/configs/experiment/semantic/default_ezsp.yaml:11)
- 默认基类语义配置中的 `loss_type` 在 `default.yaml` 定义。  
  [configs/model/semantic/default.yaml:7](e:/code/python/PointSpace/reference_code/superpoint_transformer/configs/model/semantic/default.yaml:7)

---

## 8. 预测从超点回传到体素/原始点

这是你关心的“链式传播”部分，代码上是**索引映射分发**：

1. `voxel_semantic_pred(super_index)`：level-1 超点预测 -> level-0 体素  
   [src/utils/output_semantic.py:114](e:/code/python/PointSpace/reference_code/superpoint_transformer/src/utils/output_semantic.py:114)
2. `full_res_semantic_pred(...)`：level-1 -> level-0 -> raw points  
   [src/utils/output_semantic.py:139](e:/code/python/PointSpace/reference_code/superpoint_transformer/src/utils/output_semantic.py:139)

---

## 9. 关键问题回答：输入到底是什么？

分三层回答：

1. **数据源输入**：原始点云（raw points）。  
2. **网络前处理后的原子层输入**：体素点（P0，`GridSampling3D` 后）。  
3. **语义主干主要操作对象**：超点层级图（P1+ 的 `NAG`，含 `edge_index/edge_attr`）。  

因此最准确说法是：

**Stage2 是“以体素为原子层、以超点图为主体计算单元”的层级图 Transformer 语义分割。**

---

## 10. 附：最小关键代码片段索引

### 10.1 Stage2 开关

```yaml
# configs/experiment/semantic/default_ezsp.yaml
model:
  training_partition_stage: False
```

### 10.2 语义前向

```python
# src/models/semantic.py
def forward(self, nag: NAG):
    x = self.net(nag)
    logits = self.head(x)
    return SemanticSegmentationOutput(logits)
```

### 10.3 SPT 是 UNet-like + NAG

```python
# src/models/components/spt.py
class SPT(nn.Module):
    """Superpoint Transformer. A UNet-like architecture processing NAG."""
```

### 10.4 图注意力的边输入

```python
# src/nn/transformer.py
def forward(self, x, norm_index, edge_index=None, edge_attr=None):
    x = self.sa(x, edge_index, edge_attr=edge_attr)
```

### 10.5 预测分发到体素/全分辨率

```python
# src/utils/output_semantic.py
def voxel_semantic_pred(...):
    return self.semantic_pred()[super_index]

def full_res_semantic_pred(...):
    return self.semantic_pred()[super_index_level0_to_level1][super_index_raw_to_level0]
```
