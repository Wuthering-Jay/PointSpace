# HPSD 第一阶段实现报告

## 1. 阶段结论

HPSD 第一阶段训练闭环已经完成。当前版本可以使用同一个 `HPSD-v1m1`
wrapper 对 PTV3 或 LitePT 进行 DINO-to-LiDAR 蒸馏，使用预计算的原生
1024 维 DINO 特征，以 DINO patch 为监督单位，不生成逐点 1024 维预测。

当前实现已经完成合成单元测试、两种 backbone 的 GPU forward/backward、
真实 `LasImageDataset` 数据链路、bfloat16 AMP、空监督 batch、checkpoint
迁移和 60,000 点真实 tile 显存测试。PCA/KMeans 分析与赋色已经完成；
relation-aware edge decoder 和 UTONIA 联合训练保留在后续 TODO 中。

## 2. 新增模块

### 2.1 公共 encoder hierarchy

文件：`pointspace/models/backbone/hpsd/hierarchy.py`

新增 `HierarchyLevel`：

```python
HierarchyLevel(
    point=point_at_level,
    input_to_level=input_point_to_token,
    level=level_id,
)
```

`build_encoder_hierarchy()` 从 encoder bottleneck 的 `pooling_parent` 链恢复
fine-to-coarse hierarchy，并逐层组合 `pooling_inverse`，生成每个输入点到各
encoder level token 的精确映射。该函数不 `pop` 或修改原 Point，因此不会
破坏后续特征使用和 autograd。

它会检查：

- hierarchy 是否存在循环；
- 实际层数是否符合 backbone 的 `num_stages`；
- 每层 pooling inverse 长度；
- 输入点映射是否越界；
- 映射是否跨 batch 样本。

### 2.2 PTV3 HPSD backbone

文件：
`pointspace/models/backbone/point_transformer_v3/point_transformer_v3m4.py`

注册名称：

```text
PT-v3m4
```

它继承 `PT-v3m3`，强制 `enc_mode=True`，只新增：

```python
backbone(data_dict, return_hierarchy=True)
```

未复制原始 PTV3 主体，也未修改已有 `PT-v3m1/PT-v3m3` 的行为或参数名称。

PTV3 m4 使用 3D RoPE 时，每个 attention head 的维度必须能被 3 整除。
正式配置采用 `(36,72,144,288,576)` channels 和 `(2,4,8,16,32)` heads，
每头维度为 18，满足约束。

### 2.3 LitePT HPSD backbone

文件：`pointspace/models/backbone/litept_v1/litept_v1m4.py`

注册名称：

```text
LitePT-v1m4
```

它继承 `LitePT-v1m3`，同样强制 encoder-only 并提供相同 hierarchy 接口。
LitePT PointRoPE CUDA kernel 要求 token head dimension 是 6 的倍数；正式配置
的每头维度为 18。

### 2.4 HPSD wrapper

文件：`pointspace/models/backbone/hpsd/hpsd_v1m1.py`

注册名称：

```text
HPSD-v1m1
```

实现内容包括：

- token-patch unique edge 构建；
- 每条边的支持点计数；
- `uniform/count/sqrt_count` 三种边权；
- patch-centric 三维特征聚合；
- 深层 encoder 特征 up-cast 与通道 concat；
- 一次 token-patch 建边和原生 1024 维 student projection；
- float32 normalized cosine loss；
- 每个样本等权的 patch loss；
- 空监督 batch 安全 backward；
- token、edge、used patch 数量日志。

## 3. Hierarchical Concat HPSD

如果只取 level 2 的原始特征计算 loss，level 3 和 level 4 位于 loss
之后，无法获得蒸馏梯度。当前实现以 `distill_level=2` 为监督尺度，
将更深层特征通过 `pooling_inverse` 精确对齐回 level 2 token：

```python
fused_level2 = concat(
    level2.feat,
    upcast(level3.feat),
    upcast(level4.feat),
)
```

空间监督单位仍是 level 2 token，但 projector 输入同时包含 level 2/3/4
语义。LitePT 的 concat 通道数为 `144+252+504=900`，PTv3 为
`144+288+576=1008`。聚合后的有效 patch 经过
`LayerNorm -> Linear(C3,1024) -> GELU -> Linear(1024,1024)`，与 DINO
teacher 计算一次 cosine loss；projector 不作用于逐点特征。

## 4. Token-Patch 关系算法

输入点 `i` 同时具有：

```text
input_to_level[i]  -> 目标层 3D token
dino_patch_index[i] -> DINO patch
dino_valid[i]       -> 是否参与监督
```

有效点生成 `(token,patch)` 关系，通过 int64 key 去重：

```python
edge_key = token * num_patches + patch
```

每个 unique key 产生一条边并记录 `point_count`。一个 token 覆盖的所有 patch
均被保留，不进行 correspondence 坐标平均，也不选择最近 patch。

当前默认边权为：

```python
weight = sqrt(point_count)
```

三维特征先在低维 backbone channel 空间聚合：

```python
patch_feat = weighted_mean(fused_token_feat[edge_token], edge_patch)
patch_pred = Linear(patch_feat)  # [U, 1024]
```

由于 student projection 发生在 patch 聚合之后，当前实现不会分配 `[N,1024]` 或
`[E,1024]` student 激活，只分配 `[U,1024]`。

## 5. 无损 DINO Patch Compaction

文件：`pointspace/datasets/transform.py`

新增 transform：

```text
CompactDinoPatches
```

它必须放在所有可能删除点的 transform 之后、`ToTensor/Collect` 之前。
操作流程：

1. 找出当前点集实际引用的 unique patch；
2. 对 `dino_feature` 只做行切片，不改变通道和数值；
3. 将点级 patch index 重映射到紧凑区间；
4. 更新 `dino_offset`；
5. 保存 `dino_source_patch_index` 以恢复原始 patch row/col。

该操作不是 DINO 特征压缩。输出仍是原生 1024 维，特征逐值不变。

真实双样本合批测试结果：

```text
points: 8192
patches per sample: 308, 450
dino_offset after collate: [308, 758]
max valid global patch index: 757
```

单元测试已确认 compaction 前后 teacher gather 和完整 patch cosine loss逐值一致。

## 6. Loss 计算

Student prediction 和 DINO teacher 在 loss 前转换为 float32：

```python
student = normalize(patch_pred.float())
teacher = normalize(dino_feature[used_patch].float())
loss_patch = 1 - sum(student * teacher)
```

默认 `sample_balanced=True`。每个样本先对自身有效 patch 求均值，再对存在
监督的 batch 样本求均值。这样有效覆盖面积较大的 tile 不会因 patch 更多而
自动获得更高权重。

如果整个 batch 没有有效 patch，模型返回与 backbone 和 student projector
计算图相连的零 loss，已经验证可以正常 backward。

## 7. 配置

LitePT 配置：

```text
configs/hpsd/pretrain-hpsd-litept-v1m4-hubei.py
```

PTV3 配置：

```text
configs/hpsd/pretrain-hpsd-ptv3-v3m4-hubei.py
```

两份配置均为不依赖 base 的完整配置，并采用：

- `LasImageDataset`；
- `coord + intensity + echo`，共 6 维输入；
- `grid_size=0.5`；
- `distill_level=2`；
- level 2/3/4 特征对齐 concat；
- 原生 DINO-1024；
- `sqrt_count`；
- concat 层级表示与单次 DINO-1024 loss；
- bfloat16 AMP；
- micro-batch 1；
- gradient accumulation 4；
- `point_max=60000`。

配置继续保留原图尺寸、像素坐标、source patch index、patch size 和 feature
grid 等字段，为后续 relation-aware edge decoder 提供数据基础。

## 8. 测试结果

### 8.1 自动单元测试

文件：`tests/models/test_hpsd.py`

结果：

```text
8 passed
```

覆盖：

- 多 token—多 patch unique edge；
- edge count；
- sqrt-count 聚合和梯度；
- 空关系；
- 样本平衡 loss；
- hierarchy concat、单次 loss 与全层梯度；
- DINO patch compaction；
- compaction 前后 loss 一致性。

### 8.2 正式配置构建

LitePT 和 PTv3 两份完整配置均使用 level 2/3/4 concat HPSD，projector
输入维数分别为 900 和 1008，并完整保留 DINO 1024 维输出。

### 8.3 真实数据 Concat GPU 测试

使用湖北真实 tile、LitePT-v1m4、bfloat16 AMP 进行完整
forward/backward：

```text
points: 60,000
valid points: 14,194
patches: 4,549
edges: 7,907
LitePT peak allocated: 699 MiB
PTv3 peak allocated: 1,446 MiB
level 0-4 gradients: all finite and non-zero
```

这些数据来自此前同一 concat 实现的真实数据 forward/backward 验证。
显存数据不包含完整 Trainer 中的 AdamW 状态和 DDP bucket，但已证明 concat
与原生 DINO-1024 可以完成训练，并且深层 encoder 能收到有效梯度。

## 9. Checkpoint 迁移测试

Tiny PTV3 distiller state 中提取全部 `backbone.*`，去掉前缀后严格加载到
同构 `PT-v3m4`：

```text
backbone tensors: 69
missing keys: 0
unexpected keys: 0
```

说明 student projector 与 backbone 命名边界清楚。下游微调可以显式只加载
`backbone.*`，不依赖宽松加载偶然匹配。

## 10. 当前未完成项目

以下内容尚未实现，状态以
`docs/dino_lidar_distillation_todo.md` 为准：

1. level 1 与 level 2 的同口径精度、显存和吞吐量对比；
2. DiTR-style point-1024 基线；
3. relation-aware edge decoder；
4. patches-per-token/tokens-per-patch 训练期统计；
5. UTONIA 联合训练；
6. 只有显存测试证明必要时才启动 PCA-512/PCA-256。

## 11. 下一步建议

下一阶段应先运行短程真实训练而不是继续增加模型结构：

1. LitePT 和 PTV3 各运行数百 iteration；
2. 检查 loss 曲线是否下降、梯度是否稳定；
3. 对比 level 1/2 的完整 AdamW/AMP 峰值显存和吞吐量；
4. 从已保存 checkpoint 做中断续训测试；
5. 将 backbone 迁移到现有语义分割配置做短程微调；
6. 之后再开始 correspondence 策略消融。

在这些闭环完成前，不建议加入 edge decoder 或 UTONIA 目标，否则训练异常
将难以定位到数据关系、层级映射、损失设计还是新增自监督分支。

## 12. 工程结构与统一训练测试入口

经过审核适配的 HPSD 代码现统一位于：

```text
pointspace/models/backbone/hpsd/
    hierarchy.py
    hpsd_v1m1.py
```

原 `pointspace/models/hpsd` 已移除。LitePT、PTV3 的适配版本分别注册为
`LitePT-v1m4`、`PT-v3m4`，不再使用 `-HPSD` 后缀。两个湖北配置文件均已
展开为完整配置，不依赖 `_base_`，同一个配置可以分别交给：

```powershell
python tools/train.py --config-file configs/hpsd/pretrain-hpsd-litept-v1m4-hubei.py
python tools/test.py --config-file configs/hpsd/pretrain-hpsd-litept-v1m4-hubei.py
```

`HPSDFeatureTester` 在测试时不读取 DINO teacher，而只读取 LAS 点特征。
`GridSample(mode="test")` 产生的多个 fragment 可以按 `batch_size_test` 批量
推理；每个 fragment 的输出利用 `index` 写回原始点位置，重复出现的点取
特征均值，最后再次归一化。输出是同名 Safetensors，主张量为
`feature: [N,C]`。默认 `feature_source="projected"`，因此 `C=1024`；也可
设置为 `backbone` 导出投影前的 512 维层级融合特征。

真实 tile 测试包含 135,441 个原始点，GridSample 生成 8 个 fragment；使用
fragment batch 4 后成功恢复并写出 `[135441,1024]`，没有遗漏点。随后通过
`utils/analyze_hpsd_features.py` 完成 PCA、MiniBatchKMeans、两种独立颜色
输出和 tile 合并测试。构造的两个 tile 各 2,000 点且重叠 1,000 个
`orig_idx`，最终 PCA 与 KMeans 两条输出均由 4,000 条 tile 记录恢复为
3,000 个按原始索引排序的唯一点。

旧 `utils/utonia` 特征提取和分析脚本已删除。新分析工具按同名 LAS/LAZ 与
Safetensors 配对，分批变换特征，并输出：

```text
output/
    analysis_model.safetensors
    pca/tiles/       pca/merged/
    kmeans/tiles/    kmeans/merged/
```

PCA 输出使用前三主成分赋 RGB；KMeans 输出同时写入 `hpsd_kmeans` extra
dimension 和确定性 cluster 颜色。合并必须依赖 `orig_idx`，从而避免用坐标
近似去重误删真实重合点。

## 13. Batch 与训练 Hook 配置整理

HPSD 配置只保留 `batch_size_train` 和 `batch_size_test`。前者表示全部 GPU 上
期望的有效训练 batch，Trainer 根据 world size 和
`gradient_accumulation_steps` 推导实际 micro-batch；后者同样先按 world
size 分配，在 `HPSDFeatureTester` 中表示每个 GPU 对单个 tile 每次前向处理
的 GridSample fragment 数量。HPSD 不进行标签验证且没有 `data.val`，因此不需要
`batch_size_val`。旧 `batch_size` 只在 `default_setup` 中作为兼容回退，不再
出现在新配置中。

`default_setup` 现允许省略验证和测试 batch：缺省时均按每 GPU 1 处理；旧
配置使用 `batch_size` 时会将解析值显式发布为 `batch_size_train`，保证
Trainer 后续逻辑一致。正数和 world-size 整除约束也改为带明确信息的异常。

两份配置使用统一 hook 顺序：`CheckpointLoader`、`RuntimeInfoHook`、
`ModelHook`、`IterationTimer`、`InformationWriter`、`CacheCleaner`、
`CheckpointSaver`。其中 CacheCleaner 同时被训练和特征 tester 使用，并在
epoch 末、训练结束、固定 step 或异常耗时条件下集中清理缓存，所以 HPSD
配置不再保留当前 Trainer 未读取的 `empty_cache` 与
`empty_cache_per_epoch`。RuntimeInfoHook 和 ModelHook 当前不是 HPSD loss
的硬依赖，但提供了与项目其他训练任务一致的运行状态与模型生命周期接口。
