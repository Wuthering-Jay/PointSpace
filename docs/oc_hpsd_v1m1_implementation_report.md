# OC-HPSD-v1m1 第一版本实现与验证报告

## 1. 实现结果

第一版本已经把“观测条件 HPSD + 真实输入 masking + CSC”接入 PointSpace，并保持原 `HPSD-v1m1` 的代码路径、注册名和测试导出行为不变。新模型注册名为 `OC-HPSD-v1m1`，支持 LitePT-v1m4 与 PT-v3m4，在一个连续训练 run 中完成 HPSD warm-up、mask/CSC 线性开启和联合训练。

在 OC-HPSD 完成下游有效性验证后，VRSR 的模型注册、实现、配置和专用测试已物理删除。通用的可视覆盖审计算子迁移到了 HPSD 分析模块，旧 VRSR checkpoint 不再提供运行时兼容入口。

## 2. 第一版本数据流

新 correspondence Safetensors 在原有 `pixel_coord`、`patch_index` 和 `valid` 之外增加：

```text
observability: [N], float16, range [0, 1]
```

`utils/tile_las_image.py` 复用已经构建的局部最高表面 DSM。令 `delta_z` 为点到局部最高表面的非负高差、`r` 为回波序号，连续可信度为：

```text
q = coverage * exp(-0.5 * (delta_z / surface_z_tolerance)^2)
    * exp(-echo_decay * (r - 1))
```

无影像覆盖点在写 correspondence 时固定为 0。`image_valid` 是影像覆盖与硬表面可见性的共同门控；回波只作为连续先验，不直接改写硬有效性。DINO 特征提取工具更新 `patch_index` 时会保留 `observability` 和其他未知扩展 tensor。

`LasImageDataset` 将该字段读为 `image_observability`，并把它加入点级同步索引。correspondence 缺少新 tensor 时，自动使用：

```text
image_observability = image_valid.float()
```

湖北数据已经用新版更新器原地迁移 correspondence，DINO 图像特征本身不需要重新提取；迁移前文件保存在独立备份目录中。

## 3. 结构化输入 masking

`GeometryGuidedMaskGenerator` 在 GPU 上按 batch sample 独立运行。候选点必须满足 `image_valid=True` 且 q 不低于配置阈值。算法按 XY block 分组，利用 block 内全部点的 Z 跨度筛选具有垂向结构的 block，再随机选择完整 block 中的高可信可视点作为 simulated-missing。若某样本没有达到垂向跨度要求的 block，可以退化为普通随机 block mask。

生成器同时约束最小 anchor 点数、最小 anchor 比例和每样本最大 mask 点数。Mask rate 为 0 时严格返回全 False。Mask 在模型 forward 内、backbone 调用之前生成，因此看到的是已经完成数据增强、GridSample 和 SphereCrop 的当前点集，也能由训练进度动态调整。

两种 m4 backbone 继承的 embedding 原本已经支持 learned mask token。新配置设置 `mask_token=True` 后，被选择点的 embedded feature 会由 mask token 替换，点坐标、点数量和层级映射保持不变。这样 CSC 不只是切换 loss 路径，而是真正面对缺失的 intensity、echo 等输入属性。

## 4. Routed token-patch edges

新实现只对全部硬有效点执行一次 `(token, patch)` 去重，并在每条唯一 edge 上同时记录：

```text
anchor_count
masked_count
anchor_q_sum
masked_q_sum
```

对于 `sqrt_count` 模式，某条监督路由的 edge 权重为：

```text
w = q_sum / sqrt(count) = sqrt(count) * mean(q)
```

Anchor route 把 concat 三维 token feature 聚合到 patch，执行原生 1024 维 HPSD cosine loss。Masked route 把多个真实 DINO patch teacher 聚合到 simulated-missing token，作为 CSC teacher。该设计保留 token 对多个 patch、patch 对多个 token 的完整多对多关系，不退化为最近 patch。

## 5. CSC 路径

第一版本仍以 level 2 作为蒸馏目标层。HPSD 使用 F2+F3+F4 concat；CSC 明确切掉 concat 的 F2 通道，只使用上采样后的 F3/F4：

```text
context = Concat(upcast(F3), upcast(F4))
prediction = LayerNorm -> Linear -> GELU -> Linear(1024)
```

只有真实存在 masked-visible support、支持点数满足阈值且 token 内 masked fraction 足够高的 token 才创建 `[M,1024]` prediction。空 anchor、空 masked target 和整个样本没有 DINO teacher 时，两套 projector 都保留在计算图中并安全产生零梯度。

总损失为：

```text
L = L_observation_hpsd + lambda_csc(progress) * L_csc
```

第一版本没有 relation loss、KNN、prototype、queue、registration confidence 和 semantic confidence。

## 6. 单次训练 curriculum

`ObservationCurriculumHook` 每个 step 根据 `epoch`、当前 iteration、每 epoch step 数和总 epoch 数重建归一化进度。默认配置为：

```text
0% - 10%   mask_rate=0, lambda_csc=0
10% - 20%  两者线性上升
20% - 100% mask_rate=0.30, lambda_csc=0.20
```

Hook 不保存独立阶段 checkpoint，也不重置 optimizer 或 scheduler。断点恢复后会由恢复的 epoch/iteration 自动回到对应 curriculum 位置。

训练日志采用紧凑键：

```text
hpsd  csc  mr  tok  edge  patch  anc  msk  ctok
```

其中 `mr` 为当前 mask rate，`anc/msk` 为 anchor/masked 点数，`ctok` 为实际 CSC token 数。

## 7. 代码结构

```text
pointspace/models/backbone/oc_hpsd/
├── __init__.py
├── ops.py                 # routed edges、q 聚合、结构化 mask
└── oc_hpsd_v1m1.py       # OC-HPSD、CSC、loss 与导出分流

pointspace/engines/hooks/
└── observation.py         # 单 run curriculum

configs/hpsd/
├── pretrain-oc-hpsd-litept-v1m4-hubei.py
└── pretrain-oc-hpsd-ptv3-v3m4-hubei.py

tests/models/
└── test_oc_hpsd.py
```

数据侧同步修改了：

```text
utils/tile_las_image.py
utils/dino/extract_dino_feature.py
pointspace/datasets/las_image.py
```

## 8. 自动测试结果

在 pointcept 环境运行：

```powershell
& D:/app/Anaconda3/envs/pointcept/python.exe -m pytest tests -q
```

第一轮完整结果为 28 passed。测试覆盖 routed edge 两路计数和 q sum、Observation-HPSD 聚合、结构化 mask anchor 预算、mask rate 0、OC-HPSD/HPSD 数值等价、CSC backward、mask token gradient、空监督、特征导出、旧 correspondence 回退、新 observability 读取、DINO 更新保留扩展 tensor、DSM 连续 q 和 curriculum 进度恢复。原 HPSD 和 VRSR 测试继续通过。

## 9. 湖北真实数据兼容验证

`E:\data\湖北\joint_tiles` 当前检测到 400 个点云、DINO 和 correspondence 配对样本。全量 72,748,646 点已迁移到 `pointspace_image_mapping_v3`，`image_valid` 比例为 43.36%，所有 q 均有限且位于 0-1。配置化实测单样本读取约 0.09-0.10 秒。

### 9.1 LitePT-v1m4

在一个约 101,038 点、7,880 patch 的真实增强样本上，BF16 forward/backward 成功：

| 项目 | 结果 |
| --- | ---: |
| level-2 token | 16,814 |
| anchor point | 18,784 |
| masked point | 8,007 |
| CSC token | 2,958 |
| 单次未热身测量 | 约 8.01 s |
| peak allocated | 约 1235.9 MiB |

### 9.2 PT-v3m4

在 SphereCrop 后 60,000 点、4,608 patch 的真实样本上，BF16 forward/backward 成功：

| 项目 | 结果 |
| --- | ---: |
| level-2 token | 9,527 |
| anchor point | 10,196 |
| masked point | 4,345 |
| CSC token | 1,437 |
| 单次未热身测量 | 约 7.04 s |
| peak allocated | 约 1497.8 MiB |

这些未热身时间包含首次算子和缓存成本，只用于证明真实路径能够完成，不能用于模型速度结论。

## 10. 同 batch 热身后资源对比

在同一个真实 LitePT batch、同一进程、各自一次 warm-up 后执行 3 次 forward/backward 平均：

| 模型 | 平均时间 | Peak allocated |
| --- | ---: | ---: |
| HPSD | 0.4646 s | 1291.0 MiB |
| OC-HPSD | 0.4921 s | 1317.6 MiB |
| 增量 | +5.93% | +2.06% |

OC-HPSD 当次产生 8,186 个 masked point 和 3,053 个 CSC token。结果低于 TODO 中时间 +25%、显存 +20% 的 Go/No-Go 上限。该结果是功能阶段的单 batch micro-benchmark，不替代完整 epoch 吞吐、不同 tile 分布和多 GPU 测量。

## 11. 测试导出兼容性

使用 PTv3 测试数据的一个 98,529 点 fragment，在输入没有任何 `dino_*` 字段的情况下成功得到：

```text
point_feature: [98529, 1024], float32
```

这证明 `return_point_feature=True` 不进入 masking、routed edges 或 CSC，可继续由 `HPSDFeatureTester` 合并 GridSample fragments 并写 Safetensors。

## 12. 运行命令

LitePT：

```powershell
& D:/app/Anaconda3/envs/pointcept/python.exe tools/train.py `
  --config-file configs/hpsd/pretrain-oc-hpsd-litept-v1m4-hubei.py `
  --num-gpus 1
```

PTv3：

```powershell
& D:/app/Anaconda3/envs/pointcept/python.exe tools/train.py `
  --config-file configs/hpsd/pretrain-oc-hpsd-ptv3-v3m4-hubei.py `
  --num-gpus 1
```

训练前应确认配置中的 `data_root`、`pointcloud_path`、`feature_output_dir` 和 `weight` 符合当前实验路径。首次正式连续 q 实验还需要使用新版 tile 工具生成包含 `observability` 的 correspondence。

## 13. 当前边界与下一步

第一版本完成的是数据、模型、curriculum、两 backbone 和导出链路的工程闭环，不代表已经证明下游语义精度提升。下一步应先固化当前 HPSD baseline，然后训练 Observation-HPSD、random block mask+CSC 和 geometry-guided mask+CSC。只有 masked CSC 在 frozen probe，特别是低 q 与真实不可视子集上产生稳定收益，才进入 relation loss 或更复杂 q 建模。

真实不可视点仍然没有直接 DINO teacher。第一版本不会给它们伪造 patch，也不会声称每个不可视 token 都获得了视觉语义。后续研究验证必须加入真实不可视 activation gradient coverage、q/高度/回波分层 probe 和 teacher coverage 退化曲线。
