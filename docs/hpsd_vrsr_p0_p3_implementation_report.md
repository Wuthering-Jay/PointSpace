# HPSD-VRSR P0-P3 实施与验证记录

## 1. 实施范围

本轮完成阶段性方案的首个工程里程碑：P0 数据覆盖审计、P1 HPSD 无损上下文接口、
P2 DINO 锚定的 128 维传播空间，以及 P3 样本内 Local VRSR。现有
`HPSD-v1m1` 注册名、默认前向返回、concat-HPSD loss、1024 维 projector 和
HPSDFeatureTester 导出路径均保留。P4 可靠度/机载几何约束、P5 prototype bank 和
P6 memory queue 尚未实现。

## 2. 新增与修改内容

新增 `pointspace/models/backbone/vrsr/`，其中 `ops.py` 提供 token visibility、紧凑
patch-to-token teacher 聚合、teacher purity、有界 cosine Top-K 和高度分层截取；
`vrsr_v1m1.py` 实现固定正交 DINO 投影、轻量 128D propagation head、校准损失以及
同 tile soft reference loss。新模型注册为 `HPSD-VRSR-v1m1`，继承原 HPSD，使已有
checkpoint 的 `backbone.*` 和 `student_projector.*` 键不增加前缀，新增状态只位于
`vrsr.*`。

`hpsd_v1m1.py` 只增加 `HPSDTrainContext` 和可选
`forward_train(..., return_context=True)`。默认 `forward()` 仍调用完全相同的训练计算，
context 不进入 result dict，也不会被日志系统持有。

新增两份不依赖 base 的完整配置：

- `configs/hpsd/pretrain-hpsd-vrsr-litept-v1m4-hubei.py`
- `configs/hpsd/pretrain-hpsd-vrsr-ptv3-v3m4-hubei.py`

配置默认使用 `mode="calibrate"` 执行 P2，只有校准通过并加载 P2 checkpoint 后才应通过
命令行 override 或另存完整实验配置切换至 `mode="local"`。全量审计后默认
`source_purity=0.90`。

新增 `utils/audit_hpsd_visibility.py`。它按真实训练 transform 和 encoder pooling 流式
统计数据，只保存直方图、分位数和固定容量优先级 reservoir，不保存全部逐点或逐 token
数组。审计还按每个 tile 的 5%-95% robust z 范围统计十个归一化高度层。

## 3. 梯度与边界行为

Local VRSR 中 `target_train` 保留计算图，而 neighbor 检索、source 和 soft reference
全部在 `no_grad` 下构造。单元测试证明 local loss 对不可视 target 的输入特征产生非零
梯度，同时 source 不接收 local loss 梯度。检索先按 `level.point.batch` 分组，测试构造了
“跨样本 source 更相似”的反例，确认实现不会跨 tile 取邻居。

无 source、无 target 或完全无视觉监督时，VRSR 返回与 propagation head/backbone 相连
的零损失，因此 `find_unused_parameters=False` 下仍有零梯度图，不会因为空监督 batch
破坏 DDP。Top-K 会自动收缩到实际 source 数，query 按 chunk 计算，不创建完整
`Q x S` 常驻相似度矩阵。

## 4. 自动测试

pointcept 环境执行：

```text
D:/app/Anaconda3/envs/pointcept/python.exe -m pytest \
  tests/engines/test_batch_config.py \
  tests/models/test_hpsd.py \
  tests/models/test_vrsr.py -q
```

结果为 `20 passed`。其中原有 HPSD 测试仍为 `8 passed`；新增测试覆盖上下文数值一致、
visibility、teacher 聚合/purity、chunked Top-K、target/source 梯度、跨 batch 隔离、
空监督图、calibrate 模式和 HPSD-VRSR 特征导出委托。

两份完整配置均能独立解析和构建：LitePT 的 concat 输入为 900 维，VRSR 可训练参数
265,352；PTv3 的 concat 输入为 1008 维，VRSR 可训练参数 293,216。固定 teacher
projection 是 buffer，不计入可训练参数。

## 5. 湖北真实数据审计

数据目录 `E:\data\湖北\joint_tiles` 中 400 组 LAS/LAZ、DINO feature 和 correspondence
全部配对成功。使用 LitePT-v1m4、真实训练 transform、`source_q=0.6`、
`source_purity=0.90`、最小支持点数 4 完成全部 400 tile 的流式审计。结果保存于：

```text
exp/hpsd_visibility_audit_full_p09/visibility_audit.json
exp/hpsd_visibility_audit_full_p09/visibility_audit.md
```

主要统计如下：

| 指标 | 结果 |
| --- | ---: |
| 变换后点数 | 49,396,710 |
| 点级 valid 比例 | 39.65% |
| level-2 token 数 | 7,088,569 |
| fully-invisible token 比例 | 42.65% |
| mixed token 比例 | 40.93% |
| fully-visible token 比例 | 16.42% |
| 通过阈值的 source token | 996,961 |
| 同时具有 source 和 target 的 tile | 400/400 |
| teacher purity p05 / p25 / p50 | 0.866 / 0.923 / 0.966 |

中间高度层的不可视比例最高：归一化 z=0.5-0.6 层约 50.14%，而最低层约 31.74%。这说明
缺失不是均匀随机采样，并支持后续 P4 使用软高度兼容度，而不支持简单全局复制最近
DINO patch。

## 6. 真实 GPU 前向、反向与效率

设备为 NVIDIA GeForce RTX 5070 Ti Laptop GPU，AMP 使用 bfloat16。LitePT 对同一真实
batch、同一公共 HPSD 权重和匹配随机种子分别运行原 HPSD 与 HPSD-VRSR。预热后六次
测量结果为：

| 模型 | 中位 forward+backward | 峰值 allocated memory |
| --- | ---: | ---: |
| HPSD | 0.0987 s | 1322.8 MiB |
| HPSD-VRSR | 0.1104 s | 1573.8 MiB |
| 增量 | +11.9% | +251.0 MiB |

两条路径记录的 HPSD loss 在相同 seed 下保持一致。VRSR 增量处于方案设置的 P3 预算
（时间小于 15%、绝对显存小于 300 MiB）内。

PT-v3m4 使用真实 60,000 点 crop 完成 BF16 forward/backward，峰值 allocated memory
约 1589.7 MiB，`tok=9275`、`src=832`、`tgt=4860`，未出现 OOM、NaN、跨样本映射或
FlashAttention 错误。LitePT 的 calibrate/local 两种模式也分别完成一次真实 optimizer
step；calibrate 模式 `loc=0, acc=0`，local 模式实际接受上限内 1024 个 target。

## 7. 特征导出兼容性

使用 HPSDFeatureTester 和 HPSD-VRSR 模型对真实最小 tile `3387-503_0015.las` 的 7 个
GridSample fragment 执行合并。输出 Safetensors 形状为 `[5731, 1024]`、dtype 为 fp16、
`feature_source=projected`、`format_version=4`，全部原始点均被覆盖。VRSR 在
`return_point_feature=True` 时直接委托原 HPSD 导出路径，不运行 correspondence、校准
或 local propagation。

## 8. 当前限制与下一步

本工作区没有配置所指向的已训练 concat-HPSD checkpoint 文件，因此本轮只能用同结构
原 HPSD 的 state dict 验证兼容加载：结果为 0 个 unexpected key，只缺少预期的 7 个
`vrsr.*` state（固定投影及 propagation head）。正式 P2 训练前需要确认实际 HPSD
checkpoint 路径，不能在随机 HPSD 上直接开始校准。

目前完成的是功能、梯度、资源和数据前提验证，不代表 P2/P3 已经训练收敛。下一阶段应
先运行 `mode="calibrate"`，观察 validation source 的 `pcos`、HPSD loss 和梯度范数；
通过后保存 checkpoint，再切换 `mode="local"`。在 P3 下游不可视子集获得稳定增益前，
不应实现 P4/P5。

