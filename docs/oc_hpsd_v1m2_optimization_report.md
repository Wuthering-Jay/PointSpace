# OC-HPSD-v1m2 机制优化与验证报告

## 1. 优化目标

OC-HPSD-v1m1 已在 DALES Decoder probe 上相对 HPSD 将 mIoU 从 0.6657 提升到
0.6705，说明观测条件 masking 与 CSC 的完整组合具有继续研究的价值。不过，这个
结果只能证明完整配方有效，尚不能区分结构化 masking、CSC target 选择和连续
observability 各自的贡献。v1m2 因此不引入 relation loss、队列或第二次 encoder
前向，而是先消除现有 CSC 中最明显的离散门控，并把 mask 的实际执行强度变成可
诊断量。`OC-HPSD-v1m1` 代码入口和原配置均保留，作为可复现基线。

## 2. 真实数据审计

在湖北 `joint_tiles` 的 64 个随机增强训练样本上，GridSample 后平均包含 126,221
个点，其中影像有效点平均 46,900，高可信 mask 候选平均 46,197。候选点占有效点
98.5%，表明当前 `min_observability=0.60` 主要承担可信 teacher 的安全门控，而不是
大幅缩小可见集合。

原配置的目标 `mask_rate=0.30` 还受到每样本 `max_mask_points=8192` 限制。64 个
样本的总目标预算仅为候选点的 17.3%，实际遮蔽为候选点的 17.0%。在 8192 上限
下，完整 block 采样总体执行了 98.0% 的预算，中位数达到 99.87%，因此原训练日志
中的 `mr=0.30` 并不是实际遮蔽率。v1m2 将上限温和提高到 12,288，并直接报告
实际率。

更高上限会放大一个边界情况：若随机序列中的下一个完整 block 大于剩余预算，旧
算法会停止。例如候选点为 86,229 的真实样本中，12,288 预算只执行了 9,067，利用
率为 73.8%。v1m2 保留此前选中的完整 block，只在最后一个边界 block 内随机选择
恰好足够的候选点。修正后，另一个候选点为 104,961 的真实样本完整执行了
12,288/12,288。部分选择只发生在最后一个 block，因而不会把主体策略退化成全局
随机点 mask。

## 3. 连续 CSC token 可信度

v1m1 要求 token 内 `masked_count / valid_count >= 0.5`。这会让 0.49 与 0.50
覆盖率的 token 产生完全不同的监督状态，也会丢弃具有真实 DINO teacher、但只被
部分结构化遮蔽的 token。v1m2 仍用 `completion_min_mask_fraction=0.1` 排除极弱
支持，但对其余 token 使用连续权重：

```text
w_t = min(mask_fraction_t / 0.5, 1)
      * mean_observability_t
      * sqrt(masked_count_t)
```

第一项描述该 token 的遮蔽完整度，达到 0.5 后不再继续放大；第二项描述 masked
teacher 的观测可信度；第三项只以平方根速度提高多点支持 target 的稳定性。损失先
在每个样本内按 `w_t` 归一化，再对 batch 中有 CSC target 的样本等权平均，避免
大点云或高密度 tile 主导梯度。DINO 仍保持原生 1024 维，teacher 聚合仍保留完整
token-patch 多对多关系。

真实 LitePT 样本中，在相同 8,102 个 masked point 下，CSC target 从 v1m1 的
2,896 增加到 v1m2 的 3,382，增加约 16.8%；峰值分配显存从 1225.5 MiB 到
1246.3 MiB，增加约 1.7%。这一次单样本测量包含 CUDA 冷启动差异，因此耗时数字
不用于声称加速，但证明增加的 target 没有形成显著显存压力。

## 4. 日志语义

v1m2 的 `mr` 改为 `masked/candidate`，表示真实执行的候选遮蔽率；`mu` 为
`masked/requested_budget`，用于定位 block 采样是否浪费预算；`cr` 为最终 CSC
target token 占所有带 masked route token 的比例。配置目标仍由 `mask_rate` 控制，
但不再把目标值伪装成实际统计量。旧 v1m1 日志字段不变，历史训练记录仍可按原语义
解释。

## 5. 代码与配置

核心新增实现位于 `pointspace/models/backbone/oc_hpsd/oc_hpsd_v1m2.py`，连续
权重和统计所需的通用逻辑位于 `oc_hpsd_v1m1.py` 与 `ops.py`，但默认开关保持
v1m1 原数值路径。完整训练配置为：

```text
configs/hpsd/pretrain-oc-hpsd-v1m2-litept-v1m4-hubei.py
configs/hpsd/pretrain-oc-hpsd-v1m2-ptv3-v3m4-hubei.py
```

两份配置均独立完整，不使用 base。它们采用 `completion_min_mask_fraction=0.1`、
`completion_full_weight_fraction=0.5`、`max_mask_points=12288` 和
`fill_partial_block=True`，并使用独立实验目录，不覆盖 v1m1 checkpoint。

## 6. 验证与下一步

LitePT 和 PTv3 完整配置均能构建，参数量分别为 16,239,168 与 52,929,288；v1m2
没有新增可学习参数，因此与 v1m1 参数量相同。HPSD 与 OC-HPSD 相关测试共 19 项
通过，其中包括 v1m1 零 mask 数值等价、v1m2 连续加权反传、样本平衡、边界 block
预算补齐和特征导出。

下一轮正式实验应保持数据划分、seed、epoch 和 probe 配置不变，比较 v1m1 与
v1m2。若 v1m2 提升，应再做两项最小消融：首先只启用连续 CSC 权重但保持 8192
上限，其次保持 v1m1 硬门控但提高到 12,288。这样才能判断收益来自更平滑的监督
利用，还是单纯来自更多 masked point。考虑到当前 Decoder probe 增益为 0.48 个
mIoU 点，至少还应补一个 seed；在此之前不建议继续叠加 relation loss 或更复杂的
跨样本机制。

## 7. 正式训练反馈与结论

后续正式训练显示 v1m2 带来轻微下游精度下降，因此它不替代 v1m1 主线。末轮日志中
实际候选遮蔽率达到 0.2186，masked point 和 CSC token 分别约为 51,285 和 17,981；
相对 v1m1 后期约 37,679 个 masked point 和 11,580 个 CSC token，分别增加约 36%
和 55%。相对 v1m1，HPSD edge 从约 94,291 降至 87,352，监督 patch 从约 48,050
降至 44,789，分别减少约 7.4% 和 6.8%；相对不做 masking 的原始 HPSD，它们则
减少约 22.8% 和 21.3%。`cr=0.9967` 还表明低门槛连续方案几乎接收了所有带
masked route 的 token。

这些统计更支持“增强 CSC 的同时削弱了可靠 HPSD 锚定”这一解释，而不能单独证明连续
权重本身无效。后续机制应首先解除 HPSD anchor 数量与输入 masking 强度之间的竞争，
并通过单变量消融判断连续权重，而不是继续提高 mask point 上限。
