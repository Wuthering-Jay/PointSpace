# HPSD / OC-HPSD 显存优化与数值验证报告

## 1. 优化原则

本轮只接受不改变 HPSD/OC-HPSD 监督关系、DINO 原生 1024 维、masking 机制、loss
权重和模型参数结构的优化。每个候选修改均先比较完整模型的 loss、全部参数梯度、峰值
显存和耗时；若没有真实显存收益，或者数值差异超过 CUDA 基线重复运行自身的波动，
则不进入配置。

## 2. 真实显存组成

在一个约 8.5-9.8 万点的湖北真实样本上，模型参数约 61.9 MiB，GPU 输入约
23.6 MiB，但 OC-HPSD forward/backward 峰值约 1.0-1.2 GiB。单独运行 LitePT encoder
的峰值约 893.8 MiB，加入 HPSD 后约 1185.0 MiB，再加入 OC-HPSD CSC 后约
1200.8 MiB。因此该样本中 backbone activation 约占峰值的四分之三，HPSD projector、
1024 维 cosine 和层级聚合构成主要增量，CSC 的额外峰值相对较小。

saved-tensor 审计曾显示一个逻辑形状为 `[E,900]` 的 int64 scatter index，但进一步按
唯一 storage 检查后确认它是由一维 index 扩展得到的零步长视图，并不实际占用
`E*900*8` 字节。基于逻辑 shape 估算出的数百 MiB 是错误的，不能作为优化依据。

## 3. 保留的优化

### 3.1 Projector + cosine checkpoint

`HierarchicalPatchSetDistiller` 新增 `projector_checkpoint`。训练时把
`LayerNorm -> Linear -> GELU -> Linear -> FP32 normalize -> cosine` 作为一个
checkpoint 区域，forward 仍执行完全相同的算子，backward 时重算中间激活。HPSD 的
student projector 与 OC-HPSD 的 completion projector 共用该路径。eval、特征导出和
无梯度路径不执行重算，state dict 不增加或删除任何参数键。

### 3.2 LitePT 深层 MLP checkpoint

LitePT Block 新增 `checkpoint_mlp`，默认关闭，仅在 HPSD LitePT-v1m4 配置中开启。
它只重算 attention block 内纯 tensor 的
`LayerNorm -> MLP -> DropPath` 分支，不重算 GridPooling、FlashAttention、序列化或
spconv。checkpoint 保存 DropPath RNG 状态，因此重算使用与原 forward 一致的随机
mask。当前 LitePT-v1m4 有 14 个 encoder block，其中 8 个 attention block 实际进入
该分支；纯卷积 block 保持原路径。

## 4. 数值一致性

CPU FP32 单元测试中，checkpoint 开关前后的 projector loss、Block 输出、输入梯度和
参数梯度在 `1e-7` 容差内一致。真实 CUDA BF16 完整模型中，checkpoint 相对基线的
平均参数梯度绝对差约为 `2.13e-6`；不开 checkpoint、相同 seed 重复运行两次的基线
自身波动约为 `2.17e-6`。checkpoint 的最大局部差约 0.0321，也低于当次基线重复运行
的 0.0391。这说明观察到的差异主要来自 CUDA scatter/FlashAttention 的固有非确定性，
没有证据表明 checkpoint 引入额外数值偏移。

HPSD 与 OC-HPSD 相关自动测试共 22 项通过。模型统计量 `tok/edge/patch/msk/ctok`
在开关前后完全一致，模型 state dict 结构不变，现有 checkpoint 可直接加载。

## 5. 显存与速度结果

单样本五次测量如下：

| projector checkpoint | LitePT MLP checkpoint | 峰值显存 | 平均时间 |
|---|---|---:|---:|
| 否 | 否 | 1089.9 MiB | 0.1119 s |
| 是 | 否 | 1047.3 MiB | 0.1095 s |
| 否 | 是 | 988.7 MiB | 0.1132 s |
| 是 | 是 | 932.6 MiB | 0.1184 s |

两项同时启用节省约 157.3 MiB，即 14.4%，平均耗时增加约 5.9%。在单 GPU 配置实际
使用的 micro-batch=5 上，batch 统计为约 92,628 token、93,431 edge、51,963 patch、
39,823 masked point 和 13,484 CSC token；峰值从 6745.4 MiB 降到 6008.4 MiB，
节省 736.9 MiB，即 10.9%，平均 step 时间增加约 5.4%。

这些数字来自同一 GPU、同一真实 batch 和相同随机种子的局部 benchmark，不应被解释
为不同硬件上的固定比例，但足以证明优化在正式训练规模上具有实质收益。

## 6. 被否决的候选

自定义高维 index-add 反向没有降低真实峰值，且最大局部梯度差约 0.0227，因此已完全
回滚，继续使用原生 `torch_scatter`。跳过第二次 DINO teacher normalize 的候选在
micro-batch=5 上峰值完全相同，仅产生极小数值变化，也已回滚。当前代码不包含这两项
实验路径。

## 7. 当前配置

所有 HPSD/OC-HPSD 配置已启用 `projector_checkpoint=True`。LitePT-v1m4 配置另外启用
`checkpoint_mlp=True`；PT-v3m4 暂时只启用已独立验证的 projector checkpoint，没有
把 LitePT 专用的 Block 重计算机制套入 PTv3。该优化不会修复 v1m2 的精度问题，正式
精度主线仍应使用已验证的 OC-HPSD-v1m1，显存优化与蒸馏机制版本相互独立。
