# 4.5 统一迁移协议的 81 项短程实验分析

## 结论与适用范围

本轮 81/81 项均完成训练、一次验证与 checkpoint 保存，日志未出现异常退出。DALES/OpenGF 为 1 epoch × 1 loop，Vaihingen 为 1 epoch × 5 loops。此轮目的仍是迁移协议和早期优化诊断，不能作为已经收敛的最终下游测试结果。

新的协议修复了 D-DITR/Concerto 曾经出现的严重训练—验证失配，但没有消除所有方法之间的迁移差异。Point-M2AE 在 DALES/Vaihingen 的全量微调出现明显的早期负迁移；Concerto 在这两个数据集上仍低于 Scratch；D-DITR 在 DALES 上也仍低于 Scratch。这些情况必须保留并解释，不能直接归为“先进方法在 ALS 上必然无效”。

CVRL 的 frozen linear probe 在三个数据集均为最高，显示其输出表示较易被线性分类器利用。FT 在 DALES 上最高，在 Vaihingen 上与 VPSD 几乎相同，在 OpenGF 上略低于 VPSD。现有结果支持 CVRL 的紧凑表示具有较好的早期迁移能力，但不能支持其所有指标、所有区域均显著领先。

完整数据见 [data_tables.md](data_tables.md)，逐日志指标和类别指标见 [metrics.json](metrics.json)。完成分析后，原始短程诊断日志和 checkpoint 已按要求清理；JSON 中的路径保留为实验来源记录，所需数值已经归档。本轮专用诊断脚本也已清理。

## 1. 三个数据集的主结果

以下为验证 mIoU (%)；Linear / Decoder / FT 采用本轮统一配置。

| Method | DALES Linear | Decoder | FT | OpenGF Linear | Decoder | FT | Vaihingen Linear | Decoder | FT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Scratch | 14.63 | 49.77 | 59.98 | 58.98 | 86.34 | 84.45 | 6.07 | 31.60 | 38.42 |
| Point-M2AE | 25.37 | 52.44 | 37.54 | 71.34 | 88.13 | 85.15 | 12.58 | 35.52 | 21.17 |
| MSC | 35.52 | 64.62 | 70.15 | 73.67 | 90.45 | 91.97 | 13.09 | 51.51 | 54.40 |
| Sonata | 34.51 | 64.86 | 68.12 | 81.32 | 90.94 | 91.39 | 15.94 | 49.51 | 49.12 |
| D-DITR | 23.78 | 52.82 | 51.25 | 64.76 | 84.64 | 85.67 | 12.06 | 37.89 | 39.66 |
| Concerto | 26.99 | 46.32 | 43.91 | 74.67 | 84.68 | 85.88 | 14.79 | 33.96 | 32.49 |
| VPSD | 44.09 | 66.18 | 71.16 | 79.07 | 91.52 | **92.25** | 31.71 | 53.55 | 56.87 |
| RRCL | 31.89 | **66.93** | 71.20 | 85.15 | 89.84 | 91.59 | 13.05 | 52.60 | 54.32 |
| CVRL | **52.39** | 66.29 | **72.14** | **85.52** | 90.68 | 92.13 | **38.28** | **55.02** | **56.89** |

### 1.1 CVRL 的优势在哪

- Linear：比该列第二名高 DALES +8.30 pp、OpenGF +0.37 pp、Vaihingen +6.57 pp。OpenGF 的差距很小，不能称为显著提升。
- FT：相对 Scratch 高 DALES +12.16 pp、OpenGF +7.68 pp、Vaihingen +18.47 pp。
- FT 相对 VPSD：DALES +0.98 pp、OpenGF −0.12 pp、Vaihingen +0.02 pp。联合回波监督未在这一短程 FT 上带来全面大幅提升。
- Decoder：DALES 比 RRCL 低 0.64 pp，OpenGF 比 VPSD 低 0.84 pp，Vaihingen 则最高。

解释必须区分“完整 CVRL 表示”与“CVRL encoder”：Linear 和 FT 保留 CVRL 200D 视觉—回波输出；Decoder 只使用 encoder 并换成统一 LitePT decoder，CVRL 的视觉 projector 与融合读出被移除。所以 Decoder 不是完整 CVRL 融合表示的直接检验。

此外，Linear 的表示维度和读取位置不同：多数方法为 LitePT decoder 72D，VPSD 为视觉 128D，CVRL 为 200D。这是一项实际表示读出比较，不能单独用来证明在同维度、同读取位置下 CVRL encoder 必然更好。容量控制可引用 4.4 的实验，但不能把该控制自动当成三个下游数据集均已完成的控制。

## 2. 训练—验证损失异常是否已修复

| Dataset | Method / mode | Train loss | Val loss | Val − Train |
|---|---|---:|---:|---:|
| DALES | Scratch FT | 0.9459 | 0.6454 | −0.3005 |
| DALES | Point-M2AE FT | 1.5861 | 1.2879 | −0.2982 |
| DALES | D-DITR FT | 1.1207 | 0.7938 | −0.3269 |
| DALES | Concerto FT | 1.2210 | 0.9199 | −0.3011 |
| DALES | CVRL FT | 0.6707 | 0.4715 | −0.1992 |
| OpenGF | Scratch FT | 0.4063 | 0.4800 | +0.0737 |
| OpenGF | D-DITR FT | 0.3657 | 0.4804 | +0.1147 |
| OpenGF | Concerto FT | 0.3714 | 0.4676 | +0.0962 |
| OpenGF | CVRL FT | 0.2901 | 0.2966 | +0.0065 |
| Vaihingen | Point-M2AE FT | 2.4787 | 2.3875 | −0.0912 |
| Vaihingen | D-DITR FT | 2.0030 | 1.7698 | −0.2332 |
| Vaihingen | Concerto FT | 2.0883 | 1.9504 | −0.1379 |

DALES 的 27 项中，验证损失全部低于 epoch 平均训练损失。此前 D-DITR/Concerto 的严重倒挂已经消失。OpenGF 多种方法仍有小幅正差，Scratch 也存在；它不再是这两个方法独有的异常。

这里的 Train loss 是整轮、不同历史参数、带增强和加权采样的平均；Val loss 使用训练结束参数、无增强验证数据。两者不是同一分布上的配对测量。因此，“Val > Train”本身不足以判断运行错误，“Val < Train”也不能证明没有任何 BN 分布偏差。以后若仍追查失配，应对同一固定干净训练 batch 使用最终参数比较 train/eval 模式输出及损失。

## 3. DropPath 与 BN 的证据需要分开

### 3.1 前一轮成对实验提供 DropPath 的直接证据

| DALES FT | 开启 DropPath | 关闭 DropPath | Δ mIoU (pp) |
|---|---:|---:|---:|
| D-DITR | 36.04 | 53.41 | +17.37 |
| Concerto | 32.15 | 42.13 | +9.98 |

该实验在同一诊断配置内切换 stochastic-module 状态，普通 Dropout 概率为 0，实际变化是有效的 encoder DropPath。这支持 DropPath 是此前 FT 运行异常的重要原因。

本轮同时改变了 FT LR 分组和 Precise BN，不能将全部改进都归因于 DropPath，也不能由关闭后的残余低分断言 DropPath 是唯一原因。

### 3.2 Decoder 的结果支持 Precise BN 的独立作用

相对 `exp/downstream_smoke_bncal/dales`：

| Method | Decoder：旧 → 新 | Δ (pp) | FT：旧 → 新 | Δ (pp) |
|---|---|---:|---|---:|
| Point-M2AE | 45.31 → 52.44 | +7.13 | 32.35 → 37.54 | +5.19 |
| D-DITR | 38.49 → 52.82 | +14.33 | 17.91 → 51.25 | +33.34 |
| Concerto | 17.92 → 46.32 | +28.40 | 23.84 → 43.91 | +20.07 |

Decoder 模式中的 encoder 已冻结并保持 eval；decoder 又未使用 attention，因此旧配置的有效 DropPath 本来就关闭。新的 Decoder 大幅提升主要支持“新 decoder 的训练后 BN 统计重估”有效，而非 encoder DropPath 的作用。Scratch Decoder 反而由 53.89 降到 49.77，说明 Precise BN 的效果也不是对任何初始化都必然正向。

LP 在 DALES 上基本保持原值：CVRL +0.01 pp、VPSD +0.01 pp、MSC −0.01 pp、Sonata +0.01 pp、RRCL 0.00 pp、Point-M2AE +0.01 pp。这符合冻结表示和 BN 统计保持固定的设计。

## 4. 剩余弱基线问题的具体表现

### 4.1 Point-M2AE：优先排查早期全量更新与预训练 decoder

DALES FT 比自己的 Decoder 低 14.90 pp，比 Scratch FT 低 22.44 pp。Vaihingen FT 比自己的 Decoder 低 14.35 pp，比 Scratch FT 低 17.25 pp。OpenGF FT 则比 Scratch FT 高 0.70 pp。

它的 Linear 在三个数据集均高于 Scratch，Decoder 也均高于 Scratch，说明权重并非完全没有信息。问题集中于允许全网络更新且使用预训练 decoder 的 FT 组合。

但 Decoder→FT 的比较同时改变了 encoder 是否可训练和 decoder 初始化方式，因此还不能把下降精确归因于 encoder 被破坏或 decoder 不适配中的某一个。最有辨别力的后续消融是在相同 FT 学习率、立即全网络训练下，比较“保留预训练 decoder”和“只加载 encoder、重置 decoder”。不用降低 LR，也不用 staged unfreeze。

DALES 类别 IoU 表明它不仅少数类差：Ground 91.33、Vegetation 81.07、Building 69.65；Scratch 分别为 95.61、88.16、90.96。Cars 16.96、Trucks 0、Poles 0。Vaihingen Roof 29.35、Car 0.15，也明显未充分学好。

### 4.2 D-DITR：运行正常，但多类别迁移速度仍弱

DALES FT 51.25，比 Scratch 低 8.73 pp；Vaihingen FT 39.66，比 Scratch 高 1.24 pp；OpenGF FT 85.67，比 Scratch 高 1.22 pp。

DALES 的 Ground/Vegetation/Building IoU 为 94.21/86.94/86.31，已经没有大类整体坍塌。相对 Scratch 的缺口主要来自 Power lines、Poles、Cars、Fences，分别低 23.87、19.23、8.35、10.75 pp。它不是简单“语义类别映射失效”。

### 4.3 Concerto：残余弱点不限于 FT

DALES Decoder 46.32 已低于 Scratch Decoder 49.77；FT 43.91 比 Scratch FT 低 16.07 pp。Vaihingen Decoder 33.96 高于 Scratch 31.60，但 FT 32.49 低于 Scratch 38.42。OpenGF FT 85.88 高于 Scratch 84.45。

因此 Concerto 的残余问题还包括 encoder 表示在多类 ALS 中的早期可利用程度，以及新 decoder 的学习速度。DALES Building IoU 78.60 和 Power lines 29.48 明显低于 Scratch 90.96/85.51。Vaihingen Car IoU 0.01，Roof 51.54。这些结果需要完整训练曲线及预训练适配检查，不能因为现代方法在其他领域先进就事后选择有利配置或删掉弱结果。

### 4.4 对原因可以说到什么程度

- 已证实：旧 FT 的 DropPath 可造成严重验证失配；新 decoder 的 BN 重估可显著改善部分方法。
- 有证据支持：弱方法在相同短训练预算中存在更慢或更差的优化，且问题依赖数据集任务。
- 待验证：预训练 decoder、特征幅值/有效秩、encoder 更新漂移、跨模态预训练适配、BN 深层统计误差分别贡献多少。
- 不能声称：这些方法收敛后必然低于 Scratch，或上述某一种推断已经是唯一根因。

## 5. 方法互补的类别证据

DALES FT 中，CVRL 在 Trucks/Fences/Poles/Building IoU 上为 20.25/47.99/65.48/95.14；VPSD 为 15.91/46.92/61.88/94.97；RRCL 为 12.93/45.99/65.38/94.60。CVRL 在部分小目标和结构类上保留了较好的早期优势，但 Cars 和 Power lines 没有超越两分支最佳值。

Vaihingen FT 中，RRCL Powerline IoU 49.02，VPSD 为 39.73，CVRL 为 44.73；VPSD Roof IoU 89.18，RRCL 为 86.56，CVRL 为 87.70。CVRL 在 Fence/hedge 上为 28.29，高于 VPSD 24.79 和 RRCL 18.45。这里体现的是不同目标的互补潜力与联合训练权衡，而非每类“取两分支最大值”。

OpenGF 只有 Ground/Non-ground 两类，语义需求不同且快速趋近高分；VPSD 92.25 与 CVRL 92.13 的差距不支持显著优劣判断。

## 6. 正式实验前必须明确的边界

1. 短程预算不等于收敛预算。实际 DALES 为 241 micro-batches（约 120 次 optimizer update），OpenGF 为 629（约 314 次），Vaihingen 为 48（24 次）。尤其 Vaihingen Linear 分类头远未证明收敛，不能把低分直接当作 frozen representation 上限。
2. 短程配置把完整 cosine schedule 压缩到一轮。它不是正式 10-epoch/5-loop 实验的第一轮，学习率暴露过程不同。本轮排名体现短预算学习能力，不能简单外推正式排名。
3. 最终 headline 表应统一协议，保留所有方法；若加入初始化消融，应对所有具有 pretrained decoder 的方法进行对称的规则或明确单列，避免仅优化弱基线而引入隐藏差异。
4. 当前三个数据集均配置 val=test 路径。若正式实验按该 split 选择 best checkpoint 后又在同一 split 报 test，就发生测试集用于模型选择，不能称为独立 held-out test。正式论文需要从训练 source blocks 单独划分 validation，或固定终点 checkpoint 并只在独立测试集做最终评价；当前短程阶段虽只有一次 val，也不能掩盖后续模型选择使用问题。
5. Precise BN 当前同时让所有选中 BN 使用 batch statistics，再汇总各层输入矩。其均值/方差对所采集的输入是精确矩，但并不保证深层统计等于使用最终全局 BN buffers 的 eval 网络输入分布。固定干净 batch 的 train/eval 对照可判断残余影响。
6. BN 矩用 FP32 的 `sum(x²) − n·mean²` 估计方差，在大均值、小方差通道存在消减误差风险。先前 Swiss 校准报告里 D-DITR/Concerto 深层有这种数值形态；这只是风险线索，不能作为此次残余低分的已证实原因。若继续诊断，应核验稳定在线中心矩或高精度累积与现实现的差异。
7. 本轮仅验证，没有多 fragment 的 test，不能由此回答之前“Decoder test 低于 val”的问题。正式结果还应检验两种推理路径的一致性。

## 7. 推荐的下一步顺序

### 第一优先级：固定终点预算和数据划分

保持 DropPath=0、当前学习率、FT 第一个 step 全网络训练。先确定独立 validation 划分或固定终点评价规则，然后运行完整统一训练预算。避免继续用单轮 mIoU 对预训练方法作最终排序。

### 第二优先级：一个辨别力较高的 FT 初始化实验

在 Point-M2AE、D-DITR、Concerto 上比较：

- A：当前 FT，保留 pretrained encoder + decoder。
- B：encoder-only 初始化，native decoder/head 随机初始化，从第一步全网络训练。

学习率、BN、DropPath、batch、增强、类别权重和训练预算完全相同。该实验不引入 staged unfreeze，也不要求下调学习率。Point-M2AE 优先，因为两个多类数据集均出现超过 14 pp 的 FT–Decoder 倒退。

### 第三优先级：同 batch 的诊断及 checkpoint 表示审计

使用最终参数，在一组固定干净训练点云上记录 train/eval 损失、logit 幅值、BN 输入均值/方差、encoder/P0–P4/decoder 特征 RMS 和有效秩；对比 update 前后漂移。对 BN 测试先保留快照并恢复 buffers，防止诊断本身改变 checkpoint。它可以区分“最终特征本身难分类”与“eval 统计改变了特征”。

### 最终论文可支持的陈述

现有证据支持统一确定性迁移和合适 BN 统计处理对视觉蒸馏模型至关重要，CVRL 在短预算的 frozen readout 与多类迁移上表现良好。最终下游优势、显著性以及对不同 SSL 范式的比较仍应基于完整训练、独立测试和多个种子报告。
