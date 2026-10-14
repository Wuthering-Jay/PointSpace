# 4.4 Return Structure as a Physical Complement：完备实验报告

## 1. 结论摘要

本轮在 Swiss-Audit 的固定 406 个 tile、140 个原始 source block 上完成了 P1–P4。所有 frozen readout 都复用 4.3 的 source-block 划分；线性 probe 使用相同的 100,000/100,000 个训练/测试点，cosine kNN 使用相同的 5,000/5,000 个点。主指标为 mIoU，同时保存 OA、macro-F1、逐类 IoU/Precision/Recall/F1、混淆矩阵和 source-block 配对 bootstrap 区间。

结果支持以下三个结论：

1. **回波监督并非普通坐标几何监督的替代写法。** Full RRCL 的 frozen-linear mIoU 为 0.2854，显著高于 Normal 的 0.1735 和 Local shape 的 0.2188；相对增益分别为 +0.1119 和 +0.0666，95% 配对区间均不跨零。
2. **回波表征在穿透/遮挡结构上的优势具有明确的结构选择性。** Full RRCL 相对 Local shape 在林下 Ground 与开阔 Ground 间的 Recall 差中之差为 +0.4047，95% CI [0.2626, 0.5000]；相对 Role-only 仍为 +0.1032，[0.0390, 0.1432]。在 CVRL 内部，正确加入回波支路使林下 Ground Recall 提升 +25.03 点，而开阔 Ground 仅提升 +3.60 点。
3. **CVRL 的收益来自正确逐点配对的回波子空间，而非单纯增加 72 维。** 正确的 `zV‖zR` 在线性 mIoU 上为 0.5127；随机 72D 和 source-block 内错配 `zR` 分别为 0.4501±0.0014 和 0.4539±0.0020。cosine kNN 结论一致。

但结果不支持以下过强表述：

- **不能声称 Full RRCL 在所有读出上优于 Role-only。** Role-only 的全域 frozen-linear mIoU 为 0.3006，比 Full RRCL 高 0.0152，95% CI 对 Full−Role 为 [−0.0253, −0.0011]。Full RRCL 的优势主要出现在 cosine kNN、`P_inv` 和林下 Ground 等结构性条件，而不是全域线性读出。
- **不能声称回波目标完全不受低级几何 shortcut 影响。** 仅输入强度与 XYZ 派生局部统计的轻量 MLP 已能获得 role mIoU 0.4121、context MAE 0.1508；不过它在 `P_inv`、植被内部和林下地面明显变差，且远非近乎完美预测。
- 当前结论来自单个预训练 seed。block bootstrap 衡量空间测试块不确定性，不能替代多个训练 seed。

因此，最稳妥的论文结论是：**return role/context supervision 学到了超出常规表面几何目标的结构信息；这种信息能够条件性补偿视觉表示，尤其有利于林下地面和复杂垂直结构。** 不宜写成“Full RRCL 普遍优于所有消融”或“回波只对不可见点有效”。

## 2. 实验协议与完整性

### 2.1 固定评估协议

- 数据：Swiss 100×100 m，训练 grid size 0.25 m；Swiss-Audit 共 406 tile。
- 划分：以原始 source block 为单位，测试侧 28 个 block；不同表示使用完全相同的行。
- Linear：逐点 L2 normalize，类别平衡 multinomial logistic regression。
- kNN：cosine distance、distance weighting、`k=20`。
- 指标：mIoU 为主；OA、macro-F1、逐类 IoU/Precision/Recall/F1 和 confusion matrix 为辅。
- 不确定性：P1/P2 采用 2,000 次 source-block 配对 bootstrap；P3 采用 1,000 次。
- 分层信息：LAS 类别、`q_geo` 与高度结构只在预训练完成后用于审计，不进入 SSL 输入。

### 2.2 模型与训练状态

| 模型 | 目标 | 参数量 | 有效 checkpoint | 近似训练时长 | 审计特征 |
|---|---|---:|---:|---:|---:|
| Scratch-LitePT | 无 | 15.361 M | seed-42 随机初始化 | – | 406/406 |
| Normal | 单位法向，符号不敏感 cosine | 15.361 M | epoch 25 | 6 h 22 min | 406/406 |
| Local shape | 六维 PCA 几何描述，Smooth-L1 | 15.361 M | epoch 25 | 6 h 17 min | 406/406 |
| Role-only | S/F/M/L weighted CE | 15.361 M | epoch 25 | 6 h 12 min | 406/406 |
| Context-only | K=48 soft role histogram | 15.361 M | **人工早停 epoch 20** | 6 h 11 min | 406/406 |
| Full RRCL | Role + Context | 15.366 M | epoch 25 | 8 h 50 min | 406/406 |

Context-only 的 epoch 21 未完成，未被使用；报告使用最后一个完整的 epoch-20 checkpoint。Local-shape 仅原特征导出被中断，训练本身已完成 epoch 25，本轮通过断点续传补齐剩余 34 个 tile。

严格而言，Context-only 与其余模型训练 epoch 不完全相同。因此它可以作为“人工早停后的最佳可用对照”，但 Full−Context 的差异不应被解释成纯目标因果效应。Role-only、Full RRCL 与两种几何基线仍保持 25 epoch 可比。

## 3. P1：受控内在自监督对照

### 3.1 主结果

| 表示 | Linear All | Linear `P_inv` | Linear `P_amb` | Linear `P_vis` | kNN All | kNN `P_inv` | kNN `P_amb` | kNN `P_vis` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Scratch | 0.0594 | 0.0574 | 0.0592 | 0.0613 | 0.0761 | 0.0641 | 0.0807 | 0.0803 |
| Normal | 0.1735 | 0.1616 | 0.1626 | 0.1871 | 0.1424 | 0.1441 | 0.1458 | 0.1391 |
| Local shape | 0.2188 | 0.2064 | 0.2071 | 0.2364 | 0.1852 | 0.1908 | 0.1954 | 0.1747 |
| Role-only | **0.3006** | **0.2911** | **0.2894** | **0.3134** | 0.2599 | 0.2382 | 0.2231 | **0.2765** |
| Context-only | 0.2528 | 0.2569 | 0.2485 | 0.2548 | 0.2256 | 0.2039 | 0.2266 | 0.2385 |
| Full RRCL | 0.2854 | 0.2893 | 0.2821 | 0.2892 | **0.2709** | **0.2897** | **0.2387** | 0.2646 |

All 区域的线性辅助指标为：

| 表示 | OA | macro-F1 | mIoU |
|---|---:|---:|---:|
| Scratch | 0.1924 | 0.1037 | 0.0594 |
| Normal | 0.4628 | 0.2671 | 0.1735 |
| Local shape | 0.5459 | 0.3234 | 0.2188 |
| Role-only | **0.6696** | **0.4223** | **0.3006** |
| Context-only | 0.6073 | 0.3617 | 0.2528 |
| Full RRCL | 0.6531 | 0.4028 | 0.2854 |

Full RRCL 相对各对照的全域 linear-mIoU 配对差值：

| 对照 | Full−对照 | 95% CI | 判断 |
|---|---:|---:|---|
| Scratch | +0.2260 | [0.1865, 0.2653] | 显著 |
| Normal | +0.1119 | [0.0741, 0.1470] | 显著 |
| Local shape | +0.0666 | [0.0354, 0.0966] | 显著 |
| Role-only | −0.0152 | [−0.0253, −0.0011] | Role-only 显著更高 |
| Context-only | +0.0326 | [0.0144, 0.0500] | 显著；受早停预算限制 |

### 3.2 如何解释 Role-only 与 Full RRCL 的反转

这个结果不是“Context 无效”。两种读出反映的结构不同：

- 在线性读出中，Role-only 全域比 Full RRCL 高 1.52 个百分点，在 `P_vis` 高 2.42 点；说明附加 context 目标可能牺牲了一部分直接线性可分性。
- 在 cosine kNN 中，Full RRCL 全域比 Role-only 高 1.10 点，在 `P_inv` 高 **5.16 点**，但在 `P_vis` 低 1.19 点；说明 Context 更可能改善不可见区域的局部邻域结构，而非全域线性判别边界。
- P2 中 Full RRCL 相对 Role-only 的“林下 Ground−开阔 Ground”差中之差仍为 +10.32 点，[+3.90, +14.32]；这进一步支持 Context 的价值是结构条件性的。

因此正文应把 Role 与 Context 描述为不同偏置：Role 提供更强的直接点级类别线索；Context 改善局部结构组织，尤其在 `P_inv`/林下结构中更明显。不要仅凭全域 linear mIoU 删除 Context。

### 3.3 逐类 IoU

| LAS class | Scratch | Normal | Local shape | Role-only | Context-only | Full RRCL |
|---|---:|---:|---:|---:|---:|---:|
| Unclassified | 0.0333 | 0.1293 | 0.1930 | 0.3070 | 0.2659 | **0.3090** |
| Ground | 0.1412 | 0.4197 | 0.4910 | **0.6478** | 0.5880 | 0.6090 |
| Vegetation | 0.2518 | 0.5186 | 0.6647 | 0.7612 | 0.6984 | **0.7695** |
| Building roof | 0.0414 | 0.1713 | 0.1894 | 0.3773 | 0.3483 | **0.3780** |
| Water | 0.0456 | 0.1890 | 0.2233 | **0.2391** | 0.1795 | 0.1527 |
| Wire / line | 0.0046 | 0.0135 | 0.0372 | 0.1398 | 0.1136 | **0.1433** |
| Tower | 0.0061 | 0.0289 | 0.0598 | **0.1675** | 0.0672 | 0.1395 |
| Bridge deck | 0.0496 | **0.1494** | 0.1472 | 0.1140 | 0.1287 | 0.1257 |
| Building facade | 0.0147 | 0.0892 | 0.1323 | **0.2016** | 0.1046 | 0.1743 |
| Bridge pier / cable | 0.0056 | 0.0259 | 0.0499 | 0.0503 | 0.0339 | **0.0526** |

罕见类的单 seed 数值方差可能很高，应以完整 JSON 中的 support、混淆矩阵和后续多 seed 为准。当前较稳定的模式是 Ground/Vegetation/Building roof 上的 return 监督优势；Water、Tower、Bridge 类不宜作强结论。

## 4. P2：结构分层与物理互补

### 4.1 CVRL 内部的条件增量

下表比较同一 CVRL checkpoint 的 `full−zV`，因而比独立 VPSD/RRCL 的绝对差异更接近条件增量。

| 测试层 | 指标 | 点数 | 增益 | 95% CI |
|---|---|---:|---:|---:|
| All | mIoU | 100,000 | +5.50 点 | [+3.38, +6.86] |
| `P_inv` | mIoU | 29,182 | +5.59 点 | [+3.69, +7.54] |
| `P_vis` | mIoU | 43,092 | +5.33 点 | [+1.96, +7.26] |
| Under-canopy Ground | Ground Recall | 7,084 | **+25.03 点** | [+17.39, +32.84] |
| Exposed Ground | Ground Recall | 11,448 | +3.60 点 | [+1.51, +6.57] |
| Vegetation interior | Vegetation Recall | 10,144 | +5.03 点 | [+3.45, +6.86] |
| Visible vegetation | Vegetation Recall | 12,643 | +3.84 点 | [+2.65, +5.04] |
| Multi-layer | mIoU | 65,685 | +7.10 点 | [+4.43, +8.68] |
| Single-layer | mIoU | 32,424 | +2.29 点 | [+1.28, +4.00] |

对应差中之差：Under-canopy−Exposed Ground 为 +21.43 点，[+14.31, +28.22]；Vegetation interior−visible 为 +1.18 点，[+0.16, +2.30]；Multi-layer−Single-layer 为 +4.81 点，[+1.22, +6.52]。密度限制到 16–64 点/柱后，Ground 交互方向保持，而 Multi-layer 交互区间跨零。因此最可靠的是林下 Ground，不是笼统的“所有多层区域”。

### 4.2 RRCL 与几何 pretext 的结构对照

| 模型 | Under-canopy Ground Recall | Exposed Ground Recall | Vegetation interior Recall | Visible vegetation Recall | Multi-layer mIoU | Single-layer mIoU |
|---|---:|---:|---:|---:|---:|---:|
| Normal | 0.5011 | 0.5788 | 0.6408 | 0.6522 | 0.1687 | 0.1290 |
| Local shape | 0.6210 | **0.6047** | 0.7437 | 0.7493 | 0.2155 | 0.1559 |
| Role-only | **0.9078** | 0.5901 | 0.8368 | **0.7811** | **0.3256** | **0.1775** |
| Context-only | 0.8628 | 0.5048 | 0.8080 | 0.6809 | 0.2752 | 0.1524 |
| Full RRCL | 0.9027 | 0.4818 | **0.8404** | 0.7697 | 0.3117 | 0.1610 |

Full RRCL 相对 Local shape 的结构差中之差：

- Under-canopy−Exposed Ground Recall：+0.4047，[0.2626, 0.5000]；
- Vegetation interior−Visible Recall：+0.0763，[0.0425, 0.1195]；
- Multi-layer−Single-layer mIoU：+0.0911，[0.0566, 0.1157]；密度限制后仍为 +0.0993，[0.0545, 0.1232]。

这些结果比仅报告 All mIoU 更直接地支持“physical complement”：即使 Normal/Local shape 已显式监督局部表面方向与 PCA 几何，RRCL 对林下及多层结构的相对优势仍然存在。

不过 Full RRCL 对开阔 Ground 的 Recall 低于各单任务和几何基线。这是有意义的负结果：回波目标的表征偏置不是全场景最优，视觉支路正好可以补偿这种表面语义弱项。

## 5. P3：Representation surgery

首先逐元素验证导出的 `CVRL-full = [zV(128) ‖ zR(72)]`，最大绝对误差为 0。随机控制逐维保留真实 `zR` 的边际；shuffle 控制在 source block 内分别对训练/测试行置换，保留场景分布但破坏逐点对应。每种控制 5 个 seed。

| 200D 表示 | Linear OA | Linear mIoU | Linear macro-F1 | kNN OA | kNN mIoU | kNN macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| 正确 `zV‖zR` | **0.8256** | **0.5127** | **0.6548** | **0.8166** | **0.4003** | **0.5103** |
| `zV‖random72` | 0.7447±0.0003 | 0.4501±0.0014 | 0.5991±0.0016 | 0.6929±0.0022 | 0.3419±0.0063 | 0.4664±0.0083 |
| `zV‖shuffle(zR)` | 0.7451±0.0006 | 0.4539±0.0020 | 0.6037±0.0030 | 0.6833±0.0026 | 0.3272±0.0122 | 0.4492±0.0165 |

正确配对比 random/shuffle 分别高 6.26/5.88 个 linear-mIoU 点。每个随机 seed 的配对 block CI 均不跨零。先对两支分别 L2 normalize 再拼接时，正确配对 linear mIoU 仍为 0.5122，说明线性收益不依赖原始分支范数；kNN 则从 0.4003 降为 0.3749，表明原始范数比例包含部分邻域信息。

P3 支持“逐点回波表征包含真实增量信息”，但不能证明共享浅层的梯度完全无冲突，也不能单独证明该信息只能来自激光测量过程。后一个问题由 P1/P4 联合约束。

## 6. P4：低级 shortcut 诊断

### 6.1 设置

轻量双头 MLP 只输入 13 维廉价属性：`log intensity`、六个局部高度/密度统计、六个由 XYZ-PCA 得到的 shape descriptor。它不读取 return number、number of returns、LAS class、`q_geo` 或视觉特征。训练/验证按 source block 分离；测试沿用 P1/P2 的固定 100k 行，其中 99,614 个点具有有效 role/context。目标 K 与 Swiss RRCL 配置一致，为 48。

这项实验只衡量“目标可预测性”，不能把 MLP 的 target 指标当成点云 backbone 的下游迁移能力。Context target 在固定原始 tile 的 raw KNN 上计算，而训练期 RRCL 在增强/体素采样后的当前点集上计算；二者物理含义一致但并非逐样本完全相同，这是本诊断的近似限制。

### 6.2 结果

| 区域 | Role OA | Role mIoU | Role macro-F1 | Context MAE | Context KL | Target entropy |
|---|---:|---:|---:|---:|---:|---:|
| All | 0.6244 | 0.4121 | 0.5735 | 0.1508 | 0.3654 | 0.6065 |
| `P_inv` | 0.5412 | 0.3530 | 0.5154 | 0.1622 | 0.3941 | 0.7172 |
| `P_amb` | 0.5621 | 0.3717 | 0.5355 | 0.1549 | 0.3689 | 0.7016 |
| `P_vis` | 0.7206 | 0.4342 | 0.5808 | **0.1404** | **0.3437** | 0.4707 |
| Vegetation interior | 0.4656 | 0.2671 | 0.4101 | 0.1821 | 0.4642 | 0.8592 |
| Under-canopy Ground | 0.6269 | 0.2290 | 0.3161 | 0.1881 | 0.4689 | 0.5013 |
| Multi-layer | 0.5385 | 0.3657 | 0.5333 | 0.1677 | 0.4186 | 0.7332 |

固定训练先验基线在 All 上的 role mIoU 仅 0.1064、context MAE 为 0.2279、context KL 为 0.6498。MLP 明显优于先验，说明 return 目标确实与低级几何统计相关；这符合 LiDAR 回波由局部结构产生的物理事实，并不是自动的数据泄漏证据。

另一方面，最需要 RRCL 的区域恰好更难被廉价几何解释：`P_inv` 的 role mIoU 比 `P_vis` 低 8.11 点；Vegetation interior 和 Under-canopy Ground 的 context MAE 分别升至 0.1821/0.1881。结合 P1 中 Local shape 显著落后于 RRCL，可判断当前结果**不支持“RRCL 只是把局部 PCA 几何重新编码”**。

MLP 并未达到近乎完美预测，因此按预注册停止门槛，暂不需要把 proxy-SSL 作为 4.4 主实验。若审稿人要求更强排除，可额外训练“同 LitePT、预测这 13 维代理特征”的 proxy-SSL；它应作为补充实验，而不是用 MLP target accuracy 替代迁移评估。

## 7. 论文可支撑的表述

### 7.1 可以使用

> Compared with coordinate-derived normal and local-shape objectives under the same LitePT architecture, return-role/context supervision yields substantially stronger frozen representations. Its relative advantage is most pronounced for under-canopy ground and vertically layered structures, indicating that laser-return structure provides a physical complement to surface-oriented geometry and visual supervision.

> Representation surgery further shows that the gain of the fused CVRL feature depends on the correctly aligned point-wise return subspace rather than merely adding 72 dimensions or preserving scene-level feature marginals.

### 7.2 应避免

- “Full RRCL consistently outperforms Role-only”：不成立，linear probe 中 Role-only 更高。
- “Return information only helps invisible points”：不成立，All 与 `P_vis` 也有增益；更准确的是在林下 Ground 上增益显著放大。
- “Context is always beneficial”：不成立；它改善 `P_inv` kNN/结构组织，但会牺牲部分全域线性可分性。
- “Return targets cannot be inferred from geometry”：过强；P4 已证明能被部分预测。
- “Results prove cross-region generalization”：不成立；Swiss-Audit 位于 Swiss 预训练覆盖范围内。

## 8. 仍需补充与优先级

1. **最高优先级：多训练 seed。** 至少为 Role-only、Full RRCL 和 Local shape 各补 2 个 seed，报告预训练 seed 均值±标准差。当前 block CI 不能替代它。
2. **Context-only 公平性。** 若正文需要精确比较 Full 与 Context-only，应补完相同的 25 epoch，或统一采用预注册的验证早停协议重跑所有 P1；否则明确标注 epoch 20 early-stop。
3. **Rare-class 结论降级。** Bridge、Tower、Wire 等 support 较少，逐类表保留为补充材料，不作为主要机制结论。
4. **4.5 外部迁移。** 4.4 已回答预训练覆盖域内机制；真正的下游和跨区域有效性应由 DALES 等 4.5 实验决定。
5. **可选 proxy-SSL。** 仅在审稿需要进一步排除低级统计捷径时执行，不是当前主结论成立的前置条件。

## 9. 产物索引

- P1 数值：[p1_intrinsic_ssl.json](results/section_44/p1_intrinsic_ssl.json)
- P2 数值：[p2_strata.json](results/section_44/p2_strata.json)
- P3 数值：[p3_surgery.json](results/section_44/p3_surgery.json)
- P4 数值：[p4_shortcut.json](results/section_44/p4_shortcut.json)
- P1 图：[figure_44_p1_intrinsic.png](results/section_44/figure_44_p1_intrinsic.png)
- P2 图：[figure_44_p2_structure.png](results/section_44/figure_44_p2_structure.png)
- P3 图：[figure_44_p3_surgery.png](results/section_44/figure_44_p3_surgery.png)
- P4 图：[figure_44_p4_shortcut.png](results/section_44/figure_44_p4_shortcut.png)
- P1 评估：[analyze_cvrl_44_p1.py](analyze_cvrl_44_p1.py)
- P2 分层：[analyze_cvrl_44_strata.py](analyze_cvrl_44_strata.py)
- P3 手术：[analyze_cvrl_44_surgery.py](analyze_cvrl_44_surgery.py)
- P4 shortcut：[analyze_cvrl_44_shortcut.py](analyze_cvrl_44_shortcut.py)
- 绘图：[plot_cvrl_44.py](plot_cvrl_44.py)

所有 JSON 都保存了比正文表格更完整的指标和混淆矩阵，可直接用于论文补充材料或重新排版。
