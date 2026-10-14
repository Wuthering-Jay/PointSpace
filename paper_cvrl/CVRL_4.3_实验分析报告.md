# CVRL 4.3：Foundation Knowledge Internalization and Visibility Mechanism Analysis

## 1. 实验状态与结论摘要

本报告严格对应实验章大纲 4.3-P1 至 4.3-P5。当前已经完成：

- Swiss-Audit 数据选择、纯几何 `q_geo`、`P_vis/P_amb/P_inv` 分区；
- DINO-1024、PCA-DINO-128、VPSD-1024、VPSD-128、CVRL-visual-1024、CVRL-visual-128 六级语义审计；
- VPSD-Scratch-128、VPSD-128、**独立预训练 RRCL-72**、CVRL-visual-128、联合 CVRL-return-72、CVRL-full-200 的区域分层 frozen linear probe 与 cosine kNN；
- 场景、LAS类别、相对高度三类可见性瓶颈统计；
- teacher-student 配对 cosine、类别级结果、主要误差组合与空间分布图；
- 406块tile上的影像patch可见表面类别代理冲突审计，以及林下地面/建筑立面的DOM–ALS垂直剖面实例。

当前结果给出三条较清晰的结论：

1. **视觉基础模型知识确实被点云网络内化。** VPSD-1024 的 linear mIoU 为 `0.5014`，高于直接把DINO patch特征映射到点上的 `0.3876`；VPSD-128在全域也比同构随机初始化高 `33.97` 个百分点，在`P_inv`高 `32.05` 点。这同时排除了“仅靠网络结构或随机高维映射即可得到结果”的解释。
2. **固定PCA-128存在损失，但对学生表示的损失较小。** DINO由1024D降至128D损失 `6.39` 个mIoU百分点；VPSD和CVRL视觉分支分别只损失 `2.01` 和 `1.87` 点，能力保留率约 `96%`。
3. **投影语义冲突确实存在，但具有类别选择性。** 在影像patch内至少5个有效对应点、可见表面主类纯度至少70%的保守审计下，1,405,737个审计点合格。`P_inv`与同patch可见表面主类冲突率为`41.84%`，`P_vis`为`17.48%`，差`24.35`个百分点（140个source block bootstrap 95% CI：`22.74–25.98`点）。林下地面的冲突尤甚；植被内部仍可能与可见树冠同属vegetation，不能把所有`P_inv`当作类别冲突。
4. **独立RRCL给出与VPSD不同的类别优势，CVRL融合提供稳定互补，但“只补偿P_inv”并不成立。** 独立RRCL在`P_inv`地面F1为`0.831`、VPSD为`0.744`；RRCL在建筑屋顶/桥面则明显较弱。CVRL-full相对CVRL-visual的linear mIoU在all/`P_inv`/`P_amb`/`P_vis`分别提升`5.50/5.59/5.51/5.33`点，增益近似均匀；cosine kNN在`P_inv`提升`6.45`点，在`P_amb/P_vis`也分别提升`0.92/0.80`点。

独立RRCL训练与406份特征导出已完成；其checkpoint SHA256与相同审计行协议已写入分析JSON。`q_geo`已完成跨环境目视检查并冻结阈值，但没有逐点人工标签，故只报告定性可靠性，不报告虚构的agreement/precision。本审计仍来自预训练footprint，不是外部泛化测试。

## 2. 数据和统一协议

### 2.1 Swiss-Audit

| 项目 | 数值 |
|---|---:|
| 原始100 m tile总数 | 13,530 |
| Swiss-Audit tile数 | 406 |
| 抽样比例 | 3.00% |
| 覆盖原始source block | 140 |
| 原始点总数 | 95,965,298 |
| 固定审计点 | 1,662,976（每tile 4,096点） |
| 具有有效影像patch对应的审计点 | 611,738（36.79%） |

Swiss-Audit来自已参与预训练的Swiss footprint，因此本节只用于**机制和表征审计**，不能包装成空间外推或跨区域泛化实验。LAS classification仅在预训练后作为评估标签，未进入VPSD、RRCL或CVRL训练目标。

### 2.2 Frozen评估协议

- split单位：原始Swiss source block，禁止point-level random split；
- 固定20% source block作为测试块；
- 所有特征逐行L2 normalize；
- linear probe：balanced multinomial Logistic Regression，LBFGS，`C=1`；
- linear：最多100,000训练点和100,000测试点；
- cosine kNN：5,000训练点、5,000测试点，`k=20`，distance weighting；
- 抽样采用相同的“类别×可见性区域”分层规则，随机种子42；
- 同一分析链的所有表示严格复用相同train/test行；旧版按表示改变随机种子的结果已作废；
- 独立RRCL的406份72D特征逐点索引与原审计索引对齐；100,000个regional linear测试行与CVRL完全一致；
- `P_amb`单独报告，但不进入`P_vis/P_inv`核心结论。

## 3. 4.3-P1：Foundation semantic internalization

![Semantic internalization](results/section_43/report_figures/figure_43_semantic_internalization.png)

### 3.1 六级特征结果

下表仅在具有有效DINO patch对应的点上训练和测试，从而保证teacher与student比较使用同一监督域。

| 表示 | 维度 | Linear OA | Linear mIoU | Cosine kNN mIoU |
|---|---:|---:|---:|---:|
| DINO native | 1024 | 0.6797 | 0.3876 | 0.2747 |
| PCA-DINO | 128 | 0.6456 | 0.3237 | 0.2874 |
| VPSD native | 1024 | **0.7948** | **0.5014** | **0.4645** |
| VPSD PCA | 128 | 0.7876 | 0.4813 | 0.4395 |
| CVRL visual native | 1024 | 0.7888 | 0.4812 | 0.4229 |
| CVRL visual PCA | 128 | 0.7791 | 0.4625 | 0.4205 |

主要观察：

- VPSD-1024相对DINO-1024提高 `11.38` 个linear mIoU百分点。直接patch teacher具有语义，但其空间粒度和点级标签不完全一致；VPSD通过三维encoder吸收邻域上下文后，点级线性可分性反而更高。
- CVRL视觉分支相对独立VPSD存在小幅弱化：1024D下降 `2.02` 点，128D下降 `1.88` 点。因此“joint learning完全无损”不成立，但视觉语义大部分得到保留。
- PCA对teacher影响最大：DINO能力保留率为 `83.52%`；VPSD和CVRL视觉分别为 `95.99%`、`96.11%`。固定PCA更适合作为已经内化后的永久点特征压缩，而不是替代原始teacher的全部语义容量。

### 3.2 配对teacher alignment

| Student与matched teacher | 点数 | Mean cosine | P10 | Median | P90 |
|---|---:|---:|---:|---:|---:|
| VPSD-1024 vs DINO-1024 | 611,738 | 0.6924 | 0.5304 | 0.7100 | 0.8334 |
| CVRL-visual-1024 vs DINO-1024 | 611,738 | **0.6951** | 0.5320 | 0.7136 | 0.8356 |
| VPSD-128 vs PCA-DINO-128 | 611,738 | 0.6115 | 0.3407 | 0.6489 | 0.8271 |
| CVRL-visual-128 vs PCA-DINO-128 | 611,738 | **0.6160** | 0.3457 | 0.6543 | 0.8301 |

CVRL视觉分支的teacher cosine没有低于VPSD，反而高约 `0.0045`。因此其linear probe小幅下降不能简单归因于“没有学到teacher”；更合理的解释是联合训练改变了类内/类间几何，使其teacher方向一致性保持，但下游线性决策边界略弱。该判断仍需要多seed确认。

## 4. 4.3-P2/P3：Geometry-only visibility与瓶颈规模

![Visibility bottleneck](results/section_43/report_figures/figure_43_visibility_bottleneck.png)

### 4.1 纯几何定义

`q_geo`仅使用XYZ和多偏移、自适应splat Z-buffer，不读取return number、number of returns、echo decay、first-return bonus或总observability。因此后续“return在`P_inv`更有帮助”不存在由return定义`P_inv`的循环论证。

阈值为：

- `P_inv`: `q_geo <= 0.20`；
- `P_amb`: `0.20 < q_geo < 0.60`；
- `P_vis`: `q_geo >= 0.60`。

全量95,965,298点的结果：

| 区域 | 点数 | 比例 |
|---|---:|---:|
| P_inv | 18,936,141 | 19.73% |
| P_amb | 12,983,023 | 13.53% |
| P_vis | 64,046,134 | 66.74% |

约五分之一点属于高置信`P_inv`，绝非少量异常点；若连同不确定带，直接视觉监督不能可靠解释的范围达到33.26%。

### 4.2 场景分布

| 场景 | Tile | 点数 | P_inv | P_amb | P_vis |
|---|---:|---:|---:|---:|---:|
| Vegetation-dominant | 81 | 31,119,774 | **25.32%** | 18.50% | 56.18% |
| Low-building mosaic | 81 | 17,689,905 | 20.20% | 14.37% | 65.43% |
| Infrastructure-enriched | 86 | 19,226,850 | 20.15% | 12.23% | 67.62% |
| Residual mixed | 4 | 872,161 | 17.54% | 12.80% | 69.66% |
| Building-dominant | 89 | 14,524,028 | 15.85% | 10.24% | 73.91% |
| Water-enriched | 4 | 588,550 | 13.85% | 9.79% | 76.36% |
| Ground-dominant | 61 | 11,944,030 | **8.97%** | 5.66% | 85.37% |

Vegetation-dominant的`P_inv`是Ground-dominant的2.82倍，符合冠层、下木层和多层回波造成视觉不可见点增多的物理预期。

论文正文 Fig. 5(a)另按点云组成及建筑高度构造**互斥的六个场景层**：Tall-building、Low-rise built、Dense canopy、Sparse vegetation、Flat bare ground、Mixed / other。它们覆盖全部406个tile；与上表用于初始抽样的环境标签不是同一套分类，不能把两套分层的百分比直接对位。Fig. 5(b)包含除Unclassified外全部LAS类别，Fig. 5(c)为相对地面高度层，三个面板均显示`P_inv/P_amb/P_vis`的100%堆叠比例。

### 4.3 类别与高度分布

本文统一采用以下类别名称：1 Unclassified、2 Ground、3 Vegetation、6 Building roof、9 Water、14 Wire / transport line、15 Transmission tower、17 Bridge deck、26 Building facade、27 Bridge pier & stay cable。

核心标准类别的`P_inv`比例：vegetation `25.00%`、ground `16.33%`、water `11.38%`、unclassified `10.58%`、building roof `10.21%`、bridge deck `2.53%`。高度分层结果为：

| Height AGL | P_inv | P_amb | P_vis |
|---|---:|---:|---:|
| 0-0.5 m | 12.49% | 8.28% | 79.23% |
| 0.5-2 m | 19.55% | 12.10% | 68.35% |
| 2-5 m | **25.95%** | 16.10% | 57.95% |
| 5-15 m | 23.80% | 16.26% | 59.94% |
| 15-30 m | 19.71% | 15.93% | 64.36% |
| >30 m | 16.68% | 13.62% | 69.70% |

瓶颈峰值位于2-15 m，而非最高点或地面附近。这与低矮植被、树冠内部、建筑立面/边缘等复杂垂直结构相符。

类别26的`P_inv`达到80.93%，明显偏离其他类别。[swisstopo第二采集周期官方产品说明](https://www.swisstopo.admin.ch/dam/de/sd-web/MWC6RfRAOvXU/swissSURFACE3D_Produktinfo_Zyklus_2_DE.pdf)确认：class 26为建筑立面，class 27为桥墩和高架桥斜拉索；二者在ASPRS表中使用reserved编号。立面高`P_inv`与俯视影像不可见的物理预期高度一致，也构成geometry-only分区有效性的额外证据。旧年份第一采集周期只有1/2/3/6/9/17六类，跨年份统计必须保留这一taxonomy差异。

### 4.4 定性目视验证与表述边界

项目作者已对写回LAS的连续`q_geo`和三分区进行目视检查，判断其在开阔地、建筑、植被及稀有基础设施场景下可靠，并据此冻结0.20/0.60阈值。为避免只展示人工挑选的最佳样例，绘图脚本按五种环境中位`P_inv`比例自动选择代表tile：

![q_geo qualitative examples](results/section_43/report_figures/figure_43_q_geo_qualitative_examples.png)

点云定性图采用75°正交近俯视角，在保留平面布局可读性的同时，以屏幕空间深度不连续增强高度轮廓。渲染使用项目内置的确定性静态EDL，以保持颜色清澈、跨平台可复现；点径根据投影点密度自适应确定。

另给出固定100 m视野的PCV + 无光影EDL正式渲染版本，用于观察开阔地、建筑和密集植被的形态；连续自适应点比例固定为`0.86`，无逐点白色晕圈。它只改变图像呈现，不改变上述`q_geo`数值、分区或定量结论。

![PCV final rendering](results/section_43/report_figures/figure_43_pcv_final.png)

该结果构成可复现的**定性验证**，但仍没有逐点人工二值真值。因此论文可写“visually inspected and qualitatively validated”，不能报告precision/agreement，也不能称为visibility ground truth。详细记录见`q_geo_qualitative_validation.md`。

## 5. 4.3-P4：Visible-Surface Bottleneck的表征后果

### 5.1 影像patch与不可见点的语义关联冲突

![Projection conflict](results/section_43/report_figures/figure_43_projection_conflict.png)

为把“几何不可见”连接到“蒸馏目标可能不匹配”，对每个16×16像素DINO patch，以有效point-image correspondence中的LAS类别众数作为**事后可见表面类别代理**。仅保留至少5个有效对应点且主类纯度至少70%的patch；再利用所有点保存的`pixel_coord`重新确定`P_inv/P_amb/P_vis`落入哪个patch。训练用的`patch_index=-1`只意味着该点无有效视觉监督，绝不意味着它不能在影像中定位。这个重建在406块、包括非正方形影像在内的所有tile上逐一校验了原有效点的patch编号。

| 几何区域 | 可比较审计点 | 本区域审计点覆盖率 | 与patch可见表面主类冲突率 | 140-source-block 95% CI |
|---|---:|---:|---:|---:|
| `P_vis` | 595,923 | 85.27% | 17.48% | 15.76–19.17% |
| `P_amb` | 391,248 | 83.02% | 35.42% | 33.97–36.89% |
| `P_inv` | 418,566 | 84.93% | **41.84%** | 40.66–43.01% |

`P_inv-P_vis`冲突差为**24.35个百分点**，配对source-block bootstrap 95% CI为`22.74–25.98`点。合格点共1,405,737，占固定审计点的84.53%；被排除点主要因为patch中缺少稳定的可见表面锚点。类别细分揭示显著异质性：

| LAS类 | `P_vis`冲突率 | `P_inv`冲突率 | `P_inv-P_vis` |
|---|---:|---:|---:|
| Ground | 17.4% | **91.2%** | +73.8点 |
| Vegetation | 3.2% | **1.4%** | −1.8点 |
| Building roof | 1.9% | 11.9% | +10.0点 |
| Bridge deck | 3.1% | 34.0% | +30.9点 |
| Building facade | 99.2% | 99.9% | +0.7点 |

由此可见，林下地面投影到树冠patch、立面投影到屋面patch时，直接给点赋予patch语义尤其不合适；但同属vegetation的冠层与内部即使几何深度不同，粗LAS类别仍相同。因此此指标是**LAS类别代理的投影冲突率，不是DINO预测错误率**，不能声称41.84%的DINO target一定错误。类别粗糙度、影像配准、patch边界、可见对应点筛选都会影响数值；仅作为物理机制的独立辅助证据。

![Projection mismatch cases](results/section_43/report_figures/figure_43_projection_mismatch_cases.png)

图中两个高冲突实例分别显示林下ground被树冠遮蔽、建筑facade位于roof边界下方；左列为正射影像及点的投影，中列为按影像方向对齐的ALS俯视PCV + 无光影EDL渲染（连续自适应点比例0.86），右列为直接使用**原始LAS高程**的A-A'纵剖面。两列顶视图中的同一条黑边白色实线表示纵剖切线；它不是分类边界或阴影。林地剖面不再随局部地面曲线逐点校正，以免改变实际坡面和地物形态。红色圈出目标类别的`P_inv`点，蓝色圈出对应的`P_vis`点。图中的点到像素仿射拟合先对百万量级的Swiss XY坐标中心化，两案例残差中位数均小于0.5像素，避免旧图顶视剖切线方向不一致。两实例是机制展示而非随机抽样；总体频率和不确定性由上面的406-tile统计给出。独立`q_geo`没有使用return或classification；这里的LAS标签只在**训练后分析**时使用。

### 5.2 表征后果：VPSD可见/不可见读出差距

VPSD-128在全域区域分析中的linear结果：

| 区域 | OA | mIoU | 测试点 |
|---|---:|---:|---:|
| All | 0.7611 | 0.4694 | 100,000 |
| P_inv | 0.7422 | 0.4437 | 29,182 |
| P_amb | 0.7555 | 0.4369 | 27,726 |
| P_vis | **0.7774** | **0.5020** | 43,092 |

`P_inv`相对`P_vis`下降 `5.83` 个mIoU百分点和 `3.52` 个OA百分点，支持视觉蒸馏存在可见表面偏置。Cosine kNN中的对应mIoU为`0.3815/0.3988`，差距缩小至`1.73`点；两种读出方向一致，但幅度不同。正文宜以linear probe作为主结果，并把kNN作为补充证据，不能宣称两种协议下瓶颈强度相同。

典型错误集中在：

- suburban的`P_inv` ground→vegetation；
- dense vegetation中低高度unclassified→ground；
- rare-infrastructure的可见ground→bridge deck。

这些误差同时受类别定义、局部密度与源数据分类噪声影响，不能全部归因于visibility。

## 6. 4.3-P5：超越直接可见域的语义补偿

![Regional compensation](results/section_43/report_figures/figure_43_regional_compensation.png)

### 6.1 Frozen linear probe

| 表示 | Dim | All | P_inv | P_amb | P_vis |
|---|---:|---:|---:|---:|---:|
| VPSD-Scratch-128 | 128 | 0.1297 | 0.1232 | 0.1250 | 0.1398 |
| VPSD-128 | 128 | 0.4694 | 0.4437 | 0.4369 | 0.5020 |
| **Independent RRCL-72** | **72** | **0.2854** | **0.2893** | **0.2821** | **0.2892** |
| CVRL-visual-128 | 128 | 0.4577 | 0.4337 | 0.4320 | 0.4821 |
| CVRL-return-72（联合分支） | 72 | 0.3177 | 0.3121 | 0.3076 | 0.3371 |
| CVRL-full-200 | 200 | **0.5127** | **0.4895** | **0.4871** | **0.5354** |

### 6.2 Cosine kNN

| 表示 | All | P_inv | P_amb | P_vis |
|---|---:|---:|---:|---:|
| VPSD-Scratch-128 | 0.1070 | 0.0891 | 0.0917 | 0.1274 |
| VPSD-128 | 0.3729 | 0.3815 | 0.3372 | 0.3988 |
| **Independent RRCL-72** | **0.2709** | **0.2897** | **0.2387** | **0.2646** |
| CVRL-visual-128 | 0.3702 | 0.3354 | 0.3548 | 0.4158 |
| CVRL-return-72（联合分支） | 0.2787 | 0.2667 | 0.2527 | 0.2961 |
| CVRL-full-200 | **0.4003** | **0.3999** | **0.3640** | **0.4238** |

### 6.3 互补增益

CVRL-full相对CVRL-visual的linear mIoU增益：

- All：`+5.50`；
- `P_inv`：`+5.59`；
- `P_amb`：`+5.51`；
- `P_vis`：`+5.33` 个百分点。

CVRL-full相对独立VPSD-128：All `+4.33`、`P_inv +4.58`、`P_amb +5.02`、`P_vis +3.33`；相对同构VPSD-Scratch则分别高`38.30/36.63/36.21/39.56`点。若只隔离视觉蒸馏本身，VPSD-128相对VPSD-Scratch的对应增益为`33.97/32.05/31.19/36.23`点。两组结果共同说明预训练贡献远大于随机结构先验。

为避免把空间相关点当作独立样本，使用28个held-out原始source block进行了2,000次配对block bootstrap。下表为linear mIoU差值及95%区间，单位均为百分点：

| 配对差值 | All | P_inv | P_amb | P_vis |
|---|---:|---:|---:|---:|
| VPSD − Scratch | 33.97 [28.28, 38.02] | 32.05 [25.38, 37.53] | 31.19 [25.35, 37.13] | 36.23 [29.76, 39.84] |
| CVRL-full − CVRL-visual | 5.50 [3.42, 6.83] | 5.59 [3.52, 7.57] | 5.51 [3.86, 7.07] | 5.33 [1.93, 7.31] |
| CVRL-full − VPSD | 4.33 [2.54, 6.57] | 4.58 [2.11, 7.41] | 5.02 [1.74, 7.83] | 3.33 [1.35, 5.60] |
| CVRL-full − Independent RRCL | 22.73 [17.42, 27.65] | 20.03 [14.83, 25.44] | 20.50 [15.27, 25.77] | 24.62 [18.17, 30.17] |
| CVRL-return − Independent RRCL | 3.24 [1.61, 4.97] | 2.28 [0.82, 3.82] | 2.55 [0.87, 4.18] | 4.79 [2.53, 6.88] |

全部区间均高于0，说明上述linear优势不是由少数空间块偶然造成；但这不改变kNN读出下的差异边界。

kNN给出更明显的区域差别：CVRL-full相对CVRL-visual在`P_inv`提高 `6.45` 点，而在`P_amb/P_vis`分别提高 `0.92/0.80` 点；相对独立VPSD，在`P_inv`也高 `1.84` 点。按mIoU，CVRL-full在四个区域和两种读出下均高于独立VPSD；但该优势来自单一审计划分，且kNN样本较少，不能将其直接解释为跨区域泛化结论。

`P_inv`类别级别，full相对visual的F1增益主要为：ground `+17.22`、vegetation `+9.63`、bridge deck `+7.63`、wire `+5.15`、unclassified `+4.08`、building `+2.11`。water下降 `2.97` 点，说明互补并非对所有类别无条件有效。

### 6.4 独立RRCL的类别选择性与融合后的变化

![P_inv class complementarity](results/section_43/report_figures/figure_43_pinv_class_complementarity.png)

独立训练的RRCL总体mIoU低于VPSD，**但并非所有类别都低**。下表单独保留逐类F1，以便考察具体类别的精确率与召回率平衡；所有方法使用完全相同的`P_inv`测试点：

| `P_inv`类别 | 支持点 | VPSD F1 | Independent RRCL F1 | CVRL-full F1 | RRCL−VPSD（95% CI） | Full−VPSD（95% CI） |
|---|---:|---:|---:|---:|---:|---:|
| Ground | 9,224 | 0.744 | **0.831** | **0.885** | +8.65 [4.30, 13.68]点 | +14.08 [10.56, 18.57]点 |
| Vegetation | 10,144 | 0.844 | 0.880 | **0.928** | +3.61 [−0.04, 7.19]点 | +8.44 [6.55, 10.79]点 |
| Building roof | 4,666 | 0.753 | 0.477 | **0.771** | −27.65 [−32.23, −24.47]点 | +1.79 [−0.21, 4.01]点 |
| Bridge deck | 583 | 0.576 | 0.325 | **0.634** | −25.04 [−36.13, −13.83]点 | +5.80 [1.47, 11.26]点 |
| Building facade | 374 | 0.503 | 0.251 | **0.531** | −25.25 [−44.84, −7.79]点 | +2.78 [0.56, 7.97]点 |

区间以28个held-out source block、2,000次配对重采样得到。RRCL相对VPSD的`P_inv` ground优势稳健；vegetation的正点估计区间贴近并略跨0，不能单独宣称显著。RRCL在roof/bridge/facade明显不足，而CVRL-full在这些视觉类别没有出现同等级崩落，并在ground/vegetation获得额外收益。这比“RRCL普遍优于VPSD”或“增益仅发生在P_inv”更符合证据。

联合checkpoint的return子空间比独立RRCL在线性读出All/`P_inv`上高3.24/2.28个mIoU百分点（后者CI `[0.82, 3.82]`），但二者训练结构和联合梯度不同，不能仅凭该差值证明视觉知识已流入return分支。独立RRCL的`P_inv` kNN mIoU为0.2897，高于其`P_vis`的0.2646；这说明回波表征的局部邻域在不可见点上未必更差，但不同区域类别构成不同，不能直接解释为“无可见性退化”的因果结论。

空间错例图进一步选择了`P_inv`中“VPSD预测错误、CVRL-full改正”的代表tile：vegetation、building、bridge deck分别找到655、354、41个被修复测试点；自动选择tile中的修复点分别为45、28、15个，黑圈标出位置。

![Representative recovery cases](results/section_43/report_figures/figure_43_representative_error_recovery.png)

### 6.5 结论边界

- `CVRL-return72`是联合checkpoint中的return子空间，不是独立训练RRCL。它单独的absolute score较低并不否定互补性；关键是与visual拼接后full显著超过两个分支。
- 当前结果支持“有意义的互补子空间”，但尚未排除200D维度增加本身的作用。4.4 representation surgery仍需加入dimension-matched random concat和shuffled-return control。
- VPSD-Scratch全量特征已经完成，`P_inv` linear mIoU仅`0.1232`，显著低于训练后VPSD的`0.4437`；4.3的Scratch与独立RRCL缺口均已闭合。

## 7. 对4.3研究问题的最终判定

| 研究问题 | 当前判定 | 证据强度 |
|---|---|---|
| Foundation semantics是否进入ALS encoder？ | 是。VPSD/CVRL visual显著高于直接DINO point mapping | 强 |
| PCA-128是否保留主要能力？ | 对学生保留约96%，对teacher只保留约84% | 强 |
| Joint learning是否完全不损伤visual subspace？ | 否，linear mIoU约损失1.9-2.0点，但teacher cosine未下降 | 中-强 |
| Visible-Surface Bottleneck是否具有实际规模？ | 是，P_inv占19.73%，并集中于植被和2-15 m结构 | 强 |
| VPSD是否在P_inv更弱？ | Linear与kNN均支持，前者差距更大 | 中等 |
| Return是否主要补偿P_inv？ | linear增益近似均匀；kNN在P_inv增益最明显 | 中等，读出相关 |
| CVRL是否兼得视觉与结构？ | 在本审计划分中，linear与kNN的mIoU均在各区域高于独立VPSD | 中-强 |
| 是否已证明超越Scratch？ | 是；VPSD在P_inv高32.05个linear mIoU点 | 强 |
| 是否已完成独立RRCL对照？ | 是；406块72D特征、同点同split probe、分类别和block bootstrap均完成 | 强，但只在审计域内 |
| q_geo是否完成验证？ | 跨环境目视检查通过；无逐点人工精度 | 定性完成 |

## 8. 论文写作建议

正文可以使用以下谨慎表述：

> VPSD internalizes foundation-model semantics into a point-native representation: despite operating without image features at inference, its frozen linear separability exceeds both direct point-to-patch teacher assignment and an architecture-matched random initialization. A qualitatively validated geometry-only audit identifies approximately one fifth of ALS returns as highly inaccessible to overhead imagery. The fused CVRL representation improves linear mIoU over its visual branch by 5.59 points on the inaccessible subset and by 5.33–5.51 points elsewhere; cosine kNN shows a larger 6.45-point gain on the inaccessible subset versus 0.80–0.92 points elsewhere, supporting complementary but readout-dependent visual-return structure.

不宜在当前阶段写：

- “q_geo达到某个precision”——只有目视定性验证，尚无逐点人工真值；
- “joint learning完全不损伤VPSD”——存在约1.9-2.0点linear mIoU下降；
- “独立RRCL全面优于VPSD的P_inv”——它只在地面显著占优，在屋顶、桥面和立面显著较弱；
- “return收益只出现在P_inv”——linear结果显示各区域均有约5.3-5.6点增益；
- “CVRL-full的优势已证明跨区域泛化”——本节所有probe仍位于预训练footprint内；
- “Swiss-Audit证明跨区域泛化”——Audit来自预训练footprint。

## 9. 指标释义与解读边界

### 9.1 分类与表征指标

- **OA（Overall Accuracy）**：测试点中预测正确点的比例。它会被ground、vegetation等大类主导，不能单独反映稀有类别能力。
- **Precision / Recall / F1**：均按类别计算。Precision衡量“预测为该类的点有多少是真的”，Recall衡量“该类真值点有多少被找回”，F1是二者的调和平均。混淆矩阵以**行为真值、列为预测**。
- **mIoU（mean Intersection over Union）**：对每类计算`TP/(TP+FP+FN)`，再对测试中并集非零的类别等权平均，是本报告比较语义可分性的主指标。它不会让ground等大类按点数支配平均值，但稀有类别仍会带来较大方差。逐类F1仅用于类别选择性分析，不与mIoU混称。
- **Frozen linear probe**：冻结特征提取器，仅训练一个线性分类器。它回答“类别是否已经能被一个线性边界读出”，而不是网络完整微调后的上限。本报告使用balanced multinomial logistic regression，以减轻类别不平衡影响。
- **Cosine kNN**：不拟合参数，使用L2归一化特征的余弦邻近点投票。它反映特征局部邻域是否按语义聚集。其采样规模小于linear probe，绝对值和方差不可与linear结果直接等同；两者趋势一致时证据最强。

### 9.2 蒸馏、压缩与配对指标

- **Paired cosine**：同一点的student特征与matched teacher特征的余弦相似度，范围为`[-1,1]`，越高表示方向越一致。Mean给整体一致性，P10/Median/P90描述困难点、典型点和高一致性点。它不是语义分类准确率：cosine更高不保证类间间隔更好。
- **PCA保留率**：报告中的“保留约96%”是`128D mIoU / native mIoU`，属于任务能力保留率，不是PCA explained-variance ratio。两者回答的问题不同。
- **feature_dim**：表示最终读出的维度，例如VPSD-128、RRCL-72、CVRL-full-200。更高维可能本身带来容量收益，因此full与单分支比较还需dimension-matched random concat等控制。
- **百分点（pp）**：例如`0.4895-0.4337=0.0558`记为约`+5.59`个百分点（以未四舍五入的原值计算），不等于相对提升5.59%。若需相对提升，应再除以基线值。

### 9.3 q_geo与区域统计

- **q_geo**：只由XYZ、多偏移自适应splat Z-buffer得到的连续几何可见性分数，范围`[0,1]`；越高表示点越稳定地处于从上方可观察的局部表面。它不是概率；当前已通过跨环境目视定性检查，但没有逐点人工真值校准。
- **P_inv / P_amb / P_vis**：分别由`q_geo<=0.20`、`0.20<q_geo<0.60`、`q_geo>=0.60`得到的互斥集合。`P_amb`是阈值敏感带，核心结论应优先比较高置信`P_inv`与`P_vis`。
- **region ratio / point count**：区域点数除以对应场景、类别或高度层的总点数。它描述几何瓶颈规模，不代表该区域分类难度。
- **Height AGL**：点高程相对局部地面基准的高度，而非绝对海拔，用于区分地表、低矮物、冠层/建筑层。局部地面估计误差会影响bin边界附近点。
- **Mismatch统计**：只统计探针预测错误的`真值→预测`组合，用于定位错误模式；它同时受类别频率、源LAS标签噪声和probe能力影响，不能直接当作网络因果解释。
- **投影冲突率**：对具有有效像素坐标的审计点，在同一个DINO patch内用可见LAS点的多数类别构造二维投影锚点，统计该锚点类别与目标点LAS类别不一致的比例。仅使用至少5个可见锚点且多数类占比≥70%的patch；它是标签代理诊断，不等于DINO教师的预测错误率，也不能证明视觉监督实际向每个不可见点传播。
- **audit_rows / teacher_visible_rows**：前者是固定分层抽取的审计点数，后者是其中具有有效影像patch对应的点数。DINO与student的六级配对比较只使用后者；全域`P_inv/P_vis`分析使用全部audit rows。
- **配对block bootstrap 95% CI**：以原始Swiss source block为重采样单位，对同一批测试点上的两种表示成对计算mIoU差值；逐类补充表仍计算逐类F1差值。区间不跨0表示差值在空间块变化下较稳定；它反映审计集内的不确定性，不代表跨城市或跨数据集置信区间。

### 9.4 Scratch应如何定义

“Scratch”必须随研究问题匹配，不能用一个随机网络回答所有问题：

1. **4.3视觉知识内化主对照：VPSD-Scratch。** 使用与VPSD完全相同的encoder、投影器和128D输出，仅把预训练权重换成seed-42随机初始化。这样`VPSD pretrained - VPSD scratch`才隔离视觉蒸馏带来的增益。随机特征的frozen probe只是“未训练表征下限”，不是通常意义的监督从头训练结果。
2. **完整CVRL方法的架构控制：CVRL-Scratch。** 在下游任务上让同一个完整CVRL架构从随机初始化开始监督训练，才能排除200D维度和双分支架构本身的收益。这应放在下游主表，而不是替代4.3的VPSD-Scratch。
3. **Plain LitePT-Scratch**仍应作为参数更少的实用基线，但不能单独承担“同架构公平控制”。

本次已生成并完成严格加载与真实GPU前向：

| 临时权重 | 参数量 | 冒烟输出 | 状态 |
|---|---:|---:|---|
| `paper_cvrl/tmp/vpsd_scratch_swiss100_seed42.pth` | 14,403,852 | 4,096 × 128 | finite，strict load通过 |
| `paper_cvrl/tmp/cvrl_scratch_swiss100_seed42.pth` | 17,340,380 | 4,096 × 200 | finite，strict load通过 |

4.3的VPSD-Scratch frozen probe已复用完全相同的split、输入和审计点，结果有效地表示“未训练表征下限”。但它不能代替下游主表中的监督Scratch；后者仍须让CVRL或plain LitePT在相同标注集、训练预算和多随机种子下从头训练。

## 10. 结果文件

- 原始定量结果：`G:\data\swiss\joint_tiles_100\Swiss-Audit\analysis\section_43_report.json`
- q_geo全量统计：`G:\data\swiss\joint_tiles_100\Swiss-Audit\analysis\q_geo_summary.json`
- 完整指标CSV：`paper_cvrl/results/section_43/section_43_key_metrics.csv`
- Block bootstrap：`G:\data\swiss\joint_tiles_100\Swiss-Audit\analysis\section_43_block_bootstrap.json`及`paper_cvrl/results/section_43/section_43_block_bootstrap.csv`
- 独立RRCL增量审计：`paper_cvrl/add_rrcl_to_section43.py`；类别bootstrap：`paper_cvrl/results/section_43/section_43_class_bootstrap.csv`
- 投影冲突定量：`G:\data\swiss\joint_tiles_100\Swiss-Audit\analysis\section_43_projection_conflict.json`及`paper_cvrl/results/section_43/projection_conflict.csv`
- 审计特征（含VPSD-Scratch）：`G:\data\swiss\joint_tiles_100\Swiss-Audit\features`
- 固定probe逐点预测：`G:\data\swiss\joint_tiles_100\Swiss-Audit\analysis\predictions`
- q_geo可视化点云：`G:\data\swiss\joint_tiles_100\Swiss-Audit\q_geo_pointcloud`（classification 1/2/3对应P_inv/P_amb/P_vis；原类别保存在`original_classification`）
- 独立RRCL配置：`configs/rrcl/pretrain-rrcl-v1m1-litept-v1m4-swiss-100.py`
- Scratch初始化脚本：`paper_cvrl/init_scratch_cvrl_checkpoint.py`
- q_geo定性验证记录：`paper_cvrl/q_geo_qualitative_validation.md`
- 可重复导出脚本：`paper_cvrl/export_swiss_audit_features.py`
- 定量分析脚本：`paper_cvrl/analyze_cvrl_43.py`
- Block bootstrap脚本：`paper_cvrl/bootstrap_cvrl_43.py`
- 绘图脚本：`paper_cvrl/plot_cvrl_43_results.py`
- 投影冲突及错例绘图：`paper_cvrl/plot_projection_conflict.py`
- 正式点云渲染：`paper_cvrl/plot_pcv_final.py`（100 m、PCV + 无光影EDL、自适应点比例0.86）
- 论文正文专用两图：`figure_5_visibility_bottleneck.png`与`figure_6_projection_conflict_mechanism.png`；制图脚本`paper_cvrl/plot_section43_paper_figures.py`，图注`paper_cvrl/section43_main_text_figures.md`。新版Fig. 6的两幅定性案例取自200 m tile，并用`paper_cvrl/prepare_section43_200m_cases.py`生成0.18 m半径的纯几何可见性sidecar；上文旧版100 m案例图仍作为历史分析图保留，不能把两套参数的分区直接混算。独立图像只保存PNG。
- 4.3详尽报告的九幅补充图位于`paper_cvrl/results/section_43/report_figures`；论文正文专用的Fig. 5/6保留在`paper_cvrl/results/section_43`根目录，避免混用历史分析图和正式图。
- 所有独立定量图和点云图只保存PNG；本Markdown报告另行生成单份排版PDF，便于完整阅读、审阅和归档。
