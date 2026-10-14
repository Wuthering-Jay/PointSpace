# 4.4 Return Structure as a Physical Complement：实验纲要与执行计划

本计划对应《CVRL 实验章大纲 V6》4.4-P1至P4。4.4属于**机制与表征审计**：要检验回波监督的物理特异性、受益结构和与视觉子空间的互补，而不是要求独立RRCL在全域胜过VPSD。Swiss-Audit处于Swiss预训练覆盖范围内，不能把本节结果写成跨区域泛化；外部迁移留给4.5。

## 1. 从4.3出发的待证命题

固定Swiss-Audit、同source-block split、相同测试点上的已知结果（linear mIoU）：

| 表示 | 维度 | All | P_inv | P_vis |
|---|---:|---:|---:|---:|
| VPSD-128 | 128 | 0.4694 | 0.4437 | 0.5020 |
| 独立RRCL-72 | 72 | 0.2854 | 0.2893 | 0.2892 |
| CVRL视觉支路 zV | 128 | 0.4577 | 0.4337 | 0.4821 |
| CVRL完整表示 zV‖zR | 200 | 0.5127 | 0.4895 | 0.5354 |

CVRL-full相对本模型zV的linear mIoU增益在`P_inv/P_vis`为`+5.59/+5.33`个百分点，**几乎均匀**；cosine kNN为`+6.45/+0.80`点，更偏向`P_inv`。因此现有证据支持互补，但尚不足以声称“回波只/主要补偿不可见点”。4.4的主问题应收敛为：

1. 与**其他内在自监督目标**相比，RRCL是否对穿透/遮挡结构更有利？
2. 这种优势在类、相对高度和局部多层几何控制后是否仍存在？
3. `zV‖zR`的收益是否来自**正确的逐点对应关系**，而非额外72维或整体范数变化？
4. role/context目标可预测性是否只是局部高度、密度、强度的shortcut？

所有预期都应写成可证伪假设。如果P2的结构交互不成立，应报告“广义互补”，而非强行解释为定向穿透补偿。

## 2. 统一评估协议（在看结果前冻结）

- **数据**：Swiss 100×100 m、grid 0.25 m；审计沿用406 tile、140原始source block、固定点索引及已有28个held-out测试block。ALS标签只用于预训练后的读出与分层，绝不进入SSL输入。
- **输入**：所有P1模型使用同一`XYZ+I`、增强、GridSample、SphereCrop和有效点定义；return number/number of returns只作RRCL监督，不能进入任何模型输入。Swiss的无效return值保持`return_role=-1`并从role/context目标排除。
- **计算预算**：P1固定LitePT-v1m4编码器与原生解码器、25 epoch、相同训练tile和有效optimizer step数、batch/梯度积累/优化器/学习率计划；所有候选输出同一72D点表示。头参数量、被监督点比例、初始共享主干梯度范数和实际训练时间一并记录。不同pretext的原始loss值不可直接比大小。
- **公平监督域**：P1主表中几何目标也只在RRCL可用的有效return点集上计loss，以排除监督覆盖率差异；允许一个“几何目标全点监督”的补充敏感性结果，但不能混入主表。
- **读出**：沿用4.3的固定source-block split、100k/100k类别×区域分层的balanced linear probe与5k/5k cosine kNN；逐行L2归一化，超参数只在训练block确定。主指标mIoU，辅以OA、逐类IoU/F1；P1各方法使用完全相同的测试行。
- **不确定性**：在原始source block上配对bootstrap（建议2,000次、固定seed），报告差值与95% CI。预训练随机性另列seed均值/标准差，不以同一checkpoint的block bootstrap替代多seed。
- **分析口径**：区域、类别和高度层的划分由`q_geo`、XYZ及后验LAS类别得到，不由return target/预测结果定义。每个预设stratum公布点数、source-block数；样本不足者只展示描述性结果，不给稳定优势断言。

## 3. P1：受控内在自监督对照

**核心对照集**（均不使用影像teacher）：

| 组别 | 监督目标 | 作用 | 状态 |
|---|---|---|---|
| Scratch-LitePT | 无预训练 | 架构下限；不是额外pretext | 需同结构72D导出 |
| Normal | 当前采样点集的局部法向 | 表面方向对照 | 待实现 |
| Local shape descriptor | 局部PCA导出的linearity、planarity、scattering、curvature、anisotropy、verticality六维目标 | 坐标几何监督对照，不引入遮蔽机制 | 离线目标与训练配置已实现 |
| Relative height（扩展） | 相对局部地面高度或高度分箱 | 检验低级垂直几何shortcut | 可后置 |
| Role-only | S/F/M/L四类 | RRCL点级组成消融 | 待增加单任务开关 |
| Context-only | K=32、不含中心点的S/F/M/L比例 | RRCL邻域组成消融 | 待增加单任务开关 |
| Full RRCL | role+context | 物理监督主模型 | 已训练一seed |

Normal与Local shape descriptor共用离线几何计算：当前Swiss数据采用K=16，在0.25 m体素质心参考点集上对每个原始LAS点做XYZ欧氏KNN/PCA，将目标写成LAS extra dimensions。协方差特征值按λ1≥λ2≥λ3排序；六维目标分别为`(λ1−λ2)/λ1`、`(λ2−λ3)/λ1`、`λ3/λ1`、`λ3/sum(λ)`、`(λ1−λ3)/λ1`、`1−|n_z|`。Normal使用符号不敏感的`1−|cos|`，shape使用Smooth-L1。稀疏/退化邻域仅根据几何条件设有效掩码，不使用回波、LAS类或影像信息；训练时再与有效return掩码求交，使监督域接近RRCL。目标来自完整原始tile上的体素参考点，**不是**裁剪/随机删除/微扰后严格重新计算，因此需在论文披露这一离线近似；K=48与其他体素尺度的敏感性应单独输出，不能混入K=16目录。由于输入也包含XYZ，这一基线测试的是**几何自监督表示的迁移效果**，不能被解释为无坐标捷径的重建任务。

role/context拆分应保持相同LitePT主体、72D导出及训练步数；仅将另一loss梯度关掉，记录对应有效监督点数。为了避免“某个loss较大所以赢”，在训练集小样本上检查每个任务对共享主干的梯度范数；固定标定系数后不得依据审计测试表现逐一调优。

**主表**：模型、pretext目标/有效监督比例、参数量、25-epoch训练时间、All/`P_vis`/`P_inv` linear mIoU、对应kNN mIoU、相对Full RRCL的配对CI。正常法向可能在平坦表面更强；这并不否定RRCL，而应检验结构差异是否符合其监督物理。

## 4. P2：回波监督在哪些结构上帮助最大？

首轮已用现有独立RRCL、VPSD、CVRL zV/zR/full特征完成P2，详见[首轮实验报告](CVRL_4.4_P2_P3_首轮实验报告.md)；**无需等待P1新训练**。预注册两种互补对比：

1. 独立RRCL−几何pretext：检验return相对一般内在SSL的结构特异性；新baseline训练完成后补齐。
2. CVRL-full−同checkpoint的zV：检验return子空间的**条件增量**；再与独立VPSD、独立RRCL并排报告，但不将不同结构的绝对分数当成纯因果消融。

预设分层（先统计样本量，随后冻结阈值）：

- `P_inv/P_amb/P_vis`：沿用纯几何q_geo高置信分区。
- **Vegetation interior**：LAS vegetation且`P_inv`，按相对地面高度2–5、5–15、>15 m分层；冠层顶面`P_vis`作为同类对照。
- **Under-canopy ground**：后验Ground，`P_inv`，在同一局部XY邻域上方有足量高出≥3 m的点；`P_vis`且无上覆层的exposed ground为对照。上覆判定只用XYZ，不用return。
- **Multi-layer columns**：固定XY网格内存在多个明显分离的高度层；同密度/同高度范围的单层网格为对照。层数由XYZ占据分箱计算，不能按number of returns划分。
- **Building/vegetation transition**：在有限XY距离内同时有Building/Vegetation LAS点的空间带；其外的屋顶内部/植被单一覆盖区为对照。该层可能受标签边界噪声影响，应报告纯度/敏感性。
- **Exposed roof/ground**：`P_vis`且局部单层的屋顶、地面，作为“RRCL未必占优”的负对照。

全类混合层报告mIoU；**以真值类别定义的层**（如under-canopy ground）主要报告该类别Recall和该空间窗内逐类IoU，避免在仅含一种真值的点集上把“mIoU”误当常规分割指标。还应报告每层的类别组成、相对高度、密度及return有效率，并做同类/同高度/同密度标准化的增益比较，以减少Simpson悖论。

核心检验量为`[full−zV]`在penetration层与相应exposed对照层的**差中之差**，使用source-block配对CI。若CI跨0，则只能说return提供总体互补，不称其“定向补偿不可见域”。主要图建议为“结构层×表示”的mIoU/类Recall点估计与CI森林图，而非仅放彩色点云截图；一幅小型垂直剖面展示成功和失败各一例。

## 5. P3：Representation surgery（先于新训练完成）

首轮已在同一批审计行上比较：`zV(128)`、`zR(72)`、`zV‖zR(200)`、`zV‖random72`、`zV‖shuffle(zR)`，并附独立RRCL-72作为参照。所有拼接对照沿用4.3相同的整体逐行L2标准化和固定probe训练/测试行；另做“先对两支各自L2再拼接”的敏感性检查，防止范数比例成为隐藏因素。具体结果见[首轮实验报告](CVRL_4.4_P2_P3_首轮实验报告.md)。

- **random72**：每点独立、固定seed、72D，并匹配真实zR的尺度/协方差或逐维边际；至少5个随机seed。不得从标签、return或q_geo生成随机向量。
- **shuffle(zR)**：在同一个原始source block内、训练/测试分开置换zR行，保持边际与总体空间域，却打断与zV及点标签的逐点对应；不按真值类别置换。固定5个置换seed。可加更严格的“同q_geo×高度粗分箱内置换”敏感性版本，但不能用LAS标签限定分箱。
- **成功判据**：full在All及预设结构层显著高于两种200D对照；若shuffle仍保留大部分增益，说明场景层统计而非逐点配准可能是主要来源。结果应同时给linear mIoU与kNN mIoU、配对block CI及随机控制的seed分布。

注意CVRL的zV/zR共享浅层参数，P3只能证明**已学到的两子空间存在互补信息**，不能宣称两路训练梯度完全独立或互不影响。

## 6. P4：低级shortcut诊断

两条诊断不可混为一谈：

1. **目标可预测性**：训练轻量MLP，仅输入同一采样点集的局部高度统计、log密度、平面性/粗糙度、可用时强度，预测role及context分布；排除return number、number of returns、LAS类别、q_geo和视觉特征。与RRCL头报告相同有效点掩码下的role OA/macro-F1、四类F1、context MAE/KL/target entropy，并按P2结构层分解。它回答目标是否存在低级统计捷径。
2. **表示迁移价值**：MLP没有与RRCL可比的点云backbone，不能拿其目标准确率冒充下游transfer。若MLP目标预测很强，再训练一个同LitePT的“局部高度/密度代理目标”预训练模型，并在同一frozen probe上比较mIoU。只有这个模型才能作为shortcut-SSL迁移对照。

可能结果与解释：handcrafted MLP能预测部分role/context，并不自动推翻RRCL；若其proxy-SSL在结构层与RRCL等效，物理特异性主张须收缩。反之，即使RRCL预text准确率不是最高，只要固定读出下结构增益更稳健，仍是有效表征证据。

## 7. 执行顺序与停止门槛

| 阶段 | 工作 | 产物 | 进入下一步的门槛 |
|---|---|---|---|
| 0 | 冻结split、mIoU口径、目标有效掩码；统计P2各层点数与block数 | 审计manifest/预注册表 | 每个主层有足够独立block；稀疏层降级为探索性 |
| 1 | 用现有特征做P2和P3；实现random/shuffle及范数敏感性 | 结构分层表、surgery表/CI | 确认主效应与边界，不按测试结果改层阈值 |
| 2 | 实现P4轻量MLP、验证无target泄漏 | role/context分层诊断 | 若近乎完美预测，优先检查泄漏和统计捷径 |
| 3 | P1补训练normal、local shape descriptor、role-only、context-only；height可后置 | 受控pretext主表 | 单seed筛选后，对Full RRCL及最强几何对照各补≥3个训练seed |
| 4 | 仅在需要时训练proxy-SSL，汇总P1–P4并写4.4正文 | 主表、机制图、失败案例、局限 | 结论必须与预注册交互及CI一致 |

不要求RRCL在All分数绝对第一；更关键的是P1的公平性、P2的结构差中之差、P3超越增维/错配对照、P4不被简单shortcut完全解释。任何一环不成立，都应保留负结果并相应收窄“physical complement”措辞。

## 8. 当前资产与待开发项

**已具备**：Swiss 100 m的CVRL/VPSD/独立RRCL模型与406份对齐特征；`G:\data\swiss\joint_tiles_100\Swiss-Audit`中的审计行、q_geo、标签、原始source-block映射和线性预测；4.3固定probe与mIoU/paired block bootstrap代码。可立即开展阶段0–1。

**已完成首轮**：P2几何结构层标记、充足性检查、配对CI与初步密度敏感性；P3随机/置换五seed、整体/分支归一化；P1 normal/六维local shape离线LAS目标、GeometrySSL训练模型与配置、role/context单任务开关、统一的72D导出，均通过小点数GPU前后向测试。**待开发**：Swiss全量LAS几何预处理与P1正式训练、多seed；P2更严格的类别×密度匹配与多层几何敏感性；P4 handcrafted特征/MLP。不要直接把已有VPSD encoder-only、RRCL full-decoder、CVRL双分支的参数量和训练loss当成严格同构P1对照；P1必须在同一LitePT训练骨架下重新定义。
