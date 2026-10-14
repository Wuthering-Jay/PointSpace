# 4.5 Public Downstream Transfer：同主干基线实施计划

状态：2026-09-21 方法忠实度修订；本文件区分已经具备的组件与尚待实现的基线，不把设计稿或小样本 smoke test 当作实验结果。

## 1. 论文问题与比较边界

主表只比较 **Swiss-100 同一预训练数据、LitePT-v1m4 同一编码器/解码器规格、XYZ+intensity 同一输入、同一下游划分与评估协议** 下，不同预训练机制产生的表示。Swiss LAS 的 `classification` 仅用于预训练后的审计，不得进入任一 SSL loss。回波字段仅可供 RRCL/CVRL 作 target；DINO 与 correspondence 仅可供 2D–3D 方法作 target/matching。下游任务不得加载 Swiss 标签。

固定 LitePT 后，新增 Point-M2AE/MSC/Sonata/D-DITR/Concerto 实现应在表格里标作 **LitePT adaptation**；不可将其结果冒充原论文的原生架构、原生输入模态或官方 checkpoint 的复现。官方报告值可另列为 benchmark context，不能混入受控主表。Point-M2AE 原版以多尺度分组 token 和层级掩码重建为核心；D-DITR 是不需推理图像的视觉知识蒸馏分支，不能与需要图像输入的 DITR injection 混称。Concerto 原版同时包含 3D 自蒸馏与 2D 学习；使用离线 DINO patch 后应明确标注为 offline-teacher adaptation。

本轮采用三类判据：(1) **数学等价**：逐行分块 InfoNCE、学生 head checkpoint、仿射 `mean→project`，可直接采用但必须有稠密计算的数值/梯度对照；(2) **LAS/LitePT 必需适配**：RGB→intensity、ALS 空间尺度、正射影像离线 DINO、LitePT 网格层级，必须披露且不能称等价；(3) **非等价计算限制**：Sonata/Concerto 的有限匹配上限会改变 Sinkhorn 参与分布，不能归入前两类。8192上限完成高覆盖验收后，正式训练为控制成本采用4096（每步约11%覆盖）；仍须报告该限制。不得再加入四分区 MSC 损失、Point-M2AE coarse Chamfer 或删去 Sonata local/roll 目标。

| 方法 | Swiss 预训练输入 | 核心目标（LitePT 适配版） | 特征/权重读出 | 当前状态 |
| --- | --- | --- | --- | --- |
| Scratch | 无 | 随机初始化下游训练 | LitePT-v1m4，72D | DALES FT 配置已建立；LP/DP 不适用作预训练表示证据 |
| Point-M2AE-style (LitePT) | XYZ+I | 跨 stage 一致空间掩码、mask token 粗到细传播、细尺度 XYZ 集合重建 | 仅保留 LitePT 主干 | 层级 decoder 和单尺度 Chamfer 已按官方流程修订；FPS/KNN token 分组仍由 LitePT 网格层级替代 |
| MSC | XYZ+I | 两视图交叉掩码、view mixing、全局配对 InfoNCE、masked intensity 重建 | 仅保留 LitePT 主干 | 目标函数已按原 MSC 主配置修订并通过 CUDA 测试；RGB→intensity 是 ALS 数据模态替换，normal 为可选分支 |
| Sonata | XYZ+I | 两全局+四局部视图、mask/roll/unmask EMA 原型自蒸馏 | student LitePT 主干 | 三项目标和跨卡 Sinkhorn 已恢复；ALS 匹配点 cap 仍需报告 |
| D-DITR | XYZ+I；DINO 仅作 target | 有效对应点的 1024D cosine 回归 | 仅保留 3D LitePT 主干 | 真实 trainer 短测通过；尚未完成 20 epoch 正式训练 |
| Concerto | XYZ+I；DINO 仅作 teacher | 完整 Sonata 三目标 + offline patch 级 2D–3D 对齐 | 3D student LitePT 主干 | 四局部视图和 roll 分支已恢复；真实 LAS+DINO 多视图数据加载及合成前后向通过 |
| VPSD / RRCL / CVRL | 已实现 | 既有预训练目标 | 既有权重 | 已训练；需固定 checkpoint 与导出协议 |

### 容易使对比失效的事项

1. 共享主干不等于相同总参数量：预训练辅助 head 与 EMA teacher 可不同，但应报告主干参数量、训练总参数量与计算量。CVRL 的 200D `[128D visual | 72D return]` 与其他方法的 72D 不等宽；LP 需同时报告原生维度和预先规定的等维读出敏感性，不能事后选对 CVRL 有利的维度。
2. 预训练 budget 固定 Swiss-100 source tile 集合、20 epoch、相同有效优化步/批量与种子；多视图方法需披露视图数、每步点数和实测 GPU-hours，不能只称“同 epoch”就认为计算公平。五个基线统一使用 `batch_size_train=16, gradient_accumulation_steps=4`；训练器实际解析为每卡 micro-batch 4、累积 4 次、有效 batch 16。各方法可以有预先声明的稳定性超参，不以测试集调参。
3. 现有 `LasImageDataset`/`CVRLLasDataset` 会按 stem 加载 `dino_feature` 与 `correspondence`。3D-only 方法应使用 `LasDataset`，不读取 DINO；视觉基线按点索引保持 mapping 在 crop/GridSample 后同步，并检查无对应点、无效 patch、缺失 intensity 的边界。DINO 使用同一 1024D 离线 teacher；PCA128 可作额外消融，但不可混入主表。
4. 主表应区分“官方原理的 LitePT 适配”与“仅同类目标”：不能用普通点级 XYZ 回归替代 Point-M2AE 的多尺度遮挡，也不能用 VPSD 的 patch-set loss 冒充 D-DITR。
5. `pointspace/models` 下的 MSC、Sonata、Concerto 属于 Pointcept/PointSpace 原版接口。新实现须放入 `pointspace/models/backbone/<method>/`，使用独立注册名（建议 `<Method>-LitePT-v1m1`）。在完整替换、配置迁移和 checkpoint 兼容测试完成以前，不禁用旧注册器；完成后再移除冲突注册，避免破坏既有实验。
6. D-DITR 官方仓库为 AGPL-3.0；研究实现应记录来源与许可，优先依据论文和接口独立实现适配逻辑，不在未处理许可的情况下直接搬入其源文件。

## 2. 按方法的工程拆解

### Point-M2AE

- 使用 4 m 粗体素组抽样掩码，并从输入中**删除整组点**，所以 LitePT 的 stem/encoder/decoder 都只见可见点；2 m 细体素组嵌套在粗掩码中。现版让粗尺度可见 token 与可学习 mask token 共同经过 Transformer decoder，再将粗 mask query 传播到细尺度，融合 LitePT 的细尺度可见特征并经第二个 Transformer decoder。**只**对细尺度 masked neighborhood 预测 16 个局部 XYZ 并使用双向 Chamfer；此前额外的 coarse Chamfer 已删除，因为官方实现没有该目标。掩码比例与官方预训练配置一致为 0.8；每样本最多 512 个 coarse、1024 个 fine query，checkpoint 只减少 decoder 激活显存。训练后只导出未遮挡 LitePT 主干的 72D 点特征。
- 保留的实质差异是官方 FPS/KNN token pyramid 被 LitePT 网格层级及 ALS 尺度分组替代，以及 fine decoder 不能像官方小物体模型一样对所有 fine 可见 token 做无界注意力；当前采用局部 KNN 特征传播加 masked-query 注意力。因此论文表名仍应写 `Point-M2AE-style (LitePT)`，不能称官方模型的严格复现。正式训练前还需核验 query 覆盖率、每尺度点数和完整 batch 显存。
- 合成 CUDA 前后向、mask query 粗到细梯度和 feature export 已通过。此前真实 Swiss 单块的 loss/耗时/显存取自**增加 mask-query decoder 以前**，现版不得沿用；仍需正式 batch、多 worker 和完整训练稳定性测试。

### MSC

- `MSCLasPairDataset` 先对同一 source tile 做共同 crop，再独立增强/体素采样两视图；原坐标用于匹配，LAS 语义/回波字段被剔除。原版交叉互斥掩码、KNN 正对、0.8/0.0 view mixing 及全局负样本池 InfoNCE 均保留。删除此前并非原 MSC 的“上下/近远四分区”目标；现在按 query 行分块计算**完全相同**的 InfoNCE。
- Swiss LAS 实际无 RGB，但有 intensity。因此原版主配置的 masked RGB MSE 在本 ALS 适配中换成单通道 masked intensity MSE；与原公开主配置一致，默认关闭 normal 重建。代码保留可选 normal 分支，在启用时可对增强/采样后的点集使用 GPU KNN/PCA，且不作主干输入。匹配点 cap 保持原实现的8192上限；row chunk由256增至1024，只改变精确InfoNCE矩阵的分块执行，不改变完整负样本池或目标。RGB→I属于数据模态替换，绝不能称为数学等价优化。
- 先前 512 对、无重建、四分区的 loss/速度/显存数据已经失效，不得引用。新路径已通过合成 BF16 CUDA 前后向；正式批量吞吐、匹配覆盖率与收敛仍待验证。

### Sonata

- `Sonata-LitePT-v1m1` 现使用两全局+四局部视图；独立 mask/unmask prototype heads，4096 prototypes、4096 隐层/512 嵌入、固定 gain 的 weight-normalized prototypes，分别执行 masked global→own teacher、masked global→other teacher（roll）、unmasked local→principal global teacher 三项目标。权重 2/8、2/8、4/8；temperature、mask size/ratio 预热与 EMA 随**实际优化步**更新；Sinkhorn 支持跨 rank prototype/sample 归一化。student/teacher 是同规格 LitePT-v1m4，输出 P0 72D，故不需要原 PTv3 的 `pooling_inverse` up-cast。
- 仍有 LitePT stage/up-cast 层级不等价、ALS 空间尺度不同及有限匹配上限的差异。student head 的分块 checkpoint、`preserve_rng_state=False`（head内无随机算子）均不改变数学目标，并已有稠密loss/梯度一致性测试。512/2048/8192 cap敏感性与batch-4显存验证已完成；正式配置采用4096 cap和2048-row chunk，每步覆盖约11%。在候选集合近似稳定、各epoch独立采样的简化假设下，20 epoch累计被采样概率约为90%，但它仍不是无界匹配。旧的两视图512-prototype速度/loss数字已失效。合成与真实Swiss六视图BF16前后向及64-step trainer短测已通过，尚未有整轮收敛数据。

### D-DITR

- 核对原论文 §3.2 后确认：D-DITR 本身就是把 2D DINO patch 特征赋给对应 3D 点，以 3D 网络最终线性层逐点回归，并在有效对应点上平均 cosine loss；**不需要**预先训练 injection teacher。原文：[DINO in the Room §3.2](https://arxiv.org/html/2503.18944v2)。本实现 `D-DITR-LitePT-v1m1` 用现有 Swiss 离线 DINO-1024 与 point-to-patch mapping 复现这条目标路径，不使用 VPSD 的 patch aggregation，也不输入 DINO 给 3D 主干。
- 同一DINO patch常被多个点引用，因此实现先对紧凑patch表归一化一次，再按点gather；这与逐点重复执行teacher归一化的cosine严格对应。loss chunk由4096增至16384，并在显存验证后关闭该线性回归loss的activation checkpoint；64-step平均batch约降低14%，峰值显存为6.62 GiB。LitePT主干仍保持`checkpoint_mlp=True`，所有正式配置继续使用CacheCleaner。
- 适配差异是 DINOv3 teacher（原文主要为 DINOv2）、ALS 正射影像对应（原文为室内/车载影像）、LitePT-v1m4（原文 PTv3）、预先缓存的 teacher 以及为大场景采用的分块 checkpoint cosine 计算。训练后丢弃线性回归头，保留原生 P0 72D LitePT 权重。真实 Swiss tile 22.2k 点、3967 个有效 patch、22.2k 个有效对应点，BF16 前后向 1.9 s、峰值 0.59 GiB；仅为单样本 smoke test。

### Concerto

- `Concerto-LitePT-v1m1` 沿用恢复三项目标的 Sonata 式 EMA 原型自蒸馏，再把第一 masked 视图中同一图像 patch 的点特征求均值，以线性 predictor 对齐离线 DINO-1024 patch teacher，采用通道中心化 cosine loss。由于 predictor 为仿射层，`mean → project` 与原 Pointcept 代码的 `project → mean` 在数学上完全等价，节省 `[visible points,1024]` 中间张量。不可见点只参与 3D 分支；完全无图像对应的块恢复为完整权重的 3D 自蒸馏。2D teacher 永不进入点主干。此前两视图性能/显存数字不适用于现版。
- 三维分支现继承恢复后的 Sonata 三项目标、四局部视图及跨卡 Sinkhorn。离线 DINOv3 代替在线 DINOv2 图像编码，正射影像对应取代室内相机投影；这是明确的 teacher/数据域适配。论文表仍应标作 `Concerto (LitePT, offline DINO)`，并报告 3D intra、2D cross、有效 patch 数与对应率。旧两视图训练/显存数字失效，整轮收敛尚未验证。[原论文方法与附录](https://arxiv.org/html/2510.23607)。

## 3. 下游固定协议

DALES 为主表：LP（冻结整个预训练表示，只训练单层 `Linear(C,K)`）、DP（冻结 LitePT encoder，训练原生 LitePT decoder+seg head）、FT（完整解冻）。每个方法均从相同 Swiss checkpoint 读取主干，语义 head 随机初始化。Scratch 只作为 FT 随机初始化参考；“随机特征 LP”若列出需另注，不作为预训练基线。使用相同 DALES 官方划分、GridSample、标签映射、零填 intensity、评估器和种子；主指标 mIoU，附 mF1/OA 与各类 IoU。

已建立 [DALES 三模式独立配置](../configs/baselines/dales/README.md)：顶部 `pretrain_method` 可选择五种 SSL baseline，均指向其 `backbone_last.pth`。LP 新增 `LitePTLinearProbeSegmentor`，严格只训练 `Linear(72,8)` 且冻结的主干保持 eval；DP 只载入 stem/encoder 并使用任务 decoder；FT 载入完整 backbone，主干学习率为 head 的 0.1 倍。配置及参数冻结边界单元测试已通过；尚未有正式预训练权重，因此不能声称 DALES 结果已验证。

Vaihingen 为自然低标注量转移：优先 Scratch/Sonata/D-DITR/CVRL，LP/DP/FT，同一官方训练集，不把它写成 1% 采样。OpenGF 为任务转移：Scratch/Sonata/VPSD/RRCL/CVRL，LP/FT，二分类 ground IoU/F1/OA，按地形分层。三数据集不能跨指标求“平均总分”。至少三个种子的主要 DALES 结果报告 mean±std；若预算有限，先单种子筛选，再对最终主表补种子。

所有下游配置记录 checkpoint SHA256、模型读取/未读取的 key、冻结参数数量、实际训练参数数量、最优 epoch、数据 split 和真实耗时。LP/DP/FT 的优化器允许预先声明的协议级学习率网格，**同一协议内对所有方法一致**，不能只针对 CVRL 搜索。比较 checkpoint 选择不得使用最终 test 标签。

## 4. 实施顺序与验收

- [x] 核查现有 4.5 PDF 大纲、Swiss 数据接口、CVRL LitePT 参数与五组对照边界。
- [x] 建立 DALES LitePT-v1m4 Scratch FT 独立配置（不依赖 `_base_`）。
- [x] 建立 Swiss pretraining manifest：固定13,530个tile stem、全量文件头/尺寸与确定性128-tile内容哈希审计，确认DINO统一为1024D、return有效率99.9997%、有效correspondence比例45.63%；归档发布前可开启全文件内容哈希。
- [x] Point-M2AE LitePT 变体：一致空间掩码、两层 Transformer mask-query decoder、粗到细传播、**单尺度** masked XYZ 重建与未遮挡特征导出；合成前后向通过。
- [ ] Point-M2AE 的 FPS/KNN token pyramid 仍由 LitePT 网格层级替代；正式主表标为 `-style` 近似并报告差异。真实 trainer 短测已通过；仍需完整训练与 LP/DP/FT。
- [x] MSC：同 tile 两视图、view mixing、原版全局 InfoNCE、masked intensity 重建及 BF16 CUDA 前后向测试；可选 normal 分支也经测试。
- [ ] MSC 真实 trainer 短测与 8192 匹配上限已通过；仍需跨 epoch 收敛与下游 LP/DP/FT。可选 normal 分支启用前先审计 ALS 法向质量。
- [x] Sonata：LitePT student/teacher、两全局四局部视图、mask/roll/unmask 目标、EMA/accumulation 校验、72D 导出、跨 rank Sinkhorn 与合成 BF16 前后向；真实 Swiss 六视图 LAS 加载通过。
- [ ] Sonata/Concerto 真实 Swiss 六视图、正式 batch、512/2048/8192 敏感性和优化后64-step trainer短测已通过；正式配置采用4096 cap、2048-row chunk。继续完成分布式 Sinkhorn 一致性与20 epoch收敛。
- [x] D-DITR 论文 §3.2 loss 核对、离线 DINO 点级 adapter、同主干配置、合成与真实 Swiss 小样本前后向；确认原方法不需 injection-teacher 阶段。
- [ ] D-DITR 真实 trainer 短测已通过；继续完成 20 epoch 收敛与跨区域下游 LP/DP/FT。
- [x] Concerto offline teacher adapter、两全局四局部视图、3D/2D 双目标与 patch 聚合等价性测试、真实 LAS+DINO 六视图数据加载及合成前后向。
- [ ] Concerto 真实 trainer 短测已通过；继续完成 20 epoch 收敛、有效对应分布统计和跨区域下游 LP/DP/FT。
- [x] 五种适配方法的注册名与旧 Pointcept 模型隔离；建立统一的下游 LP/DP/FT 配置，验证骨干架构一致、权重键前缀和可训练参数边界。
- [x] 五种方法的真实 Swiss 数据加载、目标 batch 显存/吞吐与 64 micro-step trainer 稳定性已全部通过。
- [ ] 待正式 checkpoint 生成后完成加载/导出往返和 DALES LP/DP/FT 前后向 smoke test。
- [ ] DALES 主表；Vaihingen/OpenGF 子表；三种 seed、计算预算、异常/缺失实验说明。

## 来源与命名

- 章节要求：`paper_cvrl/CVRL_实验章大纲_V6_详细设计版.pdf`，4.5-P1–P5。
- Point-M2AE 官方仓库：https://github.com/ZrrSkywalker/Point-M2AE；论文：https://arxiv.org/abs/2205.14401。
- DITR / D-DITR 官方仓库：https://github.com/VisualComputingInstitute/DITR；论文：https://arxiv.org/abs/2503.18944。
- MSC/Sonata/Concerto：本仓库 `pointspace/models/` 的 Pointcept 移植版及对应官方预训练配置，后续逐一记录适配差异。
