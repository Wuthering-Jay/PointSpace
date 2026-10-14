# CVRL 4.3 Swiss-Audit 实验材料与执行清单

> 更新日期：2026-09-18  
> Audit 根目录：`G:\data\swiss\joint_tiles_100\Swiss-Audit`  
> 主 CVRL checkpoint：`exp/cvrl/swiss/pretrain-v1m1-litept-v1m4-100-full/model/model_last.pth`

## 1. 当前已经准备完成的材料

### 1.1 Swiss-Audit 数据

- [x] 对13,530块100 m tile扫描类别、点数、高程跨度和原始空间块。
- [x] 选取406块，即3.0007%。
- [x] 覆盖全部140个原始Swiss空间块。
- [x] 保留全部实际出现的LAS类别，并强化稀有类别样本。
- [x] 复制pointcloud、DINO-1024和correspondence三元组。
- [x] 使用指定的全局PCA模型生成406份DINO PCA-128，不在Audit上重新拟合PCA。
- [x] 保存选择manifest、全量tile统计、PCA SHA256和数据摘要。

环境组成：

| 启发式环境 | tile数 |
|---|---:|
| urban | 89 |
| suburban | 81 |
| dense_vegetation | 81 |
| open_ground | 61 |
| rare_infrastructure | 86 |
| water | 4 |
| mixed | 4 |

这里的环境名称由LAS类别比例确定，是用于分层抽样的**启发式标签**，不是人工场景真值。论文使用前需要从每类抽取若干tile进行人工核验。

Audit共95,965,298点，主要类别支持如下：

| LAS类 | 点数 | 通常含义 |
|---:|---:|---|
| 1 | 1,963,216 | unclassified |
| 2 | 38,750,587 | ground |
| 3 | 44,964,434 | low vegetation；Swiss供应方可能把更多植被统一放入该类 |
| 6 | 8,164,800 | building |
| 9 | 815,947 | water |
| 14 | 29,867 | wire/conductor |
| 15 | 25,603 | transmission tower |
| 17 | 938,951 | bridge deck |
| 26 | 222,612 | building facades（第二采集周期） |
| 27 | 89,281 | bridge piers and viaduct stay cables（第二采集周期） |

### 1.2 纯几何可见性

- [x] 实现`paper_cvrl/q_geo.py`。
- [x] `q_geo`只使用XYZ；不读取return、intensity或classification。
- [x] 实现互不重叠的`P_inv/P_amb/P_vis`三分区。
- [x] 实现局部地面估计和高度分层。
- [x] 对406块Audit数据完成计算。
- [x] 每块保存4096个按“类别×可见区域”分层的固定审计点索引。
- [x] 生成场景、类别、高度的分区统计。

当前工作阈值：

```text
P_inv: q_geo <= 0.20
P_amb: 0.20 < q_geo < 0.60
P_vis: q_geo >= 0.60
```

总体结果：

| 区域 | 点数 | 比例 |
|---|---:|---:|
| P_inv | 18,936,141 | 19.73% |
| P_amb | 12,983,023 | 13.53% |
| P_vis | 64,046,134 | 66.74% |

初步合理性信号：

- 开阔地`P_inv=8.97%`，密集植被`P_inv=25.32%`；
- ground为16.33%，vegetation为25.00%，building为10.21%；
- 高度分层中，0–0.5 m的`P_inv=12.49%`，2–5 m为25.95%；
- class 26的`P_inv=80.93%`，表明该供应方自定义类别需要优先核查。

这些统计说明0.20/0.60能形成大小足够且性质不同的三个集合，但阈值仍需通过人工可见性样本验证后才能冻结为论文阈值。

### 1.3 六级视觉特征链

正式比较链为：

```text
DINO-native-1024 ──固定全局PCA──> DINO-PCA-128
        │                              │
        ├── VPSD-native-1024 ──PCA──> VPSD-visual-128
        │
        └── CVRL-visual-native-1024 ──PCA──> CVRL-visual-128
```

统一约定：

- VPSD和CVRL都在原生1024D空间计算蒸馏损失；
- 两者默认/永久视觉输出为固定PCA后的128D；
- `visual_native`只用于审计，输出PCA前的1024D；
- 128D和1024D来自同一模型、同一projector，不分别训练模型；
- 分析阶段再次做row-L2，以消除固定scale对Linear/kNN的影响。

已完成：

- [x] 为VPSD加入与CVRL相同的固定PCA readout。
- [x] VPSD默认`visual`输出128D。
- [x] VPSD支持`visual_native`输出1024D。
- [x] CVRL支持`visual_native`输出1024D。
- [x] CVRL epoch audit hook同时保存`z_visual_native`和`z_visual`。
- [x] 特征tester支持仅保存固定audit point index，避免406块1024D全点特征占用数百GB。
- [x] DINO-1024与DINO-PCA-128已准备完成。

执行状态：

- [x] 训练Swiss-100 VPSD。
- [x] 导出VPSD-128。
- [x] 导出VPSD-1024。
- [x] 用现有CVRL权重导出CVRL-visual-128和CVRL-full-200。
- [x] 导出CVRL-visual-1024和CVRL-return-72。
- [x] 完成source-block frozen linear/kNN分析与图表。

## 2. 配置和代码入口

| 用途 | 文件 |
|---|---|
| Swiss-Audit选择、复制和教师PCA | `paper_cvrl/prepare_swiss_audit.py` |
| 纯几何q_geo核心 | `paper_cvrl/q_geo.py` |
| q_geo批处理及分层统计 | `paper_cvrl/swiss_audit_visibility.py` |
| 4.3完整Linear/kNN、分区与错例分析 | `paper_cvrl/analyze_cvrl_43.py` |
| 一条命令导出全部审计特征 | `paper_cvrl/export_swiss_audit_features.py` |
| 4.3论文图和CSV | `paper_cvrl/plot_cvrl_43_results.py` |
| 4.3详细结果解读 | `paper_cvrl/CVRL_4.3_实验分析报告.md` |
| Swiss VPSD训练/导出 | `configs/vpsd/pretrain-vpsd-v1m1-litept-v1m4-swiss-100.py` |
| 现有CVRL checkpoint审计导出 | `configs/cvrl/audit-cvrl-v1m1-litept-v1m4-swiss-100.py` |
| 共享原生投影/PCA readout | `pointspace/models/backbone/vpsd/projection.py` |

## 3. Audit目录结构

```text
Swiss-Audit/
├── pointcloud/                       # 406份LAS
├── dino_feature/                     # DINO native 1024D patch teacher
├── dino_feature_pca128/              # 固定全局PCA后的128D patch teacher
├── correspondence/                   # point -> pixel/patch
├── visibility/                       # q_geo、region、height、audit_index
├── features/
│   ├── vpsd_native1024/
│   ├── vpsd_pca128/
│   ├── vpsd_scratch_pca128/
│   ├── cvrl_visual_native1024/
│   ├── cvrl_visual_pca128/
│   ├── rrcl/
│   └── cvrl_full/
├── analysis/
│   ├── q_geo_summary.json
│   ├── predictions/                  # 公平probe的固定逐点预测
│   └── section_43_report.json        # 特征导出后生成
├── q_geo_pointcloud/                 # classification=1/2/3可视化点云
├── metadata/
│   └── dino_pca_model.safetensors
├── audit_manifest.csv
├── audit_summary.json
└── all_tile_statistics.json
```

由于G盘不支持Windows硬链接，三元组最终采用物理复制；原训练数据未移动、未修改。

## 4. 下一步执行顺序

### Step 1：人工冻结q_geo阈值

- [x] 将406块`q_geo`与三分区写回LAZ并完成人工目视检查。
- [x] 自动选择open ground、urban、suburban、dense vegetation、rare infrastructure的中位样例生成定性图，避免只选最佳案例。
- [x] 项目作者确认当前划分可靠；冻结0.20/0.60阈值，不再依据下游结果调整。
- [x] 明确记录：无逐点人工标签，因此只声明定性可靠，不报告precision/recall/agreement。

### Step 2：训练VPSD控制模型

配置：`configs/vpsd/pretrain-vpsd-v1m1-litept-v1m4-swiss-100.py`

与CVRL Swiss-100一致的关键条件包括：

- LitePT-v1m4 encoder；
- XYZ+intensity，grid size=0.25 m；
- P2/P3/P4层级融合；
- native DINO-1024 cosine监督；
- 相同数据增强、220k SphereCrop、batch=16、accumulation=4；
- AdamW、OneCycleLR、25 epoch和相同学习率分组。

VPSD是encoder-only，因此没有RRCL decoder与PCGrad；这是任务定义差异，不应人为补入。

### Step 3：导出六级特征

1. VPSD配置设置`feature_source="visual"`，输出128D；
2. VPSD配置设置`feature_source="visual_native"`，输出1024D；
3. CVRL Audit配置设置`feature_source="visual"`，输出128D；
4. CVRL Audit配置设置`feature_source="visual_native"`，输出1024D。

所有student特征只保存每块固定4096点，并带`point_index`，确保六级特征严格对齐。

### Step 4：内部化分析

运行`paper_cvrl/analyze_cvrl_43.py`，在具有有效影像对应的相同点集上比较：

- frozen Logistic Linear Probe；
- cosine distance-weighted kNN；
- overall/per-class指标；
- native student-teacher paired cosine；
- source-block级别train/test split，禁止点级随机泄漏。

### Step 5：可见性瓶颈

- [x] 对VPSD-128分别报告all/P_vis/P_amb/P_inv。
- [x] 按class/environment/height拆分VPSD错误。
- [x] 报告P_vis到P_inv的绝对下降和相对下降。
- [x] 将q_geo/P_inv/P_amb/P_vis写入406块可视化LAZ；原类别由`original_classification`保留。
- [x] 自动选择vegetation、building、bridge deck中VPSD错误而CVRL-full修复最多的tile并生成空间图。

### Step 6：区域补偿

- [x] 从现有CVRL checkpoint额外导出`return`和`full`特征。
- [x] 比较VPSD-128、CVRL-visual-128、CVRL-return-72、CVRL-full-200。
- [x] 分别报告all/P_vis/P_amb/P_inv。
- [x] 检查`full - visual`提升是否主要出现在P_inv。
- [x] 建立同条件独立RRCL-only配置。
- [x] 固定并验证同构VPSD/CVRL random-init Scratch权重。
- [x] 训练独立RRCL并导出406块Swiss-Audit特征；与联合return子空间分开报告，并使用同一审计点、相同probe划分。
- [x] 导出全部406块VPSD-Scratch审计特征并运行相同区域probe。
- [x] 所有表示改为严格复用相同train/test采样行，旧版跨表示随机子样本结果作废。
- [x] 以28个held-out source block完成2,000次配对bootstrap，报告Scratch、visual与full差值的95%区间。
- [x] 增加独立RRCL、联合return、CVRL-full之间的区域probe与配对bootstrap。
- [x] 进行`P_inv`分类别VPSD/RRCL/CVRL对照，报告地面、植被、屋顶、桥面、立面的F1及block bootstrap区间。
- [x] 量化点投影到二维patch时的LAS类别代理冲突，并绘制林下地面与建筑立面代表性纵剖错例；明确这不是DINO预测错误率。
- [x] 统一可见性图三面板横向朝向，使用互斥、同层级场景名称并放大`P_inv`图例。
- [x] 选定100 m + PCV + 无光影EDL，正式渲染比例0.86，清理旧的参数对比图。

## 5. 仍需注意的论文边界

1. Swiss-Audit来自已参与预训练的数据，只能证明训练分布内的内部化和机制行为，不能证明未见Swiss区域泛化。
2. Audit内部的probe split按原始空间块划分，能够避免Audit内部相邻100 m tile的点级泄漏，但不能消除它曾参与预训练这一事实。
3. 2024起第二采集周期新增class 26建筑立面、class 27桥墩/高架桥斜拉索；与第一周期六类taxonomy混合分析时应按采集年份分层解释。
4. `q_geo`阈值已经过跨环境目视定性验证并冻结，但没有逐点人工真值，不能报告人工precision/recall。
5. DINO teacher只在image-valid点上存在；六级内部化比较必须使用共同可见点。VPSD/CVRL的P_inv分析则使用student全点特征，不能为不可见点伪造teacher特征。

## 6. 已完成测试

- [x] Python语法编译通过。
- [x] Swiss VPSD和CVRL Audit配置均能成功构建模型。
- [x] 现有CVRL checkpoint严格加载：0 missing、0 unexpected keys。
- [x] 固定PCA输出维度和范数测试通过。
- [x] 合成上下表面中，`q_geo`正确区分顶层与被遮挡点。
- [x] 0.20/0.60三分区边界测试通过。
- [x] 4.3 frozen Linear/kNN及区域指标合成测试通过。
- [x] 全部现有测试合计8项pytest通过。
- [x] 用真实CVRL checkpoint和真实Audit tile完成GPU冒烟测试：固定4096点分别成功导出128D与1024D，点索引完全一致。
- [x] 128D直接导出与“聚合后的1024D再经固定PCA”平均cosine为0.999997，差异仅来自多fragment平均与FP16量化顺序。
- [x] q_geo写回406块LAZ，共95,965,298点；抽查11块点数、原类别备份和三分区编码一致。
- [x] 独立RRCL配置成功构建15,366,212参数模型，并读取真实Swiss样本（88,240个grid/crop后点）。
- [x] VPSD-Scratch与CVRL-Scratch均0 missing/0 unexpected严格加载，真实tile分别导出4,096×128与4,096×200有限值特征。
- [x] VPSD-Scratch全部406份特征完整导出，每份4,096×128且索引对齐、数值有限。
- [x] 分析协议升级为v2：同链表示严格共用probe行，并保存逐点预测供空间错例复现。
- [x] 生成跨环境q_geo定性图与vegetation/building/bridge deck代表性修复图。
- [x] 三组关键linear差值的source-block bootstrap 95%区间均不跨0。
- [x] 独立RRCL的406份特征与checkpoint时间和SHA256验证通过，维度72且与固定审计点对齐。
- [x] 投影patch索引在全部406块中与有效correspondence匹配，并兼容非正方形影像网格。
