# CVRL 实验资产与缺口审计（对应实验章大纲 V6）

> 审计日期：2026-09-17  
> 对照文档：`CVRL_实验章大纲_V6_详细设计版.pdf`  
> 审计范围：当前仓库代码、配置、`exp/` 中可复核日志与 checkpoint，以及本机可访问的数据目录。  
> 状态约定：**已有结果**＝已有可引用的定量结果；**能力具备**＝代码/配置已存在但尚无足够结果；**部分完成**＝只有部分证据链；**缺失**＝尚不能支持论文结论；**不宜比较**＝已有结果存在关键混杂因素。

## 1. 总体结论

当前工程已经具备完整的 CVRL 训练、特征导出和 DALES 三种迁移协议，且已有一个重要的初步结果：在同一 DALES Linear Probe 设置下，融合特征优于两个单分支特征，说明视觉与回波表征确有互补性。但按 V6 大纲要求，论文实验的核心证据链仍未闭合。

目前最准确的成熟度判断是：

- **方法与工程实现：较成熟。** CVRL-v1m1、累积式 PCGrad、VPSD/RRCL 分支、128D/72D/200D 输出、Linear/Decoder/Full Fine-Tuning 均已实现。
- **内部机制证据：仅有开端。** 有分支 Linear Probe，但没有正式 Swiss-Audit、几何可见性分组、四级视觉蒸馏上限审计和不可见区域增益分析。
- **公开数据迁移证据：DALES 单数据集、单种子、部分完成。** OpenGF、Vaihingen、低标注比例、跨任务与多种子结果尚缺。
- **公平基线：不足。** 现有 LitePT scratch 与 CVRL 微调在输入属性、模型结构和训练配置上不完全一致，不能直接用于正式增益结论。
- **预训练尺度对比：当前不可作因果解释。** 两个 Swiss checkpoint 同时改变了 tile 尺度、grid size、训练轮数、batch size、损失权重和 PCA 模型，不能把差异归因于单一因素。

因此，论文现阶段可以谨慎陈述：

1. CVRL 能在 Swiss 无标签数据上稳定完成联合预训练；
2. 在一次 DALES、seed=42 的严格线性头实验中，`zFull` 明显优于 `zV` 和 `zR`；
3. 预训练模型可成功用于 Linear Probe、Decoder Probe 和 Full Fine-Tuning。

现阶段**还不能**正式声称：

- CVRL 优于同架构从头训练或现有自监督方法；
- VPSD 的主要瓶颈来自不可见区域，RRCL 在该区域提供了补偿；
- CVRL 具有跨数据集、跨任务和低标注优势；
- 100 m/0.25 m 优于或劣于 200 m/0.5 m；
- 方法具有 backbone 无关性或达到 SOTA。

## 2. 两个 Swiss 预训练模型审计

### 2.1 配置与数据清单

| 项目 | 100 m / 0.25 m | 200 m / 0.5 m |
|---|---:|---:|
| 实验目录 | `exp/cvrl/swiss/pretrain-v1m1-litept-v1m4-100-full` | `exp/cvrl/swiss/pretrain-v1m1-litept-v1m4-full` |
| 配置 | `pretrain-cvrl-v1m1-litept-v1m4-swiss-100.py` | `pretrain-cvrl-v1m1-litept-v1m4-swiss.py` |
| tile 尺度 | 100 × 100 m | 200 × 200 m |
| grid size | 0.25 m | 0.5 m |
| pointcloud 文件数 | 13,530 | 3,418 |
| DINO feature 文件数 | 13,530 | 3,417 |
| correspondence 文件数 | 13,530 | 3,417 |
| 配对完整性 | 完整 | 有 1 个无 DINO/correspondence 的孤立点云 |
| epoch | 25 | 50 |
| train batch size | 16 | 20 |
| gradient accumulation | 4 | 4 |
| `lambda_vpsd` | 1.0 | 1.0 |
| `lambda_return` / `lambda_ctx` | 1.0 / 1.0 | 1.5 / 1.5 |
| PCA 模型 | 在 100 m 数据目录独立拟合 | 在 200 m 数据目录独立拟合 |
| 预训练验证 | `evaluate=False` | `evaluate=False` |
| 可用 checkpoint | `model_last.pth` | `model_last.pth` |
| DALES 已完成迁移 | 是 | 否 |

200 m 数据中缺少配对文件的样本为：

```text
swisssurface3d_2020_2714-1095_2056_5728_0000
```

该样本在正式复现实验前应写入数据清单并明确剔除，不能依赖数据集加载阶段静默跳过。

### 2.2 最终训练日志

| 模型 | total | VPSD | role | context | 日志中的 return 权重 |
|---|---:|---:|---:|---:|---:|
| 100 m / 0.25 m | 1.2918 | 0.3019 | 0.4240 | 0.5659 | 1.0 |
| 200 m / 0.5 m | 2.4277 | 0.3224 | 0.4577 | 0.6305 | 1.5 |

这里的 total loss **不能横向比较**：第二个实验对 role/context 使用了 1.5 倍权重，而且训练轮数、样本组织、PCA 和 batch size 也不同。即便三个未加权分损失数值接近，也只能说明两个训练任务都已收敛到相近量级，不能据此证明某种 tile 或 grid 更优。

### 2.3 主模型建议

建议论文主实验暂定 **100 m / 0.25 m** checkpoint，理由不是“它的 loss 更低”，而是：

- 13,530 个样本全部具有 pointcloud/DINO/correspondence 完整配对；
- 当前 DALES 的五组迁移结果全部基于该 checkpoint；
- 0.25 m 与当前 DALES 下游配置一致，减少迁移时的分辨率变化；
- 已形成连续、可追溯的配置—checkpoint—下游结果链。

200 m / 0.5 m checkpoint 可作为**第二预训练工作点或尺度鲁棒性补充实验**。若不补做受控实验，论文中只能写成“两种训练配置均可运行”，不能写成 tile/grid 消融。

若要正式研究尺度，至少应保证相同原始空间、相同数据划分、相同损失权重、相同训练更新次数、相同 PCA 变换与相同下游协议，并分别单独改变 tile size 或 grid size。当前两个 checkpoint 不满足这一条件。

## 3. 当前已经具备的实验资产

### 3.1 模型与训练能力

| 资产 | 状态 | 说明 |
|---|---|---|
| CVRL-v1m1 联合模型 | 能力具备 | 视觉 128D、回波 72D、融合 200D；完整 checkpoint 可直接迁移 |
| LitePT-v1m4 CVRL backbone | 能力具备 | 当前 Swiss 主模型使用的 backbone |
| PTv3m4 CVRL backbone | 能力具备 | 有实现和 Hubei 配置，但没有与当前 Swiss/LitePT 主实验匹配的正式结果 |
| 累积式 stage-wise symmetric PCGrad | 能力具备 | 当前正式 CVRL-v1m1 使用；10% return warm-up 后启用 |
| VPSD/RRCL 分支切换 | 能力具备 | 支持 visual、return、full 输出与训练模式 |
| RRCL evaluator | 能力具备 | role 分类和 context 重建指标已有实现 |
| 审计特征保存 | 能力具备 | hook 能保存 teacher native/PCA、`zV`、`zR`、`zFull` 及标签/样本标识 |
| 四级特征审计脚本 | 能力具备 | `utils/analyze_cvrl_audit.py` 已实现固定 tile split、L2 normalize、Logistic Linear Probe 和 cosine kNN |
| 几何/回波可见性计算 | 部分完成 | 内核能返回 geometry、return 和融合 observability；现有落盘和筛选主要使用融合值 |
| DINO 提取和全局 PCA | 能力具备 | 支持共享 PCA、采样拟合、模型单独保存和 safetensors 输出 |

关键实现入口：

- `pointspace/models/backbone/cvrl/cvrl_v1m1.py`
- `pointspace/models/backbone/cvrl/common.py`
- `pointspace/models/backbone/litept_v1/litept_v1m4_cvrl.py`
- `pointspace/models/backbone/rrcl/`
- `pointspace/engines/pcgrad.py`
- `pointspace/engines/hooks/cvrl.py`
- `utils/analyze_cvrl_audit.py`
- `pointspace/utils/surface_visibility.py`
- `utils/compute_las_visibility.py`
- `utils/dino/`

### 3.2 DALES 三种迁移协议

当前已有三类独立配置：

- Linear Probe：冻结完整 CVRL backbone，只训练严格的 `Linear(C, K)`；支持 `zV=128D`、`zR=72D`、`zFull=200D`。
- Decoder Probe：冻结 CVRL encoder，训练 LitePT 原生 decoder 和语义头。
- Full Fine-Tuning：完整 CVRL backbone 与语义头共同训练。

配置位于 `configs/cvrl/dales/`，且均加载 100 m / 0.25 m checkpoint。

### 3.3 已有 DALES 定量结果

以下结果来自同一预训练 checkpoint，但均只有 seed=42：

| 协议/特征 | mIoU | mAcc | OA | mF1 | 当前可支持的结论 |
|---|---:|---:|---:|---:|---|
| Linear Probe — `zV` | 0.5628 | 0.7327 | 0.9304 | 0.6653 | 视觉分支具有明确语义可分性 |
| Linear Probe — `zR` | 0.5036 | 0.6850 | 0.9269 | 0.6062 | 回波分支单独可用于语义迁移，但弱于视觉分支 |
| Linear Probe — `zFull` | **0.6160** | **0.7715** | **0.9518** | **0.7130** | 融合比 `zV` 高 5.32 mIoU 点，比 `zR` 高 11.24 mIoU 点；构成初步互补证据 |
| Decoder Probe | 0.7570 | 0.8388 | 0.9717 | 0.8341 | 冻结 encoder 后，轻量 decoder 可有效恢复空间细节 |
| Full Fine-Tuning | 0.7937 | 0.8730 | 0.9742 | 0.8698 | 预训练模型可稳定完成端到端迁移 |

这是当前最强的一组论文资产，尤其是 `zFull > zV > zR`。不过它仍缺少多种子方差、随机拼接/打乱回波对照，以及同架构 scratch probe。

### 3.4 现有监督结果只能作为参考，不能直接作为公平基线

仓库内旧 DALES LitePT scratch 的 mIoU 为 0.7926–0.7963，CVRL Full Fine-Tuning 为 0.7937。表面上看两者基本持平，但该比较存在明显混杂：

- 旧 scratch 使用 `XYZ + echo` 五通道，而 CVRL 使用 `XYZ + zero-filled intensity` 四通道；
- 旧 scratch 是普通 `LitePT-v1m1`，CVRL FT 是带 visual/return 分支和 200D 输出的完整 CVRL 架构；
- batch、AMP dtype、梯度累积、冻结/学习率分组等配置不同；
- 均只有一个固定 seed 的结果可直接对应；
- CVRL FT 的训练目标和参数量也不同于普通 LitePT。

因此，现有结果既不能证明 CVRL 优于 scratch，也不能证明预训练无效。正式主表必须补充**相同下游输入、相同网络结构、相同增强、相同优化器和相同种子集合**的从头训练对照。

## 4. 按 V6 大纲逐项检查

### 4.1 数据集与评估协议

| V6 要求 | 当前状态 | 已有内容 | 缺失/问题 |
|---|---|---|---|
| Swiss-Paired 预训练数据 | 部分完成 | 两种尺度数据和完整训练链；100 m 数据配对完整 | 缺正式统计：原始场景数、面积、点数、密度、可见点比例、DINO patch 数、城乡/森林等组成 |
| Swiss-Audit | 缺失 | 审计 hook 和分析脚本已具备 | 没有空间独立的固定审计划分、审计特征文件、场景类型表和正式指标结果 |
| DALES | 部分完成 | 2,904 train tiles、1,100 test tiles；已有 LP/DP/FT | 缺公平 scratch、自监督基线、多种子、低标注比例 |
| Vaihingen | 原始数据可用 | 本机目录存在 PTS/LAS、影像和参考数据 | 无 CVRL 预处理、配置、数据划分、协议和结果 |
| OpenGF | 部分完成 | 5,760 train tiles、1,188 test tiles；已有普通 LitePT baseline | 无 CVRL 迁移配置与结果 |
| LASDU | 可选资产 | 仓库有配置和部分旧结果 | 尚未决定是否进入正文/附录，无 CVRL 正式结果 |
| Hubei | 仅适合讨论 | 有大量历史开发实验和特征导出 | 架构版本与当前正式模型不一致，不应混入主定量表 |

### 4.2 预训练与统一评估设置

| V6 要求 | 当前状态 | 说明 |
|---|---|---|
| 正式预训练配置 | 部分完成 | 两个配置均完整列出参数，但尚未冻结论文唯一主配置及配置哈希 |
| checkpoint 选择规则 | 缺失 | 两个预训练均 `evaluate=False`，只有 `model_last.pth`，不存在基于独立验证集的 best checkpoint |
| 固定随机种子 | 部分完成 | 当前 seed=42；主实验仍需至少 3 seeds |
| 统一输入模态 | 缺失 | CVRL 与旧 scratch 的 echo/intensity 输入不一致 |
| 统一训练预算 | 缺失 | 25/50 epoch、batch、loss 权重和 PCA 均不同 |
| 参数量/显存/速度 | 缺失结果 | 工程可测，但尚无正式统一硬件测量表 |
| 数据划分防泄漏 | 未证明 | Swiss tile 可能来自相邻或重叠原始块；Audit 必须按原始航带/大块空间隔离，不能随机按 tile 划分 |

### 4.3 内部化、可见性瓶颈与区域补偿

| 实验 | 当前状态 | 缺失内容 |
|---|---|---|
| DINO-1024 → PCA-128 → VPSD-128 → CVRL-visual-128 四级审计 | 能力具备、无结果 | 当前实验目录没有固定 Swiss-Audit safetensors/CSV/JSON；还缺 visual-only 对照 checkpoint 或严格等价的 visual 分支抽取协议 |
| 纯几何 `q_geo` | 部分完成 | 内核可算 geometric observability，但现有 classification/observability 主要是几何与回波融合值 |
| `P_vis / P_amb / P_inv` | 缺失 | 需在 Swiss-Audit 上标定双阈值，并报告各分组点数、场景和类别分布 |
| `P_inv` 场景/类别/高度分析 | 缺失 | 需按建筑、桥梁、道路、植被及高度层统计 |
| VPSD 在可见/不可见区探测 | 缺失 | 需固定同一 probe，并报告 all/visible/ambiguous/invisible |
| Scratch/VPSD/RRCL/CVRL 分区对比 | 缺失 | 当前 DALES 只有全点集结果，没有纯几何可见性分区 |

特别需要强调：V6 的“VPSD 可见性瓶颈”验证必须使用**不含 return number/number of returns 的纯几何 `q_geo`**。当前融合 observability 包含回波先验，若用它再证明 RRCL 对不可见区域有效，会形成定义泄漏和循环论证。

### 4.4 回波互补性与融合有效性

| 实验 | 当前状态 | 缺失内容 |
|---|---|---|
| masked geometry/normal/role/context/RRCL 控制实验 | 缺失 | 有 RRCL 任务代码，但无统一 Swiss 受控结果 |
| 回波收益出现在哪里 | 缺失 | 需在 `P_inv`、植被内部、边缘、低矮地物等区域定量 |
| `zV / zR / zFull` Linear Probe | **已有单种子结果** | 缺 3 seeds、per-class 表、置信区间 |
| random concat | 缺失 | 需用同维随机特征排除“维度更高”带来的收益 |
| shuffled `zR` | 缺失 | 需打乱点级或 tile 内回波特征，排除仅由统计分布带来的收益 |
| shortcut MLP | 缺失 | 建议用高度、局部密度、强度/零强度、局部几何统计预测 return role/context，排除简单捷径 |
| PCGrad 价值 | 部分完成 | 有历史实验和当前实现，但缺在正式数据、相同预算下的 naive sum vs protected+PCGrad 对照 |

### 4.5 公开数据集迁移

| 目标 | 当前状态 | 判断 |
|---|---|---|
| DALES LP/DP/FT | 部分完成 | 三种协议均跑通，只有一个 seed，缺公平 baselines |
| DALES 自监督方法对比 | 缺失 | Sonata、Point-MAE、D-DITR、VPSD、RRCL 等尚无统一输入/骨干/预算结果 |
| Vaihingen 自然低数据 | 缺失 | 原始数据已获得，但从预处理到结果均未建立 |
| OpenGF 跨任务迁移 | 缺失 | 普通 baseline 可用，CVRL 未运行 |
| 跨数据集综合表 | 缺失 | 至少需要 DALES + OpenGF/Vaihingen 中之一形成外部有效性证据 |

Sonata/Concerto 等代码存在并不等于已完成对比。只有当其在相同 ALS 数据、相同输入、相同 backbone 或清楚标注的原生 backbone、相同训练预算下运行后，才能进入主表。

### 4.6 标签效率、消融与效率

| 实验 | 当前状态 | 最低建议 |
|---|---|---|
| DALES 1/5/10/20/100% labels | 缺失 | 主文至少 1/10/100%，完整 1/5/10/20/100% 可放附录；每档 3 seeds |
| 固定 mask | 缺失 | 先生成并版本化 sample/tile mask，所有方法共用 |
| VPSD patch-set vs pointwise | 缺失 | 至少保留一组核心对照 |
| registration shift 鲁棒性 | 缺失 | 0、轻微、中等 XY shift 三档即可 |
| native-1024/PCA-128/direct-128 | 部分资产 | PCA 工具和模型已有，但无保真度及下游结果 |
| PCA 维数 | 缺失 | 64/128/256 可放附录；正文只需论证选择 128 的依据 |
| RRCL role/context 消融 | 缺失 | role only、context only、role+context |
| joint optimization 消融 | 缺失 | naive sum、保护语义锚点、保护+累积 PCGrad |
| LitePT/PTv3 portability | 能力具备、无正式结果 | 至少同一小规模协议验证 PTv3；否则只写实现兼容，不写泛化结论 |
| 参数量/吞吐/峰值显存 | 缺失 | 同 GPU、同点数上限、同 AMP dtype、固定 warm-up 后测量 |

## 5. 当前最关键的实验风险

### 5.1 Swiss-Audit 尚未真正建立

当前两个预训练配置的 `test_pointcloud_path` 分别指向 Hubei 和 Yunnan 外部点云，作用是导出特征，不是 Swiss 内部机制审计。两个配置又都设置 `evaluate=False`，因此没有独立预训练验证指标。

应从 Swiss 原始大块或航带级别建立空间隔离的 Audit 集，覆盖至少：城区、郊区、农田、森林和复杂屋顶。必须在切 tile 前划分，避免相邻/重叠 tile 泄漏。该集合只用于固定特征审计和机制图，不参与 PCA 拟合、预训练或超参数选择。

### 5.2 当前可见性字段不等于大纲要求的几何可见性

`surface_visibility.py` 已能返回：

- `geometric_observability`
- `return_observability`
- 融合后的 `observability`

但现有工具主要将融合值写回 LAS/correspondence，并据此二值化。论文机制实验应单独落盘 `q_geo=geometric_observability`，用 `q_geo` 定义 `P_vis/P_amb/P_inv`；融合 observability 可作为工程可见性或补充实验，但不能用于主机制定义。

### 5.3 线性探测虽是线性头，但训练目标不是最标准的 LP

当前 DALES Linear Probe 的可训练模块确实只有 `Linear(C, K)`，架构上满足严格 Linear Probe；但损失使用 auto-weighted CE + Lovasz。若与外部论文的标准线性评估比较，建议：

- 主 Linear Probe 使用普通 CE、固定训练轮数、固定增强；或
- 明确命名为 “linear-head segmentation probe”，并保证所有方法完全使用同一损失。

Swiss-Audit 的 sklearn Logistic Regression/cosine kNN 与 DALES 点级分割 probe 应分开描述，不能混为同一协议。

### 5.4 Full Fine-Tuning 同时混入了预训练收益和架构收益

CVRL Full Fine-Tuning 保留完整 visual/return 分支和 200D 融合特征，而旧 scratch 是普通 LitePT。为了分离“预训练权重”和“更大/不同架构”两种因素，至少补充：

1. **Random-init Full-CVRL**：相同完整 CVRL 下游架构，从随机初始化训练；
2. **CVRL encoder-only → plain LitePT decoder**：只迁移共享 encoder，使用与 scratch 一致的 LitePT 下游结构；
3. **Plain LitePT scratch**：同样的 XYZ-only 输入和统一训练配置。

三者分别回答：完整架构本身是否更强、encoder 预训练是否有效、完整 CVRL 表征是否还有额外价值。

### 5.5 两种预训练尺度不能替代独立消融

当前 100 m/0.25 m 与 200 m/0.5 m 共有至少六个联动变化。论文可在同一 DALES/Audit 协议下把它们列为两个“预训练 operating points”，但必须注明不是单变量消融。若篇幅和算力有限，建议不要把尺度作为主贡献，只把 100 m/0.25 m 设为正式模型，200 m/0.5 m 放补充材料。

## 6. 推荐的最小可发表证据闭环

V6 全量计划规模很大。建议先完成下列五个闭环，再扩展附录实验。

### 闭环 A：视觉知识确实被点云模型内部化

在固定 Swiss-Audit 上比较：

```text
DINO-1024 → PCA-128 → VPSD/CVRL zV-128 → CVRL zFull-200
```

统一使用 tile-level split、L2 normalize、Logistic Linear Probe 和 cosine kNN，报告 overall、per-class 及建筑/桥梁/道路专项指标。额外报告 PCA-128 的累计解释方差或重建误差，证明 128D 不是未经验证的瓶颈。

### 闭环 B：纯视觉蒸馏存在几何不可见瓶颈

用纯几何 `q_geo` 定义可见/模糊/不可见区域；先用少量人工或多视角证据验证 `q_geo` 的 precision/recall，再比较 VPSD probe 在三类区域的性能。若 `P_inv` 明显更差，才能支持“visibility bottleneck”。

### 闭环 C：RRCL 在瓶颈区域提供互补信息

固定同一个下游头，比较 `zV`、`zR`、`zFull`、random concat、shuffled `zR`，分别报告 all/`P_vis`/`P_inv`。预期证据应同时满足：

- `zFull > zV`；
- 提升主要出现在 `P_inv` 或植被内部/边缘等回波敏感区域；
- random concat 和 shuffled `zR` 不能复现该提升。

当前 DALES 的 `zFull > zV > zR` 已满足第一条的单种子版本，后两条仍缺失。

### 闭环 D：公开数据迁移不是架构或输入造成的假提升

在 DALES 使用统一 XYZ-only 输入、统一模型和训练预算，至少完成：

- Plain LitePT scratch；
- Random-init Full-CVRL；
- VPSD-only；
- RRCL-only；
- CVRL；
- CVRL encoder-only transfer。

主结果至少 3 seeds，报告 mean ± std。Linear Probe、Decoder Probe、Full Fine-Tuning 分开成表，不用单个 FT 数字同时承担全部结论。

### 闭环 E：至少一个外部泛化维度

优先级建议：

1. **OpenGF 跨任务迁移**：现有 tiled 数据和 baseline 已具备，增量工程最小；
2. **DALES 低标注率**：最能体现预训练价值，建议至少 1%、10%、100%；
3. **Vaihingen 自然低数据**：论文价值高，但预处理和协议建设成本更大。

至少完成前两项中的一项，最好两项都完成。仅有 DALES 全量 FT 很难支撑完整自监督论文。

## 7. 建议的实验执行顺序

### P0：立即完成，防止后续结果不可追溯

- [ ] 冻结 100 m/0.25 m 为主 checkpoint，并记录 SHA256、文件大小、epoch、配置副本、git commit、PCA 模型 SHA256。
- [ ] 为 200 m 数据显式剔除或补齐那 1 个孤立点云，并生成 pointcloud/DINO/correspondence 配对 manifest。
- [ ] 建立空间独立 Swiss-Audit，保存原始块 ID、tile ID、场景类型和语义标签来源。
- [ ] 将纯几何 `q_geo` 与融合 `observability` 分字段保存；冻结 `P_vis/P_amb/P_inv` 阈值。
- [ ] 将 DALES 所有正式方法统一为 XYZ-only；第四通道若为兼容 stem 而补零，应在论文中明确，并确保所有对照一致。
- [ ] 固定 seeds（建议 42、3407、815）和所有 label mask。

### P1：主张成立所必需

- [ ] 跑完 Swiss-Audit 四级内部化分析。
- [ ] 完成 `q_geo` 有效性验证和 VPSD visible/invisible gap。
- [ ] 对 `zV/zR/zFull/random concat/shuffled zR` 做 3-seed probe。
- [ ] 重跑公平 DALES scratch、random-init Full-CVRL、encoder-only transfer 与 CVRL FT。
- [ ] 增加 Swiss VPSD-only、RRCL-only 的同预算 checkpoint；不能用旧 Hubei 不同架构实验替代。
- [ ] 生成 DALES per-class、building/bridge/road 或相近结构类别的专项表和混淆矩阵。

### P2：强化论文价值

- [ ] DALES 1/10/100% label efficiency，3 seeds；5%/20% 可随后补到附录。
- [ ] OpenGF scratch vs CVRL，至少 Full Fine-Tuning，最好再加 Linear/Decoder Probe。
- [ ] role only / context only / RRCL；naive sum / protected / protected+PCGrad。
- [ ] registration shift 鲁棒性和 PCA-128 保真度。

### P3：资源允许时进入附录

- [ ] Vaihingen 自然低数据实验。
- [ ] 64/128/256 PCA 维数。
- [ ] 100 m/0.25 m 与 200 m/0.5 m 在统一下游协议下的 operating-point 比较。
- [ ] PTv3 portability。
- [ ] LASDU 补充结果。

## 8. 推荐的论文表格与图对应关系

| 编号 | 建议内容 | 当前状态 |
|---|---|---|
| Table 1 | 数据集、面积、点数、密度、模态、用途 | 只有文件数；主要统计缺失 |
| Table 2 | Swiss-Audit 四级特征 Linear/kNN | 工具已有；结果缺失 |
| Table 3 | DALES LP/DP/FT 与公平 baselines，mean ± std | CVRL 单 seed 已有；公平 baseline 和方差缺失 |
| Table 4 | DALES label efficiency | 缺失 |
| Table 5 | VPSD/RRCL/joint/PCGrad/PCA 核心消融 | 缺失 |
| Table 6 | OpenGF/Vaihingen 跨数据或跨任务结果 | 缺失 |
| Table 7 | 参数量、显存、吞吐 | 缺失 |
| Fig. 1 | CVRL 完整方法图 | 设计文档和实现可支撑，需论文绘图 |
| Fig. 2 | `q_geo` 可见/模糊/不可见可视化 | 算法部分具备，正式 Audit 图缺失 |
| Fig. 3 | 四级特征 PCA/kNN/类别结构图 | 通用分析工具具备，固定审计结果缺失 |
| Fig. 4 | `P_vis/P_inv` 上 zV/zR/zFull 增益 | 缺失 |
| Fig. 5 | 标签效率曲线 | 缺失 |
| Fig. 6 | 典型建筑、桥梁、道路、植被案例 | 现有外部特征可作为候选，但需固定论文样本和统一色标 |

## 9. 建议的主表比较原则

为避免审稿时被质疑“预算不公平”或“架构不公平”，正式主表应遵循：

1. **同输入**：所有主对照使用 XYZ-only；不要让 scratch 使用 echo 而 CVRL 使用零 intensity。
2. **同划分**：同一 train/val/test，以及完全相同的低标注 mask。
3. **同训练预算**：epoch/optimizer/scheduler/augmentation/batch 或总更新数一致。
4. **同 backbone 口径**：原生方法可保留原生 backbone，但应另设“LitePT controlled”小节；不能混在一列暗示完全公平。
5. **同 probe**：Linear Probe 必须同一损失、同一训练轮数；kNN 必须同一归一化、k 和距离度量。
6. **多种子**：核心表至少 3 seeds，报告 mean ± std；单次最好值只能放补充材料。
7. **不按测试集选模型**：预训练采用固定最后 epoch，或在独立 Swiss validation 上选 best；不能根据 DALES test 反向选择 100 m/200 m checkpoint。
8. **报告参数量差异**：完整 CVRL FT 与 plain LitePT 参数空间不同，必须同时提供 random-init Full-CVRL 控制。

## 10. 最终判断

V6 大纲的科学问题组织是完整的：它不是简单堆公开数据集，而是围绕“视觉知识内部化—可见性瓶颈—回波互补—公开迁移—低标注收益”建立递进证据链。当前仓库已经完成了最难的模型工程，并通过 DALES 分支 Linear Probe 给出了互补性的第一条定量证据。

真正缺少的不是更多网络改动，而是**冻结协议、建立无泄漏 Swiss-Audit、把纯几何可见性从融合 observability 中分离、补齐公平 baseline 与多种子结果**。在这些完成之前，不建议继续扩展新模型版本；应把算力集中在可证伪、可复核的实验矩阵上。

建议论文主线以 100 m/0.25 m checkpoint 为准，200 m/0.5 m 作为补充 operating point。首先完成 Swiss-Audit、DALES 公平对照和 OpenGF/低标注实验；这三部分闭环后，CVRL 才具备从“有效工程方案”进入“论证完整论文方法”的证据基础。

