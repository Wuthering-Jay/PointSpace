# HPSD 不可视点监督传播：可实施的阶段性方案

## 1. 文档定位与结论

本文把此前的 HPSD + VRSR 概念方案收敛为能够直接排期、编码、测试和终止的工程计划。
方案依据当前 PointSpace 实现、相关论文正式版本以及截至 2026-08-17 可访问的作者官方
代码仓库制定。这里的目标不是一次性复现 DSP、AIScene、AADNet、RAC-Net 和 DGNet，
而是只提取其中已经被论文和代码共同验证、又适合机载 LiDAR 正射不可视问题的部分。

建议保持现有 concat-HPSD 完全可独立运行，在它之外新增一个训练期监督传播模型
`HPSD-VRSR-v1m1`。第一版只完成“传播空间校准 + 样本内可靠软传播”，不要立即加入
全局 prototype、跨样本队列、语义分类头或双分支增强一致性。只有前一阶段通过明确的
Go/No-Go 条件后，下一阶段才进入主线。这样可以判断增益究竟来自不可视点监督，还是
来自额外参数、正则化或更长训练。

推荐的落地顺序为：

```text
P0 数据审计
  -> P1 无损暴露 HPSD 上下文与通用算子
  -> P2 建立 DINO 锚定的 128D 传播空间
  -> P3 样本内 Local VRSR MVP
  -> P4 可靠度与机载几何软约束
  -> P5 跨样本球面 prototype fallback
  -> P6 可选的跨 tile memory queue
  -> P7 完整训练、下游验证与论文消融
```

P0 至 P4 是建议完成的主线，P5 是通过前述验证后才加入的增强项，P6 只是研究性备选。
这一区分很重要：一个轻量 EMA prototype bank 不是 DSP 意义上的跨样本 feature
reallocation，也不应在论文中这样表述。

## 2. 官方论文与代码核对结果

### 2.1 DSP：借鉴梯度传播思想，不照搬稠密 affinity

[Dense Supervision Propagation](https://arxiv.org/abs/2107.11267) 的关键贡献是通过
cross-sample feature reallocating 和 intra-sample feature redistribution 重排特征，
使有限监督的梯度能够到达未标注位置。论文还表明 cross 与 intra 不适合无条件同时
训练，分阶段训练更稳定。其原始方法面向带少量人工类别标签的室内点云，依赖共享类别、
双线性 affinity 和解码器监督；本项目面对的是 DINO teacher 缺失且缺失机制与正射
遮挡相关，不能直接复制该网络。

本方案只采用 DSP 的两个原则：监督必须对不可视 target 产生可证明的非零梯度；复杂
传播机制必须分阶段启用。没有把 DSP 的全矩阵 affinity 或双样本共享类别配对列入 MVP。
论文页面没有给出可供直接移植的官方实现，因此该部分属于“依据论文重新实现”，而非
“移植官方代码”。

### 2.2 AIScene：可借鉴场景内外分治，当前仓库不能作为代码依赖

[AIScene 论文](https://openaccess.thecvf.com/content/CVPR2025/html/Liu_Exploring_Scene_Affinity_for_Semi-Supervised_LiDAR_Semantic_Segmentation_CVPR_2025_paper.html)
使用 teacher-student 伪标签、scene 内 point erasure 以及跨多个 scene 的 patch/instance
mixing。它的重要启示是场景内一致性与场景间信息利用应被拆开评估，而不是把所有传播
统称为 cross-sample affinity。

作者的[官方仓库](https://github.com/azhuantou/AIScene)截至核对日期只有 README，尚无
可复用训练实现。因此本项目不依赖 AIScene 代码，也不实现 point erasure 或多场景
patch mixing；只借鉴“先样本内、后样本间”的实验组织原则。

### 2.3 AADNet：直接借鉴非均匀监督意识，不移植其损失

[AADNet 论文](https://ojs.aaai.org/index.php/AAAI/article/view/32680)指出稀疏监督不仅
数量少，还会空间分布不均。其两个正式模块是 label-aware downsampling 和
multiplicative dynamic entropy with asynchronous training。论文特别讨论了二维投影
造成的非均匀标注，这与机载正射可视性具有直接相关性。

[AADNet 官方仓库](https://github.com/panzhiyi/AADNet)显示，LaDS 实现在
`openpoints/dataset/data_util.py::voxelize`：每个 voxel 排序后优先保留带标签点；
MDE-AT 实现在 `openpoints/loss/build.py::AsynchronousCrossEntropy`：交替进行带熵
校准的交叉熵和熵正则。HPSD 预训练没有类别 logits 和人工 sparse label，直接移植
MDE-AT 在数学上并不成立。可实施的借鉴方式是统计每个 tile、每个高度层和每个 token
的视觉监督密度，并在 source 采样与 loss 汇总时做样本平衡，而不是把 VRSR 的
scene-balanced sampling 称为 AADNet 复现。

### 2.4 RAC-Net：借鉴“置信度不能单独代表可靠度”

[RAC-Net 论文](https://arxiv.org/abs/2303.05164)使用预测置信度与多次增强预测的不确定性
共同划分 reliable/ambiguous 点；可靠点接受 hard pseudo-label CE，模糊点仍保留 soft
KL consistency，而不是被全部丢弃。[官方仓库](https://github.com/wu-zhonghua/RAC-Net)
的核心逻辑位于 `pcr/engines/defaults.py`：对原始、PointWolf 和 affine 预测求均值与
标准差，然后用 `max_probability >= tau` 且对应类别标准差小于 `kappa` 选择可靠点。

VRSR 没有语义类别概率，也不计划为此增加多次 backbone 前向，所以不能直接使用
RAC-Net 的 confidence/uncertainty。可以等价地组合多个与检索质量相关但来源不同的
证据：source 的 DINO purity、top-1 相似度、top-1/top-2 margin、Top-K 权重熵，以及
软几何兼容度。阈值必须由 P0/P3 的实际分布确定，不能照搬 RAC-Net 的 0.7/0.05。

### 2.5 DGNet：prototype 阶段只做受控简化

[DGNet 论文](https://proceedings.neurips.cc/paper_files/paper/2024/hash/38d6af46cca4ce1f7d699bf11078cb84-Abstract-Conference.html)
在单位超球面上用 mixture of von Mises-Fisher distributions 描述语义 embedding，并
通过可靠类别初始化、soft assignment 和 Nested EM 交替优化分布参数与网络。其消融
表明 soft assignment 优于 hard assignment。

[DGNet 官方仓库](https://github.com/panzhiyi/DGNet)的主要实现位于
`openpoints/loss/build.py::vMFLoss`，包含类别均值初始化、soft posterior、EM 更新、
vMF loss、prototype discrimination 和 prediction consistency；训练循环还跨 batch
累计 prototype。VRSR 没有人工类别中心，因此不能声称复现 moVMF。P5 只实现
class-agnostic、EMA 更新的 spherical k-means bank，并保留 soft assignment。它是由
DGNet 的球面分布建模得到的工程简化，必须独立消融。

## 3. 当前 PointSpace 基线与不可破坏约束

现有 [`hpsd_v1m1.py`](../pointspace/models/backbone/hpsd/hpsd_v1m1.py) 已经完成以下
稳定能力：在 `distill_level` 构造输入点到 token 的映射；保留 token 与 DINO patch
之间完整的多对多稀疏边；把目标层和全部更深层对齐后 concat；仅对有效 patch 的聚合
特征执行 1024 维 MLP projector；以 cosine loss 对齐原生 DINO 特征。新的传播分支不得
改变这些默认语义、loss 数值、日志字段以及 HPSDFeatureTester 的导出结果。

LitePT-v1m4 和 PT-v3m4 都通过
[`hierarchy.py`](../pointspace/models/backbone/hpsd/hierarchy.py)暴露统一的
`HierarchyLevel(point, input_to_level, level)`，因此 VRSR 应只依赖这个协议，不在两个
backbone 内分别实现传播。目标层暂时沿用 HPSD 的 `distill_level=2`，也不新增
`fusion_level`；传播特征直接复用当前 `fuse_hierarchy_features()` 的 concat 结果。

[`LasImageDataset`](../pointspace/datasets/las_image.py)当前提供逐点
`dino_patch_index` 和 `dino_valid`，合批后 `dino_patch_index` 已转换为 batch 全局 patch
行号，`dino_offset` 提供样本边界。需要注意，当前 `dino_valid` 同时表示影像范围覆盖和
正射表面可见性。第一版可以把所有 `False` 都视为无 teacher target，但诊断时不能区分
“影像未覆盖”和“被遮挡”。未来 correspondence schema 可以可选增加
`dino_image_covered` 与 `dino_surface_visible`，同时继续保留
`dino_valid = covered & visible` 以兼容旧数据；这不是 P1-P4 的前置条件。

另外，两份现有配置的训练长度定义不同：LitePT 使用 `epoch=10, loop=10`，PTv3 使用
`epoch=100, loop=1`，batch 规模也不同。因此阶段调度必须以 optimizer step 比例或独立
checkpoint 阶段表示，不能写死“第 3 个 epoch 开启”。

## 4. 目标架构与梯度路径

### 4.1 总体结构

新模型建议注册为 `HPSD-VRSR-v1m1`，继承
`HierarchicalPatchSetDistiller`，使已有 checkpoint 的 `backbone.*` 和
`student_projector.*` 键保持兼容；新增参数统一置于 `vrsr.*`。原有 `HPSD-v1m1` 不启用
任何 VRSR 代码路径。

```text
输入点云 + DINO patches + correspondence
             |
      LitePT-v1m4 / PT-v3m4
             |
   hierarchy + F_H (level 2 concat)
      |                    |
      |                    +--> HPSD: token->patch 聚合 -> MLP1024 -> L_hpsd
      |
      +--> VRSR: token 可视率/teacher 聚合
                       |
                  MLP128 传播空间
                       |
             可视 source -> 不可视 target
                       |
                  L_cal + L_local
                       |
            [可选] prototype / queue
```

总损失第一版定义为：

```text
L = L_hpsd + lambda_cal * L_cal + lambda_local * L_local
```

P5 通过后才增加 `lambda_proto * L_proto`。HPSD loss 始终存在，因而传播空间即使发生
退化，也不能删除原始 DINO 蒸馏锚点。

### 4.2 128D 传播空间必须显式校准

对 level-2 token `t`，先由现有 token-patch edges 聚合对应 DINO teacher：

```text
w_tp       = sqrt(point_count_tp)
d_bar_t    = normalize(sum_p w_tp * dino_p)
support_t  = sum_p point_count_tp
q_t        = valid_point_count_t / all_point_count_t
```

使用固定、带 seed 的 1024x128 正交随机投影 `R` 把 teacher 映射到稳定的低维球面：

```text
teacher128_t = normalize(d_bar_t @ R)
student128_t = normalize(prop_head(F_H_t))
L_cal        = mean(1 - cos(student128_t, teacher128_t))
```

`R` 通过 `register_buffer` 保存进 checkpoint，使用 fp32 初始化和归一化，前向可在 AMP
下执行。`prop_head` 建议为：

```python
nn.Sequential(
    nn.LayerNorm(projector_in_channels),
    nn.Linear(projector_in_channels, 256),
    nn.GELU(),
    nn.Linear(256, 128),
)
```

连续保留 `L_cal` 比“校准若干 epoch 后完全冻结 head”更稳健，因为 VRSR loss 会持续
改变 backbone 特征分布；固定 `R` 则保证 teacher 语义坐标系不会随训练漂移。这里的
128D 只用于传播检索，HPSD 原生 1024D teacher 和 projector 均不压缩。

### 4.3 source、target 和 teacher purity

MVP 中只把 `q_t == 0` 的 fully-invisible token 设为 target。混合 token 已包含可视点并
可通过 token 共享表示接受 HPSD 监督，过早把 `0 < q <= 0.2` 也作为 target 会重复施加
监督，难以解释增益。

source 初始条件建议为：

```text
q_t >= source_q
support_t >= min_source_points
num_teacher_patches_t >= min_source_patches
purity_t >= source_purity
```

其中 teacher purity 是 token 内各 DINO patch 与聚合 teacher 的加权平均 cosine：

```text
purity_t = sum_p w_tp * cos(dino_p, d_bar_t) / sum_p w_tp
```

`source_q`、`source_purity` 等不能先验固定为 0.6、0.7。P0 应输出分位数，第一版取能保留
约 20%-40% 可视 token 的阈值，再由验证集调整。source 太少时宁可让该 tile 的 VRSR
loss 为零，也不要降低阈值强行传播。

### 4.4 Local VRSR 的严格梯度语义

同一样本内，从 source 中检索每个 target 的 Top-K 邻居。检索和 reference 构造全部在
`no_grad` 下进行，但 loss 左侧的 target 不能 detach：

```python
target_train = student128[target_idx]              # 保留梯度
with torch.no_grad():
    target_search = target_train.detach()
    source_search = student128[source_idx].detach()
    topk_sim, topk_pos = chunked_topk(target_search, source_search, k)
    weight = softmax(topk_sim / temperature, dim=-1)
    reference = normalize((weight[..., None] * source_search[topk_pos]).sum(1))

loss_local = (1.0 - (target_train * reference).sum(-1))[accepted].mean()
```

这样 `dL_local / dF_H_target` 明确非零，reference 不会被 target 反向拖动，也不会因为误把
`target_train` detach 而形成数值正常但没有训练作用的假 loss。source 只通过
`L_cal/L_hpsd` 维持 teacher 语义，P3 不让 local loss 更新 source。

检索必须按 batch sample 分组，不能让同一 forward 中不同 tile 在 P3 意外互相匹配。
每个样本设置 `max_sources` 和 `max_targets`，source 用高度分层随机采样而不是简单取前 N
个；target 超限时随机采样并记录覆盖率。相似度使用 chunked matrix multiplication，禁止
构造完整 `[num_target, num_source]` 常驻矩阵。

## 5. 分阶段实施计划

### P0：数据覆盖与规模审计，不改模型

这一阶段新增只读工具 `utils/audit_hpsd_visibility.py`，在真实训练 transform 之后统计点级
和 level-2 token 级可视性，而不是只读取原始 correspondence。工具应使用当前模型构造
hierarchy，但放在 `torch.inference_mode()` 下，不创建 projector 激活。

每个样本至少记录点数、level-2 token 数、valid 点比例、`q=0` token 比例、混合 token
比例、可作为 source 的 token 数、每 token patch 数、teacher purity、每样本有效 patch
数和高度分层覆盖。汇总只保存直方图/分位数和少量 reservoir sample，不保存全部逐点或
逐 token 数组，避免重演大规模特征分析中的内存问题。

输出建议为一个小型 `visibility_audit.json` 和一个 `visibility_audit.md`。此阶段还要记录
LitePT/PTv3 在真实 `batch_size_train` 下的基线 step time、峰值显存和 HPSD loss 分布。

Go 条件：至少 90% 的训练 tile 同时具有 source 和 fully-invisible target；或虽然低于
90%，但全数据 target 占比足够且有明确的按 tile 跳过策略。No-Go 条件：多数 tile 没有
任何高质量 source，或 `q=0` 主要来自整幅影像缺失而非局部遮挡。这时应先改善影像覆盖
或配对数据，VRSR 不能凭空创造 teacher。

### P1：上下文接口与无损基础算子

建议新增：

```text
pointspace/models/backbone/vrsr/__init__.py
pointspace/models/backbone/vrsr/ops.py
pointspace/models/backbone/vrsr/context.py
tests/models/test_vrsr_ops.py
```

对 `hpsd_v1m1.py` 只做一次无行为变化的内部重构：公开
`forward_train(input_dict, return_context=False)`，原 `forward()` 仍返回完全相同的
`loss/tok/edge/patch`。当新子类请求 context 时，额外返回 dataclass 引用：

```text
HPSDTrainContext
  point
  hierarchy
  level
  distill_feat        [T, C_H]
  edges               TokenPatchEdges
  teacher             [P, 1024]
```

context 不放入常规 result dict，避免 InformationWriter 把大 tensor 当作日志项，也避免
Trainer 修改行为。`ops.py` 实现 token visibility、patch-to-token teacher aggregation、
purity 和按 sample 分组索引；全部函数应支持空输入。

必须完成的回归测试包括：现有 HPSD 单元测试全部通过；同一随机输入重构前后 HPSD loss
在 fp32 下逐元素一致；`return_point_feature` 与 HPSDFeatureTester 不变；空 valid batch
仍能反向并产生零梯度；input-to-level 不跨 batch。

Go 条件：HPSD 回归完全通过，新增 context 不复制 `[N,1024]` 或 `[N,C_H]` 张量，基线
峰值显存和 step time 变化不超过测量噪声（建议阈值 2%）。

### P2：传播空间校准

新增 `pointspace/models/backbone/vrsr/vrsr_v1m1.py`，先只实现 `prop_head`、固定 `R`、
source 构造和 `L_cal`，`lambda_local=0`。建议模型子类注册名为 `HPSD-VRSR-v1m1`，参数
仍通过完整的 LitePT/PTv3 HPSD 配置传入。

P2 应从现有 HPSD checkpoint 开始，而不是同时随机训练 HPSD 和 propagation head。
训练初期可以保留 HPSD loss；`lambda_cal` 从 0 线性 warmup 到 0.05 或 0.1。具体值由
HPSD loss 与 calibration loss 的 backbone 梯度范数确定，目标是 VRSR 新分支的梯度范数
不超过 HPSD 的约 25%，而不是追求两个 loss 数值相近。

需要记录的简称建议为：

```text
loss, hpsd, cal, src, q0, pcos
```

其中 `pcos` 是 validation source 上 student128 与 teacher128 的平均 cosine；详细分位数
写 TensorBoard，不放在每步终端日志。

Go 条件：独立验证子集上 calibration cosine 持续上升并稳定；source retrieval 的同
teacher 邻域一致性显著优于随机投影初始值；HPSD validation loss 不恶化超过 2%；无
NaN/Inf。P2 没有通过时不能开启 local propagation。

### P3：样本内 Local VRSR MVP

P3 只加入 fully-invisible target、样本内 Top-K、stop-gradient soft reference 和
`L_local`。默认建议从以下保守参数开始，最终值必须以 P0 分位数为准：

```python
vrsr=dict(
    propagation_channels=128,
    source_q=0.6,
    target_q=0.0,
    min_source_points=4,
    min_source_patches=1,
    source_purity=None,       # 由 audit 分位数写回配置
    topk=8,
    temperature=0.10,
    max_sources=512,
    max_targets=1024,
    query_chunk_size=256,
    lambda_cal=0.05,
    lambda_local=0.02,
)
```

`lambda_local` 应在 P2 checkpoint 上从 0 warmup；第一轮实验不使用 reliability hard
threshold 和 prototype fallback，所有具备 source 的 target 只按 soft neighbor entropy
做连续权重。这样可以先回答“传播本身是否有用”，而不是同时验证五个启发式模块。

必须增加一个梯度单测：构造一个可视 source 和一个不可视 target，令 HPSD loss 对 target
为零，反向 `L_local` 后断言 target 对应 `distill_feat.grad` 非零、reference 无梯度；再把
target 错误 detach，测试应能捕获零梯度。还需覆盖无 source、无 target、Top-K 大于
source 数量、一个 batch 多 tile、AMP fp16/bf16 和不同 token 数。

Go 条件：相对 P2，训练峰值显存增加不超过 10%（或绝对不超过 300 MiB），step time
增加不超过 15%；下游线性探测或短程微调中不可视子集指标有稳定提升，且可视子集无明显
下降。建议至少两个 seed；单次 loss 更低不构成通过依据。

### P4：可靠度与机载几何软约束

P4 在 P3 有效后才加入。可靠度不使用单一 cosine 阈值，而组合：

```text
r_source   = teacher_purity * support_saturation
r_retrieval= f(top1_similarity, top1-top2_margin, normalized_topk_entropy)
r_geometry = soft_height_compatibility * soft_xy_compatibility
r_total    = r_source * r_retrieval * r_geometry
```

机载点云中屋顶/立面、冠层/林下地面的几何差异是系统性的，纯语义最近邻容易把不可视
结构拉向正射表面。第一版几何项只使用 level token 的平均 `coord`，在每个 tile 内用
robust height range（例如 z 的 5%-95% 分位差）归一化。几何项应为 soft weight，不做
固定米制 hard rejection；否则高层建筑侧面可能完全失去 source。echo/intensity gate 暂不
加入，因为当前 hierarchy 不保证这些原始字段以独立、可解释的名字保留到 level-2。

阈值确定方式采用 P0/P3 的 held-out 分布：先保留可靠度最高的 30%-50% target，逐步扩大
覆盖。RAC-Net 的启示是 ambiguous target 不应永久丢弃，但 VRSR 的第一步应让它们权重
接近零，而不是另开一次增强前向计算 KL。只有在高可靠 target 明确有效后，才对 ambiguous
target 使用低权重、较高温度的 soft consistency。

终端日志新增简称 `acc`（accepted targets）、`rel`（平均可靠度）、`ent`（Top-K 熵），
完整分布只进入 TensorBoard。Go 条件是 P4 相对 P3 提升不可视子集且 target coverage
没有塌缩；若 accepted ratio 长期低于 10%，应回退连续加权，不继续提高筛选复杂度。

### P5：跨样本 spherical prototype fallback

P5 解决的是某个 tile source 太少或完全没有 source 的情况，不替代 Local VRSR。新增：

```text
pointspace/models/backbone/vrsr/prototype_bank.py
tests/models/test_vrsr_prototype_bank.py
```

prototype bank 保存 `K x 128` 的归一化 buffer、EMA support 和 age，默认 K 从 32 开始。
只用高质量 source 更新：每隔 `update_interval` 将 source 分配到最近 prototype，按 rank
计算 feature sum/count，经 `torch.distributed.all_reduce` 后做 EMA 更新并重新归一化。
空 cluster 保持原值并增加 age；连续过期才用当前最远 source 重置。所有 bank 更新均在
`no_grad` 中完成，不进入 optimizer，也不保存历史点级 tensor。

对没有可靠 local reference 的 target，使用多个 prototype 的 soft assignment 构造
reference；有 local reference 时默认不用 prototype，避免两种教师冲突。该实现保留
DGNet 的球面归一化和 soft assignment 思想，但不实现有类别初始化、moVMF 浓度参数、
Nested EM、discriminative loss 或分类 posterior consistency。

DDP 测试必须验证两个 rank 在更新后 prototype bitwise/容差一致；resume checkpoint 后
bank、support、age 完整恢复；单卡与多卡在相同 source 集上结果一致。Go 条件是
anchor-poor tile 的 target coverage 和下游不可视指标提升，同时正常 tile 不下降；若只
降低训练 loss 而没有下游收益，prototype 分支应删除。

### P6：可选跨 tile memory queue

只有当 P5 证明跨样本先验有效、但 prototype 过度平滑时才进入 P6。queue 保存其他 tile
的高可靠 source128、少量几何摘要、tile id 和 age，固定上限如 4096-8192 条，全部 detach；
当前 target 对 queue 做同样的 chunked Top-K。它比 prototype 保留更多局部模式，但会增加
通信、检索和 stale feature 问题。

该 queue 仍不是 DSP 的双向 live-sample gradient reallocation，因为历史 source 没有
梯度。若论文贡献必须严格包含跨样本梯度重分配，需要另做 live pair 实验：同一 batch
选择两个当前样本、保留双方图、双向构造监督，并与 queue 区分命名。考虑到 PTv3 当前
单卡 micro-batch 可能为 1，live pair 会显著提高显存和数据调度复杂度，不建议作为主线。

P6 的 Go 条件必须高于普通增强项：相对 P5 的不可视子集提升应大于多 seed 方差，并且
训练耗时增加不超过 25%。否则保留 P5 或直接停在 P4。

### P7：完整训练与下游验证

建议新增两份不继承 base 的完整配置：

```text
configs/hpsd/pretrain-hpsd-vrsr-litept-v1m4-hubei.py
configs/hpsd/pretrain-hpsd-vrsr-ptv3-v3m4-hubei.py
```

每份配置都保留当前 `batch_size_train/batch_size_test` 逻辑，不恢复废弃的总
`batch_size`。阶段切换优先使用独立训练运行与 checkpoint，而不是在一个 run 中动态
修改 `requires_grad`：

```text
run A: mode=calibrate, load HPSD checkpoint
run B: mode=local,     load run A checkpoint
run C: mode=reliable,  load run B checkpoint
run D: mode=prototype, load run C checkpoint（可选）
```

这样每一阶段都能复现、回滚和独立比较，也不需要新增复杂 StageSchedulerHook。配置仍可
通过 `tools/train.py --options model.vrsr.mode=...` 做短程 smoke test；正式实验保存展开后的
完整配置。`tools/test.py` 和 `HPSDFeatureTester` 对新子类继续使用继承的
`return_point_feature` 路径，VRSR 训练分支在推理时不执行，导出的 safetensors 格式不变。

## 6. 代码接口草案

```python
@dataclass
class VRSRStats:
    token_visibility: torch.Tensor      # [T]
    valid_count: torch.Tensor           # [T]
    total_count: torch.Tensor           # [T]
    teacher128: torch.Tensor            # [S, 128]
    teacher_token: torch.Tensor         # [S]
    teacher_purity: torch.Tensor        # [S]
    teacher_support: torch.Tensor       # [S]


class VisibilityReliableSupervisor(nn.Module):
    def forward(
        self,
        distill_feat,
        level,
        edges,
        dino_feature,
        dino_valid,
        point_offset,
    ) -> dict:
        # 返回 cal/local/proto loss 及纯标量统计，不返回大 tensor。
        ...


@MODELS.register_module("HPSD-VRSR-v1m1")
class HPSDVRSRDistiller(HierarchicalPatchSetDistiller):
    def forward(self, input_dict, **kwargs):
        if kwargs.get("return_point_feature", False):
            return super().forward(input_dict, **kwargs)
        hpsd_result, context = self.forward_train(input_dict, return_context=True)
        vrsr_result = self.vrsr(context=context, input_dict=input_dict)
        return {
            "loss": hpsd_result["loss"] + vrsr_result["loss"],
            "hpsd": hpsd_result["loss"].detach(),
            "cal": vrsr_result["cal"].detach(),
            "loc": vrsr_result["local"].detach(),
            "src": vrsr_result["source_count"],
            "tgt": vrsr_result["target_count"],
            "acc": vrsr_result["accepted_count"],
        }
```

实际实现时不能把 `hpsd` 的 detached 日志项再次计入 loss；Trainer 只对统一的 `loss`
反向。所有 count 和 diagnostic tensor 都应是标量，避免训练日志和显存被无意放大。

## 7. 资源预算与复杂度控制

令 level-2 token 数为 `T`，每样本 source/target 上限为 `S/Q`，传播维数为 `D=128`，
query chunk 为 `Cq`。传播特征存储为 `O(TD)`，每个检索 chunk 的相似度为 `O(Cq*S)`，
计算量为 `O(QSD)`；它不随原始点数 N 直接生成 1024 维逐点激活。

以 `T=8000, S=512, Q=1024, Cq=256` 为例，fp32 单个相似度 chunk 约 0.5 MiB，128D
token 特征约 4 MiB；主要额外显存来自 prop_head 的 autograd activation，而不是 Top-K
索引。实际峰值仍必须在两种 backbone 和真实 batch 下测量，因为 concat `F_H` 已由 HPSD
持有、是否被 autograd 重用会影响结果。

建议硬预算如下：P2 额外 step time 小于 8%、显存小于 150 MiB；P3/P4 累计 step time
小于 15%、显存小于 300 MiB；P5 累计 step time 小于 20%。这些是工程停止线，不是理论
承诺。若超限，按顺序减小 `max_targets`、`max_sources`、`query_chunk_size`，最后才考虑把
传播维数从 128 降到 64；不要压缩 HPSD 的 1024D teacher。

## 8. 验证矩阵

### 8.1 单元测试

必须覆盖 token visibility 与手工计数一致、patch-to-token 聚合和 purity 正确、batch
边界不串样本、空边/空 source/空 target、Top-K 退化、固定投影 checkpoint 恢复、严格
detach 梯度、AMP、prototype DDP 同步及 resume。随机测试应设置确定性 seed。

### 8.2 集成测试

使用 `pointcept` 环境分别构建 LitePT-v1m4 与 PT-v3m4，执行一个真实
`LasImageDataset` batch 的 forward/backward/optimizer step；验证 loss 有限、VRSR target
梯度非零、HPSD 统计不变。再用 `tools/train.py` 运行至少 100 step smoke test，并用
`tools/test.py` 导出一个包含多个 fragment 的 tile，确认合并顺序和 safetensors 元数据
与原 HPSD 完全一致。

### 8.3 表征和下游指标

仅比较预训练 loss 不足以证明不可视点学得更好。最终至少报告：

| 维度 | 指标 |
| --- | --- |
| 训练健康度 | HPSD/cal/local loss、梯度范数、accepted ratio、NaN/OOM |
| 效率 | samples/s、step time、峰值显存、checkpoint 大小 |
| 全体下游 | OA、mAcc、mIoU 或现有任务主指标 |
| 可视子集 | 由 `dino_valid=True` 划分的指标 |
| 不可视子集 | 由 `dino_valid=False` 划分的指标 |
| 结构子集 | 若可获得，建筑侧面、冠层内部、林下地面分别统计 |
| 表征诊断 | source-target cosine、Top-K entropy、prototype occupancy |

下游验证应使用冻结线性探测和完整微调两种协议。线性探测更能判断表示本身是否改善，
完整微调则反映实际任务收益。每个关键对比至少两个 seed；若差异小于 seed 方差，不能把
该阶段合入默认配置。

## 9. 最小消融集合

为避免实验数量失控，主线只保留以下递增消融：

| 编号 | 方案 | 回答的问题 |
| --- | --- | --- |
| A | 当前 concat-HPSD | 稳定基线 |
| B | A + calibration only | 额外 head/teacher128 是否本身改变表示 |
| C | B + Local MVP | 不可视 target 的直接传播是否有效 |
| D | C + reliability/geometry | 是否减少错误传播 |
| E | D + prototype fallback | 跨样本分布先验是否帮助 anchor-poor tile |
| F | E + queue（可选） | 细粒度跨 tile memory 是否优于 prototype |

如果 C 不优于 B，应停止 VRSR 主线，而不是继续堆叠 D/E/F。若 C 有效而 D 无效，则保留
连续软传播；若 E 无效，则结论应是样本内传播已经足够，而不是继续增大 prototype 数。

## 10. 主要风险与对应处置

最严重的算法风险是 confirmation collapse：不可视 target 被拉向当前 student source，
而 source 空间本身没有 teacher 锚定。P2 的固定 teacher 投影和持续 `L_cal` 是对此的核心
防护。第二个风险是系统性几何错配，P4 的 soft geometry 只能降低风险，不能恢复正射影像
从未观察到的独有语义。因此报告和论文必须把 VRSR 定位为“利用共享语义结构改善不可视
表示”，不能宣称重建了不可见区域的真实视觉特征。

工程上最容易出现的是零梯度假实现、跨 batch 误检索、DDP prototype 漂移、日志携带大
tensor 以及阶段配置不可复现。对应措施分别是严格梯度单测、按 batch 分组、all-reduce
后的统一 EMA、只返回标量统计和独立 checkpoint 阶段。

数据上最大的限制是 `dino_valid` 混合了影像未覆盖与遮挡不可视。P0 必须至少统计空间
连续的大块无覆盖区域；若将来重新生成 correspondence，建议升级可选的双 mask，而不是
破坏旧字段。对于完全没有影像覆盖且没有任何 source 的 tile，P3 loss 必须为零，P5 也
只能提供分布先验，不能把黑色默认影像当作有效 teacher。

## 11. 建议的首个实现里程碑

首个可合并里程碑只包含 P0-P3，不包含 prototype 和 queue。其完成定义是：

1. 现有 HPSD 所有测试和导出结果不变；
2. visibility audit 能在真实湖北数据上流式完成且内存有界；
3. P2 能学到稳定的 DINO 锚定 128D source 空间；
4. P3 的不可视 target 有经过单测证明的非零梯度；
5. LitePT/PTv3 均能通过真实 batch、AMP、100-step 与 fragment export 测试；
6. 至少一次下游实验显示不可视子集改善，资源增量不越过预算。

只有这六项全部满足，才进入 P4/P5。按这个边界实施，第一版预计涉及一个 HPSD 内部无损
重构、一个新的 VRSR 包、两份完整配置、一项审计工具和约 20-30 个针对性测试，范围明确、
可回滚，也足以验证核心研究假设。

## 参考资料

- Jiacheng Wei et al., [Dense Supervision Propagation for Weakly Supervised Semantic Segmentation on 3D Point Clouds](https://arxiv.org/abs/2107.11267).
- Chuandong Liu et al., [Exploring Scene Affinity for Semi-Supervised LiDAR Semantic Segmentation](https://openaccess.thecvf.com/content/CVPR2025/html/Liu_Exploring_Scene_Affinity_for_Semi-Supervised_LiDAR_Semantic_Segmentation_CVPR_2025_paper.html), [official repository](https://github.com/azhuantou/AIScene).
- Zhiyi Pan et al., [Point Cloud Semantic Segmentation with Sparse and Inhomogeneous Annotations](https://ojs.aaai.org/index.php/AAAI/article/view/32680), [official repository](https://github.com/panzhiyi/AADNet).
- Zhonghua Wu et al., [Reliability-Adaptive Consistency Regularization for Weakly-Supervised Point Cloud Segmentation](https://arxiv.org/abs/2303.05164), [official repository](https://github.com/wu-zhonghua/RAC-Net).
- Zhiyi Pan et al., [Distribution Guidance Network for Weakly Supervised Point Cloud Semantic Segmentation](https://proceedings.neurips.cc/paper_files/paper/2024/hash/38d6af46cca4ce1f7d699bf11078cb84-Abstract-Conference.html), [official repository](https://github.com/panzhiyi/DGNet).

