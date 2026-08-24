# HPSD + VRSR 融合方案评估报告

## 1. 评估对象与结论

本报告评估 [`HPSD_VRSR_最终设计方案.docx`](../HPSD_VRSR_最终设计方案.docx)
提出的融合方案，重点判断 VRSR 是否适合作为机载激光雷达正射不可视点的学习分支，
以及它与当前 concat-HPSD、LitePT-v1m4、PT-v3m4、LasImageDataset、Trainer 和
HPSDFeatureTester 的兼容性、实现复杂度与资源消耗。

总体结论是：**VRSR 适合作为不可视点学习的第一版工程分支，但应“有条件采纳”，
不建议完全按照当前文档直接实现。** 它的 Local Reallocation 本质上是可见 source
到不可视 target 的可靠图一致性：不可视 target 在 128 维空间检索可视 source，并
被软 reference 约束。该 loss 对不可视 target 具有明确、直接的非零梯度，训练时可
删除、推理零开销，且比完整 DSP affinity 或 1024 维逐点传播节省很多资源。因此，
从工程落地、显存可控和与现有 HPSD 共存三个角度看，方向是合理的。

但当前方案还有三个必须在实现前修正的问题。第一，新增 `prop_proj` 没有任何显式
DINO 语义校准，warm-up 只训练 HPSD，并不会训练这个新投影头；随机初始化的 128D
空间未必能可靠表达 HPSD 已学到的视觉语义。第二，文档同时写了 target 用于训练和
`T.detach()` 用于检索，如果实现时把 loss 中的 target 也 detach，VRSR 对 backbone
将完全没有梯度。第三，Global Prototype Guidance 是跨样本潜在分布约束，而不是
DSP 意义上的跨样本 feature reallocation；它可以作为低成本 fallback，却不能支撑
“实现了跨样本梯度重分配”的强表述。

因此推荐把当前方案定位为 **VRSR-Lite：visibility-aware reliable graph consistency
with prototype fallback**。先修正传播空间语义锚定与梯度接口，只实现样本内 local
MVP；确认不可视子集提升后再加入 global prototype。若研究目标仍要求真正的跨样本
feature reallocation，应在后续另加 queue/live-pair cross-token 消融，而不是把全局
prototype 等同于该机制。

## 2. 方案的有效部分

### 2.1 问题定义准确

文档正确区分了随机缺标与机载 LiDAR 的系统性不可视。正射影像倾向于覆盖屋顶、
冠层和开阔地表，而建筑侧面、冠层内部和林下地面的大量有效回波没有直接像素 teacher。
这种缺失与三维结构、回波层次和观测方向相关，不能简单将最近 patch 特征复制给无效
点。文档明确允许“无可靠锚点则不传播”，并将 VRSR 设置为低权重辅助项，这一点符合
传感器信息边界，也能减少错误监督扩散。

### 2.2 连续 token visibility 比二值 valid 更适合层级 backbone

方案定义：

```text
q_i = N_i_valid / N_i
```

这比直接在 level 2 使用点级 `dino_valid` 更合理，因为一个下采样 token 可能同时
包含可视点与不可视点。`q_i` 可以区分高可视 source、低可视 target 和混合 token，
并允许后续由硬阈值扩展为连续加权。计算只需要根据 `input_to_level` 做两次
scatter count，复杂度为 `O(N)`，与当前 hierarchy 接口完全兼容。

文档给出的 `q>=0.60` source、`q<=0.20` target 可以作为覆盖审计的初始分桶，但不应
直接固化为最终超参数。第一版 loss 更建议只监督 `q=0` 的 fully-invisible token，
避免把已经通过共享 token 接受 HPSD 的混合区域误认为完全无监督；验证有效后再逐步
扩展到 `(0,0.2]`。

### 2.3 Local Reallocation 确实能给不可视点直接梯度

当前方案的核心 loss 是：

```text
reference_i = sum_j p_ij * stopgrad(h_source_j)
L_i = 1 - cos(h_target_i, stopgrad(reference_i))
```

只要 loss 中的 `h_target_i` 没有 detach，就有：

```text
dL_i / dh_target_i != 0
dL_i / dF_H_target_i
    = dL_i/dh_target_i * d(prop_proj(F_H_target_i))/dF_H_target_i
```

因此梯度会经过 `prop_proj`、concat 层级特征和共享 backbone 到达不可视 token。这比
仅依赖 transformer 感受野的间接梯度更明确，也不需要为所有不可视点生成 1024 维
DINO prediction。

需要准确表述的是：它并非原始 DSP 的 feature reallocation。DSP 在带监督位置使用
由未监督 value 重建的特征，监督 loss 经重建矩阵反向流入未监督 value；当前 VRSR
则为不可视 target 构造停止梯度的可视 reference，再直接拉近 target。两者都扩大了
监督影响范围，但梯度路径和错误模式不同。VRSR 更接近可靠图一致性或软原型传播，
工程上更稳定、覆盖 target 更直接，但更容易产生特征平滑。

### 2.4 低维传播和分阶段启用合理

在 128D 空间做 source 检索、reliability 和 prototype 匹配，同时保持 HPSD 原生
DINO-1024 不变，是合理的职责分离。传播内部降维不等于压缩 HPSD teacher，也不会
影响最终 backbone 输出。文档采用 HPSD warm-up、local、reliability、global
fallback 的渐进顺序，也比一次加入全部 loss 更容易定位问题。

### 2.5 训练期分支、推理期删除可实现

VRSR 只需要训练时的 `dino_valid`、hierarchy、F2/F3/F4 concat 和少量 prototype。
测试时可以删除 `prop_proj`、prototype 和 VRSR loss，继续导出原有 HPSD projected
特征或直接保留 backbone。这与 HPSD 的预训练目标一致，不增加下游推理时间和显存。

## 3. 必须修正的算法问题

### 3.1 128D 传播空间当前没有语义锚定

文档假定高 `q` token “已通过 HPSD 获得可靠视觉语义”，但 HPSD 的监督实际定义在
patch 聚合结果：

```text
MLP(Aggregate_patch(F_H)) <-> DINO_patch
```

这不保证每个单独 token 的 `F_H` 已经逐点等价于 DINO，更不保证一个全新随机初始化的
`prop_proj(F_H)` 保留 DINO cosine 结构。HPSD warm-up 不会优化 `prop_proj`，所以在
Local start 的第一个 iteration，Top-K 依赖的是未经校准的投影空间。128D 随机投影
有机会近似保留 F_H 距离，但这只是经验假设，不能直接把 top-1 cosine 解释为语义
可靠度，也不能预设 `tau=0.70`。

推荐在 VRSR 启动前加入一个轻量的 propagation-space calibration。利用现有
token-patch edges，把可视 token 关联的多个 DINO patch 聚合为 token teacher：

```text
d_t = normalize(sum_p sqrt(point_count_tp) * dino_p)
u_t = normalize(R_fixed * d_t),       R_fixed: 1024 -> 128
h_t = normalize(prop_proj(F_H_t))
L_cal = 1 - cos(h_t, stopgrad(u_t))
```

`R_fixed` 可以是固定随机正交投影，或在代表性 DINO 样本上离线拟合的 PCA；它只服务
传播空间，不改变 HPSD 的 1024 维 teacher。校准阶段可对 `F_H` stop-gradient，只训练
约十万参数的 `prop_proj`，随后冻结或使用很小学习率。这样 source、target 和 prototype
都处于有明确视觉语义依据且相对稳定的 128D 空间。

如果不愿增加 `L_cal`，最低要求是先冻结 HPSD backbone，对当前随机/训练后
`prop_proj` 做可视 token retrieval probe，证明同类或相似 DINO teacher 的 token
确实在 128D 空间互为近邻，再启用传播。否则 prototype 的含义和 reliability 阈值都
缺少基础。

### 3.2 detach 语义必须在接口中彻底消除歧义

文档伪代码使用：

```text
T = h[target_mask]
topk_same_sample(T.detach(), S, ...)
weighted_cosine_loss(T[valid], reference[valid], ...)
```

这个写法本身可以正确反向，但正文“VRSR 全部 stop-gradient source
target/reference”容易被误解。正确规则应写成：

```text
target_train  = h[target_mask]           # 保留梯度，用于 loss
target_search = target_train.detach()    # 仅用于 Top-K 和 reliability
source_search = h[source_mask].detach()
reference     = build_reference(...).detach()
```

只有 `target_train` 保留计算图。source、neighbor index、similarity、reliability、
reference 和 prototype 都应停止梯度。必须为此写一个梯度单元测试：构造同时包含 source
和 target 的 toy batch，要求 target 对应的 backbone/prop projection 梯度非零，source
侧只通过 HPSD 获得梯度；将 HPSD 权重临时设为零后，source feature 不应从 VRSR loss
收到梯度。

### 3.3 q 高只代表覆盖强，不代表 teacher 纯净

`q_i` 衡量可见点比例，却没有衡量一个 token 是否跨越多个视觉边界。一个 `q=1` 的
token 可能同时关联屋顶边缘、树冠和阴影中的多个差异 patch。将其作为 source 会把
混合 teacher 传播给不可视点。

推荐把 source reliability 拆成三部分：

```text
visibility_i = q_i
purity_i = weighted_mean_p cos(dino_p, aggregated_dino_i)
support_i = valid point count or token-patch edge support
```

source 至少同时满足可视比例、DINO teacher purity 和最小支持点数。若保留 HPSD
student prediction，还可把可视 token/patch 的 HPSD residual 作为附加置信度。只靠
`q>=0.60` 不足以支撑可靠传播。

### 3.4 `valid=False` 的物理原因仍被混在一起

当前 correspondence 的 `valid` 同时受正射影像 coverage 和
`surface_only_valid` 影响。无影像覆盖点与正射遮挡点不是同一种学习问题：前者可能
在另一 tile/航带中有相似可视地物，适合 global guidance；后者常是墙面、林下地面
等系统性不可视结构，最容易被屋顶或树冠 source 错误覆盖。

VRSR 第一版可继续只依赖 `dino_valid`，但正式版本应增加：

```text
dino_image_covered
dino_surface_visible
dino_valid = covered & visible & patch_in_range
```

这两个布尔字段几乎没有资源负担，却能为 source/target 分组、误传播分析和下游指标
提供必要依据。global prototype 应优先服务“无覆盖但结构可匹配”的点，对被遮挡结构
采用更严格阈值和几何约束。

### 3.5 ALS 场景不宜把几何门控推迟到 v2

文档建议 MVP 先不加 XYZ/normal gate，但 ALS 不可视性恰好与空间结构高度相关。
树冠与林下地面 XY 重叠，屋顶与立面空间接近，而它们可能在 HPSD 语义空间中因共享
上下文表现相似。纯语义 Top-K 很可能产生系统性错误，不是普通随机噪声。

第一版不需要复杂几何网络，但至少应在候选生成时加入低成本描述：归一化高度、局部
高差、intensity、echo number/number of returns 和 XY/3D 距离分桶。推荐并行保留
“局部几何候选”和“同 tile 语义候选”，再用轻量 gate 合并，而不是只在错误出现后
补救。

### 3.6 Global Prototype 不是严格的跨样本 feature reallocation

32–64 个 EMA prototype 很轻量，可以补充 anchor-poor tile，但它们只描述由高可视
source 形成的全局模式。不可视独有结构如果从未出现在 source 中，prototype 不可能
创造对应监督；scene balancing 也只能减少单 tile/高密度区域的支配，不能恢复缺失
语义模式。

此外，文档对参考工作的引用应保持准确：

- [AIScene](https://openaccess.thecvf.com/content/CVPR2025/html/Liu_Exploring_Scene_Affinity_for_Semi-Supervised_LiDAR_Semantic_Segmentation_CVPR_2025_paper.html)
  的核心实现包括 teacher-student 伪标签、scene 内 point erasure 一致性和多场景 patch
  mixing，并不是与 DSP 相同的跨样本 token affinity。
- [AADNet](https://ojs.aaai.org/index.php/AAAI/article/view/32680) 通过
  label-aware downsampling 与动态熵梯度校准处理稀疏且非均匀标注；scene-balanced
  source sampling 是本方案根据其“非均匀监督导致梯度偏置”结论作出的工程推演，
  不是 AADNet 原模块。
- [RAC-Net](https://arxiv.org/abs/2303.05164) 使用 prediction confidence 和 model
  uncertainty 衡量伪标签可靠性；VRSR 的 top-1/margin/entropy 是轻量替代，不应写成
  等价复现。
- [DGNet](https://proceedings.neurips.cc/paper_files/paper/2024/hash/38d6af46cca4ce1f7d699bf11078cb84-Abstract-Conference.html)
  使用 moVMF 分布、可靠聚类初始化和交替优化；EMA k-means prototype 只是从其分布
  指导思想得到的简化版本。

因此 global branch 可以称为 `cross-sample prototype guidance`，但不应称为完成了
DSP/AIScene 的 cross-sample feature reallocation。若后续要验证真正跨 tile 的细粒度
迁移，可增加一个 stop-gradient memory queue：保存其他 tile 的高置信 key/reference，
当前不可视 target 对 queue 做 chunked Top-K，并与 prototype 版本做精度/成本对照。

## 4. 与当前工程的兼容性

### 4.1 LitePT-v1m4 与 PT-v3m4

两种 backbone 都通过相同的 `HierarchyLevel(point, input_to_level, level)` 返回
fine-to-coarse hierarchy，且 HPSD 已有 `fuse_hierarchy_features(hierarchy,
distill_level)`。VRSR 在 level 2 使用 F2/F3/F4 concat，因此算法不依赖具体 backbone。
LitePT concat 通道为 900，PTv3 为 1008，只影响 `prop_proj` 的输入维数。

原 DOCX 在推理阶段只写了 LitePT encoder，这一点应改为 LitePT/PTv3 均支持。PTv3
当前训练是 100 epoch，而 LitePT 是 10 epoch 且 dataset loop=10，不能把文档中的
“epoch 1–3、4–5、6–7、8–10”直接用于两份配置。应把阶段定义为总 optimizer step
比例，例如：

```text
0%–30%: HPSD + prop-space calibration
30%–50%: HPSD + local
50%–70%: HPSD + reliable local
70%–100%: optional global fallback
```

若从已经收敛的 HPSD checkpoint 启动，可跳过大部分 HPSD warm-up，但仍需完成新
`prop_proj` 的校准。

### 4.2 HPSD 模型包装方式

当前 HPSD 可以通过 `return_point=True` 返回 hierarchy，但内部已经构造了一次
`distill_feat=concat(F2,up(F3),up(F4))`，返回结果中没有该张量。外部 wrapper 若从
hierarchy 再调用一次 `fuse_hierarchy_features()`，会重复分配约 `T×C_fused` 的 concat
activation。

推荐仅增加一个不改变算法的接口：

```text
HPSD.forward(..., return_context=True)
context = {
    hierarchy,
    level,
    distill_feat,
    token_patch_edges,
}
```

VRSR wrapper 消费 context 后必须把非标量对象从模型输出删除，只向 Trainer 返回
`loss` 和短日志项。这样 backbone、concat 和 token-patch unique 不会重复执行。

### 4.3 Trainer 与日志

包装模型可返回：

```text
loss = hpsd_loss + lambda_r * vrsr_loss
hp, vr, src, tgt, cov, ign
```

字段采用简称，避免恢复冗长日志。Trainer 仍只对 `loss` backward，无需改动
`tools/train.py`。VRSR prototype 若采用 DDP online update，必须在各 rank 间同步
assignment count 和 prototype sum，否则每张 GPU 会形成不同的全局原型。同步量只有
`K_proto×128`，成本很低。

### 4.4 HPSDFeatureTester 与 checkpoint

wrapper 必须透传 `return_point_feature=True`、`feature_source` 和
`normalize_feature`，或者 HPSDFeatureTester 显式 unwrap `.hpsd`。Tester 当前还读取
`projector_in_channels`，wrapper 需提供同名 property。测试导出不应加载 prototype
队列，也不应要求 correspondence。

引入 wrapper 后 checkpoint key 可能从 `backbone.*` 变为 `hpsd.backbone.*`。应提供
显式的 HPSD-only/backbone 提取与加载转换，而不是依赖 `strict=False`。现有 HPSD
checkpoint 则可以加载到 wrapper 的 `.hpsd` 子模块，新 VRSR 参数单独初始化。

### 4.5 LasImageDataset 与 transforms

当前 `dino_valid`、`dino_patch_index` 都属于 `index_valid_keys`，GridSample、随机丢点
和 Crop 会同步更新，`CompactDinoPatches` 也不会破坏 q 的计算。因此 Local MVP
不需要改变现有数据读取。需要注意 `q_i` 表示当前增强和 GridSample 后输入点的可视
比例，而不是原始 LAS 的固定比例；RandomDropout 会让 q 有轻微随机扰动。若实测波动
明显，可以在 dropout 前保存 point visibility weight，或只把这种扰动视为正则化。

## 5. 资源消耗评估

### 5.1 参数量

`LayerNorm(C)+Linear(C,128)` 的参数量约为：

| Backbone | C_fused | prop_proj 参数 |
| --- | ---: | ---: |
| LitePT-v1m4 | 900 | 约 117K |
| PT-v3m4 | 1008 | 约 131K |

即便考虑梯度、AdamW 一阶/二阶状态和 AMP master copy，也只有约 2–3 MiB。32–64 个
128D prototype 小于 0.04 MiB，参数存储可以忽略。若加入固定 DINO `1024→128`
投影，它不需要 optimizer state。

### 5.2 激活和 Top-K 峰值

以 `T=10k–14k` level-2 token、128D BF16 embedding 为例，完整 `h` 约占
2.4–3.4 MiB，连同归一化、梯度和 linear 中间量通常在十余 MiB。若 target 占
70%，target/reference 各自约 1.7–2.4 MiB。

文档中“local 只保存 `T_target×K_local`”只描述 Top-K 选择后的常驻结果，不是搜索
峰值。精确 chunked Top-K 仍会临时生成：

```text
similarity_chunk: [chunk_targets, M_src]
```

当 `chunk_targets=8192`、`M_src=1024` 时，该矩阵 BF16 约 16 MiB，FP32 约
32 MiB。搜索使用 detached embedding，不保留 autograd 图，因此可以在每个 chunk
结束后释放。Top-K 的 `8192×16` int64 index 和相似度约再增加 1–2 MiB。不同 CUDA
topk kernel 还可能分配 workspace，必须以 `torch.cuda.max_memory_allocated()` 实测。

避免 wrapper 重复创建 F_H 很重要。一个 `10k×900` BF16 concat 本身约 17 MiB，
加上反向图后可能比整个 prototype 模块更贵。复用 HPSD context 可以直接省去这部分
重复开销。

综合估计，正确 chunk、复用 F_H 且不对 source 建反向图时，VRSR-Lite 峰值显存增量
大约为 30–100 MiB。对现有完整 HPSD 通常属于约 5%–15% 量级，达到文档“低于
independent multi-level 约 +20%”的目标具有较高可行性，但在真实 LitePT 大 batch
和 PTv3 上仍需分别测量，不能仅凭张量理论值验收。

### 5.3 计算量

若每 tile 有 10,000 个 target、最多 1,024 个 source、embedding 为 128 维，完整
精确相似度约需要：

```text
10,000 × 1,024 × 128 = 1.31B multiply-accumulate / tile
```

矩阵乘法本身适合 GPU，但 Top-K 排序、按样本 ragged slicing 和多个 tile 循环会带来
额外 kernel 开销。LitePT 当前单 GPU micro-batch 可能大于 1，这一成本会随 tile 数
近似线性增加；PTv3 单 GPU micro-batch 可为 1，单步更容易控制。

建议增加 `max_targets_per_sample=2048–4096`，按 q、空间网格和随机轮换采样 target。
这样每轮只监督部分不可视 token，但多个 epoch 可获得覆盖，计算量下降 2.5–5 倍。
只有在消融证明全 target 明显更好时才取消上限。MVP 使用 PyTorch chunked matmul
和 `topk` 即可，不建议为了第一版引入 FAISS 依赖。

Prototype 匹配最多为 `T_fallback×64×128`，相对 local 搜索很小；DDP prototype
all-reduce 也可忽略。总体训练时间预计增加约 5%–20%，主要取决于 target 上限、
micro-batch 和 topk kernel，而不是参数数量。

### 5.4 推理资源

VRSR、propagation calibration 和 prototype 全部训练期删除。只要 wrapper 正确透传
HPSD/backbone 导出路径，推理参数、FLOPs 和显存增量为零。

## 6. 推荐的修正版 VRSR-Lite

推荐的最小可靠流程如下：

```text
1. HPSD 保持当前 concat + MLP + patch cosine，不改变 loss。
2. 根据 input_to_level 统计 token q、fully-invisible 和 mixed 比例。
3. 由 token-patch edges 聚合可视 token DINO teacher，计算 purity。
4. 用固定 1024->128 teacher 投影校准 prop_proj；先 detach F_H，只训练小头。
5. source = 高 q + 高 purity + 足够 support 的 token。
6. target_train = fully-invisible h，保留梯度。
7. target_search/source/reference/reliability 全部 detach。
8. chunked same-sample Top-K + 最低成本 geometry/echo gate。
9. 只对可靠 target 计算低权重 cosine consistency。
10. Local 通过 Go/No-Go 后，才加入 scene-balanced EMA prototype fallback。
```

修正版 loss 可写为：

```text
L = L_HPSD + lambda_cal * L_cal + lambda_R * L_local

L_local = sum_i w_i * (1 - cos(h_i_train, stopgrad(ref_i))) / sum_i w_i
w_i = (1-q_i) * reliability_i * purity_ref_i
```

`L_cal` 在传播空间稳定后可以关闭或只对 prop_proj 保留很小权重。`lambda_R` 从
0.01–0.05 范围开始，不能直接假定 0.05 最优。Global prototype loss 与 local
使用同一形式，但必须有更高阈值并分开记录 coverage。

## 7. 实现难度与开发顺序

### 7.1 实现难度判断

| 模块 | 难度 | 主要工作 |
| --- | --- | --- |
| q 与覆盖审计 | 低 | scatter count、分桶统计、单元测试 |
| token DINO teacher/purity | 中 | 复用 token-patch edges 做反向聚合 |
| prop-space calibration | 中 | 固定 teacher 投影、阶段冻结/解冻 |
| Local Top-K | 中 | 按 batch 分组、chunk、target/source 采样 |
| Reliability 与 geometry gate | 中 | 阈值标定、候选特征与诊断 |
| Global prototype | 中高 | EMA assignment、DDP 同步、空原型重置 |
| wrapper/tester/checkpoint | 中 | context 透传、标量输出、key 转换 |
| 真正 cross-sample queue | 高 | 检索、陈旧特征、跨域与分布式一致性 |

Local MVP 不涉及 CUDA 自定义算子，完全可以使用 PyTorch 和 torch_scatter 实现。
主要风险不在代码能否运行，而在传播空间是否真的有语义、可靠度是否可校准，以及
不可视点提升是否来自正确传播而非过度平滑。

### 7.2 推荐开发顺序

1. 冻结当前 HPSD checkpoint，新增 coverage audit，报告 raw point 与 level 0–4 的
   visible / fully-invisible / mixed token 比例。
2. 实现 token DINO teacher、purity 和 `prop_proj` calibration，并先做可视 token
   retrieval probe。
3. 实现 fully-invisible target、scene source cap、chunked Top-K 和严格 detach
   单元测试，不加入 prototype。
4. 加入最低成本的 height/echo/geometry gate，运行 HPSD 与 HPSD+Local 对照。
5. 只有 Invisible Linear Probe、低标注 fine-tune 或不可视结构子集显著提升且 Visible
   基本不下降时，再实现 reliability margin/entropy。
6. Local+Reliability 通过后，再实现 DDP-safe global prototype fallback。
7. 最后将 queue-based cross-sample Top-K 作为 prototype 的精度/成本对照，而不是
   第一版必需功能。

## 8. 验收与消融建议

最小消融应包含：

| 版本 | 目的 |
| --- | --- |
| HPSD | 固定强基线 |
| + uncalibrated Local | 检验原文档假设，作为对照而非推荐最终版 |
| + calibrated Local | 验证传播空间语义校准是否必要 |
| + purity + geometry gate | 验证 ALS 系统性错配控制 |
| + reliability | 验证覆盖/精度平衡 |
| + global prototype | 验证 anchor-poor tile 是否额外受益 |
| + cross-sample queue | 判断 prototype 是否损失过多跨 tile 细节 |

除了 DOCX 已提出的 All/Visible/Invisible Linear Probe，还应增加：

- fully-invisible token 的 VRSR-only 梯度范数；
- HPSD 与 VRSR 在共享 F_H/backbone 上的梯度 cosine；
- source teacher purity、target 被选择覆盖率和每个 source 的使用次数；
- 按 `image_covered/surface_visible` 分组的指标；
- 屋顶/立面、树冠/林下地面之间的错误匹配率；
- 每 iteration 训练时间和 `max_memory_allocated` 增量；
- prototype occupancy、跨 rank 差异和重置次数。

Go 条件应为：不可视子集指标有稳定提升，可视子集基本不下降，HPSD loss 没有持续
恶化，传播空间 retrieval 明显优于随机，attention/source 使用不塌缩，且峰值显存
增量控制在基准约 15% 内。若 uncalibrated Local 无收益，不应直接加入 prototype；
先检查传播空间语义、q/purity 和梯度路径。

## 9. 最终判断

HPSD + VRSR 的总体分工是成立的：HPSD 负责将可靠正射视觉语义注入可视 ALS 表示，
VRSR 负责把已经进入三维空间的语义以受控方式扩展到不可视 token。其 Local 分支比
完整 DSP 更适合当前万级 token 和 encoder-only 工程，资源预算较低、实现不依赖新
CUDA 算子、训练后可以完全删除，因而适合作为下一步 MVP。

不过，当前 DOCX 更像一份方向正确的融合设计，而不是可以逐行编码的最终规格。
propagation embedding 的语义校准、detach 边界、source purity、不可视原因拆分和
Top-K 真实峰值必须在实现前补齐。Global prototype 是有价值的低成本分布先验，但
不能替代真正的跨样本 feature reallocation，也不应在 Local 尚未证明有效前进入主线。

在完成上述修正后，本方案的兼容性和可实现性较高，预计新增训练显存在 30–100 MiB、
训练时间增加约 5%–20%，推理零额外成本。推荐实施顺序是
`覆盖审计 → 传播空间校准 → Local MVP → Reliability → Global Prototype → 可选
Cross-Sample Queue`，而不是一次性实现 DOCX 中全部融合模块。

## 参考资料

- Jiacheng Wei et al., [Dense Supervision Propagation for Weakly Supervised
  Semantic Segmentation on 3D Point Clouds](https://arxiv.org/abs/2107.11267).
- Chuandong Liu et al., [Exploring Scene Affinity for Semi-Supervised LiDAR
  Semantic Segmentation](https://openaccess.thecvf.com/content/CVPR2025/html/Liu_Exploring_Scene_Affinity_for_Semi-Supervised_LiDAR_Semantic_Segmentation_CVPR_2025_paper.html),
  CVPR 2025.
- Zhiyi Pan et al., [Point Cloud Semantic Segmentation with Sparse and
  Inhomogeneous Annotations](https://ojs.aaai.org/index.php/AAAI/article/view/32680),
  AAAI 2025.
- Zhonghua Wu et al., [Reliability-Adaptive Consistency Regularization for
  Weakly-Supervised Point Cloud Segmentation](https://arxiv.org/abs/2303.05164).
- Zhiyi Pan et al., [Distribution Guidance Network for Weakly Supervised Point
  Cloud Semantic Segmentation](https://proceedings.neurips.cc/paper_files/paper/2024/hash/38d6af46cca4ce1f7d699bf11078cb84-Abstract-Conference.html),
  NeurIPS 2024.

