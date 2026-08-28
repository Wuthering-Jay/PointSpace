# 面向跨模态可观测性差异的机载 LiDAR 无标签预训练方案评估

## 1. 评估结论

本文评估的对象是仓库根目录中的《面向跨模态可观测性差异的机载LiDAR视觉基础模型无标签预训练方案》。评估同时以当前 PointSpace 的 HPSD、VRSR、`LasImageDataset`、LitePT-v1m4 和 PT-v3m4 实现为工程约束，而不是只讨论抽象方法。

总体结论是：**建议停止把 VRSR 作为主线继续扩展，并将新方案作为下一条研究主线，但新方案应当“有条件接收、经过一次关键修订后再实现”**。新方案最有价值的部分不是增加了多少 loss，而是把研究问题从一般的弱监督传播重新定义为正射影像与机载 LiDAR 的跨模态可观测域不一致。这一问题定义符合 ALS 的真实观测机制，也比 VRSR 的 source-to-target 检索更容易形成清晰、具有遥感特色的论文叙事。

新方案目前最大的理论漏洞位于 Geometry-guided Observation Simulation 和 Contextual Semantic Completion（CSC）之间。PDF 认为从 CSC 输入中移除 F2、只使用上采样的 F3/F4，就能够防止目标 token 的局部身份泄漏。这一论断在当前 backbone 中并不成立：F3/F4 仍然由包括目标 token 在内的低层特征经过 pooling、卷积或注意力生成。因此，如果只是取消目标 token 的 HPSD loss，再用 F3/F4 预测同一位置的 DINO teacher，CSC 很可能退化成另一套深层 projector，并没有真正模拟输入观测缺失。

这个漏洞可以利用现有代码中已经具备的 `mask_token` 能力修正。推荐在每次 forward 之前，从高可信可视点中按 ALS 结构块选择 simulated-missing 点，并通过 backbone embedding 的 mask token 真正遮蔽这些点的输入属性；坐标和点的存在性仍保留，使网络能够依靠三维几何和周围上下文恢复语义。随后，未遮蔽的高可信点承担 HPSD anchor loss，被遮蔽但仍拥有真实 DINO 对应的点承担 CSC loss，真实不可视点不接受伪造 teacher，但参与共享 encoder 的上下文计算。该修订保留一次 encoder 前向、一次 backward 和一个连续 optimizer/scheduler，因此符合用户希望的一次训练要求。

建议的最终主线不是原 PDF 中四个模块同时上线，而是按重要性收敛为：

1. Observation-weighted HPSD，负责可靠可视表面的视觉锚定；
2. Geometry-guided input masking，负责构造真实的、可监督的视觉缺失代理任务；
3. Contextual Semantic Completion，负责从被遮蔽的三维输入恢复 DINO-shaped 表示；
4. Relational distillation 仅作为可删除的后续增强，不进入第一版 MVP。

这里的“阶段”是研发和消融顺序，不是模型训练时需要依次加载多个 checkpoint。最终主模型应在同一个训练 run 内通过连续 curriculum 调整 mask rate 和 CSC loss 权重。

## 2. 为什么可以弃用 VRSR

### 2.1 VRSR 解决问题的方式与当前目标不再一致

VRSR 先在可视 source token 上学习一个 128 维 teacher 对齐空间，再对真实不可视 target token 检索同 tile 的 Top-K source，以 detached source 表示构造软目标。该机制具有明确的 target 梯度路径，并且此前已经验证了 LitePT、PTv3、真实数据构图和显存开销。但是，它依赖“先使传播空间可用，再开启 local propagation”的训练逻辑。即使工程上可以把两个阶段放进同一个调度器，方法叙事仍围绕校准、检索、传播和可靠度筛选展开，超参数之间也存在明显耦合。

更重要的是，VRSR 默认不可视 token 可以从可视 token 集合中找到合理的语义近邻。对于屋顶与立面、树冠与林下地面、裸露地表与被植被遮挡地面，这个假设并不总成立。Top-K 相似度是在尚未充分训练的三维空间中计算的，错误匹配一旦形成，就可能通过自增强的方式持续强化。后续如果再加入 margin、熵、几何约束、prototype 或 queue，系统会变得更复杂，却仍然无法从根本上消除 source 语义不完备的问题。

新方案不再给真实不可视点构造 source-derived pseudo target，而是在有真实 DINO teacher 的可视区域模拟缺失状态，训练共享的三维条件预测能力。从研究假设上看，它把“不可见点应当像哪个可视点”改成了“在缺少局部观测属性时，三维结构和上下文能否恢复视觉语义”。后者更容易通过受控 masking、可见性退化曲线和分层 probe 验证，也不需要 KNN、graph、prototype 或跨样本队列。

### 2.2 应当弃用，但不建议本轮直接物理删除

“弃用”与“立即删除”应区分处理。VRSR 已经形成了可运行代码、配置、测试和资源基准，它目前仍是新方案的重要对照组，也是证明“显式传播为什么不是最终选择”的实验依据。直接删除会使旧 checkpoint 无法按注册名构建，也会丢失已验证的 token visibility、teacher-to-token 聚合和 chunked Top-K 等通用算子。

建议先将 VRSR 标记为 deprecated，停止继续增加功能并从默认配置、主文档和推荐训练命令中移出。新方案完成最小可运行验证并确认 HPSD baseline 不受影响后，再执行物理删除。删除前应把仍有通用价值的可见性统计和 teacher 聚合函数移动到 HPSD 或新的 observation 模块，保留一个代码 tag 或归档分支以便复现实验。这个顺序不会要求重新训练 VRSR，只是避免在替代方案尚未通过单元测试前破坏仓库的可回退性。

## 3. 新方案的科学价值

### 3.1 问题定义是成立的

方案以两个传感器观测算子表示正射影像和 ALS，并指出影像可观测支持域通常只是 ALS 支持域的子集。这一形式化虽然简单，但抓住了机载场景与室内 RGB-D、车载多相机数据的根本差异。正射影像对屋顶、树冠和裸露地表提供外观先验，而 ALS 同时包含立面、冠层内部、林下地面和多回波结构。因而，“投影坐标合法”与“物理上被影像观测”必须是两个不同概念。

此前对湖北真实数据的完整审计也支持这个问题并非边缘情形：49,396,710 个变换后点中，点级 `image_valid` 约为 39.65%；在 7,088,569 个 level-2 token 中，42.65% 完全不可视，40.93% 为可视与不可视混合，只有 16.42% 完全可视。因此，仅依靠可视点 HPSD 并假设共享网络会自然解决全部不可视结构，并不是充分论证。

### 3.2 “无标签”表述基本合理，但不宜写成完全无监督

预训练不使用人工语义标签，因此称为无标签预训练是合理的。不过训练仍接受冻结 DINO 产生的跨模态 teacher supervision，严格地说更接近 teacher-guided unsupervised representation pretraining 或 cross-modal distillation，而不是完全没有监督信号的纯自监督学习。正式论文应明确“unlabeled”指没有人工三维类别标签，避免审稿人把 DINO teacher 视为与“unsupervised”措辞冲突。

### 3.3 论文创新性强于单纯增加一个蒸馏 loss

当前方案的论文潜力主要来自三个层次。第一，提出 orthophoto-ALS cross-modal observability mismatch，并用真实 ALS 统计证明该问题具有结构性。第二，将 true-ortho 可见性、垂向穿透、回波和配准不确定性转化为蒸馏可信度，而不是把所有投影点等同处理。第三，用符合 ALS 结构的 masked cross-modal prediction 训练共享三维 encoder，而不是为真实不可视点分配伪特征。

这种定位与 DITR 的直接 VFM-to-3D 蒸馏、ScaLR 对 backbone/teacher/data scale 的强调以及 Sonata 对三维 geometric shortcut 的分析能够形成清晰关系。需要注意，方案不能声称已经证明真实不可视点获得了 DINO 语义；它提出的是一个可检验的泛化假设，必须由不可视子集的 frozen probe、few-shot 和覆盖率退化曲线验证。

## 4. 对四个模块的逐项评估

### 4.1 Observation Reliability：值得保留，但需要重新定义字段语义

PDF 使用

```text
q_i = v_i * c_i_reg * c_i_sem
```

表达物理可见性、配准可信度与视觉语义纯度的共同作用，这一思想是正确的。工程上不再拆分多个硬布尔字段：`valid` 本身统一表达点是否拥有可用于监督的正射像素/patch 对应，范围外、无数据覆盖、遮挡或非表面点均为 False。连续可信程度由独立的 q 表达：

```text
image_valid         是否允许该点提供物理可信的影像 teacher，作为统一硬门控
image_observability 点的连续可观测可信度 q，范围 [0, 1]
```

第一版不应同时实现 `v`、`c_reg` 和 `c_sem`。最稳妥的 MVP 是复用 `tile_las_image.py` 已经计算的局部 DSM 上包络，额外输出连续的表面可信度：

```text
delta_z_i = z_surface_i - z_i
q_surface_i = exp(-max(delta_z_i - tau, 0) / sigma_z)
```

其中 `tau` 可以复用 `surface_z_tolerance`，`sigma_z` 控制容差之外的平滑衰减。对于完全无影像覆盖的点，`image_valid=False`，即使几何上接近上表面也不能成为 teacher。对于 simulated-missing 样本，只从 `image_valid=True` 且 q 高的点中选择，因为这些点拥有物理可信的真实 DINO target。

PDF 中的 `rho_above` 与 `delta_z` 高度相关，同时使用容易重复惩罚林下点。建议先只采用 `delta_z`，在发现同一高度差下仍无法区分稀疏穿透结构时，再加入经过归一化的 above-occupancy。当前 `LasDataset` 只向网络暴露两维 `echo=(is_first, is_last)`，没有保留完整的 return number 和 number of returns；如果 q 在线计算依赖精细回波顺序，需要新增字段。更简单的办法是在 tile 生成阶段直接利用原 LAS 维度离线计算 q，并把结果写入 correspondence Safetensors，这样不会改变 backbone 的 6 维输入。

`c_reg` 不适合在 MVP 中用“靠近 DINO patch 边界”近似。固定 patch 边界并不等于配准误差，真实地物边界反而是最有信息的位置。若后续实现，应依据局部二维特征梯度与三维高度边缘在多个小位移下的对齐稳定性估计；如果估计成本或稳定性不满足要求，宁可不使用 `c_reg`。`c_sem` 也不能由单个 patch 的一个 1024 维向量直接得到“内部纯度”，可以使用邻域 DINO 相似度、同 token 多 patch 一致性或增强视图稳定性作为代理，但应作为后续消融而不是默认事实。

### 4.2 Observation-conditioned HPSD：兼容性最高，应当先实现

当前 HPSD 先由逐点 `input_to_level` 和 `image_patch_index` 构造去重 token-patch edges，再按 `sqrt(point_count)` 聚合 token feature，最后只对实际使用的 patch 生成 1024 维 prediction。把连续 q 接入该流程不需要改变 projector，也不需要创建逐点 `[N,1024]` 张量。

推荐明确区分“特征聚合权重”和“patch loss 可靠度”，避免同一个 q 在两个位置无意中平方。可以定义：

```text
a_tp = sqrt(n_tp) * mean(q_i | i supports edge t-p)
r_t  = n_valid(t) / n_total(t)
```

`a_tp` 用于把 token feature 聚合到 patch，`r_t` 则反映一个 token 内真实可视点的占比。第一版可以只使用 `a_tp` 完成 observation-weighted aggregation，并继续按样本平衡平均 patch loss；如果实验表明低支持 patch 仍造成噪声，再增加截断后的 patch-level reliability。不要在没有消融的情况下同时对 feature aggregation 和 final loss 重复乘完整 q。

当 `q=1` 且 mask rate 为 0 时，新实现必须数值等价于当前 HPSD。这个等价性测试比一般的“能够 forward”更重要，因为它保证新分支不会破坏已经取得的 concat-HPSD 结果。

### 4.3 Geometry-guided Observation Simulation：原稿概念正确，当前实现描述不充分

原稿提出从高可信可视集合中划分 anchor 和 simulated-missing，并让后者保留真实 DINO teacher。这个实验设计非常好，因为真实不可视点没有 teacher，无法直接验证“预测是否正确”；高可信可视点经过受控遮蔽后，仍有真实 teacher 可作为训练和评价目标。

但是，**只在 loss 中把某些点从 HPSD 移到 CSC，不构成真正的 observation simulation**。DINO 从未作为 student 输入，因此“取消直接视觉锚定”仅仅是切换监督 head。若 CSC 仍读取由目标点完整 LiDAR 属性生成的 F3/F4，它学习的仍然是该点的 3D-to-DINO regression，只是投影层不同。

推荐使用当前 backbone 已经具备的输入 mask 机制。`LitePT-v1m4` 和 `PT-v3m4` 分别继承 `litept_v1m3_utonia.py` 与 `point_transformer_v3m3_utonia.py`；两者的 embedding 都支持 `mask_token=True`，并在输入 `Point` 含有 `mask` 时，用可学习 mask token 替换相应点的 embedded feature。坐标仍然保留，所以模型知道三维结构和位置，但不能直接使用被 mask 点的强度、回波等局部属性。这与方案希望学习的 `3D geometry/context -> visual semantics` 更一致。

推荐的单次 forward 路径是：

```text
高可信 image_valid 点
        |
        +-- 未选中 mask --> HPSD anchor edges --> patch-set DINO loss
        |
        +-- 结构化选中 --> embedding mask token --> F3/F4 --> CSC DINO loss

真实 image_valid=False 点
        |
        +-- 不构造 DINO target，但正常进入 encoder，并参与周围 token 的上下文计算
```

Mask 应在模型 forward 内根据当前已经完成数据增强、GridSample 和 SphereCrop 的点生成，而不是在 DataLoader worker 中提前固定。这样 curriculum 可以在同一个训练 run 内把 mask rate 从 0 平滑升到目标值，也能确保每个样本保留足够 anchor。按 XY column、局部高度结构或序列化 block 进行向量化分组即可，不需要 KNN。

第一版只实现一种结构化 mask：在含有足够高可信点的 XY block 中成片选择点，并保证每个样本至少保留 60%-70% 的可靠 anchor。Vertical-column、boundary-aware 和 large block 三种模式不应同时上线，否则超参数和归因成本会迅速增加。建议先比较 random point mask、random block mask 和 geometry-guided vertical block mask 三项。

### 4.4 Contextual Semantic Completion：经过真实输入 masking 后才成立

在修订后的设计中，被 mask 点的输入属性已经由 learned mask token 替代，因此使用 F3/F4 预测其真实 DINO teacher 具有明确意义。推荐继续排除 F2，使用：

```text
C_t = Concat(upcast(F3)_t, upcast(F4)_t)
Z_t = MLP_CSC(C_t)
L_CSC = mean_t [1 - cosine(normalize(Z_t), stopgrad(D_t))]
```

这里的 `D_t` 不能含糊地写成“该 token 的 DINO 特征”。一个 level-2 token 可能通过多个原始点关联多个 patch，因此应沿用 HPSD 的多对多关系，只对 simulated-missing 点支持的 token-patch edge 聚合 teacher。推荐先按 edge support 和 q 聚合 patch teacher 到 token，再对 token teacher L2 归一化。没有足够高可信 masked support 的 token 不计算 CSC。

LitePT 配置中 F3+F4 通道数为 `252+504=756`，PTv3 中为 `288+576=864`。一个 `LayerNorm -> Linear(hidden=1024) -> GELU -> Linear(1024)` 的 1024 维 CSC projector 分别约增加 1.82M 和 1.93M 权重，参数规模并不大。为了保持与当前 HPSD 一致，第一版可以保留原生 1024 维 DINO teacher，不必预先压缩。

必须谨慎描述 CSC 对真实不可视点的作用。真实不可视点没有 `L_CSC`，因此该方法**不能保证每一个不可视 activation 都得到直接 teacher gradient**。它们受益于两条间接路径：一是它们作为三维上下文参与邻近 masked-visible token 的深层计算，相关 activation 可能获得梯度；二是 encoder 参数在所有点间共享，学到的 3D-to-visual mapping 会迁移到真实不可视区域。最终是否真的改善不可视表示是实验问题，而不是由 loss 形式自动保证的结论。

### 4.5 Relational Distillation：理论合理，但不应进入 MVP

DINO 空间的相对关系可能比绝对 1024 维回归更稳定，局部 cosine relation loss 作为轻量正则具有合理性。不过 HPSD 和 CSC 已经同时约束绝对方向，关系 loss 的边际收益并不确定。若 naive 实现 `[M,K,1024]` pair tensor，还会引入显著临时显存；必须分块计算或只保存标量 cosine。

因此建议第一轮实现完全不包含 relation loss。只有 Observation-HPSD 与 masked CSC 已在 frozen probe 和低可观测子集上产生稳定增益后，才增加局部关系消融。若加入，应限制在同 tile、同结构块、每个 token 8 个以内 pair，并保持 `lambda_rel` 很小。Relation loss 无收益时应能从最终方法中干净删除，而不影响主要创新。

## 5. 修订后的统一目标

令 `A` 为未遮蔽的高可信 anchor point，`M` 为从高可信可视点中采样并在输入 embedding 被遮蔽的 simulated-missing point。两者互斥，真实不可视点既不属于 A，也不属于 M。推荐目标为：

```text
L = L_obs_hpsd(A) + lambda_c(s) * L_csc(M)
```

其中 `s` 是当前全局训练进度。可选的关系项只在后续消融中写成：

```text
L = L_obs_hpsd + lambda_c(s) * L_csc + lambda_r(s) * L_rel
```

为了保持一次训练，推荐使用同一个 optimizer、scheduler 和 checkpoint 链路：

```text
0% - 10%  : mask_rate = 0，lambda_c = 0，纯 HPSD 稳定视觉锚点
10% - 20% : mask_rate 线性升至 0.30，lambda_c 线性升至 0.20
20% - 100%: 固定目标 mask rate 与 lambda_c，联合训练
```

这是一个训练 run 内的 curriculum，不重置 optimizer，不重新加载 checkpoint，也不需要先导出三维特征。为了支持断点恢复，当前进度、mask rate 和 loss weight 必须由 epoch/global step 可重建，而不能只保存在临时 Python 状态中。

## 6. 与当前代码的兼容性

### 6.1 数据生成

`utils/tile_las_image.py` 已经在生成 correspondence 时计算局部表面 DSM，并可通过 `surface_cell_size`、`surface_radius` 和 `surface_z_tolerance` 得到硬 `surface_visible`。因此最经济的修改是在同一遍计算中输出连续 `observability`，避免训练时反复做柱体统计。Correspondence Safetensors 建议增加：

```text
observability: [N], float16
```

如后续需要区分影像覆盖和表面可视性，再增加 `covered: [N], bool`。第一版没有必要把所有中间量 `delta_z`、`rho_above`、`c_reg` 都写入文件；关键是保证 q 的生成参数写进 Safetensors metadata，以便复现。

### 6.2 数据集与变换

`pointspace/datasets/las_image.py` 将 `observability` 读取为点级字段 `image_observability`，并加入 `IMAGE_POINT_KEYS`，从而在 RandomDropout、GridSample、SphereCrop 等点采样操作中与点同步。`CompactImagePatches` 保持 patch 压缩逻辑，因为 simulated-missing 点仍需要对应的 teacher patch。

Mask 在模型内生成，不新增持久化的随机 `mask` 文件。模型收到当前 batch 的 `coord/grid_coord`、`offset`、`image_valid` 和 `image_observability` 后，按样本生成 simulated mask，并在调用 backbone 之前把它放入 input dict 的 `mask`。这样两种 backbone 可以直接使用已有 embedding mask token。

### 6.3 Backbone

现有 `LitePT-v1m4` 与 `PT-v3m4` 不需要为了 MVP 再建立 m5 版本。它们继承的 m3 实现已经支持构造 learned mask token；配置中当前只是设置了 `mask_token=False`。新配置把它改为 `True` 即可。

只有当后续实验要求“在 level-2 之后再 mask token feature”，才需要为两种 backbone 暴露 stage hook。MVP 先使用输入 embedding mask，因为它代码侵入更低，而且确实遮蔽了 intensity/echo 等局部属性。坐标保留并不是泄漏错误，而是让网络用 ALS 几何完成语义预测；这与真实不可视点在推理时仍拥有三维坐标的条件一致。

### 6.4 HPSD 模型

推荐新增一个包装模型，例如 `OC-HPSD-v1m1`，继承 `HierarchicalPatchSetDistiller`，保持以下 state dict 前缀不变：

```text
backbone.*
student_projector.*
```

新增参数只位于 `completion_projector.*` 和可学习 mask token。这样原 HPSD checkpoint 可以作为可选初始化，而新模型测试导出仍可复用 `HPSDFeatureTester` 的 projected/backbone 两种路径。

OC-HPSD 直接在一次 backbone 前向后取得 hierarchy、distill feature 与 teacher，并一次性构造带 route statistics 的稀疏关系。每条唯一 token-patch edge 保存 `anchor_count`、`masked_count` 和 q sum，HPSD 使用 anchor support，CSC 使用 masked support，避免对相同逐点键执行两次昂贵的 `torch.unique`。早期为附加传播分支设计的 `HPSDTrainContext` 已随该分支退役而删除。

不建议把新逻辑直接写进 `hpsd_v1m1.py` 的默认路径。`HPSD-v1m1` 应继续作为冻结 baseline；新模型在 mask rate 为 0、q 恒为 1 时必须通过与 HPSD 的数值等价测试。

### 6.5 Curriculum 与训练器

当前 `ModelHook` 只会对继承 HookBase 的 model 调用生命周期方法，HPSD 本身是普通 `nn.Module`。因此新方案需要一个很小的专用 hook，例如 `ObservationCurriculumHook`，在 `before_step` 或 `before_epoch` 中把可重建的 global progress 传给模型。也可以实现通用的 `set_train_progress()` 接口并由 hook 递归查找。不要在模型 forward 中直接读取全局单例状态，否则单元测试、DDP 和导出会更难控制。

## 7. 计算、显存和 I/O 评估

### 7.1 一次训练与一次 encoder 前向是可实现的

修订后的方案在同一 batch 中先生成 mask，然后只运行一次 LitePT/PTv3 encoder。HPSD 和 CSC 都复用该 hierarchy，再合并 loss 做一次 backward。它不需要 VRSR 式先训练 propagation space，也不需要离线提取 student feature、构建 prototype 或 KNN 图。

需要区分“方法只需一个训练 run”和“研究验证只需训练一次”。任何可靠论文仍然需要 HPSD baseline、随机 mask、几何 mask、不同 seed 和下游协议等多次独立实验；新方案消除的是部署主方法所需的串行 checkpoint 阶段，而不是消融实验成本。

### 7.2 额外 I/O 很小

`image_observability` 若用 float16 保存，每点增加 2 字节。按此前审计的约 4,940 万点估算，总 correspondence 增量约 94 MiB，分摊到每个 tile 后通常不是读取瓶颈。q 应离线写入，避免每个 epoch 重复 DSM/垂向统计。

### 7.3 CSC 显存为低到中等增量

此前湖北数据平均每个 tile 约有 17,721 个 level-2 token。若 reliable token 足够且 mask rate 为 0.30，理论上限约为每 tile 5,300 个 CSC target。单独的 `[5300,1024]` fp16 prediction 约 10.4 MiB；训练还会保存 hidden activation、归一化和 float32 cosine 临时量，因此实际增量会高于这个数字，但仍远小于逐原始点生成 `[N,1024]`。

LitePT 当前配置使用 micro-batch 1 和梯度累积，因此单步峰值主要由一个 tile 决定。PTv3 当前总 batch 为 4，若四个样本都达到上述上限，仅 CSC output 约 42 MiB。考虑 projector backward 和 token/teacher 聚合，建议第一轮实测预算预留 100-300 MiB，而不是宣称“几乎无开销”。最终数字必须用与当前 HPSD 基准相同的 warmup 和峰值统计脚本测量。

### 7.4 Relation loss 是潜在的主要临时显存风险

若为 5,300 个 target 各取 8 个 pair 并显式 materialize `[5300,8,1024]` fp16 tensor，单个张量约 83 MiB，还未包含 student/teacher 两侧和 backward。因此 relation loss 必须 chunked，或者在后续实验证明必要时才实现。这也是将它从 MVP 移除的工程理由。

### 7.5 训练速度主要受 projector 和稀疏统计影响

输入 mask 不增加 encoder 层数，也不增加第二视图；其额外成本主要是 GPU 上的结构块分组、带 route 的 token-patch 去重，以及对 masked token 的 CSC projector。合理实现下，速度增量应低于双视图 teacher-student 方法，但目前无法仅凭静态代码给出可信百分比。建议设置 Go/No-Go 上限：相对 HPSD 单步时间增量不超过 25%，显存增量不超过 20%；超过上限时先降低 CSC target cap，而不是压缩全部 DINO teacher。

## 8. 真实不可视点到底如何学习

这是新方案最容易被过度表述的部分。修订后需要明确三种点的梯度性质。

高可信且未 mask 的 anchor 通过 HPSD 获得直接 DINO loss。高可信但被 mask 的 simulated-missing 点通过 CSC 获得直接 DINO loss，而且其输入属性已被遮蔽，因此预测只能依赖坐标、mask token 和邻域上下文。真实不可视点没有真实 DINO teacher，不获得直接跨模态 loss；当它们位于 masked-visible token 的感受野中时，会作为上下文参与后者的预测并可能获得 activation gradient，同时共享 encoder 权重也会把可见区域学到的规律应用到它们。

这种机制比 VRSR 更安全，因为它不会给林下点硬分配一个树冠 source feature；但它也比 VRSR 的显式 target loss 更弱，因为无法保证每个真实不可视 token 都得到特定监督。方案成败取决于 H3：受控遮蔽下学到的三维条件映射是否能跨观测状态泛化。必须用实验验证，而不能仅凭网络参数共享宣称已经实现“监督覆盖全部点”。

建议增加一个 gradient coverage 诊断：在固定真实 tile 上对 CSC backward，统计真实 `image_valid=False` 点或其 level-2 token 中非零 activation gradient 的比例、梯度范数分布和距 masked-visible block 的距离关系。它不能代替下游精度，但能证明 CSC 是否真的通过上下文路径触及不可视区域。

## 9. 实验设计评估

PDF 中以可观测性分层、ALS 垂向/回波分层和覆盖率退化曲线作为证据闭环，这部分设计是成熟且必要的。建议把实验分成“方法是否工作”“是否解决目标问题”“是否值得额外复杂度”三个层次。

第一层使用现有湖北无标签数据检查训练稳定性、mask/anchor 数量、HPSD 数值兼容、CSC cosine、梯度覆盖和资源开销。第二层必须在具有可靠类别的 ALS 数据上做 frozen linear probe、frozen MLP probe、few-shot 和 full fine-tuning，并分别报告高 q、低 q、`image_valid=False`、first/last return 和不同归一化高度层。第三层再比较 random mask、geometry-guided mask 和 relation loss，判断遥感结构建模是否产生超出一般 masked learning 的增益。

推荐最小消融为：

| 编号 | 模型 | 核心问题 |
| --- | --- | --- |
| B0 | 3D backbone from scratch | 是否需要视觉预训练 |
| B1 | 当前 HPSD | 稳定视觉蒸馏基线 |
| B2 | Observation-weighted HPSD | 连续可靠度是否减少错误监督 |
| B3 | B2 + random block mask + CSC | 一般 masked prediction 是否有效 |
| B4 | B2 + geometry-guided mask + CSC | ALS 观测模拟是否带来额外收益 |
| B5 | B4 + relation | 关系正则是否值得保留 |

VRSR 可以作为附加对照而不是主线必跑项。如果已有可信 VRSR checkpoint 或日志，可以直接复用；没有必要为了删除它而重新完成多阶段训练。

覆盖率退化实验应只减少可用于 loss 的高可信 teacher，而不能删除点云或修改下游标签。建议使用固定随机种子和固定 tile-level mask，比较 100%、75%、50%、25%、10% teacher coverage 下的 frozen probe。这样曲线反映的是视觉观测缺失鲁棒性，而不是数据量变化。

可观测性分层评价还需要警惕混杂因素。低 q 往往与类别、点密度、高度和回波类型相关，因此只报告低 q mIoU 不能区分“模型更适合某些类别”与“模型更能处理不可视”。应同时给出按类别条件化的 q 分层结果，或至少报告每层样本数和类别分布。

## 10. 建议的 MVP 边界

为了避免重演 VRSR 不断增加可靠度模块的复杂化过程，建议第一版严格限制为以下内容：

```text
保留：当前 concat-HPSD 与原生 1024D DINO teacher
新增：离线 image_observability
新增：GPU 上按 XY/垂向 block 生成 simulated mask
启用：两种 m4 backbone 已有的 embedding mask token
新增：anchor/masked 两路稀疏 edge support
新增：F3+F4 -> 1024D 的轻量 CSC projector
新增：单 run mask/lambda curriculum
不做：c_reg、c_sem、rho_above MLP、relation、KNN、prototype、queue
```

MVP 的成功标准建议设为：

1. `mask_rate=0, q=1` 时与 HPSD loss 和梯度数值等价；
2. LitePT-v1m4 与 PT-v3m4 均通过 CPU 小样本和真实 GPU forward/backward；
3. 每个 batch 的 anchor、masked target 数量满足约束，空监督样本安全返回零 loss；
4. HPSD 单步时间增量不超过 25%，峰值显存增量不超过 20%；
5. random block mask + CSC 至少不降低 overall frozen probe；
6. geometry-guided mask 在低 q 或真实不可视子集上优于 HPSD 和 random mask；
7. 至少两个 seed 的提升超过随机波动。

如果第 5 项失败，说明 CSC 基础机制没有成立，不应继续增加几何 mask 和 relation。如果第 5 项成立而第 6 项失败，保留一般 masked CSC，但不能把 geometry-guided observability simulation 写成主要贡献。如果只在 full fine-tuning 上提升而 frozen probe 无变化，需要谨慎判断收益是否来自初始化正则而非更好的无标签表示。

## 11. 推荐代码结构

以下是建议结构，不代表当前已经存在：

```text
pointspace/models/backbone/
├── hpsd/
│   ├── hpsd_v1m1.py                 # 冻结原 baseline
│   ├── hierarchy.py                 # 现有 hierarchy
│   └── observation_ops.py           # q 聚合、route edges、token statistics
└── oc_hpsd/
    ├── __init__.py
    ├── masking.py                   # GPU 结构化 mask 与样本约束
    └── oc_hpsd_v1m1.py             # HPSD + CSC + curriculum 状态

pointspace/engines/hooks/
└── observation_curriculum.py        # 将 global progress 传给模型

configs/hpsd/
├── pretrain-oc-hpsd-litept-v1m4-hubei.py
└── pretrain-oc-hpsd-ptv3-v3m4-hubei.py

tests/models/
├── test_observation_ops.py
└── test_oc_hpsd.py
```

`vrsr/ops.py` 中的 token visibility 和 teacher 聚合思想可以迁移，但不应把 VRSR 的 Top-K、128 维随机投影和 local loss 带入新主线。新命名最好突出 observation-conditioned HPSD，而不是继续使用“supervision propagation”，以免方法叙事再次回到弱监督传播。

## 12. VRSR 退役实施顺序

建议按以下顺序处理代码：

这一路线现已完成：通用 visibility/teacher aggregation 统计迁移到 `hpsd/analysis_ops.py`，OC-HPSD 已完成两种 backbone、真实数据和下游 probe 验证；旧传播分支的模型注册、配置、专用测试和历史方案文档已经删除。旧 checkpoint 仅能通过版本控制历史恢复，不再属于当前运行时兼容范围。

如果用户希望仓库立刻变得简洁，也可以在第一步就从默认 import 中移除 VRSR，使其不再自动注册，但暂时保留源码目录。这属于软弃用，能减少误用，同时不给尚未落地的新方案制造不可逆风险。

## 13. 对原 PDF 的具体修订建议

原 PDF 可以保留研究背景、问题形式化、实验设计和论文定位，但建议修改以下关键表述。

第一，把“从 CSC 中移除 F2 即可防止身份泄漏”改为“对 simulated-missing 点启用 backbone embedding mask token，并由 F3/F4 上下文恢复 teacher”。第二，把“真实不可见点通过共享 encoder 获得视觉语义”改成待验证的研究假设，并增加 gradient coverage 与不可视 probe。第三，将关系蒸馏移至可选增强或附录，不作为四个紧耦合核心模块之一。第四，q 当前包含 surface observability 与回波序号衰减，配准与 semantic confidence 延后。第五，统一使用 `image_valid` 硬门控和 `image_observability` 连续 q，不拆分多余布尔字段。第六，把“无需多次训练”准确写成“最终方法为单个连续训练 run；方法开发和消融仍需独立实验”。

参考文献也需要校对。PDF 中 DITR 条目的首位作者写成了 `Knaebel, K.`，与 arXiv 2503.18944 和官方仓库列出的作者不一致，应改为 Karim Abou Zeid 等。正式稿还应补充 true-ortho visibility、ALS-orthophoto registration 和遥感 LiDAR-image fusion 的权威文献，否则“遥感物理驱动”的相关工作部分仍偏视觉/自动驾驶。

## 14. 最终判定

| 维度 | 原稿评分 | 修订后预期 | 评价 |
| --- | ---: | ---: | --- |
| 问题重要性 | 9/10 | 9/10 | ALS 特有且真实数据占比高 |
| 方法新颖性 | 8/10 | 8.5/10 | 观测条件蒸馏与结构化 masking 叙事清晰 |
| 理论闭合性 | 6/10 | 8/10 | 必须修复 F3/F4 局部信息泄漏和过强结论 |
| HPSD 兼容性 | 8/10 | 9/10 | 原 HPSD 可冻结，已有 mask token 可复用 |
| 工程可实现性 | 7/10 | 8.5/10 | 单 encoder 前向可行，route edges 需认真实现 |
| 资源可控性 | 8/10 | 8/10 | CSC 为中等增量，relation 应延后 |
| 实验可证伪性 | 9/10 | 9/10 | 分层 probe 与覆盖率曲线能够闭环验证 |

最终建议是：**VRSR 可以退出研究主线；新方案值得实施，但应以“Observation-weighted HPSD + 真正输入 masking + CSC”的精简修订版开始，而不是直接照 PDF 四模块同时落地。** 这一版本保留 HPSD 已有成果，不给真实不可视点伪造 DINO 对应，不需要 KNN 或多阶段 checkpoint，且能直接利用当前两种 m4 backbone 的 mask token。它仍不能从理论上保证所有不可视点都获得语义，因此论文成败将取决于不可视分层 probe、覆盖率退化曲线和跨数据集验证，而不是总体 mIoU 的单一提升。

## 15. 参考资料

- Karim Abou Zeid et al., [DINO in the Room: Leveraging 2D Foundation Models for 3D Segmentation](https://arxiv.org/abs/2503.18944), [official repository](https://github.com/VisualComputingInstitute/DITR).
- Gilles Puy et al., [Three Pillars Improving Vision Foundation Model Distillation for LiDAR](https://openaccess.thecvf.com/content/CVPR2024/papers/Puy_Three_Pillars_Improving_Vision_Foundation_Model_Distillation_for_Lidar_CVPR_2024_paper.pdf), CVPR 2024.
- Xiaoyang Wu et al., [Sonata: Self-Supervised Learning of Reliable Point Representations](https://openaccess.thecvf.com/content/CVPR2025/html/Wu_Sonata_Self-Supervised_Learning_of_Reliable_Point_Representations_CVPR_2025_paper.html), [official repository](https://github.com/facebookresearch/sonata).
- 当前仓库 `docs/hpsd_implementation_report.md` 与 `docs/hpsd_vrsr_p0_p3_implementation_report.md` 中记录的 HPSD/VRSR 实现和真实数据验证结果。
