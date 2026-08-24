# 面向机载激光雷达不可视点的 HPSD 监督传播分支评估与设计

## 1. 结论摘要

当前 HPSD 分支建议保持现状。它以正射影像中 `dino_valid=True` 的点为桥梁，在
level 2 的 token-patch 多对多关系上将 DINO-1024 特征蒸馏到三维 encoder，结构
简单、监督来源明确，已经具备稳定的工程闭环。下一步如果要改善占原始点数约
50%–70% 的正射不可视点，不建议改变 HPSD 的 concat、patch 聚合或 projector，
也不建议直接为 `valid=False` 点伪造最近影像 patch。更合适的升级方式是在 HPSD
之外增加一个仅训练期启用的可见性感知监督传播分支，本文暂称为
**V-DSP（Visibility-aware Dense Supervision Propagation）**。

V-DSP 借鉴 Dense Supervision Propagation（DSP）的核心思想，但不能照搬论文的
完整 affinity 矩阵和语义分割 decoder。推荐方案以“可视 token”为带 DINO teacher
的监督锚点，以“完全不可视 token”为待学习 value，通过稀疏 feature
reallocation 让锚点位置的 DINO 对齐损失反向传播到不可视 token。样本内传播使用
当前 tile 的可视锚点和不可视 token；跨样本传播使用一个带置信度和 tile 检索的
EMA 原型队列，把历史样本的可视 DINO 锚点分配给当前样本的不可视 token。两者都
只在训练时存在，测试时仍然只保留原始三维 backbone，因而没有推理开销。

该方向与当前 HPSD 高度兼容，但存在一个必须正视的边界：DINO 从未观察到建筑
侧面、林下地面等位置的真实影像，因此传播分支只能把“相似可视地物的语义先验”
迁移给不可视点，不能恢复不存在的真实视觉信息。建筑屋顶与侧面、树冠与林下地面
可能具有完全不同的三维语义和几何属性，如果使用无门控的全局相似度传播，反而会
抹除机载激光雷达独有的信息。因此 V-DSP 应当始终是低权重、软目标、带置信门控的
辅助损失，而不能替代 HPSD 或成为硬伪标签生成器。

## 2. 问题定义与当前 HPSD 的真实监督覆盖

原始点级 `dino_valid=False` 比例不能直接等同于“没有受到 HPSD 梯度的表示比例”。
当前 backbone 会先进行 GridSample 和层级 pooling，一个 level 2 token 往往包含多个
输入点。只要 token 内存在至少一个可视点，它就会参与 token-patch 建边；同时，
PTv3/LitePT 的局部混合和 self-attention 也会让相邻不可视点间接获得梯度。因此在
设计传播分支之前，应先按层级统计以下覆盖率：

```text
visible_count[t] = sum(dino_valid[i] and input_to_level[i] == t)
total_count[t]   = sum(input_to_level[i] == t)

visible token:       visible_count[t] > 0
fully invisible:     visible_count[t] == 0
mixed token:         0 < visible_count[t] < total_count[t]
```

真正需要显式传播的是 `fully invisible token`。混合 token 已经通过同一个 token 表示
直接参与 HPSD；在 level 2 上继续对其传播的边际收益可能较低。如果原始点中
50%–70% 不可视，但 level 2 完全不可视 token 只占很小比例，说明 level 2 已经把
大量可视点与不可视点混合，此时应先评估 level 1 传播，而不是盲目增大传播损失。
推荐首先记录 level 0–4 的可视 token 比例、完全不可视 token 比例、混合 token
比例和每个 token 的可视点占比直方图，再最终确定 `propagation_level`。

当前 correspondence 的 `valid` 同时表达两个条件：该像素是否有真实正射影像覆盖，
以及开启 `surface_only_valid` 后该点是否为正射可见表面。两类无效点的学习含义并不
相同。影像覆盖缺口中的点只是缺少当前数据源，而建筑侧面、树冠下方和林下地面是
由观测几何导致的系统性不可视。未来若实现 V-DSP，推荐将映射格式扩展为两个独立
布尔字段：

```text
dino_image_covered:  点投影位置是否有真实正射影像数据
dino_surface_visible: 点是否通过正射表面可见性检测
dino_valid:           两者与 patch 合法性的最终合取
```

两个额外布尔数组对 60,000 点样本仅增加约 120 KB 未压缩数据，却可以分别统计、
采样和评估“无覆盖点”与“被遮挡点”。现有数据只包含 `dino_valid` 时仍可完成第一版
V-DSP，但无法可靠区分失败原因，不应从黑色像素值反推覆盖状态。

## 3. DSP 论文的关键思想

本文参考本地论文 [`2107.11267v3.pdf`](../2107.11267v3.pdf) 及其
[arXiv 页面](https://arxiv.org/abs/2107.11267)。论文正式题目为 *Dense
Supervision Propagation for Weakly Supervised Semantic Segmentation on 3D Point
Clouds*，其设定是在 S3DIS 和 ScanNet 中仅给 1% 或 10% 的输入点提供语义类别。

### 3.1 Cross-Sample Feature Reallocating

对于两个至少共享一个语义类别的样本，论文先得到 encoder 特征
`Fi ∈ R^(Ni×K)` 和 `Fj ∈ R^(Nj×K)`，再计算双线性 cross affinity：

```text
Ac = Fi Wc Fj^T
Ai = softmax_row(Ac)
Aj = softmax_row(Ac^T)
Fj_to_i = Ai Fj
Fi_to_j = Aj Fi
```

`Fj_to_i` 具有样本 i 的位置数量，但每一行都是样本 j 全部点特征的加权和。
论文使用共享 decoder 分别解码原始特征和 reallocated 特征，不对跨样本分支直接
施加弱语义标签，而是使用输出一致性损失：

```text
L_CR = || decoder(Fi) - decoder(Fj_to_i) ||_F^2
```

其关键不是“为未标注点生成标签”，而是改变反向传播路径：样本 i 的稀疏监督先
影响 `decoder(Fi)`，一致性损失再通过 affinity 和 `Fj_to_i` 流向样本 j 中与之相似
的未标注点。论文采用双向传播，所以两个样本均可成为监督源和被传播对象。

### 3.2 Intra-Sample Feature Reallocating

样本内模块使用同一特征计算 self affinity：

```text
As = Fi Ws Fi^T
A  = softmax_row(As^T)
Fi_self = A Fi
```

原始特征和重分配特征经过共享 decoder 后，一方面计算原始输出与重分配输出之间的
`L_SR`，另一方面在已有稀疏标签位置对重分配分支额外计算语义损失 `L_seg_s`。
由于重分配特征中的每一行由全样本点加权形成，标注位置上的损失可以直接把梯度
分配给未标注 value，而不是只依赖 backbone 的自然感受野。

### 3.3 两阶段训练是方法的一部分，而不是实现细节

论文先训练 cross-sample 模块，再使用 intra-sample 模块微调。其 S3DIS 消融显示，
10% 标签下 baseline、单独 CSFR、单独 ISFR、同时训练 CSFR+ISFR 和
CSFR→ISFR 分别为 66.5、67.7、68.0、66.3 和 68.6 mIoU；1% 标签下分别为
65.1、66.8、66.9、65.0 和 67.0。cross 与 intra 同时训练甚至低于 baseline，
而先 intra 后 cross 也明显弱于先 cross 后 intra。这说明两个传播目标会产生优化
干扰，不能简单作为两个 loss 一次性相加。

论文把 affinity 放在下采样的中间特征上，并且训练样本是半径 2 m 的室内子云。
两个传播模块只用于训练，推理时完全删除。论文也明确指出，当几何相似的类别具有
不同语义时仍会失败；这一风险在屋顶/侧面、树冠/林下地面等机载激光雷达结构中
会更严重。

## 4. DSP 与 HPSD 的可迁移部分和不可照搬部分

| 维度 | 原始 DSP | 当前 HPSD / ALS 场景 | 设计影响 |
| --- | --- | --- | --- |
| 监督 | 稀疏离散语义类别 | 可视点关联的连续 DINO-1024 patch 特征 | 需要构造 token 级 DINO 锚点，不存在“共同类别”真值 |
| 网络 | encoder-decoder 语义分割网络 | encoder-only LitePT/PTv3 + patch projector | 传播头不能依赖语义 decoder |
| 样本 | 2 m 室内子云 | 50 m 级 ALS tile，点和 token 更多 | 禁止完整 `T×T` affinity |
| 未监督点 | 随机缺少标签 | 由正射遮挡系统性缺失 | 错误传播具有方向性偏差，需要几何和可见性门控 |
| 跨样本配对 | 已知至少有一个共同类别 | 预训练数据可无语义标签 | 需用 DINO 原型签名近似判断重叠语义 |
| 推理 | 删除传播模块 | HPSDFeatureTester 导出 backbone/projected 特征 | 可保持零额外推理开销 |

最值得迁移的是“在有监督锚点的位置重建来自未监督 value 的特征，从而把锚点 loss
的梯度显式路由到未监督点”。最不应照搬的是全连接 affinity、依赖类别标签的样本
配对，以及对所有传播结果无条件施加同等权重。

## 5. 推荐架构：V-DSP 独立监督传播分支

### 5.1 与 HPSD 的组合方式

推荐新增一个外部包装模型，而不是修改 `HierarchicalPatchSetDistiller` 的损失定义：

```text
input
  └─ HPSD-v1m1（原样运行一次 backbone）
       ├─ L_hpsd：现有可视 patch 蒸馏
       └─ hierarchy + input_to_level
             └─ V-DSP propagation head
                    ├─ L_cross（阶段 1）
                    └─ L_intra（阶段 2）
```

包装器可以调用 HPSD 的 `forward(..., return_point=True)`，随后立即从返回字典中
移除 `point/hierarchy`，只把标量 loss 和紧凑统计项交给 Trainer。传播头复用
`fuse_hierarchy_features()` 和已有 `input_to_level`，不会第二次运行 backbone。
这样 HPSD checkpoint、现有测试导出路径和下游 backbone 权重提取逻辑均保持不变；
不启用包装器时，行为与当前版本逐值一致。

### 5.2 从 patch teacher 构造可靠的可视 token 锚点

当前 HPSD 已经建立唯一 `(token, patch, point_count)` 边。V-DSP 可沿反方向把一个
token 关联的多个 DINO patch 聚合为 token teacher：

```text
w_tp = sqrt(point_count_tp)
q_t  = normalize(sum_p(w_tp * dino_p) / sum_p(w_tp))
```

只有 `visible_count[t] > 0` 的 token 才能成为锚点。若一个 token 同时关联多个视觉
差异很大的 patch，其 teacher 本身不可靠，因此定义锚点纯度：

```text
purity_t = weighted_mean_p(cos(dino_p, q_t))
```

锚点采样同时考虑 `purity_t`、可视点占比和支持点数。teacher、纯度和锚点侧
backbone 特征在传播 loss 中应停止梯度；HPSD 仍负责直接优化这些可视特征，传播
分支的主要梯度目标是不可视 token。

### 5.3 样本内监督重分配

从当前 tile 采样 `M_a` 个高置信可视锚点和 `M_u` 个完全不可视 token。传播头使用
低维 factorized affinity，而不是原论文的 `K×K` 双线性矩阵：

```text
Q_a = Wq(stopgrad(f_anchor))       # [M_a, d_k]
K_u = Wk(f_unseen)                 # [M_u, d_k]
V_u = Wv(f_unseen)                 # [M_u, d_v]
A   = softmax((Q_a K_u^T) / sqrt(d_k) + geometry_bias)
R_a = A V_u                        # [M_a, d_v]
P_a = normalize(Hprop(R_a))        # [M_a, 1024]

L_intra = weighted_mean(1 - cos(P_a, stopgrad(q_anchor)))
```

这一路径保留了 DSP 的关键梯度语义：loss 虽然定义在有 DINO teacher 的锚点行，
但 `R_a` 的 value 全部来自不可视 token，所以梯度会经过 `A`、`V_u`、`K_u` 明确
流向不可视三维特征。`Hprop` 是独立的小投影头，输出仍为原生 1024 维；内部
`d_k=128`、`d_v=256` 只用于降低传播开销，不压缩 HPSD teacher，也不改变最终
比较空间。

`geometry_bias` 不应只使用欧氏距离。建筑侧面与屋顶距离很近、树冠与林下地面在
XY 上几乎重合，仅依赖局部距离会造成系统性错配。推荐输入归一化高度、局部高差、
回波序号/回波总数、强度以及 token 的局部几何描述，并对高度差和回波结构差异过大
的候选设置门控。第一版可以只做候选过滤和一个小 MLP bias，不需要复杂图网络。

为了避免少数不可视 token 被所有锚点重复选择，应记录 attention 的列覆盖率和
熵。可采用轮换随机采样、每锚点 top-k、mutual top-k 或带容量约束的稀疏 Sinkhorn；
第一版推荐“分层随机采样 + top-k + 熵阈值”，实现和显存最可控。

### 5.4 跨样本监督重分配

原始 DSP 要求一对 live 样本同时驻留显存并共享类别。当前 PTv3 配置经过梯度累计
后的实际 micro-batch 可以为 1，强行改为 2 会近似翻倍 backbone activation，抵消
稀疏传播带来的收益。因此推荐使用 **EMA 可视原型队列**，实现跨 iteration 的
单向 feature reallocation：

```text
queue entry = {
    tile_signature,
    anchor_key,          # EMA/stop-gradient, d_k
    dino_target,         # 原生 1024 维 fp16
    geometry_descriptor,
    confidence,
}

A_c = softmax(Q_memory K_current_unseen^T / sqrt(d_k) + geometry_bias)
R_c = A_c V_current_unseen
L_cross = weighted_mean(1 - cos(Hprop(R_c), dino_memory))
```

memory query 和 DINO target 均不反向传播，当前 tile 的不可视 key/value 接收梯度。
虽然它不是原论文同时保留两个计算图的双向传播，但每个 tile 在进入网络时都会成为
当前 value，随后其高置信可视原型再进入队列监督后续样本，长期效果仍是跨样本的
监督再分配，而且兼容 micro-batch 1。

由于没有类别标签，不能随机选择 memory tile。每个 tile 应由其高置信可视 DINO
原型形成 `tile_signature`，从队列中检索相似但不同名的 tile；只有签名相似度、
原型 mutual-nearest 匹配率和 attention 置信度同时达标时才启用 `L_cross`。无任何
可视锚点的 tile 无法可靠形成签名，第一版应跳过跨样本 loss，而不是随机借用 teacher。

队列方案的限制必须明确：历史样本一侧是 stop-gradient，因此单步不是原论文的
双向梯度重路由。若后续实验证明队列版有效且硬件允许，可增加“live-pair 模式”作为
高成本消融；不建议将其作为默认实现。

### 5.5 分阶段训练目标

推荐采用三个阶段，并始终保留 HPSD 作为可视锚点：

```text
阶段 0：L = L_hpsd
阶段 1：L = L_hpsd + lambda_cross(t) * L_cross
阶段 2：L = L_hpsd + lambda_intra(t) * L_intra
```

阶段 0 先让三维特征具备基本 DINO 语义，否则初始 affinity 接近随机，传播会把噪声
扩散到大多数不可视点。如果已有收敛的 concat-HPSD checkpoint，可以直接从阶段 1
开始。阶段 1 先学习跨 tile 的粗粒度共性，阶段 2 再用当前 tile 内关系微调，顺序与
DSP 的有效消融一致。`L_cross` 和 `L_intra` 默认不同时启用；二者都应采用 warm-up
或 sigmoid ramp，建议初始权重 0，峰值先从 HPSD 权重的 0.05–0.2 搜索，而不是
直接设为 1。

传播分支应使用 soft cosine loss，不生成永久伪标签。低纯度锚点、高 attention 熵、
跨 tile 匹配弱或几何冲突的关系直接跳过。这样即便传播判断错误，其影响仍被限制在
辅助 loss 内，不会污染 correspondence 或 DINO teacher 文件。

## 6. 计算量与显存评估

### 6.1 原始完整 affinity 不适用于当前 tile

假设目标层有 `T=10,000` 个 token，完整 self affinity 包含 1 亿个元素，单个
BF16/FP16 矩阵约 191 MiB，FP32 约 381 MiB。训练还需保留 logits、softmax、梯度
和矩阵乘法中间量，单方向通常会达到该数值的数倍。若 `T=13,500`，单个半精度
矩阵已经约 348 MiB；样本内与跨样本、双向传播和多个 batch 样本叠加后，很容易
额外占用数 GiB。原始 `Fi W Fj^T` 在 concat 通道 900/1008 下还包含昂贵的
`K×K` 变换。因此，完整 DSP 在当前 ALS 配置中不可接受。

### 6.2 推荐默认规模

第一版建议从以下保守规模开始：

| 参数 | 建议初值 | 作用 |
| --- | ---: | --- |
| `propagation_level` | 2 | 与 HPSD 建边尺度一致，先验证收益 |
| `prop_key_channels` | 128 | Q/K affinity 维度 |
| `prop_value_channels` | 256 | 被重分配 value 维度 |
| `intra_anchor_samples` | 256 | 每 tile 可视锚点上限 |
| `intra_unseen_samples` | 1024 | 每 tile 完全不可视 token 上限 |
| `attention_topk` | 16–32 | 每个锚点保留的不可视候选 |
| `cross_prototypes_per_tile` | 16–32 | 入队的可视 DINO 原型数 |
| `queue_tiles` | 256 | 跨样本检索范围 |

`256×1024` 的半精度 affinity 仅约 0.5 MiB，远小于完整 `T×T`。考虑 Q/K/V、
1024 维传播输出、autograd 和 top-k 临时量，合理实现的单 tile 样本内分支预计增加
约 20–60 MiB 峰值显存；实际数值需用现有真实 tile 和 AMP 测量，不能只按矩阵
理论大小承诺。跨样本队列若存 256 个 tile、每 tile 32 个原型，并保存 128 维 key
和 1024 维 fp16 teacher，总量约 18 MiB，加上置信度和几何描述仍可控制在约
20 MiB。队列可以常驻 CPU pinned memory，仅把检索到的少量原型异步传入 GPU。

传播头的参数量约为 1–1.3M：concat 特征到 128 维的 Q/K、到 256 维的 V，以及
`256→512→1024` 的独立 projector。AdamW 参数和状态会额外占用十余 MiB，但不会
随点数线性增长。计算上主要是 `M_a×M_u×d_k` affinity 和
`M_a×M_u×d_v` reallocation；采用上述规模时远小于对全体 token 生成逐点
1024 维特征。

### 6.3 与当前 batch 配置的关系

LitePT 配置当前 `batch_size_train=20`、`gradient_accumulation_steps=4`，单 GPU
解析出的 micro-batch 可大于 1；PTv3 当前 `batch_size_train=4` 且累计 4 步，单 GPU
micro-batch 为 1。梯度累计不会让不同 micro-step 的计算图同时存在，因此不能用它
替代 live cross-sample pair。EMA 队列方案对两种配置都成立，也不要求修改现有
batch 逻辑；live-pair 消融必须单独重新评估 PTv3 显存。

## 7. 与 LitePT、PTv3 和现有数据流的兼容性

LitePT-v1m4 和 PT-v3m4 都通过同一个 `HierarchyLevel(point,
input_to_level, level)` 接口暴露 encoder 层级，V-DSP 只依赖这一稳定接口、
`dino_valid`、`dino_patch_index` 和 `dino_feature`，因此传播算法无需区分具体
backbone。两者在 level 2 concat 后分别为 900 和 1008 通道，只影响 Q/K/V 投影
层的输入维度。

`LasImageDataset` 已把 DINO 点级字段加入 `index_valid_keys`，GridSample、随机
丢点和 Crop 会同步更新 `dino_valid` 与 patch index；`CompactDinoPatches` 也保留
所有当前可见点引用的 teacher。因此传播分支可以直接在 transform 之后构造 token
可见性和 teacher，不需要逐点重新校验。跨 batch 的 token 归属可由 hierarchy 中
`point.batch` 与 offset 获得。

传播模块只在预训练包装器中注册。测试时继续调用现有 HPSDFeatureTester 或直接
导出 backbone；V-DSP 的 Q/K/V、传播 projector 和队列均不加载到下游模型。若从
包装器 checkpoint 提取 `hpsd.backbone.*`，应提供显式前缀转换工具，避免依赖
`strict=False` 的偶然匹配。

## 8. 主要风险与控制措施

### 8.1 正射不可视并不等于随机缺标

DSP 的未标注点是随机抽取后缺少人工语义，而本项目中的不可视性与对象结构强相关。
屋顶可视但墙面不可视，树冠可视但地面和低层植被不可视。若模型只按 DINO 相似度
传播，可能把屋顶特征灌入墙面、把树冠特征灌入林下地面。必须使用三维几何和回波
门控，并在下游按可见性子集分别评估，不能只看总体指标。

### 8.2 DINO 连续特征比类别标签更严格

同一语义类别在不同季节、光照和地块上的 DINO 特征并不完全相同。原 DSP 只要求
传播后预测相同类别，而直接 cosine 对齐 1024 维 teacher 会传递更多外观细节。
因此跨样本最好使用多个 DINO 原型、较低 loss 权重和高置信匹配，不应强迫所有
同类地物收敛到单一全局原型。

### 8.3 表示塌缩与注意力捷径

如果大量不可视 value 被迫重建少数锚点，传播头可能让所有 value 接近平均 teacher，
或者 attention 永远选择少量容易 token。需要监控 teacher/student 方差、原型使用率、
attention 熵、不可视 token 被选择覆盖率以及不同 tile 的原型占用；必要时加入容量
约束或多样性正则，但不建议第一版立即引入复杂对比损失。

### 8.4 队列陈旧和跨区域域差异

EMA 队列中的 student key 会随 backbone 更新而陈旧。应保存 teacher DINO target
作为稳定监督，只对 student key 使用较短队列或 momentum 更新；跨航带、季节、
传感器差异较大时，可在 tile signature 中加入数据域标识并优先域内检索。

### 8.5 传播 loss 干扰 HPSD

论文已经实证 cross 与 intra 联合训练可能低于 baseline。本项目中 HPSD 本身也是
额外目标，干扰风险更高。必须保留 HPSD-only checkpoint、分阶段启用、loss ramp、
梯度范数监控和快速回退开关。传播分支不应修改现有 HPSD 统计、teacher 文件或
correspondence。

## 9. 实验与验收方案

第一步不是实现完整 cross queue，而是完成监督覆盖审计。在真实湖北数据上统计原始
点与 level 0–4 token 的可视/完全不可视/混合比例。如果 level 2 完全不可视 token
仍占明显比例，先在 level 2 实现样本内稀疏传播；如果 level 2 主要是混合 token，
则将传播层前移到 level 1，但仍保持 HPSD 在 level 2。

随后按以下顺序做消融，每次都从相同 HPSD checkpoint 和随机种子开始：

| 实验 | 目的 |
| --- | --- |
| HPSD baseline | 固定当前基准 |
| HPSD + intra | 验证同 tile 显式梯度传播是否有效 |
| HPSD + cross queue | 验证跨 tile 原型是否提供额外信息 |
| HPSD + cross→intra | 验证论文推荐的两阶段顺序 |
| HPSD + cross+intra joint | 仅作为负面对照，不建议默认使用 |
| 去除 geometry gate | 量化屋顶/侧面、树冠/地面错配风险 |
| level 1 vs level 2 | 确定空间分辨率与显存收益平衡 |

预训练期至少记录 `L_hpsd`、传播 loss、锚点数、完全不可视 token 数、成功传播
token 数、传播覆盖率、平均 attention 熵、跨样本检索相似度和被跳过样本比例。
下游评价不能只报告总体 mIoU；应在有真值的数据上分别报告可视点、正射遮挡点、
无影像覆盖点、建筑侧面候选和多回波林下点的指标。如果暂时没有这些细分类别，至少
按 `dino_valid` 分成 visible / invisible 两组，并报告 linear probe、全量 fine-tune
和少标注 fine-tune 的差异。

建议的第一阶段验收条件是：传播分支使完全不可视 token 获得非零且有限的直接梯度；
HPSD loss 和可视子集下游指标不显著退化；传播 attention 不塌缩到极少 token；真实
训练峰值显存增量控制在基准的约 10%–20% 内。只有 intra 通过这些条件后，才值得
实现跨样本队列。

## 10. 推荐实施顺序

1. 新增只读覆盖审计工具，统计 raw point 与各层 token 的可见性组成，不改模型。
2. 在未来新生成的 correspondence 中拆分 `image_covered` 与 `surface_visible`；旧数据
   继续兼容单一 `valid`。
3. 新增独立 V-DSP wrapper 和 token teacher 聚合函数，保持 HPSD-v1m1 不变。
4. 只实现 level 2 的 sparse intra reallocation，完成梯度、显存和下游子集消融。
5. 若 intra 有稳定收益，再实现 CPU/EMA cross-tile prototype queue 和检索门控。
6. 最后验证 cross→intra 两阶段训练；live-pair 和 Sinkhorn 仅作为后续高成本消融。

## 11. 最终判断

DSP 的监督重分配思想非常适合补充 HPSD 的“只在可视点建立直接 teacher”这一结构性
缺口，尤其是跨 tile 的监督迁移契合机载激光雷达中同一地物分散于不同样本的特点。
但原始 DSP 不能直接移植：完整 affinity 对当前 token 数过于昂贵，其共同类别假设在
无标签预训练中不存在，而且正射不可视具有强烈的结构偏差。

因此推荐把升级目标定义为“**稀疏、可见性感知、DINO 原型门控的梯度重路由**”，
而不是一般的特征平滑或硬伪标签传播。采用独立训练分支、HPSD 持续锚定、cross→intra
分阶段、level 2 起步和 EMA 原型队列后，该方案可以兼容现有 LitePT/PTv3、数据集、
Trainer 和特征导出流程，预计不会引入不可接受的推理成本，训练显存也可通过固定
采样规模控制。它值得作为 HPSD 的下一代可选增强方向，但应先以覆盖审计和 intra
MVP 验证假设，再投入跨样本模块的完整工程实现。

## 参考资料

- Jiacheng Wei et al., [Dense Supervision Propagation for Weakly Supervised
  Semantic Segmentation on 3D Point Clouds](https://arxiv.org/abs/2107.11267),
  IEEE TCSVT 34(6), 4367–4377.
- Xiang Xu and Gim Hee Lee, [Weakly Supervised Semantic Point Cloud Segmentation:
  Towards 10× Fewer Labels](https://openaccess.thecvf.com/content_CVPR_2020/html/Xu_Weakly_Supervised_Semantic_Point_Cloud_Segmentation_Towards_10x_Fewer_Labels_CVPR_2020_paper.html),
  CVPR 2020.
- Hanyu Shi et al., [Weakly Supervised Segmentation on Outdoor 4D Point Clouds
  With Temporal Matching and Spatial Graph Propagation](https://openaccess.thecvf.com/content/CVPR2022/html/Shi_Weakly_Supervised_Segmentation_on_Outdoor_4D_Point_Clouds_With_Temporal_Matching_and_CVPR_2022_paper.html),
  CVPR 2022.

