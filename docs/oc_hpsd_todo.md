# OC-HPSD 实施 TODO

## 目标与边界

OC-HPSD（Observation-Conditioned HPSD）用于替代 VRSR 研究主线。它保留当前 concat-HPSD 的可视区域视觉锚定能力，在同一次训练 run 和同一次 encoder 前向中，对高可信可视点实施结构化输入 masking，并利用 CSC（Contextual Semantic Completion）恢复其真实 DINO teacher。真实不可视点不构造伪标签，只通过共享 encoder 和上下文计算间接受益。

第一版本保持 DINO 原生 1024 维，不实现 KNN、graph、prototype、queue、registration confidence、semantic confidence 和 relational distillation。`HPSD-v1m1` 必须保持行为不变，OC-HPSD 使用独立注册名和配置。

## P0：基线与退役策略

- [x] 完成新方案理论、兼容性和资源评估。
- [x] 确认 LitePT-v1m4 与 PT-v3m4 已有 embedding mask token 能力。
- [x] 确认真实湖北数据存在显著的不可视点和 fully-invisible token。
- [x] 将 VRSR 从推荐主线标记为 deprecated。
- [x] 将仍有价值的可视覆盖审计算子迁移到 HPSD 分析模块。
- [x] 删除 VRSR 注册、实现、配置、专用测试和历史方案文档。
- [x] 旧实现仍可由版本控制历史恢复，不再保留运行时 checkpoint 兼容入口。

VRSR 已退出代码和实验主线。后续只维护 HPSD 与 OC-HPSD，不再为旧 VRSR checkpoint 提供模型注册兼容。

## P1：连续可观测度数据链路

- [x] `tile_las_image.py` 在现有 DSM 计算中生成连续 surface observability。
- [x] correspondence Safetensors 新增 `observability: [N]` float16。
- [x] 无影像覆盖点的 observability 固定为 0。
- [x] `image_valid` 作为影像覆盖与正射可见性的统一硬门控。
- [x] DINO correspondence 更新必须保留 observability 和未知扩展 tensor。
- [x] `LasImageDataset` 读取为 `image_observability` 并注册为点级字段。
- [x] correspondence 缺少 observability 时回退为 `image_valid.float()`。
- [x] 增加新旧 schema、形状、范围和点采样同步测试。
- [x] q 融合回波序号指数衰减，并完成真实数据更新与独立备份。
- [x] 点—影像关系字段统一改用 `image_` 前缀，不保留旧字段别名。

当前 observability 描述 DSM 表面接近程度与回波序号先验。配准可信度和 DINO 邻域稳定性暂不加入 q，避免一次引入不可归因的多变量。

## P2：结构化输入 Masking

- [x] 实现按样本、XY block 和垂向跨度选择 simulated-missing 点的 GPU 算子。
- [x] mask 候选只来自 `image_valid=True` 且 q 高于阈值的点。
- [x] 每个样本保留最小 anchor 数和最小 anchor 比例。
- [x] 垂向结构不足时支持退化为普通 block mask，而不是产生空训练批次。
- [x] mask rate 为 0 时严格返回全 False。
- [x] 两种 m4 backbone 配置启用 `mask_token=True`。
- [x] 验证被 mask 点的 embedded feature 被替换，同时坐标和点数量不变。

第一版不同时实现 vertical-column、boundary-aware 和 large-context 三种策略。默认只实现可向随机 block 回退的 vertical-structure-aware XY block mask。

## P3：Observation-HPSD 与 CSC

- [x] 构造一次去重的 routed token-patch edges。
- [x] 每条 edge 分别记录 anchor/masked point count 和 q sum。
- [x] Observation-HPSD 只使用 anchor support，并按 `sqrt(count) * mean(q)` 聚合。
- [x] CSC 只使用 masked-visible support 聚合 token teacher。
- [x] CSC 输入为目标层之后的 F3/F4 等深层 concat，不使用 F2。
- [x] CSC projector 保持 DINO 原生 1024 维输出。
- [x] 空 anchor、空 masked target 和单样本无监督均能安全反传零梯度。
- [x] 只为实际 CSC target 创建 `[M,1024]` prediction。
- [x] 测试导出路径不生成 mask、不读取 correspondence，并继续兼容 `HPSDFeatureTester`。

必须增加 `q=1, mask_rate=0` 与原 HPSD 的 loss/gradient 数值等价测试。该测试通过前，不能把 OC-HPSD 用作正式训练配置。

## P4：单 Run Curriculum

- [x] 新增 hook，将可重建的 global training progress 传给 OC-HPSD。
- [x] 训练前 10% 使用纯 HPSD。
- [x] 训练 10%-20% 线性提升 mask rate 和 CSC loss weight。
- [x] 后续保持目标值，optimizer 和 scheduler 不重置。
- [x] 断点恢复后 curriculum 能由 epoch/iteration 自动恢复。
- [x] 日志只输出紧凑标量：HPSD、CSC、mask rate、anchor/masked/CSC token 数。

## P5：配置与自动测试

- [x] 新增完整 LitePT-v1m4 湖北配置，不使用 base。
- [x] 新增完整 PT-v3m4 湖北配置，不使用 base。
- [x] 配置保留 RuntimeInfoHook、ModelHook、CacheCleaner 和 HPSDFeatureTester。
- [x] 增加 routed edges、q weighting、mask 约束、CSC gradient 和空监督测试。
- [x] 原 HPSD/VRSR 测试继续通过，证明没有破坏既有成果。
- [x] 在 pointcept 环境运行目标测试和完整相关测试。

## P6：真实数据与 GPU 验证

- [x] 使用 `E:\data\湖北\joint_tiles` 的旧 correspondence 验证回退读取。
- [x] 统计真实 batch 的 q、anchor、masked point、CSC token 数量。
- [x] LitePT-v1m4 完成真实 batch BF16 forward/backward。
- [x] PT-v3m4 完成裁剪真实 batch BF16 forward/backward。
- [x] `HPSDFeatureTester` 验证新模型导出路径。
- [x] 测量相对 HPSD 的 warm step 时间与峰值显存。

第一版本 Go/No-Go 资源上限为：相对 HPSD 单步时间增量不超过 25%，峰值显存增量不超过 20%。超过上限时优先限制每样本 CSC target 数，不压缩全部 DINO teacher。

## P7：训练后研究验证

- [ ] 固化 HPSD baseline checkpoint 和 probe 结果。
- [ ] 比较 HPSD、Observation-HPSD、random block mask+CSC、geometry mask+CSC。
- [ ] 至少两个 seed 验证提升超过随机波动。
- [ ] 报告 overall、可视、不可视、q 分层、归一化高度和回波分层 probe。
- [ ] 绘制 100%、75%、50%、25%、10% teacher coverage 退化曲线。
- [ ] 增加真实不可视 activation gradient coverage 诊断。
- [ ] 只有 P3/P4 产生稳定增益后，才评估轻量 relational distillation。

## 第一版本完成定义

第一版本完成意味着 P1-P5 全部完成，P6 至少完成旧数据读取、两个 backbone 的真实 forward/backward 和导出兼容验证。P7 属于需要正式训练预算的研究实验，不阻塞代码 MVP，但必须在论文结论前完成。

## P8：v1m2 机制优化

- [x] 固定 `OC-HPSD-v1m1` 注册名、配置和数值路径作为已验证基线。
- [x] 审计真实样本的候选比例、目标预算、实际 mask rate 和预算利用率。
- [x] 用连续 CSC token 可信度替代 0.5 单一硬门槛，并保持样本等权。
- [x] 为提高 mask 上限后的最后一个边界 block 增加可选预算补齐。
- [x] 日志区分实际候选遮蔽率 `mr`、预算利用率 `mu` 和 CSC token 保留率 `cr`。
- [x] 新增完整 LitePT-v1m4 与 PT-v3m4 v1m2 配置，不依赖 base。
- [x] 完成真实湖北样本 forward/backward、显存和监督覆盖验证。
- [x] 训练 v1m2 并完成下游比较；结果轻微下降，不替代 v1m1 主线。
- [ ] 将 mask 上限恢复为 8192，单独消融连续 CSC 权重。
- [ ] 评估 mask-independent HPSD 路由，解除 CSC 增强与 anchor 减少的竞争。

v1m2 不新增 encoder 前向、DINO 维度或可学习模块。它可以从 v1m1 checkpoint
加载全部模型参数，但正式对比应重新进行无标签预训练，不能只替换 loss 后直接使用
旧 checkpoint 做 probe。
