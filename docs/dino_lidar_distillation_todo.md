# HPSD 实现 TODO

本清单对应 [HPSD 设计报告](dino_lidar_distillation_design.md)。第一版保持
DINO 原生 1024 维特征，不启用 PCA 压缩；只有真实显存测试确认有必要后，
才进入降维优化阶段。

## P0：第一版训练闭环

- [x] 完成设计报告和真实数据尺度诊断。
- [x] 定义 PTV3/LitePT 共用的 `PointHierarchy` 数据结构与映射校验。
- [x] 新增 `PT-v3m4` encoder-only backbone。
- [x] 新增 `LitePT-v1m4` encoder-only backbone。
- [x] 实现输入点到指定 encoder level token 的映射。
- [x] 将目标层之后的深层特征无损 up-cast，确保全部 encoder stage 获得梯度。
- [x] 实现完整 token-patch unique edge 构建和计数。
- [x] 实现 `sqrt_count` patch-centric 三维特征聚合。
- [x] 实现原生 1024 维 student projector 和 float32 cosine loss。
- [x] 按样本平均 patch loss，正确处理空监督样本。
- [x] 注册 `HPSD-v1m1` 并提供 PTV3/LitePT 最小配置示例。
- [x] 验证蒸馏 checkpoint 可只迁移 `backbone.*`。

## P0：正确性测试

- [x] 合成层级映射测试。
- [x] 单 token 对多 patch 测试。
- [x] 单 patch 对多 token 测试。
- [x] 无效点和空 patch 测试。
- [x] batch patch offset 与样本边界测试。
- [x] PTV3 forward/backward smoke test。
- [x] LitePT forward/backward smoke test。
- [x] `LasImageDataset -> collate -> HPSD` 真实数据测试。
- [x] AMP bfloat16 下 loss 有限性测试。
- [x] 整个 batch 无有效 patch 的完整模型 backward 测试。
- [x] 正式 Trainer 的真实数据多 iteration 与梯度累积测试。
- [x] 验证 AdamW 参数更新、checkpoint 保存及模型严格回载。
- [x] 通过 `tools/train.py` 完成真实数据短训练。
- [x] 新增 `HPSDFeatureTester`，通过 `tools/test.py` 导出点特征。
- [x] 合并 GridSample test fragments 并恢复原始点顺序。
- [x] 使用 Safetensors 保存逐点 HPSD 特征与关键元数据。
- [x] 实现 PCA/KMeans 分析、分别赋色和 `orig_idx` tile 合并。

## P1：性能与数据传输优化

- [ ] 记录逐点 DiTR-style 1024 维基线显存和吞吐量。
- [x] 记录 HPSD level 2 的 token、edge、patch 数和峰值显存。
- [ ] 记录 HPSD level 1 的同口径显存与吞吐量。
- [x] 根据真实裁剪数据判断是否需要 `CompactDinoPatches`。
- [x] 实现在最终点变换之后移除未引用 patch 的无损 transform。
- [x] 验证压缩前后 teacher gather 逐值一致。
- [x] 验证压缩前后完整 HPSD loss 逐值一致。

## P2：关系增强

- [ ] 实现 edge mean pixel 和相对 patch 位置统计。
- [ ] 实现可选 relation-aware edge decoder。
- [ ] 比较 nearest、token target mean、all-edge 和 edge decoder。
- [ ] 增加 patches-per-token、tokens-per-patch 训练日志。

## P3：可选 DINO 降维

- [ ] 仅在原生 1024 维显存或吞吐量不可接受时启动。
- [ ] 实现有效 patch 均匀采样工具。
- [ ] 实现 PCA-512/PCA-256 拟合和 Safetensors 保存。
- [ ] 实现冻结 teacher projector。
- [ ] 比较原生 1024、PCA-512 和 PCA-256 的下游精度。

## P4：UTONIA 联合训练

- [ ] 在独立 HPSD 收益确认后接入 UTONIA student 指定层级。
- [ ] 保持 DINO teacher 与 EMA 3D teacher 相互独立。
- [ ] 比较纯 HPSD、纯 UTONIA 和联合训练。

## 第一版验收标准

1. PTV3 和 LitePT 使用同一个 `HPSD-v1m1` wrapper 完成 forward/backward。
2. 多 patch 对应不经过坐标平均，也不退化为最近 patch。
3. Student 仅对有效 patch 生成 `[U,1024]` prediction，不生成 `[N,1024]`。
4. batch 内每个样本等权，空监督样本不产生 NaN。
5. 默认训练保持 DINO 原始 1024 维，推理只需要普通三维 backbone。
6. 正式 Trainer 能在真实数据上完成梯度累积、优化器更新和 checkpoint 闭环。
