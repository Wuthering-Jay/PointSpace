# DALES 标签效率非单调现象：权重审计与三随机种子复现实验

## 1. 问题

现有单次实验中，Scratch 的 mIoU 随标签比例呈明显非单调变化：0.01%、0.1%、1%、10% 和 100% 分别约为 54.70、64.06、51.42、58.48 和 58.06；Point-M2AE 也存在类似波动。非单调并不违反理论，但如此大的反向变化不能直接解释为标签效率规律，必须先排除权重加载错误，再估计训练随机性。

## 2. 预训练权重审计

### Scratch

- 所有 Scratch 配置均为 `weight=None`，没有误加载其他预训练模型。
- 模型可从随机初始化完整构建。

### Point-M2AE

所有标签比例使用同一个权重：

`exp/baselines/swiss/pretrain-point-m2ae-litept-v1m4-100/model/backbone_last_bncal_swiss.pth`

审计结果如下：

- 文件大小：61,523,147 bytes（约 58.67 MiB）；
- SHA-256：`9BAC4EF74D7C3DD9940E18081E9C13BE80CB1C640F2C26F99A61F97591D57457`；
- checkpoint 来源标记：`backbone-only export`；
- 共 262 个主干张量，全部为有限值，未发现 NaN 或 Inf；
- 与当前 Point-M2AE 下游模型逐项比较时，262 个主干张量名称和形状全部匹配；
- 加载后仅缺少 `seg_head.weight` 和 `seg_head.bias`，这是下游任务按类别数重新初始化分割头的预期行为；
- 无 unexpected keys，无 shape mismatch；
- 原预训练日志中损失从约 0.0926 逐步下降至 0.0715，未发现数值发散。

因此，目前没有证据表明 Point-M2AE 的非单调曲线由 checkpoint 损坏、权重错载或网络版本不匹配导致。权重本身可以基本排除为首要原因。

## 3. 更可能的波动来源

现有结果只包含单个 seed，且稀疏监督下以下因素会被明显放大：

1. 随机初始化的分割头、数据顺序、增强和裁剪会改变每轮实际见到的稀有类别监督；
2. CE 自动类别权重与 Lovasz loss 对少量稀有类点和 batch 组成较敏感；
3. BN 精确校准所抽取的数据批次受 seed 影响；
4. 0.25% 相对阈值、patience=3 的早停可能在不同局部最优处终止；
5. 原 100% 端点来自 4.5 的 10 epoch × 5 loop 协议，而稀疏实验采用 20 epoch × 3 loop，两者此前并非严格同协议。

尤其需要注意，嵌套稀疏标注只保证高比例集合包含低比例集合，并不保证每个随机 crop 或训练 batch 中的有效监督密度单调稳定。单次运行出现反转并不足以说明增加标签有害。

## 4. 三随机种子协议

本轮只比较 Scratch 和 Point-M2AE，使用：

- seeds：42、3407、5291；
- 标签比例：0.01%、0.1%、1%、10%、100%；
- 每个配置：20 epochs、每 epoch 3 loops；
- early stopping：监控验证集 mIoU，patience=3，相对改善阈值 0.25%，warm-up=2 epochs；
- 100% 端点也使用同一训练和早停协议；
- 稀疏标签 manifest 固定为 `sparse_labels_seed42`；
- 只改变训练、数据管线与 BN 校准 seed。

总计 2 × 5 × 3 = 30 个任务。固定标签 manifest 是有意的：本轮估计的是训练随机性，而不是训练随机性与标注抽样随机性的混合。如果均值曲线仍有异常，再增加独立 manifest seeds 评估“抽到哪些标签”带来的方差。

## 5. 结果解释规则

每个标签比例应报告三次实验的：

- mIoU mean ± standard deviation；
- 三个独立值；
- 最佳 epoch；
- 是否由 early stopping 终止。

不应强制把均值修成单调。建议按以下规则判断：

- 若相邻预算均值反转小于合并标准差，视为统计波动；
- 若反转显著大于标准差，检查逐类 IoU、有效监督点数、最佳 epoch 和 BN 统计；
- 若 Scratch 与 Point-M2AE 在同一预算处同时异常，优先检查数据/训练协议；
- 若只有 Point-M2AE 异常，再进一步检查迁移优化稳定性，而不是 checkpoint 完整性。

## 6. 生成内容

- 顺序运行脚本：`paper_cvrl/run_section46_multiseed.py`
- 30 个独立配置：`configs/label_efficiency/dales/multiseed/`
- 任务清单：`configs/label_efficiency/dales/multiseed/jobs.json`
- 输出目录：`exp/label_efficiency_multiseed/dales/seed*/`

运行器会在每次启动时重新生成配置，并按 seed → 标签比例 → 方法顺序逐项执行。若某个任务的 `train.log` 已包含 `Test result:`，默认自动跳过，因而可以安全地断点续跑；使用 `--force` 才会强制重跑。

## 7. 已完成验证

- 两个 Python 文件已通过语法编译；
- 30 个配置均可被项目配置解析器读取；
- 24 个稀疏配置均使用固定 manifest，6 个 100% 配置均使用完整 `LasDataset`；
- 每个 seed 的输出路径唯一；
- Scratch、Point-M2AE 的代表性稀疏与完整配置均已完成模型构建测试；
- Point-M2AE 的代表配置已完成 checkpoint 实际加载测试。

## 8. 2026-10-03 阶段性结果

当前 30 个任务中，seed 42 与 seed 3407 的 20 个任务已经完整产生
`Test result`；seed 5291 只运行了 0.01% Scratch，且停在训练/验证阶段，尚无
测试结果。因此下面的均值和标准差都只是 **2-seed 阶段统计**，不能替代最终
3-seed 结论。

### 8.1 测试集 mIoU

| 方法 | 标签比例 | seed 42 | seed 3407 | 2-seed mean ± sample std |
|---|---:|---:|---:|---:|
| Scratch | 0.01% | 0.5364 | 0.5381 | 0.5373 ± 0.0012 |
| Scratch | 0.1% | 0.4689 | 0.6538 | 0.5614 ± 0.1307 |
| Scratch | 1% | 0.5640 | 0.6356 | 0.5998 ± 0.0506 |
| Scratch | 10% | 0.7389 | 0.5077 | 0.6233 ± 0.1635 |
| Scratch | 100% | 0.5050 | 0.7375 | 0.6213 ± 0.1644 |
| Point-M2AE | 0.01% | 0.4665 | 0.4686 | 0.4676 ± 0.0015 |
| Point-M2AE | 0.1% | 0.5886 | 0.4894 | 0.5390 ± 0.0701 |
| Point-M2AE | 1% | 0.6565 | 0.5630 | 0.6098 ± 0.0661 |
| Point-M2AE | 10% | 0.5977 | 0.7382 | 0.6680 ± 0.0993 |
| Point-M2AE | 100% | 0.7203 | 0.6645 | 0.6924 ± 0.0395 |

### 8.2 趋势是否一致

阶段均值给出了比原单次实验更合理的图景：Point-M2AE 的均值从 0.4676、
0.5390、0.6098、0.6680 到 0.6924，严格单调上升；Scratch 的均值从 0.5373、
0.5614、0.5998 到 0.6233 后，在 100% 处轻微降至 0.6213。最后这个 -0.20
个百分点的变化远小于该预算下的 16.44 个百分点 seed 标准差，不能解释为
“更多标签损害性能”。所以，原单次曲线中剧烈的非单调性目前更像是高训练
方差，而不是稳定的标签效率规律。

但不同 seed 的单次曲线仍然非常不稳定。例如 Scratch seed 42 在 10% 达到
0.7389、100% 却只有 0.5050；seed 3407 则在相同两个预算得到 0.5077 和
0.7375，方向完全相反。Point-M2AE 也在 10% 出现 14.05 个百分点的 seed
差异。这说明均值趋势虽已改善，当前协议的方差仍大到足以改变方法排序。

### 8.3 一个额外的复现性警报

原实验与 multiseed 实验中的 seed-42 配置逐行比较后，除 `save_path` 和注释外
完全相同，使用的也是同一 `sparse_labels_seed42` manifest；但部分结果仍明显
不同。例如 0.1% Scratch 从原来的 0.6406 变为 0.4689。两份日志的训练损失
轨迹从早期就已经分离，因此差异不是测试阶段类别映射或 checkpoint 选择造成
的。它更可能来自两次运行之间的运行时代码状态变化，或 CUDA/spconv、数据
加载和随机增强链路中的非确定性。现有日志没有保存 git commit、工作树 diff、
CUDA deterministic 状态和依赖快照，因而无法仅凭日志把两者进一步拆开。

这意味着本轮 3-seed 实验应当作为一个连续的软件快照完成；在 seed 5291 完成
前不要混入新的训练代码修改，也不要把 9 月 30 日的原单次 seed-42 结果当作
第三个独立 seed。

### 8.4 权重结论

Point-M2AE 的所有已完成复现实验都加载同一个
`backbone_last_bncal_swiss.pth`，日志中始终只缺少新建的
`seg_head.weight/bias`，没有 unexpected key 或 shape mismatch。当前结果没有
支持“预训练权重损坏”这一解释。相反，Scratch 同样出现巨大波动，说明主要
问题位于共同的稀疏微调协议/运行复现性，而不是 Point-M2AE checkpoint 独有。

## 9. 当前行动建议

1. 继续完成 seed 5291 剩余任务；运行器会自动跳过已有 `Test result` 的任务。
2. 完成前冻结代码与环境；至少记录 git commit/diff hash、PyTorch/CUDA/spconv
   版本及 deterministic 配置。
3. 最终以 3-seed mean ± std 报告，不再使用单 seed 曲线论证单调性。
4. 若第三个 seed 后 Scratch 仍有超过约 10 个百分点的标准差，优先取消或延后
   稀疏实验的早停，并固定 epoch 比较；随后再分别评估独立 manifest seed。
5. 论文中可报告均值随预算总体上升，但在 3-seed 完成前不应写成确定结论。
