# 4.5 Swiss 预训练表示的跨数据集下游迁移：完整实验报告

## 数据与核验

本报告解析 `exp/downstream/{dales,opengf,vaihingen}/{method}-{lin,dec,ft}/train.log`。9 个方法 × 3 种读出/微调模式 × 3 个数据集，共 **81/81** 组均有一次最终 `Test result`、对应的 `model_best.pth`，日志中没有 `ERROR`。最终测试由 `PreciseEvaluator` 加载验证 mIoU 最佳 checkpoint；不是最后一轮权重。逐类测试 IoU 的均值与日志 mIoU 一致，最大差异仅为四舍五入造成的 0.000063。

所有预训练权重来自 Swiss-100，Scratch 随机初始化。DALES 为 8 类（2,900 train / 1,100 held-out tile）；OpenGF 为过滤噪声后的 ground / non-ground 2 类（3,775 / 1,059）；Vaihingen 为 9 类（239 / 183）。表中全部数字是最终精确测试 **mIoU，单位 %**；每列粗体表示该数据集和读出方式下的最高值。

### DALES

| 方法 | Frozen linear | Decoder probe | Full FT |
|---|---:|---:|---:|
| Scratch | 16.59 | 66.56 | 58.06 |
| Point-M2AE | 37.58 | 68.96 | 64.11 |
| MSC | 46.52 | **76.89** | 79.59 |
| Sonata | 48.68 | 76.72 | 78.90 |
| D-DITR | 33.61 | 69.66 | 69.08 |
| Concerto | 35.62 | 66.75 | 54.61 |
| VPSD | 56.55 | 76.18 | 79.69 |
| RRCL | 51.29 | 76.62 | **79.96** |
| CVRL | **61.89** | 76.31 | 79.28 |

### OpenGF

| 方法 | Frozen linear | Decoder probe | Full FT |
|---|---:|---:|---:|
| Scratch | 59.88 | 89.57 | 87.81 |
| Point-M2AE | 77.17 | 90.17 | 88.84 |
| MSC | 72.77 | 92.16 | 88.89 |
| Sonata | 84.65 | 92.76 | **93.55** |
| D-DITR | 68.78 | 89.07 | 92.96 |
| Concerto | 79.41 | 89.06 | 88.67 |
| VPSD | 81.15 | **94.08** | 90.37 |
| RRCL | 87.70 | 92.79 | 88.60 |
| CVRL | **88.23** | 93.29 | 89.59 |

### Vaihingen

| 方法 | Frozen linear | Decoder probe | Full FT |
|---|---:|---:|---:|
| Scratch | 11.60 | 40.91 | 57.84 |
| Point-M2AE | 28.11 | 51.61 | 44.34 |
| MSC | 38.92 | 56.72 | 59.40 |
| Sonata | 43.70 | 57.43 | 59.34 |
| D-DITR | 31.97 | 53.01 | 53.49 |
| Concerto | 32.13 | 51.12 | 52.74 |
| VPSD | 53.14 | **59.25** | **60.75** |
| RRCL | 42.83 | 57.77 | 59.14 |
| CVRL | **56.77** | 59.12 | 60.73 |


## 主要结论

**CVRL 的最稳定优势是冻结表征的线性可分性。** 它在三个数据集的 linear probe 均排名第一。相对第二名，DALES 高 5.34 pp（VPSD），OpenGF 高 0.53 pp（RRCL），Vaihingen 高 3.63 pp（VPSD）。相对 Scratch 的提升分别为 45.30、28.35、45.17 pp。这个结果支持视觉与回波信息的联合表示对可直接读出的语义信息有益；OpenGF 的 0.53 pp 优势较小，单 seed 不能证明显著差异。

**Decoder probe 与 FT 并没有形成三数据集统一领先。** CVRL 的 decoder 在 DALES、OpenGF、Vaihingen 分别为第 4、2、2 名；FT 为第 4、4、2 名。DALES FT 的 79.28% 比 Scratch 高 21.22 pp，但比 RRCL、VPSD、MSC 分别低 0.68、0.41、0.31 pp。Vaihingen FT 的 60.73% 与 VPSD 的 60.75% 仅差 0.02 pp，合理的表述是接近持平。OpenGF FT 的 89.59% 则低于 Sonata 3.96 pp，不能用“全面最优”概括 CVRL。

**DALES 的少数类对名次敏感。** CVRL 与 RRCL 的 FT 逐类 IoU 在多数类别很接近，但 trucks 为 34.54% vs 40.85%，相差 6.31 pp；这一类单独就会对八类 mIoU 贡献约 0.79 pp 的劣势，足以解释两者总体 0.68 pp 的差距。CVRL 在 vegetation、cars、fences、buildings 上达到九种方法中的最高或并列最高逐类 IoU，但差距通常极小，不能用个别类别的最高值替代整体 mIoU。Vaihingen 的 CVRL 在 fence/hedge、roof、facade 领先，VPSD 在 impervious surfaces、car、tree 等类别更高，两者整体几乎持平。

**OpenGF 出现普遍的“decoder > FT”现象。** 9 种方法中有 7 种的 decoder mIoU 高于 FT；CVRL 为 93.29% vs 89.59%，VPSD 为 94.08% vs 90.37%，RRCL 为 92.79% vs 88.60%。Sonata 和 D-DITR 是例外。这说明 OpenGF 的 FT 下降不是 CVRL 单独的故障，更可能涉及完整更新预训练权重时的优化与泛化权衡；日志本身不能确定具体机制。不能把较弱 FT 结果直接解释为预训练表示没有价值，因为相应的 linear/decoder 结果明显更高。

**早停实际生效。** OpenGF 的 27/27 项均在 20 epoch 上限之前停下；Vaihingen 的 15/27 项早停；DALES 未配置早停，均完成 10 epoch。最终测试用最佳验证 checkpoint，因此晚期退化不会直接变成测试时的最后权重。例如 DALES Concerto FT 第 2 轮验证 mIoU 达到 53.93%，第 10 轮仅 6.42%，最终测试仍为最佳 checkpoint 的 54.61%。这说明该协议下部分方法的迁移轨迹不稳定，但仅凭这些日志无法区分优化、归一化或任务适配等原因。

## 解释边界与论文建议

1. **验证与最终测试不是独立文件集。** 三个数据集的配置均将同一 held-out 路径用于验证选模和精确测试。最终测试改变了推理/fragment 协议，但不是独立于选模的数据。因此论文应描述为“held-out split 的最佳验证 checkpoint 精确推理结果”，不能称为独立测试集的无偏泛化估计。DALES Scratch FT 等个别运行的验证与精确测试 mIoU 差异较大，也提示两种推理协议不可简单混同。
2. **不同 probe 的 CVRL 表征不完全相同。** CVRL linear 和 FT 使用完整 200D 融合表示；CVRL decoder probe 按配置只迁移共享 LitePT encoder，并训练原生 decoder，视觉/回波完整分支不作为 decoder 输入。因此 decoder 分数不能解释为完整 200D 融合特征的 decoder 表现。
3. **冻结特征维度不同。** CVRL full 为 200D，VPSD 为 128D，其他 LitePT 类输出通常为 72D。统一 LitePT 主干并不意味着读出维度和参数量完全一致；4.5 的 linear 对比应结合 4.4 的配对/随机维度控制，而非单凭 4.5 归因。
4. **单 seed 无置信区间。** 小于约 1 pp 的差异（例如 DALES 强方法 FT，Vaihingen CVRL/VPSD FT，OpenGF CVRL/RRCL linear）应报告为观察值而非统计显著性结论。建议对 CVRL、最强单任务分支和各数据集 FT 最优基线追加多 seed，尤其针对论文中的关键差距。
5. **论文正文的保守结论：** CVRL 在三种 ALS 数据集上提供最强的线性可读语义表示，并在 decoder/FT 下维持强竞争力；充分微调后没有稳定压过所有单任务或通用自监督基线。这样表述与全部 81 组结果一致。

逐轮验证 mIoU、完整测试指标及所有类别的 IoU/Precision/Recall/F1 可由同目录 `full_downstream_parsed.json` 复核；解析脚本为 `paper_cvrl/analyze_section45_logs.py`。
