# DALES 下游完整实验分析（10 epoch × 5 loop）

本报告读取 `exp/downstream/dales/{method}-{probe}/train.log`。27/27 组均有最终 `Test result`，未发现日志中的 `ERROR`。下表均为最终测试 mIoU（%）；模型选择依据是验证 mIoU 最优的 checkpoint，而非最后一轮。`lin`、`dec`、`ft` 分别为 linear probe、decoder probe、full fine-tuning。

| 预训练方法 | Linear | Decoder | Fine-tuning |
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

## 主要发现

1. CVRL 的线性语义可分性最强：比 VPSD 高 5.34 pp、比 RRCL 高 10.60 pp、比 MSC 高 15.37 pp。这是本组结果中最清晰的互补表征证据。
2. 充分微调后，CVRL 达到 79.28%，比 Scratch 高 21.22 pp，但低于 RRCL 0.68 pp、VPSD 0.41 pp、MSC 0.31 pp。不能据此声称 CVRL 在 DALES 的 FT 精度全面领先；这些小差距还需跨种子检验。
3. Decoder probe 的领先者是 MSC（76.89%）。CVRL（76.31%）与 RRCL（76.62%）、Sonata（76.72%）、VPSD（76.18%）处于相近区间。此协议下，新增 decoder 能缩小各强预训练表示之间的差距。
4. CVRL FT 的建筑 IoU 为 96.75%、汽车为 83.07%、植被为 92.86%；卡车仅 34.54%，低于 RRCL 的 40.85%。因此总体 mIoU 受少数类，特别是卡车影响，不宜只凭主要类别判断迁移优劣。
5. Concerto FT 的验证 mIoU 在第 2 轮达到 53.93%，至第 10 轮跌至 6.42%；D-DITR FT 第 7 轮达到 69.79%，此后明显回落。最终测试分别使用较早的最佳 checkpoint（54.61%、69.08%），这些轨迹提示迁移不稳定，不能把其低分简单解释成方法本身上限。

## 解释边界与后续写法

- 这 27 组是单随机种子，微小 FT 差异没有显著性结论。论文应突出 CVRL 在 frozen linear probe 的优势，同时如实报告 FT 与较强基线持平或略低。
- 当前配置的 `val_data_dir` 与 `test_data_dir` 指向同一批 DALES held-out 文件；test 是该批文件的多 fragment 精确推理，不是独立于 checkpoint 选择的数据集。应在实验协议中明确这一点，避免称为独立测试集。
- 其余预训练方法的弱迁移结果可作为当前统一 ALS 协议下的观测，但不宜归因为某个方法的本质缺陷。后续可把最佳轮数、学习曲线和类别 IoU 一并放入补充材料。

## 下一步配置

仅为 OpenGF 和 Vaihingen 启用验证 mIoU 早停：相对最优值至少提高 0.25% 才重置耐心，`patience=3`、`warmup_epochs=2`。OpenGF 改为 20 epoch × 3 train loop；Vaihingen 保持 10 epoch × 5 train loop。DALES 完成的配置保持不变。早停后仍运行最佳 checkpoint 的最终精确测试。
