# HPSD / OC-HPSD 的 DALES 下游评估

本目录使用同一套 DALES 数据划分、输入特征、随机种子、增强、优化器和训练预算，对原始 HPSD 与 OC-HPSD 进行成对比较。HPSD 预训练时使用 `coord + intensity + echo` 六维输入，因此下游也保持相同输入，并对 intensity 使用相同的稳健对数归一化，避免 embedding stem 因输入维数变化而无法加载。

`semseg-hpsd-litept-v1m4-dales-base.py` 是完整公共配置，默认等价于原始 HPSD 的 full fine-tuning。原始 HPSD 与 OC-HPSD 分别提供 `lin`、`dec` 和 `ft` 三种协议。`lin` 使用 encoder-only LitePT-v1m4，冻结整个 backbone，仅训练线性分类头；`DefaultSegmentorV2` 会把五层 encoder 特征回填到输入点并拼接为 1008 维。`dec` 使用具有相同 encoder 的 LitePT-v1m3，冻结 encoder，只训练随机初始化的 decoder 和分割头。`ft` 同样使用 LitePT-v1m3 encoder-decoder，但允许全部参数参与微调。`semseg-random-litept-v1m4-dales-ft.py` 是不加载预训练权重的随机初始化基线。

建议先运行 linear probe，因为它最直接反映冻结表示质量；随后运行 decoder probe，检查固定 encoder 在允许任务解码器适配时的表现；最后运行 full fine-tuning，比较最终任务上限。成对实验应保持相同 seed，至少补充三个 seed 后报告均值和标准差。

运行示例：

```powershell
python tools/train.py --config-file configs/dales/hpsd/semseg-hpsd-litept-v1m4-dales-lin.py
python tools/train.py --config-file configs/dales/hpsd/semseg-oc-hpsd-litept-v1m4-dales-lin.py

python tools/train.py --config-file configs/dales/hpsd/semseg-hpsd-litept-v1m4-dales-dec.py
python tools/train.py --config-file configs/dales/hpsd/semseg-oc-hpsd-litept-v1m4-dales-dec.py

python tools/train.py --config-file configs/dales/hpsd/semseg-hpsd-litept-v1m4-dales-ft.py
python tools/train.py --config-file configs/dales/hpsd/semseg-oc-hpsd-litept-v1m4-dales-ft.py
```

验证和测试日志除原有 mIoU、mAcc、OA 与逐类别 IoU/Precision/Recall/F1 外，还会输出宏平均 F1、频率加权 IoU 和 Cohen's Kappa。DALES 的 cars、trucks、power lines、fences 和 poles 占比很低，因此模型比较应以 mIoU 和宏平均 F1 为主，OA 与频率加权 IoU只能作为总体点级表现的补充。
