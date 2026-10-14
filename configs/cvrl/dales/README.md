# CVRL-v1m1 在 DALES 上的微调协议

本目录包含三个彼此独立、没有 `_base_` 依赖的配置。它们默认加载 Swiss
CVRL-v1m1 checkpoint：

`exp/cvrl/swiss/pretrain-v1m1-litept-v1m4-full/model/model_last.pth`

由于该 checkpoint 的 stem 在预训练时接收 `XYZ + intensity`，三个协议都保持
4 通道输入。DALES 没有可用的 intensity 模态，因此数据流水线通过
`EnsureFeature(overwrite=True)` 将该通道显式置零。这样既保持预训练 stem 的
权重形状，又不会用 echo 或其他人为构造值冒充强度。这里不直接沿用原 DALES
配置的 `XYZ + echo` 五通道输入，否则 stem 权重不兼容，实验也不再是严格的
预训练迁移对比。

## 1. Linear Probe

配置：`semseg-cvrl-v1m1-litept-v1m4-dales-lin.py`

- 整个 CVRL backbone 冻结并固定为 eval 模式；
- 唯一可训练层为一个严格的 `Linear(C, 8)`；
- 修改配置顶部的 `feature_source` 可选择：
  - `"visual"` 或 `"vpsd"`：128D VPSD 特征；
  - `"return"` 或 `"rrcl"`：72D RRCL 特征；
  - `"full"` 或 `"cvrl"`：200D 融合特征。

`feature_channels`、输出目录和实验名会随该选项自动调整。

## 2. Decoder Probe

配置：`semseg-cvrl-v1m1-litept-v1m4-dales-dec.py`

只从 CVRL checkpoint 加载并冻结
`return_backbone.embedding + return_backbone.enc`。新建的 LitePT 原生
GridUnpooling decoder 和 `Linear(72, 8)` 分割头参与训练。配置使用
`dec_depths=(1,1,1,1)`、`dec_conv=(True,True,True,True)`，与 CVRL return
decoder 的规模一致。冻结 encoder 的前向索引缓存与可训练 decoder 的索引
缓存使用独立名称，避免 spconv 反向复用缺少 backward indices 的 eval 缓存。

## 3. Full Fine-Tuning

配置：`semseg-cvrl-v1m1-litept-v1m4-dales-ft.py`

加载完整 200D CVRL backbone，并训练 visual、return、共享 encoder 及新分割头。
backbone 学习率为 `1e-4`，新分割头学习率为 `1e-3`。

三个配置均设置 `CheckpointLoader(required=True)`。路径错误或权重不存在时会
直接报错，而不会静默运行随机初始化实验。
