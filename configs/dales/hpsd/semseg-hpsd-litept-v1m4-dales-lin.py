_base_ = ["./semseg-hpsd-litept-v1m4-dales-base.py"]

save_path = "exp/dales/hpsd/semseg-hpsd-litept-v1m4-lin"

# encoder-only 时 DefaultSegmentorV2 将五层特征回填到输入点并 concat，
# 输出通道为 36+72+144+252+504=1008；冻结 backbone 后只训练线性 head。
model = dict(
    backbone_out_channels=1008,
    backbone=dict(type="LitePT-v1m4", enc_mode=True),
    freeze_backbone=True,
)

optimizer = dict(type="AdamW", lr=2e-3, weight_decay=2e-3)
param_dicts = []
