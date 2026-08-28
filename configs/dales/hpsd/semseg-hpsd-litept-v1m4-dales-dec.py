_base_ = ["./semseg-hpsd-litept-v1m4-dales-base.py"]

save_path = "exp/dales/hpsd/semseg-hpsd-litept-v1m4-dec"

# 固定预训练 encoder，只训练随机初始化的 decoder 和分割 head。
model = dict(
    backbone_out_channels=72,
    backbone=dict(type="LitePT-v1m3", enc_mode=False, freeze_encoder=True),
    freeze_backbone=False,
)

optimizer = dict(type="AdamW", lr=2e-3, weight_decay=2e-3)
param_dicts = [dict(keyword="block", lr=2e-4)]
