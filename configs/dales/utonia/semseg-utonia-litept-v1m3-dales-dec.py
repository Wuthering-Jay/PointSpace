_base_ = ["./semseg-utonia-litept-v1m3-dales-base.py"]

save_path = "exp/dales/utonia/semseg-litept-v1m3-dec"

model = dict(
    backbone_out_channels=72,
    backbone=dict(
        enc_mode=False,
        freeze_encoder=True,
    ),
    freeze_backbone=False,
)

optimizer = dict(type="AdamW", lr=2e-3, weight_decay=2e-3)
param_dicts = [dict(keyword="block", lr=2e-4)]
