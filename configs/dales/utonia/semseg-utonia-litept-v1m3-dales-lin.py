_base_ = ["./semseg-utonia-litept-v1m3-dales-base.py"]

save_path = "exp/dales/utonia/semseg-litept-v1m3-lin"

model = dict(
    backbone_out_channels=1008,
    backbone=dict(
        enc_mode=True,
        freeze_encoder=False,
    ),
    freeze_backbone=True,
)

optimizer = dict(type="AdamW", lr=2e-3, weight_decay=2e-3)
param_dicts = [dict(keyword="block", lr=2e-4)]
