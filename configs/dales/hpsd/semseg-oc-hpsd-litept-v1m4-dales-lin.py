_base_ = ["./semseg-hpsd-litept-v1m4-dales-base.py"]

weight = (
    "exp/hubei/hpsd/pretrain-oc-hpsd-litept-v1m4-native1024/"
    "model/model_last.pth"
)
save_path = "exp/dales/hpsd/semseg-oc-hpsd-litept-v1m4-lin"
pred_save_dir = r"E:\data\DALES\dales_las\tile\pred_oc_hpsd"
writer = dict(save_dir=pred_save_dir)

model = dict(
    backbone_out_channels=1008,
    backbone=dict(type="LitePT-v1m4", enc_mode=True),
    freeze_backbone=True,
)

optimizer = dict(type="AdamW", lr=2e-3, weight_decay=2e-3)
param_dicts = []
