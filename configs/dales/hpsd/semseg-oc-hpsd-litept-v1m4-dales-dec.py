_base_ = ["./semseg-hpsd-litept-v1m4-dales-base.py"]

weight = (
    "exp/hubei/hpsd/pretrain-oc-hpsd-litept-v1m4-native1024/"
    "model/model_last.pth"
)
save_path = "exp/dales/hpsd/semseg-oc-hpsd-litept-v1m4-dec"
pred_save_dir = r"E:\data\DALES\dales_las\tile\pred_oc_hpsd"
writer = dict(save_dir=pred_save_dir)

model = dict(
    backbone_out_channels=72,
    backbone=dict(type="LitePT-v1m3", enc_mode=False, freeze_encoder=True),
    freeze_backbone=False,
)

optimizer = dict(type="AdamW", lr=2e-3, weight_decay=2e-3)
param_dicts = [dict(keyword="block", lr=2e-4)]
