# -------------------------------------------------------
# DALES Utonia self-supervised pretraining without images.
# Backbone: LitePT-v1m3
# Input feature: xyz + echo
# This config is intentionally self-contained and does not depend on _base_.
# -------------------------------------------------------

# data path
train_data_dir = r"E:\data\DALES\dales_las\tile\train"
save_path = "exp/dales/utonia/pretrain-litept-v1m3-dales-xyz-echo"

# dataset
dataset_type = "LasDataset"
required_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8]
ignore_index = -1
grid_size = 0.5
in_channels = 5

# misc
weight = None
resume = False
test_only = False
seed = None
batch_size = 16
batch_size_train = 16
batch_size_val = 4
batch_size_test = 4
num_worker = 8
gradient_accumulation_steps = 2
mix_prob = 0.0
clip_grad = 1.0
empty_cache = True
empty_cache_per_epoch = False
sync_bn = False
enable_amp = True
amp_dtype = "bfloat16"
evaluate = False
eval_epoch = 100
find_unused_parameters = True
enable_wandb = False
wandb_project = "pointspace-dales"
wandb_key = None

# model
model = dict(
    type="Utonia-v1m1",
    # Image branch is disabled by enc2d_loss_weight=0. These fields are kept only
    # because Utonia-v1m1 requires them in the constructor.
    patch_h=1,
    patch_w=1,
    image_weight_name="none",
    image_weight_path="none",
    backbone_out_channels=504,
    embedding_channels=64,
    student_pretrained=False,
    enc2d_upcast_level=0,
    backbone=dict(
        type="LitePT-v1m3",
        in_channels=in_channels,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(36, 72, 144, 252, 504),
        enc_num_head=(2, 4, 8, 14, 28),
        enc_patch_size=(192, 192, 192, 192, 192),
        enc_conv=(True, True, True, False, False),
        enc_attn=(False, False, False, True, True),
        enc_rope_freq=(100.0, 100.0, 100.0, 100.0, 100.0),
        dec_depths=(0, 0, 0, 0),
        dec_channels=(72, 72, 144, 252),
        dec_num_head=(4, 4, 8, 14),
        dec_patch_size=(192, 192, 192, 192),
        dec_conv=(False, False, False, False),
        dec_attn=(False, False, False, False),
        dec_rope_freq=(100.0, 100.0, 100.0, 100.0),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        shuffle_orders=True,
        enable_flash=True,
        traceable=True,
        mask_token=True,
        enc_mode=True,
        freeze_encoder=False,
        shift_coords=None,
        jitter_coords=1.1,
        rescale_coords=1.2,
    ),
    teacher_custom=dict(
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
    ),
    head_in_channels=144 + 252 + 504,
    head_hidden_channels=2048,
    head_embed_channels=256,
    head_num_prototypes=2048,
    enc2d_head_in_channels=1,
    enc2d_head_hidden_channels=1,
    enc2d_head_embed_channels=1,
    enc2d_head_num_prototypes=1,
    num_global_view=2,
    num_local_view=4,
    mask_size_start=8,
    mask_size_base=20,
    mask_size_warmup_ratio=0.05,
    mask_ratio_start=0.3,
    mask_ratio_base=0.7,
    mask_ratio_warmup_ratio=0.05,
    mask_jitter=0.5,
    teacher_temp_start=0.04,
    teacher_temp_base=0.07,
    teacher_temp_warmup_ratio=0.05,
    student_temp=0.1,
    mask_loss_weight=1 / 4,
    roll_mask_loss_weight=1 / 4,
    unmask_loss_weight=2 / 4,
    enc2d_loss_weight=0.0,
    momentum_base=0.994,
    momentum_final=1,
    match_max_k=8,
    match_max_r=1.0,
    up_cast_level=2,
    enc2d_cos_shift=False,
    sonata_model_type="online",
)

# scheduler
epoch = 10
base_lr = 0.002
lr_decay = 0.9
base_wd = 0.04
final_wd = 0.2

dec_depths = model["backbone"]["enc_depths"]
param_dicts = [
    dict(
        keyword=f"enc{e}.block{b}.",
        lr=base_lr * lr_decay ** (sum(dec_depths) - sum(dec_depths[:e]) - b - 1),
    )
    for e in range(len(dec_depths))
    for b in range(dec_depths[e])
]
del dec_depths

optimizer = dict(type="AdamW", lr=base_lr, weight_decay=base_wd)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[base_lr] + [g["lr"] for g in param_dicts],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)

# transforms
pretrain_transform = [
    dict(
        type="Update",
        keys_dict={
            "index_valid_keys": (
                "coord",
                "origin_coord",
                "echo",
                "segment",
                "instance",
            )
        },
    ),
    dict(type="ZPercentileCenterShift", percentile=2.0),
    dict(type="Copy", keys_dict={"coord": "origin_coord"}),
    dict(type="GridSample", grid_size=grid_size, hash_type="fnv", mode="train"),
    dict(type="RandomDropEcho", drop_ratio=1.0, drop_application_ratio=0.2),
    dict(type="RandomDropEcho", drop_ratio=0.1, drop_application_ratio=0.5),
    dict(
        type="MultiViewGenerator",
        view_keys=("coord", "origin_coord", "echo"),
        global_view_num=2,
        global_view_scale=(0.4, 1.0),
        local_view_num=4,
        local_view_scale=(0.1, 0.4),
        global_shared_transform=[],
        global_transform=[
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.8),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
        ],
        local_transform=[
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.8),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
        ],
        max_size=65536,
        enc2d_max_size=65536,
        enc2d_scale=(0.8, 1),
    ),
    dict(type="ToTensor"),
    dict(type="Update", keys_dict={"grid_size": grid_size}),
    dict(
        type="Collect",
        keys=(
            "global_origin_coord",
            "global_coord",
            "global_offset",
            "local_origin_coord",
            "local_coord",
            "local_offset",
            "grid_size",
            "name",
        ),
        offset_keys_dict=dict(),
        global_feat_keys=("global_coord", "global_echo"),
        local_feat_keys=("local_coord", "local_echo"),
    ),
]

data = dict(
    train=dict(
        type=dataset_type,
        split="train",
        data_path=train_data_dir,
        required_class=required_classes,
        remap_class=True,
        ignore_index=ignore_index,
        test_mode=False,
        loop=5,
        transform=pretrain_transform,
    )
)

hooks = [
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="WeightDecaySchedular", base_value=base_wd, final_value=final_wd),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter", interval=5),
    dict(type="CacheCleaner", time_multiplier=5, step_clean_interval=200),
    dict(type="CheckpointSaver", save_freq=10),
]

train = dict(type="DefaultTrainer")
test = dict(type="SemSegTester", verbose=True)
