# Section 4.5 MSC baseline: original global point-pair InfoNCE and masked
# intensity reconstruction (RGB replacement), with LitePT.

data_root = r"G:\data\swiss\joint_tiles_100"
train_pointcloud_path = rf"{data_root}\pointcloud"
audit_pointcloud_path = rf"{data_root}\Swiss-Audit\pointcloud"
feature_output_dir = rf"{data_root}\Swiss-Audit\features\msc_litept"
save_path = "exp/baselines/swiss/pretrain-msc-litept-v1m4-100"
weight = f"{save_path}/model/model_last.pth"
resume = True

grid_size = 0.25
feature_keys = ("coord", "intensity")
in_channels = 4
num_classes = 1
class_names = ("unlabeled",)
ignore_index = -1

evaluate = False
test_only = False
seed = 42
num_worker = 4
prefetch_factor = 2
val_prefetch_factor = 1
persistent_workers = True
batch_size_train = 16
batch_size_val = 4
batch_size_test = 4
gradient_accumulation_steps = 4
epoch = 20
clip_grad = 3.0
sync_bn = False
enable_amp = True
amp_dtype = "bfloat16"
find_unused_parameters = False
mix_prob = 0.0
enable_wandb = False
wandb_project = "pointspace-cvrl-45"
wandb_key = None

model = dict(
    type="MSC-LitePT-v1m1",
    backbone_out_channels=72,
    mask_group_size=4.0,
    mask_rate=0.4,
    matching_max_k=8,
    matching_max_radius=0.75,
    matching_max_pair=8192,
    temperature=0.4,
    nce_chunk_size=1024,
    reconstruct_normal=False,
    reconstruct_intensity=True,
    normal_k=32,
    view1_mix_prob=0.8,
    view2_mix_prob=0.0,
    contrast_weight=1.0,
    reconstruct_weight=1.0,
    backbone=dict(
        type="LitePT-v1m4",
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
        dec_depths=(1, 1, 1, 1),
        dec_channels=(72, 72, 144, 252),
        dec_num_head=(4, 4, 8, 14),
        dec_patch_size=(192, 192, 192, 192),
        dec_conv=(True, True, True, True),
        dec_attn=(False, False, False, False),
        dec_rope_freq=(100.0, 100.0, 100.0, 100.0),
        dec_rope_enable=True,
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
        mask_token=False,
        enc_mode=False,
        freeze_encoder=False,
        shift_coords=None,
        jitter_coords=None,
        rescale_coords=None,
        checkpoint_mlp=True,
    ),
)

base_lr = 1e-3
optimizer = dict(type="AdamW", lr=base_lr, weight_decay=0.05)
scheduler = dict(type="OneCycleLR", max_lr=base_lr, pct_start=0.05,
                 anneal_strategy="cos", div_factor=10.0,
                 final_div_factor=1000.0)
param_dicts = [dict(keyword="block", lr=1e-4)]

hooks = [
    dict(type="CheckpointLoader"),
    dict(type="RuntimeInfoHook"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter", interval=10),
    dict(type="CacheCleaner", time_multiplier=2.5, step_clean_interval=200),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="EarlyStoppingHook", monitor="loss", mode="min", patience=3,
         min_delta_percent=0.5, warmup_epochs=2),
    dict(type="BackboneCheckpointExporter", filename="backbone_last.pth"),
    dict(type="PreciseEvaluator", test_last=True),
]
train = dict(type="DefaultTrainer")
test = dict(type="MSCFeatureTester", verbose=True)
feature_source = "backbone"
feature_dtype = "float16"
normalize_feature = False
feature_aggregate_on_gpu = False
feature_overwrite = False
feature_index_dir = rf"{data_root}\Swiss-Audit\visibility"
feature_index_key = "audit_index"

# Crop once before duplication, preserving substantial overlap between views.
base_transform = [
    dict(type="ZPercentileCenterShift", percentile=2.0),
    dict(type="RobustLogIntensity"),
    dict(type="SphereCrop", point_max=220000, mode="random"),
]
view_transform = [
    dict(type="Copy", keys_dict={"coord": "origin_coord"}),
    dict(type="Update", keys_dict={
        "index_valid_keys": ["coord", "intensity", "origin_coord"]
    }),
    dict(type="RandomDropout", dropout_ratio=0.1,
         dropout_application_ratio=0.2),
    dict(type="RandomRotate", angle=[-1, 1], axis="z",
         center=[0, 0, 0], p=0.5),
    dict(type="RandomScale", scale=[0.9, 1.1]),
    dict(type="RandomFlip", p=0.5),
    dict(type="RandomJitter", sigma=0.005, clip=0.02),
    dict(type="GridSample", grid_size=grid_size, hash_type="fnv",
         mode="train", return_grid_coord=True),
    dict(type="SphereCrop", point_max=110000, mode="random"),
    dict(type="ToTensor"),
    dict(type="Collect", keys=("origin_coord", "coord", "grid_coord"),
         feat_keys=feature_keys),
]

data = dict(
    num_classes=num_classes,
    ignore_index=ignore_index,
    names=class_names,
    train=dict(
        type="MSCLasPairDataset", split="train", data_root=data_root,
        data_path=train_pointcloud_path, class_weight=None,
        weighted_sampler=False, test_mode=False, loop=1,
        base_transform=base_transform,
        view1_transform=view_transform,
        view2_transform=view_transform,
    ),
    val=dict(
        type="MSCLasPairDataset", split="val", data_root=data_root,
        data_path=train_pointcloud_path, class_weight=None,
        weighted_sampler=False, test_mode=False, loop=1,
        base_transform=base_transform,
        view1_transform=view_transform,
        view2_transform=view_transform,
    ),
    test=dict(
        type="LasDataset", split="test", data_path=audit_pointcloud_path,
        test_mode=True, ignore_index=ignore_index,
        transform=[
            dict(type="ZPercentileCenterShift", percentile=2.0),
            dict(type="RobustLogIntensity"),
            dict(type="GridSample", grid_size=grid_size, hash_type="fnv",
                 mode="test", return_grid_coord=True, return_inverse=True),
        ],
        post_transform=[
            dict(type="ToTensor"),
            dict(type="Collect", keys=("coord", "grid_coord", "index"),
                 feat_keys=feature_keys),
        ],
    ),
)
