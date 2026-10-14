# P1 six-dimensional local-shape control on Swiss-100. Run the offline geometry
# utility first; descriptors are targets, not backbone inputs.

# -----------------------------------------------------------------------------
# 0. Paths and task
# -----------------------------------------------------------------------------
data_root = r"G:\data\swiss\joint_tiles_100"
train_pointcloud_path = rf"{data_root}\pointcloud_geom_p1"
audit_pointcloud_path = rf"{data_root}\Swiss-Audit\pointcloud"
feature_output_dir = rf"{data_root}\Swiss-Audit\features\geometry_local_shape"

grid_size = 0.25
feature_keys = ("coord", "intensity")
in_channels = 4
num_classes = 4
class_names = ("S", "F", "M", "L")
ignore_index = -1
geometry_k = 16  # matches existing Swiss offline geometry LAS
backbone_out_channels = 72

# -----------------------------------------------------------------------------
# 1. Runtime
# -----------------------------------------------------------------------------
save_path = "exp/geometry_ssl/swiss/pretrain-local_shape-litept-v1m4-100"
weight = f"{save_path}/model/model_last.pth"
resume = True
# Match the CVRL run exactly: no full 13,530-tile validation sweep per epoch.
# PreciseEvaluator still exports the fixed Swiss-Audit subset after training.
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

epoch = 25
clip_grad = 3.0
sync_bn = False
enable_amp = True
amp_dtype = "bfloat16"
find_unused_parameters = False
mix_prob = 0.0

enable_wandb = False
wandb_project = "pointspace-geometry-ssl"
wandb_key = None

# -----------------------------------------------------------------------------
# 2. Model: same LitePT-v1m4 backbone as independent RRCL
# -----------------------------------------------------------------------------
model = dict(
    type="GeometrySSL",
    objective="local_shape",
    backbone_out_channels=backbone_out_channels,
    smooth_l1_beta=0.1,
    head_dropout=0.1,
    ignore_index=ignore_index,
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

# -----------------------------------------------------------------------------
# 3. Optimization, hooks and feature export
# -----------------------------------------------------------------------------
base_lr = 1e-3
optimizer = dict(type="AdamW", lr=base_lr, weight_decay=0.05)
scheduler = dict(
    type="OneCycleLR",
    max_lr=base_lr,
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = [dict(keyword="block", lr=1e-4)]

hooks = [
    dict(type="CheckpointLoader"),
    dict(type="RuntimeInfoHook"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter", interval=10),
    dict(type="CacheCleaner", time_multiplier=2.5, step_clean_interval=200),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="BackboneCheckpointExporter", filename="backbone_last.pth"),
    dict(type="PreciseEvaluator", test_last=True),
]

train = dict(type="DefaultTrainer")
test = dict(type="GeometryFeatureTester", verbose=True)
feature_source = "backbone"
feature_dtype = "float16"
normalize_feature = False
feature_aggregate_on_gpu = False
feature_overwrite = False
feature_index_dir = rf"{data_root}\Swiss-Audit\visibility"
feature_index_key = "audit_index"

# -----------------------------------------------------------------------------
# 4. Data: same augmentations/crop as CVRL, without image-only tensors
# -----------------------------------------------------------------------------
train_transform = [
    dict(type="ZPercentileCenterShift", percentile=2.0),
    dict(type="RobustLogIntensity"),
    dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2),
    dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
    dict(type="RandomScale", scale=[0.9, 1.1]),
    dict(type="RandomFlip", p=0.5),
    dict(type="RandomJitter", sigma=0.005, clip=0.02),
    dict(
        type="GridSample",
        grid_size=grid_size,
        hash_type="fnv",
        mode="train",
        return_grid_coord=True,
    ),
    dict(type="SphereCrop", point_max=220000, mode="random"),
]

val_transform = [
    dict(type="ZPercentileCenterShift", percentile=2.0),
    dict(type="RobustLogIntensity"),
    dict(
        type="GridSample",
        grid_size=grid_size,
        hash_type="fnv",
        mode="train",
        return_grid_coord=True,
    ),
    dict(type="SphereCrop", point_max=220000, mode="center"),
]

pretrain_post_transform = [
    dict(type="ToTensor"),
    dict(
        type="Collect",
        keys=("coord", "grid_coord", "segment", "return_role", "geom_valid", "local_shape"),
        feat_keys=feature_keys,
    ),
]

data = dict(
    num_classes=num_classes,
    ignore_index=ignore_index,
    names=class_names,
    train=dict(
        type="LasDataset",
        split="train",
        data_root=data_root,
        data_path=train_pointcloud_path,
        class_weight=None,
        weight_sample=0.2,
        weighted_sampler=False,
        test_mode=False,
        loop=1,
        transform=train_transform,
        post_transform=pretrain_post_transform,
    ),
    val=dict(
        type="LasDataset",
        split="val",
        data_root=data_root,
        data_path=train_pointcloud_path,
        class_weight=None,
        weighted_sampler=False,
        test_mode=False,
        loop=1,
        transform=val_transform,
        post_transform=pretrain_post_transform,
    ),
    test=dict(
        type="LasDataset",
        split="test",
        data_path=audit_pointcloud_path,
        test_mode=True,
        ignore_index=ignore_index,
        transform=[
            dict(type="ZPercentileCenterShift", percentile=2.0),
            dict(type="RobustLogIntensity"),
            dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="test",
                return_grid_coord=True,
                return_inverse=True,
            ),
        ],
        post_transform=[
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "index"),
                feat_keys=feature_keys,
            ),
        ],
    ),
)
