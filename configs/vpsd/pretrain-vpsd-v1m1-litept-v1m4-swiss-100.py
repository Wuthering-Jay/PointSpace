# Swiss VPSD control for CVRL section 4.3.
# Standalone configuration; intentionally matches the CVRL-v1m1 Swiss-100
# encoder, input, augmentations, grid, optimizer and 25-epoch budget.

# -----------------------------------------------------------------------------
# 0. Paths and feature definition
# -----------------------------------------------------------------------------
data_root = r"G:\data\swiss\joint_tiles_100"
train_pointcloud_path = rf"{data_root}\pointcloud"
dino_feature_path = rf"{data_root}\dino_feature"
pca_model_path = (
    rf"{data_root}\dino_feature_pca128\dino_pca_model.safetensors"
)
audit_pointcloud_path = rf"{data_root}\Swiss-Audit\pointcloud"

grid_size = 0.25
feature_keys = ("coord", "intensity")
in_channels = 4
ignore_index = -1

# visual: fixed-PCA 128D permanent representation (default paper feature)
# visual_native: native DINO-aligned 1024D representation before PCA
# backbone: pre-projector P2/P3/P4 concatenation (diagnostic only)
feature_source = "visual"
feature_output_dir = (
    rf"{data_root}\Swiss-Audit\features\vpsd_pca128"
    if feature_source in {"visual", "projected"}
    else rf"{data_root}\Swiss-Audit\features\vpsd_native1024"
    if feature_source == "visual_native"
    else rf"{data_root}\Swiss-Audit\features\vpsd_backbone"
)

# -----------------------------------------------------------------------------
# 1. Runtime and optimization
# -----------------------------------------------------------------------------
save_path = "exp/vpsd/swiss/pretrain-v1m1-litept-v1m4-100"
weight = f"{save_path}/model/model_last.pth"
resume = True
evaluate = False
test_only = False
seed = 42

num_worker = 4
prefetch_factor = 2
persistent_workers = True
batch_size_train = 16
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
wandb_project = "pointspace-vpsd"
wandb_key = None

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

# -----------------------------------------------------------------------------
# 2. VPSD: native 1024D supervision, fixed-PCA 128D permanent output
# -----------------------------------------------------------------------------
model = dict(
    type="VPSD-v1m1",
    distill_level=2,
    level_channels=(36, 72, 144, 252, 504),
    teacher_channels=1024,
    visual_channels=128,
    pca_path=pca_model_path,
    require_pca=True,
    pca_output_normalize=True,
    pca_output_scale=128**0.5,
    edge_weight="sqrt_count",
    sample_balanced=True,
    validate_mapping=False,
    fuse_deeper_features=True,
    projector_hidden_channels=1024,
    projector_checkpoint=True,
    aggregate_channel_chunk_size=64,
    projector_patch_chunk_size=4096,
    normalize_dtype="amp",
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
        checkpoint_mlp=True,
        mask_token=False,
        enc_mode=True,
        freeze_encoder=False,
        shift_coords=None,
        jitter_coords=None,
        rescale_coords=None,
    ),
)

# -----------------------------------------------------------------------------
# 3. Hooks and feature export
# -----------------------------------------------------------------------------
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="RuntimeInfoHook"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter", interval=10),
    dict(type="CacheCleaner", time_multiplier=2.5, step_clean_interval=200),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=True),
]

train = dict(type="DefaultTrainer")
test = dict(type="VPSDFeatureTester", verbose=True)

feature_dtype = "float16"
normalize_feature = False
feature_aggregate_on_gpu = False
feature_overwrite = False
# Only store the fixed, class/visibility-stratified audit rows. The model still
# sees complete GridSample fragments, but 1024D disk output stays compact.
feature_index_dir = rf"{data_root}\Swiss-Audit\visibility"
feature_index_key = "audit_index"

# -----------------------------------------------------------------------------
# 4. Data
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
    dict(type="CompactImagePatches"),
]

data = dict(
    train=dict(
        type="LasImageDataset",
        split="train",
        data_root=data_root,
        data_path=train_pointcloud_path,
        dino_feature_path=dino_feature_path,
        test_mode=False,
        loop=1,
        transform=train_transform,
        post_transform=[
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=(
                    "coord",
                    "grid_coord",
                    "dino_feature",
                    "image_patch_index",
                    "image_valid",
                    "dino_offset",
                ),
                feat_keys=feature_keys,
            ),
        ],
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
