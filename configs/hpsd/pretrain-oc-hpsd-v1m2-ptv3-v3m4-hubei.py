# OC-HPSD-v1m2 PT-v3m4：完整独立配置，不依赖 base。

data_root = r"E:\data\湖北\joint_tiles"
pointcloud_path = rf"{data_root}\pointcloud"
feature_output_dir = rf"{data_root}\hpsd_feature\oc_hpsd_ptv3_v3m4"
grid_size = 0.5
feature_keys = ("coord", "intensity", "echo")
in_channels = 6
# 只有连续点-影像可观测度 q 不低于该值的点才可成为结构化 masking 候选。
min_observability = 0.60

save_path = "exp/hubei/hpsd/pretrain-oc-hpsd-v1m2-ptv3-v3m4-native1024"
weight = (
    "exp/hubei/hpsd/pretrain-oc-hpsd-v1m2-ptv3-v3m4-native1024/"
    "model/model_last.pth"
)
resume = False
evaluate = False
test_only = False
seed = 42

num_worker = 4
batch_size_train = 4
batch_size_test = 4
gradient_accumulation_steps = 4
epoch = 100
clip_grad = 3.0
sync_bn = False
enable_amp = True
amp_dtype = "bfloat16"
find_unused_parameters = False

enable_wandb = False
wandb_project = "pointspace-oc-hpsd"
wandb_key = None
mix_prob = 0.0
param_dicts = None

base_lr = 0.001
optimizer = dict(type="AdamW", lr=base_lr, weight_decay=0.05)
scheduler = dict(
    type="OneCycleLR",
    max_lr=base_lr,
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)

hooks = [
    dict(type="CheckpointLoader"),
    dict(type="RuntimeInfoHook"),
    dict(type="ModelHook"),
    dict(type="ObservationCurriculumHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter", interval=10),
    dict(type="CacheCleaner", time_multiplier=5, step_clean_interval=200),
    dict(type="CheckpointSaver", save_freq=None),
]
train = dict(type="DefaultTrainer")
test = dict(type="HPSDFeatureTester", verbose=True)

feature_source = "projected"
feature_dtype = "float16"
normalize_feature = True
feature_aggregate_on_gpu = True
feature_overwrite = False

model = dict(
    type="OC-HPSD-v1m2",
    distill_level=2,
    level_channels=(36, 72, 144, 288, 576),
    teacher_channels=1024,
    edge_weight="sqrt_count",
    sample_balanced=True,
    validate_mapping=False,
    fuse_deeper_features=True,
    projector_hidden_channels=1024,
    projector_checkpoint=True,
    completion_hidden_channels=1024,
    completion_min_points=1,
    completion_min_mask_fraction=0.1,
    # mask_fraction 达到 0.5 后取满权重；更低覆盖率按比例平滑降权。
    completion_full_weight_fraction=0.5,
    mask_rate=0.30,
    lambda_csc=0.20,
    curriculum_start=0.10,
    curriculum_warmup=0.10,
    masking=dict(
        block_size=4.0,
        min_observability=min_observability,
        min_vertical_span=1.0,
        min_anchor_points=64,
        min_anchor_ratio=0.65,
        max_mask_points=12288,
        fallback_random_block=True,
        # 完整 block 超出剩余预算时，仅在最后一个边界 block 内补足预算。
        fill_partial_block=True,
    ),
    backbone=dict(
        type="PT-v3m4",
        in_channels=in_channels,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(36, 72, 144, 288, 576),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(192, 192, 192, 192, 192),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        shuffle_orders=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        traceable=True,
        mask_token=True,
        enc_mode=True,
    ),
)

data = dict(
    train=dict(
        type="LasImageDataset",
        split="train",
        data_root=data_root,
        test_mode=False,
        loop=1,
        transform=[
            dict(type="ZPercentileCenterShift", percentile=2.0),
            dict(type="RobustLogIntensity"),
            dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="SphereCrop", point_max=60000, mode="random"),
            dict(type="CompactImagePatches"),
        ],
        post_transform=[
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=(
                    "coord",
                    "grid_coord",
                    "dino_feature",
                    "image_pixel_coord",
                    "image_patch_index",
                    "image_valid",
                    "image_observability",
                    "dino_offset",
                    "image_source_patch_index",
                    "image_original_size",
                    "image_padded_size",
                    "image_feature_size",
                    "image_patch_size",
                ),
                feat_keys=feature_keys,
            ),
        ],
    ),
    test=dict(
        type="LasDataset",
        split="test",
        data_path=pointcloud_path,
        test_mode=True,
        ignore_index=-1,
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
