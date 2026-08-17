# HPSD PT-v3m4：训练与点特征提取完整配置。
# tools/train.py 和 tools/test.py 均直接读取本文件，不依赖 base 配置。

data_root = r"E:\data\湖北\joint_tiles"
pointcloud_path = rf"{data_root}\pointcloud"
grid_size = 0.5
feature_keys = ("coord", "intensity", "echo")
in_channels = 6

weight = "exp/hubei/hpsd/pretrain-ptv3-v3m4-fusion-native1024/model/model_last.pth"
resume = False
evaluate = False
test_only = False
seed = None
save_path = "exp/hubei/hpsd/pretrain-ptv3-v3m4-fusion-native1024"

num_worker = 4
batch_size_train = 4
batch_size_test = 4  # 全部 GPU 配置值；每 GPU fragment batch = 4 // world_size
gradient_accumulation_steps = 4
epoch = 100
eval_epoch = 100
clip_grad = 3.0

sync_bn = False
enable_amp = True
amp_dtype = "bfloat16"
find_unused_parameters = False
enable_wandb = False
wandb_project = "pointspace-hpsd"
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
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter", interval=10),
    dict(type="CacheCleaner", time_multiplier=5, step_clean_interval=200),
    dict(type="CheckpointSaver", save_freq=None),
]
train = dict(type="DefaultTrainer")
test = dict(type="HPSDFeatureTester", verbose=True)

feature_output_dir = rf"{data_root}\hpsd_feature\ptv3_v3m4_fusion"
feature_source = "projected"
feature_dtype = "float16"
normalize_feature = True
feature_aggregate_on_gpu = True
feature_overwrite = False

model = dict(
    type="HPSD-v1m1",
    # 首个 True 自动推断为参考 level 2。
    fusion_levels=(False, False, True, True, True),
    fusion_channels=512,
    level_weight_init=(0.0, 0.0, 1.0, 0.5, 0.25),
    level_weight_floor=0.05,
    level_channels=(36, 72, 144, 288, 576),
    teacher_channels=1024,
    edge_weight="sqrt_count",
    sample_balanced=True,
    validate_mapping=False,
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
        mask_token=False,
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
            dict(type="CompactDinoPatches"),
        ],
        post_transform=[
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=(
                    "coord",
                    "grid_coord",
                    "dino_feature",
                    "dino_pixel_coord",
                    "dino_patch_index",
                    "dino_valid",
                    "dino_offset",
                    "dino_source_patch_index",
                    "dino_original_size",
                    "dino_padded_size",
                    "dino_feature_size",
                    "dino_patch_size",
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
