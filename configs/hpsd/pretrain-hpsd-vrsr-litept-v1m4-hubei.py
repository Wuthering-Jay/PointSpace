# HPSD-VRSR LitePT-v1m4：DINO 蒸馏与不可视 token 监督传播完整配置。
# 本文件不依赖 base；可直接由 tools/train.py 和 tools/test.py 调用。

data_root = r"E:\data\湖北\joint_tiles"
pointcloud_path = r"E:\data\云南\data\tile"
feature_output_dir = r"E:\data\云南\hpsd_feature\litept_v1m4_vrsr"
grid_size = 0.5
feature_keys = ("coord", "intensity", "echo")
in_channels = 6

save_path = "exp/hubei/hpsd/pretrain-litept-v1m4-vrsr"
# P2 首次训练加载已收敛的 concat-HPSD；CheckpointLoader 默认 strict=False，
# 只缺少新增的 vrsr.* 参数。P3 可把 weight 改为 P2 checkpoint。
weight = "exp/hubei/hpsd/pretrain-litept-v1m4-concat-native1024/model/model_last.pth"
resume = False
evaluate = False
test_only = False
seed = 42

num_worker = 8
batch_size_train = 20
batch_size_test = 1
gradient_accumulation_steps = 4

epoch = 10
clip_grad = 3.0
sync_bn = False
enable_amp = True
amp_dtype = "bfloat16"
find_unused_parameters = False

enable_wandb = False
wandb_project = "pointspace-hpsd-vrsr"
wandb_key = None
mix_prob = 0.0

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
param_dicts = [dict(keyword="block", lr=1e-4)]

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

feature_source = "projected"
feature_dtype = "float16"
normalize_feature = True
feature_aggregate_on_gpu = False
feature_overwrite = False

model = dict(
    type="HPSD-VRSR-v1m1",
    distill_level=2,
    level_channels=(36, 72, 144, 252, 504),
    teacher_channels=1024,
    edge_weight="sqrt_count",
    sample_balanced=True,
    validate_mapping=False,
    fuse_deeper_features=True,
    projector_hidden_channels=1024,
    vrsr=dict(
        # 默认安全地执行 P2；校准通过后加载 P2 checkpoint 并改为 local。
        mode="calibrate",
        propagation_channels=128,
        hidden_channels=256,
        projection_seed=3407,
        source_q=0.6,
        # 第一版只监督 fully-invisible token；不要提前扩展到 mixed token。
        target_q=0.0,
        min_source_points=4,
        min_source_patches=1,
        # 全量审计 p05=0.866、p25=0.923，0.90 可过滤低一致性 source
        # 同时保留充足锚点；仍应在正式训练的 validation 统计上复核。
        source_purity=0.90,
        topk=8,
        temperature=0.1,
        max_sources=512,
        max_targets=1024,
        query_chunk_size=256,
        lambda_cal=0.05,
        lambda_local=0.02,
    ),
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
        loop=10,
        transform=[
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
