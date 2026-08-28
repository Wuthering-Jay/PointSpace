# OC-HPSD LitePT-v1m4：观测条件 HPSD + 结构化输入 masking + CSC。
# 本配置完整独立，不依赖 base；最终方法在一个连续训练 run 中完成。

# -------------------------------------------------------
# 0. 数据路径与点输入特征
# -------------------------------------------------------
data_root = r"E:\data\湖北\joint_tiles"
pointcloud_path = r"E:\data\云南\data\tile"
feature_output_dir = r"E:\data\云南\hpsd_feature\oc_hpsd_litept_v1m4"
grid_size = 0.5
feature_keys = ("coord", "intensity", "echo")
in_channels = 6
# 只有连续点-影像可观测度 q 不低于该值的点才可成为结构化 masking 候选。
min_observability = 0.60

# -------------------------------------------------------
# 1. Checkpoint 与运行控制
# -------------------------------------------------------
save_path = "exp/hubei/hpsd/pretrain-oc-hpsd-litept-v1m4-native1024"
weight = (
    "exp/hubei/hpsd/pretrain-oc-hpsd-litept-v1m4-native1024/"
    "model/model_last.pth"
)
resume = False
evaluate = False
test_only = False
seed = 42

# -------------------------------------------------------
# 2. Batch、训练循环与精度
# -------------------------------------------------------
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

# -------------------------------------------------------
# 3. 日志、优化器与 hook
# -------------------------------------------------------
enable_wandb = False
wandb_project = "pointspace-oc-hpsd"
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
    # 按全局 step 更新 mask rate 与 CSC 权重；断点恢复无需保存临时阶段状态。
    dict(type="ObservationCurriculumHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter", interval=10),
    dict(type="CacheCleaner", time_multiplier=2.5, step_clean_interval=100),
    dict(type="CheckpointSaver", save_freq=None),
]

train = dict(type="DefaultTrainer")
test = dict(type="HPSDFeatureTester", verbose=True)

# -------------------------------------------------------
# 4. 测试特征导出
# -------------------------------------------------------
feature_source = "projected"
feature_dtype = "float16"
normalize_feature = True
feature_aggregate_on_gpu = False
feature_overwrite = False

# -------------------------------------------------------
# 5. OC-HPSD 模型
# -------------------------------------------------------
model = dict(
    type="OC-HPSD-v1m1",
    distill_level=2,
    level_channels=(36, 72, 144, 252, 504),
    teacher_channels=1024,
    edge_weight="sqrt_count",
    sample_balanced=True,
    validate_mapping=False,
    fuse_deeper_features=True,
    projector_hidden_channels=1024,
    # HPSD 与 CSC projector 均在反向时重算，保持 DINO 原生 1024 维。
    projector_checkpoint=True,
    # CSC 只读取 level 2 之后的 F3/F4，上采样后映射到原生 1024D DINO。
    completion_hidden_channels=1024,
    completion_min_points=1,
    completion_min_mask_fraction=0.5,
    # 同一训练 run：前 10% 纯 HPSD，之后 10% 线性打开 mask 与 CSC。
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
        max_mask_points=8192,
        fallback_random_block=True,
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
        # 只重算纯 tensor 的深层 MLP 分支，保留随机状态与前向语义。
        checkpoint_mlp=True,
        # simulated-missing 点在 embedding 后由 learned token 替换输入属性。
        mask_token=True,
        enc_mode=True,
    ),
)

# -------------------------------------------------------
# 6. 数据
# -------------------------------------------------------
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
    # 导出阶段只读取点云，不生成 mask，也不需要 DINO/correspondence。
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
