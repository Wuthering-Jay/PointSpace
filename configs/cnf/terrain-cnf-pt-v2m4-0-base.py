# ════════════════════════════════════════════════════════════════════════════
# Terrain CNF (Conditional Neural Field) — PT-v2m4 Backbone
#
# 基于机载激光点云地面点的隐式神经表示：
#   - ClassFilter 过滤地面点 (class 1)
#   - TerrainImplicitSampler 将地面点拆分为 support / query
#     (三策略：随机抽稀 + 矩形空洞 + 拉普拉斯地形敏感采样)
#   - Backbone (PT-v2m4) 编码 support → per-point features
#   - DualBranchCNFHead:
#       · PTV2 向量交叉注意力插值 (Softplus 保证 C² 可导)
#       · Base 分支: raw(x,y) + F_query → 低频大走势 pred_base
#       · Detail 分支: γ(x,y) [Fourier PE] + F_query → 高频残差 pred_detail
#   - 多频解耦损失 + detach 梯度隔离
#   - 测试时生成规则密集网格 → 隐式曲面输出
# ════════════════════════════════════════════════════════════════════════════

# -------------------------------------------------------
# 0. Path settings
# -------------------------------------------------------
train_data_dir = r"E:\data\云南遥感中心\第二批\ground-only\disk03\tile\train"
val_data_dir   = r"E:\data\云南遥感中心\第二批\ground-only\disk03\tile\val"
test_data_dir  = r"E:\data\云南遥感中心\第二批\ground-only\disk03\tile\val"
pred_save_dir  = r"E:\data\云南遥感中心\第二批\ground-only\disk03\tile\pred_cnf"
save_path = "exp/cnf/terrain-cnf-pt-v2m4-0-base"

# -------------------------------------------------------
# 1. General settings
# -------------------------------------------------------
grid_size = 0.5
ignore_index = -1
dataset_type = "LasDataset"
# Only keep ground class (class 1 in DALES)
ground_class = [2]
feature_keys = ["coord"]
in_channels = 3

# -------------------------------------------------------
# 2. Checkpoint / run control
# -------------------------------------------------------
weight = "exp/cnf/terrain-cnf-pt-v2m4-0-base/model/model_last.pth"  # path to pretrained weights (None to train from scratch)
resume = True
evaluate = True     # run validation after each epoch
test_only = False
seed = 42

# -------------------------------------------------------
# 3. Resource & batch settings
# -------------------------------------------------------
batch_size_train = 4
batch_size_val = 2
batch_size_test = 1
num_worker = 0
gradient_accumulation_steps = 2

# -------------------------------------------------------
# 4. Training loop
# -------------------------------------------------------
epoch = 10
clip_grad = None

# -------------------------------------------------------
# 5. Precision & performance
# -------------------------------------------------------
enable_amp = True
amp_dtype = "float16"
sync_bn = False
find_unused_parameters = False

# -------------------------------------------------------
# 6. Logging
# -------------------------------------------------------
enable_wandb = False
wandb_project = "pointspace-cnf"
wandb_key = None
mix_prob = 0.0

# -------------------------------------------------------
# 7. Model — DefaultCNF (Backbone + DualBranchCNFHead)
# -------------------------------------------------------
model = dict(
    type="DefaultCNF",
    reg_weight=1.0,
    backbone=dict(
        type="PT-v2m4",
        in_channels=in_channels,
        patch_embed_depth=1,
        patch_embed_channels=24,
        patch_embed_groups=6,
        patch_embed_neighbours=24,
        enc_depths=(2, 2, 2),
        enc_channels=(36, 72, 144),
        enc_groups=(6, 12, 24),
        enc_neighbours=(32, 32, 32),
        dec_depths=(1, 1, 1),
        dec_channels=(24, 36, 72),
        dec_groups=(4, 6, 12),
        dec_neighbours=(32, 32, 32),
        grid_sizes=(
            3 * grid_size,
            7.5 * grid_size,
            18.75 * grid_size,
        ),
        attn_qkv_bias=True,
        pe_multiplier=False,
        pe_bias=True,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        enable_checkpoint=False,
        unpool_backend="interp",
    ),
    head=dict(
        type="DualBranchCNFHead",
        backbone_out_channels=24,   # must match backbone dec_channels[0]
        query_dim=2,                # predict z from (x, y)
        num_targets=1,              # scalar output (terrain height)
        k_neighbors=16,             # KNN for IDW anchor + grouped features
        hidden_dim=256,             # fuse layer hidden dimension
        num_freqs=6,                # Fourier PE octaves for detail branch
        base_hidden_dims=[128, 64],
        detail_hidden_dims=[128, 64],
    ),
    criteria=None,  # use built-in multi-frequency loss
)

# -------------------------------------------------------
# 8. Optimizer & scheduler
# -------------------------------------------------------
optimizer = dict(type="AdamW", lr=1e-3, weight_decay=1e-2)
scheduler = dict(
    type="CosineAnnealingLR",
    total_steps=epoch,
)
param_dicts = None

# -------------------------------------------------------
# 9. Hooks
# -------------------------------------------------------
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter", interval=10),
    dict(type="CacheCleaner", time_multiplier=5, step_clean_interval=None),
    dict(type="CnfEvaluator", log_interval=10),
    dict(type="CheckpointSaver", save_freq=None),
]

# -------------------------------------------------------
# 10. Train / test engine & writer
# -------------------------------------------------------
train = dict(type="DefaultTrainer")
test = dict(type="CnfTester")
writer = dict(
    type="LASWriter",
    save_dir=pred_save_dir,
    source_dir=test_data_dir,   # borrow CRS/scale/offset from source LAS
    classification=1,           # mark CNF output as ground (DALES class 2); set None to skip
)

# CNF test-time parameters
query_dim = 2                  # query coordinate dimensionality
query_resolution = 0.25        # output grid spacing (metres)
query_batch_size = 50_000    # max query points per forward pass
compute_derivatives = True    # compute slope & curvature maps

# -------------------------------------------------------
# 11. Dataset
# -------------------------------------------------------
data = dict(
    num_classes=1,          # CNF is a regression task
    ignore_index=ignore_index,
    names=["terrain_z"],
    train=dict(
        type=dataset_type,
        split="train",
        data_path=train_data_dir,
        test_mode=False,
        loop=1,
        transform=[
            # 1. Centre Z near ground level
            dict(type="ZPercentileCenterShift", percentile=2.0),
            # 2. Keep only ground points
            dict(type="ClassFilter", keep_classes=ground_class, class_key="segment"),
            # 3. Light augmentation (before query split)
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            # 4. Split into support / query
            dict(
                type="TerrainImplicitSampler",
                random_ratio=0.1,
                feature_ratio=0.1,
                max_blocks=5,
                block_size_range=(10.0, 50.0),
                feature_resolution=2.0,
                max_query_ratio=0.6,
            ),
            # 5. Compute grid_coord for backbone (no downsampling!)
            # dict(type="GridCoordinate", grid_size=grid_size),
        ],
        post_transform=[
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=["coord", "query_coord", "query_gt", "query_gt_low"],
                offset_keys_dict=dict(
                    offset="coord",
                    query_offset="query_coord",
                ),
                feat_keys=feature_keys,
            ),
        ],
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_path=val_data_dir,
        test_mode=False,
        loop=1,
        transform=[
            dict(type="ZPercentileCenterShift", percentile=2.0),
            dict(type="ClassFilter", keep_classes=ground_class, class_key="segment"),
            dict(
                type="TerrainImplicitSampler",
                random_ratio=0.1,
                feature_ratio=0.1,
                max_blocks=5,
                block_size_range=(10.0, 50.0),
                feature_resolution=2.0,
                max_query_ratio=0.6,
            ),
            # dict(type="GridCoordinate", grid_size=grid_size),
        ],
        post_transform=[
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=["coord", "query_coord", "query_gt", "query_gt_low"],
                offset_keys_dict=dict(
                    offset="coord",
                    query_offset="query_coord",
                ),
                feat_keys=feature_keys,
            ),
        ],
    ),
    test=dict(
        type=dataset_type,
        split="test",
        data_path=test_data_dir,
        test_mode=True,
        # Test: ClassFilter + GridSample (test mode) for backbone encoding
        # Query grid is generated by CnfTester at inference time
        transform=[
            dict(type="ZPercentileCenterShift", percentile=2.0),
            dict(type="ClassFilter", keep_classes=ground_class, class_key="segment"),
            # mode="train": returns one downsampled dict, not a list of fragments.
            # For CNF we only need one clean support representation — no multi-fragment
            # averaging is required (unlike semseg where every raw point must be covered).
            # dict(
            #     type="GridSample",
            #     grid_size=grid_size,
            #     hash_type="fnv",
            #     mode="train",
            #     return_grid_coord=True,
            # ),
        ],
        aug_transform=[
            [dict(type="RandomScale", scale=[1, 1])],
        ],
        post_transform=[
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=["coord"],
                feat_keys=feature_keys,
            ),
        ],
    ),
)
