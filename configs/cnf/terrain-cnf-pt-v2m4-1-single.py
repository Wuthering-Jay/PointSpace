# ════════════════════════════════════════════════════════════════════════════
# Terrain CNF — PT-v2m4 Backbone — Dual-KNN Single-Branch Head
#
# 双重 KNN 语义-几何交织架构：
#   - CategoryAwareDownsample: 地面点全密度保留，非地面点体素降采样
#   - TerrainImplicitSampler:  开放限制(max_query_ratio=0.9) + 极端大空洞
#   - Backbone (PT-v2m5) → 全量点(含树冠/建筑) per-point features
#   - SingleBranchCNFHead (Dual-KNN):
#       · Ground Branch: 仅地面 KNN → IDW z_anchor + feat_anchor
#       · Semantic Branch: 全量 KNN → class_embed + Z-Fourier 高差编码
#       · 交叉注意力融合 → MLP → residual → pred_z = z_anchor + residual
#   - 损失: SmoothL1(pred_z, query_gt, beta=0.1) + terrain-complexity 加权
# ════════════════════════════════════════════════════════════════════════════

# -------------------------------------------------------
# 0. Path settings
# -------------------------------------------------------
train_data_dir = r"E:\data\云南遥感中心\第二批\disk03\tile\train"
val_data_dir   = r"E:\data\云南遥感中心\第二批\disk03\tile\val"
test_data_dir  = r"E:\data\云南遥感中心\3.13稀疏点云_道尔补全点云\模拟点样例数据\模拟点样例数据\01原始数据\las\tile"
pred_save_dir  = r"E:\data\云南遥感中心\3.13稀疏点云_道尔补全点云\模拟点样例数据\模拟点样例数据\01原始数据\las\pred_cnf"
save_path = "exp/cnf/terrain-cnf-pt-v2m4-4-single"

# -------------------------------------------------------
# 1. General settings
# -------------------------------------------------------
grid_size = 1.0
ignore_index = -1
dataset_type = "LasDataset"
ground_class = 2
feature_keys = ["coord"]
in_channels = 3

# -------------------------------------------------------
# 2. Checkpoint / run control
# -------------------------------------------------------
weight = "exp/cnf/terrain-cnf-pt-v2m4-4-single/model/model_last.pth"
resume = True
evaluate = True
test_only = False
seed = 42

# -------------------------------------------------------
# 3. Resource & batch settings
# -------------------------------------------------------
batch_size_train = 4
batch_size_val = 1
batch_size_test = 1
num_worker = 0
gradient_accumulation_steps = 4

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
# 7. Model — DefaultCNF (Backbone + SingleBranchCNFHead)
# -------------------------------------------------------
model = dict(
    type="DefaultCNF",
    reg_weight=0.0,       # no regularization for single branch
    terrain_alpha=2.0,    # terrain complexity weighting: W = 1 + alpha * |gt - z_anchor|
    ohem_ratio = 0.5,
    normal_weight=10.0,
    enable_normal_loss=True,
    filter_non_ground=False,
    ground_class=ground_class,
    backbone=dict(
        type="PT-v2m5",
        in_channels=in_channels,
        use_cls_embed=True,       # LAS semantic class embedding
        num_classes=32,           # max LAS classification code
        cls_embed_dim=16,         # embedding dimension
        patch_embed_depth=1,
        patch_embed_channels=24,
        patch_embed_groups=6,
        patch_embed_neighbours=24,
        enc_depths=(1, 1, 1),
        enc_channels=(36, 72, 144),
        enc_groups=(6, 12, 24),
        enc_neighbours=(24, 24, 24),
        dec_depths=(1, 1, 1),
        dec_channels=(24, 36, 72),
        dec_groups=(4, 6, 12),
        dec_neighbours=(24, 24, 24),
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
        type="SingleBranchCNFHead",
        backbone_out_channels=64,
        query_dim=2,
        num_targets=1,
        k_neighbors=16,
        hidden_dim=256,
        z_num_freqs=4,
        ground_class=2,
        num_classes=32,
        class_embed_dim=16,
        attn_groups=4,
    ),
    criteria=None,  # use built-in SmoothL1 loss
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
    dict(type="CacheCleaner", time_multiplier=5, step_clean_interval=250),
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
    source_dir=test_data_dir,
    classification=1,
)

# CNF test-time parameters
query_dim = 2
query_resolution = 0.5
query_batch_size = 20000
compute_derivatives = True

# -------------------------------------------------------
# 11. Dataset  (compute_gt_low=False → no query_gt_low)
# -------------------------------------------------------
data = dict(
    num_classes=1,
    ignore_index=ignore_index,
    names=["terrain_z"],
    train=dict(
        type=dataset_type,
        split="train",
        data_path=train_data_dir,
        test_mode=False,
        loop=4,
        weighted_sampler="terrain",
        transform=[
            dict(type="ZPercentileCenterShift", percentile=2.0),
            dict(type="CategoryAwareDownsample", grid_size=grid_size, ground_class=ground_class),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(
                type="TerrainImplicitSampler",
                random_ratio=0.1,
                feature_ratio=0.1,
                max_blocks=3,
                block_size_range=(5.0, 30.0),
                feature_resolution=2.0,
                max_query_ratio=0.9,          # 放开限制
                extreme_hole_prob=0.3,        # 30% 概率触发极端大空洞
                query_max=4096,
                compute_gt_low=False,
                ground_class=ground_class,
            ),
        ],
        post_transform=[
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=["coord", "segment", "query_coord", "query_gt", "query_normal_gt"],
                optional_keys=["query_normal_gt", "segment"],
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
        loop=4,
        transform=[
            dict(type="ZPercentileCenterShift", percentile=2.0),
            dict(type="CategoryAwareDownsample", grid_size=grid_size, ground_class=ground_class),
            dict(
                type="TerrainImplicitSampler",
                random_ratio=0.1,
                feature_ratio=0.1,
                max_blocks=3,
                block_size_range=(5.0, 30.0),
                feature_resolution=2.0,
                max_query_ratio=0.9,          # 放开限制
                extreme_hole_prob=0.3,        # 30% 概率触发极端大空洞
                query_max=4096,
                compute_gt_low=False,
                ground_class=ground_class,
            ),
        ],
        post_transform=[
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=["coord", "segment", "query_coord", "query_gt", "query_normal_gt"],
                optional_keys=["query_normal_gt", "segment"],
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
        transform=[
            dict(type="ZPercentileCenterShift", percentile=2.0),
            dict(type="CategoryAwareDownsample", grid_size=grid_size, ground_class=ground_class),
        ],
        aug_transform=[
            [dict(type="RandomScale", scale=[1, 1])],
        ],
        post_transform=[
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=["coord", "segment"],
                optional_keys=["segment"],
                offset_keys_dict=dict(
                    offset="coord",
                ),
                feat_keys=feature_keys,
            ),
        ],
    ),
)
