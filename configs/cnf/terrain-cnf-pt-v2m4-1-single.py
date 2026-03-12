# ════════════════════════════════════════════════════════════════════════════
# Terrain CNF — PT-v2m4 Backbone — Single-Branch Head
#
# 单分支 CNF 隐式曲面重建：
#   - ClassFilter 过滤地面点
#   - TerrainImplicitSampler  (compute_gt_low=False，无低频真值)
#   - Backbone (PT-v2m4) → per-point features
#   - SingleBranchCNFHead:
#       · KNN + IDW anchor
#       · Linear PE + Fourier PE + relative_z 全部融合在一个分支
#       · pred_z = z_anchor + MLP(fused_features)
#   - 损失: SmoothL1(pred_z, query_gt, beta=0.1)
# ════════════════════════════════════════════════════════════════════════════

# -------------------------------------------------------
# 0. Path settings
# -------------------------------------------------------
train_data_dir = r"E:\data\云南遥感中心\第二批\ground-only\disk03\tile\train"
val_data_dir   = r"E:\data\云南遥感中心\第二批\ground-only\disk03\tile\val"
test_data_dir  = r"E:\data\云南遥感中心\第二批\ground-only\disk03\tile\val"
pred_save_dir  = r"E:\data\云南遥感中心\第二批\ground-only\disk03\tile\pred_cnf_single"
save_path = "exp/cnf/terrain-cnf-pt-v2m4-3-single"

# -------------------------------------------------------
# 1. General settings
# -------------------------------------------------------
grid_size = 0.5
ignore_index = -1
dataset_type = "LasDataset"
ground_class = [2]
feature_keys = ["coord"]
in_channels = 3

# -------------------------------------------------------
# 2. Checkpoint / run control
# -------------------------------------------------------
weight = "exp/cnf/terrain-cnf-pt-v2m4-3-single/model/model_last.pth"
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
    normal_weight=0.5,
    backbone=dict(
        type="PT-v2m4",
        in_channels=in_channels,
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
        backbone_out_channels=24,   # must match backbone dec_channels[0]
        query_dim=2,                # predict z from (x, y)
        num_targets=1,              # scalar output (terrain height)
        k_neighbors=24,             # KNN for IDW anchor + grouped features
        hidden_dim=256,             # fusion hidden dimension
        num_freqs=6,                # Fourier PE octaves
        mlp_hidden_dims=[128, 64],
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
query_resolution = 0.25
query_batch_size = 10000
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
            dict(type="ClassFilter", keep_classes=ground_class, class_key="segment"),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(
                type="TerrainImplicitSampler",
                random_ratio=0.1,
                feature_ratio=0.1,
                max_blocks=5,
                block_size_range=(5.0, 50.0),
                feature_resolution=2.0,
                max_query_ratio=0.5,
                compute_gt_low=False,
            ),
        ],
        post_transform=[
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=["coord", "query_coord", "query_gt", "query_normal_gt"],
                optional_keys=["query_normal_gt"],
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
            dict(type="ClassFilter", keep_classes=ground_class, class_key="segment"),
            dict(
                type="TerrainImplicitSampler",
                random_ratio=0.1,
                feature_ratio=0.1,
                max_blocks=5,
                block_size_range=(5.0, 50.0),
                feature_resolution=2.0,
                max_query_ratio=0.5,
                compute_gt_low=False,
            ),
        ],
        post_transform=[
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=["coord", "query_coord", "query_gt", "query_normal_gt"],
                optional_keys=["query_normal_gt"],
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
            dict(type="ClassFilter", keep_classes=ground_class, class_key="segment"),
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
