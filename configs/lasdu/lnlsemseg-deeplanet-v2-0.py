# -------------------------------------------------------
# 0. Path settings
# -------------------------------------------------------
train_data_dir = r"E:\data\LASDU\tile\train_noisy"
val_data_dir = r"E:\data\LASDU\tile\test"
test_data_dir = r"E:\data\LASDU\tile\test"
pred_save_dir = r"E:\data\LASDU\tile\pred_lnl"
save_path = "exp/lasdu/semseg-deeplanet-v2-lnl"

# -------------------------------------------------------
# 1. General settings
# -------------------------------------------------------
num_classes = 5
grid_size = 0.5
ignore_index = -1
dataset_type = "LasDataset"
required_classes = [0, 1, 2, 3, 4]
class_names = [
    "ground",
    "buildings",
    "trees",
    "low vegetation",
    "artifacts",
]
feature_keys = ["coord", "echo", "intensity"]
in_channels = 6

# -------------------------------------------------------
# 2. Checkpoint / run control
# -------------------------------------------------------
# weight = "exp/lasdu/semseg-deeplanet-v2-sparse/model/model_best.pth"   # path to pretrained / fine-tune weight
weight = None
resume = True      # resume from the latest checkpoint
evaluate = True     # run evaluation after each training epoch
test_only = False   # skip training, run test only
seed = 42           # fixed seed (None = auto-random, value is logged)

# -------------------------------------------------------
# 3. Resource & batch settings
# -------------------------------------------------------
batch_size_train = 8       # effective batch = micro_batch × gradient_accumulation_steps
                           #   micro_batch = batch_size_train // gradient_accumulation_steps
batch_size_val = 2         # None → auto 1 per GPU (no gradient → less memory than train)
batch_size_test = 2        # None → auto 1 per GPU; >1 = fragments per forward in SemSegTester
num_worker = 0            # total dataloader workers across all GPUs
gradient_accumulation_steps = 2  # effective batch = 2, micro_batch per step = 3

# -------------------------------------------------------
# 4. Training loop
# -------------------------------------------------------
epoch = 100        # total epochs; val runs after every epoch
clip_grad = None    # gradient clipping (None = disabled)

# -------------------------------------------------------
# 5. Precision & performance
# -------------------------------------------------------
enable_amp = True
amp_dtype = "float16"
sync_bn = False
find_unused_parameters = False

# -------------------------------------------------------
# 6. Logging & augmentation
# -------------------------------------------------------
enable_wandb = False
wandb_project = "pointspace-lasdu"
wandb_key = None    # set or run `wandb login` beforehand
mix_prob = 0.0      # MixUp / CutMix probability

# -------------------------------------------------------
# 7. Model
# -------------------------------------------------------
model = dict(
    type="DeepLNLSegmentor",
    num_classes=num_classes,
    backbone_out_channels=64,
    backbone=dict(
        type="DeepLANet-v2",
        in_channels=in_channels,
        patch_embed_depth=1,
        patch_embed_channels=32,
        patch_embed_neighbours=16,
        enc_depths=(4, 4, 12, 4),
        enc_channels=(64, 128, 256, 512),
        enc_neighbours=(16, 16, 16, 16),
        dec_depths=(1, 1, 1, 1),
        dec_channels=(64, 128, 256, 512),
        dec_neighbours=(16, 16, 16, 16),
        grid_sizes=(
            3 * grid_size / 2,
            7.5 * grid_size / 2,
            15 * grid_size / 2,
            37.5 * grid_size / 2,
        ),  # x3, x2.5, x2.5, x2.5
        drop_path_rate=0.2,
        enable_checkpoint=False,
        unpool_backend="interp",
        # 深层网络稳定性优化
        enable_deep_supervision=True,   # 启用 HDS (混合深监督)
        enable_layer_scale=True,        # 启用 LayerScale
        layer_scale_init_value=1e-5,    # LayerScale 初始值
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1, auto_class_weight=True),
        #dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
    # 辅助损失配置 (Hybrid Deep Supervision)
    aux_criteria=[
        dict(type="CrossEntropyLoss", loss_weight=0.4, ignore_index=-1, auto_class_weight=True),
    ],
    aux_channels=(64, 128, 256, 512), # 辅助头通道数，需与 enc_channels 对应
    aux_dropout=0.1,
    aux_weights=(0.1, 0.2, 0.3, 0.4), # 各 stage 的辅助损失权重比例 (越深的层权重越大)
    shallow_stage=2,      # 物理 Stage 2 (提供纯净局部几何先验)
    bottleneck_stage=4,   # 物理 Stage 4 (Encoder最深处，提供纯净全局语义)
    max_alpha=2.0,        # [V3新增] CDCS 散度衰减的最大系数
    warmup_epochs=4,     # [V3调整] 前 N 个 epoch 为纯热身期 (alpha=0)
    rampup_epochs=15,     # [V3新增] 热身期结束后，alpha 爬升的过渡期轮数
    base_tau_pseudo=0.90, # [V3调整] 基础伪标签阈值 (结合自适应机制)
    pseudo_weight=0.1,    # PSSM 伪标签损失权重
    ignore_index=ignore_index,      # 数据集中的未分类/忽略标签 ID
)

# -------------------------------------------------------
# 8. Optimizer & scheduler
# -------------------------------------------------------
optimizer = dict(type="AdamW", lr=1e-3, weight_decay=1e-2)
scheduler = dict(
    type="CosineAnnealingLR",
    total_steps=epoch,
)
param_dicts = None # example: [dict(keyword="block", lr_scale=0.1)]

# -------------------------------------------------------
# 9. Hooks
# -------------------------------------------------------
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="RuntimeInfoHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter", interval=10),
    dict(type="CacheCleaner", time_multiplier=5, step_clean_interval=100),
    dict(type="SemSegEvaluator",log_interval=10),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
]

# -------------------------------------------------------
# 10. Train / test engine & writer
# -------------------------------------------------------
train = dict(type="DefaultTrainer")
test = dict(type="SemSegTester")
writer = dict(type="LASWriter", save_dir=pred_save_dir, source_dir=test_data_dir)

# -------------------------------------------------------
# 11. Dataset
# -------------------------------------------------------
data = dict(
    num_classes=num_classes,
    ignore_index=ignore_index,
    names=class_names,
    train=dict(
        type=dataset_type,
        split="train",
        data_path=train_data_dir,  # Use specific train path
        required_class=required_classes,  # Filter unwanted classes
        remap_class=True,              # Remap to continuous
        class_weight='sqrt',           # Recommended: sqrt method
        weight_sample=1.0,             # Use 100% of data for weight computation
        weighted_sampler=True,         # Enable WeightedRandomSampler
        test_mode=False,
        loop=2,
        # Data augmentation
        transform=[
            dict(type="ZPercentileCenterShift", percentile=2.0),
            dict(type="RobustLogIntensity", clip_min=-3.0, clip_max=3.0),
            dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.05, clip=0.1),
            dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
        ],
        post_transform=[
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=["coord", "segment"],  # 只需要coord和segment
                feat_keys=feature_keys,
            ),
        ],
    ),
    val=dict(
        type=dataset_type,
        split="test",
        data_path=val_data_dir,  # Use specific val path
        required_class=required_classes,
        remap_class=True,
        ignore_index=ignore_index,
        test_mode=False,
        loop=2,  # Validation doesn't need loop
        # Validation uses minimal transforms (no random augmentation for deterministic eval)
        transform=[
            dict(type="ZPercentileCenterShift", percentile=2.0),
            dict(type="RobustLogIntensity", clip_min=-3.0, clip_max=3.0),
            dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.05, clip=0.1),
            dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
        ],
        post_transform=[
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=["coord", "segment"],  # 只需要coord和segment
                feat_keys=feature_keys,
            ),
        ],
    ),
    test=dict(
        type=dataset_type,
        split="test",
        data_path=test_data_dir,
        required_class=required_classes,
        remap_class=True,
        ignore_index=ignore_index,
        test_mode=True,
        # Base transform
        transform=[
            dict(type="ZPercentileCenterShift", percentile=2.0),
            dict(type="RobustLogIntensity", clip_min=-3.0, clip_max=3.0),
            dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="test",
                return_grid_coord=True
            ),
        ],
        aug_transform=[
            # [dict(type="RandomScale", scale=[0.9, 0.9])],
            # [dict(type="RandomScale", scale=[0.95, 0.95])],
            [dict(type="RandomScale", scale=[1, 1])],
            # [dict(type="RandomScale", scale=[1.05, 1.05])],
            # [dict(type="RandomScale", scale=[1.1, 1.1])],
            # [dict(type="RandomScale", scale=[0.9, 0.9]), dict(type="RandomFlip", p=1)],
            # [dict(type="RandomScale", scale=[0.95, 0.95]), dict(type="RandomFlip", p=1)],
            # [dict(type="RandomScale", scale=[1, 1]), dict(type="RandomFlip", p=1)],
            # [dict(type="RandomScale", scale=[1.05, 1.05]), dict(type="RandomFlip", p=1)],
            # [dict(type="RandomScale", scale=[1.1, 1.1]), dict(type="RandomFlip", p=1)],
        ],
        post_transform=[
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=["coord", "index"],
                feat_keys=feature_keys,
            ),
        ],
    ),
)
