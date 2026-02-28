# -------------------------------------------------------
# 0. Path settings
# -------------------------------------------------------
train_data_dir = r"E:\data\DALES\dales_las\tile\train"
val_data_dir = r"E:\data\DALES\dales_las\tile\test"
test_data_dir = r"E:\data\DALES\dales_las\tile\test"
pred_save_dir = r"E:\data\DALES\dales_las\tile\pred"
save_path = "exp/dales/semseg-pt-v2m4-0-base"

# -------------------------------------------------------
# 1. General settings
# -------------------------------------------------------
num_classes = 8
grid_size = 0.5
ignore_index = -1
dataset_type = "LasDataset"
required_classes = [1, 2, 3, 4, 5, 6, 7, 8]
class_names = [
    "ground",
    "vegetation",
    "cars",
    "trucks",
    "power lines",
    "fences",
    "poles",
    "buildings",
]
feature_keys = ["coord", "echo"]
in_channels = 5

# -------------------------------------------------------
# 2. Checkpoint / run control
# -------------------------------------------------------
weight = None       # path to pretrained / fine-tune weight
resume = False      # resume from the latest checkpoint
evaluate = True     # run evaluation after each training epoch
test_only = False   # skip training, run test only
seed = 42           # fixed seed (None = auto-random, value is logged)

# -------------------------------------------------------
# 3. Resource & batch settings
# -------------------------------------------------------
batch_size_train = 2       # total across all GPUs
batch_size_val = 2         # None → auto 1 per GPU
batch_size_test = 2        # None → auto 1 per GPU; >1 = fragments per forward in SemSegTester
num_worker = 8             # total dataloader workers across all GPUs
gradient_accumulation_steps = 2

# -------------------------------------------------------
# 4. Training loop
# -------------------------------------------------------
epoch = 2        # total epochs
eval_epoch = epoch    # evaluate & save checkpoint every N epochs
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
wandb_project = "pointspace-dales"
wandb_key = None    # set or run `wandb login` beforehand
mix_prob = 0.0      # MixUp / CutMix probability

# -------------------------------------------------------
# 7. Model
# -------------------------------------------------------
model = dict(
    type="DefaultSegmentorV2",
    num_classes=num_classes,
    backbone_out_channels=24,
    backbone=dict(
        type="PT-v2m4",
        in_channels=in_channels,
        patch_embed_depth=1,
        patch_embed_channels=24,
        patch_embed_groups=6,
        patch_embed_neighbours=24,
        enc_depths=(2, 2, 2, 2),
        enc_channels=(48, 96, 192, 256),
        enc_groups=(6, 12, 24, 32),
        enc_neighbours=(32, 32, 32, 32),
        dec_depths=(1, 1, 1, 1),
        dec_channels=(24, 48, 96, 192),
        dec_groups=(4, 6, 12, 24),
        dec_neighbours=(32, 32, 32, 32),
        grid_sizes=(
            3 * grid_size,
            7.5 * grid_size,
            18.75 * grid_size,
            45.875 * grid_size,
        ),  # x3, x2.5, x2.5, x2.5
        attn_qkv_bias=True,
        pe_multiplier=False,
        pe_bias=True,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        enable_checkpoint=False,
        unpool_backend="interp",  # map / interp
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1, auto_class_weight=True),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
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
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter", interval=10),
    dict(type="CacheCleaner", time_multiplier=5),
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
        weight_sample=0.2,             # Use 20% of data for weight computation
        weighted_sampler=True,         # Enable WeightedRandomSampler
        test_mode=False,
        loop=1,
        # Data augmentation
        transform=[
            dict(type="ZPercentileCenterShift", percentile=2.0),
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
        split="val",
        data_path=val_data_dir,  # Use specific val path
        required_class=required_classes,
        remap_class=True,
        ignore_index=ignore_index,
        test_mode=False,
        loop=1,  # Validation doesn't need loop
        # Validation uses minimal transforms (no random augmentation for deterministic eval)
        transform=[
            dict(type="ZPercentileCenterShift", percentile=2.0),
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
