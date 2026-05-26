# -------------------------------------------------------
# 0. Path settings
# -------------------------------------------------------
train_data_dir = r"E:\data\LASDU\tile\train"
val_data_dir = r"E:\data\LASDU\tile\test"
test_data_dir = r"E:\data\LASDU\tile\test"
pred_save_dir = r"E:\data\LASDU\tile\pred"
save_path = "exp/lasdu/semseg-pt-v3m1-0-base"

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
weight = "exp/lasdu/semseg-pt-v3m1-0-base/model/model_last.pth"   # path to pretrained / fine-tune weight
# weight = None
resume = True      # resume from the latest checkpoint
evaluate = True     # run evaluation after each training epoch
test_only = False   # skip training, run test only
seed = 42           # fixed seed (None = auto-random, value is logged)

# -------------------------------------------------------
# 3. Resource & batch settings
# -------------------------------------------------------
batch_size_train = 12      # effective batch = micro_batch × gradient_accumulation_steps
                           #   micro_batch = batch_size_train // gradient_accumulation_steps
batch_size_val = 2         # None → auto 1 per GPU (no gradient → less memory than train)
batch_size_test = 2        # None → auto 1 per GPU; >1 = fragments per forward in SemSegTester
num_worker = 0            # total dataloader workers across all GPUs
gradient_accumulation_steps = 2  # effective batch = 2, micro_batch per step = 3

# -------------------------------------------------------
# 4. Training loop
# -------------------------------------------------------
epoch = 20        # total epochs; val runs after every epoch
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
    type="DefaultSegmentorV2",
    num_classes=num_classes,
    backbone_out_channels=64,
    backbone=dict(
        type="PT-v3m1",
        in_channels=in_channels,
        order=["z", "z-trans", "hilbert", "hilbert-trans"],
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(96, 96, 96, 96, 96),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(96, 96, 96, 96),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=True,
        enable_flash=False,
        upcast_attention=False,
        upcast_softmax=False,
        enc_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("Dales"),
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
        loop=20,
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
                keys=["coord", "segment", "grid_coord"],  # 只需要coord和segment
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
        loop=10,  # Validation doesn't need loop
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
                keys=["coord", "segment", "grid_coord"],  # 只需要coord和segment
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
                keys=["coord", "index", "grid_coord"],
                feat_keys=feature_keys,
            ),
        ],
    ),
)
