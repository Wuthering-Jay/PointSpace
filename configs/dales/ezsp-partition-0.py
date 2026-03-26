# -------------------------------------------------------
# DALES EZ-SP Partition Training Configuration
#
# This config trains the EZ-SP partition network to learn
# point features suitable for semantic superpoint clustering.
#
# Phase 1: Partition Learning (this config)
# - Lightweight TinySparseCNN feature extraction
# - Contrastive loss on graph edges
# - GPU-accelerated partition during validation
#
# Reference: EZ-SP (https://arxiv.org/abs/2402.04991)
# -------------------------------------------------------

# -------------------------------------------------------
# 0. Path settings
# -------------------------------------------------------
train_data_dir = r"E:\data\DALES\dales_las\tile\train"
val_data_dir = r"E:\data\DALES\dales_las\tile\test"
test_data_dir = r"E:\data\DALES\dales_las\tile\test"
save_path = "exp/dales/ezsp-partition"

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
# Input features: coord (xyz) + echo (1) + normalized (3) = 7
# Or just coord + echo = 5 if no normals computed
feature_keys = ["coord", "echo"]
in_channels = 5

# -------------------------------------------------------
# 2. Checkpoint / run control
# -------------------------------------------------------
weight = None
resume = True
evaluate = True
test_only = False
seed = 42

# -------------------------------------------------------
# 3. Resource & batch settings
# -------------------------------------------------------
batch_size_train = 8
batch_size_val = 4
batch_size_test = 2
num_worker = 4
gradient_accumulation_steps = 2

# -------------------------------------------------------
# 4. Training loop
# -------------------------------------------------------
epoch = 200  # Partition training typically needs more epochs
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
wandb_project = "ezsp-dales"
wandb_key = None
mix_prob = 0.0

# -------------------------------------------------------
# 7. Model - EZ-SP Partition Network
# -------------------------------------------------------
model = dict(
    type="EZSPPartitionSegmentor",
    backbone=dict(
        type="TinySparseCNN",
        in_channels=in_channels,
        # EZ-SP default: 3 conv blocks with 32 channels
        # dim_hf -> 32 -> 32 -> 32
        channels=[32, 32, 32],
        kernel_sizes=[7, 3, 3],  # From SPT DALES config
    ),
    partition_criteria=[
        dict(
            type="EZSPContrastiveLoss",
            affinity_temperature=1.0,  # Controls feature similarity sensitivity
            focal_gamma=1.0,  # Focal loss parameter
            adaptive_sampling_ratio=0.7,  # Balance inter/intra class edges
            num_classes=num_classes,
            k=8,  # KNN neighbors for graph construction
            loss_weight=1.0,
            train_only=True,
        ),
    ],
    partition_cfg=dict(
        reg=0.02,  # Regularization (coarseness control)
        min_size=30,  # Minimum superpoint size
        k=8,  # KNN neighbors
        edge_weight_mode="unit",
        verbose=False,
    ),
    compute_partition_on_val=True,
)

# -------------------------------------------------------
# 8. Optimizer & scheduler
# -------------------------------------------------------
optimizer = dict(type="AdamW", lr=5e-4, weight_decay=1e-4)
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
    dict(type="RuntimeInfoHook"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter", interval=10),
    dict(type="CacheCleaner", time_multiplier=5, step_clean_interval=100),
    # No SemSegEvaluator for partition training - we evaluate partition quality
    dict(type="CheckpointSaver", save_freq=None),
]

# -------------------------------------------------------
# 10. Train / test engine
# -------------------------------------------------------
train = dict(type="DefaultTrainer")
test = dict(type="SemSegTester")  # For validation loop

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
        data_path=train_data_dir,
        required_class=required_classes,
        remap_class=True,
        class_weight="sqrt",
        weight_sample=0.2,
        weighted_sampler=False,  # Don't need for partition training
        test_mode=False,
        loop=5,
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
                keys=["coord", "segment"],  # Need segment for contrastive loss
                feat_keys=feature_keys,
            ),
        ],
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_path=val_data_dir,
        required_class=required_classes,
        remap_class=True,
        ignore_index=ignore_index,
        test_mode=False,
        loop=1,
        transform=[
            dict(type="ZPercentileCenterShift", percentile=2.0),
            dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="train",  # Use train mode for consistent grid
                return_grid_coord=True,
            ),
        ],
        post_transform=[
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=["coord", "segment"],
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
        transform=[
            dict(type="ZPercentileCenterShift", percentile=2.0),
            dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="test",
                return_grid_coord=True,
            ),
        ],
        aug_transform=[
            [dict(type="RandomScale", scale=[1, 1])],
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
