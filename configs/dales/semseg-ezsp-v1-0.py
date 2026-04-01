# ==============================================================================
# DALES EZ-SP (End-to-End Superpoint Transformer) Configuration
# ==============================================================================
#
# Two-stage training configuration for EZ-SP on DALES dataset:
#   Stage 1 (partition learning): Train CNN to learn boundary-aligned features
#   Stage 2 (semantic segmentation): Train transformer on superpoint graphs
#
# Reference: Superpoint Transformer - Robert et al.
#   Paper: https://arxiv.org/abs/2306.08045
#   Code: https://github.com/drprojects/superpoint_transformer
#
# ==============================================================================

# -------------------------------------------------------
# 0. Path settings
# -------------------------------------------------------
train_data_dir = r"E:\data\DALES\dales_las\tile\train"
val_data_dir = r"E:\data\DALES\dales_las\tile\test"
test_data_dir = r"E:\data\DALES\dales_las\tile\test"
pred_save_dir = r"E:\data\DALES\dales_las\tile\pred"
save_path = "exp/dales/semseg-ezsp-v1-0"

# -------------------------------------------------------
# 1. General settings
# -------------------------------------------------------
num_classes = 8
grid_size = 0.1  # Smaller grid for better partition boundaries
# CRITICAL: In SPT/EZ-SP, ignore_label = num_classes (NOT -1!)
# Void/ignored annotations are placed in the (num_classes)-th column of histogram
ignore_index = num_classes  # = 8 for DALES
dataset_type = "LasDataset"
required_classes = [1, 2, 3, 4, 5, 6, 7, 8]
class_names = [
    "ground",       # 0 - stuff class
    "vegetation",   # 1 - stuff class
    "cars",         # 2
    "trucks",       # 3
    "power_lines",  # 4
    "fences",       # 5
    "poles",        # 6
    "buildings",    # 7
]
stuff_classes = [0, 1]  # Large background classes (ground, vegetation)

# Features: coord(3) + intensity(1) + echo(1) = 5
# Note: DALES has no RGB, using intensity instead
feature_keys = ["coord", "intensity"]
in_channels = 5  # coord(3) + intensity(1) + elevation(1)

# -------------------------------------------------------
# 2. EZ-SP Training Stage Control
# -------------------------------------------------------
# Set to True for Stage 1 (partition learning)
# Set to False for Stage 2 (semantic segmentation)
training_partition_stage = False  # Change to True for Stage 1

# Stage 1 checkpoint (required for Stage 2)
# Set this to your Stage 1 checkpoint path when running Stage 2
stage1_checkpoint = None  # e.g., "exp/dales/semseg-ezsp-v1-0-stage1/model/model_best.pth"

# -------------------------------------------------------
# 3. Checkpoint / run control
# -------------------------------------------------------
weight = stage1_checkpoint  # Auto-load Stage 1 weights for Stage 2
resume = True
evaluate = True
test_only = False
seed = 42

# -------------------------------------------------------
# 4. Resource & batch settings
# -------------------------------------------------------
batch_size_train = 4  # Smaller batch for EZ-SP (memory intensive)
batch_size_val = 2
batch_size_test = 1
num_worker = 4
gradient_accumulation_steps = 2

# -------------------------------------------------------
# 5. Training loop
# -------------------------------------------------------
# Stage 1: ~100 epochs for partition learning
# Stage 2: ~600 epochs for semantic segmentation
epoch = 100 if training_partition_stage else 600
clip_grad = 1.0  # Gradient clipping for stability

# -------------------------------------------------------
# 6. Precision & performance
# -------------------------------------------------------
enable_amp = True
amp_dtype = "float16"
sync_bn = False
find_unused_parameters = True  # Required for two-stage model

# -------------------------------------------------------
# 7. Logging
# -------------------------------------------------------
enable_wandb = False
wandb_project = "pointspace-dales-ezsp"
wandb_key = None
mix_prob = 0.0  # No MixUp for EZ-SP

# -------------------------------------------------------
# 8. Model - EZ-SP Configuration
# -------------------------------------------------------

# SparseCNN configuration (shared between stages)
sparse_cnn_config = dict(
    type="EZ-SparseCNN",
    in_channels=in_channels,
    channels=[32, 32, 32],  # 3-layer CNN
    kernel_size=7,
    norm="gn",  # GraphNorm for better generalization
    activation="leaky_relu",
    residual=False,
    global_residual=False,
)

# Partition module configuration
partition_config = dict(
    type="GreedyContourPriorPartition",
    reg=2e-2,  # Regularization
    min_size=[5, 15, 70],  # Min superpoint size per level (3 levels)
    k_adjacency=10,  # KNN for adjacency graph
)

# Alternative: Simpler partition module for faster experiments
partition_config_simple = dict(
    type="GreedyContourPriorPartitionSimple",
    k_adjacency=10,
    grid_size=grid_size,
    num_levels=2,
)

# Partition criterion configuration (Stage 1)
partition_criterion_config = dict(
    type="PartitionCriterion",
    gamma=1.0,  # Focal loss gamma
    alpha=0.5,  # Class balance
    temperature=1.0,  # Affinity temperature
    adaptive_sampling=True,
    adaptive_sampling_ratio=0.9,  # Minority class ratio after sampling
    num_classes=num_classes,
    loss_weight=1.0,
)

# SPT Transformer configuration (Stage 2)
# Reference: configs/model/semantic/spt-3.yaml
transformer_config = dict(
    type="EZSPTransformerSimple",
    num_classes=num_classes,
    in_channels=32,  # From SparseCNN output
    hidden_dim=64,
    num_heads=16,  # Following original SPT
    num_blocks=3,  # 3 transformer blocks
    ffn_ratio=1,  # No FFN expansion (following DALES config)
    dropout=0.0,
    use_pos=True,
    use_diameter=True,
)

# Full model configuration
if training_partition_stage:
    # Stage 1: Partition learning
    model = dict(
        type="EZSPPartitionSegmentor",
        training_partition_stage=True,
        num_classes=num_classes,
        sparse_cnn=sparse_cnn_config,
        partition_module=partition_config_simple,  # Use simple for faster training
        partition_criterion=partition_criterion_config,
        backbone_out_channels=32,
    )
else:
    # Stage 2: Semantic segmentation
    model = dict(
        type="EZSPPartitionSegmentor",
        training_partition_stage=False,
        num_classes=num_classes,
        sparse_cnn=sparse_cnn_config,
        partition_module=partition_config_simple,
        transformer=transformer_config,
        freeze_cnn=True,  # Freeze pretrained CNN
        backbone_out_channels=32,
        # Loss configuration following SPT conventions
        loss_type='ce_kl',  # Level-1: CE, Level-2+: KL divergence
        multi_stage_loss_lambdas=[1, 50],  # Level-1 weight=1, Level-2+ weight=50
        criteria=[
            dict(
                type="CrossEntropyLoss",
                loss_weight=1.0,
                ignore_index=num_classes,  # MUST be num_classes, not -1!
                auto_class_weight=True,  # Automatic class weighting
            ),
            dict(
                type="LovaszLoss",
                mode="multiclass",
                loss_weight=0.5,
                ignore_index=num_classes,  # MUST be num_classes, not -1!
            ),
        ],
    )

# -------------------------------------------------------
# 9. Optimizer & scheduler
# -------------------------------------------------------
if training_partition_stage:
    # Stage 1: Higher LR for partition learning
    optimizer = dict(type="AdamW", lr=0.01, weight_decay=1e-4)
    scheduler = dict(
        type="CosineAnnealingLR",
        total_steps=epoch,
        eta_min=1e-6,
    )
else:
    # Stage 2: Lower LR following original SPT
    optimizer = dict(type="AdamW", lr=0.005, weight_decay=1e-4)
    scheduler = dict(
        type="CosineAnnealingLRWithWarmup",
        total_steps=epoch,
        eta_min=1e-6,
        warmup_epochs=20,
        warmup_init_lr=1e-6,
    )

# Transformer LR scaling (following original SPT)
# Transformer blocks use 0.1x learning rate to prevent gradient explosion
param_dicts = [
    dict(keyword="transformer", lr_scale=0.1),
]

# -------------------------------------------------------
# 10. Hooks
# -------------------------------------------------------
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="RuntimeInfoHook"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter", interval=10),
    dict(type="CacheCleaner", time_multiplier=5, step_clean_interval=100),
    dict(type="SemSegEvaluator", log_interval=10),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
]

# -------------------------------------------------------
# 11. Train / test engine & writer
# -------------------------------------------------------
train = dict(type="DefaultTrainer")
test = dict(type="SemSegTester")
writer = dict(type="LASWriter", save_dir=pred_save_dir, source_dir=test_data_dir)

# -------------------------------------------------------
# 12. Dataset
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
        class_weight='sqrt',  # Following original SPT weighted_loss_smooth='sqrt'
        weight_sample=0.2,
        weighted_sampler=True,
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
                keys=["coord", "segment"],
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
                mode="train",
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

# ==============================================================================
# Usage Instructions
# ==============================================================================
#
# Stage 1: Partition Learning
# ---------------------------
# 1. Set training_partition_stage = True (line 63)
# 2. Set epoch = 100 (or desired)
# 3. Run: python tools/train.py --config-file configs/dales/semseg-ezsp-v1-0.py
# 4. Save the checkpoint path for Stage 2
#
# Stage 2: Semantic Segmentation
# ------------------------------
# 1. Set training_partition_stage = False (line 63)
# 2. Set stage1_checkpoint = "path/to/stage1/model_best.pth" (line 68)
# 3. Set epoch = 600 (or desired)
# 4. Run: python tools/train.py --config-file configs/dales/semseg-ezsp-v1-0.py
#
# Testing
# -------
# 1. Set test_only = True
# 2. Set weight = "path/to/final/model.pth"
# 3. Run: python tools/train.py --config-file configs/dales/semseg-ezsp-v1-0.py
#
# ==============================================================================
