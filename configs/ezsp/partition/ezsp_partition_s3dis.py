"""
EZ-SP Stage 1: Partition Learning on S3DIS

This configuration trains the SparseCNN to learn point features that
produce good superpoint partitions aligned with semantic boundaries.

Usage:
    python tools/train.py --config-file configs/ezsp/partition/ezsp_partition_s3dis.py
"""

_base_ = [
    "../../pointcept/_base_/default_runtime.py",
    "../_base_/ezsp_base.py",
]

# Dataset settings
data_root = "data/s3dis"
num_classes = 13
names = [
    "ceiling", "floor", "wall", "beam", "column",
    "window", "door", "table", "chair", "sofa",
    "bookcase", "board", "clutter",
]

# Model configuration
model = dict(
    type="EZSPPartitionSegmentor",
    training_partition_stage=True,
    num_classes=num_classes,
    backbone_out_channels=32,
    sparse_cnn=dict(
        type="EZ-SparseCNN",
        in_channels=6,
        channels=[32, 32, 32],
        kernel_size=3,
        norm="gn",
        activation="relu",
        residual=True,
    ),
    partition_module=dict(
        type="GreedyContourPriorPartition",
        reg=2e-2,
        min_size=[5, 30, 90],
        k_adjacency=10,
        spatial_weight=None,
        edge_weight_mode="unit",
    ),
    partition_criterion=dict(
        type="PartitionCriterion",
        num_classes=num_classes,
        loss_function=dict(type="BinaryFocalLoss", gamma=1.0, alpha=0.5),
        temperature=1.0,
        adaptive_sampling=True,
        adaptive_sampling_ratio=0.9,
    ),
)

# Data configuration
data = dict(
    num_classes=num_classes,
    ignore_index=-1,
    names=names,
    train=dict(
        type="S3DISDataset",
        split=["Area_1", "Area_2", "Area_3", "Area_4", "Area_6"],
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
            dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            dict(type="ChromaticJitter", p=0.95, std=0.05),
            dict(
                type="GridSample",
                grid_size=0.04,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="SphereCrop", sample_rate=0.8, mode="random"),
            dict(type="SphereCrop", point_max=80000, mode="random"),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment"),
                feat_keys=("color", "normal"),
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        type="S3DISDataset",
        split="Area_5",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="GridSample",
                grid_size=0.04,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment"),
                feat_keys=("color", "normal"),
            ),
        ],
        test_mode=False,
    ),
)

# Training settings
batch_size = 12
num_worker = 4
epoch = 100
eval_epoch = 10

# Optimizer
optimizer = dict(type="AdamW", lr=0.002, weight_decay=0.005)
scheduler = dict(
    type="OneCycleLR",
    max_lr=0.002,
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)

# Hooks
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="CheckpointSaver", save_freq=None),
]
