"""
EZ-SP Stage 2 Simple Configuration

Minimal SPT transformer for quick experimentation and debugging.
Uses EZSPTransformerSimple with reduced complexity.

Author: PointSpace Team
"""

_base_ = ["ezsp_base.py"]

# Model configuration for Stage 2
model = dict(
    type="EZSPPartitionSegmentor",
    training_partition_stage=False,  # Stage 2: semantic segmentation
    num_classes=13,
    freeze_cnn=True,
    backbone_out_channels=32,
    
    # SparseCNN (pretrained from Stage 1)
    sparse_cnn=dict(
        type="EZ-SparseCNN",
        in_channels=6,
        channels=[32, 32, 32],
        kernel_size=3,
        dilation=1,
        norm="gn",
        activation="relu",
        residual=True,
        global_residual=False,
    ),
    
    # Partition module
    partition_module=dict(
        type="GreedyContourPriorPartition",
        reg=2e-2,
        min_size=[5, 30, 90],
        k_adjacency=10,
    ),
    
    # Simple SPT Transformer (1 down + 1 up stage)
    transformer=dict(
        type="EZSPTransformerSimple",
        num_classes=13,
        in_channels=32,
        hidden_dim=64,
        num_heads=4,
        num_blocks=2,
        use_pos=True,
    ),
    
    # Loss
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
    ],
)

# Training settings
optimizer = dict(
    type="AdamW",
    lr=0.001,
    weight_decay=0.01,
)

lr_scheduler = dict(
    type="OneCycleLR",
    max_lr=0.001,
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)

data = dict(
    num_classes=13,
    ignore_index=-1,
    train=dict(batch_size=4, num_workers=4),
    val=dict(batch_size=1, num_workers=1),
    test=dict(batch_size=1, num_workers=1),
)

train = dict(
    max_epochs=50,
    eval_freq=5,
    checkpoint_freq=5,
    log_freq=50,
)

hooks = [
    dict(type="CheckpointHook", interval=5, max_keep=10),
    dict(type="IterTimerHook"),
    dict(type="InformationHook"),
    dict(type="SemSegEvaluator", interval=5),
    dict(type="CheckInvalidLossHook", interval=50),
]

# Load Stage 1 pretrained weights
# load_from = "path/to/stage1_checkpoint.pth"
