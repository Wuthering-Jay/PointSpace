"""
EZ-SP Stage 2 Configuration with SPT Transformer

This configuration is for Stage 2 (semantic segmentation) of EZ-SP training.
It uses the SPT (Superpoint Transformer) network for processing superpoint graphs.

Usage:
    1. First train Stage 1 (partition learning) with ezsp_stage1_*.py config
    2. Then train Stage 2 with this config, loading Stage 1 weights

Author: PointSpace Team
"""

_base_ = ["ezsp_base.py"]

# Model configuration for Stage 2
model = dict(
    type="EZSPPartitionSegmentor",
    training_partition_stage=False,  # Stage 2: semantic segmentation
    num_classes=13,
    freeze_cnn=True,  # Freeze pretrained CNN from Stage 1
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
    
    # Partition module (same as Stage 1)
    partition_module=dict(
        type="GreedyContourPriorPartition",
        reg=2e-2,
        min_size=[5, 30, 90],
        k_adjacency=10,
        spatial_weight=None,
        edge_weight_mode="unit",
    ),
    
    # SPT Transformer configuration
    transformer=dict(
        type="EZSPTransformer",
        num_classes=13,
        in_channels=32,  # Must match sparse_cnn output
        
        # Architecture: 3 down stages + 2 up stages
        nano=False,  # Use PointStage for Level-0 processing
        point_mlp=[32, 64],  # Project CNN features to 64-dim
        point_drop=0.1,
        
        # Down stages (encoder)
        down_dim=[64, 128, 256],
        down_in_mlp=[
            [64, 64],      # Stage 1: keep 64-dim
            [128, 128],    # Stage 2: keep 128-dim
            [256, 256],    # Stage 3: keep 256-dim
        ],
        down_out_mlp=None,  # No output projection
        down_num_heads=[4, 8, 8],
        down_num_blocks=[2, 2, 2],
        down_ffn_ratio=4,
        down_residual_drop=0.1,
        down_attn_drop=0.1,
        down_drop_path=0.1,
        
        # Up stages (decoder)
        up_dim=[128, 64],  # 2 up stages
        up_in_mlp=[
            [256, 128],  # Stage 1: concat(128, 128) -> 128
            [128, 64],   # Stage 2: concat(64, 64) -> 64
        ],
        up_out_mlp=None,
        up_num_heads=[8, 4],
        up_num_blocks=[2, 2],
        up_ffn_ratio=4,
        up_residual_drop=0.1,
        up_attn_drop=0.1,
        up_drop_path=0.1,
        
        # Transformer settings
        use_pos=True,  # Use normalized positions
        pool="max",    # Max pooling for downsampling
        fusion="cat",  # Concatenation fusion for skip connections
        qk_dim=8,
        qkv_bias=True,
        k_rpe=False,   # Relative positional encoding
        q_rpe=False,
        v_rpe=False,
    ),
    
    # Semantic segmentation loss
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
)

# Training settings for Stage 2
optimizer = dict(
    type="AdamW",
    lr=0.001,
    weight_decay=0.01,
)

# Learning rate schedule
lr_scheduler = dict(
    type="OneCycleLR",
    max_lr=0.001,
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)

# Data settings
data = dict(
    num_classes=13,
    ignore_index=-1,
    train=dict(
        batch_size=4,
        num_workers=4,
    ),
    val=dict(
        batch_size=1,
        num_workers=1,
    ),
    test=dict(
        batch_size=1,
        num_workers=1,
    ),
)

# Training settings
train = dict(
    max_epochs=100,
    eval_freq=5,
    checkpoint_freq=5,
    log_freq=50,
)

# Hooks
hooks = [
    dict(type="CheckpointHook", interval=5, max_keep=10),
    dict(type="IterTimerHook"),
    dict(type="InformationHook"),
    dict(
        type="SemSegEvaluator",
        interval=5,
    ),
    dict(type="CheckInvalidLossHook", interval=50),
]

# Load Stage 1 pretrained weights
# Set this path before training
# load_from = "path/to/stage1_checkpoint.pth"
