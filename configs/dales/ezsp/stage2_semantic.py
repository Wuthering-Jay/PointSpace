"""
EZ-SP DALES Stage 2 - Semantic Segmentation

前提: 完成Stage 1训练
运行: python tools/train.py --config-file configs/dales/ezsp/stage2_semantic.py --num-gpus 1
"""

train_data_dir = r"E:\data\DALES\dales_las\tile\train"
val_data_dir = r"E:\data\DALES\dales_las\tile\test"
test_data_dir = r"E:\data\DALES\dales_las\tile\test"
pred_save_dir = r"E:\data\DALES\dales_las\tile\pred"
save_path = "exp/dales/ezsp-stage2"

num_classes = 8
ignore_index = num_classes
grid_size = 0.25
dataset_type = "LasDataset"
required_classes = [1, 2, 3, 4, 5, 6, 7, 8]
class_names = [
    "ground", 
    "vegetation", 
    "cars", 
    "trucks", 
    "power_lines", 
    "fences", 
    "poles", 
    "buildings"
    ]

feature_keys = ["coord", "echo"]
in_channels = 5

weight = r"exp\dales\ezsp-stage2\model\model_last.pth"
resume = True
evaluate = True
test_only = False
seed = 42

batch_size_train = 8
batch_size_val = 2
batch_size_test = 3
num_worker = 0
gradient_accumulation_steps = 2

# Training settings - Following official DALES Stage2 config
epoch = 10  # 官方用600，这里先用100测试
clip_grad = 5.0

# Mixed precision training - Now enabled with FP16-stable custom LayerNorm
# Our LayerNorm uses eps=1e-3 and FP32 accumulation for stability
enable_amp = True
amp_dtype = "float16"
sync_bn = False
find_unused_parameters = True

enable_wandb = False
wandb_project = "pointspace-dales-ezsp"
wandb_key = None
mix_prob = 0.0

sparse_cnn_config = dict(
    type="EZ-SparseCNN",
    in_channels=in_channels,
    channels=[32, 64, 64],
    kernel_size=3,
    dilation=1,
    norm="gn",
    norm_eps=1e-4,
    activation="relu",
    residual=True,
    global_residual=False,
    last_norm=True,
    last_activation=False,
    frozen=False
)

partition_config = dict(
    type="GreedyContourPriorPartition",
    reg=[0.015, 0.05, 0.15],
    min_size=[3, 15, 50],
    k_adjacency=10,
    spatial_weight=0.05,
    edge_weight_mode="affinity_latent_distance",
    d_0=None,
    w_adjacency=0.0,
    max_iterations=-1,
    edge_reduce="add",
    build_edge_features=False,
    build_vertical_features=False,
)

transformer_config = dict(
    type="EZSPTransformer",
    num_classes=num_classes,
    in_channels=67,
    nano=True,
    down_dim=[64, 128, 256],
    down_in_mlp=[[67, 64], [134, 128], [201, 256]],
    down_num_heads=[4, 8, 16],
    down_num_blocks=[2, 2, 2],
    down_ffn_ratio=1.0,
    up_dim=[128, 64],
    up_in_mlp=[[454, 128], [259, 64]],
    up_num_heads=[8, 4],
    up_num_blocks=[1, 1],
    up_ffn_ratio=1.0,
    use_pos=True,
    pool="max",
    fusion="cat",
)

model = dict(
    type="EZSPPartitionSegmentor",
    training_partition_stage=False,
    num_classes=num_classes,
    sparse_cnn=sparse_cnn_config,
    partition_module=partition_config,
    transformer=transformer_config,
    freeze_cnn=True,
    backbone_out_channels=64,
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=num_classes, auto_class_weight=True),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=num_classes),
    ],
)

optimizer = dict(type="AdamW", lr=1e-3, weight_decay=1e-2)
scheduler = dict(
    type="CosineAnnealingLR",
    total_steps=epoch,
)

# Differential learning rate for transformer (official: 0.1x base lr)
param_dicts = [dict(keyword="transformer", lr_scale=0.1)]

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
        ignore_index=ignore_index,
        class_weight='sqrt',
        weight_sample=0.2,
        weighted_sampler=True,
        test_mode=False,
        loop=5,
        transform=[
            dict(type="ZPercentileCenterShift", percentile=2.0),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="SaveNodeIndex", key="sub"),
            dict(
                type="GridSampling3D", 
                size=grid_size, mode="mean", 
                hist_key="segment", 
                hist_size=num_classes + 1,
                quantize_coords=False,
                feat_keys=feature_keys),
        ],
        post_transform=[
            dict(type="ToTensor"),
            dict(type="Collect", 
                 keys=["coord", "segment", "sub", "num_raw_points", "grid_size"], 
                 feat_keys=feature_keys),
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
            dict(type="Copy", keys_dict=dict(segment="segment_raw")),
            dict(type="SaveNodeIndex", key="sub"),
            dict(
                type="GridSampling3D", 
                size=grid_size, 
                mode="mean", 
                hist_key="segment", 
                hist_size=num_classes + 1, 
                quantize_coords=False, 
                feat_keys=feature_keys),
        ],
        post_transform=[
            dict(type="ToTensor"),
            dict(
                type="Collect", 
                keys=["coord", "segment", "segment_raw", "sub", "num_raw_points", "grid_size"], 
                feat_keys=feature_keys),
        ],
    ),
    test=dict(
        type=dataset_type,
        split="test",
        data_path=test_data_dir,
        required_class=required_classes,
        remap_class=True,
        ignore_index=ignore_index,
        test_mode=False,  # Use val mode for simple single-pass inference
        transform=[
            dict(type="ZPercentileCenterShift", percentile=2.0),
            dict(type="Copy", keys_dict=dict(segment="segment_raw")),
            dict(type="SaveNodeIndex", key="sub"),
            dict(
                type="GridSampling3D", 
                size=grid_size, 
                mode="mean", 
                hist_key="segment", 
                hist_size=num_classes + 1, 
                quantize_coords=False, 
                feat_keys=feature_keys),
        ],
        post_transform=[
            dict(type="ToTensor"),
            dict(
                type="Collect", 
                keys=["coord", "segment", "segment_raw", "sub", "num_raw_points", "grid_size"], 
                feat_keys=feature_keys),
        ],
    ),
)

hooks = [
    dict(type="CheckpointLoader", strict=False),
    dict(type="RuntimeInfoHook"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter", interval=10),
    dict(type="CacheCleaner", time_multiplier=5, step_clean_interval=100),
    dict(type="SemSegEvaluator", log_interval=10),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
    # dict(type="NaNInfDetectorTrainerHook", raise_on_nan=True, raise_on_inf=False, check_input=True, verbose=True, check_interval=1),
]


train = dict(type="DefaultTrainer")
test = dict(type="SuperpointSemSegTester")  # Use simple tester for superpoint models
writer = dict(type="LASWriter", save_dir=pred_save_dir, source_dir=test_data_dir)