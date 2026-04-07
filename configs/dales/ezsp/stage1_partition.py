"""
EZ-SP DALES Stage 1 Configuration - Partition Training

完全基于 semseg-ezsp-v1-0.py 的 Stage 1 配置。

运行: python tools/train.py --config-file configs/dales/ezsp/stage1_partition.py --num-gpus 1
"""

train_data_dir = r"E:\data\DALES\dales_las\tile\train"
val_data_dir = r"E:\data\DALES\dales_las\tile\test"
test_data_dir = r"E:\data\DALES\dales_las\tile\test"
pred_save_dir = r"E:\data\DALES\dales_las\tile\pred"
save_path = "exp/dales/ezsp-stage1"

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

weight = None
resume = True
evaluate = True
test_only = False
seed = 42

batch_size_train = 16
batch_size_val = 2
batch_size_test = 2
num_worker = 4
gradient_accumulation_steps = 2

epoch = 1
clip_grad = 5.0

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

partition_criterion_config = dict(
    type="PartitionCriterion",
    gamma=1.0,
    alpha=0.5,
    temperature=0.1,
    adaptive_sampling=True,
    adaptive_sampling_ratio=0.75,
    num_classes=num_classes,
    loss_weight=1.0,
)

model = dict(
    type="EZSPPartitionSegmentor",
    training_partition_stage=True,
    num_classes=num_classes,
    sparse_cnn=sparse_cnn_config,
    partition_module=partition_config,
    partition_criterion=partition_criterion_config,
    backbone_out_channels=64,
    use_voxel_to_point=False,
    voxel_to_point_decoder=None,
)

optimizer = dict(type="AdamW", lr=0.01, weight_decay=1e-4)
scheduler = dict(type="CosineAnnealingLR", total_steps=epoch, eta_min=1e-6)
param_dicts = None

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
            # 精简的数据增强（移除dropout和jitter，加快加载）
            dict(type="ZPercentileCenterShift", percentile=2.0),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            # 保存原始索引（必须在voxelization之前）
            dict(type="SaveNodeIndex", key="sub"),
            # Voxelization（EZ-SP核心，无法优化）
            dict(
                type="GridSampling3D",
                size=grid_size,
                mode="mean",
                hist_key="segment",
                hist_size=num_classes + 1,
                quantize_coords=False,  # 不量化坐标，节省计算
                feat_keys=feature_keys,
            ),
        ],
        post_transform=[
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=["coord", "segment", "sub", "num_raw_points", "grid_size"],  # 保留grid_size供sparsify使用
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
        loop=5,
        transform=[
            dict(type="ZPercentileCenterShift", percentile=2.0),
            dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(type="SaveNodeIndex", key="sub"),
            dict(type="Copy", keys_dict={"segment": "segment_raw"}),
            dict(
                type="GridSampling3D",
                size=grid_size,
                mode="mean",
                hist_key="segment",
                hist_size=num_classes + 1,
                quantize_coords=True,
                feat_keys=feature_keys,
            ),
        ],
        post_transform=[
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=["coord", "grid_coord", "segment", "sub", "num_raw_points", "segment_raw", "grid_size"],
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
            dict(type="SaveNodeIndex", key="sub"),
            dict(type="Copy", keys_dict={"segment": "segment_raw"}),
            dict(
                type="GridSampling3D",
                size=grid_size,
                mode="mean",
                hist_key="segment",
                hist_size=num_classes + 1,
                quantize_coords=True, 
                feat_keys=feature_keys,
            ),
        ],
        aug_transform=[
            [dict(type="RandomScale", scale=[1, 1])],
        ],
        post_transform=[
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=["coord", "grid_coord", "segment", "sub", "num_raw_points", "grid_size"],
                optional_keys=["segment_raw"],
                feat_keys=feature_keys,
            ),
        ],
    ),
)

hooks = [
    dict(type="CheckpointLoader"),
    dict(type="RuntimeInfoHook"),
    dict(type="ModelHook"),
    # dict(type="NaNInfDetectorTrainerHook", raise_on_nan=False, raise_on_inf=False, check_input=True, verbose=True, check_interval=1),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter", interval=10),
    dict(type="CacheCleaner", time_multiplier=5, step_clean_interval=100),
    dict(type="EZSPPartitionEvaluator", log_interval=10),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
]
train = dict(type="DefaultTrainer")
test = dict(type="EZSPPartitionTester")
writer = dict(type="LASWriter", save_dir=pred_save_dir, source_dir=test_data_dir)

