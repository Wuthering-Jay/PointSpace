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

partition_feature_keys = ["intensity"]
point_hf_keys = partition_feature_keys
post_cnn_point_hf_keys = [
    "intensity",
    "linearity",
    "planarity",
    "scattering",
    "verticality",
    "elevation",
]
feature_keys = partition_feature_keys
in_channels = len(partition_feature_keys)

weight = r"exp\dales\ezsp-stage1\model\model_last.pth"
resume = False
evaluate = True
test_only = False
seed = 42

batch_size_train = 8
batch_size_val = 2
batch_size_test = 3
num_worker = 0
gradient_accumulation_steps = 2

# Training settings - Following official DALES Stage2 config
epoch = 600 
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
    channels=[32, 32, 32],
    kernel_size=[7, 3, 3],
    dilation=1,
    norm="gn",
    norm_eps=1e-4,
    activation="leakyrelu",
    residual=False,
    global_residual=False,
    last_norm=True,
    last_activation=False,
    frozen=False
)

partition_config = dict(
    type="GreedyContourPriorPartition",
    reg=[0.015, 0.05, 0.15],
    min_size=[5, 15, 70],
    k_adjacency=10,
    spatial_weight=0.05,
    edge_weight_mode="affinity_latent_distance",
    d_0=None,
    w_adjacency=0.0,
    max_iterations=-1,
    edge_reduce="add",
    build_edge_features=True,
    build_vertical_features=False,
)

transformer_config = dict(
    type="EZSPTransformer",
    num_classes=num_classes,
    in_channels=32,
    nano=False,
    point_cnn_blocks=True,
    point_mlp_on_cnn_feats=True,
    point_mlp=[41, 64, 128],
    down_dim=[64, 128, 256],
    down_in_mlp=[[132, 64], [68, 128], [132, 256]],
    down_num_heads=[4, 8, 16],
    down_num_blocks=[2, 2, 2],
    down_ffn_ratio=1.0,
    up_dim=[128, 64],
    up_in_mlp=[[451, 128], [227, 64]],
    up_num_heads=[8, 4],
    up_num_blocks=[1, 1],
    up_ffn_ratio=1.0,
    use_pos=True,
    use_node_hf=False,
    use_diameter=True,
    use_diameter_parent=False,
    down_pool_dim=[64, 128],
    pool="max",
    fusion="cat",
)

graph_transform_config = dict(
    type="HierarchyGraphTransform",
    enabled=True,
    training_only=True,
    apply_levels="1+",
    max_nodes=0,
    max_edges=0,
    n_min_edges=0,
    n_max_edges=0,
    add_self_loops=True,
    pos_jitter_std=0.0,
    pos_jitter_trunc=0.0,
    edge_attr_jitter_std=0.0,
    edge_attr_jitter_trunc=0.0,
)

model = dict(
    type="EZSPPartitionSegmentor",
    training_partition_stage=False,
    num_classes=num_classes,
    sparse_cnn=sparse_cnn_config,
    partition_module=partition_config,
    transformer=transformer_config,
    freeze_cnn=True,
    backbone_out_channels=32,
    post_cnn_keys=post_cnn_point_hf_keys,
    graph_transform=graph_transform_config,
    criteria=[
        dict(type="WeightedFocalLoss", loss_weight=1.0, gamma=2.0, ignore_index=num_classes, auto_class_weight=True),
    ],
)

optimizer = dict(type="AdamW", lr=5e-3, weight_decay=1e-4)
scheduler = dict(
    type="CosineAnnealingLR",
    total_steps=epoch,
)

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
                feat_keys=partition_feature_keys),
            dict(type="PointFeatures", keys=["linearity", "planarity", "scattering", "verticality"], k=25, k_min=10),
            dict(type="GroundElevation", model="ransac", xy_grid=5.0, scale=20.0),
        ],
        post_transform=[
            dict(type="ToTensor"),
            dict(type="Collect", 
                 keys=["coord", "segment", "sub", "num_raw_points", "grid_size", "intensity", "linearity", "planarity", "scattering", "verticality", "elevation"], 
                 feat_keys=partition_feature_keys),
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
                feat_keys=partition_feature_keys),
            dict(type="PointFeatures", keys=["linearity", "planarity", "scattering", "verticality"], k=25, k_min=10),
            dict(type="GroundElevation", model="ransac", xy_grid=5.0, scale=20.0),
        ],
        post_transform=[
            dict(type="ToTensor"),
            dict(
                type="Collect", 
                keys=["coord", "segment", "segment_raw", "sub", "num_raw_points", "grid_size", "intensity", "linearity", "planarity", "scattering", "verticality", "elevation"], 
                feat_keys=partition_feature_keys),
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
                feat_keys=partition_feature_keys),
            dict(type="PointFeatures", keys=["linearity", "planarity", "scattering", "verticality"], k=25, k_min=10),
            dict(type="GroundElevation", model="ransac", xy_grid=5.0, scale=20.0),
        ],
        post_transform=[
            dict(type="ToTensor"),
            dict(
                type="Collect", 
                keys=["coord", "segment", "segment_raw", "sub", "num_raw_points", "grid_size", "intensity", "linearity", "planarity", "scattering", "verticality", "elevation"], 
                feat_keys=partition_feature_keys),
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
