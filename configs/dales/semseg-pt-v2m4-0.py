# ==============================================================================
# Path Settings (集中管理所有路径)
# ==============================================================================
train_data_path = r"E:\data\DALES\dales_las\tile\train"
val_data_path = r"E:\data\DALES\dales_las\tile\test"
test_data_path = r"E:\data\DALES\dales_las\tile\test"

save_path = "exp/dales/semseg-pt-v2m4-0-base"
saver_output_path = r"E:\data\DALES\dales_las\tile\pred"

# weight = "exp/dales/semseg-pt-v2m4-0-base/model/model_best.pth"
# weight = None  # None = train from scratch

# ==============================================================================
# Runtime Settings
# ==============================================================================
resume = False  # whether to resume training process
evaluate = True  # evaluate after each epoch training process
test_only = False  # test only without training
seed = 1  # None = random seed, or set a fixed number for reproducibility

num_worker = 8  # total workers in all GPUs
batch_size = 4  # total batch size in all GPUs (adjusted for LAS data)
batch_size_val = 4  # auto adapt to bs 1 for each GPU
batch_size_test = 4  # fragments per batch during testing (within a single scene)

epoch = 10  # total training epochs (reduced for LAS dataset)

gradient_accumulation_steps = 1  # gradient accumulation steps
clip_grad = None  # gradient clipping (None = disabled, or set a float value)

mix_prob = 0.0  # probability of mixup augmentation (0.8 = 80% mixup, 20% original)

sync_bn = False  # synchronized batch normalization across GPUs
enable_amp = True  # automatic mixed precision for faster training
amp_dtype = "float16"  # AMP data type: "float16" or "bfloat16"
find_unused_parameters = False  # for distributed training

param_dicts = None  # example: [dict(keyword="block", lr_scale=0.1)]
enable_wandb = False  # enable Weights & Biases logging
wandb_project = "pointspace-dales"  # W&B project name
wandb_key = None  # W&B API key (None = use wandb login)

train = dict(type="DefaultTrainer")
test = dict(type="SemSegTester", verbose=True)

# ==============================================================================
# Settings
# ==============================================================================
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

# ==============================================================================
# Hooks
# ==============================================================================
hooks = [
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter", interval=10),  # Log every 10 iterations during training
    dict(type="CacheCleaner", ratio_threshold=10, time_threshold=3, clean_interval=None, log_clean=True)
]

# ==============================================================================
# Model Settings
# ==============================================================================
# Two modes are supported:
# Mode 1: backbone + loss (backbone contains internal head, num_classes > 0)
# Mode 2: backbone + head + loss (backbone outputs features, external head)
model = dict(
    type="DefaultSegmentor",
    backbone=dict(
        type="PT-v2m4",
        in_channels=5,
        num_classes=None,  # None / 0 = feature-only mode (no internal head)
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
    # External head for semantic segmentation
    head=dict(
        type="SemSegHead",
        in_channels=24,
        num_classes=num_classes,
        hidden_channels=24,
        num_layers=2,  # MLP with 1 hidden layer
        dropout=0.0,
        bn=True,
    ),
    criteria=[
        dict(
            type="CrossEntropyLoss",
            loss_weight=1.0,
            ignore_index=ignore_index,
            inject_class_weight=True,  # Enable class weight injection from dataset
        ),
        dict(
            type="LovaszLoss", 
            mode="multiclass", 
            loss_weight=1.0, 
            ignore_index=ignore_index,
        ),
    ],
)

# ==============================================================================
# Optimizer & Scheduler Settings
# ==============================================================================
optimizer = dict(type="AdamW", lr=1e-3, weight_decay=1e-2)
scheduler = dict(
    type="CosineAnnealingLR",
    total_steps=epoch,
)

# ==============================================================================
# Dataset Settings
# ==============================================================================
data = dict(
    num_classes=num_classes,  # Remapped classes
    ignore_index=ignore_index,
    names=class_names,
    # ==============================================================================
    # Training Dataset
    # ==============================================================================
    train=dict(
        type=dataset_type,
        split="train",
        data_path=train_data_path,  # Use specific train path
        required_class=required_classes,  # Filter unwanted classes
        remap_class=True,              # Remap to continuous [0-6]
        class_weight='sqrt',           # Recommended: sqrt method
        weight_sample=0.2,             # Use 20% of data for weight computation
        weighted_sampler=True,         # Enable WeightedRandomSampler
        test_mode=False,
        loop=5,
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
                feat_keys=["coord","echo"],
            ),
        ],
    ),
    
    # ==============================================================================
    # Validation Dataset
    # ==============================================================================
    val=dict(
        type=dataset_type,
        split="val",
        data_path=val_data_path,  # Use specific val path
        required_class=required_classes,
        remap_class=True,
        ignore_index=ignore_index,
        test_mode=False,
        loop=5,  # Validation doesn't need loop
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
                feat_keys=["coord","echo"],
            ),
        ],
    ),
    
    # ==============================================================================
    # Test Dataset (with TTA)
    # ==============================================================================
    test=dict(
        type=dataset_type,
        split="test",
        data_path=test_data_path,
        required_class=required_classes,  # Filter to classes 1-8 (same as training!)
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
                feat_keys=["coord","echo"],
            ),
        ],
    ),
)

# ==============================================================================
# Saver Settings (LasSaver for preserving LAS attributes)
# ==============================================================================
# Set save_path to None to disable saving test results
# saver_output_path = None  # Uncomment to disable saving
saver = dict(
    type="LasSaver",
    save_path=saver_output_path,        # Results save path (None = no saving)
    input_path=test_data_path,          # Original LAS files path
    # id2class will be automatically obtained from dataset during testing
    # id2class={0: 2, 1: 3, 2: 4, 3: 5, 4: 6, 5: 7, 6: 8},
    output_format='las',  # 'las' or 'laz'
    compress=False,       # Enable LAZ compression
)
