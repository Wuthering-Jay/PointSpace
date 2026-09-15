# RRCL-v1 on DALES LAS tiles.
# Backbone input is XYZ only. LAS return attributes are supervision targets.

# -------------------------------------------------------
# 0. Paths and task definition
# -------------------------------------------------------
train_data_dir = r"E:\data\DALES\dales_las\tile\train"
val_data_dir = r"E:\data\DALES\dales_las\tile\test"
save_path = "exp/dales/rrcl-litept-v1m1"
feature_output_dir = r"E:\data\DALES\dales_las\tile\rrcl_feature\litept_v1m1-2"

dataset_type = "RRCLLasDataset"
num_classes = 4
class_names = ["S", "F", "M", "L"]
ignore_index = -1
grid_size = 0.25
knn_k = 32
feature_keys = ["coord"]
in_channels = 3

# -------------------------------------------------------
# 1. Run control and resources
# -------------------------------------------------------
weight = f"{save_path}/model/model_last.pth"
resume = True
evaluate = True
test_only = False
seed = 42

batch_size_train = 20
batch_size_val = 8
batch_size_test = 8
num_worker = 8
gradient_accumulation_steps = 2
epoch = 2
clip_grad = None

enable_amp = True
amp_dtype = "float16"
sync_bn = False
find_unused_parameters = False
enable_wandb = False
wandb_project = "pointspace-dales-rrcl"
wandb_key = None
mix_prob = 0.0

# -------------------------------------------------------
# 2. RRCL model
# -------------------------------------------------------
model = dict(
    type="RRCL",
    backbone_out_channels=72,
    knn_k=knn_k,
    lambda_ctx=1.0,
    head_dropout=0.1,
    ignore_index=ignore_index,
    backbone=dict(
        type="LitePT-v1m1",
        in_channels=in_channels,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(36, 72, 144, 252, 504),
        enc_num_head=(2, 4, 8, 14, 28),
        enc_patch_size=(192, 192, 192, 192, 192),
        enc_conv=(True, True, True, True, True),
        enc_attn=(False, False, False, True, True),
        enable_rope=False,
        enc_rope_freq=(100.0, 100.0, 100.0, 100.0, 100.0),
        dec_depths=(1, 1, 1, 1),
        dec_channels=(72, 72, 144, 252),
        dec_num_head=(4, 4, 8, 14),
        dec_patch_size=(192, 192, 192, 192),
        dec_conv=(True, True, True, True),
        dec_attn=(False, False, False, False),
        dec_rope_freq=(100.0, 100.0, 100.0, 100.0),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        shuffle_orders=True,
        enc_mode=False,
    ),
    criteria=[
        dict(
            type="CrossEntropyLoss",
            loss_weight=1.0,
            ignore_index=ignore_index,
            auto_class_weight=True,
        ),
    ],
)

# -------------------------------------------------------
# 3. Optimization and hooks
# -------------------------------------------------------
optimizer = dict(type="AdamW", lr=1e-3, weight_decay=2e-3)
scheduler = dict(type="CosineAnnealingLR", total_steps=epoch)
param_dicts = [dict(keyword="block", lr=1e-4)]

hooks = [
    dict(type="CheckpointLoader"),
    dict(type="RuntimeInfoHook"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter", interval=10),
    dict(type="CacheCleaner", time_multiplier=5, step_clean_interval=200),
    dict(type="RRCLEvaluator", log_interval=10),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="BackboneCheckpointExporter", filename="backbone_last.pth"),
    dict(type="PreciseEvaluator", test_last=True),
]

train = dict(type="DefaultTrainer")
test = dict(type="RRCLFeatureTester", verbose=True)

# Feature export keeps the raw 72-D decoder feature by default. Set True when
# cosine-distance analysis requires unit-normalized rows.
feature_source = "backbone"
feature_dtype = "float16"
normalize_feature = False
feature_aggregate_on_gpu = False
feature_overwrite = False

# -------------------------------------------------------
# 4. Data
# -------------------------------------------------------
train_transform = [
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
]

val_transform = [
    dict(type="ZPercentileCenterShift", percentile=2.0),
    dict(
        type="GridSample",
        grid_size=grid_size,
        hash_type="fnv",
        mode="train",
        return_grid_coord=True,
    ),
]

post_transform = [
    dict(type="ToTensor"),
    dict(
        type="Collect",
        keys=["coord", "return_role", "grid_coord"],
        feat_keys=feature_keys,
    ),
]

test_post_transform = [
    dict(type="ToTensor"),
    dict(
        type="Collect",
        keys=["coord", "grid_coord", "index"],
        feat_keys=feature_keys,
    ),
]

data = dict(
    num_classes=num_classes,
    ignore_index=ignore_index,
    names=class_names,
    train=dict(
        type=dataset_type,
        split="train",
        data_path=train_data_dir,
        required_class=None,
        remap_class=False,
        class_weight="sqrt",
        weight_sample=0.2,
        weighted_sampler=False,
        test_mode=False,
        loop=5,
        transform=train_transform,
        post_transform=post_transform,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_path=val_data_dir,
        required_class=None,
        remap_class=False,
        class_weight=None,
        weighted_sampler=False,
        test_mode=False,
        loop=1,
        transform=val_transform,
        post_transform=post_transform,
    ),
    test=dict(
        type=dataset_type,
        split="test",
        data_path=val_data_dir,
        required_class=None,
        remap_class=False,
        class_weight=None,
        weighted_sampler=False,
        test_mode=True,
        transform=[
            dict(type="ZPercentileCenterShift", percentile=2.0),
            # dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="test",
                return_grid_coord=True,
                return_inverse=True,
            ),
        ],
        aug_transform=[[dict(type="RandomScale", scale=[1.0, 1.0])]],
        post_transform=test_post_transform,
    ),
)
