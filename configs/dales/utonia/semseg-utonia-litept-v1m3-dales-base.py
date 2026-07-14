# -------------------------------------------------------
# Common DALES semantic segmentation config for Utonia LitePT-v1m3 downstream.
# This local base is self-contained and does not depend on Pointcept runtime base.
# -------------------------------------------------------

train_data_dir = r"E:\data\DALES\dales_las\tile\train"
val_data_dir = r"E:\data\DALES\dales_las\tile\test"
test_data_dir = r"E:\data\DALES\dales_las\tile\test"
pred_save_dir = r"E:\data\DALES\dales_las\tile\pred"
save_path = "exp/dales/utonia/semseg-litept-v1m3-base"

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
feature_keys = ["coord", "echo"]
in_channels = 5

weight = "exp/dales/utonia/pretrain-litept-v1m3-dales-xyz-echo/model/model_last.pth"
resume = False
evaluate = True
test_only = False
seed = 42

batch_size = 16
batch_size_train = 16
batch_size_val = 8
batch_size_test = 8
num_worker = 8
gradient_accumulation_steps = 2

epoch = 10
eval_epoch = 10
clip_grad = None

empty_cache = False
empty_cache_per_epoch = False
enable_amp = True
amp_dtype = "bfloat16"
sync_bn = False
find_unused_parameters = False

enable_wandb = False
wandb_project = "pointspace-dales"
wandb_key = None
mix_prob = 0.0

model = dict(
    type="DefaultSegmentorV2",
    num_classes=num_classes,
    backbone_out_channels=72,
    backbone=dict(
        type="LitePT-v1m3",
        in_channels=in_channels,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(36, 72, 144, 252, 504),
        enc_num_head=(2, 4, 8, 14, 28),
        enc_patch_size=(192, 192, 192, 192, 192),
        enc_conv=(True, True, True, False, False),
        enc_attn=(False, False, False, True, True),
        enc_rope_freq=(100.0, 100.0, 100.0, 100.0, 100.0),
        dec_depths=(0, 0, 0, 0),
        dec_channels=(72, 72, 144, 252),
        dec_num_head=(4, 4, 8, 14),
        dec_patch_size=(192, 192, 192, 192),
        dec_conv=(False, False, False, False),
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
        enable_flash=True,
        traceable=True,
        mask_token=False,
        enc_mode=False,
        freeze_encoder=False,
        shift_coords=None,
        jitter_coords=1.1,
        rescale_coords=1.2,
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1, auto_class_weight=True),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
    freeze_backbone=False,
)

optimizer = dict(type="AdamW", lr=1e-3, weight_decay=2e-3)
scheduler = dict(type="CosineAnnealingLR", total_steps=epoch)
param_dicts = [dict(keyword="block", lr=1e-4)]

hooks = [
    dict(
        type="CheckpointLoader",
        keywords="module.student.backbone",
        replacement="module.backbone",
    ),
    dict(type="RuntimeInfoHook"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter", interval=10),
    dict(type="CacheCleaner", time_multiplier=5, step_clean_interval=200),
    dict(type="SemSegEvaluator", log_interval=10),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
]

train = dict(type="DefaultTrainer")
test = dict(type="SemSegTester")
writer = dict(type="LASWriter", save_dir=pred_save_dir, source_dir=test_data_dir)

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
        weighted_sampler=True,
        test_mode=False,
        loop=5,
        transform=[
            dict(type="ZPercentileCenterShift", percentile=2.0),
            dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2),
            dict(type="RandomDropEcho", drop_ratio=0.1, drop_application_ratio=0.3),
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
                keys=["coord", "segment", "grid_coord"],
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
            dict(type="Copy", keys_dict={"segment": "origin_segment"}),
            dict(type="ZPercentileCenterShift", percentile=2.0),
            dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_inverse=True,
            ),
        ],
        post_transform=[
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=["coord", "grid_coord", "segment", "origin_segment", "inverse"],
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
                return_inverse=True,
            ),
        ],
        aug_transform=[
            [dict(type="RandomScale", scale=[1, 1])],
        ],
        post_transform=[
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=["coord", "grid_coord", "index"],
                feat_keys=feature_keys,
            ),
        ],
    ),
)
