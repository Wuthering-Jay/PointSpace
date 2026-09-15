# Official CVRL-v1m1 with the CVRL-specific PT-v3m4 backbone. This standalone file
# restores native 1024D VPSD supervision and exposes a fixed
# PCA-compressed 128D visual representation. Return remains 96D.
data_root = r"E:\data\湖北\joint_tiles"
train_pointcloud_path = rf"{data_root}\pointcloud"
test_pointcloud_path = rf"{data_root}\pointcloud"
dino_feature_path = rf"{data_root}\dino_feature"
pca_model_path = rf"{data_root}\dino_feature_pca128\dino_pca_model.safetensors"
grid_size = 0.5
feature_keys = ("coord", "intensity")
in_channels = 4
num_classes = 4
class_names = ("S", "F", "M", "L")
ignore_index = -1
knn_k = 32

mode = "full"  # visual: 128D; return: 96D; full: 224D
visual_channels = 128
return_channels = 96
backbone_out_channels = {
    "visual": visual_channels,
    "return": return_channels,
    "full": visual_channels + return_channels,
}[mode]

save_path = f"exp/hubei/cvrl/pretrain-v1m1-ptv3-v3m4-{mode}"
weight = f"{save_path}/model/model_last.pth"
resume = False
evaluate = True
test_only = False
seed = 42

num_worker = 4
batch_size_train = 12
batch_size_val = 4
batch_size_test = 4
gradient_accumulation_steps = 4
epoch = 10
clip_grad = 3.0
sync_bn = False
enable_amp = True
amp_dtype = "bfloat16"
find_unused_parameters = False
enable_wandb = False
wandb_project = "pointspace-cvrl"
wandb_key = None
mix_prob = 0.0
param_dicts = None

base_lr = 1e-3
optimizer = dict(type="AdamW", lr=base_lr, weight_decay=0.05)
scheduler = dict(
    type="OneCycleLR",
    max_lr=base_lr,
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)

# Offline PCA prerequisite. The following lines are documentation only; do not
# uncomment them as CVRL config fields. The executable values are already set
# in the __main__ block of utils/dino/pca_dino_feature.py:
# target_dim=128, sample_ratio=1.0, max_samples=1000000,
# max_sample_files=None, normalize_input=True, whiten=False,
# svd_solver="randomized", iterated_power=7, n_oversamples=64,
# power_iteration_normalizer="QR", output_dtype="float32".
# Run once before training:
# conda run -n pointspace python utils/dino/pca_dino_feature.py

model = dict(
    type="CVRL-v1m1",
    mode=mode,
    backbone_out_channels=backbone_out_channels,
    visual_channels=visual_channels,
    return_channels=return_channels,
    native_teacher_channels=1024,
    projector_checkpoint=True,
    return_warmup_ratio=0.1,
    pcgrad_start_ratio=0.1,
    pcgrad_log_interval=100,
    edge_weight="sqrt_count",
    sample_balanced=True,
    validate_mapping=False,
    aggregate_channel_chunk_size=64,
    visual_loss_chunk_size=4096,
    knn_k=knn_k,
    lambda_vpsd=1.0,
    lambda_return=1.0,
    lambda_ctx=1.0,
    head_dropout=0.1,
    ignore_index=ignore_index,
    decoder_detach_stages=(2, 3, 4),
    backbone=dict(
        type="CVRL-v1m1-backbone",
        mode=mode,
        level_channels=(24, 48, 96, 192, 384),
        distill_level=2,
        visual_channels=visual_channels,
        return_channels=return_channels,
        native_channels=1024,
        native_hidden_channels=1024,
        pca_path=pca_model_path,
        require_pca=True,
        # Preserve PCA direction while matching visual/return feature scale.
        pca_output_normalize=True,
        pca_output_scale=visual_channels**0.5,
        visual_token_chunk_size=8192,
        return_backbone=dict(
            type="PT-v3m4",
            in_channels=in_channels,
            order=("z", "z-trans", "hilbert", "hilbert-trans"),
            stride=(2, 2, 2, 2),
            enc_depths=(1, 1, 2, 4, 2),
            enc_channels=(24, 48, 96, 192, 384),
            enc_num_head=(2, 3, 6, 12, 24),
            enc_patch_size=(48, 48, 48, 48, 48),
            dec_depths=(1, 1, 1, 1),
            dec_channels=(96, 96, 128, 192),
            dec_num_head=(6, 6, 8, 12),
            dec_patch_size=(48, 48, 48, 48),
            mlp_ratio=3,
            qkv_bias=True,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
            drop_path=0.3,
            pre_norm=True,
            shuffle_orders=True,
            enable_rpe=False,
            enable_flash=True,
            upcast_attention=False,
            upcast_softmax=False,
            checkpoint_cpe=True,
            checkpoint_attention=True,
            checkpoint_mlp=True,
            random_order=True,
            traceable=True,
            mask_token=False,
            enc_mode=False,
        ),
    ),
    criteria=[
        dict(
            type="CrossEntropyLoss",
            loss_weight=1.0,
            ignore_index=ignore_index,
            auto_class_weight=True,
        )
    ],
)

hooks = [
    dict(type="CheckpointLoader"),
    dict(type="RuntimeInfoHook"),
    dict(type="ModelHook"),
    dict(type="CVRLTrainingSchedule"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter", interval=10),
    dict(type="CacheCleaner", time_multiplier=2.5, step_clean_interval=100),
    dict(type="CVRLEvaluator", log_interval=10),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=True),
]
train = dict(type="CVRLPCGradTrainer")
test = dict(type="CVRLFeatureTester", verbose=True)

feature_output_dir = rf"{data_root}\cvrl_feature\cvrl_v1m1_ptv3_v3m4_{mode}"
feature_source = mode
feature_dtype = "float16"
normalize_feature = False
feature_aggregate_on_gpu = False
feature_overwrite = False

train_transform = [
    dict(type="ZPercentileCenterShift", percentile=2.0),
    dict(type="RobustLogIntensity"),
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
if mode in {"visual", "full"}:
    train_transform.append(dict(type="CompactImagePatches"))

val_transform = [
    dict(type="ZPercentileCenterShift", percentile=2.0),
    dict(type="RobustLogIntensity"),
    dict(
        type="GridSample",
        grid_size=grid_size,
        hash_type="fnv",
        mode="train",
        return_grid_coord=True,
    ),
]
if mode in {"visual", "full"}:
    val_transform.append(dict(type="CompactImagePatches"))

pretrain_keys = ["coord", "grid_coord"]
if mode in {"visual", "full"}:
    pretrain_keys.extend(
        ["dino_feature", "image_patch_index", "image_valid", "dino_offset"]
    )
if mode in {"return", "full"}:
    pretrain_keys.append("return_role")

pretrain_post_transform = [
    dict(type="ToTensor"),
    dict(
        type="Collect",
        keys=tuple(pretrain_keys),
        feat_keys=feature_keys,
    ),
]

if mode == "visual":
    pretrain_dataset_type = "LasImageDataset"
elif mode == "return":
    pretrain_dataset_type = "RRCLLasDataset"
else:
    pretrain_dataset_type = "CVRLLasDataset"

train_dataset = dict(
    type=pretrain_dataset_type,
    split="train",
    data_root=data_root,
    data_path=train_pointcloud_path,
    class_weight="sqrt" if mode in {"return", "full"} else None,
    weight_sample=0.2,
    weighted_sampler=False,
    test_mode=False,
    loop=10,
    transform=train_transform,
    post_transform=pretrain_post_transform,
)
val_dataset = dict(
    type=pretrain_dataset_type,
    split="val",
    data_root=data_root,
    data_path=train_pointcloud_path,
    class_weight=None,
    weighted_sampler=False,
    test_mode=False,
    loop=1,
    transform=val_transform,
    post_transform=pretrain_post_transform,
)
if mode in {"visual", "full"}:
    train_dataset["dino_feature_path"] = dino_feature_path
    val_dataset["dino_feature_path"] = dino_feature_path

data = dict(
    num_classes=num_classes,
    ignore_index=ignore_index,
    names=class_names,
    train=train_dataset,
    val=val_dataset,
    test=dict(
        type="LasDataset",
        split="test",
        data_path=test_pointcloud_path,
        test_mode=True,
        ignore_index=ignore_index,
        transform=[
            dict(type="ZPercentileCenterShift", percentile=2.0),
            dict(type="RobustLogIntensity"),
            dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="test",
                return_grid_coord=True,
                return_inverse=True,
            ),
        ],
        post_transform=[
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "index"),
                feat_keys=feature_keys,
            ),
        ],
    ),
)
