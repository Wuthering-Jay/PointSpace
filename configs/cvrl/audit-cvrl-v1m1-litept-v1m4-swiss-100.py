# Export fixed Swiss-Audit CVRL visual representations from the official
# Swiss-100 checkpoint. Switch only ``feature_source`` between visual (128D)
# and visual_native (1024D); both come from the same trained projector.

data_root = r"G:\data\swiss\joint_tiles_100"
audit_pointcloud_path = rf"{data_root}\Swiss-Audit\pointcloud"
pca_model_path = (
    rf"{data_root}\dino_feature_pca128\dino_pca_model.safetensors"
)

grid_size = 0.25
feature_keys = ("coord", "intensity")
in_channels = 4
ignore_index = -1
num_classes = 4
class_names = ("S", "F", "M", "L")

feature_source = "visual"
feature_output_subdir = {
    "visual": "cvrl_visual_pca128",
    "visual_native": "cvrl_visual_native1024",
    "return": "rrcl",
    "full": "cvrl_full",
    "backbone": "cvrl_full",
}[feature_source]
feature_output_dir = rf"{data_root}\Swiss-Audit\features\{feature_output_subdir}"

save_path = "exp/cvrl/swiss/pretrain-v1m1-litept-v1m4-100-full"
weight = f"{save_path}/model/model_last.pth"
resume = False
evaluate = False
test_only = True
seed = 42

num_worker = 2
batch_size_train = 1  # required by shared runtime setup; unused in test_only
batch_size_test = 4
sync_bn = False
enable_amp = True
amp_dtype = "bfloat16"
find_unused_parameters = False

model = dict(
    type="CVRL-v1m1",
    mode="full",
    backbone_out_channels=200,
    visual_channels=128,
    return_channels=72,
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
    knn_k=48,
    lambda_vpsd=1.0,
    lambda_return=1.0,
    lambda_ctx=1.0,
    head_dropout=0.1,
    ignore_index=ignore_index,
    decoder_detach_stages=(2, 3, 4),
    backbone=dict(
        type="CVRL-v1m1-backbone",
        mode="full",
        level_channels=(36, 72, 144, 252, 504),
        distill_level=2,
        visual_channels=128,
        return_channels=72,
        native_channels=1024,
        native_hidden_channels=1024,
        pca_path=pca_model_path,
        require_pca=True,
        pca_output_normalize=True,
        pca_output_scale=128**0.5,
        visual_token_chunk_size=8192,
        return_backbone=dict(
            type="LitePT-v1m4",
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
            dec_depths=(1, 1, 1, 1),
            dec_channels=(72, 72, 144, 252),
            dec_num_head=(4, 4, 8, 14),
            dec_patch_size=(192, 192, 192, 192),
            dec_conv=(True, True, True, True),
            dec_attn=(False, False, False, False),
            dec_rope_freq=(100.0, 100.0, 100.0, 100.0),
            dec_rope_enable=True,
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
            jitter_coords=None,
            rescale_coords=None,
            checkpoint_mlp=True,
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

test = dict(type="CVRLFeatureTester", verbose=True)
feature_dtype = "float16"
normalize_feature = False
feature_aggregate_on_gpu = False
feature_overwrite = False
feature_index_dir = rf"{data_root}\Swiss-Audit\visibility"
feature_index_key = "audit_index"

data = dict(
    num_classes=num_classes,
    ignore_index=ignore_index,
    names=class_names,
    test=dict(
        type="LasDataset",
        split="test",
        data_path=audit_pointcloud_path,
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
