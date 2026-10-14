data_root = 'G:/data/swiss/joint_tiles_100'
train_pointcloud_path = 'G:/data/swiss/joint_tiles_100/pointcloud'
audit_pointcloud_path = 'G:/data/swiss/joint_tiles_100/Swiss-Audit/pointcloud'
feature_output_dir = 'G:/data/swiss/joint_tiles_100/Swiss-Audit/features/point_m2ae_litept'
save_path = 'exp/baselines/other_execution/point_m2ae_baseline'
weight = None
resume = False
grid_size = 0.25
feature_keys = ('coord', 'intensity')
in_channels = 4
num_classes = 1
class_names = ('unlabeled', )
ignore_index = -1
evaluate = False
test_only = False
seed = 42
num_worker = 2
prefetch_factor = 2
val_prefetch_factor = 1
persistent_workers = True
batch_size_train = 16
batch_size_val = 4
batch_size_test = 4
gradient_accumulation_steps = 4
epoch = 20
clip_grad = 3.0
sync_bn = False
enable_amp = True
amp_dtype = 'bfloat16'
find_unused_parameters = False
mix_prob = 0.0
enable_wandb = False
wandb_project = 'pointspace-cvrl-45'
wandb_key = None
model = dict(
    type='PointM2AE-LitePT-v1m1',
    backbone_out_channels=72,
    fine_channels=72,
    coarse_channels=504,
    coarse_level=4,
    mask_group_size=4.0,
    fine_group_size=2.0,
    mask_ratio=0.8,
    max_coarse_groups=512,
    max_fine_groups=1024,
    fine_points_per_group=16,
    min_group_points=4,
    neighbor_k=8,
    fine_loss_weight=1.0,
    backbone=dict(
        type='LitePT-v1m4',
        in_channels=4,
        order=('z', 'z-trans', 'hilbert', 'hilbert-trans'),
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
        checkpoint_mlp=True))
base_lr = 0.001
optimizer = dict(type='AdamW', lr=0.001, weight_decay=0.05)
scheduler = dict(
    type='OneCycleLR',
    max_lr=0.001,
    pct_start=0.05,
    anneal_strategy='cos',
    div_factor=10.0,
    final_div_factor=1000.0)
param_dicts = [dict(keyword='block', lr=0.0001)]
hooks = [
    dict(type='RuntimeInfoHook'),
    dict(type='ModelHook'),
    dict(type='IterationTimer', warmup_iter=1),
    dict(type='InformationWriter', interval=1),
    dict(type='CacheCleaner', time_multiplier=2.5, step_clean_interval=200)
]
train = dict(type='DefaultTrainer')
test = dict(type='PointM2AEFeatureTester', verbose=True)
feature_source = 'backbone'
feature_dtype = 'float16'
normalize_feature = False
feature_aggregate_on_gpu = False
feature_overwrite = False
feature_index_dir = 'G:/data/swiss/joint_tiles_100/Swiss-Audit/visibility'
feature_index_key = 'audit_index'
train_transform = [
    dict(type='ZPercentileCenterShift', percentile=2.0),
    dict(type='RobustLogIntensity'),
    dict(
        type='RandomDropout', dropout_ratio=0.2,
        dropout_application_ratio=0.2),
    dict(
        type='RandomRotate', angle=[-1, 1], axis='z', center=[0, 0, 0], p=0.5),
    dict(type='RandomScale', scale=[0.9, 1.1]),
    dict(type='RandomFlip', p=0.5),
    dict(type='RandomJitter', sigma=0.005, clip=0.02),
    dict(
        type='GridSample',
        grid_size=0.25,
        hash_type='fnv',
        mode='train',
        return_grid_coord=True),
    dict(type='SphereCrop', point_max=220000, mode='random')
]
val_transform = [
    dict(type='ZPercentileCenterShift', percentile=2.0),
    dict(type='RobustLogIntensity'),
    dict(
        type='GridSample',
        grid_size=0.25,
        hash_type='fnv',
        mode='train',
        return_grid_coord=True),
    dict(type='SphereCrop', point_max=220000, mode='center')
]
pretrain_post_transform = [
    dict(type='ToTensor'),
    dict(
        type='Collect',
        keys=('coord', 'grid_coord'),
        feat_keys=('coord', 'intensity'))
]
data = dict(
    num_classes=1,
    ignore_index=-1,
    names=('unlabeled', ),
    train=dict(
        type='LasDataset',
        split='train',
        data_root='G:\data\swiss\joint_tiles_100',
        data_path='G:\data\swiss\joint_tiles_100\pointcloud',
        class_weight=None,
        weighted_sampler=False,
        test_mode=False,
        loop=1,
        transform=[
            dict(type='ZPercentileCenterShift', percentile=2.0),
            dict(type='RobustLogIntensity'),
            dict(
                type='RandomDropout',
                dropout_ratio=0.2,
                dropout_application_ratio=0.2),
            dict(
                type='RandomRotate',
                angle=[-1, 1],
                axis='z',
                center=[0, 0, 0],
                p=0.5),
            dict(type='RandomScale', scale=[0.9, 1.1]),
            dict(type='RandomFlip', p=0.5),
            dict(type='RandomJitter', sigma=0.005, clip=0.02),
            dict(
                type='GridSample',
                grid_size=0.25,
                hash_type='fnv',
                mode='train',
                return_grid_coord=True),
            dict(type='SphereCrop', point_max=220000, mode='random')
        ],
        post_transform=[
            dict(type='ToTensor'),
            dict(
                type='Collect',
                keys=('coord', 'grid_coord'),
                feat_keys=('coord', 'intensity'))
        ]),
    val=dict(
        type='LasDataset',
        split='val',
        data_root='G:\data\swiss\joint_tiles_100',
        data_path='G:\data\swiss\joint_tiles_100\pointcloud',
        class_weight=None,
        weighted_sampler=False,
        test_mode=False,
        loop=1,
        transform=[
            dict(type='ZPercentileCenterShift', percentile=2.0),
            dict(type='RobustLogIntensity'),
            dict(
                type='GridSample',
                grid_size=0.25,
                hash_type='fnv',
                mode='train',
                return_grid_coord=True),
            dict(type='SphereCrop', point_max=220000, mode='center')
        ],
        post_transform=[
            dict(type='ToTensor'),
            dict(
                type='Collect',
                keys=('coord', 'grid_coord'),
                feat_keys=('coord', 'intensity'))
        ]),
    test=dict(
        type='LasDataset',
        split='test',
        data_path='G:\data\swiss\joint_tiles_100\Swiss-Audit\pointcloud',
        test_mode=True,
        ignore_index=-1,
        transform=[
            dict(type='ZPercentileCenterShift', percentile=2.0),
            dict(type='RobustLogIntensity'),
            dict(
                type='GridSample',
                grid_size=0.25,
                hash_type='fnv',
                mode='test',
                return_grid_coord=True,
                return_inverse=True)
        ],
        post_transform=[
            dict(type='ToTensor'),
            dict(
                type='Collect',
                keys=('coord', 'grid_coord', 'index'),
                feat_keys=('coord', 'intensity'))
        ]))
max_train_steps = 8
