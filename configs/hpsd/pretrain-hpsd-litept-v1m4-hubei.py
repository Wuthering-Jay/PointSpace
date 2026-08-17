# HPSD LitePT-v1m4：训练与点特征提取完整配置。
# tools/train.py 和 tools/test.py 均直接读取本文件，不依赖 base 配置。

# -------------------------------------------------------
# 0. 数据路径与点输入特征
# -------------------------------------------------------
data_root = r"E:\data\湖北\joint_tiles"
pointcloud_path = r"E:\data\云南\data\tile"
feature_output_dir = r"E:\data\云南\hpsd_feature\litept_v1m4_concat"
grid_size = 0.5  # 训练体素及测试 fragment 划分使用相同空间分辨率
feature_keys = ("coord", "intensity", "echo")  # 3 + 1 + 2 = 6 维
in_channels = 6

# -------------------------------------------------------
# 1. Checkpoint 与运行控制
# -------------------------------------------------------
save_path = "exp/hubei/hpsd/pretrain-litept-v1m4-concat-native1024"
weight = "exp/hubei/hpsd/pretrain-litept-v1m4-concat-native1024/model/model_last.pth"
resume = False
evaluate = False  # HPSD 预训练没有语义标签验证集，不构建 data.val
test_only = False
seed = 42

# -------------------------------------------------------
# 2. Batch 与数据加载
# -------------------------------------------------------
num_worker = 8
batch_size_train = 20  # 全部 GPU 上期望的有效训练 batch
batch_size_test = 1  # 全部 GPU 配置值；每 GPU fragment batch = 4 // world_size
gradient_accumulation_steps = 4  # 单卡时 micro-batch=1，累计 4 步更新

# -------------------------------------------------------
# 3. 训练循环与数值精度
# -------------------------------------------------------
epoch = 10
clip_grad = 3.0

sync_bn = False
enable_amp = True
amp_dtype = "bfloat16"
find_unused_parameters = False

# -------------------------------------------------------
# 4. 日志与优化器
# -------------------------------------------------------
enable_wandb = False
wandb_project = "pointspace-hpsd"
wandb_key = None
mix_prob = 0.0

base_lr = 0.001
optimizer = dict(type="AdamW", lr=base_lr, weight_decay=0.05)
scheduler = dict(
    type="OneCycleLR",
    max_lr=base_lr,
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)

# optimizer = dict(type="AdamW", lr=1e-3, weight_decay=2e-3)
# scheduler = dict(
#     type="CosineAnnealingLR",
#     total_steps=epoch,
# )
param_dicts = [dict(keyword="block", lr=1e-4)] # example: [dict(keyword="block", lr_scale=0.1)]

hooks = [
    dict(type="CheckpointLoader"),
    dict(type="RuntimeInfoHook"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter", interval=10),
    dict(type="CacheCleaner", time_multiplier=5, step_clean_interval=200),
    dict(type="CheckpointSaver", save_freq=None),
]

train = dict(type="DefaultTrainer")
test = dict(type="HPSDFeatureTester", verbose=True)

# -------------------------------------------------------
# 5. HPSD 测试特征导出
# -------------------------------------------------------
# projected：导出 concat 层级表示投影后的 DINO 对齐 1024 维特征；
# backbone：导出投影前的 concat 层级表示。
feature_source = "projected"
feature_dtype = "float16"  # Safetensors 输出类型，仅支持 float16/float32
normalize_feature = True  # fragment 合并后再次执行 L2 归一化
feature_aggregate_on_gpu = False  # GPU index_add 更快，但会占用 N*C*4 字节
feature_overwrite = False

# -------------------------------------------------------
# 6. HPSD 蒸馏模型
# -------------------------------------------------------
model = dict(
    type="HPSD-v1m1",
    # 以 level 2 的 token 尺度建立 token-patch 对应关系。
    distill_level=2,
    # 必须与 backbone enc_channels 完全一致，用于检查层级输出并确定融合维数。
    level_channels=(36, 72, 144, 252, 504),
    # 保持 DINOv3 ViT-L 原生通道，不预先使用 PCA 压缩 teacher。
    teacher_channels=1024,
    # 一个 token-patch edge 可能由多个点支持，sqrt_count 可抑制密度偏置。
    edge_weight="sqrt_count",
    # 每个 tile 先独立平均 patch loss，再在 batch 内对 tile 等权平均。
    sample_balanced=True,
    # correspondence 由离线工具确定性生成；关闭逐步越界检查以减少同步开销。
    validate_mapping=False,
    # 将 level 3/4 通过 pooling_inverse 对齐到 level 2 后按通道 concat。
    fuse_deeper_features=True,
    # concat 后只对有效 patch 使用轻量 MLP 映射到 DINO 原生 1024 维。
    projector_hidden_channels=1024,
    backbone=dict(
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
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        shuffle_orders=True,
        enable_flash=True,
        traceable=True,  # 保留 pooling_parent/inverse，构造输入点到层级 token 映射
        mask_token=False,
        enc_mode=True,  # HPSD 只预训练 encoder，不构建语义分割 decoder
    ),
)

# -------------------------------------------------------
# 7. 训练数据：点云 + DINO patch + 点到 patch 关系
# -------------------------------------------------------
data = dict(
    train=dict(
        type="LasImageDataset",
        split="train",
        data_root=data_root,
        test_mode=False,
        loop=10,
        transform=[
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
            # 裁剪后仅保留被当前点引用的 patch；不改变 DINO 值和 1024 维通道。
            dict(type="CompactDinoPatches"),
        ],
        post_transform=[
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=(
                    "coord",
                    "grid_coord",
                    "dino_feature",
                    "dino_pixel_coord",
                    "dino_patch_index",
                    "dino_valid",
                    "dino_offset",
                    "dino_source_patch_index",
                    "dino_original_size",
                    "dino_padded_size",
                    "dino_feature_size",
                    "dino_patch_size",
                ),
                feat_keys=feature_keys,
            ),
        ],
    ),
    # 测试仅需要点云。HPSDFeatureTester 不重新读取影像、DINO 或 correspondence。
    test=dict(
        type="LasDataset",
        split="test",
        data_path=pointcloud_path,
        test_mode=True,
        ignore_index=-1,
        transform=[
            dict(type="ZPercentileCenterShift", percentile=2.0),
            dict(type="RobustLogIntensity"),
            dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="test",  # 返回覆盖全部原始点的 fragment 列表
                return_grid_coord=True,
                return_inverse=True,
            ),
        ],
        post_transform=[
            dict(type="ToTensor"),
            dict(
                type="Collect",
                # index 用于把多个 fragment 的预测累加回原始点顺序。
                keys=("coord", "grid_coord", "index"),
                feat_keys=feature_keys,
            ),
        ],
    ),
)
