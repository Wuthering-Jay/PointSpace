# 本文件是 `pretrain-utonia-v1m1-0-base_stagev1.py` 的中文注释版副本。
# 目的：
# 1. 保留原始配置的可执行结构和值，便于直接对照和调试。
# 2. 用中文解释每个主要配置块的含义、影响范围和典型用途。
# 3. 不修改原始配置文件，避免影响现有实验。

# 继承默认运行时配置。
# default_runtime.py 通常会定义日志、分布式训练、随机种子、默认 hooks、
# checkpoint 路径、runner/trainer 默认行为等项目级通用设置。
_base_ = ["../_base_/default_runtime.py"]


# -----------------------------------------------------------------------------
# 基础训练与运行参数
# -----------------------------------------------------------------------------

# 输入图像裁剪后的高度。
# 这里使用 518，是为了能被 patch_size=14 整除：
# 518 / 14 = 37，因此后续 patch_h 会是 37。
crop_h = 518

# 输入图像裁剪后的宽度。
# 同样 518 / 14 = 37，因此 patch_w 也是 37。
crop_w = 518

# 图像编码器的 patch 大小。
# DINOv2 ViT-G/14 使用 14x14 patch，因此这里与图像 backbone 对齐。
patch_size = 14

# 总 batch size，注释中说明是所有 GPU 上的总 batch size。
# 如果使用多卡训练，通常每卡 batch size = batch_size / world_size。
batch_size = 256  # bs: total bs in all gpus

# 数据加载 worker 数。
# 这里设置很大，说明预期运行环境可能是大规模多卡/多进程训练。
# 实际使用时需要确认机器 CPU 核数、共享内存和文件系统吞吐是否足够。
num_worker = 1024

# 混合数据概率，当前为 0，表示不启用 mix 类增强或混合采样逻辑。
mix_prob = 0.0

# 梯度裁剪阈值。
# 用于限制梯度范数，减少大模型预训练时的梯度爆炸风险。
clip_grad = 1.0

# 每轮/每步后是否清理 CUDA cache。
# True 可以缓解显存碎片，但可能略微降低速度。
empty_cache = True

# 启用自动混合精度训练。
# 对大模型预训练很重要，可以降低显存占用并提升吞吐。
enable_amp = True

# AMP 使用 bfloat16。
# bfloat16 比 float16 动态范围更大，在 A100/H100 等硬件上常用于稳定训练。
amp_dtype = "bfloat16"

# 是否在训练过程中执行评估。
# 预训练配置通常不做常规验证集评估，因此为 False。
evaluate = False

# 分布式训练中是否查找未使用参数。
# Utonia 这类 student/teacher、多分支、多 loss 结构可能存在某些分支在特定 step 不参与反传，
# 因此这里设置 True 更稳，但会带来一点额外开销。
find_unused_parameters = True


# -----------------------------------------------------------------------------
# 模型配置
# -----------------------------------------------------------------------------

model = dict(
    # 模型注册名。框架会根据该字符串在 registry 中构建 Utonia v1m1 模型。
    type="Utonia-v1m1",

    # 图像 patch 网格高度和宽度。
    # 518x518 图像经过 14x14 patch 划分后得到 37x37 patch map。
    patch_h=crop_h // patch_size,
    patch_w=crop_w // patch_size,

    # 2D 图像 teacher/backbone 权重名称。
    # dinov2_vitg14_reg 表示使用带 register tokens 的 DINOv2 ViT-G/14。
    image_weight_name="dinov2_vitg14_reg",

    # 图像权重路径或 HuggingFace 模型标识。
    image_weight_path="facebook/dinov2-with-registers-giant",

    # 3D backbone 输出通道数或融合后的输出维度配置。
    # 该值需要与 Utonia 模型内部特征融合/投影逻辑匹配。
    backbone_out_channels=1332,

    # 模型内部 embedding 通道数。
    # 常用于坐标、颜色、法线等输入特征的初始嵌入或融合表示。
    embedding_channels=64,

    # student 是否加载预训练权重。
    # False 表示本阶段从当前配置初始化 student，teacher/2D 权重另由 image_weight_path 控制。
    student_pretrained=False,

    # 2D encoder 特征上采样/类型转换级别。
    # 具体语义由 Utonia 实现决定，通常影响 2D 特征取出或插值过程中的精度处理。
    enc2d_upcast_level=3,

    # -------------------------------------------------------------------------
    # 3D backbone 配置：student 与 teacher 共享的 Point Transformer 结构基础。
    # -------------------------------------------------------------------------
    backbone=dict(
        # 3D backbone 注册名。这里使用 PT-v3m3，即 Point Transformer v3 的某个变体。
        type="PT-v3m3",

        # 输入点特征通道数。
        # 通常可能包含 coord/color/normal 等特征，例如 xyz + rgb + normal = 9。
        in_channels=9,

        # 点云序列化/空间排序方式。
        # z、z-trans、hilbert、hilbert-trans 提供不同空间遍历顺序，
        # 多顺序可以增强模型对空间局部结构的表达。
        order=("z", "z-trans", "hilbert", "hilbert-trans"),

        # 各 stage 的下采样步长。
        # 4 个 stride 对应从浅层到深层的层级降采样。
        stride=(2, 2, 2, 2),

        # encoder 每个 stage 的 block 数。
        # 总 block 数为 3 + 3 + 3 + 12 + 3 = 24。
        enc_depths=(3, 3, 3, 12, 3),

        # encoder 每个 stage 的通道数。
        # 随着下采样加深，通道从 54 增长到 576。
        enc_channels=(54, 108, 216, 432, 576),

        # 每个 stage 的 attention head 数。
        # 通道越大，head 数也越多。
        enc_num_head=(3, 6, 12, 24, 32),

        # 每个 stage 的 patch size。
        # 这里统一为 1024，表示每个局部 attention/group 的点数量上限或窗口大小。
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),

        # Transformer MLP 隐藏层扩展比例。
        mlp_ratio=4,

        # QKV 线性层是否使用 bias。
        qkv_bias=True,

        # attention qk 缩放因子。
        # None 表示使用默认的 1/sqrt(head_dim)。
        qk_scale=None,

        # attention 权重 dropout。
        attn_drop=0.0,

        # projection dropout。
        proj_drop=0.0,

        # stochastic depth/drop path 概率。
        # student 使用 0.3 增强正则化；teacher 下面会覆盖为 0。
        drop_path=0.3,

        # 是否随机打乱多种空间排序顺序。
        # 有助于增强模型对不同点序表示的鲁棒性。
        shuffle_orders=True,

        # 是否使用 pre-norm Transformer 结构。
        # pre-norm 通常在深层 Transformer 中更稳定。
        pre_norm=True,

        # 是否启用相对位置编码。
        enable_rpe=False,

        # 是否启用 FlashAttention 或类似高效 attention 内核。
        enable_flash=True,

        # 是否将 attention 计算上转为更高精度。
        upcast_attention=False,

        # 是否将 softmax 上转为更高精度。
        upcast_softmax=False,

        # encoder-only 模式。
        # 预训练特征学习通常只需要 encoder 表征。
        enc_mode=True,

        # 是否开启 traceable 模式，便于模型跟踪、导出或特定框架调试。
        traceable=True,

        # 是否使用 mask token。
        # Utonia 预训练包含 mask/unmask 相关 loss，因此需要 mask token。
        mask_token=True,

        # RoPE 位置编码的 base 参数。
        rope_base=10,

        # 坐标平移增强配置，仅用于RoPE位置编码
        # None 表示不额外平移坐标，具体增强主要由 transform 处理。
        shift_coords=None,

        # 坐标 jitter 强度。
        jitter_coords=1.1,

        # 坐标 rescale 强度。
        rescale_coords=1.2,
    ),

    # teacher 分支的自定义覆盖参数。
    # teacher 通常作为 EMA 模型，不需要 dropout/drop_path 这类随机正则。
    teacher_custom=dict(
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
    ),

    # 3D projection/head 输入通道，与 backbone 最后一层通道 576 对齐。
    head_in_channels=576,

    # 3D head 隐藏层通道数。
    head_hidden_channels=4096,

    # 3D head 输出 embedding 通道数。
    head_embed_channels=256,

    # 3D head 原型数量。
    # 用于 DINO/iBOT 风格的 prototype/cluster 分布预测。
    head_num_prototypes=4096,

    # 2D encoder head 输入通道。
    # DINOv2 ViT-G/14 的特征维度通常为 1536。
    enc2d_head_in_channels=1536,

    # 2D head 隐藏层通道数。
    enc2d_head_hidden_channels=4096,

    # 2D head 输出 embedding 通道数。
    enc2d_head_embed_channels=256,

    # 2D head 原型数量，与 3D head 对齐。
    enc2d_head_num_prototypes=4096,

    # 每个样本生成的全局视图数量。
    num_global_view=2,

    # 每个样本生成的局部视图数量。
    num_local_view=4,

    # mask 尺寸调度的起始值。
    mask_size_start=10,

    # mask 尺寸调度的目标/基础值。
    mask_size_base=40,

    # mask size warmup 比例，占总训练进度的 5%。
    mask_size_warmup_ratio=0.05,

    # mask ratio 起始值。
    mask_ratio_start=0.3,

    # mask ratio 目标/基础值。
    mask_ratio_base=0.7,

    # mask ratio warmup 比例。
    mask_ratio_warmup_ratio=0.05,

    # mask 抖动强度，避免固定 mask 模式。
    mask_jitter=0.5,

    # teacher temperature 起始值。
    # 较低温度会让 teacher 输出分布更尖锐。
    teacher_temp_start=0.04,

    # teacher temperature 最终/基础值。
    teacher_temp_base=0.07,

    # teacher temperature warmup 比例。
    teacher_temp_warmup_ratio=0.05,

    # student temperature。
    student_temp=0.1,

    # masked token loss 权重。
    mask_loss_weight=1 / 8,

    # roll mask loss 权重。
    # 具体含义取决于 Utonia 实现，通常与 mask 区域或视图滚动匹配有关。
    roll_mask_loss_weight=1 / 8,

    # unmasked token loss 权重。
    unmask_loss_weight=2 / 8,

    # 2D-3D/2D encoder 对齐 loss 权重。
    # 这里占 4/8，是总 loss 中最重的一项。
    enc2d_loss_weight=4 / 8,

    # teacher EMA 动量起始值。
    # 训练初期 teacher 更新相对快一些。
    momentum_base=0.994,

    # teacher EMA 动量最终值。
    # 训练后期趋近 1，teacher 更新更慢、更稳定。
    momentum_final=1,

    # 2D-3D 匹配时每个元素最多考虑的候选数。
    match_max_k=8,

    # 2D-3D 匹配半径上限。
    match_max_r=0.32,

    # 模型内部 upcast 级别。
    up_cast_level=0,

    # 是否对 2D encoder 使用 cosine shift。
    enc2d_cos_shift=True,

    # Sonata 相关模型类型标识。
    # 此处为 online，表示使用在线分支/在线模式。
    sonata_model_type="online",
)


# -----------------------------------------------------------------------------
# 学习率、权重衰减和优化器配置
# -----------------------------------------------------------------------------

# 总训练 epoch 数。
epoch = 100

# 基础学习率。
base_lr = 0.004

# layer-wise learning rate decay。
# 越靠近输入的浅层通常学习率越小，越靠近输出的深层学习率越大。
lr_decay = 0.9  # layer-wise lr decay

# 权重衰减起始值。
# 具体调度由 WeightDecaySchedular hook 负责。
base_wd = 0.04  # wd scheduler enable in hooks

# 权重衰减最终值。
# 预训练中逐渐增大 weight decay 是常见做法，可提升泛化。
final_wd = 0.2  # wd scheduler enable in hooks

# 读取 backbone 各 stage 的深度，用于为每个 encoder block 构造独立学习率。
dec_depths = model["backbone"]["enc_depths"]

# 为 encoder 中每个 block 创建参数组配置。
# keyword 用于匹配参数名，例如 enc0.block0.、enc3.block11.。
# lr 使用 layer-wise decay：
# - 越浅层指数越大，学习率越小。
# - 越深层越接近 base_lr。
param_dicts = [
    dict(
        keyword=f"enc{e}.block{b}.",
        lr=base_lr * lr_decay ** (sum(dec_depths) - sum(dec_depths[:e]) - b - 1),
    )
    for e in range(len(dec_depths))
    for b in range(dec_depths[e])
]

# 删除临时变量，避免被配置系统误收集或污染命名空间。
del dec_depths

# AdamW 优化器。
# weight_decay 这里给 base_wd，但实际训练中还会被 WeightDecaySchedular 动态调度。
optimizer = dict(type="AdamW", lr=base_lr, weight_decay=base_wd)

# OneCycleLR 学习率调度器。
# max_lr 是一个列表：第一个是默认参数组学习率，后面是 layer-wise 参数组学习率。
scheduler = dict(
    type="OneCycleLR",
    max_lr=[base_lr] + [g["lr"] for g in param_dicts],

    # 前 5% 训练进度用于 warmup 到 max_lr。
    pct_start=0.05,

    # 使用 cosine 退火。
    anneal_strategy="cos",

    # 初始学习率 = max_lr / div_factor。
    div_factor=10.0,

    # 最终学习率 = 初始学习率 / final_div_factor。
    final_div_factor=1000.0,
)


# -----------------------------------------------------------------------------
# 图像归一化参数
# -----------------------------------------------------------------------------

# ImageNet 默认均值与标准差。
# DINOv2 图像分支通常使用 ImageNet 风格归一化。
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


# -----------------------------------------------------------------------------
# 室外自动驾驶类数据增强：Waymo 使用
# -----------------------------------------------------------------------------

outdoor_transform = [
    # 指定后续索引/过滤时认为有效的字段。
    # outdoor 数据中保留 coord、origin_coord、color、normal、superpoint、
    # strength、instance 以及图像点云对应关系等字段。
    dict(
        type="Update",
        keys_dict={
            "index_valid_keys": (
                "coord",
                "origin_coord",
                "color",
                "normal",
                "superpoint",
                "strength",
                "instance",
                "correspondence",
                "global_correspondence",
            )
        },
    ),

    # 图像增强。
    # 负责裁剪到 518x518，并同步生成 37x37 patch 网格相关信息。
    dict(
        type="ImgAugmentation",
        crop_h=crop_h,
        crop_w=crop_w,
        patch_h=crop_h // patch_size,
        patch_w=crop_w // patch_size,
        patch_size=patch_size,
        imgtransforms=[
            # 图像颜色扰动，概率 0.95。
            dict(type="ImgChromaticJitter", p=0.95, std=0.05),

            # 图像高斯模糊，概率 0.5。
            dict(type="ImgGaussianBlur", p=0.5),

            # 按 ImageNet 统计量归一化图像。
            dict(
                type="Imgnormalize",
                mean=IMAGENET_DEFAULT_MEAN,
                std=IMAGENET_DEFAULT_STD,
            ),
        ],
    ),

    # 将当前 coord 复制为 origin_coord。
    # origin_coord 用于保留增强前或采样前的原始坐标，便于后续建立对应关系。
    dict(type="Copy", keys_dict={"coord": "origin_coord"}),

    # 室外点云先整体缩放到较小尺度。
    # Waymo 等自动驾驶场景尺度很大，先缩小有助于统一到网络处理范围。
    dict(type="RandomScale", scale=[0.18, 0.22]),

    # 网格采样，下采样点云。
    # grid_size=0.01 表示按 1cm 网格做采样；hash_type="fnv" 表示使用 FNV hash。
    dict(type="GridSample", grid_size=0.01, hash_type="fnv", mode="train"),

    # 随机丢弃全部颜色，应用概率 0.2。
    # 让模型不能过度依赖 RGB。
    dict(type="RandomDropColor", drop_ratio=1.0, drop_application_ratio=0.2),

    # 随机丢弃 10% 颜色，应用概率 0.5。
    dict(type="RandomDropColor", drop_ratio=0.1, drop_application_ratio=0.5),

    # 随机丢弃全部法线，应用概率 0.2。
    dict(type="RandomDropNormal", drop_ratio=1.0, drop_application_ratio=0.2),

    # 随机丢弃 10% 法线，应用概率 0.5。
    dict(type="RandomDropNormal", drop_ratio=0.1, drop_application_ratio=0.5),

    # 多视图生成器。
    # 从同一场景生成 2 个 global view 和 4 个 local view，
    # 供 Utonia 的多视图自监督目标使用。
    dict(
        type="MultiViewGenerator",

        # 多视图中需要保留并同步处理的字段。
        view_keys=("coord", "origin_coord", "color", "correspondence", "normal"),

        # 是否选择帧。室外序列数据可能有多帧/多 sweep。
        if_frame_selected=True,

        # 全局视图数量与裁剪比例范围。
        global_view_num=2,
        global_view_scale=(0.4, 1.0),

        # 局部视图数量与裁剪比例范围。
        local_view_num=4,
        local_view_scale=(0.1, 0.4),

        # 全局视图共享增强。
        # 这里仅做颜色归一化。
        global_shared_transform=[
            dict(type="NormalizeColor"),
        ],

        # 每个 global view 独立应用的几何增强。
        global_transform=[
            # 可选的中心平移，这里关闭。
            # dict(type="CenterShift", apply_z=True),

            # 绕 z 轴大角度随机旋转。
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.8),

            # 绕 x/y 轴小角度随机旋转，模拟轻微姿态扰动。
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.8),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.8),

            # 随机翻转。
            dict(type="RandomFlip", p=0.5),

            # 裁剪点云范围。
            # 注意前面已经缩放 0.18~0.22，这里的范围也乘了 0.2。
            dict(
                type="PointClip",
                point_cloud_range=(
                    -75.2 * 0.2,
                    -75.2 * 0.2,
                    -4 * 0.2,
                    75.2 * 0.2,
                    75.2 * 0.2,
                    2 * 0.2,
                ),
            ),

            # 小幅随机缩放与抖动。
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomJitter", sigma=0.0025, clip=0.01),

            # 室外增强中暂未启用弹性形变。
            # dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
        ],

        # 每个 local view 独立应用的几何增强。
        # 与 global_transform 基本一致，但最后额外做 NormalizeColor。
        local_transform=[
            # dict(type="CenterShift", apply_z=True),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.8),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.8),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.8),
            dict(type="RandomFlip", p=0.5),
            dict(
                type="PointClip",
                point_cloud_range=(
                    -75.2 * 0.2,
                    -75.2 * 0.2,
                    -4 * 0.2,
                    75.2 * 0.2,
                    75.2 * 0.2,
                    2 * 0.2,
                ),
            ),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomJitter", sigma=0.0025, clip=0.01),
            # dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            dict(type="NormalizeColor"),
        ],

        # 每个视图最多保留的 3D 点数。
        max_size=32768,

        # 2D encoder 对应关系最多保留的点/patch 数。
        enc2d_max_size=32768,

        # 2D 分支使用的尺度采样范围。
        enc2d_scale=(0.8, 1),
    ),

    # 转为 tensor。
    dict(type="ToTensor"),

    # 写入 grid_size 元信息，供模型或 collate 阶段使用。
    dict(type="Update", keys_dict={"grid_size": 0.01}),

    # 收集最终送入模型的数据字段。
    dict(
        type="Collect",
        keys=(
            "global_origin_coord",
            "global_coord",
            "global_offset",
            "local_origin_coord",
            "local_coord",
            "local_offset",
            "grid_size",
            "name",
            "images",
            "global_correspondence",
            "img_num",
        ),
        offset_keys_dict=dict(),

        # global/local view 输入特征由坐标、颜色、法线组成。
        global_feat_keys=("global_coord", "global_color", "global_normal"),
        local_feat_keys=("local_coord", "local_color", "local_normal"),
    ),
]


# -----------------------------------------------------------------------------
# 物体类数据增强：PartNet 使用
# -----------------------------------------------------------------------------

obj_transform = [
    # 物体数据额外包含 segment 字段。
    dict(
        type="Update",
        keys_dict={
            "index_valid_keys": (
                "coord",
                "origin_coord",
                "color",
                "normal",
                "superpoint",
                "strength",
                "segment",
                "instance",
                "correspondence",
                "global_correspondence",
            )
        },
    ),

    # 图像增强，与 outdoor_transform 保持一致。
    dict(
        type="ImgAugmentation",
        crop_h=crop_h,
        crop_w=crop_w,
        patch_h=crop_h // patch_size,
        patch_w=crop_w // patch_size,
        patch_size=patch_size,
        imgtransforms=[
            dict(type="ImgChromaticJitter", p=0.95, std=0.05),
            dict(type="ImgGaussianBlur", p=0.5),
            dict(
                type="Imgnormalize",
                mean=IMAGENET_DEFAULT_MEAN,
                std=IMAGENET_DEFAULT_STD,
            ),
        ],
    ),

    # 物体坐标归一化。
    # 对 PartNet 这类 CAD/物体数据，归一化可以消除不同物体尺度差异。
    dict(type="NormalizeCoord"),

    # 归一化后再做较大范围随机缩放，增强尺度鲁棒性。
    dict(type="RandomScale", scale=[0.5, 1.5]),

    # 保存 origin_coord，供多视图和 2D-3D 对应关系使用。
    dict(type="Copy", keys_dict={"coord": "origin_coord"}),

    # 可选 z 方向平移，当前关闭。
    # dict(type="ZShift", apply_center=True),

    # 网格采样。
    dict(type="GridSample", grid_size=0.01, hash_type="fnv", mode="train"),

    # 随机丢弃颜色和法线，增强对几何/外观缺失的鲁棒性。
    dict(type="RandomDropColor", drop_ratio=1.0, drop_application_ratio=0.2),
    dict(type="RandomDropColor", drop_ratio=0.1, drop_application_ratio=0.5),
    dict(type="RandomDropNormal", drop_ratio=1.0, drop_application_ratio=0.2),
    dict(type="RandomDropNormal", drop_ratio=0.1, drop_application_ratio=0.5),

    # 多视图生成。
    # 物体数据的 global/local view scale 都偏大，
    # 说明更强调保留完整物体结构，而不是像室外/室内场景那样裁很小局部。
    dict(
        type="MultiViewGenerator",
        global_view_num=2,
        global_view_scale=(0.8, 1.0),
        local_view_num=4,
        local_view_scale=(0.6, 0.8),

        # 多个 global view 共享的颜色增强。
        global_shared_transform=[
            dict(
                type="RandomColorJitter",
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.02,
                p=0.8,
            ),
            dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            # dict(type="ChromaticJitter", p=0.95, std=0.05),
            dict(type="NormalizeColor"),
        ],

        # global view 几何增强。
        global_transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="RandomShift", shift=((-0.2, 0.2), (-0.2, 0.2), (-0.2, 0.2))),
            dict(type="RandomScale", scale=[0.5, 1.5]),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.8),
            dict(type="RandomRotate", angle=[-1, 1], axis="x", p=0.8),
            dict(type="RandomRotate", angle=[-1, 1], axis="y", p=0.8),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
        ],

        # local view 几何增强和颜色增强。
        # local 分支额外再次做颜色扰动，增加不同视图间外观差异。
        local_transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="RandomShift", shift=((-0.2, 0.2), (-0.2, 0.2), (-0.2, 0.2))),
            dict(type="RandomScale", scale=[0.5, 1.5]),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.8),
            dict(type="RandomRotate", angle=[-1, 1], axis="x", p=0.8),
            dict(type="RandomRotate", angle=[-1, 1], axis="y", p=0.8),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            # dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
            dict(
                type="RandomColorJitter",
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.02,
                p=0.8,
            ),
            dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            # dict(type="ChromaticJitter", p=0.95, std=0.05),
            dict(type="NormalizeColor"),
        ],

        # 物体数据允许保留更多点。
        max_size=65536,
        enc2d_max_size=65536,
        enc2d_scale=(0.8, 1),
    ),
    dict(type="ToTensor"),
    dict(type="Update", keys_dict={"grid_size": 0.01}),
    dict(
        type="Collect",
        keys=(
            "global_origin_coord",
            "global_coord",
            "global_offset",
            "local_origin_coord",
            "local_coord",
            "local_offset",
            "grid_size",
            "name",
            "images",
            "global_correspondence",
            "img_num",
        ),
        offset_keys_dict=dict(),
        global_feat_keys=("global_coord", "global_color", "global_normal"),
        local_feat_keys=("local_coord", "local_color", "local_normal"),
    ),
]


# -----------------------------------------------------------------------------
# 室内场景类数据增强：ScanNet 与 Structured3D 使用
# -----------------------------------------------------------------------------

indoor_transform = [
    # 室内数据与物体数据一样保留 segment 字段。
    dict(
        type="Update",
        keys_dict={
            "index_valid_keys": (
                "coord",
                "origin_coord",
                "color",
                "normal",
                "superpoint",
                "strength",
                "segment",
                "instance",
                "correspondence",
                "global_correspondence",
            )
        },
    ),

    # 图像增强。
    dict(
        type="ImgAugmentation",
        crop_h=crop_h,
        crop_w=crop_w,
        patch_h=crop_h // patch_size,
        patch_w=crop_w // patch_size,
        patch_size=patch_size,
        imgtransforms=[
            dict(type="ImgChromaticJitter", p=0.95, std=0.05),
            dict(type="ImgGaussianBlur", p=0.5),
            dict(
                type="Imgnormalize",
                mean=IMAGENET_DEFAULT_MEAN,
                std=IMAGENET_DEFAULT_STD,
            ),
        ],
    ),

    # 记录原始坐标。
    dict(type="Copy", keys_dict={"coord": "origin_coord"}),

    # 室内场景整体缩放到 0.45~0.55。
    # 与 outdoor 的 0.18~0.22 相比，室内场景原始尺度通常更小，因此缩放幅度不同。
    dict(type="RandomScale", scale=[0.45, 0.55]),

    # 网格采样。
    dict(type="GridSample", grid_size=0.01, hash_type="fnv", mode="train"),

    # 颜色和法线随机丢弃。
    dict(type="RandomDropColor", drop_ratio=1.0, drop_application_ratio=0.2),
    dict(type="RandomDropColor", drop_ratio=0.1, drop_application_ratio=0.5),
    dict(type="RandomDropNormal", drop_ratio=1.0, drop_application_ratio=0.2),
    dict(type="RandomDropNormal", drop_ratio=0.1, drop_application_ratio=0.5),

    # 多视图生成。
    # 室内场景 global/local 的裁剪比例与 outdoor 一致：
    # global 0.4~1.0，local 0.1~0.4。
    dict(
        type="MultiViewGenerator",

        # 可选视图字段设置，当前关闭。
        # view_keys=("coord", "origin_coord", "color", "normal"),

        global_view_num=2,
        global_view_scale=(0.4, 1.0),
        local_view_num=4,
        local_view_scale=(0.1, 0.4),

        # 室内场景使用更强的颜色增强。
        global_shared_transform=[
            dict(
                type="RandomColorJitter",
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.02,
                p=0.8,
            ),
            dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            # dict(type="ChromaticJitter", p=0.95, std=0.05),
            dict(type="NormalizeColor"),
        ],

        # global view 几何增强。
        global_transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.8),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.8),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.8),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.0025, clip=0.01),

            # 室内场景启用较弱弹性形变。
            dict(type="ElasticDistortion", distortion_params=[[0.1, 0.2], [0.4, 0.8]]),
        ],

        # local view 增强。
        # 与 global view 类似，但额外在 local 上再做颜色扰动。
        local_transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.8),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.8),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.8),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.0025, clip=0.01),
            dict(type="ElasticDistortion", distortion_params=[[0.1, 0.2], [0.4, 0.8]]),
            # dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
            dict(
                type="RandomColorJitter",
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.02,
                p=0.8,
            ),
            dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            # dict(type="ChromaticJitter", p=0.95, std=0.05),
            dict(type="NormalizeColor"),
        ],

        # 室内场景最多保留 65536 点。
        max_size=65536,
        enc2d_max_size=65536,
        enc2d_scale=(0.8, 1),
    ),

    dict(type="ToTensor"),
    dict(type="Update", keys_dict={"grid_size": 0.01}),
    dict(
        type="Collect",
        keys=(
            "global_origin_coord",
            "global_coord",
            "global_offset",
            "local_origin_coord",
            "local_coord",
            "local_offset",
            "grid_size",
            "name",
            "images",
            "global_correspondence",
            "img_num",
        ),
        offset_keys_dict=dict(),
        global_feat_keys=("global_coord", "global_color", "global_normal"),
        local_feat_keys=("local_coord", "local_color", "local_normal"),
    ),
]


# -----------------------------------------------------------------------------
# 数据集配置
# -----------------------------------------------------------------------------

# 数据集权重。
# None 表示不在这里显式指定各数据集采样权重，使用 ConcatDataset 或默认逻辑。
data_weight = None

# 数据长度限制。
# None 表示不在这里显式截断整体数据长度。
data_length = None

data = dict(
    train=dict(
        # 多数据集拼接。
        # Stage v1 使用 4 个数据源：Waymo、PartNet、ScanNet、Structured3D。
        type="ConcatDataset",
        datasets=[
            # -----------------------------------------------------------------
            # Waymo 10 Hz：室外自动驾驶点云 + 图像数据
            # -----------------------------------------------------------------
            dict(
                type="WaymoImagePointDataset",

                # 是否使用多 sweep 点云。
                if_sweep=True,

                # 是否加载图像数据。
                if_img=True,

                # 使用 3 帧 sweep。
                sweeps=3,

                # sweep 间隔为 1。
                sweep_gap=1,

                # 图像裁剪与 patch 设置传入数据集。
                crop_h=crop_h,
                crop_w=crop_w,
                patch_size=patch_size,

                # 使用 training 和 validation split 参与预训练。
                split=["training", "validation"],

                # 数据根目录。
                data_root="data/waymo",

                # 使用室外增强流水线。
                transform=outdoor_transform,

                # 训练模式。
                test_mode=False,

                # 数据集重复次数。
                loop=1,
            ),

            # -----------------------------------------------------------------
            # PartNet：物体级数据
            # -----------------------------------------------------------------
            dict(
                type="PartNetDataDataset",
                crop_h=crop_h,
                crop_w=crop_w,
                patch_size=patch_size,
                split=["train"],
                data_root="data/partnet_data_v0",

                # 使用物体增强流水线。
                transform=obj_transform,
                test_mode=False,
                loop=1,
            ),

            # -----------------------------------------------------------------
            # ScanNet：室内 RGB-D/点云场景数据
            # -----------------------------------------------------------------
            dict(
                type="DefaultImagePointDataset",
                crop_h=crop_h,
                crop_w=crop_w,
                patch_size=patch_size,
                split=["train", "val", "test"],
                data_root="data/scannet",

                # 使用室内增强流水线。
                transform=indoor_transform,
                test_mode=False,
                loop=1,
            ),

            # -----------------------------------------------------------------
            # Structured3D：合成室内场景数据
            # -----------------------------------------------------------------
            dict(
                type="DefaultImagePointDataset",
                crop_h=crop_h,
                crop_w=crop_w,
                patch_size=patch_size,
                split=["train", "val", "test"],
                data_root="data/structured3d",
                transform=indoor_transform,
                test_mode=False,
                loop=1,
            ),
        ],
    )
)


# -----------------------------------------------------------------------------
# Hooks
# -----------------------------------------------------------------------------

hooks = [
    # 加载 checkpoint。
    # 可能用于恢复训练或加载预训练权重，具体行为由 CheckpointLoader 实现和 runtime 配置决定。
    dict(type="CheckpointLoader"),

    # 模型相关 hook。
    # 通常负责 teacher/student EMA、温度调度、mask 调度或模型内部状态更新等。
    dict(type="ModelHook"),

    # 权重衰减调度 hook。
    # 从 base_wd=0.04 调度到 final_wd=0.2。
    dict(type="WeightDecaySchedular", base_value=base_wd, final_value=final_wd),

    # 迭代计时器。
    # warmup_iter=2 表示前 2 次迭代可能不纳入稳定耗时统计。
    dict(type="IterationTimer", warmup_iter=2),

    # 信息写入 hook。
    # 通常负责日志、指标、学习率、loss 等信息输出。
    dict(type="InformationWriter"),

    # checkpoint 保存 hook。
    # 每 10 个 epoch 或特定单位保存一次，具体单位由框架实现决定。
    dict(type="CheckpointSaver", save_freq=10),
]
