"""
测试 DeepLANet 深层网络稳定性优化
测试零初始化、LayerScale 和混合深监督功能
"""

import torch
from pointspace.models.builder import build_model


def test_zero_initialization():
    """测试零初始化: 验证残差分支初始输出为 0"""
    print("\n=== 测试零初始化 ===")

    # V1 模型
    cfg_v1 = {
        "type": "DeepLANet-V1",
        "in_channels": 6,
        "patch_embed_depth": 1,
        "patch_embed_channels": 32,
        "enc_depths": [1, 1],
        "enc_channels": [64, 96],
        "dec_depths": [1, 1],
        "dec_channels": [32, 64],
        "grid_sizes": [0.1, 0.2],
    }
    model_v1 = build_model(cfg_v1)

    # 检查 V1 中所有 Block 的 norm3.norm.weight 是否为 0
    zero_init_count_v1 = 0
    for name, param in model_v1.named_parameters():
        if "norm3.norm.weight" in name:
            if torch.allclose(param, torch.zeros_like(param)):
                zero_init_count_v1 += 1

    print(f"V1: 找到 {zero_init_count_v1} 个零初始化的 norm3.norm.weight")
    assert zero_init_count_v1 > 0, "V1 零初始化失败"

    # V2 模型
    cfg_v2 = {
        "type": "DeepLANet-V2",
        "in_channels": 6,
        "patch_embed_depth": 1,
        "patch_embed_channels": 32,
        "enc_depths": [1, 1],
        "enc_channels": [64, 96],
        "dec_depths": [1, 1],
        "dec_channels": [32, 64],
        "grid_sizes": [0.1, 0.2],
    }
    model_v2 = build_model(cfg_v2)

    # 检查 V2 中所有 ResLFEBlock 的 norm2.norm.weight 是否为 0
    zero_init_count_v2 = 0
    for name, param in model_v2.named_parameters():
        if "norm2.norm.weight" in name and "blocks.blocks" in name:
            if torch.allclose(param, torch.zeros_like(param)):
                zero_init_count_v2 += 1

    print(f"V2: 找到 {zero_init_count_v2} 个零初始化的 norm2.norm.weight")
    assert zero_init_count_v2 > 0, "V2 零初始化失败"

    print("✓ 零初始化测试通过")


def test_layer_scale():
    """测试 LayerScale: 验证 gamma 参数是否正确创建"""
    print("\n=== 测试 LayerScale ===")

    # V1 模型 (启用 LayerScale)
    cfg_v1 = {
        "type": "DeepLANet-V1",
        "in_channels": 6,
        "patch_embed_depth": 1,
        "patch_embed_channels": 32,
        "enc_depths": [1, 1],
        "enc_channels": [64, 96],
        "dec_depths": [1, 1],
        "dec_channels": [32, 64],
        "grid_sizes": [0.1, 0.2],
        "enable_layer_scale": True,
        "layer_scale_init_value": 1e-6,
    }
    model_v1 = build_model(cfg_v1)

    # 检查 V1 中是否有 gamma 参数
    gamma_count_v1 = 0
    for name, param in model_v1.named_parameters():
        if "gamma" in name:
            gamma_count_v1 += 1
            # 验证初始值是否正确
            expected_value = 1e-6
            if torch.allclose(param, torch.full_like(param, expected_value), atol=1e-7):
                print(f"  V1 gamma 参数 {name} 初始化为 {expected_value}")

    print(f"V1: 找到 {gamma_count_v1} 个 gamma 参数")
    assert gamma_count_v1 > 0, "V1 LayerScale gamma 参数未创建"

    # V2 模型 (启用 LayerScale)
    cfg_v2 = {
        "type": "DeepLANet-V2",
        "in_channels": 6,
        "patch_embed_depth": 1,
        "patch_embed_channels": 32,
        "enc_depths": [1, 1],
        "enc_channels": [64, 96],
        "dec_depths": [1, 1],
        "dec_channels": [32, 64],
        "grid_sizes": [0.1, 0.2],
        "enable_layer_scale": True,
        "layer_scale_init_value": 1e-5,
    }
    model_v2 = build_model(cfg_v2)

    # 检查 V2 中是否有 gamma 参数
    gamma_count_v2 = 0
    for name, param in model_v2.named_parameters():
        if "gamma" in name:
            gamma_count_v2 += 1
            expected_value = 1e-5
            if torch.allclose(param, torch.full_like(param, expected_value), atol=1e-6):
                print(f"  V2 gamma 参数 {name} 初始化为 {expected_value}")

    print(f"V2: 找到 {gamma_count_v2} 个 gamma 参数")
    assert gamma_count_v2 > 0, "V2 LayerScale gamma 参数未创建"

    print("✓ LayerScale 测试通过")


def test_deep_supervision():
    """测试混合深监督: 验证中间特征是否正确返回"""
    print("\n=== 测试混合深监督 ===")

    # 创建测试数据
    batch_size = 2
    num_points = 1000
    in_channels = 6

    coord = torch.rand(num_points, 3)
    feat = torch.rand(num_points, in_channels)
    offset = torch.tensor([500, 1000])

    data_dict = {
        "coord": coord,
        "feat": feat,
        "offset": offset,
    }

    # V1 模型 (启用深监督)
    cfg_v1 = {
        "type": "DeepLANet-V1",
        "in_channels": 6,
        "patch_embed_depth": 1,
        "patch_embed_channels": 32,
        "enc_depths": [1, 1],
        "enc_channels": [64, 96],
        "dec_depths": [1, 1],
        "dec_channels": [32, 64],
        "grid_sizes": [0.1, 0.2],
        "enable_deep_supervision": True,
    }
    model_v1 = build_model(cfg_v1)
    model_v1.eval()

    with torch.no_grad():
        output_v1 = model_v1(data_dict)

    # 检查是否返回了 aux_outputs
    assert hasattr(output_v1, "aux_outputs"), "V1 未返回 aux_outputs"
    assert len(output_v1.aux_outputs) == 2, f"V1 应该返回 2 个中间特征，但返回了 {len(output_v1.aux_outputs)}"

    print(f"V1 中间特征数量: {len(output_v1.aux_outputs)}")
    for i, aux_points in enumerate(output_v1.aux_outputs):
        coord_aux, feat_aux, offset_aux = aux_points
        print(f"  Stage {i}: coord {coord_aux.shape}, feat {feat_aux.shape}, offset {offset_aux.shape}")

    # V2 模型 (启用深监督)
    cfg_v2 = {
        "type": "DeepLANet-V2",
        "in_channels": 6,
        "patch_embed_depth": 1,
        "patch_embed_channels": 32,
        "enc_depths": [1, 1],
        "enc_channels": [64, 96],
        "dec_depths": [1, 1],
        "dec_channels": [32, 64],
        "grid_sizes": [0.1, 0.2],
        "enable_deep_supervision": True,
    }
    model_v2 = build_model(cfg_v2)
    model_v2.eval()

    with torch.no_grad():
        output_v2 = model_v2(data_dict)

    # 检查是否返回了 aux_outputs
    assert hasattr(output_v2, "aux_outputs"), "V2 未返回 aux_outputs"
    assert len(output_v2.aux_outputs) == 2, f"V2 应该返回 2 个中间特征，但返回了 {len(output_v2.aux_outputs)}"

    print(f"V2 中间特征数量: {len(output_v2.aux_outputs)}")
    for i, aux_points in enumerate(output_v2.aux_outputs):
        coord_aux, feat_aux, offset_aux = aux_points
        print(f"  Stage {i}: coord {coord_aux.shape}, feat {feat_aux.shape}, offset {offset_aux.shape}")

    print("✓ 混合深监督测试通过")


def test_forward_consistency():
    """测试前向传播一致性: 启用和不启用优化时输出形状应该一致"""
    print("\n=== 测试前向传播一致性 ===")

    # 创建测试数据
    batch_size = 2
    num_points = 1000
    in_channels = 6

    coord = torch.rand(num_points, 3)
    feat = torch.rand(num_points, in_channels)
    offset = torch.tensor([500, 1000])

    data_dict = {
        "coord": coord,
        "feat": feat,
        "offset": offset,
    }

    # V1: 不启用优化
    cfg_v1_base = {
        "type": "DeepLANet-V1",
        "in_channels": 6,
        "patch_embed_depth": 1,
        "patch_embed_channels": 32,
        "enc_depths": [1, 1],
        "enc_channels": [64, 96],
        "dec_depths": [1, 1],
        "dec_channels": [32, 64],
        "grid_sizes": [0.1, 0.2],
    }
    model_v1_base = build_model(cfg_v1_base)
    model_v1_base.eval()

    with torch.no_grad():
        output_v1_base = model_v1_base(data_dict)

    # V1: 启用所有优化
    cfg_v1_opt = {
        **cfg_v1_base,
        "enable_layer_scale": True,
        "layer_scale_init_value": 1e-6,
        "enable_deep_supervision": True,
    }
    model_v1_opt = build_model(cfg_v1_opt)
    model_v1_opt.eval()

    with torch.no_grad():
        output_v1_opt = model_v1_opt(data_dict)

    # 验证输出形状一致
    assert output_v1_base.feat.shape == output_v1_opt.feat.shape, "V1 优化前后输出形状不一致"
    print(f"V1 输出形状: {output_v1_opt.feat.shape}")
    print(f"V1 启用优化后有 aux_outputs: {hasattr(output_v1_opt, 'aux_outputs')}")

    # V2: 不启用优化
    cfg_v2_base = {
        "type": "DeepLANet-V2",
        "in_channels": 6,
        "patch_embed_depth": 1,
        "patch_embed_channels": 32,
        "enc_depths": [1, 1],
        "enc_channels": [64, 96],
        "dec_depths": [1, 1],
        "dec_channels": [32, 64],
        "grid_sizes": [0.1, 0.2],
    }
    model_v2_base = build_model(cfg_v2_base)
    model_v2_base.eval()

    with torch.no_grad():
        output_v2_base = model_v2_base(data_dict)

    # V2: 启用所有优化
    cfg_v2_opt = {
        **cfg_v2_base,
        "enable_layer_scale": True,
        "layer_scale_init_value": 1e-6,
        "enable_deep_supervision": True,
    }
    model_v2_opt = build_model(cfg_v2_opt)
    model_v2_opt.eval()

    with torch.no_grad():
        output_v2_opt = model_v2_opt(data_dict)

    # 验证输出形状一致
    assert output_v2_base.feat.shape == output_v2_opt.feat.shape, "V2 优化前后输出形状不一致"
    print(f"V2 输出形状: {output_v2_opt.feat.shape}")
    print(f"V2 启用优化后有 aux_outputs: {hasattr(output_v2_opt, 'aux_outputs')}")

    print("✓ 前向传播一致性测试通过")


if __name__ == "__main__":
    print("开始测试 DeepLANet 深层网络稳定性优化...")

    try:
        test_zero_initialization()
        test_layer_scale()
        test_deep_supervision()
        test_forward_consistency()

        print("\n" + "=" * 60)
        print("所有测试通过！")
        print("=" * 60)

    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
