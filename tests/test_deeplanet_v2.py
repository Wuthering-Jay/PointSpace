"""
DeepLANet V2 数据流测试脚本
测试轻量级 VFR 模块和 ResLFE Block 的数据流通性
"""

import torch
import torch.nn as nn
import sys
sys.path.insert(0, "e:/code/python/PointSpace")

import pointops


def test_positional_encoding_encoder():
    """测试 PositionalEncodingEncoder 模块"""
    from pointspace.models.backbone.deeplanet.deeplanet_v2 import (
        PositionalEncodingEncoder,
        compute_stage_positional_encoding,
    )

    print("=" * 50)
    print("Testing PositionalEncodingEncoder...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模拟输入
    n = 1000
    k = 16
    embed_channels = 64

    coord = torch.randn(n, 3, device=device)
    offset = torch.tensor([500, 1000], dtype=torch.int32, device=device)

    # KNN查询
    reference_index, dist = pointops.knn_query(k, coord, offset)

    # 计算位置编码
    pos_encoding = compute_stage_positional_encoding(coord, reference_index, dist)

    # 实例化模块
    pe_encoder = PositionalEncodingEncoder(embed_channels).to(device)

    # 前向传播
    pe_feat = pe_encoder(pos_encoding)

    print(f"  Input position encoding: {pos_encoding.shape}")
    print(f"  Output PE feature: {pe_feat.shape}")

    assert pe_feat.shape == (n, embed_channels), f"Expected {(n, embed_channels)}, got {pe_feat.shape}"
    print("  PositionalEncodingEncoder: PASSED")
    return True


def test_vfr_module():
    """测试 VFRModule 模块"""
    from pointspace.models.backbone.deeplanet.deeplanet_v2 import VFRModule

    print("=" * 50)
    print("Testing VFRModule...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模拟输入
    n = 1000
    k = 16
    embed_channels = 64

    coord = torch.randn(n, 3, device=device)
    feat = torch.randn(n, embed_channels, device=device)
    offset = torch.tensor([500, 1000], dtype=torch.int32, device=device)

    # KNN查询
    reference_index, dist = pointops.knn_query(k, coord, offset)

    # 实例化模块
    vfr = VFRModule().to(device)

    # 前向传播
    output = vfr(feat, coord, reference_index)

    print(f"  Input feat: {feat.shape}")
    print(f"  Reference index: {reference_index.shape}")
    print(f"  Output: {output.shape}")

    assert output.shape == feat.shape, f"Expected {feat.shape}, got {output.shape}"
    print("  VFRModule: PASSED")
    return True


def test_reslfe_block():
    """测试 ResLFEBlock 模块"""
    from pointspace.models.backbone.deeplanet.deeplanet_v2 import (
        ResLFEBlock,
        PositionalEncodingEncoder,
        compute_stage_positional_encoding,
    )

    print("=" * 50)
    print("Testing ResLFEBlock...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模拟输入
    n = 1000
    k = 16
    embed_channels = 64

    coord = torch.randn(n, 3, device=device)
    feat = torch.randn(n, embed_channels, device=device)
    offset = torch.tensor([500, 1000], dtype=torch.int32, device=device)

    points = [coord, feat, offset]

    # KNN查询和位置编码
    reference_index, dist = pointops.knn_query(k, coord, offset)
    pos_encoding = compute_stage_positional_encoding(coord, reference_index, dist)

    # PE encoder
    pe_encoder = PositionalEncodingEncoder(embed_channels).to(device)
    pe = pe_encoder(pos_encoding)

    # 实例化模块
    block = ResLFEBlock(embed_channels).to(device)

    # 前向传播
    output = block(points, pe, reference_index)
    out_coord, out_feat, out_offset = output

    print(f"  Input coord: {coord.shape}")
    print(f"  Input feat: {feat.shape}")
    print(f"  PE feature: {pe.shape}")
    print(f"  Output coord: {out_coord.shape}")
    print(f"  Output feat: {out_feat.shape}")

    assert out_coord.shape == coord.shape
    assert out_feat.shape == feat.shape
    print("  ResLFEBlock: PASSED")
    return True


def test_block_sequence():
    """测试 BlockSequence 模块"""
    from pointspace.models.backbone.deeplanet.deeplanet_v2 import BlockSequence

    print("=" * 50)
    print("Testing BlockSequence...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模拟输入
    n = 1000
    k = 16
    embed_channels = 64
    depth = 2

    coord = torch.randn(n, 3, device=device)
    feat = torch.randn(n, embed_channels, device=device)
    offset = torch.tensor([500, 1000], dtype=torch.int32, device=device)

    points = [coord, feat, offset]

    # 实例化模块
    block_seq = BlockSequence(depth, embed_channels, k).to(device)

    # 前向传播
    output = block_seq(points)
    out_coord, out_feat, out_offset = output

    print(f"  Input coord: {coord.shape}")
    print(f"  Input feat: {feat.shape}")
    print(f"  Output coord: {out_coord.shape}")
    print(f"  Output feat: {out_feat.shape}")

    assert out_coord.shape == coord.shape
    assert out_feat.shape == feat.shape
    print("  BlockSequence: PASSED")
    return True


def test_vfr_patch_embed():
    """测试 VFRPatchEmbed 模块"""
    from pointspace.models.backbone.deeplanet.deeplanet_v2 import VFRPatchEmbed

    print("=" * 50)
    print("Testing VFRPatchEmbed...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模拟输入
    n = 1000
    k = 8
    in_channels = 6
    embed_channels = 48
    depth = 1

    coord = torch.randn(n, 3, device=device)
    feat = torch.randn(n, in_channels, device=device)
    offset = torch.tensor([500, 1000], dtype=torch.int32, device=device)

    points = [coord, feat, offset]

    # 实例化模块
    patch_embed = VFRPatchEmbed(depth, in_channels, embed_channels, k).to(device)

    # 前向传播
    output = patch_embed(points)
    out_coord, out_feat, out_offset = output

    print(f"  Input coord: {coord.shape}")
    print(f"  Input feat: {feat.shape}")
    print(f"  Output coord: {out_coord.shape}")
    print(f"  Output feat: {out_feat.shape}")

    assert out_coord.shape == coord.shape
    assert out_feat.shape == (n, embed_channels)
    print("  VFRPatchEmbed: PASSED")
    return True


def test_encoder_decoder():
    """测试 Encoder 和 Decoder 模块"""
    from pointspace.models.backbone.deeplanet.deeplanet_v2 import Encoder, Decoder

    print("=" * 50)
    print("Testing Encoder and Decoder...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模拟输入
    n = 1000
    in_channels = 48
    embed_channels = 96
    depth = 1
    grid_size = 0.1
    k = 16

    coord = torch.randn(n, 3, device=device)
    feat = torch.randn(n, in_channels, device=device)
    offset = torch.tensor([500, 1000], dtype=torch.int32, device=device)

    points = [coord, feat, offset]

    # 测试 Encoder
    encoder = Encoder(depth, in_channels, embed_channels, grid_size, k).to(device)
    enc_output, cluster = encoder(points)
    enc_coord, enc_feat, enc_offset = enc_output

    print(f"  Encoder Input coord: {coord.shape}")
    print(f"  Encoder Input feat: {feat.shape}")
    print(f"  Encoder Output coord: {enc_coord.shape}")
    print(f"  Encoder Output feat: {enc_feat.shape}")
    print(f"  Cluster: {cluster.shape}")

    # 测试 Decoder
    decoder = Decoder(embed_channels, in_channels, in_channels, depth, k).to(device)
    dec_output = decoder(enc_output, points, cluster)
    dec_coord, dec_feat, dec_offset = dec_output

    print(f"  Decoder Output coord: {dec_coord.shape}")
    print(f"  Decoder Output feat: {dec_feat.shape}")

    assert dec_coord.shape == coord.shape
    assert dec_feat.shape == (n, in_channels)
    print("  Encoder and Decoder: PASSED")
    return True


def test_full_backbone():
    """测试完整的 DeepLANetV2 backbone"""
    from pointspace.models.backbone.deeplanet.deeplanet_v2 import DeepLANetV2
    from pointspace.models.utils.structure import Point

    print("=" * 50)
    print("Testing DeepLANetV2 Full Backbone...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模拟输入
    n = 2000
    in_channels = 6

    coord = torch.randn(n, 3, device=device)
    feat = torch.randn(n, in_channels, device=device)
    offset = torch.tensor([1000, 2000], dtype=torch.int32, device=device)

    # 创建 Point 对象
    data_dict = {
        "coord": coord,
        "feat": feat,
        "offset": offset,
    }
    point = Point(data_dict)

    # 使用较小的配置加速测试
    model = DeepLANetV2(
        in_channels=in_channels,
        patch_embed_depth=1,
        patch_embed_channels=48,
        patch_embed_neighbours=8,
        enc_depths=(1, 1, 1, 1),
        enc_channels=(96, 192, 384, 512),
        enc_neighbours=(16, 16, 16, 16),
        dec_depths=(1, 1, 1, 1),
        dec_channels=(48, 96, 192, 384),
        dec_neighbours=(16, 16, 16, 16),
        grid_sizes=(0.06, 0.12, 0.24, 0.48),
        drop_path_rate=0.1,
        enable_checkpoint=False,
        unpool_backend="map",
    ).to(device)

    # 前向传播
    output_point = model(point)

    print(f"  Input coord: {coord.shape}")
    print(f"  Input feat: {feat.shape}")
    print(f"  Output feat: {output_point.feat.shape}")

    # 验证输出
    assert output_point.feat.shape[0] == n
    assert output_point.feat.shape[1] == 48  # dec_channels[0]
    print(f"  Expected output feat dim: {48}, Actual: {output_point.feat.shape[1]}")
    print("  DeepLANetV2 Full Backbone: PASSED")
    return True


def test_backward():
    """测试反向传播"""
    from pointspace.models.backbone.deeplanet.deeplanet_v2 import DeepLANetV2
    from pointspace.models.utils.structure import Point

    print("=" * 50)
    print("Testing Backward Pass...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模拟输入
    n = 1000
    in_channels = 6

    coord = torch.randn(n, 3, device=device, requires_grad=False)
    feat = torch.randn(n, in_channels, device=device, requires_grad=True)
    offset = torch.tensor([500, 1000], dtype=torch.int32, device=device)

    # 创建 Point 对象
    data_dict = {
        "coord": coord,
        "feat": feat,
        "offset": offset,
    }
    point = Point(data_dict)

    # 使用较小的配置
    model = DeepLANetV2(
        in_channels=in_channels,
        patch_embed_depth=1,
        patch_embed_channels=32,
        patch_embed_neighbours=8,
        enc_depths=(1, 1),
        enc_channels=(64, 128),
        enc_neighbours=(8, 8),
        dec_depths=(1, 1),
        dec_channels=(32, 64),
        dec_neighbours=(8, 8),
        grid_sizes=(0.1, 0.2),
        drop_path_rate=0.0,
        enable_checkpoint=False,
        unpool_backend="map",
    ).to(device)

    # 前向传播
    output_point = model(point)

    # 计算损失并反向传播
    loss = output_point.feat.sum()
    loss.backward()

    print(f"  Loss: {loss.item()}")
    print(f"  Gradient exists: {feat.grad is not None}")
    if feat.grad is not None:
        print(f"  Gradient shape: {feat.grad.shape}")
        print(f"  Gradient mean: {feat.grad.mean().item():.6f}")

    assert feat.grad is not None, "Gradient should exist"
    print("  Backward Pass: PASSED")
    return True


def test_v1_v2_comparison():
    """比较 V1 和 V2 的参数量和计算效率"""
    from pointspace.models.backbone.deeplanet.deeplanet_v1 import DeepLANetV1
    from pointspace.models.backbone.deeplanet.deeplanet_v2 import DeepLANetV2
    from pointspace.models.utils.structure import Point
    import time

    print("=" * 50)
    print("Testing V1 vs V2 Comparison...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模拟输入
    n = 2000
    in_channels = 6

    coord = torch.randn(n, 3, device=device)
    feat = torch.randn(n, in_channels, device=device)
    offset = torch.tensor([1000, 2000], dtype=torch.int32, device=device)

    data_dict = {
        "coord": coord,
        "feat": feat,
        "offset": offset,
    }

    # 相同配置
    config = {
        "in_channels": in_channels,
        "patch_embed_depth": 1,
        "patch_embed_channels": 48,
        "patch_embed_neighbours": 8,
        "enc_depths": (1, 1, 2, 1),
        "enc_channels": (96, 192, 384, 512),
        "enc_neighbours": (16, 16, 16, 16),
        "dec_depths": (1, 1, 1, 1),
        "dec_channels": (48, 96, 192, 384),
        "dec_neighbours": (16, 16, 16, 16),
        "grid_sizes": (0.06, 0.12, 0.24, 0.48),
        "drop_path_rate": 0.1,
        "enable_checkpoint": False,
        "unpool_backend": "map",
    }

    # V1
    print("\n  Creating V1 model...")
    model_v1 = DeepLANetV1(**config).to(device)
    params_v1 = sum(p.numel() for p in model_v1.parameters())

    # V2
    print("  Creating V2 model...")
    model_v2 = DeepLANetV2(**config).to(device)
    params_v2 = sum(p.numel() for p in model_v2.parameters())

    print(f"\n  V1 Parameters: {params_v1:,}")
    print(f"  V2 Parameters: {params_v2:,}")
    print(f"  Parameter reduction: {(1 - params_v2 / params_v1) * 100:.2f}%")

    # 预热
    with torch.no_grad():
        _ = model_v1(Point(data_dict))
        _ = model_v2(Point(data_dict))

    # 测试速度 - V1
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.time()
    with torch.no_grad():
        for _ in range(5):
            _ = model_v1(Point(data_dict))

    if device.type == "cuda":
        torch.cuda.synchronize()

    time_v1 = (time.time() - start) / 5

    # 测试速度 - V2
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.time()
    with torch.no_grad():
        for _ in range(5):
            _ = model_v2(Point(data_dict))

    if device.type == "cuda":
        torch.cuda.synchronize()

    time_v2 = (time.time() - start) / 5

    print(f"\n  V1 Inference time: {time_v1 * 1000:.2f} ms")
    print(f"  V2 Inference time: {time_v2 * 1000:.2f} ms")
    print(f"  Speedup: {time_v1 / time_v2:.2f}x")

    print("\n  V1 vs V2 Comparison: PASSED")
    return True


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("DeepLANet V2 Data Flow Tests")
    print("Lightweight VFR + ResLFE Architecture")
    print("=" * 60)

    tests = [
        ("PositionalEncodingEncoder", test_positional_encoding_encoder),
        ("VFRModule", test_vfr_module),
        ("ResLFEBlock", test_reslfe_block),
        ("BlockSequence", test_block_sequence),
        ("VFRPatchEmbed", test_vfr_patch_embed),
        ("Encoder/Decoder", test_encoder_decoder),
        ("Full Backbone", test_full_backbone),
        ("Backward Pass", test_backward),
        ("V1 vs V2 Comparison", test_v1_v2_comparison),
    ]

    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, "PASSED" if result else "FAILED"))
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, f"ERROR: {e}"))

    # 打印测试摘要
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    for name, status in results:
        status_icon = "[PASS]" if status == "PASSED" else "[FAIL]"
        print(f"  {status_icon} {name}: {status}")

    passed = sum(1 for _, s in results if s == "PASSED")
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")

    return all(s == "PASSED" for _, s in results)


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
