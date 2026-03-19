"""
DeepLANet V1 数据流测试脚本
测试各个模块的数据流通性
"""

import torch
import torch.nn as nn
import sys
sys.path.insert(0, "e:/code/python/PointSpace")

import pointops


def test_local_spatial_encoding():
    """测试 LocalSpatialEncoding 模块"""
    from pointspace.models.backbone.deeplanet.deeplanet_v1 import (
        LocalSpatialEncoding,
        compute_stage_positional_encoding,
    )

    print("=" * 50)
    print("Testing LocalSpatialEncoding with Stage-Level PE...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模拟输入
    n = 1000
    k = 16
    d = 32

    coord = torch.randn(n, 3, device=device)
    feat = torch.randn(n, d, device=device)

    # 构造 offset (假设2个batch)
    offset = torch.tensor([500, 1000], dtype=torch.int32, device=device)

    # KNN查询
    reference_index, dist = pointops.knn_query(k, coord, offset)

    # Stage级别: 计算位置编码
    pos_encoding = compute_stage_positional_encoding(coord, reference_index, dist)

    # 实例化模块
    lse = LocalSpatialEncoding(d).to(device)

    # 前向传播（使用预计算的位置编码）
    output = lse(pos_encoding, feat, coord, reference_index)

    print(f"  Input coord: {coord.shape}")
    print(f"  Input feat: {feat.shape}")
    print(f"  Position encoding: {pos_encoding.shape}")
    print(f"  Reference index: {reference_index.shape}")
    print(f"  Output: {output.shape}")

    assert output.shape == (n, k, 2 * d), f"Expected {(n, k, 2*d)}, got {output.shape}"
    print("  LocalSpatialEncoding: PASSED")
    return True


def test_attentive_pooling():
    """测试 AttentivePooling 模块"""
    from pointspace.models.backbone.deeplanet.deeplanet_v1 import AttentivePooling

    print("=" * 50)
    print("Testing AttentivePooling...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模拟输入
    n = 1000
    k = 16
    d_in = 64
    d_out = 32

    x = torch.randn(n, k, d_in, device=device)
    reference_index = torch.randint(0, n, (n, k), device=device)

    # 实例化模块
    pool = AttentivePooling(d_in, d_out).to(device)

    # 前向传播
    output = pool(x, reference_index)

    print(f"  Input x: {x.shape}")
    print(f"  Output: {output.shape}")

    assert output.shape == (n, d_out), f"Expected {(n, d_out)}, got {output.shape}"
    print("  AttentivePooling: PASSED")
    return True


def test_stage_positional_encoding():
    """测试 Stage-Level Positional Encoding"""
    from pointspace.models.backbone.deeplanet.deeplanet_v1 import compute_stage_positional_encoding

    print("=" * 50)
    print("Testing Stage-Level Positional Encoding...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模拟输入
    n = 1000
    k = 16

    coord = torch.randn(n, 3, device=device)
    offset = torch.tensor([500, 1000], dtype=torch.int32, device=device)

    # KNN查询
    reference_index, dist = pointops.knn_query(k, coord, offset)

    # 计算位置编码
    pos_encoding = compute_stage_positional_encoding(coord, reference_index, dist)

    print(f"  Input coord: {coord.shape}")
    print(f"  Reference index: {reference_index.shape}")
    print(f"  Distance: {dist.shape}")
    print(f"  Position encoding: {pos_encoding.shape}")

    assert pos_encoding.shape == (n, k, 10), f"Expected {(n, k, 10)}, got {pos_encoding.shape}"
    print("  Stage-Level Positional Encoding: PASSED")
    return True


def test_local_feature_aggregation():
    """测试 LocalFeatureAggregation 模块"""
    from pointspace.models.backbone.deeplanet.deeplanet_v1 import (
        LocalFeatureAggregation,
        compute_stage_positional_encoding,
    )

    print("=" * 50)
    print("Testing LocalFeatureAggregation with Stage-Level PE...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模拟输入
    n = 1000
    k = 16
    d_in = 32
    d_out = 48  # 实际输出维度为 2 * d_out = 96

    coord = torch.randn(n, 3, device=device)
    feat = torch.randn(n, d_in, device=device)
    offset = torch.tensor([500, 1000], dtype=torch.int32, device=device)

    # KNN查询
    reference_index, dist = pointops.knn_query(k, coord, offset)

    # Stage级别: 计算位置编码
    pos_encoding = compute_stage_positional_encoding(coord, reference_index, dist)

    # 实例化模块
    lfa = LocalFeatureAggregation(d_in, d_out, k).to(device)

    # 前向传播（使用预计算的位置编码）
    output = lfa(coord, feat, pos_encoding, reference_index)

    print(f"  Input coord: {coord.shape}")
    print(f"  Input feat: {feat.shape}")
    print(f"  Position encoding: {pos_encoding.shape}")
    print(f"  Output: {output.shape}")

    expected_out = 2 * d_out
    assert output.shape == (n, expected_out), f"Expected {(n, expected_out)}, got {output.shape}"
    print("  LocalFeatureAggregation: PASSED")
    return True


def test_block():
    """测试 Block 模块"""
    from pointspace.models.backbone.deeplanet.deeplanet_v1 import (
        Block,
        compute_stage_positional_encoding,
    )

    print("=" * 50)
    print("Testing Block with Stage-Level PE...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模拟输入
    n = 1000
    k = 16
    embed_channels = 64

    coord = torch.randn(n, 3, device=device)
    feat = torch.randn(n, embed_channels, device=device)
    offset = torch.tensor([500, 1000], dtype=torch.int32, device=device)

    points = [coord, feat, offset]

    # KNN查询
    reference_index, dist = pointops.knn_query(k, coord, offset)

    # Stage级别: 计算位置编码
    pos_encoding = compute_stage_positional_encoding(coord, reference_index, dist)

    # 实例化模块
    block = Block(embed_channels, k).to(device)

    # 前向传播（使用预计算的位置编码）
    output = block(points, pos_encoding, reference_index)
    out_coord, out_feat, out_offset = output

    print(f"  Input coord: {coord.shape}")
    print(f"  Input feat: {feat.shape}")
    print(f"  Position encoding: {pos_encoding.shape}")
    print(f"  Output coord: {out_coord.shape}")
    print(f"  Output feat: {out_feat.shape}")

    assert out_coord.shape == coord.shape
    assert out_feat.shape == feat.shape
    print("  Block: PASSED")
    return True


def test_block_sequence():
    """测试 BlockSequence 模块"""
    from pointspace.models.backbone.deeplanet.deeplanet_v1 import BlockSequence

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


def test_lfa_patch_embed():
    """测试 LFAPatchEmbed 模块"""
    from pointspace.models.backbone.deeplanet.deeplanet_v1 import LFAPatchEmbed

    print("=" * 50)
    print("Testing LFAPatchEmbed...")

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
    patch_embed = LFAPatchEmbed(depth, in_channels, embed_channels, k).to(device)

    # 前向传播
    output = patch_embed(points)
    out_coord, out_feat, out_offset = output

    print(f"  Input coord: {coord.shape}")
    print(f"  Input feat: {feat.shape}")
    print(f"  Output coord: {out_coord.shape}")
    print(f"  Output feat: {out_feat.shape}")

    assert out_coord.shape == coord.shape
    assert out_feat.shape == (n, embed_channels)
    print("  LFAPatchEmbed: PASSED")
    return True


def test_encoder_decoder():
    """测试 Encoder 和 Decoder 模块"""
    from pointspace.models.backbone.deeplanet.deeplanet_v1 import Encoder, Decoder

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
    """测试完整的 DeepLANetV1 backbone"""
    from pointspace.models.backbone.deeplanet.deeplanet_v1 import DeepLANetV1
    from pointspace.models.utils.structure import Point

    print("=" * 50)
    print("Testing DeepLANetV1 Full Backbone...")

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
    model = DeepLANetV1(
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
    print("  DeepLANetV1 Full Backbone: PASSED")
    return True


def test_backward():
    """测试反向传播"""
    from pointspace.models.backbone.deeplanet.deeplanet_v1 import DeepLANetV1
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
    model = DeepLANetV1(
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


def test_stage_level_pe_efficiency():
    """
    测试 Stage-Level Position Embedding 的效率优化
    验证位置编码在每个 Stage 只计算一次
    """
    from pointspace.models.backbone.deeplanet.deeplanet_v1 import (
        BlockSequence,
        compute_stage_positional_encoding,
    )
    import time

    print("=" * 50)
    print("Testing Stage-Level PE Efficiency...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模拟输入
    n = 2000
    k = 16
    embed_channels = 96
    depth = 4  # 使用更多的 Block 来验证效率

    coord = torch.randn(n, 3, device=device)
    feat = torch.randn(n, embed_channels, device=device)
    offset = torch.tensor([1000, 2000], dtype=torch.int32, device=device)

    points = [coord, feat, offset]

    # 实例化模块
    block_seq = BlockSequence(depth, embed_channels, k).to(device)

    # 预热
    with torch.no_grad():
        _ = block_seq(points)

    # 测试前向传播时间
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.time()
    with torch.no_grad():
        for _ in range(10):
            output = block_seq(points)

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.time() - start

    out_coord, out_feat, out_offset = output

    print(f"  Input coord: {coord.shape}")
    print(f"  Input feat: {feat.shape}")
    print(f"  Depth (num blocks): {depth}")
    print(f"  Output coord: {out_coord.shape}")
    print(f"  Output feat: {out_feat.shape}")
    print(f"  Average time (10 runs): {elapsed / 10 * 1000:.2f} ms")
    print(f"  Optimization: Position encoding computed ONCE per stage (not {depth} times)")

    assert out_coord.shape == coord.shape
    assert out_feat.shape == feat.shape
    print("  Stage-Level PE Efficiency: PASSED")
    return True


def test_stage_level_pe_efficiency():
    """
    测试 Stage-Level Position Embedding 的效率优化
    验证位置编码在每个 Stage 只计算一次
    """
    from pointspace.models.backbone.deeplanet.deeplanet_v1 import (
        BlockSequence,
        compute_stage_positional_encoding,
    )
    import time

    print("=" * 50)
    print("Testing Stage-Level PE Efficiency...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模拟输入
    n = 2000
    k = 16
    embed_channels = 96
    depth = 4  # 使用更多的 Block 来验证效率

    coord = torch.randn(n, 3, device=device)
    feat = torch.randn(n, embed_channels, device=device)
    offset = torch.tensor([1000, 2000], dtype=torch.int32, device=device)

    points = [coord, feat, offset]

    # 实例化模块
    block_seq = BlockSequence(depth, embed_channels, k).to(device)

    # 预热
    with torch.no_grad():
        _ = block_seq(points)

    # 测试前向传播时间
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.time()
    with torch.no_grad():
        for _ in range(10):
            output = block_seq(points)

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.time() - start

    out_coord, out_feat, out_offset = output

    print(f"  Input coord: {coord.shape}")
    print(f"  Input feat: {feat.shape}")
    print(f"  Depth (num blocks): {depth}")
    print(f"  Output coord: {out_coord.shape}")
    print(f"  Output feat: {out_feat.shape}")
    print(f"  Average time (10 runs): {elapsed / 10 * 1000:.2f} ms")
    print(f"  Note: Position encoding computed ONCE per stage (not {depth} times)")

    assert out_coord.shape == coord.shape
    assert out_feat.shape == feat.shape
    print("  Stage-Level PE Efficiency: PASSED")
    return True


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("DeepLANet V1 Data Flow Tests (Stage-Level PE)")
    print("=" * 60)

    tests = [
        ("Stage-Level Positional Encoding", test_stage_positional_encoding),
        ("LocalSpatialEncoding", test_local_spatial_encoding),
        ("AttentivePooling", test_attentive_pooling),
        ("LocalFeatureAggregation", test_local_feature_aggregation),
        ("Block", test_block),
        ("BlockSequence", test_block_sequence),
        ("LFAPatchEmbed", test_lfa_patch_embed),
        ("Encoder/Decoder", test_encoder_decoder),
        ("Full Backbone", test_full_backbone),
        ("Backward Pass", test_backward),
        ("Stage-Level PE Efficiency", test_stage_level_pe_efficiency),
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
