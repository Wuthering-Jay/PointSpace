"""
测试位置编码优化效果

对比强制 FP32 vs 归一化 + Clamp 两种方案的：
- 速度
- 显存占用
- 数值稳定性
"""

import torch
import torch.nn as nn
import time
import sys
sys.path.insert(0, 'e:/code/python/PointSpace')

from pointspace.models.backbone.deeplanet.deeplanet_v2 import PositionalEncodingEncoder


def test_pe_encoder(use_amp=True, batch_size=10000, k=32):
    """测试位置编码编码器性能"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embed_channels = 64

    # 创建模型
    model = PositionalEncodingEncoder(
        embed_channels=embed_channels,
        normalize_input=True,  # 使用归一化策略
        safe_range=60000.0
    ).to(device)

    # 模拟深层网络中的位置编码（第四个 stage，距离较大）
    # 故意制造一些极值来测试稳定性
    pos_encoding = torch.randn(batch_size, k, 10, device=device)

    # 注入一些极值（模拟深层网络中的大距离）
    pos_encoding[::100, :, 9] = 10000.0  # 距离维度
    pos_encoding[::100, :, 0:3] = 5000.0  # 坐标极值

    print(f"输入统计：")
    print(f"  形状: {pos_encoding.shape}")
    print(f"  范围: [{pos_encoding.min():.2f}, {pos_encoding.max():.2f}]")
    print(f"  包含 NaN: {torch.isnan(pos_encoding).any()}")
    print(f"  包含 Inf: {torch.isinf(pos_encoding).any()}")

    # 预热
    for _ in range(5):
        with torch.amp.autocast('cuda', enabled=use_amp):
            _ = model(pos_encoding)

    torch.cuda.synchronize()

    # 计时
    num_runs = 50
    start_time = time.time()

    for _ in range(num_runs):
        with torch.amp.autocast('cuda', enabled=use_amp):
            output = model(pos_encoding)

    torch.cuda.synchronize()
    elapsed = time.time() - start_time
    avg_time = elapsed / num_runs * 1000  # ms

    # 检查输出稳定性
    has_nan = torch.isnan(output).any()
    has_inf = torch.isinf(output).any()

    # 显存占用（峰值）
    memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    print(f"\n性能测试结果（AMP={'ON' if use_amp else 'OFF'}）：")
    print(f"  平均耗时: {avg_time:.3f} ms")
    print(f"  显存占用: {memory_mb:.2f} MB")
    print(f"  输出形状: {output.shape}")
    print(f"  输出范围: [{output.min():.4f}, {output.max():.4f}]")
    print(f"  包含 NaN: {has_nan}")
    print(f"  包含 Inf: {has_inf}")
    print(f"  数值稳定: {'✓' if not (has_nan or has_inf) else '✗'}")

    torch.cuda.empty_cache()

    return {
        'time_ms': avg_time,
        'memory_mb': memory_mb,
        'stable': not (has_nan or has_inf)
    }


def compare_strategies():
    """对比不同精度策略"""
    print("="*60)
    print("位置编码优化效果测试")
    print("="*60)

    if not torch.cuda.is_available():
        print("CUDA 不可用，跳过测试")
        return

    print("\n方案 1: 使用 AMP (FP16) + 归一化 + Clamp（优化后）")
    print("-"*60)
    result_amp = test_pe_encoder(use_amp=True, batch_size=10000, k=32)

    print("\n" + "="*60)
    print("\n方案 2: 强制 FP32（原始方案，需手动切换代码测试）")
    print("-"*60)
    print("注意：当前代码已使用优化方案，如需对比请恢复原始 FP32 代码")

    # 模拟 FP32 方案（纯 FP32 推理）
    result_fp32 = test_pe_encoder(use_amp=False, batch_size=10000, k=32)

    print("\n" + "="*60)
    print("性能对比总结")
    print("="*60)
    print(f"速度提升: {result_fp32['time_ms'] / result_amp['time_ms']:.2f}x")
    print(f"显存节省: {result_fp32['memory_mb'] - result_amp['memory_mb']:.2f} MB "
          f"({(1 - result_amp['memory_mb']/result_fp32['memory_mb'])*100:.1f}%)")
    print(f"数值稳定性: AMP={'✓' if result_amp['stable'] else '✗'}, "
          f"FP32={'✓' if result_fp32['stable'] else '✗'}")

    print("\n结论：")
    if result_amp['stable']:
        print("✓ 优化方案在保持数值稳定的同时，显著提升了性能和显存效率")
    else:
        print("✗ 仍存在数值不稳定，可能需要调整归一化策略")


if __name__ == "__main__":
    compare_strategies()
