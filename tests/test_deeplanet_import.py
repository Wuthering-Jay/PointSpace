"""
DeepLANet V1 简单导入测试
验证模块可以正确导入
"""

import sys
sys.path.insert(0, "e:/code/python/PointSpace")

print("Testing imports...")

try:
    from pointspace.models.backbone.deeplanet.deeplanet_v1 import (
        PointBatchNorm,
        compute_stage_positional_encoding,
        LocalSpatialEncoding,
        AttentivePooling,
        LocalFeatureAggregation,
        Block,
        BlockSequence,
        GridPool,
        UnpoolWithSkip,
        Encoder,
        Decoder,
        LFAPatchEmbed,
        DeepLANetV1,
    )
    print("✓ All modules imported successfully!")

    # 验证关键函数存在
    assert callable(compute_stage_positional_encoding)
    print("✓ compute_stage_positional_encoding function exists")

    # 验证模型可以实例化
    model = DeepLANetV1(in_channels=6)
    print("✓ DeepLANetV1 model instantiated successfully")
    print(f"  - Model has {sum(p.numel() for p in model.parameters())} parameters")

    print("\nAll imports and basic checks passed!")

except Exception as e:
    print(f"✗ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
