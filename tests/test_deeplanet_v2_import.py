"""
DeepLANet V2 快速导入测试
验证模块可以正确导入
"""

import sys
sys.path.insert(0, "e:/code/python/PointSpace")

print("Testing DeepLANet V2 imports...")

try:
    from pointspace.models.backbone.deeplanet.deeplanet_v2 import (
        PointBatchNorm,
        compute_stage_positional_encoding,
        PositionalEncodingEncoder,
        VFRModule,
        ResLFEBlock,
        BlockSequence,
        GridPool,
        UnpoolWithSkip,
        Encoder,
        Decoder,
        VFRPatchEmbed,
        DeepLANetV2,
    )
    print("✓ All V2 modules imported successfully!")

    # 验证关键函数存在
    assert callable(compute_stage_positional_encoding)
    print("✓ compute_stage_positional_encoding function exists")

    # 验证 VFR 模块可以实例化
    vfr = VFRModule()
    print("✓ VFRModule instantiated successfully")

    # 验证模型可以实例化
    model = DeepLANetV2(in_channels=6)
    print("✓ DeepLANetV2 model instantiated successfully")
    print(f"  - Model has {sum(p.numel() for p in model.parameters()):,} parameters")

    print("\nAll imports and basic checks passed!")
    print("DeepLANet V2 is ready to use! 🎉")

except Exception as e:
    print(f"✗ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
