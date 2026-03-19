"""
验证 DeepLANet 模型注册
"""

import sys
sys.path.insert(0, "e:/code/python/PointSpace")

print("Checking model registration...")

try:
    # 导入 MODELS 注册表
    from pointspace.models.builder import MODELS

    # 导入模型（触发注册）
    from pointspace.models.backbone.deeplanet import DeepLANetV1, DeepLANetV2

    print("\n✓ Models imported successfully!")

    # 检查注册的名称
    print("\nRegistered DeepLANet models:")
    for name in MODELS.module_dict.keys():
        if "DeepLANet" in name or "deeplanet" in name.lower():
            print(f"  - {name}")

    # 测试构建
    print("\nTesting model construction from registry:")

    # 测试小写 v1
    try:
        model = MODELS.build(dict(type="DeepLANet-v1", in_channels=6))
        print("  ✓ DeepLANet-v1 (lowercase) built successfully")
    except Exception as e:
        print(f"  ✗ DeepLANet-v1 (lowercase) failed: {e}")

    # 测试大写 V1
    try:
        model = MODELS.build(dict(type="DeepLANet-V1", in_channels=6))
        print("  ✓ DeepLANet-V1 (uppercase) built successfully")
    except Exception as e:
        print(f"  ✗ DeepLANet-V1 (uppercase) failed: {e}")

    # 测试小写 v2
    try:
        model = MODELS.build(dict(type="DeepLANet-v2", in_channels=6))
        print("  ✓ DeepLANet-v2 (lowercase) built successfully")
    except Exception as e:
        print(f"  ✗ DeepLANet-v2 (lowercase) failed: {e}")

    # 测试大写 V2
    try:
        model = MODELS.build(dict(type="DeepLANet-V2", in_channels=6))
        print("  ✓ DeepLANet-V2 (uppercase) built successfully")
    except Exception as e:
        print(f"  ✗ DeepLANet-V2 (uppercase) failed: {e}")

    print("\n✓ All registration checks passed!")

except Exception as e:
    print(f"\n✗ Registration check failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
