"""
测试 segMaxVerts 参数功能
"""
import torch
import numpy as np
from pointseg.functions import segment_point

def test_segmax_verts():
    print("=" * 60)
    print("测试 segMaxVerts 参数")
    print("=" * 60)

    # 创建一个简单的平面点云 (10x10 grid = 100 points)
    np.random.seed(42)
    n_points = 100
    x = np.linspace(0, 1, 10)
    y = np.linspace(0, 1, 10)
    xx, yy = np.meshgrid(x, y)
    z = np.zeros_like(xx) + np.random.randn(*xx.shape) * 0.01  # 轻微噪声

    vertices = np.column_stack([xx.ravel(), yy.ravel(), z.ravel()]).astype(np.float32)

    # 所有点的法向量都指向 z 轴（平面）
    normals = np.zeros((n_points, 3), dtype=np.float32)
    normals[:, 2] = 1.0

    # 构建 KNN 边 (k=4)
    from sklearn.neighbors import NearestNeighbors
    k = 4
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(vertices)
    _, indices = nbrs.kneighbors(vertices)

    source = np.repeat(np.arange(n_points), k)
    target = indices[:, 1:].flatten()
    edges = np.column_stack([source, target]).astype(np.int64)

    # 转换为 tensor
    vertices_t = torch.tensor(vertices, dtype=torch.float32)
    normals_t = torch.tensor(normals, dtype=torch.float32)
    edges_t = torch.tensor(edges, dtype=torch.int64)

    # 测试 1: 无 segMaxVerts 限制
    print("\n[测试1] 无最大点数限制 (segMaxVerts=-1)")
    idx_no_limit = segment_point(vertices_t, normals_t, edges_t, kThresh=0.5, segMinVerts=5, segMaxVerts=-1)
    unique_labels_no_limit = torch.unique(idx_no_limit)
    sizes_no_limit = [(idx_no_limit == l).sum().item() for l in unique_labels_no_limit]
    print(f"  超点数量: {len(unique_labels_no_limit)}")
    print(f"  各超点大小: {sorted(sizes_no_limit, reverse=True)}")
    print(f"  最大超点: {max(sizes_no_limit)} 点")

    # 测试 2: 设置 segMaxVerts = 30
    print("\n[测试2] 设置 segMaxVerts=30")
    idx_max30 = segment_point(vertices_t, normals_t, edges_t, kThresh=0.5, segMinVerts=5, segMaxVerts=30)
    unique_labels_30 = torch.unique(idx_max30)
    sizes_30 = [(idx_max30 == l).sum().item() for l in unique_labels_30]
    print(f"  超点数量: {len(unique_labels_30)}")
    print(f"  各超点大小: {sorted(sizes_30, reverse=True)}")
    print(f"  最大超点: {max(sizes_30)} 点")

    # 测试 3: 设置 segMaxVerts = 15
    print("\n[测试3] 设置 segMaxVerts=15")
    idx_max15 = segment_point(vertices_t, normals_t, edges_t, kThresh=0.5, segMinVerts=5, segMaxVerts=15)
    unique_labels_15 = torch.unique(idx_max15)
    sizes_15 = [(idx_max15 == l).sum().item() for l in unique_labels_15]
    print(f"  超点数量: {len(unique_labels_15)}")
    print(f"  各超点大小: {sorted(sizes_15, reverse=True)}")
    print(f"  最大超点: {max(sizes_15)} 点")

    # 验证结果
    print("\n" + "=" * 60)
    print("验证结果")
    print("=" * 60)

    success = True
    if max(sizes_30) > 30:
        print(f"[FAIL] segMaxVerts=30 时最大超点 ({max(sizes_30)}) 超过限制")
        success = False
    else:
        print(f"[PASS] segMaxVerts=30 时最大超点 ({max(sizes_30)}) <= 30")

    if max(sizes_15) > 15:
        print(f"[FAIL] segMaxVerts=15 时最大超点 ({max(sizes_15)}) 超过限制")
        success = False
    else:
        print(f"[PASS] segMaxVerts=15 时最大超点 ({max(sizes_15)}) <= 15")

    # 验证限制越严格，超点数量越多
    if len(unique_labels_15) >= len(unique_labels_30) >= len(unique_labels_no_limit):
        print(f"[PASS] 限制越严格超点数量越多: {len(unique_labels_no_limit)} <= {len(unique_labels_30)} <= {len(unique_labels_15)}")
    else:
        print(f"[INFO] 超点数量变化: no_limit={len(unique_labels_no_limit)}, max30={len(unique_labels_30)}, max15={len(unique_labels_15)}")

    print("\n" + ("测试通过!" if success else "测试失败!"))
    return success

if __name__ == "__main__":
    test_segmax_verts()
