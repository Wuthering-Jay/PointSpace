"""
测试 z_sensitivity 和 segMaxVerts 参数效果
"""
import torch
import numpy as np
from pointseg.functions import segment_point
from sklearn.neighbors import NearestNeighbors
import open3d as o3d

def test_z_sensitivity_effect():
    print("=" * 60)
    print("测试 Z 轴敏感度参数效果")
    print("=" * 60)

    # 创建一个两层的平面点云 (下层 50 点 + 上层 50 点)
    np.random.seed(42)
    n_per_layer = 50

    # 下层: z=0
    x1 = np.random.rand(n_per_layer) * 10
    y1 = np.random.rand(n_per_layer) * 10
    z1 = np.zeros(n_per_layer) + np.random.randn(n_per_layer) * 0.01
    layer1 = np.column_stack([x1, y1, z1])

    # 上层: z=1 (高度差 1m)
    x2 = np.random.rand(n_per_layer) * 10
    y2 = np.random.rand(n_per_layer) * 10
    z2 = np.ones(n_per_layer) + np.random.randn(n_per_layer) * 0.01
    layer2 = np.column_stack([x2, y2, z2])

    vertices_original = np.vstack([layer1, layer2]).astype(np.float32)
    n_points = vertices_original.shape[0]

    def segment_with_z_scale(z_scale):
        """使用指定的 z 缩放系数进行分割"""
        vertices = vertices_original.copy()
        vertices[:, 2] *= z_scale

        # 计算法向量
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vertices)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2.0, max_nn=30))
        normals = np.asarray(pcd.normals).astype(np.float32)

        # 构建边
        k = 6
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(vertices)
        _, indices = nbrs.kneighbors(vertices)
        source = np.repeat(np.arange(n_points), k)
        target = indices[:, 1:].flatten()
        edges = np.column_stack([source, target]).astype(np.int64)

        # 分割
        vertices_t = torch.tensor(vertices, dtype=torch.float32)
        normals_t = torch.tensor(normals, dtype=torch.float32)
        edges_t = torch.tensor(edges, dtype=torch.int64)

        idx = segment_point(vertices_t, normals_t, edges_t,
                           kThresh=0.3, segMinVerts=10, segMaxVerts=60)
        return idx.numpy()

    # 测试不同的 z_sensitivity
    print("\n[测试1] z_sensitivity = 1.0 (默认)")
    idx_1 = segment_with_z_scale(1.0)
    n_sp_1 = len(np.unique(idx_1))
    sizes_1 = [(idx_1 == i).sum() for i in np.unique(idx_1)]
    print(f"  超点数量: {n_sp_1}")
    print(f"  超点大小: {sorted(sizes_1, reverse=True)}")

    print("\n[测试2] z_sensitivity = 3.0 (增强高度差异)")
    idx_3 = segment_with_z_scale(3.0)
    n_sp_3 = len(np.unique(idx_3))
    sizes_3 = [(idx_3 == i).sum() for i in np.unique(idx_3)]
    print(f"  超点数量: {n_sp_3}")
    print(f"  超点大小: {sorted(sizes_3, reverse=True)}")

    print("\n[测试3] z_sensitivity = 0.3 (弱化高度差异)")
    idx_03 = segment_with_z_scale(0.3)
    n_sp_03 = len(np.unique(idx_03))
    sizes_03 = [(idx_03 == i).sum() for i in np.unique(idx_03)]
    print(f"  超点数量: {n_sp_03}")
    print(f"  超点大小: {sorted(sizes_03, reverse=True)}")

    print("\n" + "=" * 60)
    print("验证 segMaxVerts=60 约束")
    print("=" * 60)
    print(f"测试1 最大超点: {max(sizes_1)} {'✓' if max(sizes_1) <= 60 else '✗'}")
    print(f"测试2 最大超点: {max(sizes_3)} {'✓' if max(sizes_3) <= 60 else '✗'}")
    print(f"测试3 最大超点: {max(sizes_03)} {'✓' if max(sizes_03) <= 60 else '✗'}")

    print("\n" + "=" * 60)
    print("结论:")
    print(f"  z_sensitivity=3.0 时更容易分层: {n_sp_3} 个超点")
    print(f"  z_sensitivity=1.0 时默认行为: {n_sp_1} 个超点")
    print(f"  z_sensitivity=0.3 时更易合并: {n_sp_03} 个超点")
    print("=" * 60)

if __name__ == "__main__":
    test_z_sensitivity_effect()
