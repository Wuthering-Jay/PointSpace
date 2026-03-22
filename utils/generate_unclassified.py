import os
import argparse
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
import laspy
from tqdm import tqdm

# ==========================================
# ⚙️ 配置区域 (请根据你的数据集修改类别 ID)
# ==========================================
# ISPRS 类别参考: 1-Powerline, 2-LowVeg, 3-ImpSurf, 4-Car, 5-Fence, 6-Roof, 7-Facade, 8-Shrub, 9-Tree
# LASDU 类别参考: 1-Ground, 2-Buildings, 3-Trees, 4-LowVeg, 5-Artifacts
UNCLASSIFIED_ID = 5  # 生成的未分类标签 ID (通常使用 0 或 255)

# 需要被“极大概率遗弃”的长尾类别 ID 列表 (如：汽车、人造物、栅栏)
LONG_TAIL_CLASSES = [3, 4]  
LONG_TAIL_DROP_RATE = 0.40  # 60% 的长尾目标会被变成未分类

# 核心类别 ID 列表 (如：地面、建筑、树木)
CORE_CLASSES = [0, 1, 2]

# 几何复杂度阈值 (Z轴法向量绝对值，越小越陡峭/粗糙)
COMPLEXITY_THRESHOLD = 0.75  # 约等于坡度大于 45 度
COMPLEXITY_DROP_RATE = 0.40 # 复杂区域有 40% 的概率被遗弃

# 边界侵蚀参数
BOUNDARY_RADIUS = 2       # 搜索异类邻居的半径 (米)
BOUNDARY_DROP_RATE = 0.15   # 边界点有 60% 概率变成未分类

def process_point_cloud(points, labels):
    """
    核心逻辑：模拟测绘生产流中的算法失效与人工遗弃
    """
    new_labels = labels.copy()
    num_points = len(points)
    
    # ---------------------------------------------------------
    # 法则 C: 几何复杂度拒判 (利用 Open3D 极速计算法向量)
    # ---------------------------------------------------------
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    # 估算法向量
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=15))
    pcd.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 10000.]))
    
    normals = np.asarray(pcd.normals)
    # Z方向法向量绝对值越小，代表越陡峭或表面越粗糙
    is_complex = np.abs(normals[:, 2]) < COMPLEXITY_THRESHOLD
    
    # 仅对核心类（地面、建筑）应用复杂度拒判
    is_core = np.isin(labels, CORE_CLASSES)
    complex_drop_mask = np.random.rand(num_points) < COMPLEXITY_DROP_RATE
    
    new_labels[is_core & is_complex & complex_drop_mask] = UNCLASSIFIED_ID
    
    # ---------------------------------------------------------
    # 法则 A: 长尾目标的语义遗弃
    # ---------------------------------------------------------
    is_long_tail = np.isin(labels, LONG_TAIL_CLASSES)
    tail_drop_mask = np.random.rand(num_points) < LONG_TAIL_DROP_RATE
    new_labels[is_long_tail & tail_drop_mask] = UNCLASSIFIED_ID
    
    # ---------------------------------------------------------
    # 法则 B: 基于形态学的边界侵蚀 (使用 KDTree)
    # ---------------------------------------------------------
    # 为了加速，我们只在尚未变成未分类的点上寻找边界
    valid_mask = (new_labels != UNCLASSIFIED_ID)
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_indices) > 0:
        valid_points = points[valid_indices]
        valid_labels = new_labels[valid_indices]
        
        tree = cKDTree(valid_points)
        # 查询邻域，n_jobs=-1 使用多线程加速
        neighbors_list = tree.query_ball_point(valid_points, r=BOUNDARY_RADIUS, workers=-1)
        
        # 记录需要变成未分类的索引
        boundary_drop_indices = []
        for idx, neighbors in enumerate(neighbors_list):
            if len(neighbors) <= 1:
                continue
            # 如果邻域内存在和自己不同的标签，说明在边界上
            local_labels = valid_labels[neighbors]
            if local_labels.min() != local_labels.max(): # 极速判断是否有异类
                if np.random.rand() < BOUNDARY_DROP_RATE:
                    boundary_drop_indices.append(valid_indices[idx])
                    
        new_labels[boundary_drop_indices] = UNCLASSIFIED_ID

    return new_labels

def process_file(input_path, output_path):
    """
    处理单个 LAS/LAZ 文件
    """
    print(f"Reading: {input_path}")
    las = laspy.read(input_path)
    
    # 提取坐标和原始标签
    points = np.vstack((las.x, las.y, las.z)).transpose()
    labels = np.array(las.classification)
    
    # 核心降级处理
    print("  Applying workflow-driven degradation...")
    new_labels = process_point_cloud(points, labels)
    
    # 统计信息
    original_unclass = np.sum(labels == UNCLASSIFIED_ID)
    new_unclass = np.sum(new_labels == UNCLASSIFIED_ID)
    print(f"  Unclassified points: {original_unclass} -> {new_unclass} ({(new_unclass/len(labels))*100:.1f}%)")
    
    # 写回新标签并保存
    las.classification = new_labels
    las.write(output_path)
    print(f"Saved: {output_path}")

def process_path(input_path, output_dir=None):
    """
    处理文件或文件夹
    """
    if os.path.isfile(input_path):
        if output_dir is None:
            output_dir = os.path.dirname(input_path)
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.basename(input_path)
        name, ext = os.path.splitext(filename)
        out_file = os.path.join(output_dir, f"{name}{ext}")
        process_file(input_path, out_file)
        
    elif os.path.isdir(input_path):
        if output_dir is None:
            output_dir = input_path
        os.makedirs(output_dir, exist_ok=True)
        
        files = [f for f in os.listdir(input_path) if f.lower().endswith(('.las', '.laz'))]
        for f in tqdm(files, desc="Processing files"):
            in_file = os.path.join(input_path, f)
            out_file = os.path.join(output_dir, f"{os.path.splitext(f)[0]}{os.path.splitext(f)[1]}")
            process_file(in_file, out_file)
    else:
        print("Invalid input path.")

if __name__ == "__main__":

    input = r"E:\data\LASDU\tile\train_sparse"  # 示例输入路径
    output = r"E:\data\LASDU\tile\train_noisy"  # 示例输出

    process_path(input, output)