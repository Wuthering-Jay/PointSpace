import os
import glob
import time
import laspy
import torch
import numpy as np
import open3d as o3d
import matplotlib.colors as mcolors
from sklearn.neighbors import NearestNeighbors
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

# 导入你刚刚编译好的底层库
from pointseg.functions import segment_point

def generate_distinct_colors(num_colors):
    """
    生成鲜艳且区分度高的随机颜色 (16-bit, 适用于 LAS 文件)
    利用 HSV 色彩空间，固定高饱和度和高亮度，随机色相。
    """
    h = np.random.rand(num_colors)
    s = np.random.uniform(0.8, 1.0, num_colors) # 高饱和度
    v = np.random.uniform(0.8, 1.0, num_colors) # 高亮度
    hsv = np.column_stack((h, s, v))
    
    rgb_float = mcolors.hsv_to_rgb(hsv) # 转换到 0.0-1.0 的 RGB
    rgb_16bit = (rgb_float * 65535).astype(np.uint16) # 映射到 LAS 标准的 16-bit
    return rgb_16bit

def process_single_las(input_file, output_file, kThresh=0.02, segMinVerts=20, segMaxVerts=8192, edge_k=10, normal_radius=1.0, z_sensitivity=1.0, verbose=True):
    """
    处理单个 LAS 文件的超点分割

    Returns:
        dict: 包含处理结果的字典 {filename, num_points, num_superpoints, max_sp_size, time}
    """
    basename = os.path.basename(input_file)
    if verbose:
        print(f"[{time.strftime('%H:%M:%S')}] 开始处理: {basename}")
    t0 = time.time()

    # ==========================
    # 1. 读取点云数据
    # ==========================
    las = laspy.read(input_file)
    # 获取真实的物理坐标 (已应用 scale 和 offset)
    xyz = np.vstack((las.x, las.y, las.z)).transpose()
    N = xyz.shape[0]
    if verbose:
        print(f"  -> 成功读取 {N} 个点.")

    # ==========================
    # 1.5 应用 Z 轴敏感度调整
    # ==========================
    if z_sensitivity != 1.0:
        if verbose:
            print(f"  -> 应用 Z 轴敏感度系数: {z_sensitivity}")
        xyz_scaled = xyz.copy()
        xyz_scaled[:, 2] *= z_sensitivity
    else:
        xyz_scaled = xyz

    # ==========================
    # 2. 计算几何特征 (法向量) - 使用缩放后的坐标
    # ==========================
    if verbose:
        print("  -> 正在估算法向量...")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_scaled)
    # 注意：normal_radius 需根据你数据的真实物理尺度调整（无人机测绘通常设为 0.5m - 2.0m）
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30))
    pcd.orient_normals_towards_camera_location([0., 0., 0.])
    normals = np.asarray(pcd.normals)

    # ==========================
    # 3. 构建拓扑图 (Edges) - 使用缩放后的坐标
    # ==========================
    if verbose:
        print("  -> 正在构建 KNN 拓扑图...")
    k_neighbors = edge_k
    # n_jobs=1 避免并行时嵌套多线程导致资源竞争
    nbrs = NearestNeighbors(n_neighbors=k_neighbors+1, algorithm='kd_tree', n_jobs=1).fit(xyz_scaled)
    _, indices = nbrs.kneighbors(xyz_scaled)
    
    source_nodes = np.repeat(np.arange(N), k_neighbors)
    target_nodes = indices[:, 1:].flatten()
    edges = np.vstack((source_nodes, target_nodes)).T

    # ==========================
    # 4. 执行极速超点分割
    # ==========================
    if verbose:
        print(f"  -> 正在执行 C++ Superpoint 分割 (kThresh={kThresh}, segMaxVerts={segMaxVerts})...")
    vertices_tensor = torch.tensor(xyz_scaled, dtype=torch.float32)
    normals_tensor = torch.tensor(normals, dtype=torch.float32)
    edges_tensor = torch.tensor(edges, dtype=torch.int64)

    sp_idx_tensor = segment_point(
        vertices=vertices_tensor,
        normals=normals_tensor,
        edges=edges_tensor,
        kThresh=kThresh,
        segMinVerts=segMinVerts,
        segMaxVerts=segMaxVerts
    )
    sp_idx = sp_idx_tensor.numpy()

    num_sp = len(np.unique(sp_idx))
    sp_sizes = [np.sum(sp_idx == i) for i in np.unique(sp_idx)]
    max_sp_size = max(sp_sizes)
    if verbose:
        print(f"  -> 分割完成! 共生成 {num_sp} 个超点模块.")
        print(f"  -> 超点大小统计: 最小={min(sp_sizes)}, 最大={max_sp_size}, 平均={np.mean(sp_sizes):.1f}")

    # ==========================
    # 5. 可视化上色与属性写入
    # ==========================
    if verbose:
        print("  -> 正在上色并写入新字段...")

    # 检查原始 LAS 是否支持 RGB，如果不支持，强制转换为支持 RGB 的点格式
    fmt = las.header.point_format.id
    if fmt in [0, 1]:  # LAS 1.2 无颜色格式
        las = laspy.convert(las, point_format_id=3)
    elif fmt == 6:     # LAS 1.4 无颜色格式
        las = laspy.convert(las, point_format_id=7)

    # 生成颜色并赋值
    distinct_colors = generate_distinct_colors(num_sp)
    point_colors = distinct_colors[sp_idx]

    las.red = point_colors[:, 0]
    las.green = point_colors[:, 1]
    las.blue = point_colors[:, 2]

    # 写入真实的超点 ID 到额外字段 (Extra Bytes)
    try:
        # 添加名为 "superpoint_id" 的 int32 自定义维度
        las.add_extra_dim(laspy.ExtraBytesParams(name="superpoint_id", type=np.int32))
    except ValueError:
        pass # 如果该字段已存在则忽略

    las.superpoint_id = sp_idx

    # 保存文件
    las.write(output_file)
    elapsed = time.time() - t0
    if verbose:
        print(f"[{time.strftime('%H:%M:%S')}] 处理完毕! 耗时: {elapsed:.2f}秒. 结果保存至: {output_file}\n")

    return {
        'filename': basename,
        'num_points': N,
        'num_superpoints': num_sp,
        'max_sp_size': max_sp_size,
        'time': elapsed
    }


def _process_worker(args):
    """多进程 worker 函数"""
    input_file, output_file, params = args
    try:
        result = process_single_las(
            input_file, output_file,
            kThresh=params['kThresh'],
            segMinVerts=params['segMinVerts'],
            segMaxVerts=params['segMaxVerts'],
            edge_k=params['edge_k'],
            normal_radius=params['normal_radius'],
            z_sensitivity=params['z_sensitivity'],
            verbose=False  # 并行时关闭详细输出
        )
        return {'status': 'success', **result}
    except Exception as e:
        return {'status': 'error', 'filename': os.path.basename(input_file), 'error': str(e)}


def batch_validate_superpoints(input_path, output_dir, kThresh=0.02, segMinVerts=20, segMaxVerts=8192,
                               edge_k=10, normal_radius=1.0, z_sensitivity=1.0, num_workers=1):
    """
    支持单文件或文件夹批处理，支持多进程并行

    Args:
        input_path: 输入文件或文件夹路径
        output_dir: 输出文件夹
        kThresh: 分割阈值，越大块越大
        segMinVerts: 最小超点点数
        segMaxVerts: 最大超点点数（-1表示不限制）
        edge_k: KNN边数
        normal_radius: 法向量搜索半径
        z_sensitivity: Z轴敏感度系数（>1 增强Z轴差异，<1 弱化Z轴差异）
        num_workers: 并行进程数（1=串行，>1=并行）
    """
    os.makedirs(output_dir, exist_ok=True)

    if os.path.isfile(input_path):
        files = [input_path]
    elif os.path.isdir(input_path):
        files = glob.glob(os.path.join(input_path, "*.las")) + glob.glob(os.path.join(input_path, "*.laz"))
    else:
        raise ValueError("输入路径无效！")

    total_files = len(files)
    print(f"共发现 {total_files} 个文件待处理.")

    # 串行模式
    if num_workers <= 1:
        print("使用串行模式处理...")
        total_time = 0
        for f in files:
            out_f = os.path.join(output_dir, "sp_" + os.path.basename(f))
            result = process_single_las(f, out_f, kThresh, segMinVerts, segMaxVerts, edge_k, normal_radius, z_sensitivity, verbose=True)
            total_time += result['time']
        print(f"\n全部完成! 总耗时: {total_time:.2f}秒")
        return

    # 并行模式
    print(f"使用 {num_workers} 进程并行处理...")
    t_start = time.time()

    params = {
        'kThresh': kThresh,
        'segMinVerts': segMinVerts,
        'segMaxVerts': segMaxVerts,
        'edge_k': edge_k,
        'normal_radius': normal_radius,
        'z_sensitivity': z_sensitivity
    }

    # 构建任务列表
    tasks = []
    for f in files:
        out_f = os.path.join(output_dir, "sp_" + os.path.basename(f))
        tasks.append((f, out_f, params))

    # 并行执行
    completed = 0
    errors = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_process_worker, task): task for task in tasks}
        for future in as_completed(futures):
            result = future.result()
            completed += 1
            if result['status'] == 'success':
                print(f"[{completed}/{total_files}] {result['filename']}: "
                      f"{result['num_points']} 点 -> {result['num_superpoints']} 超点, "
                      f"耗时 {result['time']:.2f}s")
            else:
                errors.append(result)
                print(f"[{completed}/{total_files}] {result['filename']}: 错误 - {result['error']}")

    total_time = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"全部完成! 处理 {total_files} 个文件, 总耗时: {total_time:.2f}秒")
    print(f"平均每个文件: {total_time/total_files:.2f}秒")
    if errors:
        print(f"失败文件数: {len(errors)}")
    print(f"{'='*60}")

if __name__ == "__main__":
    # ==============================
    # 在这里修改你的测试参数
    # ==============================
    INPUT_PATH = r"E:\data\云南遥感中心\第二批\disk03\processed_其他073"       # 可以是单个文件 "./data/test.las" 或文件夹 "./data"
    OUTPUT_DIR = r"E:\data\云南遥感中心\第二批\disk03\processed_其他073"  # 输出文件夹

    # 核心调参指南：
    # kThresh: 控制分割粒度。对于测绘数据，0.01~0.05 之间微调。越大块越大，越小越碎。
    # segMinVerts: 强行合并小于此点数的孤立碎块。
    # segMaxVerts: 最大超点点数，防止超点过大（-1表示不限制）。
    # normal_radius: 法向量搜索半径，建议设置为点云平均间距的 3-5 倍（比如 1.0 米）。
    # z_sensitivity: Z轴敏感度系数。
    #   - >1: 增强高度差异的影响，容易在不同高度层分割（适合多层建筑）
    #   - <1: 弱化高度差异，更关注平面几何（适合地形起伏较大的场景）
    #   - =1: 默认，xyz 三轴同等权重
    # num_workers: 并行进程数
    #   - 1: 串行处理（默认，适合调试）
    #   - >1: 并行处理，建议设为 CPU 核心数的 1/2 到 2/3（如 8 核设 4-6）
    #   - 注意：每个进程会占用一定内存，进程数过多可能导致内存不足

    batch_validate_superpoints(
        input_path=INPUT_PATH,
        output_dir=OUTPUT_DIR,
        kThresh=0.005,
        segMinVerts=5,
        segMaxVerts=2048,
        edge_k=10,
        normal_radius=2.0,
        z_sensitivity=2.0,
        num_workers=8  # 并行进程数，设为 1 则串行
    )