import os
import numpy as np
import matplotlib.pyplot as plt
import laspy
from pathlib import Path
from tqdm import tqdm

def get_paper_ready_colors(labels):
    """
    为不同的标签分配适合论文发表的高级颜色
    """
    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)
    
    # 【配色方案选择】
    # 方案 A (默认): 'turbo' - 适合超点/聚类类别较多(>20)的情况。色彩丰富、感知均匀、质感高级。
    # 方案 B: 'tab20' - 适合类别较少(<20)的情况。最经典的论文分类标准柔和配色。
    # 方案 C: 'Set3' - 莫兰迪色系/马卡龙色系，非常淡雅，适合浅色背景展示。
    cmap = plt.get_cmap('tab20') 
    
    color_palette = cmap(np.linspace(0, 1, num_classes))
    
    # 打乱颜色顺序，防止相邻/连续的标签颜色过于相近
    np.random.seed(42)  
    np.random.shuffle(color_palette)
    
    # 将标签映射为 0 ~ num_classes-1 的索引 (极速向量化搜索)
    sort_idx = np.argsort(unique_labels)
    sorted_labels = unique_labels[sort_idx]
    indices = np.searchsorted(sorted_labels, labels)
    
    # LAS 格式的颜色要求是 16 位无符号整数 (0 - 65535)
    r = (color_palette[indices, 0] * 65535).astype(np.uint16)
    g = (color_palette[indices, 1] * 65535).astype(np.uint16)
    b = (color_palette[indices, 2] * 65535).astype(np.uint16)
    
    return r, g, b

def ensure_rgb_format(las_data):
    """
    确保 LAS 数据的格式支持 RGB 颜色。如果不支持则静默转换为 Format 3。
    """
    if las_data.point_format.id in [2, 3, 5, 7, 8, 9, 10, 11]:
        return las_data
    
    try:
        return laspy.convert(las_data, point_format_id=3)
    except AttributeError:
        new_header = laspy.LasHeader(point_format=3, version=las_data.header.version)
        for extra_dim in las_data.header.point_format.extra_dimension_names:
            new_header.add_extra_dim(las_data.header.point_format.dimension_by_name(extra_dim))
            
        new_las = laspy.LasData(new_header)
        for dim in las_data.point_format.dimension_names:
            if dim in new_las.point_format.dimension_names:
                setattr(new_las, dim, getattr(las_data, dim))
        return new_las

def process_single_file(filepath, output_base_dir):
    """
    处理单个 LAS/LAZ 文件，使用 tqdm 接管输出
    """
    filename = os.path.basename(filepath)
    las = laspy.read(filepath)
    
    # 动态寻找所有以 'superpoint_level_' 开头的字段
    superpoint_dims = [dim for dim in las.point_format.dimension_names if dim.startswith('superpoint_level_')]
    
    if not superpoint_dims:
        return False # 返回 False 表示跳过

    # 预先确保点云格式支持颜色写入
    las = ensure_rgb_format(las)

    # 内层进度条：展示当前文件下不同层级的处理进度
    for sp_dim in tqdm(superpoint_dims, desc=f"渲染层级 ({filename})", leave=False, colour='cyan'):
        labels = getattr(las, sp_dim)
        
        # 赋予高级感配色
        r, g, b = get_paper_ready_colors(labels)
        
        las.red = r
        las.green = g
        las.blue = b
        
        # 创建子文件夹并保存
        level_out_dir = os.path.join(output_base_dir, sp_dim)
        os.makedirs(level_out_dir, exist_ok=True)
        out_filepath = os.path.join(level_out_dir, filename)
        
        las.write(out_filepath)
        
    return True

def process_pointclouds(input_path, output_dir):
    """
    主控函数：带有简洁进度指示的批处理
    """
    if not os.path.exists(input_path):
        print(f"❌ 错误: 输入路径 '{input_path}' 不存在。")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    files_to_process = []
    
    if os.path.isfile(input_path):
        if input_path.lower().endswith(('.las', '.laz')):
            files_to_process.append(Path(input_path))
        else:
            print("❌ 错误: 指定的文件不是 .las 或 .laz 格式。")
            return
            
    elif os.path.isdir(input_path):
        search_paths = [
            Path(input_path).rglob('*.las'),
            Path(input_path).rglob('*.laz'),
            Path(input_path).rglob('*.LAS'),
            Path(input_path).rglob('*.LAZ')
        ]
        for generator in search_paths:
            files_to_process.extend(list(generator))
            
    if not files_to_process:
        print("⚠️ 未找到任何需要处理的 .las 或 .laz 文件。")
        return

    # 外层进度条：展示文件总进度
    print("\n🚀 开始处理超点聚类赋色任务...")
    success_count = 0
    with tqdm(total=len(files_to_process), desc="总体文件进度", colour='green') as pbar:
        for file in files_to_process:
            processed = process_single_file(str(file), output_dir)
            if processed:
                success_count += 1
            pbar.update(1)
            
    print(f"\n🎉 任务完成！共成功处理并赋色 {success_count} 个文件。结果已保存至: {output_dir}")

# ==========================================
# 运行示例
# ==========================================
if __name__ == "__main__":
    # 你可以在这里修改为你的实际路径
    # INPUT_PATH 可以是单个文件路径，比如 'data/cloud.laz' 
    # 也可以是文件夹路径，比如 'data/pointclouds'
    INPUT_PATH = r"E:\data\DALES\dales_las\tile\pred" 
    OUTPUT_DIR = r"E:\data\DALES\dales_las\tile\pred-visualized"
    
    process_pointclouds(INPUT_PATH, OUTPUT_DIR)