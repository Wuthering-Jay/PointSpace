import numpy as np
import laspy
from pathlib import Path

def filter_single_las(input_file: Path, output_file: Path, keep_classes: list):
    """
    处理单个 LAS/LAZ 文件，仅保留指定类别的点
    """
    print(f"正在处理: {input_file.name} ...")
    
    # 1. 读取点云
    las = laspy.read(input_file)
    
    # 2. 生成布尔掩码 (Mask)
    # np.isin 可以非常高效地判断数组中的元素是否在目标列表中
    mask = np.isin(las.classification, keep_classes)
    
    # 统计保留的点数
    keep_count = np.sum(mask)
    if keep_count == 0:
        print(f"  -> 警告: {input_file.name} 中没有包含类别 {keep_classes} 的点，将输出空文件！")
    
    # 3. 过滤点云
    # 直接利用布尔掩码对 las.points 进行切片，丢弃不需要的点
    las.points = las.points[mask]
    
    # 4. 【关键步骤】更新头文件
    # 因为点数减少了，且点云的地理包围盒 (Min/Max) 可能缩小了
    # 必须调用此方法重新计算头文件记录，否则生成的文件可能会报错或边界错误
    las.update_header()
    
    # 5. 写出文件
    las.write(output_file)
    print(f"  -> 已保存至: {output_file} (保留点数: {keep_count} / 原始点数: {len(mask)})")


def extract_las_by_class(input_path: str, output_path: str, keep_classes: list):
    """
    通用接口：过滤点云类别，仅保留指定列表中的点
    
    参数:
    - input_path: 输入路径 (文件或文件夹)
    - output_path: 输出路径 (文件或文件夹)
    - keep_classes: 列表形式的保留类别，如 [2, 6] 表示只保留地面和建筑
    """
    in_p = Path(input_path)
    out_p = Path(output_path)
    
    if not in_p.exists():
        raise FileNotFoundError(f"输入路径不存在: {in_p}")

    # 情况 1: 单个文件
    if in_p.is_file():
        if in_p.suffix.lower() not in ['.las', '.laz']:
            raise ValueError("输入文件必须是 .las 或 .laz 格式")
            
        if out_p.is_dir() or not out_p.suffix:
            out_p.mkdir(parents=True, exist_ok=True)
            out_file = out_p / in_p.name
        else:
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_file = out_p
            
        filter_single_las(in_p, out_file, keep_classes)

    # 情况 2: 文件夹批量处理
    elif in_p.is_dir():
        out_p.mkdir(parents=True, exist_ok=True)
        for file_path in in_p.rglob("*"):
            if file_path.suffix.lower() in ['.las', '.laz']:
                rel_path = file_path.relative_to(in_p)
                out_file = out_p / rel_path
                out_file.parent.mkdir(parents=True, exist_ok=True)
                
                filter_single_las(file_path, out_file, keep_classes)
    
    print("\n所有过滤提取任务已完成！")

# ==========================================
# 使用示例
# ==========================================
if __name__ == "__main__":
    
    INPUT_SRC = r"E:\data\云南遥感中心\第二批\ground\disk03\train"      # 替换为实际输入路径
    OUTPUT_DST = r"E:\data\云南遥感中心\第二批\ground-only\disk03\train"    # 替换为实际输出路径
    
    # 定义你要保留的类别列表。
    # 例如：通常 2 代表地面 (Ground)，6 代表建筑物 (Building)
    # classes_to_keep = [2, 6]
    classes_to_keep = [2]  # 仅保留地面类别

    
    try:
        extract_las_by_class(
            input_path=INPUT_SRC, 
            output_path=OUTPUT_DST, 
            keep_classes=classes_to_keep
        )
    except Exception as e:
        print(f"运行出错: {e}")