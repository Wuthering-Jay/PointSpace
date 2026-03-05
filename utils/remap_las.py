import os
import numpy as np
import laspy
from pathlib import Path

def process_single_las(input_file: Path, output_file: Path, mapping_rule: dict, default_class: int = None):
    """
    处理单个 LAS/LAZ 文件，进行类别映射
    """
    print(f"正在处理: {input_file.name} ...")
    
    # 读取点云文件（laspy 会保留所有的头文件、点格式和 VLR 等信息）
    las = laspy.read(input_file)
    orig_classes = las.classification
    
    # =========================================================
    # 核心逻辑：根据 default_class 初始化查找表 (LUT)
    # =========================================================
    if default_class is not None:
        # 情况A：存在 default_class，未指定的类别全都映射到 default_class
        lut = np.full(256, default_class, dtype=np.uint8)
    else:
        # 情况B：default_class 为 None，未指定的类别保持原样 (0->0, 1->1...)
        lut = np.arange(256, dtype=np.uint8)
    
    # 根据 mapping_rule 覆盖查找表中的特定值
    for old_class, new_class in mapping_rule.items():
        if 0 <= old_class <= 255:
            lut[old_class] = new_class
        else:
            print(f"警告: 类别值 {old_class} 超出 0-255 范围，已跳过。")
            
    # 【瞬间映射】通过数组索引一次性完成千万级点的类别替换
    new_classes = lut[orig_classes]
    
    # 将修改后的类别数组赋值回 las 对象
    las.classification = new_classes
    
    # 写出到目标路径（保持原有所有结构不变）
    las.write(output_file)
    print(f"已保存至: {output_file}")


def remap_las_classification(input_path: str, output_path: str, mapping_rule: dict, default_class: int = None):
    """
    通用接口：根据映射规则修改点云类别，支持单文件或文件夹批量处理
    
    参数:
    - input_path: 输入路径 (文件或文件夹)
    - output_path: 输出路径 (文件或文件夹)
    - mapping_rule: 字典形式的映射规则，如 {4: 3, 5: 3}
    - default_class: 如果为 None，其他类别保持原样；如果指定了数字(如 0)，其他类别全映射为该数字。
    """
    in_p = Path(input_path)
    out_p = Path(output_path)
    
    if not in_p.exists():
        raise FileNotFoundError(f"输入路径不存在: {in_p}")

    # 情况 1: 处理单个文件
    if in_p.is_file():
        if in_p.suffix.lower() not in ['.las', '.laz']:
            raise ValueError("输入文件必须是 .las 或 .laz 格式")
            
        if out_p.is_dir() or not out_p.suffix:
            out_p.mkdir(parents=True, exist_ok=True)
            out_file = out_p / in_p.name
        else:
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_file = out_p
            
        process_single_las(in_p, out_file, mapping_rule, default_class)

    # 情况 2: 处理整个文件夹
    elif in_p.is_dir():
        out_p.mkdir(parents=True, exist_ok=True)
        # rglob("*") 会递归遍历子文件夹，并保持原有目录层级结构
        for file_path in in_p.rglob("*"):
            if file_path.suffix.lower() in ['.las', '.laz']:
                rel_path = file_path.relative_to(in_p)
                out_file = out_p / rel_path
                out_file.parent.mkdir(parents=True, exist_ok=True)
                
                process_single_las(file_path, out_file, mapping_rule, default_class)
    
    print("\n所有处理已完成！")

# ==========================================
# 使用示例
# ==========================================
if __name__ == "__main__":
    
    INPUT_SRC = r"E:\data\云南遥感中心\第二批\disk03\train"      # 替换为实际的输入文件或文件夹
    OUTPUT_DST = r"E:\data\云南遥感中心\第二批\ground\disk03\train"    # 替换为实际的输出文件或文件夹
    
    # 规则字典：将 2 映射为 2（可省略，写上更明确），4 和 5 映射为 3
    my_mapping = {
        2: 2,
        # 4: 3,
        # 5: 3
    }

    # ==========================
    # 场景一：其他类别映射到 0
    # ==========================
    print(">>> 执行场景一: 其他类别映射到 0")
    remap_las_classification(
        input_path=INPUT_SRC, 
        output_path=OUTPUT_DST, 
        mapping_rule=my_mapping, 
        default_class=1          # 这里传入 0
    )

    # ==========================
    # 场景二：其他类别保持原样
    # ==========================
    # print("\n>>> 执行场景二: 其他类别保持原样")
    # remap_las_classification(
    #     input_path=INPUT_SRC, 
    #     output_path=OUTPUT_DST + "_keep", 
    #     mapping_rule=my_mapping, 
    #     default_class=None       # 这里传入 None
    # )