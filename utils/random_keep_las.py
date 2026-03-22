import os
import argparse
import numpy as np
import laspy
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_file(input_file, output_file, keep_ratio, convert_label):
    logging.info(f"Processing: {input_file}")
    
    las = laspy.read(input_file)
    classes = las.classification
    
    unique_classes = np.unique(classes)
    # 默认将所有标签初始化为 convert_label
    new_classes = np.full_like(classes, convert_label)
    
    for c in unique_classes:
        if c == convert_label:
            # 如果原始类别就是转换目标类别，保留它们（已在 new_classes 中初始化）
            continue
            
        # 找到属于当前类别的所有点的索引
        class_indices = np.where(classes == c)[0]
        num_keep = int(len(class_indices) * keep_ratio)
        
        if num_keep > 0:
            # 随机选择保留的索引
            keep_indices = np.random.choice(class_indices, num_keep, replace=False)
            new_classes[keep_indices] = c
            
    # 更新标签
    las.classification = new_classes
    las.write(output_file)
    logging.info(f"Saved to: {output_file}")


def main():
    
    input_path = Path(r"E:\data\LASDU\train")
    output_path = Path(r"E:\data\LASDU\train_sparse")
    keep_ratio = 0.01
    convert_label = 5
    
    if input_path.is_file():
        if output_path.is_dir():
            output_file = output_path / input_path.name
        else:
            output_file = output_path
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
        process_file(input_path, output_file, keep_ratio, convert_label)
        
    elif input_path.is_dir():
        output_path.mkdir(parents=True, exist_ok=True)
        files = list(input_path.glob("*.las")) + list(input_path.glob("*.laz"))
        
        for file in files:
            output_file = output_path / file.name
            process_file(file, output_file, keep_ratio, convert_label)
    else:
        logging.error("Input path does not exist")

if __name__ == "__main__":
    main()
