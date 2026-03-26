import os
import numpy as np
import argparse
import glob

def write_ply_with_scalars(filename, coords, scalars_dict):
    """将坐标和多个标量场写入 PLY 文件，方便 CloudCompare 读取"""
    num_points = coords.shape[0]
    
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        
        # 动态写入标量场头部
        for name in scalars_dict.keys():
            f.write(f"property float {name}\n")
            
        f.write("end_header\n")
        
        # 组装数据并保存
        scalars_list = [scalars_dict[k].reshape(-1, 1) for k in scalars_dict.keys()]
        data = np.hstack([coords] + scalars_list)
        np.savetxt(f, data, fmt='%.6f ' * 3 + '%.4f ' * len(scalars_list))

def process_single_file(npz_file):
    """处理单个 npz 文件"""
    try:
        data = np.load(npz_file)
        out_file = npz_file.replace(".npz", ".ply")
        
        write_ply_with_scalars(out_file, data['coord'], {
            "target_noisy": data['target'],
            "network_pred": data['pred'],
            "robust_weight": data['weight'],
            "kl_divergence": data['kl'],
            "pseudo_labels": data['pseudo']
        })
        return True
    except Exception as e:
        print(f"❌ Error processing {npz_file}: {e}")
        return False

if __name__ == "__main__":


    path = r"debug_dumps"

    # 场景 1: 传入的是单个文件
    if os.path.isfile(path) and path.endswith('.npz'):
        print(f"⚙️ Processing single file: {path}")
        if process_single_file(path):
            print(f"✅ Successfully converted to PLY!")

    # 场景 2: 传入的是整个文件夹
    elif os.path.isdir(path):
        search_pattern = os.path.join(path, "*.npz")
        npz_files = glob.glob(search_pattern)
        
        if not npz_files:
            print(f"⚠️ No .npz files found in directory: {path}")
        else:
            print(f"🚀 Found {len(npz_files)} .npz files in {path}. Starting batch conversion...")
            
            # 尝试导入 tqdm 显示进度条，如果没有就用普通打印
            try:
                from tqdm import tqdm
                iterator = tqdm(npz_files, desc="Converting")
            except ImportError:
                iterator = npz_files
                
            success_count = 0
            for f in iterator:
                if process_single_file(f):
                    success_count += 1
                if not isinstance(iterator, list): # 如果不是 tqdm
                    print(f"  -> Converted {os.path.basename(f)}")
                    
            print(f"\n🎉 Batch conversion complete! Successfully converted {success_count}/{len(npz_files)} files.")
            print(f"📂 You can now drag the .ply files from {path} into CloudCompare.")

    else:
        print("❌ Invalid path provided. Please provide a valid .npz file or directory.")