import whitebox
import os

wbt = whitebox.WhiteboxTools()

# 1. 定义文件路径
input_las = r"E:\data\云南遥感中心\第二批\disk01\val\processed_城镇12.las"
clean_las = input_las.replace(".las", "_denoised.las")      # 去噪后的中间文件
ground_las = input_las.replace(".las", "_ground_final.las") # 最终的地面点文件

wbt.set_working_dir(os.path.dirname(input_las))

print("第一步：正在执行孤立噪声点剔除 (Lidar Remove Outliers)...")

# 2. 执行去噪算法 (参数严格按位置传递)
wbt.lidar_remove_outliers(
    input_las,            # 输入文件
    clean_las,            # 去噪后的输出文件
    radius=2.0,           # 局部搜索半径。可以和地面滤波的 radius 保持一致或略小
    elev_diff=2.0,        # [关键] 高度差阈值（米）。如果某点低于/高于周围2米内的局部基准面达2米，即判定为噪声
    use_median=True       # [推荐开启] True 表示使用局部“中位数”作为基准，对极端极低地下点具有极强的鲁棒性
)

if os.path.exists(clean_las):
    print(f"去噪完成！已生成中间干净点云: {clean_las}")
    print("\n第二步：基于干净的点云执行地面滤波 (Lidar Ground Point Filter)...")
    
    # 3. 基于“去噪后”的数据提取地面
    wbt.lidar_ground_point_filter(
        clean_las,                # 注意：这里传入的是去噪后的 clean_las，而不是最初的 input_las
        ground_las,               
        radius=5,               # 提取地面的搜索半径 (可稍大)
        min_neighbours=0,         
        slope_threshold=15.0,     # 根据地形走势设定的坡度阈值
        height_threshold=0.25,     # 控制贴地精度的严苛高度差
        classify=True,           
        slope_norm=True,          
        height_above_ground=False
    )
    
    print(f"\n全部处理完成！最终地面点云已保存至: {ground_las}")
else:
    print("去噪步骤失败，请检查输入数据或控制台输出。")

# 可选：如果你不需要保留去噪的中间文件，可以在末尾用 os.remove(clean_las) 删除它