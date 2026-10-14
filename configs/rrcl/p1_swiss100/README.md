# CVRL 4.4-P1：Swiss-100 同构预训练对照

五组均使用Swiss-100、XYZ+intensity、LitePT-v1m4完整编码器/解码器、72D最终点特征、25 epoch、batch 16、梯度累积4、相同增强/采样和优化器。LAS语义`classification`仅为审计标签；任何目标都不进入`feat`。几何两组的loss还限定于有效return点，以尽量对齐RRCL监督覆盖域。

| 目标 | 可直接传给`tools/train.py --config-file`的配置 |
|---|---|
| Normal prediction | [geometry_normal.py](geometry_normal.py) |
| 6D Local shape descriptor | [geometry_local_shape.py](geometry_local_shape.py) |
| Role-only | [rrcl_role_only.py](rrcl_role_only.py) |
| Context-only | [rrcl_context_only.py](rrcl_context_only.py) |
| 完整RRCL | [pretrain-rrcl-v1m1-litept-v1m4-swiss-100.py](../pretrain-rrcl-v1m1-litept-v1m4-swiss-100.py)；已有独立RRCL权重可作首个seed |

四份新配置均完整列举参数，**不依赖`*base`**；完整RRCL继续用已存在且同样自包含的配置。`evaluate=False`与原Swiss-100实验一致，训练后`PreciseEvaluator`会对Swiss-Audit导出72D backbone特征。几何配置使用`GeometryFeatureTester`，role/context使用`RRCLFeatureTester`。输出分别进入独立实验及特征文件夹，不覆盖现有模型。

## 先生成离线目标

在[compute_las_local_geometry.py](../../../utils/compute_las_local_geometry.py)底部直接编辑参数，然后在pointspace环境运行：

```powershell
D:/app/anaconda3/envs/pointspace/python.exe utils/compute_las_local_geometry.py
```

默认读取`G:\data\swiss\joint_tiles_100\pointcloud`，向单独的`pointcloud_geom_p1`写出同名LAS，不触碰原文件。程序支持跳过已存在输出、`num_workers`和`max_files`；首次建议`max_files=2`试算并核对，再恢复`None`跑全量。**Normal与Local shape两组共同使用同一份离线输出，无需分别预处理。**原目录有13,530个LAS，总体积约126 GB；新extra dimensions使输出可能接近原体积的两倍，应预留约250 GB输出空间。现有K=16输出可以直接用GPU后端续跑，不必重算；大块LAS的几何计算明显加快，但整个流程可能受LAS写盘限制。

附加字段：`normal_x/y/z`、`geom_linearity`、`geom_planarity`、`geom_scattering`、`geom_curvature`、`geom_anisotropy`、`geom_verticality`、`geom_valid`。LAS原有点数、坐标、return、classification、intensity与其他属性原样保留。Dataset会把六维字段组成`local_shape[N,6]`，把有效标记读作`geom_valid[N]`。这些字段仅作为目标，不在`feature_keys=("coord", "intensity")`中。

## 训练

例如：

```powershell
D:/app/anaconda3/envs/pointspace/python.exe tools/train.py --config-file configs/rrcl/p1_swiss100/geometry_normal.py
```

其余三份新配置只需替换路径；完整RRCL则使用上表中的原配置。`role-only`不建context head、不做KNN；`context-only`不建role head且不扫描角色类别权重；完整RRCL保持原有实现与权重。所有模型的点特征读取统一为72D backbone输出。测试脚本[smoke_p1_training.py](../../../paper_cvrl/smoke_p1_training.py)已在一块LAS上以2048点完成**五组**CUDA前向和反向，**不代表25-epoch正式训练已经运行**。

## 方法口径与限制

- 当前Swiss离线目标来自原始完整LAS的0.25 m体素质心参考点、K=16 XYZ-KNN/PCA；已有部分输出采用K=16，续跑不可在同一目录混入K=48。GPU续跑使用`backend="gpu"`、`num_workers=2`和`overwrite=False`；两个worker可重叠GPU计算与LAS读写，更多worker会自动限制为2。后续K=48敏感性应单独输出。裁剪、随机丢点、微小jitter之后局部邻域会变化，因此它是与训练点集接近、但非严格同一邻域的固定目标。
- 六维量中`linearity+planarity+scattering=1`、`anisotropy=linearity+planarity`，存在代数相关；采用六维目标是为了覆盖经典描述符，后续应补四维去冗余敏感性。
- Normal的PCA正负号不唯一，loss使用`1−|cos|`；对近垂直立面尤其必要。六维描述量使用Smooth-L1，`verticality=1−|normal_z|`。`RandomRotate`和`RandomFlip`会同步变换LAS normal；描述量对当前Z轴旋转、XY翻转和等比例缩放不变。
- Normal和shape只在`geom_valid ∩ 有效return_role`上计loss。正式报告必须记录各组有效监督比例、总步数、实际时间、梯度/目标量纲；不能直接比较不同pretext原始loss值。
