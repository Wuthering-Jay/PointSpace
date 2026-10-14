# CVRL 数据集规模统计

统计日期：2026-10-02。

## 主表

| Dataset | Statistical split | Spatial units | Coverage area (km²) | Points used | Mean density (pts/m²) | Effective classes |
|---|---:|---:|---:|---:|---:|---:|
| Swiss-Paired | complete pretraining set | 13,530 × 100 m tiles | 135.300 | 3,439,144,277 | 25.42 | 9 |
| Swiss-Audit | complete audit subset | 406 × 100 m tiles | 4.060 | 95,965,298 | 23.64 | 9 |
| DALES | train only | 29 × 500 m source tiles | 7.250 | 361,670,122 | 49.89 | 8 |
| Vaihingen | train only | 1 source scene | 0.155 | 753,876 | 4.86 | 9 |
| OpenGF | train only | 151 × 500 m source tiles | 37.750 | 462,999,513 | 12.26 | 2 |

`Points used` 和平均密度遵循网络实际数据协议。Swiss-Paired 与 Swiss-Audit 为无标签预训练/审计点云，不做语义类别过滤；DALES、Vaihingen 和 OpenGF 只统计训练划分，且剔除训练时不参与损失的类别。

## 有效类别

- **Swiss-Paired / Swiss-Audit（9 类）**：Ground、Vegetation、Building roof、Water、Wire / transport line、Transmission tower、Bridge deck、Building facade、Bridge pier & stay cable。原始 class 1 `Unclassified` 不计入有效类别。
- **DALES（8 类）**：Ground、Vegetation、Cars、Trucks、Power lines、Fences、Poles、Buildings。原始 class 0 不参与训练。
- **Vaihingen（9 类）**：Powerline、Low vegetation、Impervious surfaces、Car、Fence / hedge、Roof、Facade、Shrub、Tree。该数据中的 class 0 是 Powerline，并非 unclassified/noise，因此保留。
- **OpenGF（2 类）**：Non-ground、Ground。原始 class 0 noise/unclassified 不参与训练。

## 统计口径

### Swiss-Paired

- 路径：`G:\data\swiss\joint_tiles_100\pointcloud`。
- 13,530 个 100 m × 100 m tile，来自 140 个不同的 1 km source block。
- 140 个 source block 的平面坐标均唯一，不存在同一地理方格的跨年份重复覆盖。
- 生成时 `overlap_factor=1`，因此没有多偏移分块造成的重复面积和重复点。
- 点数取自全量 Swiss 预训练 manifest：`paper_cvrl/results/section_45/swiss100_pretrain_manifest.json`。
- 覆盖面积按 13,530 个有效完整影像覆盖 tile 的名义面积相加。

### Swiss-Audit

- 路径：`G:\data\swiss\joint_tiles_100\Swiss-Audit\pointcloud`。
- 它是 Swiss-Paired 中确定性选择的 406 个 100 m tile，不是额外或空间外数据。
- 点数来自 `Swiss-Audit/analysis/q_geo_summary.json` 的全量 95,965,298 点，而不是 frozen probe 使用的抽样点数。

### DALES

- 原始训练路径：`E:\data\DALES\dales_las\train`。
- 29 个 500 m × 500 m source tiles，对应 7.25 km²。
- 原始点数为 368,667,682；其中 class 0 有 6,997,560 点。
- 当前下游协议只保留 LAS 类别 1–8，因此实际使用 361,670,122 点，主表密度为 49.89 pts/m²。
- 若按未过滤原始点计算，密度为 50.85 pts/m²。

### Vaihingen

- 原始训练文件：`E:\data\Vaihingen\Vaihingen3D_Traininig.las`。
- 使用原始 source LAS 统计，而不是 `tile/train` 中由多偏移网格产生的重叠训练 tile；后者会重复计算同一点。
- 覆盖面积按 LAS 头部 XY 范围计算：382.86 m × 405.29 m = 155,169.33 m²。
- 训练类别为 0–8，原始 753,876 点全部有效。

### OpenGF

- 原始训练路径：`E:\data\OpenGF\train`。
- 151 个 500 m × 500 m source tiles，对应 37.75 km²。
- 原始点数为 463,852,970；其中 class 0 noise/unclassified 有 853,457 点。
- 当前下游协议仅保留类别 1–2，因此实际使用 462,999,513 点，主表密度为 12.26 pts/m²。
- 若按未过滤原始点计算，密度为 12.29 pts/m²。

## 计算原则

平均密度统一计算为：

\[
\rho = \frac{N_{\mathrm{used}}}{A_{\mathrm{coverage}}}.
\]

DALES、Vaihingen 和 OpenGF 均从未切割的训练 source 文件统计，避免训练 tile 的重叠、边缘合并或 `orig_idx` 重复影响数据集规模。面积表示平面覆盖面积，不是三维曲面面积；点数为 GridSample、SphereCrop 和数据增强之前的原始有效点数。
