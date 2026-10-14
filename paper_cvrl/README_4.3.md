# CVRL 4.3 实验资产索引

本目录的4.3材料分为**论文正文图**与**审计报告补充结果**。两者共享Swiss-Audit，但不可把不同范围/参数的定性案例分区直接合并到总体统计中。

## 交付结果

| 用途 | 文件 |
|---|---|
| 正文 Fig. 5：可见性瓶颈 | [`results/section_43/figure_5_visibility_bottleneck.png`](results/section_43/figure_5_visibility_bottleneck.png) |
| 正文 Fig. 6：投影冲突机制 | [`results/section_43/figure_6_projection_conflict_mechanism.png`](results/section_43/figure_6_projection_conflict_mechanism.png) |
| 正文两图图注与案例限定 | [`section43_main_text_figures.md`](section43_main_text_figures.md) |
| 详尽分析报告 | [`CVRL_4.3_实验分析报告.md`](CVRL_4.3_实验分析报告.md)、[`CVRL_4.3_实验分析报告.pdf`](CVRL_4.3_实验分析报告.pdf) |
| 报告补充图（9幅） | [`results/section_43/report_figures`](results/section_43/report_figures) |
| 可核对数值 | `results/section_43/*.csv`；主评估指标为mIoU，逐类补充分析保留F1 |

`results/section_43/case_visibility_200m`仅含Fig. 6两个200 m案例的几何可见性sidecar。`backend_render_cache`和`pcv_*_metrics.csv`是渲染方式消融的历史审计产物，不作为正文结果。`tmp`中的随机初始化checkpoint和冒烟测试样本仍用于复现Scratch对照，**并非可任意清除的缓存**。

## 数据与计算边界

- 406个Swiss-Audit tile、固定点索引、特征、预测和主分析JSON在`G:\data\swiss\joint_tiles_100\Swiss-Audit`，没有复制进本仓库。总体可见性统计约9,600万点；frozen probe始终采用原始source block划分。
- Fig. 5及Fig. 6(a)使用Swiss-Audit总体分析。Fig. 6(b)(c)使用`G:\data\swiss\joint_tiles`的两个200 m tile及本目录的case-only可见性sidecar；这两例仅作物理机制说明。
- mIoU从固定测试行的混淆矩阵计算：逐类`TP/(TP+FP+FN)`，对并集非零的类取平均。历史`macro_F1`字段仍留在原始JSON作为辅助值，但报告表格、总体增益图和区域bootstrap使用mIoU。逐类F1表及其bootstrap仍明确标作F1。
- `q_geo`纯几何，不读取return或LAS类别。Fig. 6(b)(c)的“visible-surface proxy”则是评估后、按同一影像patch的有效LAS对应点多数类构造的标签代理，不是DINO类别预测。

## 代码入口与复现顺序

这些脚本均保留IDE直接运行的模块级路径参数。不要在重新制图时运行特征提取或重新拟合probe，除非确实要重做上游实验。

1. 数据准备：`prepare_swiss_audit.py`、`swiss_audit_visibility.py`、`export_swiss_audit_features.py`；可见性定义在`q_geo.py`。这些是上游步骤，现有审计目录已完成。
2. 固定协议分析：`analyze_cvrl_43.py`；独立RRCL合并为`add_rrcl_to_section43.py`。`refresh_section43_miou.py`从现有混淆矩阵补齐mIoU，无需重新训练probe；`bootstrap_cvrl_43.py`重算source-block配对mIoU区间。投影冲突单独由`analyze_projection_conflict.py`处理。
3. 正文图：先运行`prepare_section43_200m_cases.py`核验/产生两份case sidecar，再运行`plot_section43_paper_figures.py`。Fig. 5/6只由后者写入结果根目录。
4. 报告补充图：运行`plot_cvrl_43_results.py`、`plot_pcv_final.py`、`plot_projection_conflict.py`。九幅报告图输出到`report_figures`，不会与Fig. 5/6混放。
5. 报告PDF：修改Markdown和图后运行`build_section43_pdf.py`，再检查分页与图表。

`plot_pcv_extent_comparison.py`、`plot_pcv_edge_edl_point_size*.py`、`render_pointcloud_backends.py`和`blender_cycles_pointcloud.py`记录PCV/EDL/Blender探索，不属于4.3主分析流水线；保留是为了追溯渲染方案的选择。`CVRL_4.3_Swiss_Audit_TODO.md`及旧实验资产审计是历史计划文件，最终状态以本索引和分析报告为准。

## 快速核查

在仓库根目录运行：

```powershell
& D:/app/anaconda3/envs/pointspace/python.exe -m pytest tests/test_cvrl_section43_analysis.py tests/test_section43_paper_figures.py tests/test_section43_projection_alignment.py tests/test_swiss_audit_visibility.py -q
```

本节报告使用mIoU；若论文正文中先前手工誊写了macro-F1表，更新正文时应以新报告数值重新排版，不能仅替换指标名。

也可直接运行`paper_cvrl/validate_section43_assets.py`，检查两份Markdown的插图链接、96组mIoU与混淆矩阵的一致性，以及bootstrap CSV的指标标记。
