# q_geo 定性目视验证记录

## 验证状态

- 日期：2026-09-18
- 数据：Swiss-Audit，406个100 m tile
- 判定对象：连续`q_geo`及固定阈值形成的`P_inv/P_amb/P_vis`
- 阈值：`P_inv <= 0.20`，`P_vis >= 0.60`，中间为`P_amb`
- 检查载体：`G:\data\swiss\joint_tiles_100\Swiss-Audit\q_geo_pointcloud`
- 人工结论：项目作者完成目视检查，认为当前划分可靠，可用于4.3机制分析。
- 决策：冻结0.20/0.60阈值，后续不得依据下游分类结果调节阈值。

## 目视检查要点

检查时同时观察连续`q_geo`、三分区、原始高度结构和原LAS类别，重点覆盖：

1. 开阔地和平屋顶的稳定可见表面；
2. 尖顶、屋面边缘和建筑立面附近；
3. 树冠顶层、冠层内部和低矮植被；
4. 桥面、线塔及其他稀有基础设施；
5. 水面及局部密度变化区域。

可复现的代表性样例由`plot_cvrl_43_results.py`按各环境`P_inv`比例的中位tile自动选择，输出为`figure_43_q_geo_qualitative_examples.png`，避免只展示人工挑选的最佳案例。

## 论文表述边界

本记录支持“经跨环境目视检查，几何分区与预期表面/内部结构一致”的定性表述。由于没有逐点人工二值标签，不能报告人工precision、recall、IoU或agreement，也不应把`q_geo`称作visibility ground truth。论文中应称其为**经过定性验证的geometry-only operational partition**。

