# Fig. 3–4 method figures

Fig. 3 使用 Swiss 正射影像的真实屋顶—植被边界作背景，点与 token 支撑范围为
机制示意；Fig. 4 使用 Swiss-Audit 真实样本
`swisssurface3d_2024_2679-1248_2056_5728.copc_0033`，读取 LAS 坐标、GPS time、
`return_number` 和 `number_of_returns`。绘图参数分别保存在同名 JSON 中，
SVG 为可编辑版本。

## Fig. 3 — VPSD mechanisms

![VPSD mechanisms](figure_3_vpsd_patch_set_alignment.png)

**Suggested caption (English).** **Fig. 3. Point-wise assignment and patch–token
support under a small illustrative registration offset.** The background is
a real Swiss orthophoto crop at a roof–vegetation boundary. In both panels,
the schematic ALS locations or token supports remain fixed while the image
patch grid shifts by the same offset $\delta$. (a) Three boundary-near points
switch their discrete teacher assignment from $t_j$ to $t_{j+1}$. (b) A much
smaller number of spatially extended token supports continue to overlap patch
$j$, illustrating how patch-set aggregation can retain shared evidence under
a small local offset. The points, token footprints, and offset are schematic;
the figure does not report a measured robustness rate or guarantee invariance.

作图与解释边界：

- 影像裁剪来自真实 Swiss tile；图上 patch 网格为机制示意，不声称与裁剪图的
  实际 DINO patch 边界精确对齐。左侧 8 个虚拟点中有 3 个在边界处改变目标；
  细橙箭头表示监督目标指向相邻 patch，**不表示点本身移动**。
- 右侧仅画 3 个半透明的 token 空间支撑范围，不再把 token 画成密集点，也不再
  划分 Enc 2/3/4。$u_1/u_2/u_3$ 仅为集合成员示例，不意味着实际集合只有
  三个 token；这些轮廓也不是已训练 LitePT 的实际感受野或对应关系。
- 同一小偏移下，较宽的空间支撑可以继续与 patch 相交；但 **token 数较少本身
  不足以保证鲁棒性**。真实效果还取决于支撑范围、聚合权重、patch 边界、偏移
  方向与幅度。因此图中不报告定量稳健率。真实 Swiss-Audit 的 3-pixel 局部
  诊断保留在[旧版脚本](../../plot_fig3_vpsd_alignment.py)中，不作为正式 Fig. 3。

Source: [plot_fig3_vpsd_schematic.py](../../plot_fig3_vpsd_schematic.py).  
Metadata: [figure_3_vpsd_patch_set_alignment.json](figure_3_vpsd_patch_set_alignment.json).

## Fig. 4 — RRCL sensing physics

![RRCL sensing physics](figure_4_rrcl_return_physics.png)

**Suggested caption (English).** **Fig. 4. ALS return roles and local return
context in RRCL.** (a) Three enlarged real ALS examples show single returns on
exposed ground and a roof, and a GPS-time- and XY-consistent F/M/L sequence.
The narrow gray profile below locates all three examples within the same
Swiss-Audit tile. Only selected returns use the S/F/M/L palette: the same S
role can occur on different semantic surfaces. (b) The two center points share
role S, but their 48-neighbor current-set XYZ KNN supports have different
return-role compositions. Actual neighbors are colored by role over a faint
background of nearby real points; the adjacent four-bin histograms summarize
their local contexts. Together, point-wise role and local context characterize
different aspects of the laser–scene interaction.

代码一致性说明：

- 角色映射严格执行 `S → F → M → L` 判定，确保 `r=n=1` 首先归入 S。
- (a) 改用平坦的 Swiss-Audit 城区 tile；85 m × 1 m 来源剖面包括地面、植被、
  屋顶、立面。三张真实样例卡片是该剖面的局部高程视窗，不是虚构点云。
  完整三回波序列 `return_number=1/2/3`、`number_of_returns=3`，且同 GPS
  time、最大水平间距不超过 1 m。仅凭 GPS time 不能认定同一脉冲，因此脚本同时
  检验了空间邻近性。单回波分别从裸露地面和屋顶取真实点。
- (b) 当前点集代理采用原始 tile 的 0.25 m 体素确定性每格一点，然后排除中心点
  计算真实 48-NN。它匹配训练中先体素采样、再算 KNN 的顺序，但**不是**带随机
  取样和增强的某个具体训练 batch。两例中心都为 S，实际计数由同目录 JSON
  记录。局部邻域使用 PointSpace 静态点云渲染器与两个案例共用的斜向相机，
  不显示三维坐标框；背景为轻微高程灰度，并通过静态渲染器的 Lit-EDL 强化
  几何边缘。背景点和 KNN 角色色点使用基于共同显示密度选定的相同 5 px splat
  半径，白心黑边点为中心。当前真实邻域背景扩展至中心周围 8 m，两个实例
  使用相同物理投影视窗，而不是
  分别缩放。四-bin 直方图的
  顺序固定为 S/F/M/L，数值均来自真实计数。
- 正式版不使用方框，仅用浅灰虚线分隔实例与分图；`(a)`、`(b)` 分图题位于
  各自内容正下方，实例不使用数字/字母编号。(a) 来源剖面保持真实水平—高程
  比例，三回波例逐点标出 `r=1/2/3`。
- (b) 用细虚线椭圆圈出真实 48-NN，连接箭头从该区域边缘指向相邻直方图。
- 同角色对照只依据回波直方图和 XYZ 几何挑选；LAS 类别仅用于挑选裸地、屋顶
  两个单回波示例，**不进入** RRCL 目标或邻域对照选择，也不再用于图中配色。
  全图唯一配色语义为 S 蓝、F 橙、M 紫、L 绿。
- Return metadata 只用于构造预训练目标，不进入 LiDAR backbone 输入。

Source: [plot_fig4_rrcl_mechanism.py](../../plot_fig4_rrcl_mechanism.py).  
Metadata: [figure_4_rrcl_return_physics.json](figure_4_rrcl_return_physics.json).
