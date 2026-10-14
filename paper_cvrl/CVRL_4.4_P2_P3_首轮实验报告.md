# 4.4 首轮执行报告：P2结构分层、P3表示手术与P1几何目标

> 本文件保留首轮执行记录；P1--P4 的最终统一结果、全指标和修正后的结论见
> [4.4 完备实验报告](CVRL_4.4_完备实验报告.md)。下文末尾关于“P1 尚未训练”的状态已过期。

本报告是Swiss-Audit的**预训练覆盖域内机制审计**，不是跨区域泛化结论。分析复用4.3已冻结的原始source-block划分及100,000个相同测试点（28个held-out source block、78个tile）；线性读出使用相同的100,000训练点、逐行L2归一化、类别平衡LogisticRegression。mIoU沿用4.3口径。置信区间以source block为重采样单位；P2重复2,000次，P3重复1,000次。所有P2结构标签都在预训练**完成后**从XYZ、q_geo或LAS类别构造，不进入模型。

## P2：回波信息在什么结构上有帮助

下表的增益为同一CVRL checkpoint的`full(200D) − visual(128D)`，不是独立RRCL与VPSD的因果差异。Ground/Vegetation的类条件组使用该类别Recall；混合类别组使用mIoU。单位均为百分点。

| 测试层 | 点数 | source block | 增益 | 95%配对CI |
|---|---:|---:|---:|---:|
| 全部测试点，mIoU | 100,000 | 28 | +5.50 | [+3.38, +6.86] |
| `P_inv`，mIoU | 29,182 | 28 | +5.59 | [+3.69, +7.54] |
| `P_vis`，mIoU | 43,092 | 28 | +5.33 | [+1.96, +7.26] |
| 林下Ground，Recall | 7,084 | 27 | **+25.03** | [+17.39, +32.84] |
| 开阔Ground，Recall | 11,448 | 28 | +3.60 | [+1.51, +6.57] |
| 内部Vegetation，Recall | 10,144 | 27 | +5.03 | [+3.45, +6.86] |
| 可见Vegetation，Recall | 12,643 | 28 | +3.84 | [+2.65, +5.04] |
| 多层柱状区域，mIoU | 65,685 | 28 | +7.10 | [+4.43, +8.68] |
| 单层柱状区域，mIoU | 32,424 | 28 | +2.29 | [+1.28, +4.00] |

林下Ground定义为`Ground ∩ P_inv`且同一1 m XY柱内至少有8个点、上方至少3 m处**确有Vegetation类点**；开阔Ground为`Ground ∩ P_vis`且无任何该高度的上覆点。Vegetation标签仅用于事后定义评估层，不进入模型。多层柱状区域由1 m XY柱内的XYZ高度跨度≥3 m且点数≥8定义，不读取return字段。它只是一个快速、可复现的**高度跨度代理**，不是严格的离散层数估计。1 m柱内同时含Vegetation与Building roof的边界层仅1,915个点/21个block，full增益+1.36点、CI跨零；不能据此宣称边界改善。

配对source-block差中之差：

| 对比：`full−visual`在第一层减第二层 | 差中之差 | 95% CI |
|---|---:|---:|
| 林下Ground − 开阔Ground（Recall） | **+21.43点** | [+14.31, +28.22] |
| 内部Vegetation − 可见Vegetation（Recall） | +1.18点 | [+0.16, +2.30] |
| 多层 − 单层（mIoU） | +4.81点 | [+1.22, +6.52] |

一个重要的密度敏感性：原始林下/开阔Ground的1 m柱点数中位数分别为38/14，不能把上述交互当作已完全控制密度。在两边都限制于**16–64点/柱**后，分别保留5,243/4,725点，中位数仍为34/20；Ground差中之差为+26.28点，CI[+18.06,+33.45]，方向未变，但这只是**重叠密度范围限制**，并非严格密度匹配。多层/单层在相同限制后的差中之差为+4.00点，CI[−0.94,+6.19]，已不能确认稳定交互。因此“回波在多层结构普遍更强”仍需更严的类别×密度匹配和P1几何基线；当前更坚实的发现是林下Ground。

独立RRCL与独立VPSD的全域mIoU分别为0.2854/0.4694，**不能**声称RRCL整体更优；但在上述林下Ground同类测试点，独立RRCL/VPSD的Recall为0.9027/0.7336，高16.91点。联合模型的visual/full为0.6947/0.9449，方向一致，仍不能排除训练架构差异。测试点中return无效比例全域为0.39%，林下Ground为0.03%，因此该组结果不是大量无效return点所驱动。

## P3：Representation surgery

已逐元素验证导出特征满足`CVRL-full = [zV(128)‖zR(72)]`，最大绝对误差为0。以下在**同一固定训练/测试点**上重新训练线性probe；random72对真实zR每一维独立置换，精确保留各维边际但破坏维间/逐点对应；shuffle(zR)在每个原始source block内且训练/测试分开置换，保留完整zR边际及block分布。两个对照各重复5个seed，绝不按标签或q_geo置换。

| 200D表示，整体逐行L2 | Linear mIoU All | Cosine kNN mIoU All |
|---|---:|---:|
| 正确配对`zV‖zR` | **0.5127** | **0.4003** |
| `zV‖random72`，5-seed均值±SD | 0.4501±0.0014 | 0.3419±0.0063 |
| `zV‖shuffle(zR)`，5-seed均值±SD | 0.4539±0.0020 | 0.3272±0.0122 |

作为维度不同的参照，单独`zV`和`zR`的4.3 linear mIoU为0.4577和0.3177，kNN为0.3702和0.2787。完整特征比五次随机控制平均高6.26个linear百分点，比五次block内置换平均高5.88点。每个置换/随机seed的full−control配对block CI均不跨零；例如seed 11的random为[+3.75,+7.48]点，shuffle为[+3.43,+7.01]点。

范数敏感性：若先将zV、zR分别L2归一化再拼接，正确配对的linear mIoU仍为0.5122，random/shuffle分别为0.4447/0.4484；因此线性互补不依赖原始支路范数比例。但kNN从整体归一化的0.4003降至0.3749，说明原始范数比例对局部邻域结构可能有用，不能笼统宣称范数完全无关。

P3支持的是：**正确逐点配对的回波子空间含有增量信息，收益不是多72维或场景级边际本身造成。**它不能证明共享浅层梯度没有竞争，也不能证明return物理信息是唯一来源；后者须等待P1同构几何SSL对照。

## P1构造状态更新：Normal与六维Local shape descriptor

原先的曲率/平面度两维原型已扩展为正式P1的Normal与六维Local shape descriptor。实现见[离线LAS计算工具](../utils/compute_las_local_geometry.py)、[GeometrySSL模型](../pointspace/models/backbone/geometry_ssl/geometry_ssl_v1m1.py)和[五组训练配置说明](../configs/rrcl/p1_swiss100/README.md)。计算仅依赖XYZ，不读return、LAS类别、q_geo或影像；训练loss再与有效return掩码求交。Normal采用符号不敏感cosine loss，shape采用Smooth-L1。两个几何任务、role-only、context-only与完整RRCL均已通过2048点GPU前后向测试。**Swiss全量离线几何LAS尚未生成、五组正式P1训练尚未启动。**

[初始原型测试](../tests/test_section44_geometry_targets.py)与[正式离线工具测试](../tests/test_compute_las_local_geometry.py)均通过。离线工具以0.25 m体素质心作KNN参考，并把目标写回原LAS每一点；它并不等于训练时随机裁剪/丢点/微扰后的在线KNN，后续方法节必须明确说明，且应做K与体素尺度敏感性。不能对Swiss-Audit每tile仅抽样的4,096点计算训练目标，否则会显著改变局部物理尺度。

下一步运行全量离线LAS计算，再让Normal、Local shape、Role-only、Context-only与Full RRCL按同预算训练、导出72D表示，并在固定Swiss-Audit协议下比较；此时才能回答“回波优于一般局部几何目标”这一P1核心命题。

## 复现与产物

- P2：[脚本](analyze_cvrl_44_strata.py)、[完整数值](results/section_44/p2_strata.json)、[结构层图](results/section_44/figure_44_p2_structure.png)。
- P3：[脚本](analyze_cvrl_44_surgery.py)、[完整数值](results/section_44/p3_surgery.json)、[表示手术图](results/section_44/figure_44_p3_surgery.png)。
- 绘图：[脚本](plot_cvrl_44.py)。

IDE中分别运行`analyze_cvrl_44_strata.py`、`analyze_cvrl_44_surgery.py`、`plot_cvrl_44.py`即可更新。结果JSON中保留逐组点数、block数、类别构成、return无效率、各表示分数、配对CI及五个seed单独结果。
