# 4.6 DALES 标签效率曲线：实验方案与实施状态

## 研究问题

在相同的 DALES 完整点云几何、Swiss 预训练权重和 FT 网络结构下，比较 0.01%、0.1%、1%、10%、100% **训练点标签**的 mIoU 曲线。100% 的既有 4.5 DALES FT 结果可先作为参考端点；由于本节稀疏实验改为 20 epoch × 3 loop，该端点并非严格同训练日程，正式曲线最好补跑匹配协议的 100%。主问题是 CVRL 是否在低标签预算下比 Scratch、单任务 VPSD/RRCL 以及其他预训练方法更高效；不能预设其胜出。

## 标签预算定义

- 分母是 DALES 训练目录中 LAS 类别 1–8 的有效标签点；类别 0 等无效点不计入分母。与 4.5 一样先过滤无效类别，随后保留全部有效类别点的几何输入，但只允许被选中的点参与 CE + Lovasz 监督。
- 固定 `seed=42`，对每个源 tile 的原始点序号生成一个确定的随机 rank。正式实验阈值 0.0001、0.001、0.01、0.10 形成严格嵌套的四档标签集。所有方法共享完全相同的点级标注，不按方法重抽。
- 采样不按类别强行均衡，主实验保持自然类别比例。类别权重和 tile 级 weighted sampler 仅由该档已选中的标签统计，避免读取预算外的类别真值。完整点云的类别过滤仍与 4.5 协议一致。
- 稀疏标签在 `ClassFilter` 后设为 `ignore_index=-1`，点的 XYZ/强度仍进入网络。随后沿用原有随机增强、GridSample 和 SphereCrop；因此预先分配的标注比例与某个具体 batch 的有效标签比例可能略有差异。记录各档的实际原始标注数，并在运行时抽样审计体素采样后的有效标签数。
- 验证与测试保持 100% 标签，不进行稀疏化；除训练标签预算外，稀疏实验使用同一数据路径、grid size、batch/accumulation、20 epoch × 3 loop、损失、学习率和模型选择规则。以验证 mIoU 进行相对阈值 0.25%、patience=3、warmup=2 的早停。

## 存储方案与已生成数据

不复制或修改 13.27 GB 的 DALES LAS 训练集。`utils/create_sparse_label_manifests.py` 仅在 `E:\data\DALES\dales_las\tile\sparse_labels_seed42` 保存 2,900 个小 NPZ（最大 10% 的原始点序号和共用 rank）及一个统计 JSON，合计约 182.2 MB。源文件尺寸、修改时间和点数在加载时校验。全部档位共用同一套索引，绝不需要多份 LAS。

训练集共有 368,667,682 个原始点，其中类别 1–8 的有效点为 361,670,122 个：

| 名义预算 | 实际有效标签数 | 实际比例 | Cars | Trucks | Power lines | Poles | 零标注 tile |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.01% | 36,334 | 0.01005% | 256 | 73 | 86 | 34 | 1 |
| 0.1% | 362,513 | 0.1002% | 2,570 | 792 | 781 | 281 | 0 |
| 1% | 3,618,303 | 1.0004% | 26,175 | 7,550 | 7,936 | 2,838 | 0 |
| 10% | 36,165,902 | 9.9997% | 258,933 | 74,932 | 79,898 | 27,711 | 0 |

0.01% 的尾类尤其稀疏，应保留每类 IoU，并至少对关键低预算结论增加随机种子敏感性分析。不能为了得到好看的结果而事后改动 seed 或重新抽标签。

## 配置与指标

`utils/generate_dales_label_efficiency_configs.py` 已生成 `configs/label_efficiency/dales` 下 9 种方法 × 4 个正式稀疏预算 = 36 份独立 FT 配置。此前的 5% 配置文件保留，但不在本轮命令中执行。配置使用 `SparseLabelLasDataset`，验证/测试仍为普通 `LasDataset`。100% 可以先引用 4.5 的 9 组 FT 日志，但它们使用 10 epoch × 5 loop、未采用本节的 20 epoch × 3 loop 和早停；严格的同协议标签效率曲线需要另行补跑 100% 端点，不能把现有端点当作完全同条件结果。

主指标为完整 held-out 数据的 mIoU；同时保留 8 类 IoU、OA、mAcc、最佳验证轮次。主图以对数尺度表示标签比例（0.01/0.1/1/10/100），各方法为曲线；建议把 Scratch、VPSD、RRCL、CVRL 高亮，其他方法细线或分组展示，以免 9 条曲线重叠。辅助表报告各预算 CVRL 相对 Scratch、VPSD、RRCL 的百分点差以及尾类结果；可报告 log-budget 轴下的面积，但不能替代逐档 mIoU。

100% 曲线端点采用现有 4.5 FT 测试 mIoU：Scratch 58.06、Point-M2AE 64.11、MSC 79.59、Sonata 78.90、D-DITR 69.08、Concerto 54.61、VPSD 79.69、RRCL 79.96、CVRL 79.28（单位：%）。

4.5 的 DALES `val` 与 `test` 指向同一批 held-out 文件，最终测试采用多 fragment 推理；论文中必须如实陈述其不构成独立于模型选择的数据集。

## 测试与执行顺序

### Scratch 精度坍缩预试验

在九方法正式曲线之前，先以相同 Scratch FT 协议测试 0.01%、0.03%、0.1%、0.3%、1% 五档。它们与主实验共用 seed=42 的嵌套索引，但配置及结果独立保存在 `configs/label_efficiency/dales/scratch_collapse` 和 `exp/label_efficiency/dales/scratch_collapse`，不会覆盖九方法实验。五档实际有效标注比例为 0.01005%、0.02999%、0.10023%、0.30018%、1.00044%。最低档只有 34 个 pole、73 个 truck 标签；这是压力测试，不能将尾类 IoU 的单 seed 波动误认为明确的表示能力变化。

0.01% 档有 1/2,900 个原始 tile 没有有效标注，稀疏训练的 weighted sampler 对其赋零采样权重。另有少量 tile 在增强和体素采样后暂时失去全部标注；数据集会重取有监督样本，防止全 `ignore_index` 微批次产生 NaN。验证和精确测试仍为完整标签，预测 LAS 不保存。该保护只保证监督有效，并不增加标注预算。

- [x] 固定嵌套标签抽样；真实 DALES 全量生成和类别覆盖审计。
- [x] 验证原 LAS 不变、源文件变化检测、有效标签掩码不删除几何点。
- [x] 验证四档实际样本、标签范围、损失有限，2-worker DataLoader 可读取；最低 0.01% Scratch FT 做单次 GPU 前向/反向。
- [x] 解析 36 份配置，检查全部使用 20 epoch × 3 loop 和 0.25% 相对早停，验证/测试不稀疏化。
- [ ] 正式训练 36 组，保留每轮验证曲线与最终逐类指标；训练前记录首批若干 tile 的 GridSample 后有效标签比例。
- [ ] 将 100% 的既有 FT 结果作为参考端点；若绘制严格同协议曲线，补跑 100% 的 20 epoch × 3 loop 对照。
- [ ] 针对 0.01% 和 0.1% 对 CVRL、Scratch、VPSD、RRCL 增加预先规定的独立种子，报告均值及波动；其他方法的单 seed 结果明确标注。
- [ ] 绘制标签效率曲线并分析优势是否随预算变化、尾类是否受有限标签数主导。

运行单项的例子：

```powershell
& D:/app/anaconda3/envs/pointspace/python.exe tools/train.py --config-file configs/label_efficiency/dales/semseg-cvrl-litept-v1m4-dales-ft-0p1pct.py
```
