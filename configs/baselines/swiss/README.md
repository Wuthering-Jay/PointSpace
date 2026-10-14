# 4.5 Swiss-100 同主干基线（实施中）

当前 LitePT 适配配置（仍需正式训练验收）：

| 配置 | 模型 | 输入 | 验证状态 |
| --- | --- | --- | --- |
| [pretrain-point-m2ae-litept-v1m4-100.py](pretrain-point-m2ae-litept-v1m4-100.py) | `PointM2AE-LitePT-v1m1` | 单视图 XYZ+I | 更新后 mask-query decoder 的合成 BF16 前后向和 72D 导出通过；仍是 `-style` 近似 |
| [pretrain-msc-litept-v1m4-100.py](pretrain-msc-litept-v1m4-100.py) | `MSC-LitePT-v1m1` | 同一 LAS 的两种增强视图 XYZ+I | 原版全局 InfoNCE、view mixing 与 masked intensity 重建的合成 BF16 前后向通过；normal head 可选 |
| [pretrain-sonata-litept-v1m4-100.py](pretrain-sonata-litept-v1m4-100.py) | `Sonata-LitePT-v1m1` | 两全局+四局部视图 XYZ+I | 三目标合成 BF16 前后向、EMA 累积步与真实 Swiss 六视图加载通过 |
| [pretrain-d-ditr-litept-v1m4-100.py](pretrain-d-ditr-litept-v1m4-100.py) | `D-DITR-LitePT-v1m1` | XYZ+I；离线 DINO-1024 仅作点级 target | 合成与真实 Swiss LAS+DINO+对应关系 BF16 前后向通过 |
| [pretrain-concerto-litept-v1m4-100.py](pretrain-concerto-litept-v1m4-100.py) | `Concerto-LitePT-v1m1` | XYZ+I 六视图；离线 DINO-1024 仅作 patch target | 合成 BF16 前后向与真实 Swiss LAS+DINO 六视图加载通过 |

五者使用同一 `LitePT-v1m4` 主干、0.25 m GridSample、20 epoch 和 XYZ+I 输入，但**并非官方原生网络复现**。所有配置均设置 `batch_size_train=16` 与 `gradient_accumulation_steps=4`；PointSpace 训练器因此使用每卡 micro-batch 4、累积 4 次，有效 batch 为 16。辅助head只在预训练使用；导出的`backbone.*`与普通LitePT同形。MSC保留原全局InfoNCE、交叉掩码、0.8/0.0 view mixing和重建目标，但Swiss无RGB，改用intensity；原公开配置的normal重建为关闭，代码仍提供可选GPU KNN/PCA normal分支。MSC的精确InfoNCE row chunk由256增至1024，仅减少kernel调度，不改变8192负样本池。Point-M2AE关闭自身小decoder的activation checkpoint，但LitePT主干配置保持一致。D-DITR将重复引用的teacher patch预归一化一次，以16384点分块计算同一逐点cosine，并因显存充足关闭loss checkpoint。Sonata/Concerto保留mask/roll/unmask、四局部视图和4096原型；正式训练采用每项目标随机采样4096个匹配点，并以2048行为块执行数学等价的head checkpoint。该设置约覆盖每步11%的候选对应，20 epoch在候选集合近似稳定且独立采样的假设下累计被采样概率约90%，但仍须披露为有限匹配近似。Point-M2AE仍是LitePT的`-style`近似：原版FPS/KNN token pyramid被网格层级取代。D-DITR/Concerto使用离线DINOv3-1024而非原版在线DINOv2。全部不读取LAS semantic或return role。所有正式配置保留`CacheCleaner`。方法差异详见[4.5计划](../../../paper_cvrl/CVRL_4.5_基线适配与迁移实验计划.md)。

在项目根目录、`pointspace` 环境中启动：

```powershell
D:/app/anaconda3/envs/pointspace/python.exe tools/train.py --config-file configs/baselines/swiss/pretrain-point-m2ae-litept-v1m4-100.py
D:/app/anaconda3/envs/pointspace/python.exe tools/train.py --config-file configs/baselines/swiss/pretrain-msc-litept-v1m4-100.py
D:/app/anaconda3/envs/pointspace/python.exe tools/train.py --config-file configs/baselines/swiss/pretrain-sonata-litept-v1m4-100.py
D:/app/anaconda3/envs/pointspace/python.exe tools/train.py --config-file configs/baselines/swiss/pretrain-d-ditr-litept-v1m4-100.py
D:/app/anaconda3/envs/pointspace/python.exe tools/train.py --config-file configs/baselines/swiss/pretrain-concerto-litept-v1m4-100.py
```

这五项**尚未完成全量训练**。目标 batch 的显存/吞吐验证和 64 micro-step（16 次参数更新）真实训练短测已全部通过，无 NaN/OOM；详见 [首轮工程验证报告](../../../paper_cvrl/CVRL_4.5_首轮工程验证报告.md)。优化后 Sonata/Concerto 的短测平均 batch 时间分别约为1.54/1.55 s，较原8192-cap、256-row-chunk设置约加速4.6/5.2倍；完整20 epoch单卡仍需约29 h量级，正式实验应记录实际GPU-hours。

五份配置均启用`EarlyStoppingHook`监控epoch平均训练loss：前2个epoch只热身，之后若连续3个epoch未相对历史最佳值改善超过0.1%，则在保存该epoch的`model_last.pth`后干净停止；状态保存在`model/early_stopping.json`并支持断点续训。百分比阈值相对`abs(best)`计算，因而同一Hook也可用于mIoU、OA、MAE等不同量纲指标。批量启动脚本会跳过达到目标epoch或已经早停的checkpoint，并续训未完成checkpoint：

```powershell
D:/app/anaconda3/envs/pointspace/python.exe paper_cvrl/run_section45_pretraining.py
```
