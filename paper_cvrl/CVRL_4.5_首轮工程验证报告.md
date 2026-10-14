# CVRL 4.5 首轮工程验证报告

日期：2026-09-22。本文只报告正式训练前的工程验收结果，不将单个随机初始化 batch 的 loss、速度或显存解释为方法性能。

## 1. 已完成的验证

五种 LitePT 适配基线及统一 DALES 迁移协议的测试共 `19 passed`：Point-M2AE、MSC、Sonata、D-DITR、Concerto 的目标函数、梯度、特征导出和配置约束均通过；DALES strict linear probe、decoder probe、full fine-tuning 及 Scratch 边界也通过。

Swiss-100 manifest 对 `pointcloud`、`dino_feature` 和 `correspondence` 三个目录进行了全量 stem、文件头和张量结构检查：

| 项目 | 结果 |
| --- | ---: |
| 对齐 tile | 13,530 |
| 原始点 | 3,439,144,277 |
| DINO teacher | 13,530/13,530 均为 1024D |
| LAS point format | format 1: 10,388；format 6: 3,142 |
| 确定性内容审计 | seed 45，128 tile，31,297,106 点 |
| 有效 return 字段 | 99.9997% |
| 有效 point-to-patch correspondence | 45.6324% |
| tile 平均 observability | 0.5881 |
| inventory SHA256 | `91d0fb0ef156f0198a1a55176978a76bd5c5f0469d3306357e25b7789e5f4b54` |

普通模式仅对固定128个tile执行完整内容哈希，避免每次重复读取约268 GiB资产；归档发布时将`HASH_ALL_FILES=True`即可生成全文件SHA256。

## 2. 真实 Swiss batch 验收

环境为 RTX 4080、BF16 autocast、batch size 1。所有方法都使用各自正式 Swiss 配置，未经缩小模型或裁剪目标。计时为单进程首个真实样本，仅用于发现异常路径，不作为论文速度比较。

| 方法 | 输入规模摘要 | Loss | 前向+反向 / s | 峰值分配显存 / GiB | 结果 |
| --- | --- | ---: | ---: | ---: | --- |
| Point-M2AE-style | 88,272 points | 0.1338 | 0.911 | 0.406 | Pass |
| MSC | 82,096 / 85,235 points；8,192 pairs | 10.4535 | 0.247 | 2.151 | Pass |
| Sonata | 89,756 / 83,811 global；4×8,192 local | 8.4043 | 0.807 | 2.616 | Pass |
| D-DITR | 89,756 points；77,591 supervised | 0.9910 | 0.126 | 1.259 | Pass |
| Concerto | 88,272 / 89,572 global；4×8,192 local；3,969 patches | 9.2567 | 0.701 | 2.716 | Pass |

Sonata/Concerto 的 `matched_pairs=12,288` 是早期2048 cap的三项目标累计值；该batch-1结果只是起步smoke test。后续先以8192 cap完成高覆盖和显存验收，再根据真实训练器成本将正式配置调整为4096 cap。

## 3. 已修复的真实瓶颈

原 `LasImageDataset._index_assets()` 在 Windows 大目录上使用 `Path.iterdir()+is_file()`，D-DITR 与 Concerto 建立13,530组三模态索引分别需要约65 s和58 s，而网络单步仅需约0.13 s和0.70 s。改成复用目录枚举元数据的 `os.scandir()` 后：

- 单独 D-DITR 数据集初始化降至约0.247 s；
- 完整真实批次测试中的 D-DITR 数据准备为0.289 s；
- Concerto 数据准备为0.381 s；
- stem唯一性、后缀过滤和缺失资产报错语义保持不变；
- 相关9项单元测试重新通过。

该修复主要消除启动和多次构建 loader 的无效等待，不改变样本内容或训练目标。

## 4. 正式训练前验收与执行顺序

### 4.1 匹配上限首轮敏感性（已完成）

固定相同真实 tile、数据增强结果、模型初始化和前向随机种子，仅改变 `matching_max_pair`：

| 方法 | cap | 累计matched pairs | matching fraction | Loss | 峰值显存/GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sonata | 512 | 3,072 | 1.48% | 8.4027 | 2.367 |
| Sonata | 2,048 | 12,288 | 5.90% | 8.4026 | 2.647 |
| Sonata | 8,192 | 49,152 | 23.61% | 8.4024 | 3.612 |
| Concerto | 512 | 3,072 | 1.46% | 9.2908 | 2.412 |
| Concerto | 2,048 | 12,288 | 5.83% | 9.2912 | 2.681 |
| Concerto | 8,192 | 49,152 | 23.34% | 9.2912 | 3.652 |

随机初始化首步 loss 几乎不随 cap 变化，**不能**据此认定不同cap等价；参与率才是关键差异。8192相对2048增加约0.97 GiB单样本峰值。8192在真实micro-batch 4下虽能运行，但长期训练过慢；最终以4096作为覆盖率和计算成本的折中，并在论文中明确披露随机有限匹配近似。

### 4.2 目标 batch 验证（已完成）

训练器会把 `batch_size_train` 先除以 `gradient_accumulation_steps` 再构建 DataLoader。因此统一配置 `16/4` 的真实含义是：**每卡 micro-batch 4、累积 4 次、有效 batch 16**。按这一真实协议重新验证结果如下：

| 方法/设置 | micro-batch | 有效 batch | 前向+反向/s | 峰值显存/GiB | 决策 |
| --- | ---: | ---: | ---: | ---: | --- |
| Point-M2AE-style | 4 | 16 | 1.589 | 1.184 | 采用 |
| MSC | 4 | 16 | 0.567 | 8.293 | 采用 |
| Sonata cap 2048 | 4 | 16 | 2.095 | 10.557 | 仅作 cap 参照 |
| Sonata cap 8192 | 4 | 16 | 3.236 | 13.630 | 高覆盖验收基线；匹配覆盖21.72% |
| D-DITR | 4 | 16 | 0.369 | 4.548 | 采用 |
| Concerto cap 2048 | 4 | 16 | 1.489 | 10.380 | 仅作 cap 参照 |
| Concerto cap 8192 | 4 | 16 | 2.914 | 13.607 | 高覆盖验收基线；匹配覆盖21.80% |

所有正式配置现在使用同一真实 batch 协议，不再存在先前误读为“有效 batch 64”的不一致性。

### 4.3 短程稳定性（已完成）

五种方法都通过了实际 `DefaultTrainer` 路径的 64 micro-step / 16 optimizer-update 检查，包含 BF16 AMP、梯度累积、正式 DataLoader 与 Sonata/Concerto EMA hook。五项进程返回码均为 0，loss 均为有限值，无 NaN、OOM 或梯度链路错误。

| 方法 | 首个记录 loss | 最后记录 loss | 相对变化 | 64-step 墙钟时间/s | 观察 |
| --- | ---: | ---: | ---: | ---: | --- |
| Point-M2AE-style | 0.1284 | 0.0986 | -23.21% | 50.1 | 重建 loss 总体下降 |
| MSC | 10.2875 | 9.1407 | -11.15% | 68.6 | NCE 下降，正负相似度间隔扩大 |
| Sonata（优化后） | 8.3717 | 8.2922 | -0.95% | 137.3 | mask/roll/unmask均稳定；平均batch约1.54 s |
| D-DITR | 1.0114 | 0.9039 | -10.63% | 72.8 | cosine loss 稳定下降 |
| Concerto（优化后） | 9.2902 | 8.7052 | -6.30% | 139.9 | intra/cross两分支均有效；平均batch约1.55 s |

这一短测只证明数值和工程稳定性。由于20-epoch OneCycleLR在短测期间几乎仍处于起始学习率，不能据此比较最终收敛质量。原8192-cap、256-row-chunk训练器短测中，Sonata/Concerto平均batch约7.05/8.10 s，ETA约132/152 h；改为4096随机匹配、2048-row chunk并关闭无随机算子checkpoint的RNG状态保存后，平均batch降至约1.54/1.55 s，对应约4.6×/5.2×加速，20 epoch ETA约29 h量级（RTX 4080单卡，仅作量级参考）。两全局、四局部、EMA teacher、4096 prototypes、Sinkhorn及mask/roll/unmask目标均保留；唯一改变目标采样分布的措施是8192→4096 cap，必须作为计算限制披露。

曾测试但未采用的方案包括：把多视图拼成三个大batch统一过主干，以及关闭LitePT MLP activation checkpoint。二者在孤立单步中偶有收益，但在真实训练器连续更新时因峰值显存和分配压力反而显著变慢，因此未进入正式实现。项目在长时间训练中依赖`CacheCleaner`控制缓存堆积，因此所有正式配置继续保留周期和自适应清理；短基准需把清理事件视为真实训练成本的一部分。

### 4.4 执行参数消融（已完成）

使用同一Sonata真实训练路径各运行8个micro-step；墙钟时间包含进程启动、数据集构建和首批预热，因此只用于同机相对比较：

| matching cap | head chunk | 8-step墙钟/s | 相对8192/256 |
| ---: | ---: | ---: | ---: |
| 8,192 | 256 | 87.2 | 1.00× |
| 8,192 | 1,024 | 75.7 | 1.15× |
| 4,096 | 1,024 | 50.2 | 1.74× |
| 4,096 | 2,048 | 50.1 | 1.74× |
| 4,096 | 4,096 | 49.6 | 1.76× |

短基准中2048与4096 row chunk差异小于2%，而较长连续测试中2048更稳定，因此正式配置取2048，不将8-step噪声解释为4096必然更快。真正的大幅收益来自避免8192 cap下原型头反向的持续分配压力；64-step端到端结果是正式速度判断的主要依据。

### 4.5 其余三种基线的等价执行优化（已完成）

在保留`CacheCleaner`、统一LitePT主干配置和所有pretext目标的条件下，对其余方法进行了8-step筛选和64-step复验：

| 方法 | 正式调整 | 8-step平均batch/s | 64-step平均batch/s | 峰值显存/GiB |
| --- | --- | ---: | ---: | ---: |
| Point-M2AE | 关闭小型coarse/fine decoder checkpoint | 0.171（原0.185） | 受本轮数据等待影响为0.229 | 1.342 |
| MSC | 精确InfoNCE row chunk 256→1024 | 0.423（原0.452） | 0.431（原0.434） | 7.884 |
| D-DITR | teacher patch只归一化一次；chunk 4096→16384；关闭loss checkpoint | 0.349（原0.434） | 0.382（原0.445） | 6.621 |

Point-M2AE的GPU计算段稳定减少，但其总墙钟受LAS读取波动主导，不应宣称确定的端到端加速比例。MSC在较长测试中的收益不足1%，调整仍保持，因为它不增加目标近似且未提高显存。D-DITR的64-step平均batch降低约14%，是三者中最稳定的实际收益；显存从原约4.55 GiB升至6.62 GiB，仍明显低于16 GiB设备上限。关闭LitePT MLP checkpoint对MSC和D-DITR没有稳定收益，因此三者主干仍使用与CVRL相同的`checkpoint_mlp=True`。所有配置均保留周期和自适应CacheCleaner。

后续顺序：

1. 短程数值稳定性已通过；完整训练期间继续记录 loss、有效匹配率、峰值显存与实际 GPU-hours。
2. 冻结正式超参后完成 20 epoch Swiss-100 预训练，保存 `backbone_last.pth` 及配置/manifest 摘要。
3. 每个checkpoint先完成权重加载与特征导出往返，再启动DALES单种子LP/DP/FT筛选。
4. 协议确认后补足三种seed，并以mIoU mean±std形成4.5主表；训练GPU-hours与峰值显存作为计算成本附表。

## 5. 可复现资产

- 数据清单生成：`paper_cvrl/build_section45_manifest.py`
- 全量JSON：`paper_cvrl/results/section_45/swiss100_pretrain_manifest.json`
- 逐tile CSV：`paper_cvrl/results/section_45/swiss100_pretrain_manifest.csv`
- 固定stem：`paper_cvrl/results/section_45/swiss100_pretrain_stems.txt`
- 真实batch验证：`paper_cvrl/validate_section45_real_batches.py`
- 验证结果：`paper_cvrl/results/section_45/real_batch_validation.json`
- 匹配上限敏感性：`paper_cvrl/benchmark_section45_matching_cap.py`
- 敏感性结果：`paper_cvrl/results/section_45/matching_cap_sensitivity.json`
- 正式batch验证：`paper_cvrl/benchmark_section45_configured_batches.py`
- 正式batch结果：`paper_cvrl/results/section_45/configured_batch_validation.json`
- Sonata执行参数基准：`paper_cvrl/benchmark_sonata_execution.py`
- 执行参数结果：`paper_cvrl/results/section_45/sonata_execution_benchmark.json`
- 其余基线执行筛选：`paper_cvrl/benchmark_other_baseline_execution.py`
- 其余基线筛选结果：`paper_cvrl/results/section_45/other_execution_benchmark.json`
- 优化后正式batch显存：`paper_cvrl/results/section_45/optimized_configured_batch_validation.json`
- 优化后64-step结果：`paper_cvrl/results/section_45/optimized_stability_summary.json`
- 短程稳定性脚本：`paper_cvrl/run_section45_stability.py`
- 短程结果：`paper_cvrl/results/section_45/short_stability_summary.json`
