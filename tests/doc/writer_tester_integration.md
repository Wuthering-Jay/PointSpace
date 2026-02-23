# Writer × Tester 集成文档

## 概述

PointSpace 的 Writer 系统已完整嵌入到 `SemSegTester` 和 `DINOSemSegTester` 中，
替换了原来针对各数据集采用不同保存方法的内联代码。

系统分为两个层次：

| 层次 | 用途 | 接口 | 选择方式 |
|------|------|------|----------|
| **Benchmark Writer** | 竞赛/评测提交格式 | `BaseBenchmarkWriter` | 根据 `cfg.data.test.type` 自动创建 |
| **General Writer** | 实际生产输出 (LAS/PLY/PCD) | `BaseWriter` | 通过 `cfg.writer` 配置 |

---

## 1. Benchmark Writer（竞赛提交格式）

### 1.1 工作原理

Tester 在 `test()` 方法中自动根据数据集类型创建对应的 Benchmark Writer：

```python
from pointspace.writers.benchmark import create_benchmark_writer

benchmark_writer = create_benchmark_writer(
    dataset_type=self.cfg.data.test.type,  # 如 "ScanNetDataset"
    save_dir=save_path,                     # 通常是 cfg.save_path/result/
    dataset=self.test_loader.dataset,       # 数据集对象
)
```

### 1.2 支持的数据集

| 数据集类型 | Writer 类 | 提交格式 | 关键属性 |
|-----------|-----------|---------|---------|
| `ScanNetDataset` | `ScanNetBenchmarkWriter` | `.txt` (class2id 映射) | `dataset.class2id` |
| `ScanNet200Dataset` | `ScanNetBenchmarkWriter` | 同上 | `dataset.class2id` |
| `ScanNetPPDataset` | `ScanNetPPBenchmarkWriter` | `.txt` (逗号分隔 top-3) | `topk=3` |
| `SemanticKITTIDataset` | `SemanticKITTIBenchmarkWriter` | `.label` (uint32 二进制) | `dataset.learning_map_inv` |
| `NuScenesDataset` | `NuScenesBenchmarkWriter` | `.bin` (uint8 + submission.json) | pred+1 偏移 |
| `S3DISDataset` | `S3DISBenchmarkWriter` | `.pth` (6-fold 指标) | `dataset.split` |

### 1.3 生命周期

```
┌─────────────────────────────────────────────┐
│  create_benchmark_writer()  ← 自动选择      │
│              ↓                               │
│  setup()           ← 创建目录/metadata       │
│              ↓                               │
│  ┌─── for 每个样本 ───┐                     │
│  │  write()          ← 写提交文件            │
│  │  pred_for_eval()  ← 变换 pred 用于评测    │
│  └────────────────────┘                     │
│              ↓                               │
│  finalize()        ← 收尾（如 S3DIS .pth）   │
└─────────────────────────────────────────────┘
```

#### 各方法说明

| 方法 | 调用时机 | 默认行为 |
|------|---------|---------|
| `setup()` | 评测循环前，仅主进程 | 创建 `submit/` 目录 |
| `write(data_name, pred)` | 推理每个样本后 | **抽象方法**，子类必须实现 |
| `pred_for_eval(pred)` | `write()` 之后 | 返回原 `pred`（ScanNet++ 返回 `pred[:, 0]`） |
| `finalize(**kwargs)` | 评测循环结束、指标聚合后 | 空操作（S3DIS 保存 `.pth`） |

### 1.4 topk 属性

`BaseBenchmarkWriter.topk` 控制模型输出的解码方式：

- `topk=1`（默认）：使用 `pred.max(1)[1]`（argmax）
- `topk=3`（ScanNet++）：使用 `pred.topk(3, dim=1)[1]`

Tester 中的代码：
```python
if benchmark_writer is not None and benchmark_writer.topk > 1:
    pred = pred.topk(benchmark_writer.topk, dim=1)[1].data.cpu().numpy()
else:
    pred = pred.max(1)[1].data.cpu().numpy()
```

---

## 2. General Writer（通用输出格式）

### 2.1 配置方式

在运行时配置中添加 `writer` 字段：

```python
# configs/_base_/default_runtime.py 或具体实验配置
writer = dict(
    type="LASWriter",
    save_dir="output/las/",
    source_dir="data/raw/",  # 可选：使用源文件作为坐标模板
)
```

设置为 `None`（默认）则不启用通用 Writer。

### 2.2 Tester 中的使用

```python
# 自动从 cfg.writer 构建
general_writer = None
writer_cfg = getattr(self.cfg, "writer", None)
if writer_cfg is not None:
    general_writer = build_writer(writer_cfg)

# ... 推理后 ...
if general_writer is not None:
    general_writer.write(data_name, pred_sem=pred)
```

### 2.3 注意事项

- General Writer 与 Benchmark Writer **互不依赖**，可同时启用
- 当前 Tester 中 `coord` 不直接可用；如需 LASWriter 写入坐标，建议配置 `source_dir` 从源文件获取
- 未来可扩展数据集返回完整坐标以支持无源文件的直接写入

---

## 3. 修改前后对比

### 修改前（内联 if/elif 链）

```python
# 创建 submit 目录
if (self.cfg.data.test.type == "ScanNetDataset"
    or self.cfg.data.test.type == "ScanNet200Dataset"
    or self.cfg.data.test.type == "ScanNetPPDataset") and comm.is_main_process():
    make_dirs(os.path.join(save_path, "submit"))
elif self.cfg.data.test.type == "SemanticKITTIDataset" and comm.is_main_process():
    make_dirs(os.path.join(save_path, "submit"))
elif self.cfg.data.test.type == "NuScenesDataset" and comm.is_main_process():
    import json
    make_dirs(...)
    json.dump(...)

# ...推理后...
if (...== "ScanNetDataset" or ...== "ScanNet200Dataset"):
    np.savetxt(...)
elif ...== "ScanNetPPDataset":
    np.savetxt(...)
    pred = pred[:, 0]
elif ...== "SemanticKITTIDataset":
    submit = np.vectorize(...)(submit)
    submit.tofile(...)
elif ...== "NuScenesDataset":
    np.array(pred + 1).astype(np.uint8).tofile(...)

# ...finalize...
if ...== "S3DISDataset":
    torch.save(...)
```

### 修改后（Writer 系统）

```python
# 创建 Benchmark Writer（一行代替整个 if/elif 链）
benchmark_writer = create_benchmark_writer(
    dataset_type=self.cfg.data.test.type,
    save_dir=save_path,
    dataset=self.test_loader.dataset,
)
if benchmark_writer is not None and comm.is_main_process():
    benchmark_writer.setup()

# ...推理后...
if benchmark_writer is not None:
    benchmark_writer.write(data_name, pred)
    pred = benchmark_writer.pred_for_eval(pred)

# ...finalize...
if benchmark_writer is not None:
    benchmark_writer.finalize(intersection=intersection, union=union, target=target)
```

**优势：**
- 消除了 Tester 中所有数据集相关的条件分支
- 每个数据集的提交逻辑封装在独立类中，符合开闭原则
- 新增数据集只需：(1) 创建 Writer 子类 (2) 在 `_DATASET_TO_WRITER` 注册

---

## 4. 添加新数据集的 Benchmark Writer

### 步骤

1. **创建 Writer 文件** `pointspace/writers/benchmark/my_dataset_writer.py`：

```python
import os
import numpy as np
from .base_benchmark_writer import BaseBenchmarkWriter

class MyDatasetBenchmarkWriter(BaseBenchmarkWriter):
    topk = 1  # 或其他值

    def __init__(self, save_dir, dataset=None):
        super().__init__(save_dir, dataset)
        # 从 dataset 提取必要属性

    def setup(self):
        os.makedirs(os.path.join(self.save_dir, "submit"), exist_ok=True)

    def write(self, data_name, pred, **kwargs):
        # 写入提交文件
        ...

    # 可选重写：
    # def pred_for_eval(self, pred): ...
    # def finalize(self, **kwargs): ...
```

2. **注册到工厂** `pointspace/writers/benchmark/builder.py`：

```python
from .my_dataset_writer import MyDatasetBenchmarkWriter

_DATASET_TO_WRITER = {
    ...
    "MyDatasetType": MyDatasetBenchmarkWriter,  # 新增
}
```

3. **更新 `__init__.py`** 导出新类（可选）。

4. **编写测试** `tests/test_benchmark_writers.py` 中新增测试类。

---

## 5. 文件结构

```
pointspace/writers/
├── __init__.py              # 统一导出
├── builder.py               # WRITERS 注册表 + build_writer()
├── base_writer.py           # BaseWriter ABC (通用)
├── las_writer.py            # LASWriter (完整实现)
├── ply_writer.py            # PLYWriter (占位)
├── pcd_writer.py            # PCDWriter (占位)
└── benchmark/               # Benchmark 提交格式 Writer
    ├── __init__.py           # 统一导出
    ├── builder.py            # create_benchmark_writer() 工厂
    ├── base_benchmark_writer.py  # BaseBenchmarkWriter ABC
    ├── scannet_writer.py     # ScanNet / ScanNet200
    ├── scannetpp_writer.py   # ScanNet++ (topk=3)
    ├── semantic_kitti_writer.py  # SemanticKITTI (.label)
    ├── nuscenes_writer.py    # NuScenes (.bin + json)
    └── s3dis_writer.py       # S3DIS (.pth 6-fold)
```

## 6. 测试

```bash
# 运行所有 Writer 测试
python -m pytest tests/test_writers.py tests/test_benchmark_writers.py -v

# 仅运行 Benchmark Writer 测试
python -m pytest tests/test_benchmark_writers.py -v

# 运行特定数据集的 Writer 测试
python -m pytest tests/test_benchmark_writers.py -k "ScanNetPP" -v
```
