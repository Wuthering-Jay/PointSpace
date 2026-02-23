# Writer 系统设计文档

## 概述

`WRITERS` 注册表解耦了 Tester 的推理逻辑和结果保存逻辑。推理完成后，Tester 可以将坐标和预测结果传给任意 Writer，由 Writer 负责序列化为目标格式。

## 架构

```
pointspace/writers/
├── __init__.py        # 汇总导出
├── builder.py         # WRITERS 注册表 + build_writer()
├── base_writer.py     # BaseWriter 抽象基类
├── las_writer.py      # LASWriter（完整实现）
├── ply_writer.py      # PLYWriter（占位符）
└── pcd_writer.py      # PCDWriter（占位符）
```

### 类层次

```
BaseWriter (ABC)
├── LASWriter      ← 完整实现
├── PLYWriter      ← NotImplementedError
└── PCDWriter      ← NotImplementedError
```

## 核心接口

```python
class BaseWriter(ABC):
    def __init__(self, save_dir: str): ...

    @abstractmethod
    def write(self, data_name: str, coord: np.ndarray, **kwargs) -> str:
        """
        Args:
            data_name: 场景名称（不含扩展名）
            coord: 点坐标 (N, 3)
            **kwargs:
                pred_sem       语义分割标签 (N,)
                pred_ins       实例分割 ID (N,)
                pred_panoptic  全景分割标签（预留）
                pred_bbox      3D 检测框（预留）
                pred_reg       回归值（预留）
                color          RGB 颜色 (N, 3)
                extra_dims     任意自定义维度 {name: (data, dtype)}
        Returns:
            str: 输出文件的完整路径
        """
```

## 使用方式

### 在代码中使用

```python
from pointspace.writers import build_writer

# 构建 Writer
writer = build_writer(dict(
    type="LASWriter",
    save_dir="output/results/",
    source_dir="data/raw/",       # 可选：保留原始 LAS 头信息和维度
    compressed=True,              # 可选：输出 .laz
))

# 写入语义分割结果
writer.write("scene_001", coord, pred_sem=pred_labels)

# 写入语义 + 实例分割结果
writer.write("scene_001", coord, pred_sem=pred_labels, pred_ins=instance_ids)

# 写入自定义维度
writer.write("scene_001", coord, extra_dims={
    "confidence": (conf_array, np.float32),
    "height_above_ground": (height_array, np.float64),
})
```

### 未来在配置文件中使用（设想）

```python
# configs/my_experiment.py
test = dict(
    type="SemSegTester",
    verbose=True,
    writer=dict(
        type="LASWriter",
        save_dir="exp/my_experiment/las_output/",
        source_dir="data/scannet/raw/",
    ),
)
```

## LASWriter 详细说明

### 两种工作模式

| 模式 | 触发条件 | 行为 |
|------|---------|------|
| 有源文件 | `source_dir` 不为 None 且找到同名文件 | 读取原始 LAS/LAZ 文件，保留所有头信息、元数据和已有维度，仅覆写/追加推理结果字段 |
| 无源文件 | `source_dir` 为 None，或未找到文件（fallback） | 从零创建 point_format=2 的 LAS 文件，自动计算 scale/offset |

### 任务字段映射

| kwargs key | LAS 字段 | 数据类型 | 状态 |
|-----------|----------|---------|------|
| `pred_sem` | `classification` | uint8 | ✅ 已实现 |
| `pred_ins` | `instance_id` (ExtraBytes) | int32 | ✅ 已实现 |
| `pred_panoptic` | - | - | 🔲 预留 |
| `pred_bbox` | - | - | 🔲 预留 |
| `pred_reg` | - | - | 🔲 预留 |
| `extra_dims` | 任意 ExtraBytes | 任意 | ✅ 已实现 |

### 异常处理

- **源文件缺失**：发出 `RuntimeWarning` 并自动回退到无源文件模式
- **源文件读取失败**：发出 `RuntimeWarning` 并回退
- **点数不匹配**：抛出 `ValueError`
- **laspy 未安装**：实例化时抛出 `ImportError`

## 扩展指南

### 添加新格式

1. 在 `pointspace/writers/` 下创建新文件（如 `e57_writer.py`）
2. 继承 `BaseWriter`，实现 `write()` 方法
3. 使用 `@WRITERS.register_module()` 装饰器注册
4. 在 `__init__.py` 中导入

```python
# e57_writer.py
from .builder import WRITERS
from .base_writer import BaseWriter

@WRITERS.register_module()
class E57Writer(BaseWriter):
    def write(self, data_name, coord, **kwargs):
        ...
```

### 添加新任务

在已有 Writer 的 `_apply_predictions()` 方法中添加新的 elif 分支即可，无需修改 `write()` 的公开接口。

## 测试

```bash
conda activate pointcept
python -m pytest tests/test_writers.py -v
```

测试覆盖 20 个用例：

| 测试类 | 覆盖内容 |
|--------|---------|
| `TestWriterRegistry` | 注册表注册、build_writer 构建、未知类型报错 |
| `TestLASWriterCreateMode` | 纯坐标写入、RGB 颜色、LAZ 压缩 |
| `TestLASWriterSemSeg` | classification 字段写入 |
| `TestLASWriterInsSeg` | instance_id ExtraBytes 写入 |
| `TestLASWriterMultiTask` | 语义 + 实例联合写入、extra_dims |
| `TestLASWriterSourceMode` | 有源文件模式保留原始数据、缺失文件回退 |
| `TestLASWriterValidation` | 点数不匹配校验 |
| `TestPlaceholderWriters` | PLY/PCD 占位符抛出 NotImplementedError |
| `TestBaseWriterAbstract` | 抽象类不可实例化、子类约束 |

## 涉及文件

| 文件 | 说明 |
|------|------|
| `pointspace/writers/__init__.py` | 包初始化与导出 |
| `pointspace/writers/builder.py` | WRITERS 注册表 + build_writer |
| `pointspace/writers/base_writer.py` | BaseWriter 抽象基类 |
| `pointspace/writers/las_writer.py` | LASWriter 完整实现 |
| `pointspace/writers/ply_writer.py` | PLYWriter 占位符 |
| `pointspace/writers/pcd_writer.py` | PCDWriter 占位符 |
| `tests/test_writers.py` | 测试文件（20 个用例） |
| `tests/doc/writer_system.md` | 本文档 |
