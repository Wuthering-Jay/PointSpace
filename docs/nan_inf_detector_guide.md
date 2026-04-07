# NaN/Inf Detection Hook - Quick Reference

## 功能特性

✅ **精确定位**：追踪到具体哪一层产生NaN/Inf  
✅ **详细统计**：显示受影响值的数量和百分比  
✅ **灵活配置**：可选择抛异常、仅警告或静默记录  
✅ **输入检查**：可同时检查模块输入和输出  
✅ **条件启用**：支持动态启用/禁用  
✅ **递归注册**：自动注册到所有子模块  
✅ **配置集成**：可直接在配置文件中启用  

---

## 使用方式

### 方式1：配置文件集成（推荐）

在配置文件的 `hooks` 列表中添加：

```python
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="RuntimeInfoHook"),
    dict(type="ModelHook"),
    # NaN/Inf检测hook（按需启用）
    dict(type="NaNInfDetectorTrainerHook", 
         raise_on_nan=True,      # 检测到NaN时抛异常
         verbose=False,          # 只在有问题时打印
         check_interval=10),     # 每10步检查一次（节省性能）
    # ... 其他hooks ...
]
```

**配置参数**：
- `raise_on_nan: bool` - 检测到NaN时抛异常（默认False，不中断训练）
- `raise_on_inf: bool` - 检测到Inf时抛异常（默认False）
- `print_stats: bool` - 打印检测统计（默认True）
- `check_input: bool` - 同时检查输入（默认False）
- `verbose: bool` - 打印所有层信息（默认False）
- `check_interval: int` - 检查间隔步数（默认1，每步检查）
- `enabled_epochs: list` - 仅在特定epoch启用（默认None，所有epoch）

**使用示例**：
```python
# 示例1: 初次调试（立即停止）
dict(type="NaNInfDetectorTrainerHook", raise_on_nan=True, check_interval=1)

# 示例2: 定期检查（不中断训练）
dict(type="NaNInfDetectorTrainerHook", raise_on_nan=False, check_interval=100)

# 示例3: 只在前几个epoch检查
dict(type="NaNInfDetectorTrainerHook", enabled_epochs=[0, 1, 2])
```

---

### 方式2：独立使用

```python
from pointspace.engines.hooks import detect_nan_inf

# 注册到模型
detector = detect_nan_inf(model, raise_on_nan=True)

# 训练/推理
output = model(input)

# 移除hook
detector.remove()
```

---

## 快速开始

### 最简单用法
```python
from pointspace.engines.hooks import detect_nan_inf

# 注册到模型
detector = detect_nan_inf(model)

# 训练/推理
output = model(input)

# 移除hook
detector.remove()
```

### 完整配置
```python
from pointspace.engines.hooks import NaNInfDetectorHook

detector = NaNInfDetectorHook(
    raise_on_nan=True,      # 检测到NaN时抛出异常
    raise_on_inf=False,     # Inf时仅警告
    print_stats=True,       # 打印详细统计
    check_input=False,      # 不检查输入（仅检查输出）
    verbose=False,          # 仅在有问题时打印
)

detector.register(model)
# ... 训练 ...
detector.print_summary()
detector.remove()
```

---

## 使用场景

### 场景1：初次调试（找出第一个NaN）
```python
detector = NaNInfDetectorHook(
    raise_on_nan=True,      # 立即停止
    verbose=True,           # 打印所有层信息
)
detector.register(model)

try:
    output = model(input)
except RuntimeError:
    print(f"First NaN in: {detector.get_first_issue_module()}")
finally:
    detector.remove()
```

**配置文件方式**：
```python
dict(type="NaNInfDetectorTrainerHook", raise_on_nan=True, verbose=True)
```

---

### 场景2：长时间训练（定期检查）
```python
detector = NaNInfDetectorHook(
    raise_on_nan=False,     # 不中断训练
    enabled=False,          # 默认禁用
)
detector.register(model)

for step in range(10000):
    # 每100步检查一次
    detector.enabled = (step % 100 == 0)
    
    output = model(input)
    loss.backward()
    optimizer.step()

detector.print_summary()
detector.remove()
```

**配置文件方式**（更简单）：
```python
dict(type="NaNInfDetectorTrainerHook", raise_on_nan=False, check_interval=100)
```

---

### 场景3：针对性检查（只监控特定模块）
```python
detector = NaNInfDetectorHook()

# 只监控怀疑有问题的模块
detector.register(model.encoder)  # 只监控encoder
# detector.register(model.decoder)  # 不监控decoder

output = model(input)
detector.remove()
```

---

### 场景4：EZ-SP训练诊断
```python
from pointspace.engines.hooks import NaNInfDetectorHook

# 注册到所有关键模块
detector = NaNInfDetectorHook(raise_on_nan=True)
detector.register(sparse_cnn, prefix="cnn")
detector.register(partition_module, prefix="partition")

# 训练循环
for batch in dataloader:
    point_out = sparse_cnn(point)
    nag = partition_module(...)
    loss, _ = criterion(nag)
    # 如果有NaN会立即抛出异常并指出位置

detector.remove()
```

**配置文件方式**：
```python
# 在配置文件中启用
dict(type="NaNInfDetectorTrainerHook", raise_on_nan=True)
```

---

## 输出示例

### 正常情况（verbose=True）
```
[OK] Linear.0 (Linear) output: shape=(32, 20), range=[-2.3456, 3.1234], mean=0.1234
[OK] ReLU.1 (ReLU) output: shape=(32, 20), range=[0.0000, 3.1234], mean=0.5678
```

### 检测到NaN
```
[NaN DETECTED] Module: sparse_cnn.blocks.1 (SparseConvBlock)
  Stage: output
  Tensor index: 0
  Shape: (178602, 32)
  NaN count: 89301/5715264 (1.56%)
  dtype: torch.float32, device: cuda:0
```

### 汇总报告
```
================================================================================
NaN/Inf Detection Summary - 2 issues found
================================================================================

NaN Issues (2):
  - sparse_cnn.blocks.1 (SparseConvBlock): 89301/5715264 values, shape=(178602, 32)
  - partition.level_0 (PartitionModule): 12/10000 values, shape=(100, 100)

================================================================================
```

---

## API参考

### NaNInfDetectorHook类（独立使用）

**构造函数参数**：
- `raise_on_nan: bool` - 检测到NaN时抛异常（默认True）
- `raise_on_inf: bool` - 检测到Inf时抛异常（默认False）
- `print_stats: bool` - 打印检测统计（默认True）
- `check_input: bool` - 同时检查输入（默认False）
- `enabled: bool` - 是否激活（默认True）
- `verbose: bool` - 打印所有层信息（默认False）

**主要方法**：
- `register(model, prefix="")` - 注册到模型及其子模块
- `remove()` - 移除所有hook
- `reset_stats()` - 重置统计信息
- `print_summary()` - 打印检测汇总
- `get_first_issue_module()` - 获取第一个有问题的模块名

**属性**：
- `enabled` - 动态启用/禁用
- `detections` - 检测记录列表
- `hooks` - 已注册的hook列表

---

### NaNInfDetectorTrainerHook类（配置文件使用）

**构造函数参数**（在配置文件中作为dict参数）：
- `raise_on_nan: bool` - 检测到NaN时抛异常（默认False）
- `raise_on_inf: bool` - 检测到Inf时抛异常（默认False）
- `print_stats: bool` - 打印检测统计（默认True）
- `check_input: bool` - 同时检查输入（默认False）
- `verbose: bool` - 打印所有层信息（默认False）
- `check_interval: int` - 检查间隔步数（默认1）
- `enabled_epochs: list` - 仅在特定epoch启用（默认None）

**生命周期方法**（自动调用）：
- `before_train()` - 训练开始前注册检测器
- `before_step()` - 每步前控制启用状态
- `after_epoch()` - 每epoch后打印汇总
- `after_train()` - 训练结束后清理

---

## 注意事项

⚠️ **性能影响**：Hook会增加约10-20%计算开销，调试完成后记得禁用或移除  
⚠️ **内存占用**：`verbose=True`会打印大量日志  
⚠️ **多GPU**：自动处理分布式训练，会注册到正确的模块  
⚠️ **清理**：独立使用时务必调用`detector.remove()`；配置文件方式会自动清理  

---

## 测试脚本

```bash
# 基础示例
python tests/test_ezsp/test_nan_inf_detector.py

# 实际训练集成
python tests/test_ezsp/example_nan_detector_usage.py
```

---

## 与训练流程集成

### 方法1：配置文件（推荐）

在 `configs/dales/semseg-ezsp-v1-0.py` 中：

```python
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="RuntimeInfoHook"),
    # 添加NaN检测hook
    dict(type="NaNInfDetectorTrainerHook", 
         raise_on_nan=True, 
         check_interval=1),
    # ... 其他hooks ...
]
```

### 方法2：代码集成（如果需要更细粒度控制）

在`tools/train.py`中添加：

```python
from pointspace.engines.hooks import NaNInfDetectorHook

# 创建trainer后
if cfg.get('debug_nan', False):
    detector = NaNInfDetectorHook(
        raise_on_nan=True,
        verbose=cfg.get('debug_verbose', False),
    )
    detector.register(trainer.model)
    # 训练结束后会自动打印summary
```

在配置文件中启用：
```python
debug_nan = True  # 启用NaN检测
debug_verbose = False  # 不打印所有层
```

---

## 常见问题

### Q: 如何只在特定阶段启用？
A: 使用 `enabled_epochs` 参数：
```python
dict(type="NaNInfDetectorTrainerHook", enabled_epochs=[0, 1, 2])  # 只在前3个epoch检查
```

### Q: 如何降低性能影响？
A: 增加 `check_interval`：
```python
dict(type="NaNInfDetectorTrainerHook", check_interval=100)  # 每100步检查一次
```

### Q: 如何在训练中途启用？
A: 在配置文件中注释掉hook，训练时取消注释并重启即可。

### Q: 检测到NaN后如何定位问题？
A: 
1. 查看第一个出现NaN的模块名
2. 使用 `verbose=True` 查看该模块之前的所有层输出
3. 检查该模块的输入是否已经包含NaN（设置 `check_input=True`）

---

## 与AMP配合使用

```python
# 配置文件中
enable_amp = True
amp_dtype = "float16"

hooks = [
    # NaN检测会自动处理AMP环境
    dict(type="NaNInfDetectorTrainerHook", raise_on_nan=True),
    # ...
]
```

检测器会正确处理混合精度训练中的tensor类型转换。

