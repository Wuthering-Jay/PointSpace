# Batched Fragment Inference for SemSegTester

## 背景

`SemSegTester` 用于大场景点云语义分割推理。由于大场景点云无法一次性塞入显存，数据集会将每个场景预先切分为多个 **fragment（碎片）**，通过滑动窗口分块推理，最后将碎片预测拼接（merge）回原始点云：

```python
pred[idx_part[bs:be], :] += pred_part[bs:be]
```

原始实现中，`fragment_batch_size` 被硬编码为 `1`，即每次只推理一个 fragment。这对于显存充裕的场景而言效率较低。

## 改动内容

### 1. 新增配置项 `fragment_batch_size`

**文件**: `configs/_base_/default_runtime.py`

```python
fragment_batch_size = 1  # batch size for fragment inference in SemSegTester (>1 to speed up)
```

- 默认值为 `1`，与原始行为完全一致（向后兼容）。
- 设置为 `> 1` 时，每次推理多个 fragment，提高 GPU 利用率。

### 2. 修改 `SemSegTester.test()` 和 `DINOSemSegTester.test()`

**文件**: `pointspace/engines/test.py`

核心改动：

```python
# 之前（硬编码 bs=1）
for i in range(len(fragment_list)):
    fragment_batch_size = 1
    s_i, e_i = i * fragment_batch_size, min(...)

# 之后（可配置 bs）
fragment_batch_size = getattr(self.cfg, "fragment_batch_size", 1)
num_fragments = len(fragment_list)
batch_num = int(np.ceil(num_fragments / fragment_batch_size))
for i in range(batch_num):
    s_i = i * fragment_batch_size
    e_i = min((i + 1) * fragment_batch_size, num_fragments)
```

### 3. 日志格式更新

日志从按每个 fragment 输出改为按每个 batch 输出，显示累计处理的 fragment 数：

```
# 之前（fragment_batch_size=1，15个fragment）
Test: 1/10-scene0001, Batch: 0/15
Test: 1/10-scene0001, Batch: 1/15
...
Test: 1/10-scene0001, Batch: 14/15

# 之后（fragment_batch_size=4，15个fragment）
Test: 1/10-scene0001, Fragment: 4/15
Test: 1/10-scene0001, Fragment: 8/15
Test: 1/10-scene0001, Fragment: 12/15
Test: 1/10-scene0001, Fragment: 15/15
```

## 使用方式

在你的配置文件中设置 `fragment_batch_size`：

```python
# 例如在 configs/scannet/semseg-... 配置中
fragment_batch_size = 4  # 每次推理4个fragment
```

或者在命令行中覆盖：

```bash
python tools/test.py --config-file configs/xxx.py --options fragment_batch_size=4
```

## 向后兼容性

- `fragment_batch_size` 通过 `getattr(self.cfg, "fragment_batch_size", 1)` 获取，旧配置无需修改。
- 当 `fragment_batch_size=1` 时，行为与原始代码完全一致。

## 测试

测试文件: `tests/test_fragment_batch.py`

运行：

```bash
conda activate pointcept
python -m pytest tests/test_fragment_batch.py -v
```

测试覆盖：

| 测试类 | 测试内容 |
|--------|---------|
| `TestFragmentBatchPartitioning` | 批次划分逻辑：整除、余数、边界情况、全覆盖 |
| `TestPredictionAccumulation` | bs=1 与 bs>1 的预测结果数值一致性 |
| `TestLoggingFormat` | 日志格式正确（累计 fragment 数 / 总数） |
| `TestGetAttrFallback` | 配置缺失时默认值为 1 |

## 涉及文件

| 文件 | 类型 |
|------|------|
| `configs/_base_/default_runtime.py` | 新增配置项 |
| `pointspace/engines/test.py` | 修改 `SemSegTester` 和 `DINOSemSegTester` |
| `tests/test_fragment_batch.py` | 新增测试 |
| `tests/doc/batched_fragment_inference.md` | 本文档 |
