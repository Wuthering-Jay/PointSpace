# AMP性能问题分析

## 🚨 问题现象

用户报告：
- **AMP ON** (enable_amp=True) + SparseCNN FP32强制：更慢、更耗内存
- **AMP OFF** (enable_amp=False) + 纯FP32：更快、更省内存

这与预期相反！

## 🔍 根本原因

### dtype转换开销

```
模块流程（AMP ON）：
Input (FP16) 
  ↓
SparseCNN (强制FP32)
  ↓ 输出FP32
  转换 → FP16 (开销!)
  ↓
Partition (FP16环境)
  ↓ 
某些操作需要FP32
  转换 → FP32 (开销!)
  ↓
继续...
```

vs

```
模块流程（AMP OFF）：
Input (FP32)
  ↓
SparseCNN (FP32)
  ↓
Partition (FP32)
  ↓
一切都是FP32，无转换
```

### 内存开销

```
AMP ON:
- FP32 tensors in SparseCNN
- FP16 tensors in other modules
- Conversion buffers
- 两套数据共存
→ 总内存反而更高！

AMP OFF:
- 只有FP32 tensors
- 无conversion buffers
- 一套数据
→ 更少内存
```

### Tensor Cores利用率

- **AMP ON**：FP32/FP16混合，无法充分利用tensor cores
- **AMP OFF**：纯FP32，CUDA调度更高效

## ✅ 解决方案

### 推荐：使用 AMP OFF

```python
# configs/dales/semseg-ezsp-v1-0.py
enable_amp = False  # 纯FP32，最佳性能
amp_dtype = "float16"  # 被忽略
```

**理由**：
1. ✅ 更快（无dtype转换）
2. ✅ 更省内存（无冗余数据）
3. ✅ 训练和验证一致
4. ✅ 无spconv算法问题（FP32支持完整）

### 何时使用 AMP ON？

只在以下情况：
- 超大模型，内存极度紧张
- 且验证数据尺寸与训练完全一致
- 且愿意接受10-20%速度损失

对于DALES Stage 1训练：
- 数据稀疏，内存不是瓶颈
- **推荐 AMP OFF**

## 📊 性能对比（预期）

| 配置 | 速度 | 内存 | 稳定性 | 推荐度 |
|------|------|------|--------|--------|
| AMP OFF (纯FP32) | 快 | 中 | ⭐⭐⭐⭐⭐ | ✅ **推荐** |
| AMP ON + SparseCNN FP32 | 慢 | 高 | ⭐⭐⭐⭐ | ❌ |
| AMP ON + 纯FP16 | 最快 | 低 | ⭐⭐ (eval失败) | ❌ |

## 🧪 验证性能

运行benchmark：
```bash
python tests/test_ezsp/benchmark_amp_configs.py
```

这会对比：
1. AMP OFF (纯FP32)
2. AMP ON + SparseCNN FP32
3. AMP ON + 纯FP16 (可能eval失败)

## 🎯 最终建议

### 当前最佳配置

```python
enable_amp = False  # 禁用AMP
```

**效果**：
- ✅ 最快的训练速度
- ✅ 最少的内存使用
- ✅ 训练/验证完全稳定
- ✅ 无spconv问题

### GraphNorm保护仍然有效

即使 `enable_amp=False`，GraphNorm中的fp32转换仍然起作用：
```python
# graph_norm.py
x_fp32 = x.float()  # 提升数值稳定性
# ... 归一化 ...
return x_norm.to(x.dtype)
```

这提供了归一化的数值稳定性，无需全局AMP。

## 📝 更新配置

已更新 `configs/dales/semseg-ezsp-v1-0.py`：
```python
# Option A: AMP OFF - RECOMMENDED
enable_amp = False

# Option B: AMP ON (commented out)
# enable_amp = True  # 如果内存极度紧张
```

## 🎓 经验总结

1. **混合精度不总是更快**
   - dtype转换有开销
   - 内存管理有开销
   - 需要实测验证

2. **稀疏计算的特殊性**
   - 已经很省内存
   - AMP收益有限
   - 纯FP32可能更优

3. **稳定性 > 理论优化**
   - 能跑 > 理论快一点
   - 实测 > 假设

4. **简单即美**
   - 纯FP32：简单、稳定、快
   - 混合精度：复杂、可能更慢
   - 从简单开始！
