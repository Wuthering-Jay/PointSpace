# DALES EZ-SP Configuration Fixes - Complete Summary

## 修复的问题列表

### 1. ✅ 特征维度不匹配 (RuntimeError: 159221x4 vs 5x32)
**问题**: 配置设置 `in_channels=5` (coord+echo)，但实际数据只有4维 (coord+intensity)

**修复**:
```python
# configs/dales/semseg-ezsp-v1-0.py
feature_keys = ["coord", "intensity"]
in_channels = 4  # coord(3) + intensity(1)
```

---

### 2. ✅ stuff_classes 参数混淆
**问题**: `stuff_classes` 用于实例分割，在纯语义分割中无用

**修复**: 注释掉此参数并添加说明
```python
# NOTE: stuff_classes is for instance segmentation (future use), not used in semantic seg
# stuff_classes = [0, 1]
```

---

### 3. ✅ ignore_index 不一致
**问题**: train配置缺少 `ignore_index` 参数，导致使用默认值 -1，而val/test使用8

**修复**:
```python
# configs/dales/semseg-ezsp-v1-0.py
train=dict(
    ...
    ignore_index=ignore_index,  # CRITICAL: must match EZ-SP convention (num_classes)
    ...
)
```

---

### 4. ✅ 类别映射逻辑错误
**问题**: 原始逻辑在 `ignore_index >= 0` 时从1开始映射，导致映射后的类别不是 [0, 1, ..., n-1]

**修复**:
```python
# pointspace/datasets/las.py
# 修改前
start_id = 0 if self.ignore_index < 0 else 1
self.class2id = {c: start_id + i for i, c in enumerate(valid_classes)}

# 修改后
self.class2id = {c: i for i, c in enumerate(valid_classes)}  # 始终从0开始
```

**流程**:
1. **Filter**: 筛选有效类别 [1,2,3,4,5,6,7,8]，移除类别 [0]
2. **Remap**: 将有效类映射到 [0,1,2,3,4,5,6,7]
3. **Apply**: 数据加载时，有效类用新ID，移除的类用 ignore_index=8

---

### 5. ✅ grid_size 未传递到模型
**问题**: GridSample设置了 `grid_size`，但Collect没有收集它

**修复**:
```python
# configs/dales/semseg-ezsp-v1-0.py - 所有数据集
post_transform=[
    dict(type="ToTensor"),
    dict(
        type="Collect",
        keys=["coord", "segment", "grid_size"],  # 添加 grid_size
        feat_keys=feature_keys,
    ),
],
```

---

### 6. ✅ grid_size 在 GridSample 中未设置
**问题**: GridSample transform 没有将 `grid_size` 存入 data_dict

**修复**:
```python
# pointspace/datasets/transform.py - GridSample (train mode)
data_dict = index_operator(data_dict, idx_unique)
data_dict["grid_size"] = self.grid_size  # 添加这行

# GridSample (test mode)
data_part = index_operator(data_dict, batch_indices, duplicate=True)
data_part["index"] = batch_indices
data_part["grid_size"] = self.grid_size  # 添加这行

# GridSample_Maxloop 同样处理
```

---

### 7. ✅ grid_size 维度处理错误
**问题**: ToTensor 将 float 转为 `FloatTensor([x])`，导致形状不匹配

**修复**:
```python
# pointspace/models/utils/structure.py - Point.sparsify()
grid_size = self.grid_size
if isinstance(grid_size, torch.Tensor):
    grid_size = grid_size.item() if grid_size.numel() == 1 else grid_size
```

---

### 8. ✅ **collate_fn 错误地拼接 grid_size** (关键修复!)
**问题**: batch collate时，`grid_size` 被 `torch.cat` 拼接，从 `[1]` 变成 `[2]`（batch_size=2）

**原因**: `collate_fn` 对所有tensor key都调用 `torch.cat`

**修复**:
```python
# pointspace/datasets/utils.py - collate_fn
batch = {
    key: (
        (
            collate_fn([d[key] for d in batch])
            if "offset" not in key and key != "grid_size"  # 关键: grid_size不拼接
            else (
                torch.cumsum(...)  # offset处理
                if "offset" in key
                else batch[0][key]  # grid_size: 使用第一个样本的值
            )
        )
        ...
    )
    for key in batch[0]
}
```

---

### 9. ✅ 模型输出值类型错误
**问题**: `partition_output.get()` 返回 Python int，但hook期望tensor

**修复方案1** (在模型中):
```python
# pointspace/models/segmentor/ezsp_segmentor.py
return {
    "loss": loss,
    "n_inter_edge": torch.tensor(partition_output.get("n_inter_edge", 0), dtype=torch.float32),
    ...
}
```

**修复方案2** (在hook中，更健壮):
```python
# pointspace/engines/hooks/misc.py
for key in self.model_output_keys:
    value = model_output_dict[key]
    if isinstance(value, torch.Tensor):
        self.trainer.storage.put_scalar(key, value.item())
    else:
        self.trainer.storage.put_scalar(key, float(value))
```

---

### 10. ✅ 日志输出优化
**问题**: 类别映射的日志顺序容易引起误解

**修复**: 优化日志，清晰展示3步流程
```
✓ Auto class remapping ENABLED (EZ-SP compatible):

  Step 1 - Filter classes:
    All classes found: [0, 1, 2, 3, 4, 5, 6, 7, 8]
    Keep: [1, 2, 3, 4, 5, 6, 7, 8]
    Remove: [0]

  Step 2 - Remap valid classes to continuous [0, 1, ..., 7]:
    Original classes: [1, 2, 3, 4, 5, 6, 7, 8]
    Remapped to:      [0, 1, 2, 3, 4, 5, 6, 7]
    Mapping: {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7}

  Step 3 - Data loading will apply mapping:
    Valid classes (e.g., 1) → Remapped ID (e.g., 0)
    Removed classes [0] → ignore_index=8

  ✓ VERIFIED: ignore_index=8 != any remapped class [0-7] (no conflict)
```

---

## 修改的文件总结

1. **configs/dales/semseg-ezsp-v1-0.py**
   - 修正 `in_channels=4`, `feature_keys`
   - 注释 `stuff_classes`
   - train配置添加 `ignore_index`
   - 所有Collect添加 `"grid_size"`

2. **pointspace/datasets/las.py**
   - 修正类别映射逻辑（始终从0开始）
   - 优化日志输出

3. **pointspace/datasets/transform.py**
   - GridSample: 设置 `data_dict["grid_size"]`
   - GridSample_Maxloop: 同样处理

4. **pointspace/models/utils/structure.py**
   - Point.sparsify(): 处理tensor grid_size (`.item()`)

5. **pointspace/datasets/utils.py**
   - collate_fn: 防止 grid_size 被 concatenate

6. **pointspace/models/segmentor/ezsp_segmentor.py**
   - 将输出值转为tensor

7. **pointspace/engines/hooks/misc.py**
   - 处理scalar和tensor输出值

---

## 验证测试

创建的测试文件：
- `tests/test_ezsp/test_dales_config.py` - 配置验证
- `tests/test_ezsp/test_las_dataset_fixes.py` - 数据集修复验证
- `tests/test_ezsp/test_grid_size_propagation.py` - grid_size传播测试
- `tests/test_ezsp/test_collate_grid_size.py` - collate_fn测试
- `tests/test_ezsp/demo_class_remapping.py` - 类别映射演示

---

## 最终状态

✅ 所有配置问题已修复
✅ 类别映射逻辑正确 ([0-7] with ignore_index=8)
✅ grid_size 正确传递和处理
✅ batch collate 正确处理 grid_size
✅ 模型输出值类型正确
✅ 日志清晰明确

**训练现在应该可以正常启动！** 🎉
