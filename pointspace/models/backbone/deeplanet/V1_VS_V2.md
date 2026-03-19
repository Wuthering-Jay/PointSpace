# DeepLANet V1 vs V2 设计对比

## 架构对比图

```
┌─────────────────────────────────────────────────────────────┐
│                    Point Transformer V2                      │
│  Block: fc1 -> GroupedVectorAttention -> fc3 (skip)         │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ 替代
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      DeepLANet V1                            │
│  Block: fc1 -> LocalFeatureAggregation -> fc3 (skip)        │
│                                                               │
│  LFA 内部:                                                    │
│    mlp1: d_in -> d_out//2                                    │
│    lse1 + pool1: [n,k,10] -> [n,k,2d] -> [n,d//2]          │
│    lse2 + pool2: [n,k,10] -> [n,k,2d] -> [n,d]             │
│    mlp2: d -> 2*d (维度翻倍)                                 │
│    输出: 2*d_out                                              │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ 轻量化
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      DeepLANet V2 (推荐)                     │
│  ResLFE Block: Front-Linear -> VFR -> Post-Linear (skip)    │
│                                                               │
│  ResLFE 内部:                                                 │
│    Front-Linear: d -> d (先投影，后 grouping)                │
│    PE Fusion: feat + pe (加法，不翻倍)                        │
│    VFR: rel_feat = f_j - f_i, max_pool(rel_feat)            │
│    Post-Linear: d -> d (维度保持)                            │
│    输出: d (维度不变)                                          │
└─────────────────────────────────────────────────────────────┘
```

## 核心差异

### 1. 局部聚合模块

| 特性 | V1: LocalFeatureAggregation | V2: VFRModule |
|------|----------------------------|---------------|
| **设计** | 复杂，两轮编码+池化 | 极简，仅边缘特征+max pooling |
| **参数** | 多层 MLP (10->d, d->2d) | **无参数** |
| **计算** | LSE + AttentivePooling × 2 | `f_j - f_i` + max |
| **输出维度** | 2*d_out (翻倍) | d (保持) |
| **复杂度** | O(N·K·d²) | **O(N·K·d)** |

### 2. 位置编码融合方式

| 方面 | V1 | V2 |
|------|----|----|
| **PE 形式** | [n, k, 10] 原始编码 | [n, C] 编码后特征 |
| **融合方式** | Concatenate [spatial, feat] | **Addition: feat + pe** |
| **维度影响** | [n, k, d] -> [n, k, 2d] | **[n, d] + [n, d] = [n, d]** |
| **优势** | 信息丰富 | **更高效，显存友好** |

### 3. Block 结构

#### V1: Block
```python
def forward(points, pos_encoding, reference_index):
    identity = feat
    feat = act(norm1(fc1(feat)))              # [n, c]
    feat = lfa(coord, feat, pos_encoding, ri) # [n, c], LFA 内部复杂
    feat = act(norm2(feat))
    feat = norm3(fc3(feat))
    feat = identity + drop_path(feat)
    return [coord, feat, offset]
```

#### V2: ResLFEBlock
```python
def forward(points, pe, reference_index):
    identity = feat
    feat = act(norm1(fc1(feat)))     # Front-Linear
    feat = feat + pe                  # PE 融合 (加法)
    feat = vfr(feat, coord, ri)       # VFR (极轻量)
    feat = norm2(fc2(feat))           # Post-Linear
    feat = identity + drop_path(feat) # 残差
    return [coord, feat, offset]
```

### 4. BlockSequence

#### V1
```python
# Stage 级别计算位置编码
pos_encoding = compute_stage_positional_encoding(...)  # [n, k, 10]
# 直接传递给 Block
for block in blocks:
    points = block(points, pos_encoding, reference_index)
```

#### V2
```python
# Stage 级别计算位置编码
pos_encoding = compute_stage_positional_encoding(...)  # [n, k, 10]
# 编码为点级特征
pe = pe_encoder(pos_encoding)  # [n, k, 10] -> [n, C]
# 传递给 Block
for block in blocks:
    points = block(points, pe, reference_index)
```

## 计算复杂度分析

假设:
- 点数: N
- 邻域大小: K
- 特征维度: d
- Block 数量: B (per stage)

### V1: LocalFeatureAggregation

```
1. PE 计算 (Stage级别): O(N·K)          [只计算1次] ✓
2. mlp1: O(N·d·d)
3. lse1: O(N·K·10·d) + O(N·K·d)
4. pool1 (attention): O(N·K·d²)
5. lse2: O(N·K·10·d) + O(N·K·d)
6. pool2 (attention): O(N·K·d²)
7. mlp2: O(N·d²)
8. shortcut: O(N·d²)

Per Block: O(N·K·d²)
Per Stage (B blocks): O(B·N·K·d²)
```

### V2: VFRModule + ResLFEBlock

```
1. PE 计算 (Stage级别): O(N·K)          [只计算1次] ✓
2. PE encoder: O(N·K·10·d) + O(N·K·d)   [只计算1次] ✓
3. fc1: O(N·d²)
4. PE fusion: O(N·d)                     [加法，几乎0成本]
5. VFR grouping: O(N·K·d)
6. VFR max: O(N·K·d)                     [无参数，极快]
7. fc2: O(N·d²)

Per Block: O(N·d²)  [主要是 Linear]
Per Stage (B blocks): O(B·N·d²)  [不依赖 K!]
```

### 复杂度对比

| 操作 | V1 | V2 | 加速比 |
|------|----|----|--------|
| **Stage开销** | O(B·N·K·d²) | O(B·N·d²) | **K倍** |
| **邻域依赖** | 强 (attention) | 弱 (max pool) | - |
| **显存占用** | [n,k,2d] 中间结果 | [n,d] 中间结果 | **2K倍** |

## 参数量对比

假设 `d = 96`, 一个 Block 的参数量:

### V1 Block
```
fc1: d×d = 9,216
LFA:
  - mlp1: d×(d//2) = 4,608
  - lse1.mlp: 10×(d//2) = 480
  - pool1.score: (d)×(d) = 9,216
  - pool1.mlp: d×(d//2) = 4,608
  - lse2.mlp: 10×(d//2) = 480
  - pool2.score: d×d = 9,216
  - pool2.mlp: d×d = 9,216
  - mlp2: d×2d = 18,432
  - shortcut: d×2d = 18,432
fc3: d×d = 9,216

Total: ~92,920 参数
```

### V2 Block
```
fc1: d×d = 9,216
VFR: 0 参数 ✓
fc2: d×d = 9,216

Total: ~18,432 参数
```

**参数减少**: `(92,920 - 18,432) / 92,920 = 80%` 🎉

注: PE encoder 在 Stage 级别共享，不计入单个 Block。

## 何时使用哪个版本?

### 使用 V1 (DeepLANet-v1)
- ✓ 需要最高精度
- ✓ 计算资源充足
- ✓ 对显存和速度不敏感
- ✓ 需要更强的特征表达能力

### 使用 V2 (DeepLANet-v2) ⭐ 推荐
- ✓ 需要高效推理
- ✓ 显存/算力有限
- ✓ 精度要求适中
- ✓ 大规模点云或实时应用
- ✓ **默认首选**

## 总结

| 维度 | 获胜者 |
|------|--------|
| 精度 | V1 (略优) |
| 速度 | **V2 (显著优)** |
| 参数量 | **V2 (显著少)** |
| 显存 | **V2 (显著少)** |
| 代码复杂度 | **V2 (更简洁)** |
| **综合推荐** | **V2** ⭐ |

V2 通过极简的 VFR 设计和高效的 PE 融合方式，在保持合理精度的同时，大幅提升了计算效率和显存效率，是工程实践的首选。
