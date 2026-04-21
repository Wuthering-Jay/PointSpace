# Point Transformer V3m1 详解

本文聚焦 `pointspace/models/backbone/point_transformer_v3/point_transformer_v3m1_base.py`，说明它是如何处理点云、如何构造 patch、序列和多 order，并跟踪每一层中点结构的变化。

阅读目标：

- 看懂 `Point` 在进入网络后如何被序列化
- 看懂 `serialized_order / serialized_inverse / pad / unpad` 的作用
- 看懂 patch 是如何形成、如何在 attention 中使用
- 看懂 encoder / decoder 每层中点数、通道数、结构字段如何变化
- 看懂多 order 时每个 block 如何轮换使用不同序列顺序

## 本次新增内容

这次补充了两块更细的说明，你可以直接跳到对应章节：

- `4.5` Flash Attention 路径与普通路径的差异
- `7.2.1` 下采样时序列是怎么变粗的
- `7.2.2` pooling 后 patch 如何变化
- `8.3` 上采样时序列和 patch 的变化
- `8.4` decoder 中 patch 的语义

如果你之前没看到变化，主要是因为它们被放在原报告的中后段。下面这些章节就是这次补充的实际内容。

---

## 1. 这版 V3m1 的整体思路

V3m1 不是“直接在点集上做 attention”，而是先把点云转成一种可序列化的结构，再在序列 patch 上做 attention。

它的主路径可以概括为：

```text
data_dict
  -> Point
  -> serialization
  -> sparsify
  -> embedding
  -> encoder stages
  -> decoder stages
  -> Point(feat updated)
```

核心特点有 3 个：

1. `serialization` 把点按照空间顺序编码成多个 order
2. `SerializedAttention` 在每个 patch 上做局部自注意力
3. `SerializedPooling / SerializedUnpooling` 负责层级下采样与回填

---

## 2. 输入阶段：Point 和序列化字段

入口代码：

```python
def forward(self, data_dict):
    point = Point(data_dict)
    point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
    point.sparsify()

    point = self.embedding(point)
    point = self.enc(point)
    if not self.enc_mode:
        point = self.dec(point)
    return point
```

### 2.1 `Point(data_dict)` 后有什么

通常至少包含：

- `coord`: `[N, 3]`
- `feat`: `[N, C]`
- `offset`: `[B]`

如果是训练管线生成的样本，还可能包含：

- `grid_size`
- `condition`
- `context`
- `name`
- `split`
- `color`

### 2.2 `serialization()` 新增了什么

序列化后，`Point` 中会多出这些核心字段：

- `order`
- `serialized_depth`
- `serialized_code`
- `serialized_order`
- `serialized_inverse`

这一步是 V3m1 的关键。

它不是单一顺序，而是**多个 order 并存**。  
`self.order` 在构造时被设为：

```python
self.order = [order] if isinstance(order, str) else order
```

所以：

- 如果传的是字符串，比如 `"z"`，内部会变成 `["z"]`
- 如果传的是 tuple/list，比如 `("z", "z-trans")`，就会同时保留两个 order

这意味着一个 `Point` 不是只有一套排序，而是有多套序列视图。

---

## 3. serialization 的含义

`serialization()` 的本质是把空间坐标编码成整数 code，再按 code 排序，得到序列顺序。

在 `Point` 的结构代码里，`serialized_order` 和 `serialized_inverse` 的形状大致是：

```text
serialized_order   : [num_orders, N]
serialized_inverse : [num_orders, N]
serialized_code    : [num_orders, N]
```

对每个 order：

- `serialized_order[i]` 是第 `i` 种序列顺序
- `serialized_inverse[i]` 是它的逆映射

### 3.1 直观理解

如果某个 order 把点排序成：

```text
original points:   p0 p1 p2 p3 p4
serialized order:  p2 p0 p4 p1 p3
```

那么 `serialized_inverse` 就告诉你：

```text
serialized -> original
```

也就是把 attention 输出再映射回原点顺序。

---

## 4. patch 是怎么来的

V3m1 的 patch 不是显式 `knn` 或 `ball query`，而是来自序列化后的连续片段。

### 4.1 `SerializedAttention` 的 patch 生成

关键代码：

```python
pad, unpad, cu_seqlens = self.get_padding_and_inverse(point)

order = point.serialized_order[self.order_index][pad]
inverse = unpad[point.serialized_inverse[self.order_index]]

qkv = self.qkv(point.feat)[order]
```

这里的逻辑是：

1. 先对每个 batch 做 padding
2. 再按当前 order 把点排成连续 patch
3. 用 `patch_size` 把点集切成固定长度的序列块

### 4.2 `pad / unpad` 是什么

在 `get_padding_and_inverse()` 中：

- `pad` 表示 padding 后的新索引
- `unpad` 表示把 padded 序列恢复成原始点索引的映射

这一步主要解决两个问题：

1. 每个 batch 的点数不一定能被 `patch_size` 整除
2. `flash_attn` 需要连续的 patch 序列

### 4.3 patch_size 的意义

`patch_size` 是每个 attention patch 中的点数。

比如：

- `patch_size = 48`
- 某个 batch 经过序列化后有 960 个点

那么它会被切成：

```text
20 个 patch
每个 patch 48 个点
```

也就是说，attention 的计算单位不是整张点云，而是序列化之后的 patch。

### 4.4 patch 与序列的关系

对 V3m1 来说，patch 不是先做几何分组再排序，而是先有序列，再按固定长度切片。

可以把它理解为：

```text
空间点云
  -> 序列化 order
  -> 连续索引序列
  -> 每 K 个点组成一个 patch
```

所以 patch 的成员来自“序列邻接”，而序列邻接又来源于 `serialization()` 的空间编码。

---

## 4.5 Flash Attention 路径与普通路径的差异

`SerializedAttention` 支持两种运行模式：

- `enable_flash=True`
- `enable_flash=False`

这两条路径在 `qkv` 生成、attention 计算方式、mask/补齐方式上都不同。

### 4.5.1 `enable_flash=True` 时

代码中：

```python
feat = flash_attn.flash_attn_varlen_qkvpacked_func(
    qkv.to(torch.bfloat16).reshape(-1, 3, H, C // H),
    cu_seqlens,
    max_seqlen=self.patch_size,
    dropout_p=self.attn_drop if self.training else 0,
    softmax_scale=self.scale,
).reshape(-1, C)
```

这个路径的特点是：

1. `qkv` 被打包成 `[-1, 3, H, C/H]`
2. 使用 `cu_seqlens` 描述每个 patch 的起始位置
3. 由 flash attention kernel 直接做变长序列的注意力
4. 不显式构造 `[K, K]` 的 attention matrix

### 4.5.2 `enable_flash=False` 时

代码中：

```python
q, k, v = (
    qkv.reshape(-1, K, 3, H, C // H).permute(2, 0, 3, 1, 4).unbind(dim=0)
)
attn = (q * self.scale) @ k.transpose(-2, -1)
attn = self.softmax(attn)
feat = (attn @ v).transpose(1, 2).reshape(-1, C)
```

这个路径的特点是：

1. 显式构造 `q, k, v`
2. 显式构造 `attn: [num_patches, H, K, K]`
3. 可以插入 `RPE`
4. 更容易 debug

### 4.5.3 两条路径的本质区别

| 维度 | flash 路径 | 非 flash 路径 |
|---|---|---|
| attention 计算 | kernel 内完成 | Python/PyTorch 显式完成 |
| 是否显式构造 `attn` 矩阵 | 否 | 是 |
| 是否支持 RPE | 否，代码里直接禁用 | 是 |
| 数值精度 | 以 `bfloat16` 打包 | 由 PyTorch 张量控制 |
| 调试友好性 | 低 | 高 |
| 性能 | 更高 | 更低 |

### 4.5.4 为什么 flash 路径不能用 RPE

在构造里有显式约束：

```python
assert enable_rpe is False
assert upcast_attention is False
assert upcast_softmax is False
```

因为 flash attention 的 kernel 更适合固定的打包输入，不方便再插入额外的显式 RPE 加法。

### 4.5.5 对 patch 的影响

两条路径对 patch 的定义是一样的，都是依赖：

```python
patch_size = K
```

但实现形式不同：

- 非 flash：先 reshape 成 `[num_patches, K, ...]`
- flash：通过 `cu_seqlens` 告诉 kernel 每个 patch 的变长边界

所以 flash 并没有改变 patch 的数学定义，只是改变了实现方式。

---

## 5. SerializedAttention 的实际计算过程

代码片段：

```python
qkv = self.qkv(point.feat)[order]

q, k, v = (
    qkv.reshape(-1, K, 3, H, C // H).permute(2, 0, 3, 1, 4).unbind(dim=0)
)
attn = (q * self.scale) @ k.transpose(-2, -1)
feat = (attn @ v).transpose(1, 2).reshape(-1, C)
feat = feat[inverse]
point.feat = self.proj(feat)
```

### 5.1 张量形状变化

假设：

- `N` = 当前层点数
- `K` = `patch_size`
- `H` = `num_heads`
- `C` = channels

那么：

```text
point.feat             : [N, C]
qkv                    : [N, 3C]
q, k, v                : [num_patches, H, K, C/H]
attn                   : [num_patches, H, K, K]
feat                   : [N, C]
```

### 5.2 `serialized_inverse` 的作用

注意力是在 `order` 上算的，但输出需要回到原始点顺序。

所以最后：

```python
feat = feat[inverse]
```

这一步把 patch 顺序中的结果重新对齐到点云原始顺序。

### 5.3 `RPE` 的位置

如果启用了相对位置编码：

```python
attn = attn + self.rpe(self.get_rel_pos(point, order))
```

它对 patch 内的相对坐标进行编码，增强局部几何感知。

---

## 6. Block 的层内结构

核心 block：

```python
class Block(PointModule):
    def forward(self, point: Point):
        shortcut = point.feat
        point = self.cpe(point)
        point.feat = shortcut + point.feat
        shortcut = point.feat
        if self.pre_norm:
            point = self.norm1(point)
        point = self.drop_path(self.attn(point))
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm1(point)

        shortcut = point.feat
        if self.pre_norm:
            point = self.norm2(point)
        point = self.drop_path(self.mlp(point))
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm2(point)
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point
```

### 6.1 Block 内部流程

每个 block 都有三段：

1. `cpe`
2. `attention`
3. `mlp`

并且每段都有 residual。

### 6.2 `cpe` 是什么

`cpe` 的定义是：

```python
self.cpe = PointSequential(
    spconv.SubMConv3d(...),
    nn.Linear(channels, channels),
    norm_layer(channels),
)
```

这说明 block 不只是纯 attention，还引入了稀疏卷积上下文增强。

### 6.3 `pre_norm`

如果 `pre_norm=True`：

- 先 norm
- 再 attention / mlp

如果 `pre_norm=False`：

- 先计算
- 再 norm

这是 transformer 常见的 pre-norm / post-norm 选择。

---

## 7. Encoder 层级变化

编码器结构是：

- 第 0 stage：直接堆 `Block`
- 第 `s > 0` stage：先 `SerializedPooling`
- 再堆 `Block`

代码片段：

```python
for s in range(self.num_stages):
    enc = PointSequential()
    if s > 0:
        enc.add(
            SerializedPooling(
                in_channels=enc_channels[s - 1],
                out_channels=enc_channels[s],
                stride=stride[s - 1],
                ...
            ),
            name="down",
        )
    for i in range(enc_depths[s]):
        enc.add(Block(...), name=f"block{i}")
    self.enc.add(module=enc, name=f"enc{s}")
```

### 7.1 每层变化

如果记第 `s` 层输入为：

```text
P_s = [N_s, C_s]
```

那么：

- 第 0 stage：`N` 不变，`C` 变为 `enc_channels[0]`
- 第 `s > 0` stage：
  - 下采样后点数减少到 `N_{s+1}`
  - 通道数升到 `enc_channels[s]`
  - 然后 block 内保持 `N_{s+1}`、`C_{s+1}` 不变

### 7.2 `SerializedPooling` 的字段变化

Pooling 后点结构会变成新的 `Point`，同时保留：

- `pooling_inverse`
- `pooling_parent`

它还会更新：

- `coord`
- `feat`
- `grid_coord`
- `serialized_code`
- `serialized_order`
- `serialized_inverse`
- `serialized_depth`
- `batch`

也就是说，下采样不是单纯降采样，而是重新生成了一套更粗粒度的序列结构。

### 7.2.1 下采样时序列是怎么变粗的

`SerializedPooling` 最关键的一行是：

```python
code = point.serialized_code >> pooling_depth * 3
```

这表示它不是重新做一次完全新的序列化，而是把原来的序列 code 直接右移。

可以理解成：

```text
fine serialized code
  -> remove low bits
  -> coarse serialized code
```

因为一个 grid cell 的编码会占用若干 bit，所以右移相当于把空间分辨率降低。

### 7.2.2 pooling 后 patch 会怎样变化

假设进入 pooling 前：

```text
serialized_order : [O, N]
patch_size       : K
num_patches      : ceil(N / K)
```

pooling 后：

```text
serialized_order : [O, N']
patch_size       : 仍然是 K
num_patches      : 变成 ceil(N' / K)
```

注意：

- `patch_size` 本身没有变
- 变的是点数 `N'`
- 以及这些点对应的更粗粒度序列顺序

### 7.2.3 pooling 对 order 的影响

`SerializedPooling` 并不会删除 order 维度，而是对每个 order 同步生成：

```python
code = code[:, head_indices]
order = torch.argsort(code)
inverse = ...
```

这说明：

1. 多 order 在下采样后仍然保留
2. 只是每个 order 上的点集变少、排序更粗
3. 后续 attention 仍然可以按 `order_index` 取对应视图

### 7.3 `serialized_depth` 如何变化

在 pooling 里：

```python
point_dict["serialized_depth"] = point.serialized_depth - pooling_depth
```

所以随着层级加深：

- `serialized_depth` 会递减
- 表示当前点云对应的序列 cube 粒度变粗了

---

## 8. Decoder 层级变化

decoder 用 `SerializedUnpooling` 恢复点级特征。

代码片段：

```python
dec.add(
    SerializedUnpooling(
        in_channels=dec_channels[s + 1],
        skip_channels=enc_channels[s],
        out_channels=dec_channels[s],
        norm_layer=bn_layer,
        act_layer=act_layer,
    ),
    name="up",
)
```

### 8.1 Unpool 的流程

```python
parent = point.pop("pooling_parent")
inverse = point.pop("pooling_inverse")
point = self.proj(point)
parent = self.proj_skip(parent)
parent.feat = parent.feat + point.feat[inverse]
return parent
```

### 8.2 变化含义

假设当前 decoder 输入为：

```text
P_low = [N_low, C_low]
P_high = [N_high, C_high]
```

则 unpool 后：

- 点数从 `N_low` 恢复到 `N_high`
- 通道数被投影到 `out_channels`
- 特征通过 skip connection 与上采样结果融合

### 8.3 为什么这叫“回填”

因为它不是纯插值，而是：

1. 从低分辨率点集拿到特征
2. 通过 `inverse` 填回父层点
3. 再和 skip 分支融合

这和 point cloud segmentation 里常见的 decoder 非常类似。

### 8.3 上采样时序列和 patch 的变化

unpool 的核心不是重新构造新的 patch，而是把低分辨率 patch 的结果回填到高分辨率 parent 点上。

可以把它理解成：

```text
decoder 输入
  -> 低分辨率 patch 序列
  -> parent / inverse 回填
  -> 高分辨率点序列恢复
```

更具体地说：

1. 当前层的点已经是较粗粒度序列
2. `pooling_parent` 保存着上一级更细粒度点
3. `pooling_inverse` 告诉你当前低分辨率点对应 parent 中哪些位置
4. `point.feat[inverse]` 把低分辨率特征传播回高分辨率点

### 8.4 decoder 中 patch 的语义

decoder 里的 `patch_size` 没变，但点数变多了，所以：

- 一个 patch 的“长度”仍是 `K`
- 但 patch 数量增加
- 点在空间上也更细了

这意味着 decoder 的 attention 是在更高分辨率、更多 patch 上重新做局部建模。

### 8.5 形象化理解

如果 encoder 第 `s` 层是：

```text
N_s points -> group into patches -> attention -> downsample
```

那么 decoder 第 `s` 层是：

```text
coarse points -> unpool to finer points -> regroup into patches -> attention
```

所以 decoder 不是单纯“插值恢复”，而是“恢复后再做一次 patch attention 精炼”。

---

## 9. 多 order 情况是怎么工作的

这是 V3m1 很重要的一点。

### 9.1 order 在构造时就可以是多个

```python
self.order = [order] if isinstance(order, str) else order
```

例如：

```python
order = ("z", "z-trans")
```

那么：

```python
self.order = ["z", "z-trans"]
```

### 9.2 每个 block 轮换 order

在 encoder / decoder 里，block 的 `order_index` 这样设置：

```python
order_index=i % len(self.order)
```

这意味着：

- 第 0 个 block 用第 0 种 order
- 第 1 个 block 用第 1 种 order
- 第 2 个 block 又回到第 0 种 order
- 以此循环

### 9.3 多 order 的实际意义

不同 order 会给同一批点构造不同的序列邻接关系。

例如：

- `z` 可能更偏向一种 Z-order / space-filling 的空间遍历
- `z-trans` 可能是一个变体顺序，用来改变邻域的局部组合

多 order 的好处是：

1. 同一层内不只依赖一种序列视角
2. 模型更容易学到不同局部排列下的稳定表示
3. 让 patch attention 不是“单一排序偏置”

### 9.4 多 order 在数据结构中的表现

`Point` 中相关字段是按 order 维度保存的：

```text
serialized_code    : [O, N]
serialized_order   : [O, N]
serialized_inverse : [O, N]
```

其中 `O = len(self.order)`。

当某个 block 使用 `order_index=k` 时，它只取第 `k` 行：

```python
order = point.serialized_order[self.order_index][pad]
inverse = unpad[point.serialized_inverse[self.order_index]]
```

所以多 order 不是重复建模，而是**同一网络在不同层轮流切换序列视角**。

---

## 10. 一层一层看张量和字段怎么变

下面用一个抽象例子说明整个变化链。

### 10.1 输入

```text
coord  : [N, 3]
feat   : [N, Cin]
offset : [B]
```

### 10.2 serialization 后

新增字段：

```text
serialized_code    : [O, N]
serialized_order   : [O, N]
serialized_inverse : [O, N]
serialized_depth   : scalar
grid_coord         : [N, 3]
```

### 10.3 embedding 后

```text
feat : [N, C0]
```

其中 `C0 = enc_channels[0]`。

### 10.4 encoder stage 0

```text
N -> N
C0 -> C0
```

只做 block，不降采样。

### 10.5 encoder stage 1

先 pooling：

```text
N -> N1
C0 -> C1
serialized_depth 减少
grid_coord 更粗
```

再 block：

```text
N1 -> N1
C1 -> C1
```

### 10.6 encoder stage s

重复类似过程：

```text
N_s -> N_{s+1}
C_s -> C_{s+1}
```

### 10.7 decoder stage

先 unpool：

```text
N_low -> N_high
C_low -> Cout
```

再 block：

```text
N_high -> N_high
Cout -> Cout
```

### 10.8 最终输出

```text
point.feat = final decoder feature
```

---

## 11. 关键代码块索引

下面这些片段最值得结合原文件一起看。

### 11.1 `forward`

```python
point = Point(data_dict)
point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
point.sparsify()
point = self.embedding(point)
point = self.enc(point)
if not self.enc_mode:
    point = self.dec(point)
return point
```

### 11.2 多 order 入口

```python
self.order = [order] if isinstance(order, str) else order
...
order_index=i % len(self.order)
```

### 11.3 attention patch 形成

```python
pad, unpad, cu_seqlens = self.get_padding_and_inverse(point)
order = point.serialized_order[self.order_index][pad]
inverse = unpad[point.serialized_inverse[self.order_index]]
qkv = self.qkv(point.feat)[order]
```

### 11.4 pooling

```python
code = point.serialized_code >> pooling_depth * 3
...
point_dict["pooling_inverse"] = cluster
point_dict["pooling_parent"] = point
```

### 11.5 unpooling

```python
parent = point.pop("pooling_parent")
inverse = point.pop("pooling_inverse")
point = self.proj(point)
parent = self.proj_skip(parent)
parent.feat = parent.feat + point.feat[inverse]
```

---

## 12. 结论

V3m1 的核心不是“attention 本身”，而是“**序列化表示 + patch attention + serialized pooling/unpooling + 多 order 轮换**”这一整套结构。

你可以把它理解成：

- `serialization` 决定点云如何被切成 patch
- `patch_size` 决定每次 attention 的局部上下文范围
- `order` 决定 patch 的空间遍历方式
- `multi-order` 让不同层/不同 block 使用不同遍历视角
- `pooling/unpooling` 负责跨尺度结构变化

如果要继续深入，下一步最值得看的就是：

1. `get_padding_and_inverse()` 如何保证 patch 连续性
2. `serialized_code >> pooling_depth * 3` 为什么能实现层级下采样
3. `PointSequential` 如何把 `Point`、`spconv` 和普通 `nn.Module` 串起来
