# LitePT-v1m1 相对 PTv3 的改进分析

本文基于当前仓库中的实现进行分析，主要参考：

- `pointspace/models/backbone/litept/litept_v1m1.py`
- `pointspace/models/backbone/point_transformer_v3/point_transformer_v3m1_base.py`
- `libs/pointrope/setup.py`
- `libs/pointrope/pointrope.cpp`
- `libs/pointrope/kernels.cu`

## 1. LitePT 是什么

LitePT 可以看作是 PTv3 的一个轻量化变体，但它不是简单地“把通道数缩小”。它更像是围绕如下目标重新裁剪了一版骨干：

- 尽量把注意力只放在真正需要的高层 stage
- 低层更多依赖稀疏卷积或局部卷积建模
- 用更便宜的 3D RoPE 替代更重的相对位置编码设计
- 通过更激进的结构开关减少 decoder 和中低层 attention 的成本

因此，LitePT 的轻量化是“结构性裁剪 + 位置编码替换 + stage 功能重分配”的组合，而不是单一缩放。

## 2. LitePT 相对 PTv3 的主要改进

### 2.1 注意力只保留在高层 stage

PTv3 的 `Block` 默认每层都是“局部几何卷积 + 注意力 + MLP”的完整块，encoder / decoder 中大多数 stage 都会用 attention。

LitePT 则显式引入了两组结构开关：

- `enc_conv=(True, True, True, False, False)`
- `enc_attn=(False, False, False, True, True)`

这意味着默认配置下：

- encoder 前 3 个 stage 只做卷积分支，不做 attention
- encoder 后 2 个 stage 只做 attention 分支，不做卷积分支

这是一种非常强的“分工式裁剪”：

- 低层点数多，attention 最贵，所以直接砍掉
- 高层点数少，再放 attention，计算收益比更高

这比 PTv3 的“每层都带 attention”更节省算力和显存。

### 2.2 decoder 默认几乎被裁掉

LitePT 默认配置：

- `dec_depths=(0, 0, 0, 0)`
- `dec_attn=(False, False, False, False)`
- `dec_conv=(False, False, False, False)`

也就是说，默认 LitePT 更接近 encoder-heavy 的设计，decoder 基本不承担复杂建模，只保留必要的上采样框架。

相比 PTv3 默认 decoder 仍然是完整 Transformer block 叠加，LitePT 在这里进一步压缩了参数量、激活占用和训练开销。

### 2.3 通道设计更“轻”

LitePT 默认通道：

- `enc_channels=(36, 72, 144, 252, 504)`

PTv3 m1 默认通道：

- `enc_channels=(32, 64, 128, 256, 512)`

乍看并不是每层都更小，但 LitePT 的关键不是单层宽度，而是：

- 大量低层 attention 被移除
- decoder 大量 block 被移除
- 结构分支按 stage 开关

因此整体 FLOPs 和激活成本会明显下降。

### 2.4 patch size 更大，attention 更偏高层全局化

LitePT 默认：

- `enc_patch_size=(1024, 1024, 1024, 1024, 1024)`

PTv3 m1 默认：

- `enc_patch_size=(48, 48, 48, 48, 48)`

LitePT 的逻辑不是在所有层都跑这种大 patch attention，而是：

- 低层根本不跑 attention
- 高层 token 数已经经过 pooling 大幅减少
- 这时再配合大 patch，相当于在高层增强更大感受野的全局建模

所以 LitePT 是把“大 patch attention”放在更少、更深、更稀疏的层上使用。

### 2.5 相对位置编码思路切换为 PointROPE

PTv3 m1 主要依赖：

- serialized local attention
- 可选 RPE

LitePT 则明确引入了 `PointROPEAttention`，用 3D RoPE 对 `q/k` 做旋转位置编码，再送入 attention。

这带来的变化是：

- 位置信息直接注入 `q/k`
- 不需要像 RPE 那样构造额外的 attention bias 表
- 更适合与 flash attention 类实现拼接，因为 RoPE 先作用在 token 上，再做标准 attention

这属于从“加 bias 的相对位置编码”切换到“旋转式位置编码”的设计变化。

### 2.6 pooling 之后按需重新 serialization

LitePT 的 `GridPooling` 多了两个很重要的参数：

- `re_serialization`
- `serialization_order`

只有当下一个 stage 需要 attention 时，才在 pooling 后重新 serialization。

这个点很关键，因为 serialization 本身也有代价。LitePT 把它从“默认每层都准备好”变成“只有 attention stage 才触发”，进一步减少无效开销。

## 3. 可以把 LitePT 概括成什么风格

如果用一句话概括，LitePT 更像：

“把 PTv3 改造成高层 attention、低层卷积、极简 decoder、RoPE 注入位置编码的轻量级混合骨干。”

其核心不是单纯缩小模型，而是：

- attention 更少
- attention 更晚
- decoder 更浅
- 位置编码更适合 flash 风格实现

## 4. 为什么 RoPE 这里还要单独写 CUDA 实现

`litept_v1m1.py` 里 `PointROPE` 有两套实现：

- 优先使用 `pointrope` CUDA 扩展
- 扩展不可用时回退到纯 PyTorch 实现

这是因为 PointROPE 在 LitePT 里不是一个很小的边角步骤，而是 attention 前对 `q/k` 的批量旋转变换。这个步骤在高 token 数、高 head 数时会比较贵。

### 4.1 PyTorch 版本做了什么

纯 PyTorch fallback 的流程大致是：

1. 先为频率构造 `cos/sin` 表
2. 对每个 token 的 xyz 位置做 embedding 取表
3. 把特征分成 x/y/z 三个子空间
4. 对每个子空间调用 `rotate_half`
5. 做逐元素乘法和加法
6. 再把三部分拼回去

这个实现优点是：

- 易读
- 容易验证正确性
- 不需要编译扩展

但缺点也很明显：

- kernel 数量多
- 中间张量多
- embedding / chunk / cat / rotate / mul / add 都会产生额外调度和访存

### 4.2 CUDA 版本做了什么

CUDA 实现在 `libs/pointrope/kernels.cu` 中，本质上做的是一个 fused kernel：

- 直接按 token 组织 block
- 在 kernel 内根据 xyz 位置和频率计算 `cos/sin`
- 在共享内存里读取 token 的局部数据
- 直接完成 `(u, v)` 旋转
- 原地写回结果

也就是说，CUDA 版把 PyTorch 版里原本分散的很多操作融合成了一次 kernel。

这带来的好处是：

- 更少的 kernel launch
- 更少的中间张量
- 更好的访存局部性
- 前向和反向都能复用同一个旋转算子思路

## 5. CUDA 实现和 Torch 实现的具体差别

### 5.1 算法层面

两者在数学上做的是同一件事：

- 将 head_dim 按 xyz 三个子空间划分
- 每个子空间再按 `(u, v)` 成对旋转
- 旋转角由位置和频率共同决定

所以它们的目标结果是一致的。

### 5.2 执行层面

Torch 版本：

- 由很多通用算子拼出来
- 更灵活
- 易调试
- 但 launch 多、访存碎

CUDA 版本：

- 一个定制 kernel 直接做完
- 原地修改 `tokens`
- 前向和反向都复用同一个旋转逻辑
- 更适合频繁调用的大规模训练

### 5.3 数据类型层面

LitePT 里有一句注释很关键：

`# workround to make pointrope cuda float32 happy`

也就是当前调用时会先：

- `q = self.rope(q.float(), pos).to(q.dtype)`
- `k = self.rope(k.float(), pos).to(k.dtype)`

这说明当前 `pointrope` CUDA 路径虽然底层 dispatch 支持 `half/bfloat16/float`，但实际作者在模型里仍然选择：

- 先转 `float32`
- RoPE 计算完成后再转回原 dtype

原因通常是：

- 数值更稳
- 避免某些半精度路径下 trig / 旋转误差积累
- 避免自定义核在混合精度训练里出现边界问题

因此它的 CUDA 版优化重点更偏向：

- kernel fusion
- 内存访问效率

而不是“极致半精度吞吐”。

## 6. pointrope CUDA 扩展是如何编译安装的

仓库里已经给了标准安装脚本：

- `libs/pointrope/setup.py`

它使用的是 PyTorch 官方扩展机制：

- `torch.utils.cpp_extension.CUDAExtension`
- `BuildExtension`

编译源文件：

- `pointrope.cpp`
- `kernels.cu`

并且会自动使用：

- `torch.cuda.get_gencode_flags()`

来生成当前环境可用的 CUDA 架构编译参数。

### 6.1 最直接的安装方法

在你的虚拟环境激活后，进入仓库根目录执行：

```bash
cd libs/pointrope
pip install -v .
```

如果你希望以开发模式安装：

```bash
cd libs/pointrope
pip install -e .
```

安装成功后，在你的虚拟环境里应该可以直接：

```python
import pointrope
```

### 6.2 对环境的要求

至少需要这些条件：

- 当前虚拟环境里的 `torch` 是 CUDA 版本
- 系统里有可用的 `nvcc`
- `nvcc` 对应的 CUDA toolkit 与 PyTorch CUDA ABI 基本兼容
- 编译器环境正常

在 Linux / WSL 下通常更顺畅；Windows 下也可以编，但常见问题更多，尤其是：

- Visual Studio Build Tools
- CUDA toolkit 版本匹配
- `nvcc` 和当前 `torch` 所用 CUDA 版本不一致

### 6.3 如何验证是否真的启用了 CUDA 版

LitePT 里已经写了回退提示：

- 如果 `import pointrope` 失败，会打印
  - `CUDA implementation unavailable ... Using slower Pytorch fallback.`

所以最简单的验证方式是：

1. 启动 Python
2. `import pointspace.models.backbone.litept.litept_v1m1`
3. 看是否出现 fallback 提示

如果没有这条提示，并且：

```python
import pointrope
```

本身也成功，那么大概率就是 CUDA 扩展已经装好了。

## 7. 实际使用上的建议

### 7.1 如果你关心速度

优先保证 `pointrope` CUDA 扩展可用，因为 LitePT 的 RoPE 会频繁作用在 attention 输入上。  
如果一直走 PyTorch fallback，LitePT 的高层 attention 部分会平白多出不少开销。

### 7.2 如果你关心稳定性

LitePT 当前实现里先把 `q/k` 转成 `float32` 再做 RoPE，这本身就是偏稳健的策略。  
如果你后续要进一步做半精度极限优化，建议先单独验证：

- 前向一致性
- 反向梯度稳定性
- 不同 dtype 下的数值偏差

### 7.3 如果你关心“为什么 LitePT 适合做轻量变体”

原因不只是 attention 少，而是它把整条建模路径重新分配了：

- 低层：卷积负责局部建模
- 高层：RoPE attention 负责长程依赖
- decoder：尽量瘦身

这使得它比“全层 Transformer 化”的 PTv3 更容易在速度、显存和精度之间做平衡。

## 8. 总结

LitePT 相对 PTv3 的核心改进可以概括为四点：

1. 把 attention 从低层大量移除，只保留在高层
2. 大幅裁剪 decoder，减少不必要的恢复阶段开销
3. 用 PointROPE 替代更重的位置编码方式
4. 通过可选 CUDA 扩展把 RoPE 这一步做成 fused kernel，提高实际运行效率

而 PointROPE 之所以还要单独写 CUDA，不是因为 PyTorch 实现做不到，而是因为：

- PyTorch 版太碎
- attention 前频繁调用时开销明显
- fused CUDA kernel 更适合这种重复、高吞吐、规则化的旋转操作

如果你的目标是把 LitePT 真正跑快，那么除了 attention 后端，`pointrope` CUDA 扩展是否启用，也是一个很关键的性能点。
