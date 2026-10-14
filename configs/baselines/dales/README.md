# Section 4.5 DALES 统一下游协议

三份独立配置不依赖 `_base_`：

| 模式 | 配置 | 可训练部分 |
| --- | --- | --- |
| 严格 linear probe | [semseg-litept-v1m4-pretrained-lin.py](semseg-litept-v1m4-pretrained-lin.py) | 单层 `Linear(72,8)`；整个 LitePT 保持 eval 且冻结 |
| decoder probe | [semseg-litept-v1m4-pretrained-dec.py](semseg-litept-v1m4-pretrained-dec.py) | 新 LitePT decoder 与语义线性头；stem/encoder 冻结 |
| full fine-tuning | [semseg-litept-v1m4-pretrained-ft.py](semseg-litept-v1m4-pretrained-ft.py) | 完整 LitePT 与语义线性头 |

在每份文件顶部设置 `pretrain_method` 为 `point-m2ae`、`msc`、`sonata`、`d-ditr` 或 `concerto`。配置会加载对应 Swiss 预训练输出的 `model/backbone_last.pth`；该文件尚不存在时必须先完成相应预训练，不会静默退化为 Scratch。五种方法使用同一 LitePT-v1m4、DALES 划分、0.25 m GridSample、零填 intensity、类别映射、损失和测试器。decoder probe 只载入预训练 stem/encoder，随机初始化任务 decoder。Scratch FT 对照见 [semseg-litept-v1m4-scratch-ft.py](semseg-litept-v1m4-scratch-ft.py)。

例如选择 `sonata` 后：

```powershell
D:/app/anaconda3/envs/pointspace/python.exe tools/train.py --config-file configs/baselines/dales/semseg-litept-v1m4-pretrained-lin.py
D:/app/anaconda3/envs/pointspace/python.exe tools/train.py --config-file configs/baselines/dales/semseg-litept-v1m4-pretrained-dec.py
D:/app/anaconda3/envs/pointspace/python.exe tools/train.py --config-file configs/baselines/dales/semseg-litept-v1m4-pretrained-ft.py
```

当前仅验证了配置、冻结边界与权重键协议；尚无这五种方法的完整 Swiss 权重或 DALES 主表结果。
