_base_ = ["./semseg-hpsd-litept-v1m4-dales-base.py"]

# 相同下游结构和 seed、但不加载任何预训练权重，用于衡量预训练净收益。
weight = None
save_path = "exp/dales/hpsd/semseg-random-litept-v1m4-ft"
pred_save_dir = r"E:\data\DALES\dales_las\tile\pred_random"
writer = dict(save_dir=pred_save_dir)
