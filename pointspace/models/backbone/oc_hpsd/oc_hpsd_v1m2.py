"""带连续 CSC 可信度加权的第二版观测条件 HPSD。"""

from pointspace.models.builder import MODELS

from .oc_hpsd_v1m1 import ObservationConditionedHPSD


@MODELS.register_module("OC-HPSD-v1m2")
class ObservationConditionedHPSDV1M2(ObservationConditionedHPSD):
    """平滑利用部分遮蔽 token，同时保持 v1m1 的单次 encoder 路径。

    v1m1 以 ``mask_fraction`` 硬门控 CSC token。v1m2 保留一个较低的噪声
    过滤阈值，再根据遮蔽覆盖率、masked support 的平均连续可观测度和支持点
    数量生成连续权重。权重在每个样本内部归一化，因此点数较多的 tile 不会
    主导 batch loss。旧模型的参数名、state dict 和数值路径均不改变。
    """

    def __init__(
        self,
        completion_min_mask_fraction=0.1,
        completion_full_weight_fraction=0.5,
        **kwargs,
    ):
        super().__init__(
            completion_min_mask_fraction=completion_min_mask_fraction,
            completion_soft_weight=True,
            completion_full_weight_fraction=completion_full_weight_fraction,
            report_compact_stats=True,
            **kwargs,
        )
