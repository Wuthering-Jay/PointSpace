"""
Criteria Builder

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from pointspace.utils.registry import Registry

LOSSES = Registry("losses")


class Criteria(object):
    def __init__(self, cfg=None):
        self.cfg = cfg if cfg is not None else []
        self.criteria = []
        for loss_cfg in self.cfg:
            self.criteria.append(LOSSES.build(cfg=loss_cfg))

    def set_class_weight(self, class_weight):
        """Inject *class_weight* into every loss that has ``auto_class_weight=True``.

        Called by the Trainer after the dataset computes class weights.
        Only losses that expose both ``auto_class_weight`` and
        ``set_class_weight()`` will be affected.
        """
        for c in self.criteria:
            if getattr(c, "auto_class_weight", False) and hasattr(
                c, "set_class_weight"
            ):
                c.set_class_weight(class_weight)

    def __call__(self, pred, target):
        if len(self.criteria) == 0:
            # loss computation occur in model
            return pred
        loss = 0
        for c in self.criteria:
            loss += c(pred, target)
        return loss


def build_criteria(cfg):
    return Criteria(cfg)
