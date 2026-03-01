"""remains unchanged header"""
from collections import OrderedDict
from pointspace.utils.registry import Registry

LOSSES = Registry("losses")


class Criteria(object):
    def __init__(self, cfg=None):
        self.cfg = cfg if cfg is not None else []
        self.criteria = []
        for loss_cfg in self.cfg:
            self.criteria.append(LOSSES.build(cfg=loss_cfg))
        # Populated after every __call__: {short_name: raw_unweighted_loss_tensor}
        self._last_individual_losses: OrderedDict = OrderedDict()

    def _criterion_short_name(self, criterion, index):
        """Derive a short display name from the criterion class name.

        Examples:
            CrossEntropyLoss  -> ce
            LovaszLoss        -> lovasz
            SmoothCELoss      -> smooth_ce
        """
        import re
        name = type(criterion).__name__
        # Strip trailing 'Loss' / 'loss'
        name = re.sub(r'[Ll]oss$', '', name)
        # CamelCase -> snake_case
        name = re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower().strip('_')
        if not name:
            name = f"loss{index}"
        return name

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
            self._last_individual_losses = OrderedDict()
            return pred
        loss = 0
        individual = OrderedDict()
        for idx, c in enumerate(self.criteria):
            weighted = c(pred, target)          # raw * loss_weight
            loss += weighted
            # Recover raw (unweighted) value for display only
            w = getattr(c, 'loss_weight', 1.0)
            raw = weighted / w if (w and w != 0) else weighted
            name = self._criterion_short_name(c, idx)
            # Deduplicate names
            if name in individual:
                name = f"{name}_{idx}"
            individual[name] = raw.detach()
        self._last_individual_losses = individual
        return loss


def build_criteria(cfg):
    return Criteria(cfg)
