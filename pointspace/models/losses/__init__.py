from .builder import build_criteria, LOSSES

from .misc import (
    CrossEntropyLoss,
    SmoothCELoss,
    DiceLoss,
    FocalLoss,
    BinaryFocalLoss,
    MSELoss,
    L1Loss,
    SmoothL1Loss,
    HuberLoss,
)
from .lovasz import LovaszLoss
from .superpoint import SuperpointConsistencyLoss
