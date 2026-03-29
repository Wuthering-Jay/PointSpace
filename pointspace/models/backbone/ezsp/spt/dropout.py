"""
Dropout Components for SPT.

This module provides dropout layers including DropPath (stochastic depth).

Reference: src/nn/dropout.py from Superpoint Transformer

Author: PointSpace Team
"""

import torch
from torch import nn


__all__ = ["DropPath", "drop_path"]


def drop_path(
    x: torch.Tensor,
    drop_prob: float = 0.0,
    training: bool = False,
    scale_by_keep: bool = True,
) -> torch.Tensor:
    """Drop paths (Stochastic Depth) per sample.

    When applied in main path of residual blocks, this randomly drops
    entire samples during training.

    Credit: https://github.com/rwightman/pytorch-image-models

    Args:
        x: Input tensor.
        drop_prob: Probability of dropping a path.
        training: Whether in training mode.
        scale_by_keep: Whether to scale output by keep probability.

    Returns:
        Tensor with dropped paths.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    # Work with different dim tensors, not just 2D ConvNets
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample.

    When applied in main path of residual blocks, this randomly drops
    entire samples during training.

    Credit: https://github.com/rwightman/pytorch-image-models

    Args:
        drop_prob: Probability of dropping a path.
        scale_by_keep: Whether to scale output by keep probability.
    """

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        """Initialize DropPath.

        Args:
            drop_prob: Probability of dropping a path during training.
            scale_by_keep: Whether to scale output by keep probability.
        """
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor.

        Returns:
            Output tensor with stochastic depth applied.
        """
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self) -> str:
        return f"drop_prob={round(self.drop_prob, 3):0.3f}"
