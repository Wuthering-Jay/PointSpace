"""
Fusion and Unpool Operators for SPT.

This module provides feature fusion and unpooling operations for the
UNet-like decoder in Superpoint Transformer.

Reference: src/nn/fusion.py, src/nn/unpool.py from Superpoint Transformer

Author: PointSpace Team
"""

import torch
from torch import nn
from typing import Optional


__all__ = [
    "fusion_factory",
    "BaseFusion",
    "CatFusion",
    "AdditiveFusion",
    "TakeFirstFusion",
    "TakeSecondFusion",
    "IndexUnpool",
]


def fusion_factory(mode: str) -> nn.Module:
    """Return the fusion class from an input string.

    Args:
        mode: Fusion mode. Options:
            - 'cat', 'concatenate', 'concatenation', '|': Concatenation
            - 'residual', 'additive', '+': Addition
            - 'first', '1', '1st': Take first input
            - 'second', '2', '2nd': Take second input

    Returns:
        Fusion module.

    Raises:
        NotImplementedError: If mode is unknown.
    """
    if mode in ["cat", "concatenate", "concatenation", "|"]:
        return CatFusion()
    elif mode in ["residual", "additive", "+"]:
        return AdditiveFusion()
    elif mode in ["first", "1", "1st"]:
        return TakeFirstFusion()
    elif mode in ["second", "2", "2nd"]:
        return TakeSecondFusion()
    else:
        raise NotImplementedError(f"Unknown fusion mode='{mode}'")


class BaseFusion(nn.Module):
    """Base class for feature fusion operations.

    Handles None inputs gracefully: if either input is None, returns the other.
    """

    def forward(
        self,
        x1: Optional[torch.Tensor],
        x2: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Forward pass.

        Args:
            x1: First input tensor or None.
            x2: Second input tensor or None.

        Returns:
            Fused tensor, or None if both inputs are None.
        """
        if x1 is None and x2 is None:
            return None
        if x1 is None:
            return x2
        if x2 is None:
            return x1
        return self._func(x1, x2)

    def _func(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
    ) -> torch.Tensor:
        """Actual fusion operation. Subclasses must implement this."""
        raise NotImplementedError


class CatFusion(BaseFusion):
    """Concatenation fusion along channel dimension."""

    def _func(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return torch.cat((x1, x2), dim=1)


class AdditiveFusion(BaseFusion):
    """Additive (residual) fusion."""

    def _func(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return x1 + x2


class TakeFirstFusion(BaseFusion):
    """Take first input only."""

    def _func(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return x1


class TakeSecondFusion(BaseFusion):
    """Take second input only."""

    def _func(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return x2


class IndexUnpool(nn.Module):
    """Simple unpooling operation based on index selection.

    Redistributes (i+1)-level features to i-level nodes based on their
    parent indexing. Each child node receives the features of its parent.
    """

    def forward(self, x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Parent features of shape (Np, C).
            idx: LongTensor of shape (Nc,) where idx[i] is the parent index
                of child node i.

        Returns:
            Unpooled features of shape (Nc, C).
        """
        return x.index_select(0, idx)
