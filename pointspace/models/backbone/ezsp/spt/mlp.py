"""
MLP Components for SPT.

This module provides MLP building blocks used in the Superpoint Transformer.

Reference: src/nn/mlp.py from Superpoint Transformer

Author: PointSpace Team
"""

import torch
from torch import nn
from typing import List, Optional, Union, Type

from pointspace.models.backbone.ezsp.spt.norm import BatchNorm, INDEX_BASED_NORMS


__all__ = ["MLP", "FFN", "Classifier", "mlp"]


def mlp(
    dims: List[int],
    activation: Optional[nn.Module] = None,
    last_activation: bool = True,
    norm: Optional[Type[nn.Module]] = BatchNorm,
    last_norm: bool = True,
    drop: Optional[float] = None,
    device: Union[str, torch.device] = "cpu",
) -> nn.ModuleList:
    """Helper to build MLP-like structures.

    Args:
        dims: List of channel sizes. Expects len(dims) >= 2.
        activation: Non-linearity module. Default is LeakyReLU.
        last_activation: Whether the last layer should have an activation.
        norm: Normalization class. Can be None (e.g., for FFN).
            Must be instantiable using norm(in_channels).
        last_norm: Whether the last layer should have a normalization.
        drop: Dropout probability in [0, 1]. No dropout if None or < 0.
        device: Device on which to create the MLP.

    Returns:
        nn.ModuleList of layers.
    """
    assert len(dims) >= 2, "Need at least 2 dims for input and output"

    if activation is None:
        activation = nn.LeakyReLU()

    # Only use bias if no normalization is applied
    bias = norm is None

    # Iteratively build the layers based on dims
    modules = []
    for i in range(1, len(dims)):
        modules.append(nn.Linear(dims[i - 1], dims[i], bias=bias, device=device))
        if norm is not None and (last_norm or i < len(dims) - 1):
            modules.append(norm(dims[i]).to(device))
        if activation is not None and (last_activation or i < len(dims) - 1):
            # Clone activation to avoid sharing state
            modules.append(activation)

    # Add final dropout if required
    if drop is not None and drop > 0:
        modules.append(nn.Dropout(drop, inplace=True))

    return nn.ModuleList(modules)


class MLP(nn.Module):
    """MLP operating on features [N, D] tensors.

    You can think of it as a series of 1x1 conv -> 1D batch norm -> activation.
    Supports index-based normalization layers (LayerNorm, InstanceNorm, etc.)
    that require a batch index.
    """

    def __init__(
        self,
        dims: List[int],
        activation: Optional[nn.Module] = None,
        last_activation: bool = True,
        norm: Optional[Type[nn.Module]] = BatchNorm,
        last_norm: bool = True,
        drop: Optional[float] = None,
        device: Union[str, torch.device] = "cpu",
    ):
        """Initialize MLP.

        Args:
            dims: List of channel sizes [in_dim, hidden_dims..., out_dim].
            activation: Activation function. Default is LeakyReLU.
            last_activation: Whether to apply activation after last layer.
            norm: Normalization layer class. None for no normalization.
            last_norm: Whether to apply norm after last layer.
            drop: Dropout probability.
            device: Device for parameters.
        """
        super().__init__()
        if activation is None:
            activation = nn.LeakyReLU()
        self.mlp = mlp(
            dims,
            activation=activation,
            last_activation=last_activation,
            norm=norm,
            last_norm=last_norm,
            drop=drop,
            device=device,
        )
        self.out_dim = dims[-1]

    def forward(self, x: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input features of shape [N, D].
            batch: Optional batch indices of shape [N] for index-based norms.

        Returns:
            Output features of shape [N, out_dim].
        """
        # Manually iterate to pass batch index for special normalization layers
        for module in self.mlp:
            if isinstance(module, INDEX_BASED_NORMS):
                x = module(x, batch=batch)
            else:
                x = module(x)
        return x


class FFN(MLP):
    """Feed-Forward Network as used in Transformers.

    By convention, these MLPs have 2 Linear layers and no normalization,
    the last layer has no activation and an optional dropout may be applied
    on the output features.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        out_dim: Optional[int] = None,
        activation: Optional[nn.Module] = None,
        drop: Optional[float] = None,
        device: Union[str, torch.device] = "cpu",
    ):
        """Initialize FFN.

        Args:
            dim: Input dimension.
            hidden_dim: Hidden layer dimension. Defaults to dim.
            out_dim: Output dimension. Defaults to dim.
            activation: Activation function. Default is LeakyReLU.
            drop: Dropout probability on output.
            device: Device for parameters.
        """
        # Build the channel sizes for the 2 linear layers
        hidden_dim = hidden_dim or dim
        out_dim = out_dim or dim
        channels = [dim, hidden_dim, out_dim]

        if activation is None:
            activation = nn.LeakyReLU()

        super().__init__(
            channels,
            activation=activation,
            last_activation=False,
            norm=None,
            last_norm=False,
            drop=drop,
            device=device,
        )


class Classifier(nn.Module):
    """A simple fully-connected head with no activation and no normalization."""

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        bias: bool = True,
        device: Union[str, torch.device] = "cpu",
    ):
        """Initialize Classifier.

        Args:
            in_dim: Input feature dimension.
            num_classes: Number of output classes.
            bias: Whether to use bias in linear layer.
            device: Device for parameters.
        """
        super().__init__()
        self.classifier = nn.Linear(in_dim, num_classes, bias=bias, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input features of shape [N, in_dim].

        Returns:
            Logits of shape [N, num_classes].
        """
        return self.classifier(x)
