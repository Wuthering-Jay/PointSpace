"""
Utility functions for SPT neural network components.

Reference: src/utils/nn.py from Superpoint Transformer

Author: PointSpace Team
"""

import torch
from torch import nn
from typing import Optional, Union, Callable


__all__ = ["init_weights", "build_qk_scale_func"]


def init_weights(
    m: nn.Module,
    linear: Optional[str] = None,
    rpe: Optional[str] = None,
    activation: str = "leaky_relu",
) -> None:
    """Manual weight initialization.

    Allows setting specific init modes for certain modules. In particular,
    the linear and RPE layers are initialized with Xavier uniform
    initialization by default.

    Reference: https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf

    Supported initializations:
        - 'xavier_uniform'
        - 'xavier_normal'
        - 'kaiming_uniform'
        - 'kaiming_normal'
        - 'trunc_normal'

    Args:
        m: Module to initialize.
        linear: Initialization method for linear layers.
        rpe: Initialization method for RPE layers.
        activation: Activation function name for gain calculation.
    """
    from pointspace.models.backbone.ezsp.spt.attention import SelfAttentionBlock

    linear = "xavier_uniform" if linear is None else linear
    rpe = linear if rpe is None else rpe

    if isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
        return

    if isinstance(m, nn.Linear):
        _linear_init(m, method=linear, activation=activation)
        return

    if isinstance(m, SelfAttentionBlock):
        if m.k_rpe is not None:
            _linear_init(m.k_rpe, method=rpe, activation=activation)
        if m.q_rpe is not None:
            _linear_init(m.q_rpe, method=rpe, activation=activation)
        return

    # Handle torchsparse Conv3d if available
    try:
        from torchsparse import nn as spnn

        if isinstance(m, spnn.Conv3d):
            _conv_init(m, method=linear, activation=activation)
            return
    except ImportError:
        pass


def _linear_init(
    m: nn.Module,
    method: str = "xavier_uniform",
    activation: str = "leaky_relu",
) -> None:
    """Initialize linear layer weights.

    Args:
        m: Linear module to initialize.
        method: Initialization method.
        activation: Activation function name for gain calculation.
    """
    gain = torch.nn.init.calculate_gain(activation)

    if m.bias is not None:
        nn.init.constant_(m.bias, 0)

    if method == "xavier_uniform":
        nn.init.xavier_uniform_(m.weight, gain=gain)
    elif method == "xavier_normal":
        nn.init.xavier_normal_(m.weight, gain=gain)
    elif method == "kaiming_uniform":
        nn.init.kaiming_uniform_(m.weight, nonlinearity=activation)
    elif method == "kaiming_normal":
        nn.init.kaiming_normal_(m.weight, nonlinearity=activation)
    elif method == "trunc_normal":
        nn.init.trunc_normal_(m.weight, std=0.02)
    else:
        raise NotImplementedError(f"Unknown initialization method: {method}")


def _conv_init(
    m: nn.Module,
    method: str = "xavier_uniform",
    activation: str = "leaky_relu",
) -> None:
    """Initialize torchsparse Conv3d weights.

    Args:
        m: Conv3d module to initialize.
        method: Initialization method.
        activation: Activation function name for gain calculation.
    """
    gain = torch.nn.init.calculate_gain(activation)

    if m.bias is not None:
        nn.init.constant_(m.bias, 0)

    # Transpose kernel for correct fan_in/fan_out computation
    # torchsparse stores weights as (kernel_volume, in_channels, out_channels)
    # while nn.init expects (out_channels, in_channels, kernel_volume)
    if m.kernel.dim() == 3:
        kernel = m.kernel.permute(2, 1, 0)
    elif m.kernel.dim() == 2:
        kernel = m.kernel.permute(1, 0)
    else:
        raise ValueError(
            f"Kernel has {m.kernel.dim()} dimensions, expected 2 or 3"
        )

    if method == "xavier_uniform":
        nn.init.xavier_uniform_(kernel, gain=gain)
    elif method == "xavier_normal":
        nn.init.xavier_normal_(kernel, gain=gain)
    elif method == "kaiming_uniform":
        nn.init.kaiming_uniform_(kernel, nonlinearity=activation)
    elif method == "kaiming_normal":
        nn.init.kaiming_normal_(kernel, nonlinearity=activation)
    elif method == "trunc_normal":
        nn.init.trunc_normal_(kernel, std=0.02)
    else:
        raise NotImplementedError(f"Unknown initialization method: {method}")


def build_qk_scale_func(
    dim: int,
    num_heads: int,
    qk_scale: Optional[Union[float, str]],
) -> Callable:
    """Build the QK-scale function for attention.

    This function follows the template: f(s), where `s` is `edge_index[0]`.

    Args:
        dim: Feature dimension.
        num_heads: Number of attention heads.
        qk_scale: Scaling mode. Options:
            - None: Default 1/(sqrt(d)*sqrt(g))
            - float: Use as-is
            - 'd': 1/sqrt(d)
            - 'g': 1/sqrt(g)
            - 'd+g' or 'g+d': 1/(sqrt(d)+sqrt(g))
            - 'd.g', 'd*g', etc.: 1/(sqrt(d)*sqrt(g))

    Returns:
        Scaling function that takes source indices.
    """
    # Default: 1/(sqrt(dim)*sqrt(num_neighbors))
    if qk_scale is None:

        def f(s):
            D = (dim // num_heads) ** -0.5
            G = (s.bincount() ** -0.5)[s].view(-1, 1, 1)
            return D * G

        return f

    # Scalar value
    if not isinstance(qk_scale, str):

        def f(s):
            return qk_scale

        return f

    # Parse string
    qk_scale = qk_scale.lower().replace(" ", "")

    if qk_scale in ["d+g", "g+d"]:

        def f(s):
            D = (dim // num_heads) ** -0.5
            G = (s.bincount() ** -0.5)[s].view(-1, 1, 1)
            return D + G

        return f

    if qk_scale in ["dg", "gd", "d*g", "g*d", "d.g", "g.d"]:

        def f(s):
            D = (dim // num_heads) ** -0.5
            G = (s.bincount() ** -0.5)[s].view(-1, 1, 1)
            return D * G

        return f

    if qk_scale == "d":

        def f(s):
            D = (dim // num_heads) ** -0.5
            return D

        return f

    if qk_scale == "g":

        def f(s):
            G = (s.bincount() ** -0.5)[s].view(-1, 1, 1)
            return G

        return f

    raise ValueError(f"Unable to build QK scaling scheme for qk_scale='{qk_scale}'")
