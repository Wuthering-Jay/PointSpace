"""
Weight Initialization for EZ-SP Transformer

Implements Xavier/Kaiming initialization for Linear layers and special
handling for LayerNorm to improve FP16/AMP stability.

Based on official SPT implementation:
reference_code/superpoint_transformer/src/utils/nn.py

Author: PointSpace Team
"""

import torch
import torch.nn as nn


def init_weights(module, linear='xavier_uniform', rpe='xavier_uniform', activation='leaky_relu'):
    """
    Initialize module weights
    
    Supported init methods:
      - 'xavier_uniform'  (default, good for tanh/sigmoid)
      - 'xavier_normal'
      - 'kaiming_uniform' (good for ReLU/LeakyReLU)
      - 'kaiming_normal'
      - 'trunc_normal'
    
    Args:
        module: nn.Module to initialize
        linear: Initialization method for Linear layers
        rpe: Initialization method for RPE (relative position encoding) layers
        activation: Activation function name (for gain calculation)
    """
    
    # LayerNorm: weight=1.0, bias=0
    if isinstance(module, nn.LayerNorm):
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
        if module.weight is not None:
            nn.init.constant_(module.weight, 1.0)
        return
    
    # BatchNorm1d: weight=1.0, bias=0
    if isinstance(module, nn.BatchNorm1d):
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
        if module.weight is not None:
            nn.init.constant_(module.weight, 1.0)
        return
    
    # Linear layers
    if isinstance(module, nn.Linear):
        _linear_init(module, method=linear, activation=activation)
        return
    
    # Self-Attention RPE layers (if exists)
    from pointspace.models.backbone.ezsp.spt.attention import SelfAttentionBlock
    if isinstance(module, SelfAttentionBlock):
        if hasattr(module, 'k_rpe') and module.k_rpe is not None:
            _linear_init(module.k_rpe, method=rpe, activation=activation)
        if hasattr(module, 'q_rpe') and module.q_rpe is not None:
            _linear_init(module.q_rpe, method=rpe, activation=activation)
        if hasattr(module, 'v_rpe') and module.v_rpe is not None:
            _linear_init(module.v_rpe, method=rpe, activation=activation)
        return


def _linear_init(module, method='xavier_uniform', activation='leaky_relu'):
    """
    Initialize a Linear layer
    
    Args:
        module: nn.Linear module
        method: Initialization method
        activation: Activation function (for gain calculation)
    """
    # Calculate gain based on activation
    if method in ['xavier_uniform', 'xavier_normal']:
        gain = nn.init.calculate_gain(activation)
    else:
        gain = 1.0
    
    # Bias -> 0
    if module.bias is not None:
        nn.init.constant_(module.bias, 0)
    
    # Weight initialization
    if method == 'xavier_uniform':
        nn.init.xavier_uniform_(module.weight, gain=gain)
    elif method == 'xavier_normal':
        nn.init.xavier_normal_(module.weight, gain=gain)
    elif method == 'kaiming_uniform':
        nn.init.kaiming_uniform_(module.weight, nonlinearity=activation)
    elif method == 'kaiming_normal':
        nn.init.kaiming_normal_(module.weight, nonlinearity=activation)
    elif method == 'trunc_normal':
        nn.init.trunc_normal_(module.weight, std=0.02)
    else:
        raise ValueError(f"Unknown init method: {method}")


def apply_weight_init(model, linear='xavier_uniform', rpe='xavier_uniform', activation='leaky_relu'):
    """
    Apply weight initialization to entire model
    
    Args:
        model: nn.Module to initialize
        linear: Method for Linear layers
        rpe: Method for RPE layers
        activation: Activation function name
        
    Example:
        >>> from pointspace.models.backbone.ezsp.spt.weight_init import apply_weight_init
        >>> model = EZSPTransformer(...)
        >>> apply_weight_init(model)
    """
    if model is None:
        return
    init_fn = lambda m: init_weights(m, linear=linear, rpe=rpe, activation=activation)
    model.apply(init_fn)
