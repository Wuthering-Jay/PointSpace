"""
NaN/Inf Detection Hook for Deep Learning Model Debugging

This module provides a forward hook that detects NaN/Inf values in model outputs
during forward pass, helping diagnose numerical instability issues.

Usage 1 - Standalone:
    from pointspace.engines.hooks import NaNInfDetector
    
    detector = NaNInfDetector(raise_on_nan=True)
    detector.register(model)
    # ... training ...
    detector.remove()

Usage 2 - In Config (Recommended):
    hooks = [
        dict(type="NaNInfDetectorHook", raise_on_nan=True, verbose=False),
        # ... other hooks ...
    ]

Author: PointSpace Team
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Tuple, Any
import logging

from .default import HookBase
from .builder import HOOKS

logger = logging.getLogger(__name__)


class NaNInfDetectorHook:
    """
    Forward hook for detecting NaN/Inf values in model outputs.
    
    This hook can be registered to any PyTorch module to monitor its outputs
    for numerical issues during forward pass.
    
    Features:
    - Detects both NaN and Inf values
    - Reports module name and layer type
    - Counts affected values
    - Optional input checking
    - Configurable behavior (raise/warn)
    - Detailed statistics
    
    Args:
        raise_on_nan: bool - Raise exception when NaN detected (default: True)
        raise_on_inf: bool - Raise exception when Inf detected (default: False)
        print_stats: bool - Print detection statistics (default: True)
        check_input: bool - Also check module inputs (default: False)
        enabled: bool - Whether hook is active (default: True)
        verbose: bool - Print info for every layer (default: False)
    
    Example:
        >>> detector = NaNInfDetectorHook(raise_on_nan=True)
        >>> detector.register(model)
        >>> output = model(input)  # Will raise if NaN detected
        >>> detector.print_summary()
        >>> detector.remove()
    """
    
    def __init__(
        self,
        raise_on_nan: bool = True,
        raise_on_inf: bool = False,
        print_stats: bool = True,
        check_input: bool = False,
        enabled: bool = True,
        verbose: bool = False,
    ):
        self.raise_on_nan = raise_on_nan
        self.raise_on_inf = raise_on_inf
        self.print_stats = print_stats
        self.check_input = check_input
        self.enabled = enabled
        self.verbose = verbose
        
        # Storage
        self.hooks = []
        self.detections = []  # List of (module_name, issue_type, stats)
        self.forward_count = 0
        
    def register(self, model: nn.Module, prefix: str = ""):
        """
        Register hook to model and all its children.
        
        Args:
            model: PyTorch module to monitor
            prefix: Name prefix for nested modules
        """
        # Register to current module
        hook = model.register_forward_hook(
            self._make_hook(prefix if prefix else model.__class__.__name__)
        )
        self.hooks.append(hook)
        
        # Recursively register to children
        for name, child in model.named_children():
            child_name = f"{prefix}.{name}" if prefix else name
            self.register(child, prefix=child_name)
        
        if not prefix:  # Only log for root
            logger.info(f"NaN/Inf detector registered to {len(self.hooks)} modules")
    
    def remove(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        logger.info("NaN/Inf detector hooks removed")
    
    def reset_stats(self):
        """Reset detection statistics."""
        self.detections.clear()
        self.forward_count = 0
    
    def _make_hook(self, module_name: str):
        """Create a forward hook for given module name."""
        def hook(module, input, output):
            if not self.enabled:
                return
            
            self.forward_count += 1
            
            # Check input if requested
            if self.check_input:
                self._check_tensors(
                    input, 
                    module_name, 
                    stage="input",
                    module_type=module.__class__.__name__
                )
            
            # Check output
            self._check_tensors(
                output,
                module_name,
                stage="output",
                module_type=module.__class__.__name__
            )
        
        return hook
    
    def _check_tensors(
        self,
        data: Any,
        module_name: str,
        stage: str = "output",
        module_type: str = "Unknown"
    ):
        """
        Check tensor(s) for NaN/Inf values.
        
        Args:
            data: Tensor or tuple/list/dict of tensors
            module_name: Name of the module
            stage: "input" or "output"
            module_type: Type of the module (e.g., "Linear", "Conv2d")
        """
        tensors = self._extract_tensors(data)
        
        for i, tensor in enumerate(tensors):
            if tensor is None or not isinstance(tensor, torch.Tensor):
                continue
            
            # Skip integer and boolean tensors (can't have NaN/Inf)
            if tensor.dtype not in (torch.float16, torch.float32, torch.float64, torch.bfloat16, torch.complex64, torch.complex128):
                if self.verbose:
                    logger.info(
                        f"[SKIP] {module_name} ({module_type}) {stage}: "
                        f"shape={tuple(tensor.shape)}, dtype={tensor.dtype} (non-float)"
                    )
                continue
            
            # Check for NaN
            has_nan = torch.isnan(tensor).any().item()
            if has_nan:
                nan_count = torch.isnan(tensor).sum().item()
                total = tensor.numel()
                
                msg = (
                    f"[NaN DETECTED] Module: {module_name} ({module_type})\n"
                    f"  Stage: {stage}\n"
                    f"  Tensor index: {i}\n"
                    f"  Shape: {tuple(tensor.shape)}\n"
                    f"  NaN count: {nan_count}/{total} ({100*nan_count/total:.2f}%)\n"
                    f"  dtype: {tensor.dtype}, device: {tensor.device}"
                )
                
                self.detections.append({
                    'module': module_name,
                    'type': module_type,
                    'stage': stage,
                    'issue': 'NaN',
                    'count': nan_count,
                    'total': total,
                    'shape': tuple(tensor.shape),
                    'dtype': str(tensor.dtype),
                })
                
                if self.print_stats:
                    logger.error(msg)
                
                if self.raise_on_nan:
                    raise RuntimeError(msg)
            
            # Check for Inf
            has_inf = torch.isinf(tensor).any().item()
            if has_inf:
                inf_count = torch.isinf(tensor).sum().item()
                total = tensor.numel()
                
                msg = (
                    f"[Inf DETECTED] Module: {module_name} ({module_type})\n"
                    f"  Stage: {stage}\n"
                    f"  Tensor index: {i}\n"
                    f"  Shape: {tuple(tensor.shape)}\n"
                    f"  Inf count: {inf_count}/{total} ({100*inf_count/total:.2f}%)\n"
                    f"  dtype: {tensor.dtype}, device: {tensor.device}"
                )
                
                self.detections.append({
                    'module': module_name,
                    'type': module_type,
                    'stage': stage,
                    'issue': 'Inf',
                    'count': inf_count,
                    'total': total,
                    'shape': tuple(tensor.shape),
                    'dtype': str(tensor.dtype),
                })
                
                if self.print_stats:
                    logger.warning(msg)
                
                if self.raise_on_inf:
                    raise RuntimeError(msg)
            
            # Verbose mode: print stats for every layer
            if self.verbose and not (has_nan or has_inf):
                # Skip non-floating point tensors in verbose mode
                if tensor.dtype in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
                    min_val = tensor.min().item() if tensor.numel() > 0 else 0
                    max_val = tensor.max().item() if tensor.numel() > 0 else 0
                    mean_val = tensor.mean().item() if tensor.numel() > 0 else 0
                    logger.info(
                        f"[OK] {module_name} ({module_type}) {stage}: "
                        f"shape={tuple(tensor.shape)}, "
                        f"range=[{min_val:.4f}, {max_val:.4f}], "
                        f"mean={mean_val:.4f}"
                    )
                else:
                    # For integer/bool tensors, just print shape and dtype
                    logger.info(
                        f"[OK] {module_name} ({module_type}) {stage}: "
                        f"shape={tuple(tensor.shape)}, dtype={tensor.dtype}"
                    )
    
    def _extract_tensors(self, data: Any) -> List[torch.Tensor]:
        """Extract all tensors from various data structures."""
        if data is None:
            return []
        
        if isinstance(data, torch.Tensor):
            return [data]
        
        if isinstance(data, (tuple, list)):
            tensors = []
            for item in data:
                tensors.extend(self._extract_tensors(item))
            return tensors
        
        if isinstance(data, dict):
            tensors = []
            for value in data.values():
                tensors.extend(self._extract_tensors(value))
            return tensors
        
        # For custom objects with tensor attributes
        if hasattr(data, '__dict__'):
            tensors = []
            for value in data.__dict__.values():
                if isinstance(value, torch.Tensor):
                    tensors.append(value)
            return tensors
        
        return []
    
    def print_summary(self):
        """Print summary of all detections."""
        if not self.detections:
            logger.info("✓ No NaN/Inf detected in forward pass")
            return
        
        logger.error("=" * 80)
        logger.error(f"NaN/Inf Detection Summary - {len(self.detections)} issues found")
        logger.error("=" * 80)
        
        # Group by issue type
        nan_detections = [d for d in self.detections if d['issue'] == 'NaN']
        inf_detections = [d for d in self.detections if d['issue'] == 'Inf']
        
        if nan_detections:
            logger.error(f"\nNaN Issues ({len(nan_detections)}):")
            for det in nan_detections:
                logger.error(
                    f"  - {det['module']} ({det['type']}): "
                    f"{det['count']}/{det['total']} values, "
                    f"shape={det['shape']}"
                )
        
        if inf_detections:
            logger.error(f"\nInf Issues ({len(inf_detections)}):")
            for det in inf_detections:
                logger.error(
                    f"  - {det['module']} ({det['type']}): "
                    f"{det['count']}/{det['total']} values, "
                    f"shape={det['shape']}"
                )
        
        logger.error("=" * 80)
    
    def get_first_issue_module(self) -> Optional[str]:
        """Get name of first module that produced NaN/Inf."""
        if self.detections:
            return self.detections[0]['module']
        return None
    
    def __enter__(self):
        """Context manager entry."""
        self.enabled = True
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.print_summary()
        return False


# Convenience function
def detect_nan_inf(
    model: nn.Module,
    raise_on_nan: bool = True,
    raise_on_inf: bool = False,
    verbose: bool = False,
) -> NaNInfDetectorHook:
    """
    Convenience function to create and register NaN/Inf detector.
    
    Args:
        model: PyTorch module to monitor
        raise_on_nan: Raise exception on NaN detection
        raise_on_inf: Raise exception on Inf detection
        verbose: Print stats for every layer
    
    Returns:
        NaNInfDetectorHook instance (call .remove() when done)
    
    Example:
        >>> detector = detect_nan_inf(model)
        >>> output = model(input)
        >>> detector.remove()
    """
    detector = NaNInfDetectorHook(
        raise_on_nan=raise_on_nan,
        raise_on_inf=raise_on_inf,
        verbose=verbose,
    )
    detector.register(model)
    return detector


# ============================================================================
# Trainer-compatible Hook (for config-based usage)
# ============================================================================

@HOOKS.register_module()
class NaNInfDetectorTrainerHook(HookBase):
    """
    Trainer-compatible NaN/Inf detector hook for config-based usage.
    
    This hook wraps the standalone NaNInfDetectorHook to work with
    the PointSpace trainer system. It can be configured directly in
    your config file.
    
    Args:
        raise_on_nan: bool - Raise exception when NaN detected (default: False in training)
        raise_on_inf: bool - Raise exception when Inf detected (default: False)
        print_stats: bool - Print detection statistics (default: True)
        check_input: bool - Also check module inputs (default: False)
        verbose: bool - Print info for every layer (default: False)
        check_interval: int - Check every N steps (default: 1, check every step)
        enabled_epochs: list - Only enable during specific epochs (default: all epochs)
    
    Config Example:
        hooks = [
            dict(type="NaNInfDetectorTrainerHook", 
                 raise_on_nan=True, 
                 verbose=False,
                 check_interval=10),
            # ... other hooks ...
        ]
    """
    
    def __init__(
        self,
        raise_on_nan: bool = False,  # Don't interrupt training by default
        raise_on_inf: bool = False,
        print_stats: bool = True,
        check_input: bool = False,
        verbose: bool = False,
        check_interval: int = 1,
        enabled_epochs: Optional[List[int]] = None,
    ):
        super().__init__()
        self.raise_on_nan = raise_on_nan
        self.raise_on_inf = raise_on_inf
        self.print_stats = print_stats
        self.check_input = check_input
        self.verbose = verbose
        self.check_interval = check_interval
        self.enabled_epochs = enabled_epochs
        
        self.detector = None
        self._step_count = 0
    
    def before_train(self):
        """Register detector to model before training starts."""
        # Create detector
        self.detector = NaNInfDetectorHook(
            raise_on_nan=self.raise_on_nan,
            raise_on_inf=self.raise_on_inf,
            print_stats=self.print_stats,
            check_input=self.check_input,
            enabled=True,  # Will be controlled by check_interval
            verbose=self.verbose,
        )
        
        # Register to trainer's model
        if hasattr(self.trainer, 'model'):
            import pointspace.utils.comm as comm
            if comm.get_world_size() > 1:
                # Distributed training - register to module
                model = self.trainer.model.module
            else:
                model = self.trainer.model
            
            self.detector.register(model, prefix="model")
            logger.info(
                f"NaN/Inf detector registered (check_interval={self.check_interval})"
            )
        else:
            logger.warning("Trainer has no model attribute, NaN detector not registered")
    
    def before_step(self):
        """Control detector enable state based on interval."""
        if self.detector is None:
            return
        
        # Check if current epoch should be monitored
        if self.enabled_epochs is not None:
            current_epoch = getattr(self.trainer, 'epoch', 0)
            if current_epoch not in self.enabled_epochs:
                self.detector.enabled = False
                return
        
        # Check interval
        self._step_count += 1
        self.detector.enabled = (self._step_count % self.check_interval == 0)
    
    def after_epoch(self):
        """Print summary after each epoch."""
        if self.detector and self.detector.detections:
            self.detector.print_summary()
            self.detector.reset_stats()
    
    def after_train(self):
        """Clean up hooks after training."""
        if self.detector:
            self.detector.print_summary()
            self.detector.remove()
            logger.info("NaN/Inf detector hooks removed")


# Alias for backward compatibility and convenience
NaNInfDetector = NaNInfDetectorHook

