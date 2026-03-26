"""
Superpoint Consistency Loss

Encourages prediction consistency within geometrically coherent superpoints.
Only active during training; disabled during validation/testing.
"""

import torch
import torch.nn as nn

try:
    from torch_scatter import scatter_mean
    TORCH_SCATTER_AVAILABLE = True
except ImportError:
    TORCH_SCATTER_AVAILABLE = False

from .builder import LOSSES


@LOSSES.register_module()
class SuperpointConsistencyLoss(nn.Module):
    """
    Superpoint Consistency Loss.

    Encourages points within the same superpoint to have similar predictions,
    while being tolerant of superpoints that may have been over-segmented
    (i.e., contain points from multiple semantic classes).

    The loss computes the L2 distance between each point's prediction probability
    and the mean probability of its superpoint. Superpoints with high internal
    disagreement (conflict) are excluded from the loss computation.

    Args:
        conflict_margin (float): Tolerance threshold for superpoint conflicts.
            Superpoints with internal disagreement above this threshold are
            excluded from the loss. Typical values: 0.05 ~ 0.2.
            Smaller values -> more conservative, only use highly consistent superpoints.
            Larger values -> more aggressive, use more superpoints but risk noise.
            Default: 0.1
        loss_weight (float): Weight multiplier for this loss term.
            Default: 1.0
        train_only (bool): If True, returns zero loss during evaluation.
            Default: True
        warmup_epochs (int): Number of epochs to wait before activating this loss.
            During warmup, the loss returns 0 to allow the network to stabilize.
            Default: 0 (no warmup)

    Forward Args:
        logits (torch.Tensor): Network output logits, shape (N, C)
        superpoint (torch.Tensor): Superpoint indices for each point, shape (N,)
            Points with superpoint < 0 are ignored.

    Returns:
        torch.Tensor: Scalar loss value
    """

    def __init__(self, conflict_margin=0.1, loss_weight=1.0, train_only=True, warmup_epochs=0):
        super().__init__()
        if not TORCH_SCATTER_AVAILABLE:
            raise ImportError(
                "SuperpointConsistencyLoss requires torch_scatter. "
                "Install it via: pip install torch-scatter"
            )
        self.conflict_margin = conflict_margin
        self.loss_weight = loss_weight
        self.train_only = train_only
        self.warmup_epochs = warmup_epochs

    def forward(self, logits, superpoint):
        """
        Compute superpoint consistency loss.

        Args:
            logits: Network output logits, shape (N, C)
            superpoint: Superpoint indices, shape (N,)

        Returns:
            Scalar loss tensor
        """
        # Skip during evaluation if train_only=True
        if self.train_only and not self.training:
            return logits.sum() * 0.0  # Differentiable zero

        # Skip during warmup period (read current epoch from RuntimeInfoHook)
        if self.warmup_epochs > 0:
            try:
                from pointspace.engines.hooks.misc import RuntimeInfoHook
                current_epoch = RuntimeInfoHook.state.get("epoch", 0)
                if current_epoch < self.warmup_epochs:
                    return logits.sum() * 0.0
            except ImportError:
                # Fallback: if RuntimeInfoHook not available, skip warmup check
                pass

        # Validate inputs
        if superpoint is None:
            return logits.sum() * 0.0

        # Filter out invalid superpoints (< 0)
        valid_mask = superpoint >= 0
        if not valid_mask.any():
            return logits.sum() * 0.0

        logits = logits[valid_mask]
        superpoint = superpoint[valid_mask]

        # Remap superpoint IDs to contiguous range [0, num_superpoints)
        unique_sp, sp_ids = torch.unique(superpoint, return_inverse=True)
        num_superpoints = len(unique_sp)

        if num_superpoints == 0:
            return logits.sum() * 0.0

        # 1. Convert logits to probability distribution (N, C)
        probs = torch.softmax(logits, dim=1)

        # 2. Compute mean prediction probability for each superpoint
        # sp_mean_probs shape: (num_superpoints, C)
        sp_mean_probs = scatter_mean(probs, sp_ids, dim=0)
        sp_mean_probs_detached = sp_mean_probs.detach()

        # 3. Broadcast mean probability back to each point (N, C)
        point_sp_mean = sp_mean_probs_detached[sp_ids]

        # 4. Compute L1 distance between each point and its superpoint mean (N,)
        # Larger distance -> point is more "inconsistent" with its superpoint
        point_diff_l1 = torch.sum(torch.abs(probs - point_sp_mean), dim=1)

        # 5. Compute conflict score for each superpoint (num_superpoints,)
        # High conflict -> superpoint contains diverse predictions (possibly over-segmented)
        sp_conflict = scatter_mean(point_diff_l1, sp_ids, dim=0)

        # 6. Generate exemption mask: superpoints with conflict > margin are excluded
        valid_sp_mask = (sp_conflict < self.conflict_margin).float()

        # 7. Broadcast mask back to points (N,)
        valid_point_mask = valid_sp_mask[sp_ids]

        # 8. Compute loss only for points in valid (consistent) superpoints
        masked_diff = point_diff_l1 * valid_point_mask
        valid_count = valid_point_mask.sum()

        if valid_count < 1:
            return logits.sum() * 0.0

        loss = masked_diff.sum() / (valid_count + 1e-6)

        return loss * self.loss_weight
