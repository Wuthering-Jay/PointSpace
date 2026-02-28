"""
Misc Losses

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .builder import LOSSES


@LOSSES.register_module()
class CrossEntropyLoss(nn.Module):
    def __init__(
        self,
        class_weight=None,
        size_average=None,
        reduce=None,
        reduction="mean",
        label_smoothing=0.0,
        loss_weight=1.0,
        ignore_index=-1,
        auto_class_weight=False,
    ):
        super(CrossEntropyLoss, self).__init__()
        self.loss_weight = loss_weight
        self.auto_class_weight = auto_class_weight
        self._ignore_index = ignore_index
        self._size_average = size_average
        self._reduce = reduce
        self._reduction = reduction
        self._label_smoothing = label_smoothing
        self._build_loss(
            torch.tensor(class_weight, dtype=torch.float)
            if class_weight is not None
            else None
        )

    def _build_loss(self, weight):
        self.loss = nn.CrossEntropyLoss(
            weight=weight,
            size_average=self._size_average,
            ignore_index=self._ignore_index,
            reduce=self._reduce,
            reduction=self._reduction,
            label_smoothing=self._label_smoothing,
        )

    def set_class_weight(self, class_weight):
        """Inject class weights computed from the dataset (called by Criteria)."""
        if isinstance(class_weight, np.ndarray):
            class_weight = torch.from_numpy(class_weight).float()
        elif isinstance(class_weight, (list, tuple)):
            class_weight = torch.tensor(class_weight, dtype=torch.float)
        self._build_loss(class_weight)

    def forward(self, pred, target):
        # Move weight to the same device as pred (handles CPU tests / GPU train)
        if self.loss.weight is not None and self.loss.weight.device != pred.device:
            self.loss.weight = self.loss.weight.to(pred.device)
        return self.loss(pred, target) * self.loss_weight


@LOSSES.register_module()
class SmoothCELoss(nn.Module):
    def __init__(self, smoothing_ratio=0.1):
        super(SmoothCELoss, self).__init__()
        self.smoothing_ratio = smoothing_ratio

    def forward(self, pred, target):
        eps = self.smoothing_ratio
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)
        loss = -(one_hot * log_prb).total(dim=1)
        loss = loss[torch.isfinite(loss)].mean()
        return loss


@LOSSES.register_module()
class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.5, logits=True, reduce=True, loss_weight=1.0):
        """Binary Focal Loss
        <https://arxiv.org/abs/1708.02002>`
        """
        super(BinaryFocalLoss, self).__init__()
        assert 0 < alpha < 1
        self.gamma = gamma
        self.alpha = alpha
        self.logits = logits
        self.reduce = reduce
        self.loss_weight = loss_weight

    def forward(self, pred, target, **kwargs):
        """Forward function.
        Args:
            pred (torch.Tensor): The prediction with shape (N)
            target (torch.Tensor): The ground truth. If containing class
                indices, shape (N) where each value is 0≤targets[i]≤1, If containing class probabilities,
                same shape as the input.
        Returns:
            torch.Tensor: The calculated loss
        """
        if self.logits:
            bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        else:
            bce = F.binary_cross_entropy(pred, target, reduction="none")
        pt = torch.exp(-bce)
        alpha = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal_loss = alpha * (1 - pt) ** self.gamma * bce

        if self.reduce:
            focal_loss = torch.mean(focal_loss)
        return focal_loss * self.loss_weight


@LOSSES.register_module()
class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma=2.0,
        alpha=0.5,
        class_weight=None,
        reduction="mean",
        loss_weight=1.0,
        ignore_index=-1,
        auto_class_weight=False,
    ):
        """Focal Loss
        <https://arxiv.org/abs/1708.02002>`

        Args:
            gamma: Focusing parameter.
            alpha: Balancing factor (scalar or per-class list).
            class_weight: Per-class weight tensor/list applied to the BCE term.
                Injected automatically when *auto_class_weight=True*.
            auto_class_weight: If True, accept injected class weights from
                ``Criteria.set_class_weight()``.
        """
        super(FocalLoss, self).__init__()
        assert reduction in (
            "mean",
            "sum",
        ), "AssertionError: reduction should be 'mean' or 'sum'"
        assert isinstance(
            alpha, (float, list)
        ), "AssertionError: alpha should be of type float"
        assert isinstance(gamma, float), "AssertionError: gamma should be of type float"
        assert isinstance(
            loss_weight, float
        ), "AssertionError: loss_weight should be of type float"
        assert isinstance(ignore_index, int), "ignore_index must be of type int"
        self.gamma = gamma
        self.alpha = alpha
        self.class_weight = (
            torch.tensor(class_weight, dtype=torch.float)
            if class_weight is not None
            else None
        )
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self.auto_class_weight = auto_class_weight

    def set_class_weight(self, class_weight):
        """Inject class weights computed from the dataset."""
        if isinstance(class_weight, np.ndarray):
            class_weight = torch.from_numpy(class_weight).float()
        elif isinstance(class_weight, (list, tuple)):
            class_weight = torch.tensor(class_weight, dtype=torch.float)
        self.class_weight = class_weight

    def forward(self, pred, target, **kwargs):
        """Forward function.
        Args:
            pred (torch.Tensor): The prediction with shape (N, C) where C = number of classes.
            target (torch.Tensor): The ground truth. If containing class
                indices, shape (N) where each value is 0≤targets[i]≤C−1, If containing class probabilities,
                same shape as the input.
        Returns:
            torch.Tensor: The calculated loss
        """
        # [B, C, d_1, d_2, ..., d_k] -> [C, B, d_1, d_2, ..., d_k]
        pred = pred.transpose(0, 1)
        # [C, B, d_1, d_2, ..., d_k] -> [C, N]
        pred = pred.reshape(pred.size(0), -1)
        # [C, N] -> [N, C]
        pred = pred.transpose(0, 1).contiguous()
        # (B, d_1, d_2, ..., d_k) --> (B * d_1 * d_2 * ... * d_k,)
        target = target.view(-1).contiguous()
        assert pred.size(0) == target.size(
            0
        ), "The shape of pred doesn't match the shape of target"
        valid_mask = target != self.ignore_index
        target = target[valid_mask]
        pred = pred[valid_mask]

        if len(target) == 0:
            return 0.0

        num_classes = pred.size(1)
        target = F.one_hot(target, num_classes=num_classes)

        alpha = self.alpha
        if isinstance(alpha, list):
            alpha = pred.new_tensor(alpha)
        pred_sigmoid = pred.sigmoid()
        target = target.type_as(pred)
        one_minus_pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
        focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * one_minus_pt.pow(
            self.gamma
        )

        loss = (
            F.binary_cross_entropy_with_logits(pred, target, reduction="none")
            * focal_weight
        )

        # Apply per-class weight if available (auto-injected or manual)
        if self.class_weight is not None:
            cw = self.class_weight.to(loss.device)
            # loss shape: (N, C) — multiply each class column by its weight
            loss = loss * cw.unsqueeze(0)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.total()
        return self.loss_weight * loss


@LOSSES.register_module()
class DiceLoss(nn.Module):
    def __init__(self, smooth=1, exponent=2, loss_weight=1.0, ignore_index=-1):
        """DiceLoss.
        This loss is proposed in `V-Net: Fully Convolutional Neural Networks for
        Volumetric Medical Image Segmentation <https://arxiv.org/abs/1606.04797>`_.
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.exponent = exponent
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

    def forward(self, pred, target, **kwargs):
        # [B, C, d_1, d_2, ..., d_k] -> [C, B, d_1, d_2, ..., d_k]
        pred = pred.transpose(0, 1)
        # [C, B, d_1, d_2, ..., d_k] -> [C, N]
        pred = pred.reshape(pred.size(0), -1)
        # [C, N] -> [N, C]
        pred = pred.transpose(0, 1).contiguous()
        # (B, d_1, d_2, ..., d_k) --> (B * d_1 * d_2 * ... * d_k,)
        target = target.view(-1).contiguous()
        assert pred.size(0) == target.size(
            0
        ), "The shape of pred doesn't match the shape of target"
        valid_mask = target != self.ignore_index
        target = target[valid_mask]
        pred = pred[valid_mask]

        pred = F.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        target = F.one_hot(
            torch.clamp(target.long(), 0, num_classes - 1), num_classes=num_classes
        )

        total_loss = 0
        for i in range(num_classes):
            if i != self.ignore_index:
                num = torch.sum(torch.mul(pred[:, i], target[:, i])) * 2 + self.smooth
                den = (
                    torch.sum(
                        pred[:, i].pow(self.exponent) + target[:, i].pow(self.exponent)
                    )
                    + self.smooth
                )
                dice_loss = 1 - num / den
                total_loss += dice_loss
        loss = total_loss / num_classes
        return self.loss_weight * loss


# =====================================================================
#  Regression Losses
# =====================================================================


@LOSSES.register_module()
class MSELoss(nn.Module):
    """Mean Squared Error loss for point-wise regression.

    Args:
        reduction (str): ``'mean'`` or ``'sum'``.
        loss_weight (float): Multiplier applied to the final loss.
        ignore_value (float | None): If not None, points whose target
            equals this value are excluded from the loss computation.
    """

    def __init__(self, reduction="mean", loss_weight=1.0, ignore_value=None):
        super().__init__()
        self.loss_weight = loss_weight
        self.ignore_value = ignore_value
        self.loss = nn.MSELoss(reduction=reduction)

    def forward(self, pred, target):
        pred = pred.reshape(-1)
        target = target.reshape(-1).float()
        if self.ignore_value is not None:
            mask = target != self.ignore_value
            pred = pred[mask]
            target = target[mask]
        if pred.numel() == 0:
            return pred.sum() * 0.0  # differentiable zero
        return self.loss(pred, target) * self.loss_weight


@LOSSES.register_module()
class L1Loss(nn.Module):
    """L1 (Mean Absolute Error) loss for point-wise regression.

    Args:
        reduction (str): ``'mean'`` or ``'sum'``.
        loss_weight (float): Multiplier applied to the final loss.
        ignore_value (float | None): If not None, points whose target
            equals this value are excluded from the loss computation.
    """

    def __init__(self, reduction="mean", loss_weight=1.0, ignore_value=None):
        super().__init__()
        self.loss_weight = loss_weight
        self.ignore_value = ignore_value
        self.loss = nn.L1Loss(reduction=reduction)

    def forward(self, pred, target):
        pred = pred.reshape(-1)
        target = target.reshape(-1).float()
        if self.ignore_value is not None:
            mask = target != self.ignore_value
            pred = pred[mask]
            target = target[mask]
        if pred.numel() == 0:
            return pred.sum() * 0.0
        return self.loss(pred, target) * self.loss_weight


@LOSSES.register_module()
class SmoothL1Loss(nn.Module):
    """Smooth L1 loss for point-wise regression.

    Combines the advantages of L1 and L2 loss – quadratic near zero,
    linear far from zero, controlled by *beta*.

    Args:
        beta (float): Threshold at which the loss transitions from
            quadratic to linear. Default 1.0.
        reduction (str): ``'mean'`` or ``'sum'``.
        loss_weight (float): Multiplier applied to the final loss.
        ignore_value (float | None): If not None, points whose target
            equals this value are excluded.
    """

    def __init__(self, beta=1.0, reduction="mean", loss_weight=1.0, ignore_value=None):
        super().__init__()
        self.loss_weight = loss_weight
        self.ignore_value = ignore_value
        self.loss = nn.SmoothL1Loss(beta=beta, reduction=reduction)

    def forward(self, pred, target):
        pred = pred.reshape(-1)
        target = target.reshape(-1).float()
        if self.ignore_value is not None:
            mask = target != self.ignore_value
            pred = pred[mask]
            target = target[mask]
        if pred.numel() == 0:
            return pred.sum() * 0.0
        return self.loss(pred, target) * self.loss_weight


@LOSSES.register_module()
class HuberLoss(nn.Module):
    """Huber loss for point-wise regression.

    Equivalent to Smooth L1 with *delta* controlling the quadratic/linear
    transition.

    Args:
        delta (float): Transition threshold. Default 1.0.
        reduction (str): ``'mean'`` or ``'sum'``.
        loss_weight (float): Multiplier applied to the final loss.
        ignore_value (float | None): If not None, points whose target
            equals this value are excluded.
    """

    def __init__(self, delta=1.0, reduction="mean", loss_weight=1.0, ignore_value=None):
        super().__init__()
        self.loss_weight = loss_weight
        self.ignore_value = ignore_value
        self.loss = nn.HuberLoss(delta=delta, reduction=reduction)

    def forward(self, pred, target):
        pred = pred.reshape(-1)
        target = target.reshape(-1).float()
        if self.ignore_value is not None:
            mask = target != self.ignore_value
            pred = pred[mask]
            target = target[mask]
        if pred.numel() == 0:
            return pred.sum() * 0.0
        return self.loss(pred, target) * self.loss_weight
