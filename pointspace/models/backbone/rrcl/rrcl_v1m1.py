"""RRCL (Return Role and Context Learning) for LiDAR self-supervision."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from pointspace.models.losses import build_criteria
from pointspace.models.utils.structure import Point
from .ops import (
    NUM_RETURN_ROLES,
    build_knn_return_context,
    build_return_role,
)

from pointspace.models.builder import MODELS, build_model


@MODELS.register_module()
class RRCL(nn.Module):
    """Return-role and neighborhood-context pretraining wrapper."""

    def __init__(
        self,
        backbone_out_channels,
        backbone=None,
        criteria=None,
        knn_k=32,
        lambda_ctx=1.0,
        objective="full",
        head_dropout=0.1,
        ignore_index=-1,
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.knn_k = int(knn_k)
        self.lambda_ctx = float(lambda_ctx)
        if objective not in {"full", "role", "context"}:
            raise ValueError("RRCL objective must be 'full', 'role', or 'context'")
        self.objective = objective
        self.ignore_index = int(ignore_index)
        self.role_head = (
            self._build_head(backbone_out_channels, head_dropout)
            if objective in {"full", "role"} else None
        )
        self.context_head = (
            self._build_head(backbone_out_channels, head_dropout)
            if objective in {"full", "context"} else None
        )

    @staticmethod
    def _build_head(channels, dropout):
        return nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels, NUM_RETURN_ROLES),
        )

    @staticmethod
    def _unwind_point(point):
        if not isinstance(point, Point):
            return point
        while "pooling_parent" in point.keys():
            if "pooling_inverse" not in point.keys():
                raise KeyError("pooled Point is missing pooling_inverse")
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = parent
        return point.feat

    def extract_point_feature(self, input_dict, normalize=False):
        """Return decoder features in the current fragment's input order."""
        point = Point(input_dict)
        feat = self._unwind_point(self.backbone(point))
        feature_point_mask = input_dict.get("feature_point_mask")
        if feature_point_mask is not None:
            feature_point_mask = feature_point_mask.reshape(-1).bool()
            if feature_point_mask.numel() != feat.shape[0]:
                raise ValueError(
                    "feature_point_mask length does not match RRCL features: "
                    f"{feature_point_mask.numel()} != {feat.shape[0]}"
                )
            feat = feat[feature_point_mask]
        if normalize:
            feat = F.normalize(feat.float(), dim=-1, eps=1e-12)
        return feat

    def forward(
        self,
        input_dict,
        return_point_feature=False,
        feature_source="backbone",
        normalize_feature=False,
    ):
        # Feature export intentionally needs neither return_role nor KNN.  It
        # is the exact backbone representation retained after SSL pretraining.
        if return_point_feature:
            if feature_source != "backbone":
                raise ValueError("RRCL only supports feature_source='backbone'")
            return {
                "point_feature": self.extract_point_feature(
                    input_dict, normalize=normalize_feature
                )
            }
        if "return_role" not in input_dict:
            raise KeyError("RRCL requires the supervision key 'return_role'")
        role_target = input_dict["return_role"].reshape(-1).long()
        # Only context/full need KNN; role-only must not pay for this target.
        if self.context_head is not None:
            # Build before the backbone so temporary [N, K] indices are freed
            # before activation memory reaches its peak.
            context_target, context_valid = build_knn_return_context(
                coord=input_dict["coord"],
                offset=input_dict["offset"],
                return_role=role_target,
                k=self.knn_k,
                ignore_index=self.ignore_index,
            )
        point = Point(input_dict)
        feat = self._unwind_point(self.backbone(point))
        output = {}
        loss = feat.sum() * 0.0
        if self.role_head is not None:
            role_logits = self.role_head(feat)
            role_valid = role_target != self.ignore_index
            loss_role = (
                self.criteria(role_logits, role_target)
                if role_valid.any() else role_logits.sum() * 0.0
            )
            loss = loss + loss_role
            output["loss/role"] = loss_role.detach()
        # Suppress the trainer's generic cross_entropy duplicate and clear
        # stale criterion logging when the role objective is inactive.
        self.criteria._last_individual_losses.clear()
        if self.context_head is not None:
            context_logits = self.context_head(feat)
            if context_valid.any():
                log_prob = F.log_softmax(
                    context_logits[context_valid].float(), dim=-1
                )
                loss_ctx = -(
                    context_target[context_valid] * log_prob
                ).sum(dim=-1).mean()
            else:
                loss_ctx = context_logits.sum() * 0.0
            loss = loss + self.lambda_ctx * loss_ctx
            output["loss/ctx"] = loss_ctx.detach()
        output["loss"] = loss
        if not self.training:
            if self.role_head is not None:
                output["role_logits"] = role_logits
            if self.context_head is not None:
                output.update(
                    context_logits=context_logits,
                    context_target=context_target,
                    context_valid=context_valid,
                )
        return output
