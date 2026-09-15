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
        head_dropout=0.1,
        ignore_index=-1,
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.knn_k = int(knn_k)
        self.lambda_ctx = float(lambda_ctx)
        self.ignore_index = int(ignore_index)
        self.role_head = self._build_head(backbone_out_channels, head_dropout)
        self.context_head = self._build_head(backbone_out_channels, head_dropout)

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
        # Build targets before the backbone so temporary [N, K] KNN indices
        # are released before activation memory reaches its peak.
        context_target, context_valid = build_knn_return_context(
            coord=input_dict["coord"],
            offset=input_dict["offset"],
            return_role=role_target,
            k=self.knn_k,
            ignore_index=self.ignore_index,
        )
        point = Point(input_dict)
        feat = self._unwind_point(self.backbone(point))
        role_logits = self.role_head(feat)
        context_logits = self.context_head(feat)
        role_valid = role_target != self.ignore_index
        if role_valid.any():
            loss_role = self.criteria(role_logits, role_target)
        else:
            loss_role = role_logits.sum() * 0.0
        # RRCL exposes task-level sub-losses below. Suppress the
        # trainer's generic per-criterion ``cross_entropy`` duplicate, also
        # clearing stale entries when a malformed batch has no valid roles.
        self.criteria._last_individual_losses.clear()
        if context_valid.any():
            log_prob = F.log_softmax(
                context_logits[context_valid].float(), dim=-1
            )
            loss_ctx = -(
                context_target[context_valid] * log_prob
            ).sum(dim=-1).mean()
        else:
            loss_ctx = context_logits.sum() * 0.0
        loss = loss_role + self.lambda_ctx * loss_ctx

        output = {
            "loss": loss,
            "loss/role": loss_role.detach(),
            "loss/ctx": loss_ctx.detach(),
        }
        if not self.training:
            output.update(
                role_logits=role_logits,
                context_logits=context_logits,
                context_target=context_target,
                context_valid=context_valid,
            )
        return output
