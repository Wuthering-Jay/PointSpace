"""Coordinate-derived geometric pretext baselines for CVRL Section 4.4 P1."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from pointspace.models.builder import MODELS, build_model
from pointspace.models.utils.structure import Point


@MODELS.register_module()
class GeometrySSL(nn.Module):
    """Predict offline normals or six local-shape descriptors.

    The backbone receives only the configured ``feat`` (XYZ + intensity for
    Swiss). LAS return attributes select the fairness mask but are never
    appended to ``feat`` or used as geometry targets.
    """

    def __init__(
        self,
        backbone,
        backbone_out_channels,
        objective="normal",
        head_dropout=0.1,
        smooth_l1_beta=0.1,
        ignore_index=-1,
    ):
        super().__init__()
        if objective not in {"normal", "local_shape"}:
            raise ValueError("GeometrySSL objective must be 'normal' or 'local_shape'")
        self.objective = objective
        self.backbone = build_model(backbone)
        self.ignore_index = int(ignore_index)
        self.smooth_l1_beta = float(smooth_l1_beta)
        if self.smooth_l1_beta <= 0:
            raise ValueError("smooth_l1_beta must be positive")
        channels = int(backbone_out_channels)
        self.head = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(channels, 3 if objective == "normal" else 6),
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

    @staticmethod
    def _backbone_input(input_dict):
        # Keep targets and labels entirely outside Point, even though LitePT
        # currently reads only feat/coord/grid_coord/offset.
        excluded = {"normal", "local_shape", "geom_valid", "return_role", "echo", "segment"}
        return Point({key: value for key, value in input_dict.items() if key not in excluded})

    def extract_point_feature(self, input_dict, normalize=False):
        feature = self._unwind_point(self.backbone(self._backbone_input(input_dict)))
        feature_point_mask = input_dict.get("feature_point_mask")
        if feature_point_mask is not None:
            feature_point_mask = feature_point_mask.reshape(-1).bool()
            if feature_point_mask.numel() != feature.shape[0]:
                raise ValueError("feature_point_mask length does not match geometry features")
            feature = feature[feature_point_mask]
        if normalize:
            feature = F.normalize(feature.float(), dim=-1, eps=1e-12)
        return feature

    def forward(
        self,
        input_dict,
        return_point_feature=False,
        feature_source="backbone",
        normalize_feature=False,
    ):
        if return_point_feature:
            if feature_source != "backbone":
                raise ValueError("GeometrySSL only supports backbone feature export")
            return {"point_feature": self.extract_point_feature(input_dict, normalize_feature)}

        target_key = "normal" if self.objective == "normal" else "local_shape"
        for key in (target_key, "geom_valid", "return_role"):
            if key not in input_dict:
                raise KeyError(f"GeometrySSL requires supervision key {key!r}")
        feature = self._unwind_point(self.backbone(self._backbone_input(input_dict)))
        predicted = self.head(feature)
        target = input_dict[target_key].reshape(-1, predicted.shape[1]).float()
        valid = input_dict["geom_valid"].reshape(-1).bool()
        role = input_dict["return_role"].reshape(-1)
        if target.shape != predicted.shape or valid.numel() != predicted.shape[0] or role.numel() != predicted.shape[0]:
            raise ValueError("Geometry target, mask, return role and backbone feature lengths differ")
        valid = valid & (role != self.ignore_index) & torch.isfinite(target).all(dim=1)
        if self.objective == "normal":
            valid = valid & (torch.linalg.vector_norm(target, dim=1) > 1e-6)
        if valid.any():
            if self.objective == "normal":
                prediction_unit = F.normalize(predicted[valid].float(), dim=1, eps=1e-12)
                target_unit = F.normalize(target[valid], dim=1, eps=1e-12)
                # PCA normal orientation is ambiguous, especially on facades.
                loss = (1 - (prediction_unit * target_unit).sum(dim=1).abs()).mean()
            else:
                loss = F.smooth_l1_loss(
                    predicted[valid].float(), target[valid], beta=self.smooth_l1_beta
                )
        else:
            loss = predicted.sum() * 0.0
        output = {"loss": loss, f"loss/{self.objective}": loss.detach()}
        if not self.training:
            output.update(geometry_prediction=predicted, geometry_target=target,
                          geometry_valid=valid)
        return output
