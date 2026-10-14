"""Concerto-style joint 3D self-distillation and offline 2D patch prediction.

The 3D branch is the ALS-sized Sonata adaptation. Image features are frozen
precomputed targets, never inputs. A linear projector is used after mean
pooling within each image patch; because it is affine, this is exactly
equivalent to projecting every point and then averaging within the patch.
"""

import torch
from torch import nn
from torch.nn import functional as F

from pointspace.models.builder import MODELS
from pointspace.models.backbone.sonata.sonata_litept_v1m1 import SonataLitePT


@MODELS.register_module("Concerto-LitePT-v1m1")
class ConcertoLitePT(SonataLitePT):
    def __init__(self, teacher_channels=1024, intra_weight=0.5,
                 cross_weight=0.5, cross_loss_scale=10.0,
                 cross_cos_shift=True, **kwargs):
        super().__init__(**kwargs)
        if teacher_channels <= 0 or intra_weight < 0 or cross_weight < 0:
            raise ValueError("Invalid Concerto loss/channel configuration")
        if intra_weight + cross_weight <= 0 or cross_loss_scale <= 0:
            raise ValueError("Concerto requires a positive loss weight and scale")
        self.patch_projector = nn.Linear(kwargs.get("backbone_out_channels", 72),
                                         teacher_channels)
        self.teacher_channels = teacher_channels
        self.intra_weight = float(intra_weight)
        self.cross_weight = float(cross_weight)
        self.cross_loss_scale = float(cross_loss_scale)
        self.cross_cos_shift = bool(cross_cos_shift)

    def _cross_modal_loss(self, point_feature, data_dict):
        for key in ("image_patch_index", "image_valid", "dino_feature"):
            if key not in data_dict:
                raise KeyError(f"Concerto requires {key} supervision field")
        patch_index = data_dict["image_patch_index"].long()
        valid = data_dict["image_valid"].bool() & (patch_index >= 0)
        if patch_index.shape != (len(point_feature),) or valid.shape != patch_index.shape:
            raise ValueError("Concerto first-view correspondence length mismatch")
        teacher = data_dict["dino_feature"]
        if teacher.ndim != 2 or teacher.shape[1] != self.teacher_channels:
            raise ValueError("Concerto DINO channel mismatch")
        if not valid.any():
            return point_feature.new_zeros(()), 0
        index = patch_index[valid]
        if int(index.max()) >= len(teacher):
            raise ValueError("Concerto patch index exceeds DINO feature length")
        used, inverse, counts = torch.unique(index, sorted=True,
                                             return_inverse=True,
                                             return_counts=True)
        # Accumulate the compact 72D features, not a huge [N, 1024] tensor.
        pooled = point_feature.new_zeros((len(used), point_feature.shape[1]))
        pooled.index_add_(0, inverse, point_feature[valid])
        pooled = pooled / counts[:, None]
        predicted = self.patch_projector(pooled).float()
        target = teacher[used].float()
        if self.cross_cos_shift:
            predicted = predicted - predicted.mean(-1, keepdim=True)
            target = target - target.mean(-1, keepdim=True)
        loss = (1 - F.cosine_similarity(predicted, target, dim=-1,
                                        eps=1e-6)).mean() * self.cross_loss_scale
        return loss, len(used)

    def forward(self, data_dict, return_point_feature=False,
                feature_source="backbone", normalize_feature=False):
        if return_point_feature:
            return super().forward(
                data_dict, return_point_feature=True,
                feature_source=feature_source,
                normalize_feature=normalize_feature,
            )
        result = super().forward(data_dict, _return_student_view1=True)
        student_feature = result.pop("_student_view1_feature")
        cross, num_patches = self._cross_modal_loss(student_feature, data_dict)
        intra = result["loss"]
        if num_patches == 0:
            # Keep image-free blocks at the normal 3D objective scale.
            result["loss"] = intra
        else:
            result["loss"] = intra * self.intra_weight + cross * self.cross_weight
        result["intra"] = intra.detach()
        result["cross"] = cross.detach()
        result["supervised_patches"] = student_feature.new_tensor(num_patches,
                                                                   dtype=torch.long)
        return result
