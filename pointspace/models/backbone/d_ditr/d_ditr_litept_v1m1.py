"""D-DITR point-level cosine distillation adapted to Swiss LAS + LitePT.

The regression head is discarded after pretraining. The only backbone input
is XYZ(+I); offline DINO features and point-to-patch mapping are targets only.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from pointspace.models.builder import MODELS, build_model


@MODELS.register_module("D-DITR-LitePT-v1m1")
class DDITRLitePT(nn.Module):
    def __init__(self, backbone, backbone_out_channels=72,
                 teacher_channels=1024, loss_chunk_size=4096,
                 checkpoint_loss=True, normalize_teacher_once=True):
        super().__init__()
        if backbone.get("type") != "LitePT-v1m4" or backbone.get("enc_mode") is not False:
            raise ValueError("D-DITR-LitePT-v1m1 requires full LitePT-v1m4")
        if backbone_out_channels != backbone["dec_channels"][0]:
            raise ValueError("backbone_out_channels must match LitePT P0 width")
        if teacher_channels <= 0 or loss_chunk_size <= 0:
            raise ValueError("D-DITR teacher channels and loss chunk must be positive")
        self.backbone = build_model(backbone)
        self.regression_head = nn.Linear(backbone_out_channels, teacher_channels)
        self.teacher_channels = teacher_channels
        self.loss_chunk_size = loss_chunk_size
        self.checkpoint_loss = checkpoint_loss
        self.normalize_teacher_once = bool(normalize_teacher_once)

    def _chunk_loss(self, student, teacher):
        prediction = F.normalize(
            self.regression_head(student).float(), dim=-1, eps=1e-6
        )
        if not self.normalize_teacher_once:
            teacher = F.normalize(teacher.float(), dim=-1, eps=1e-6)
        return (1 - (prediction * teacher).sum(dim=-1)).sum()

    def forward(self, data_dict, return_point_feature=False,
                feature_source="backbone", normalize_feature=False):
        if feature_source != "backbone":
            raise ValueError("D-DITR feature_source must be 'backbone'")
        if not return_point_feature:
            for key in ("image_patch_index", "image_valid", "dino_feature"):
                if key not in data_dict:
                    raise KeyError(f"D-DITR requires {key} supervision field")
            patch_index = data_dict["image_patch_index"].long()
            valid = data_dict["image_valid"].bool() & (patch_index >= 0)
            indices = torch.nonzero(valid, as_tuple=False).flatten()
            if indices.numel() == 0:
                raise RuntimeError("D-DITR batch has no valid point-to-image correspondences")
        point_feature = self.backbone(data_dict).feat
        if return_point_feature:
            if "feature_point_mask" in data_dict:
                point_feature = point_feature[data_dict["feature_point_mask"]]
            if normalize_feature:
                point_feature = F.normalize(point_feature.float(), dim=-1)
            return {"point_feature": point_feature}

        if patch_index.shape != (len(point_feature),) or valid.shape != patch_index.shape:
            raise ValueError("D-DITR point-to-patch mapping length mismatch")
        teacher = data_dict["dino_feature"]
        if teacher.ndim != 2 or teacher.shape[1] != self.teacher_channels:
            raise ValueError("D-DITR teacher feature channel mismatch")
        if int(patch_index[indices].max()) >= len(teacher):
            raise ValueError("D-DITR patch index exceeds teacher feature length")
        if self.normalize_teacher_once:
            # A patch is commonly referenced by many points. Normalizing the
            # compact patch table once is mathematically equivalent to
            # repeatedly normalizing every gathered copy in cosine loss.
            with torch.no_grad():
                teacher = F.normalize(teacher.float(), dim=-1, eps=1e-6)
        loss_sum = point_feature.new_zeros((), dtype=torch.float32)
        for chunk in indices.split(self.loss_chunk_size):
            student_chunk = point_feature[chunk]
            teacher_chunk = teacher[patch_index[chunk]]
            if self.checkpoint_loss and self.training:
                chunk_loss = checkpoint(self._chunk_loss, student_chunk,
                                        teacher_chunk, use_reentrant=False,
                                        preserve_rng_state=False)
            else:
                chunk_loss = self._chunk_loss(student_chunk, teacher_chunk)
            loss_sum = loss_sum + chunk_loss
        loss = loss_sum / indices.numel()
        return dict(loss=loss, cosine=loss.detach(),
                    visible_fraction=valid.float().mean().detach(),
                    supervised_points=indices.new_tensor(indices.numel()))
