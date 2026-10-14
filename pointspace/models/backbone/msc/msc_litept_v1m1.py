"""Masked Scene Contrast on LitePT, with chunked *exact* InfoNCE.

The point-pair contrast, cross masks, view mixing, and masked radiometric
reconstruction follow MSC. Swiss intensity replaces unavailable RGB; masked
normal reconstruction remains optional. Row chunking changes only peak memory,
not the negative pool or objective.
"""

from __future__ import annotations

import random
import torch
import torch.nn as nn
import torch.nn.functional as F

import pointops

from pointspace.models.builder import MODELS, build_model


def _counts(offset):
    return torch.diff(F.pad(offset.long(), (1, 0)))


@MODELS.register_module("MSC-LitePT-v1m1")
class MSCLitePT(nn.Module):
    def __init__(
        self,
        backbone: dict,
        backbone_out_channels: int = 72,
        mask_group_size: float = 4.0,
        mask_rate: float = 0.4,
        matching_max_k: int = 8,
        matching_max_radius: float = 0.5,
        matching_max_pair: int = 8192,
        temperature: float = 0.4,
        nce_chunk_size: int = 256,
        reconstruct_normal: bool = False,
        reconstruct_intensity: bool = True,
        normal_k: int = 32,
        view1_mix_prob: float = 0.8,
        view2_mix_prob: float = 0.0,
        contrast_weight: float = 1.0,
        reconstruct_weight: float = 1.0,
    ):
        super().__init__()
        if backbone.get("type") != "LitePT-v1m4" or backbone.get("enc_mode") is not False:
            raise ValueError("MSC-LitePT-v1m1 requires full LitePT-v1m4")
        if backbone_out_channels != backbone["dec_channels"][0]:
            raise ValueError("backbone_out_channels must match LitePT P0 width")
        if not 0 <= mask_rate <= 0.5:
            raise ValueError("MSC disjoint view masks require mask_rate <= 0.5")
        if min(mask_group_size, matching_max_k, matching_max_radius,
               matching_max_pair, temperature, nce_chunk_size, normal_k) <= 0:
            raise ValueError("Invalid MSC mask/matching/partition configuration")
        self.backbone = build_model(backbone)
        self.mask_token = nn.Parameter(torch.zeros(1, backbone["in_channels"]))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.mask_group_size = mask_group_size
        self.mask_rate = mask_rate
        self.matching_max_k = matching_max_k
        self.matching_max_radius = matching_max_radius
        self.matching_max_pair = matching_max_pair
        self.temperature = temperature
        self.nce_chunk_size = nce_chunk_size
        self.reconstruct_normal = reconstruct_normal
        self.reconstruct_intensity = reconstruct_intensity
        self.normal_k = normal_k
        if not (0 <= view1_mix_prob <= 1 and 0 <= view2_mix_prob <= 1):
            raise ValueError("MSC view mixing probabilities must be in [0, 1]")
        self.view1_mix_prob = view1_mix_prob
        self.view2_mix_prob = view2_mix_prob
        self.contrast_weight = contrast_weight
        self.reconstruct_weight = reconstruct_weight
        self.normal_head = nn.Linear(backbone_out_channels, 3) if reconstruct_normal else None
        self.intensity_head = (nn.Linear(backbone_out_channels, 1)
                               if reconstruct_intensity else None)

    def _cross_view_masks(self, first, second):
        coord1, coord2 = first["origin_coord"], second["origin_coord"]
        offset1, offset2 = first["offset"], second["offset"]
        mask1 = torch.zeros(len(coord1), dtype=torch.bool, device=coord1.device)
        mask2 = torch.zeros(len(coord2), dtype=torch.bool, device=coord2.device)
        start1 = start2 = 0
        for end1, end2 in zip(offset1.tolist(), offset2.tolist()):
            union = torch.cat((coord1[start1:end1], coord2[start2:end2]))
            origin = union.amin(dim=0)
            cell = torch.floor((union.float() - origin.float()) / self.mask_group_size).long()
            _, inverse = torch.unique(cell, dim=0, return_inverse=True)
            group_count = int(inverse.max()) + 1
            group_mask_count = int(group_count * self.mask_rate)
            permutation = torch.randperm(group_count, device=union.device)
            first_groups = permutation[:group_mask_count]
            second_groups = permutation[group_mask_count:2 * group_mask_count]
            mask1[start1:end1] = torch.isin(inverse[:end1 - start1], first_groups)
            mask2[start2:end2] = torch.isin(inverse[end1 - start1:], second_groups)
            start1, start2 = end1, end2
        return mask1, mask2

    def _view_features(self, view, mask, mix_prob):
        masked_feat = torch.where(mask[:, None], self.mask_token.to(view["feat"].dtype), view["feat"])
        backbone_input = dict(
            coord=view["coord"], feat=masked_feat,
            grid_coord=view["grid_coord"], offset=view["offset"],
        )
        if len(view["offset"]) > 1 and random.random() < mix_prob:
            offset = view["offset"]
            backbone_input["offset"] = torch.cat(
                (offset[1:-1:2], offset[-1:]))
        point = self.backbone(backbone_input)
        return point.feat

    def _matched_pairs(self, first, second):
        coord1, coord2 = first["origin_coord"], second["origin_coord"]
        offset1, offset2 = first["offset"], second["offset"]
        counts1, counts2 = _counts(offset1), _counts(offset2)
        if len(counts1) != len(counts2) or int(counts2.min()) == 0:
            raise ValueError("MSC views must contain points from the same nonempty tiles")
        origin = []
        start = 0
        for end in offset1.tolist():
            origin.append(coord1[start:end].amin(0))
            start = end
        origin = torch.stack(origin)
        batch1 = torch.repeat_interleave(torch.arange(len(origin), device=coord1.device), counts1)
        batch2 = torch.repeat_interleave(torch.arange(len(origin), device=coord1.device), counts2)
        local1 = (coord1.float() - origin[batch1].float()).contiguous()
        local2 = (coord2.float() - origin[batch2].float()).contiguous()
        k = min(self.matching_max_k, int(counts2.min()))
        with torch.no_grad():
            neighbor = pointops.knn_query_idx(
                k, local2, offset2.int(), local1, offset1.int()
            ).long()
            distance2 = (local2[neighbor] - local1[:, None]).square().sum(-1)
            eligible = distance2 <= self.matching_max_radius**2
            random_score = torch.rand_like(distance2).masked_fill(~eligible, -1)
            selected_neighbor = neighbor.gather(1, random_score.argmax(1, keepdim=True)).squeeze(1)
            valid = eligible.any(1)
            anchor = torch.nonzero(valid, as_tuple=False).flatten()
            if anchor.numel() > self.matching_max_pair:
                anchor = anchor[torch.randperm(anchor.numel(), device=anchor.device)
                                [:self.matching_max_pair]]
        return anchor, selected_neighbor[anchor]

    def _contrastive_loss(self, feature1, feature2, anchor, positive):
        if anchor.numel() < 2:
            raise RuntimeError(
                "MSC found fewer than two matched points per tile; "
                "check matching_max_radius and view transforms"
            )
        z1 = F.normalize(feature1[anchor].float(), dim=1)
        z2 = F.normalize(feature2[positive].float(), dim=1)
        # All selected positives share one global negative pool, as in MSC.
        # Splitting query rows is exactly equivalent to a full N x N matrix.
        loss_sum = z1.new_zeros(())
        positive_sum = z1.new_zeros(())
        negative_sum = z1.new_zeros(())
        for start in range(0, len(anchor), self.nce_chunk_size):
            stop = min(start + self.nce_chunk_size, len(anchor))
            similarity = z1[start:stop] @ z2.T
            labels = torch.arange(start, stop, device=similarity.device)
            loss_sum = loss_sum + F.cross_entropy(
                similarity / self.temperature, labels, reduction="sum")
            positive_sum = positive_sum + similarity.gather(1, labels[:, None]).sum().detach()
            negative_sum = negative_sum + (similarity.sum() -
                similarity.gather(1, labels[:, None]).sum()).detach()
        count = len(anchor)
        return (loss_sum / count, positive_sum / count,
                negative_sum / (count * (count - 1)), count)

    @torch.no_grad()
    def _estimate_normals(self, view):
        if "normal" in view:
            return F.normalize(view["normal"].float(), dim=-1)
        coord = view["origin_coord"].float().contiguous()
        offset = view["offset"].int()
        k = min(self.normal_k, int(_counts(offset).min()))
        index = pointops.knn_query_idx(k, coord, offset, coord, offset).long()
        with torch.autocast(device_type=coord.device.type, enabled=False):
            local = coord[index] - coord[:, None]
            local = local - local.mean(1, keepdim=True)
            covariance = local.transpose(1, 2) @ local / max(k, 1)
            normal = torch.linalg.eigh(covariance).eigenvectors[:, :, 0]
        normal = torch.where(normal[:, 2:3] < 0, -normal, normal)
        return F.normalize(normal, dim=-1)

    def _normal_loss(self, feature1, feature2, first, second, mask1, mask2):
        losses = []
        for feature, view, mask in ((feature1, first, mask1),
                                    (feature2, second, mask2)):
            if mask.any():
                target = self._estimate_normals(view)[mask]
                prediction = F.normalize(self.normal_head(feature[mask]).float(), dim=-1)
                losses.append((1 - F.cosine_similarity(prediction, target)).sum())
        if not losses:
            return feature1.sum() * 0
        return torch.stack(losses).sum() / (mask1.sum() + mask2.sum()).clamp_min(1)

    def forward(
        self, data_dict, return_point_feature=False,
        feature_source="backbone", normalize_feature=False,
    ):
        if return_point_feature:
            if feature_source != "backbone":
                raise ValueError("MSC feature_source must be 'backbone'")
            feature = self.backbone(data_dict).feat
            if "feature_point_mask" in data_dict:
                feature = feature[data_dict["feature_point_mask"]]
            if normalize_feature:
                feature = F.normalize(feature.float(), dim=-1)
            return {"point_feature": feature}
        first = {key[6:]: value for key, value in data_dict.items() if key.startswith("view1_")}
        second = {key[6:]: value for key, value in data_dict.items() if key.startswith("view2_")}
        mask1, mask2 = self._cross_view_masks(first, second)
        feature1 = self._view_features(first, mask1, self.view1_mix_prob)
        feature2 = self._view_features(second, mask2, self.view2_mix_prob)
        anchor, positive_index = self._matched_pairs(first, second)
        nce, positive, negative, num_pairs = self._contrastive_loss(
            feature1, feature2, anchor, positive_index)
        normal = (self._normal_loss(feature1, feature2, first, second, mask1, mask2)
                  if self.reconstruct_normal else nce.new_zeros(()))
        intensity = nce.new_zeros(())
        if self.reconstruct_intensity:
            if first["feat"].shape[1] != 4 or second["feat"].shape[1] != 4:
                raise ValueError("MSC intensity reconstruction expects XYZ+I input")
            numerator = nce.new_zeros(())
            for feature, view, mask in ((feature1, first, mask1),
                                        (feature2, second, mask2)):
                if mask.any():
                    prediction = self.intensity_head(feature[mask]).float()
                    target = view["feat"][mask, -1:].float()
                    numerator = numerator + F.mse_loss(prediction, target, reduction="sum")
            intensity = numerator / (mask1.sum() + mask2.sum()).clamp_min(1)
        loss = (self.contrast_weight * nce +
                self.reconstruct_weight * (normal + intensity))
        return dict(
            loss=loss,
            nce=nce.detach(),
            normal=normal.detach(),
            intensity=intensity.detach(),
            positive_similarity=positive,
            negative_similarity=negative,
            matched_pairs=torch.tensor(num_pairs, device=loss.device),
            mask_fraction=0.5 * (mask1.float().mean() + mask2.float().mean()),
        )
