"""Large-scene, LitePT-based adaptation of Point-M2AE pretraining.

The original Point-M2AE uses an FPS/KNN token pyramid. This variant keeps
LitePT's grid-pooling hierarchy fixed for a same-backbone Swiss comparison:
coarse spatial groups are masked consistently at every scale and absent from
the encoder. Learned mask queries propagate through two decoder stages; only
the fine masked XYZ neighborhoods are reconstructed, as in the source method.
The native LitePT hierarchy replaces the paper's FPS/KNN token hierarchy.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import pointops

from pointspace.models.builder import MODELS, build_model


def _offset_counts(offset: torch.Tensor) -> torch.Tensor:
    return torch.diff(F.pad(offset.long(), (1, 0)))


def _group_points(coord: torch.Tensor, size: float):
    """Return voxel cells, inverse point-to-group map and group counts."""
    origin = coord.amin(dim=0)
    cell = torch.floor((coord.float() - origin.float()) / size).long()
    extent = cell.amax(dim=0) + 1
    if int(torch.prod(extent).item()) >= 2**62:
        unique_cell, inverse, counts = torch.unique(
            cell, dim=0, sorted=True, return_inverse=True, return_counts=True
        )
    else:
        key = (cell[:, 0] * extent[1] + cell[:, 1]) * extent[2] + cell[:, 2]
        unique_key, inverse, counts = torch.unique(
            key, sorted=True, return_inverse=True, return_counts=True
        )
        unique_cell = torch.stack(
            (
                unique_key // (extent[1] * extent[2]),
                (unique_key // extent[2]) % extent[1],
                unique_key % extent[2],
            ),
            dim=1,
        )
    center = origin.float() + (unique_cell.float() + 0.5) * size
    return center, inverse, counts


def _sample_group_targets(
    coord: torch.Tensor,
    center: torch.Tensor,
    inverse: torch.Tensor,
    counts: torch.Tensor,
    selected: torch.Tensor,
    points_per_group: int,
    size: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample equal-size local-coordinate sets without per-group Python loops."""
    if selected.numel() == 0:
        return center.new_empty((0, 3)), coord.new_empty((0, points_per_group, 3))
    sorted_point = torch.argsort(inverse, stable=True)
    group_start = F.pad(counts.cumsum(0)[:-1], (1, 0))
    draw = torch.randint(
        0, 2**31 - 1,
        (selected.numel(), points_per_group),
        device=coord.device,
    )
    point_index = sorted_point[
        group_start[selected, None] + draw.remainder(counts[selected, None])
    ]
    centers = center[selected]
    target = (coord[point_index].float() - centers[:, None, :]) / size
    return centers, target.clamp(-0.5, 0.5)


@MODELS.register_module("PointM2AE-LitePT-v1m1")
class PointM2AELitePT(nn.Module):
    """Mask whole 3D groups; decode coarse-to-fine and reconstruct fine XYZ."""

    def __init__(
        self,
        backbone: dict,
        backbone_out_channels: int = 72,
        fine_channels: int = 72,
        coarse_channels: int = 504,
        coarse_level: int = 4,
        mask_group_size: float = 4.0,
        fine_group_size: float = 2.0,
        mask_ratio: float = 0.8,
        max_coarse_groups: int = 512,
        max_fine_groups: int = 1024,
        fine_points_per_group: int = 16,
        min_group_points: int = 4,
        neighbor_k: int = 8,
        fine_loss_weight: float = 1.0,
        checkpoint_decoder: bool = True,
    ):
        super().__init__()
        if not (0 < fine_group_size < mask_group_size):
            raise ValueError("fine_group_size must be in (0, mask_group_size)")
        scale_ratio = mask_group_size / fine_group_size
        if abs(scale_ratio - round(scale_ratio)) > 1e-6:
            raise ValueError("mask_group_size must be an integer multiple of fine_group_size")
        if not 0 < mask_ratio < 1:
            raise ValueError("mask_ratio must be in (0, 1)")
        if min(max_coarse_groups, max_fine_groups, neighbor_k,
               min_group_points, fine_points_per_group) < 1:
            raise ValueError("group limits, neighbor_k and target sizes must be positive")
        if backbone.get("type") != "LitePT-v1m4" or backbone.get("enc_mode") is not False:
            raise ValueError("PointM2AE-LitePT-v1m1 requires full LitePT-v1m4")
        if backbone_out_channels != fine_channels:
            raise ValueError("backbone_out_channels must match fine_channels")
        if fine_channels != backbone["dec_channels"][0]:
            raise ValueError("fine_channels must match LitePT's P0 decoder width")
        if not 0 <= coarse_level < len(backbone["enc_channels"]):
            raise ValueError("coarse_level is outside the LitePT encoder")
        if coarse_channels != backbone["enc_channels"][coarse_level]:
            raise ValueError("coarse_channels must match the selected encoder stage")
        self.backbone = build_model(backbone)
        self.coarse_level = coarse_level
        self.mask_group_size = float(mask_group_size)
        self.fine_group_size = float(fine_group_size)
        self.mask_ratio = mask_ratio
        self.max_coarse_groups = max_coarse_groups
        self.max_fine_groups = max_fine_groups
        self.fine_points_per_group = fine_points_per_group
        self.min_group_points = min_group_points
        self.neighbor_k = neighbor_k
        self.fine_loss_weight = fine_loss_weight
        self.checkpoint_decoder = bool(checkpoint_decoder)
        self.coarse_mask_token = nn.Parameter(torch.zeros(1, coarse_channels))
        self.fine_mask_token = nn.Parameter(torch.zeros(1, fine_channels))
        nn.init.trunc_normal_(self.coarse_mask_token, std=0.02)
        nn.init.trunc_normal_(self.fine_mask_token, std=0.02)
        self.coarse_pos_embed = nn.Sequential(
            nn.Linear(3, coarse_channels), nn.GELU(),
            nn.Linear(coarse_channels, coarse_channels),
        )
        self.coarse_decoder = nn.TransformerEncoderLayer(
            d_model=coarse_channels, nhead=6,
            dim_feedforward=2 * coarse_channels, dropout=0.0,
            batch_first=True, norm_first=True,
        )
        self.coarse_to_fine = nn.Linear(coarse_channels, fine_channels)
        self.fine_decoder = nn.Sequential(
            nn.Linear(3 * fine_channels + 3, fine_channels), nn.GELU(),
            nn.Linear(fine_channels, fine_channels),
        )
        self.fine_pos_embed = nn.Sequential(
            nn.Linear(3, fine_channels), nn.GELU(),
            nn.Linear(fine_channels, fine_channels),
        )
        self.fine_decoder_block = nn.TransformerEncoderLayer(
            d_model=fine_channels, nhead=4,
            dim_feedforward=2 * fine_channels, dropout=0.0,
            batch_first=True, norm_first=True,
        )
        self.fine_head = nn.Sequential(
            nn.Linear(fine_channels + 3, fine_channels * 2), nn.GELU(),
            nn.Linear(fine_channels * 2, fine_points_per_group * 3),
        )

    def _mask_and_targets(self, data_dict):
        coord = data_dict["coord"]
        offset = data_dict["offset"]
        visible = torch.ones(coord.shape[0], dtype=torch.bool, device=coord.device)
        origins = []
        centers = {"coarse": [], "fine": []}
        targets = {"coarse": [], "fine": []}
        query_counts = {"coarse": [], "fine": []}
        visible_counts = []
        start = 0
        for end in offset.tolist():
            if end <= start:
                raise ValueError("Each input batch sample must contain points")
            local = coord[start:end]
            origins.append(local.amin(dim=0))
            coarse_center, coarse_inverse, coarse_count = _group_points(
                local, self.mask_group_size
            )
            group_count = coarse_count.numel()
            # Keep at least one group and enough points visible per sample.
            num_mask = min(int(group_count * self.mask_ratio), group_count - 1)
            selected_mask = torch.randperm(group_count, device=coord.device)[:num_mask]
            local_visible = ~torch.isin(coarse_inverse, selected_mask)
            if int(local_visible.sum()) < max(16, self.neighbor_k):
                local_visible = torch.ones_like(local_visible)
                selected_mask = selected_mask[:0]
            visible[start:end] = local_visible
            visible_counts.append(int(local_visible.sum()))

            coarse_eligible = selected_mask[
                coarse_count[selected_mask] >= self.min_group_points
            ]
            coarse_eligible = coarse_eligible[
                torch.randperm(coarse_eligible.numel(), device=coord.device)
                [: self.max_coarse_groups]
            ]
            c_center = coarse_center[coarse_eligible]
            c_target = local.new_empty((len(c_center), 0, 3))
            centers["coarse"].append(c_center)
            targets["coarse"].append(c_target)
            query_counts["coarse"].append(c_center.shape[0])

            fine_center, fine_inverse, fine_count = _group_points(
                local, self.fine_group_size
            )
            fine_masked = torch.unique(fine_inverse[~local_visible])
            fine_eligible = fine_masked[
                fine_count[fine_masked] >= self.min_group_points
            ]
            fine_eligible = fine_eligible[
                torch.randperm(fine_eligible.numel(), device=coord.device)
                [: self.max_fine_groups]
            ]
            f_center, f_target = _sample_group_targets(
                local, fine_center, fine_inverse, fine_count,
                fine_eligible, self.fine_points_per_group, self.fine_group_size,
            )
            centers["fine"].append(f_center)
            targets["fine"].append(f_target)
            query_counts["fine"].append(f_center.shape[0])
            start = end

        filtered = {
            key: data_dict[key][visible]
            for key in ("coord", "grid_coord", "feat")
        }
        filtered["offset"] = torch.tensor(
            visible_counts, device=coord.device, dtype=offset.dtype
        ).cumsum(0)
        return (
            filtered,
            torch.stack(origins),
            {scale: (torch.cat(centers[scale]), torch.cat(targets[scale]),
                     torch.tensor(query_counts[scale], device=coord.device,
                                  dtype=offset.dtype).cumsum(0))
             for scale in ("coarse", "fine")},
            visible.float().mean(),
        )

    def _query_features(self, source, centers, query_offset, origins):
        if centers.shape[0] == 0:
            return source.feat.new_empty((0, source.feat.shape[1] + 3))
        source_counts = _offset_counts(source.offset)
        query_counts = _offset_counts(query_offset)
        k = min(self.neighbor_k, int(source_counts.min()))
        if k < 1:
            raise RuntimeError("No visible reference point in a batch sample")
        source_batch = torch.repeat_interleave(
            torch.arange(len(origins), device=centers.device), source_counts
        )
        query_batch = torch.repeat_interleave(
            torch.arange(len(origins), device=centers.device), query_counts
        )
        reference_local = (source.coord.float() - origins[source_batch].float()).contiguous()
        query_local = (centers.float() - origins[query_batch].float()).contiguous()
        with torch.no_grad():
            neighbor = pointops.knn_query_idx(
                k, reference_local, source.offset.int(),
                query_local, query_offset.int(),
            ).long()
            distance = (reference_local[neighbor] - query_local[:, None]).square().sum(-1)
            weight = (distance + 1e-4).rsqrt()
            weight = weight / weight.sum(dim=1, keepdim=True)
        feature = (source.feat[neighbor] * weight.to(source.feat.dtype)[..., None]).sum(1)
        return torch.cat((feature, (query_local / 100).to(feature.dtype)), dim=-1)

    @staticmethod
    def _chamfer(prediction, target):
        if prediction.shape[0] == 0:
            return prediction.sum() * 0
        with torch.autocast(device_type="cuda", enabled=False):
            distance = torch.cdist(prediction.float(), target.float()).square()
            return 0.5 * (distance.amin(-1).mean() + distance.amin(-2).mean())

    @staticmethod
    def _propagate_coarse(decoded, coarse_center, coarse_offset,
                          fine_center, fine_offset):
        """Carry decoded masked coarse queries into masked fine queries."""
        result = decoded.new_zeros((len(fine_center), decoded.shape[1]))
        c_start = f_start = 0
        for c_end, f_end in zip(coarse_offset.tolist(), fine_offset.tolist()):
            if f_end > f_start and c_end > c_start:
                nearest = torch.cdist(
                    fine_center[f_start:f_end].float(),
                    coarse_center[c_start:c_end].float(),
                ).argmin(-1)
                result[f_start:f_end] = decoded[c_start:c_end][nearest]
            c_start, f_start = c_end, f_end
        return result

    def _decode_coarse(self, coarse, centers, query_offset, origins):
        """Decode visible and masked coarse tokens together, per source tile."""
        decoded = []
        visible_start = query_start = 0
        for sample, (visible_end, query_end) in enumerate(
                zip(coarse.offset.tolist(), query_offset.tolist())):
            visible_feat = coarse.feat[visible_start:visible_end]
            visible_coord = (coarse.coord[visible_start:visible_end].float()
                             - origins[sample].float()) / 100.0
            masked_coord = (centers[query_start:query_end].float()
                            - origins[sample].float()) / 100.0
            if query_end > query_start:
                positions = torch.cat((visible_coord, masked_coord), dim=0)
                tokens = torch.cat((visible_feat,
                    self.coarse_mask_token.expand(query_end - query_start, -1)), dim=0)
                positions = self.coarse_pos_embed(positions).to(tokens.dtype)
                sequence = (tokens + positions)[None]
                output = (checkpoint(
                    self.coarse_decoder, sequence, use_reentrant=False,
                    preserve_rng_state=False,
                ) if self.training and self.checkpoint_decoder
                          else self.coarse_decoder(sequence))[0]
                decoded.append(output[len(visible_feat):])
            visible_start, query_start = visible_end, query_end
        if not decoded:
            return coarse.feat.new_empty((0, coarse.feat.shape[1]))
        return torch.cat(decoded, dim=0)

    def _decode_fine(self, features, positions, offset):
        decoded = []
        start = 0
        for end in offset.tolist():
            if end > start:
                sequence = features[start:end] + self.fine_pos_embed(
                    positions[start:end]).to(features.dtype)
                sequence = sequence[None]
                output = (checkpoint(
                    self.fine_decoder_block, sequence, use_reentrant=False,
                    preserve_rng_state=False,
                ) if self.training and self.checkpoint_decoder
                          else self.fine_decoder_block(sequence))[0]
                decoded.append(output)
            start = end
        return (torch.cat(decoded, dim=0) if decoded else
                features.new_empty(features.shape))

    def forward(
        self,
        data_dict,
        return_point_feature=False,
        feature_source="backbone",
        normalize_feature=False,
    ):
        if return_point_feature:
            if feature_source != "backbone":
                raise ValueError("PointM2AE feature_source must be 'backbone'")
            point = self.backbone(data_dict)
            feature = point.feat
            if "feature_point_mask" in data_dict:
                feature = feature[data_dict["feature_point_mask"]]
            if normalize_feature:
                feature = F.normalize(feature.float(), dim=-1)
            return {"point_feature": feature}
        filtered, origins, target, visible_fraction = self._mask_and_targets(data_dict)
        point, hierarchy = self.backbone(filtered, return_hierarchy=True)
        coarse = hierarchy[self.coarse_level].point
        c_center, c_target, c_offset = target["coarse"]
        f_center, f_target, f_offset = target["fine"]
        if c_center.shape[0] == 0 and f_center.shape[0] == 0:
            raise RuntimeError("No masked group meets min_group_points; check group sizes")
        f_feature = self._query_features(point, f_center, f_offset, origins)
        c_decoded = self._decode_coarse(coarse, c_center, c_offset, origins)
        propagated = self._propagate_coarse(
            c_decoded, c_center, c_offset, f_center, f_offset)
        f_decoded = self.fine_decoder(torch.cat((
            f_feature[:, :-3], self.fine_mask_token.expand(len(f_feature), -1),
            self.coarse_to_fine(propagated), f_feature[:, -3:]), dim=-1))
        f_decoded = self._decode_fine(f_decoded, f_feature[:, -3:], f_offset)
        f_prediction = self.fine_head(torch.cat((f_decoded, f_feature[:, -3:]), dim=-1))
        f_prediction = f_prediction.reshape(-1, self.fine_points_per_group, 3)
        loss_fine = self._chamfer(f_prediction, f_target)
        loss = self.fine_loss_weight * loss_fine
        return dict(
            loss=loss,
            loss_fine=loss_fine.detach(),
            visible_fraction=visible_fraction.detach(),
        )
