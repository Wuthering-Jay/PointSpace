import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
import torch_cluster
from peft import LoraConfig, get_peft_model
from collections import OrderedDict

from pointspace.models.losses import build_criteria
from pointspace.models.utils.structure import Point
from pointspace.models.utils import offset2batch
from .builder import MODELS, build_model


@MODELS.register_module()
class DefaultSegmentor(nn.Module):
    def __init__(self, backbone=None, criteria=None, aux_criteria=None, aux_weights=None):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.aux_criteria = build_criteria(aux_criteria) if aux_criteria else None
        self.aux_weights = aux_weights

    def forward(self, input_dict):
        if "condition" in input_dict.keys():
            # PPT (https://arxiv.org/abs/2308.09718)
            # currently, only support one batch one condition
            input_dict["condition"] = input_dict["condition"][0]
        
        out = self.backbone(input_dict)
        aux_logits_list = []
        if isinstance(out, dict) and "seg_logits" in out:
            seg_logits = out["seg_logits"]
            aux_logits_list = out.get("aux_logits", [])
        elif isinstance(out, tuple) and len(out) >= 2:
            seg_logits, aux_logits_list = out[0], out[1]
        else:
            seg_logits = out

        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict = dict()

            # Auxiliary loss from intermediate heads when present.
            if len(aux_logits_list) > 0 and self.aux_criteria is not None:
                aux_weights = self.aux_weights if self.aux_weights is not None else [1.0] * len(aux_logits_list)
                aux_loss = 0.0
                for i, aux_logits in enumerate(aux_logits_list):
                    if i >= len(aux_weights):
                        break
                    stage_loss = self.aux_criteria(aux_logits, input_dict["segment"])
                    weighted_loss = stage_loss * aux_weights[i]
                    aux_loss = aux_loss + weighted_loss
                loss = loss + aux_loss
                return_dict["l_aux"] = aux_loss

            return_dict["loss"] = loss
            return return_dict
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)
        # test
        else:
            return dict(seg_logits=seg_logits)


@MODELS.register_module()
class DefaultSegmentorV2(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
        aux_criteria=None,
        aux_channels=None,
        aux_dropout=0.1,
        aux_weights=None,
        freeze_backbone=False,
    ):
        super().__init__()
        import pointops
        self.pointops = pointops
        self.num_classes = num_classes
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.aux_criteria = build_criteria(aux_criteria) if aux_criteria else None
        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.aux_heads = None
        self.aux_weights = None
        if aux_channels is not None and len(aux_channels) > 0:
            self.aux_heads = nn.ModuleList()
            for ch in aux_channels:
                head = nn.Sequential(
                    nn.Dropout(p=aux_dropout),
                    nn.Linear(ch, num_classes),
                )
                self.aux_heads.append(head)

            # Per-stage weights for auxiliary heads.
            if aux_weights is not None:
                assert len(aux_weights) == len(aux_channels), (
                    f"aux_weights length ({len(aux_weights)}) must match "
                    f"aux_channels length ({len(aux_channels)})"
                )
                self.aux_weights = aux_weights
            else:
                # Default every auxiliary stage to weight 1.0.
                self.aux_weights = tuple([1.0] * len(aux_channels))

    def forward(self, input_dict, return_point=False):
        point = Point(input_dict)
        
        origin_coord = point.coord.clone()
        origin_offset = point.offset.clone()

        point = self.backbone(point)
        # Backbone added after v1.5.0 return Point instead of feat and use DefaultSegmentorV2
        # TODO: remove this part after make all backbone return Point only.
        if isinstance(point, Point):
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            feat = point.feat
        else:
            feat = point
        seg_logits = self.seg_head(feat)
        return_dict = dict()
        if return_point:
            # PCA evaluator parse feat and coord in point
            return_dict["point"] = point

        # Collect auxiliary outputs for hybrid deep supervision.
        aux_logits_list = []
        if (
            self.training
            and self.aux_heads is not None
            and self.aux_criteria is not None
            and hasattr(point, "aux_outputs")
            and point.aux_outputs is not None
        ):
            aux_outputs = point.aux_outputs

            for i, aux_points in enumerate(aux_outputs):
                if i >= len(self.aux_heads):
                    break

                aux_coord, aux_feat, aux_offset = aux_points

                # Interpolate low-resolution stage features back to raw points.
                with torch.amp.autocast('cuda', enabled=False):
                    interpolated_feat = self.pointops.interpolation(
                        aux_coord.float(), origin_coord.float(), aux_feat.float(), aux_offset, origin_offset
                    )

                # Produce auxiliary logits at raw-point resolution.
                aux_logits = self.aux_heads[i](interpolated_feat)
                aux_logits_list.append(aux_logits)

        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            
            # Auxiliary loss.
            if len(aux_logits_list) > 0 and self.aux_criteria is not None:
                aux_loss = 0.0
                for i, aux_logits in enumerate(aux_logits_list):
                    stage_loss = self.aux_criteria(aux_logits, input_dict["segment"])
                    # Apply the configured per-stage weight.
                    weighted_loss = stage_loss * self.aux_weights[i]
                    aux_loss = aux_loss + weighted_loss
                loss = loss + aux_loss
                return_dict["l_aux"] = aux_loss

            return_dict["loss"] = loss
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
            return_dict["seg_logits"] = seg_logits
        # test
        else:
            return_dict["seg_logits"] = seg_logits
        return return_dict


@MODELS.register_module()
class DefaultLORASegmentorV2(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
        freeze_backbone=False,
        use_lora=False,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        backbone_path=None,
        keywords=None,
        replacements=None,
    ):
        super().__init__()
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.keywords = keywords
        self.replacements = replacements
        self.backbone = build_model(backbone)
        backbone_weight = torch.load(
            backbone_path,
            map_location=lambda storage, loc: storage.cuda(),
        )
        self.backbone_load(backbone_weight)

        self.criteria = build_criteria(criteria)
        self.freeze_backbone = freeze_backbone
        self.use_lora = use_lora

        if self.use_lora:
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["qkv"],
                # target_modules=["query", "value"],
                lora_dropout=lora_dropout,
                bias="none",
            )
            self.backbone.enc = get_peft_model(self.backbone.enc, lora_config)

        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        if self.use_lora:
            for name, param in self.backbone.named_parameters():
                if "lora_" in name:
                    param.requires_grad = True
        self.backbone.enc.print_trainable_parameters()

    def backbone_load(self, checkpoint):
        weight = OrderedDict()
        for key, value in checkpoint["state_dict"].items():
            if not key.startswith("module."):
                key = "module." + key  # xxx.xxx -> module.xxx.xxx
            # Now all keys contain "module." no matter DDP or not.
            if self.keywords in key:
                key = key.replace(self.keywords, self.replacements)
            key = key[7:]  # module.xxx.xxx -> xxx.xxx
            if key.startswith("backbone."):
                key = key[9:]
            weight[key] = value
        load_state_info = self.backbone.load_state_dict(weight, strict=False)
        print(f"Missing keys: {load_state_info[0]}")
        print(f"Unexpected keys: {load_state_info[1]}")

    def forward(self, input_dict, return_point=False):
        point = Point(input_dict)
        if self.freeze_backbone and not self.use_lora:
            with torch.no_grad():
                point = self.backbone(point)
        else:
            point = self.backbone(point)

        if isinstance(point, Point):
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            feat = point.feat
        else:
            feat = point

        seg_logits = self.seg_head(feat)
        return_dict = dict()
        if return_point:
            return_dict["point"] = point

        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
            return_dict["seg_logits"] = seg_logits
        else:
            return_dict["seg_logits"] = seg_logits
        return return_dict


@MODELS.register_module()
class DINOEnhancedSegmentor(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
        freeze_backbone=False,
    ):
        super().__init__()
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.backbone = build_model(backbone) if backbone is not None else None
        self.criteria = build_criteria(criteria)
        self.freeze_backbone = freeze_backbone
        if self.backbone is not None and self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, input_dict, return_point=False):
        point = Point(input_dict)
        if self.backbone is not None:
            if self.freeze_backbone:
                with torch.no_grad():
                    point = self.backbone(point)
            else:
                point = self.backbone(point)
            point_list = [point]
            while "unpooling_parent" in point_list[-1].keys():
                point_list.append(point_list[-1].pop("unpooling_parent"))
            for i in reversed(range(1, len(point_list))):
                point = point_list[i]
                parent = point_list[i - 1]
                assert "pooling_inverse" in point.keys()
                inverse = point.pooling_inverse
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = point_list[0]
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pooling_inverse
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            feat = [point.feat]
        else:
            feat = []
        dino_coord = input_dict["dino_coord"]
        dino_feat = input_dict["dino_feat"]
        dino_offset = input_dict["dino_offset"]
        idx = torch_cluster.knn(
            x=dino_coord,
            y=point.origin_coord,
            batch_x=offset2batch(dino_offset),
            batch_y=offset2batch(point.origin_offset),
            k=1,
        )[1]

        feat.append(dino_feat[idx])
        feat = torch.concatenate(feat, dim=-1)
        seg_logits = self.seg_head(feat)
        return_dict = dict()
        if return_point:
            # PCA evaluator parse feat and coord in point
            return_dict["point"] = point
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
            return_dict["seg_logits"] = seg_logits
        # test
        else:
            return_dict["seg_logits"] = seg_logits
        return return_dict


@MODELS.register_module()
class DefaultClassifier(nn.Module):
    def __init__(
        self,
        backbone=None,
        criteria=None,
        num_classes=40,
        backbone_embed_dim=256,
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.num_classes = num_classes
        self.backbone_embed_dim = backbone_embed_dim
        self.cls_head = nn.Sequential(
            nn.Linear(backbone_embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, input_dict):
        point = Point(input_dict)
        point = self.backbone(point)
        # Backbone added after v1.5.0 return Point instead of feat
        # And after v1.5.0 feature aggregation for classification operated in classifier
        # TODO: remove this part after make all backbone return Point only.
        if isinstance(point, Point):
            point.feat = torch_scatter.segment_csr(
                src=point.feat,
                indptr=nn.functional.pad(point.offset, (1, 0)),
                reduce="mean",
            )
            feat = point.feat
        else:
            feat = point
        cls_logits = self.cls_head(feat)
        if self.training:
            loss = self.criteria(cls_logits, input_dict["category"])
            return dict(loss=loss)
        elif "category" in input_dict.keys():
            loss = self.criteria(cls_logits, input_dict["category"])
            return dict(loss=loss, cls_logits=cls_logits)
        else:
            return dict(cls_logits=cls_logits)


@MODELS.register_module()
class DefaultRegressor(nn.Module):
    """Per-point regression head following the DefaultSegmentorV2 pattern.

    Produces a scalar (or multi-target) regression prediction for every
    input point. Uses the same backbone -> Point -> pooling_parent
    unwinding logic as ``DefaultSegmentorV2``.

    The regression target is read from an **existing data field**
    (e.g. ``"intensity"``) rather than a dedicated key,
    eliminating the need for a purpose-built ``regression_target`` asset.

    Convention:
        - Input target key : configurable via ``target_key`` (default ``"intensity"``)
        - Output key        : ``"reg_pred"``  (float32, shape N or N x D)

    Args:
        num_targets (int): Number of regression targets per point.
            Default 1 (scalar regression).
        backbone_out_channels (int): Feature dimension produced by the
            backbone (or concatenated after pooling_parent unwinding).
        backbone (dict | None): Backbone config to build.
        criteria (list[dict] | None): Loss config(s) for
            ``build_criteria``. Typical choices: ``MSELoss``,
            ``L1Loss``, ``SmoothL1Loss``, ``HuberLoss``.
        target_key (str): Key in ``input_dict`` used as the regression
            ground-truth (e.g. ``"intensity"``).
        freeze_backbone (bool): Freeze backbone parameters.
    """

    def __init__(
        self,
        num_targets=1,
        backbone_out_channels=64,
        backbone=None,
        criteria=None,
        target_key="intensity",
        freeze_backbone=False,
    ):
        super().__init__()
        self.reg_head = nn.Linear(backbone_out_channels, num_targets)
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.num_targets = num_targets
        self.target_key = target_key
        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, input_dict):
        point = Point(input_dict)
        point = self.backbone(point)

        # Unwind pooling_parent hierarchy (same as DefaultSegmentorV2)
        if isinstance(point, Point):
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            feat = point.feat
        else:
            feat = point

        reg_pred = self.reg_head(feat)  # (N, num_targets)
        # Squeeze to (N,) when single target for easier downstream use
        if self.num_targets == 1:
            reg_pred = reg_pred.squeeze(-1)

        return_dict = dict()
        # train
        if self.training:
            loss = self.criteria(reg_pred, input_dict[self.target_key])
            return_dict["loss"] = loss
        # eval
        elif self.target_key in input_dict.keys():
            loss = self.criteria(reg_pred, input_dict[self.target_key])
            return_dict["loss"] = loss
            return_dict["reg_pred"] = reg_pred
        # test
        else:
            return_dict["reg_pred"] = reg_pred
        return return_dict


@MODELS.register_module()
class DefaultSemSegRegressor(nn.Module):
    """Joint semantic-segmentation + per-point regression head.

    Shares a single backbone and splits into two independent heads:

    * ``seg_head`` -> linear projection -> ``(N, num_classes)`` logits
    * ``reg_head`` -> linear projection -> ``(N,)`` or ``(N, num_targets)``

    Each head has its own loss (``seg_criteria`` / ``reg_criteria``) and a
    scalar weight (``seg_weight`` / ``reg_weight``).  The total training
    loss is ``seg_weight * seg_loss + reg_weight * reg_loss``.

    Convention (output keys):
        - ``"seg_logits"`` : semantic segmentation logits
        - ``"reg_pred"``   : regression prediction
        - ``"loss"``       : weighted sum of seg and reg losses
        - ``"seg_loss"``   : individual seg loss (eval only)
        - ``"reg_loss"``   : individual reg loss (eval only)

    Args:
        num_classes (int): Number of semantic classes.
        num_targets (int): Number of regression targets per point (default 1).
        backbone_out_channels (int): Feature dim from backbone.
        backbone (dict | None): Backbone config.
        seg_criteria (list[dict] | None): Loss configs for segmentation.
        reg_criteria (list[dict] | None): Loss configs for regression.
        seg_weight (float): Weight applied to seg loss.
        reg_weight (float): Weight applied to reg loss.
        target_key (str): Key in ``input_dict`` for regression ground-truth.
        freeze_backbone (bool): Freeze backbone parameters.
    """

    def __init__(
        self,
        num_classes,
        num_targets=1,
        backbone_out_channels=64,
        backbone=None,
        seg_criteria=None,
        reg_criteria=None,
        seg_weight=1.0,
        reg_weight=1.0,
        target_key="intensity",
        freeze_backbone=False,
    ):
        super().__init__()
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.reg_head = nn.Linear(backbone_out_channels, num_targets)
        self.backbone = build_model(backbone)
        self.seg_criteria = build_criteria(seg_criteria)
        self.reg_criteria = build_criteria(reg_criteria)
        self.num_classes = num_classes
        self.num_targets = num_targets
        self.seg_weight = seg_weight
        self.reg_weight = reg_weight
        self.target_key = target_key
        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, input_dict):
        point = Point(input_dict)
        point = self.backbone(point)

        # Unwind pooling_parent hierarchy (same as DefaultSegmentorV2)
        if isinstance(point, Point):
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            feat = point.feat
        else:
            feat = point

        seg_logits = self.seg_head(feat)
        reg_pred = self.reg_head(feat)
        if self.num_targets == 1:
            reg_pred = reg_pred.squeeze(-1)

        return_dict = dict()
        has_seg_target = "segment" in input_dict.keys()
        has_reg_target = self.target_key in input_dict.keys()

        # train: only return combined loss
        if self.training:
            seg_loss = self.seg_criteria(seg_logits, input_dict["segment"])
            reg_loss = self.reg_criteria(reg_pred, input_dict[self.target_key])
            return_dict["loss"] = self.seg_weight * seg_loss + self.reg_weight * reg_loss
        # eval: return combined loss + individual losses + predictions
        elif has_seg_target and has_reg_target:
            seg_loss = self.seg_criteria(seg_logits, input_dict["segment"])
            reg_loss = self.reg_criteria(reg_pred, input_dict[self.target_key])
            return_dict["loss"] = self.seg_weight * seg_loss + self.reg_weight * reg_loss
            return_dict["seg_loss"] = seg_loss
            return_dict["reg_loss"] = reg_loss
            return_dict["seg_logits"] = seg_logits
            return_dict["reg_pred"] = reg_pred
        # test: predictions only
        else:
            return_dict["seg_logits"] = seg_logits
            return_dict["reg_pred"] = reg_pred
        return return_dict


@MODELS.register_module()
class DeepLASegmentor(nn.Module):
    """DeepLANet segmentor with hybrid deep supervision.

    Adds auxiliary heads on encoder stages, interpolates stage features back
    to raw-point resolution, and combines the main loss with optional
    auxiliary losses.
    """

    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
        aux_criteria=None,
        aux_channels=None,
        aux_dropout=0.1,
        aux_weights=None,
        freeze_backbone=False,
    ):
        super().__init__()
        import pointops

        self.pointops = pointops
        self.num_classes = num_classes

        # Main segmentation head.
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )

        # Backbone
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.aux_criteria = build_criteria(aux_criteria) if aux_criteria else None
        # Superpoint consistency loss (train only)

        # Optionally freeze the backbone.
        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # Build auxiliary heads for encoder stages.
        # ``aux_channels`` should align with encoder stage output channels.
        self.aux_heads = None
        self.aux_weights = None
        if aux_channels is not None and len(aux_channels) > 0:
            self.aux_heads = nn.ModuleList()
            for ch in aux_channels:
                head = nn.Sequential(
                    nn.Dropout(p=aux_dropout),
                    nn.Linear(ch, num_classes),
                )
                self.aux_heads.append(head)

            # Per-stage weights for auxiliary heads.
            if aux_weights is not None:
                assert len(aux_weights) == len(aux_channels), (
                    f"aux_weights length ({len(aux_weights)}) must match "
                    f"aux_channels length ({len(aux_channels)})"
                )
                self.aux_weights = aux_weights
            else:
                # Default every auxiliary stage to weight 1.0.
                self.aux_weights = tuple([1.0] * len(aux_channels))

    def forward(self, input_dict, return_point=False):
        point = Point(input_dict)

        # Save raw coordinates and offsets for feature interpolation.
        origin_coord = point.coord.clone()
        origin_offset = point.offset.clone()

        # Backbone forward pass.
        point = self.backbone(point)

        # Unwind ``pooling_parent`` hierarchy, same as DefaultSegmentorV2.
        if isinstance(point, Point):
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            feat = point.feat
        else:
            feat = point

        # Main segmentation head.
        seg_logits = self.seg_head(feat)

        return_dict = dict()
        if return_point:
            return_dict["point"] = point

        # =====================================================================
        # Hybrid deep supervision: process auxiliary outputs.
        # =====================================================================
        aux_logits_list = []
        if (
            self.training
            and self.aux_heads is not None
            and self.aux_criteria is not None
            and hasattr(point, "aux_outputs")
            and point.aux_outputs is not None
        ):
            aux_outputs = point.aux_outputs

            for i, aux_points in enumerate(aux_outputs):
                if i >= len(self.aux_heads):
                    break

                aux_coord, aux_feat, aux_offset = aux_points

                # Interpolate stage features to raw-point resolution.
                # Run this in FP32 to avoid instability under AMP.
                import torch
                with torch.amp.autocast('cuda', enabled=False):
                    interpolated_feat = self.pointops.interpolation(
                        aux_coord.float(), origin_coord.float(), aux_feat.float(), aux_offset, origin_offset
                    )

                # Produce auxiliary logits.
                aux_logits = self.aux_heads[i](interpolated_feat)
                aux_logits_list.append(aux_logits)

        # =====================================================================
        # Compute losses.
        # =====================================================================
        if self.training:
            # Main loss.
            loss = self.criteria(seg_logits, input_dict["segment"])

            # Auxiliary loss, only when configured.
            if len(aux_logits_list) > 0 and self.aux_criteria is not None:
                aux_loss = 0.0
                for i, aux_logits in enumerate(aux_logits_list):
                    stage_loss = self.aux_criteria(aux_logits, input_dict["segment"])
                    # Apply the configured per-stage weight.
                    weighted_loss = stage_loss * self.aux_weights[i]
                    aux_loss = aux_loss + weighted_loss
                # aux_criteria 鐨?loss_weight 宸插湪 build_criteria 涓鐞?
                loss = loss + aux_loss
                return_dict["l_aux"] = aux_loss

            return_dict["loss"] = loss

        elif "segment" in input_dict.keys():
            # Eval mode: main loss only.
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
            return_dict["seg_logits"] = seg_logits

        else:
            # Test mode.
            return_dict["seg_logits"] = seg_logits

        return return_dict
