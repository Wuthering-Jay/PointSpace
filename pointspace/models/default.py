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
    def __init__(self, backbone=None, criteria=None):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def forward(self, input_dict):
        if "condition" in input_dict.keys():
            # PPT (https://arxiv.org/abs/2308.09718)
            # currently, only support one batch one condition
            input_dict["condition"] = input_dict["condition"][0]
        seg_logits = self.backbone(input_dict)
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss)
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
        freeze_backbone=False,
    ):
        super().__init__()
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, input_dict, return_point=False):
        point = Point(input_dict)
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
    input point.  Uses the same backbone → Point → pooling_parent
    unwinding logic as ``DefaultSegmentorV2``.

    The regression target is read from an **existing data field**
    (e.g. ``"hag"``, ``"intensity"``) rather than a dedicated key,
    eliminating the need for a purpose-built ``regression_target`` asset.

    Convention:
        - Input target key : configurable via ``target_key`` (default ``"hag"``)
        - Output key        : ``"reg_pred"``  (float32, shape N or N×D)

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
            ground-truth (e.g. ``"hag"``, ``"intensity"``).
        freeze_backbone (bool): Freeze backbone parameters.
    """

    def __init__(
        self,
        num_targets=1,
        backbone_out_channels=64,
        backbone=None,
        criteria=None,
        target_key="hag",
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

    * ``seg_head`` – linear projection → ``(N, num_classes)`` logits
    * ``reg_head`` – linear projection → ``(N,)`` or ``(N, num_targets)``

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
        target_key="hag",
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


from .head import DualBranchCNFHead, SingleBranchCNFHead  # noqa: F401 — triggers MODELS registration


@MODELS.register_module()
class DefaultCNF(nn.Module):
    """针对连续地形隐式重建 (Continuous DEM) 的默认网络包装器。

    包含特征提取 Backbone 和双分支条件神经场 Head，
    采用 Point 数据结构传递数据给 Backbone，与 Pointcept 生态一致。

    Architecture:
        1. **Backbone** encodes the support point cloud into per-point
           feature vectors ``(N, C)`` via ``Point`` data structure.
        2. **Head** (e.g. :class:`DualBranchCNFHead`) decodes
           predictions for arbitrary *query* coordinates, returning
           ``(pred_base, pred_detail)``.

    Default loss (asymmetric dual-stream, with detach):
        - loss_base:  SmoothL1(pred_base, query_gt_low, beta=1.0)
        - loss_final: SmoothL1(pred_base.detach() + pred_detail, query_gt, beta=0.1)
        - loss_reg:   L1 mean of pred_detail * reg_weight

    Args:
        backbone (dict | None): Backbone config (e.g. PT-v2m4).
        head (dict | None): CNF head config (e.g. DualBranchCNFHead).
        criteria (dict | None): Custom loss module. If provided,
            ``compute_loss`` delegates to ``criteria(head_output,
            input_dict)``.  Otherwise uses the built-in loss.
        reg_weight (float): L1 regularization weight on the detail
            branch (only used by the built-in dual-branch loss).
        terrain_alpha (float): Terrain-complexity weighting coefficient.
            ``weight = 1 + alpha * |query_gt - z_anchor|``.  Only used by
            single-branch built-in loss.  Default 2.0.
    """

    def __init__(self, backbone=None, head=None, criteria=None,
                 reg_weight=0.01, terrain_alpha=2.0, ohem_ratio=0.5,
                 normal_weight=0.0, enable_normal_loss=True, normal_ratio=0.5,
                 filter_non_ground=False, ground_class=2):
        super().__init__()
        self.backbone = build_model(backbone) if backbone is not None else None
        self.head = build_model(head) if head is not None else None
        self.criteria = build_model(criteria) if criteria is not None else None
        self.reg_weight = reg_weight
        self.terrain_alpha = terrain_alpha
        self.ohem_ratio = ohem_ratio
        self.normal_weight = normal_weight
        self.enable_normal_loss = enable_normal_loss
        self.filter_non_ground = filter_non_ground
        self.ground_class = ground_class
    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _filter_ground(self, support_feat, support_coord, input_dict):
        """Filter out non-ground points after backbone encoding.

        Returns filtered (feat, coord, offset).  When ``filter_non_ground``
        is False or ``segment`` is absent the inputs are returned
        unchanged.
        """
        support_offset = input_dict.get("offset")
        if not self.filter_non_ground:
            return support_feat, support_coord, support_offset

        if "segment" not in input_dict:
            # segment absent at inference time (no GT labels) — skip filtering
            import logging
            logging.getLogger(__name__).warning(
                "DefaultCNF._filter_ground: filter_non_ground=True but 'segment' "
                "is missing in input_dict. Skipping ground filtering."
            )
            return support_feat, support_coord, support_offset
        cls_labels = input_dict["segment"].squeeze()
        ground_mask = (cls_labels == self.ground_class)

        if not ground_mask.any():
            ground_mask[0] = True

        support_coord = support_coord[ground_mask]
        support_feat = support_feat[ground_mask]

        if support_offset is not None:
            batch_size = support_offset.shape[0]
            batch_idx = offset2batch(support_offset)
            batch_idx_ground = batch_idx[ground_mask]
            counts = torch.bincount(batch_idx_ground, minlength=batch_size)
            support_offset = torch.cumsum(counts, dim=0).int()

        return support_feat, support_coord, support_offset

    def _run_backbone(self, input_dict):
        """Run backbone with Point wrapping and unwind pooling hierarchy.

        Returns:
            tuple: (feat, coord) — (N, C) and (N, 3) tensors.
        """
        point = Point(input_dict)
        point = self.backbone(point)

        if isinstance(point, Point):
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat(
                    [parent.feat, point.feat[inverse]], dim=-1
                )
                point = parent
            return point.feat, point.coord
        else:
            return point, input_dict["coord"]

    # ------------------------------------------------------------------
    # Public sub-methods (called by CnfTester)
    # ------------------------------------------------------------------
    def extract_feat(self, input_dict):
        """[Inference] Run backbone only, return ``{feat, coord, offset, segment}``."""
        feat, coord = self._run_backbone(input_dict)
        offset = input_dict.get("offset")
        result = dict(feat=feat, coord=coord)
        if offset is not None:
            result["offset"] = offset
        if "segment" in input_dict:
            result["segment"] = input_dict["segment"]
        return result

    def query_forward(self, support_coord, support_feat, query_coord,
                      support_segment=None):
        """[Inference] Run head only, return final prediction.

        Used by :class:`CnfTester` for chunked dense queries.
        Forces eval mode on the head so that single-branch returns a plain
        tensor and dual-branch returns ``(pred_base, pred_detail)``.
        """
        was_training = self.head.training
        self.head.eval()
        result = self.head(
            support_coord, support_feat, query_coord,
            support_segment=support_segment,
        )
        if was_training:
            self.head.train()
        if isinstance(result, tuple):
            # Dual-branch: base + detail
            return result[0] + result[1]
        return result

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    def compute_loss(self, head_output, input_dict):
        """Compute loss for both single-branch and dual-branch heads.

        For single-branch heads (training), *head_output* is
        ``(pred_z, local_z_anchor)`` or ``(pred_z, local_z_anchor, pred_normal)``
        when ``predict_normals=True``.
        For dual-branch heads, *head_output* is ``(pred_base, pred_detail)``.

        If ``self.criteria`` is set, delegates entirely to that module.
        Otherwise uses the appropriate built-in loss.
        """
        if self.criteria is not None:
            return self.criteria(head_output, input_dict)

        # ---- Single-branch loss with terrain-complexity weighting ----
        if isinstance(self.head, SingleBranchCNFHead):
            if isinstance(head_output, tuple):
                if len(head_output) == 3:
                    # New 3-tuple: (pred_z, local_z_anchor, pred_normal)
                    pred_z, local_z_anchor, pred_normal = head_output
                else:
                    # Legacy 2-tuple: (pred_z, local_z_anchor)
                    pred_z, local_z_anchor = head_output
                    pred_normal = None
            else:
                pred_z = head_output
                local_z_anchor = None
                pred_normal = None

            query_gt = input_dict["query_gt"]
            if query_gt.dim() == 1:
                query_gt = query_gt.unsqueeze(-1)
            pz = pred_z.unsqueeze(-1) if pred_z.dim() == 1 else pred_z

            # ==========================================================
            # 🌟 1. 基础 MAE 保底 (保细节、保锐利)
            # ==========================================================
            # l1_error = torch.abs(pz - query_gt)
            l1_error = torch.nn.functional.smooth_l1_loss(pz, query_gt, reduction='none', beta=1.0)
            
            # 结合您的地形复杂度加权 (如果有)
            if local_z_anchor is not None:
                with torch.no_grad():
                    anchor = local_z_anchor.unsqueeze(-1) if local_z_anchor.dim() == 1 else local_z_anchor
                    idw_error = torch.abs(query_gt - anchor)
                    terrain_weight = torch.clamp(1.0 + self.terrain_alpha * idw_error, max=5.0)
                weighted_l1 = l1_error * terrain_weight
            else:
                weighted_l1 = l1_error

            # 之前的 OHEM (25%) 用于回传基础 L1
            num_keep_25 = int(weighted_l1.shape[0] * self.ohem_ratio)
            if num_keep_25 > 0:
                loss_l1 = torch.mean(torch.topk(weighted_l1.view(-1), k=num_keep_25)[0])
            else:
                loss_l1 = torch.mean(weighted_l1)

            # ==========================================================
            # 🌟 2. 专杀 RMSE：辅助 MSE Loss
            # 用 L2 的平方特性去压制大误差，但不给太大权重，防止把地形抹平
            # ==========================================================
            # l2_error = (pz - query_gt) ** 2
            l2_error = F.mse_loss(pz, query_gt, reduction='none')
            # 为了防止平地的 0.1 米误差也被 L2 过度关注，我们只对 OHEM 选出的 25% 难点施加 L2
            if num_keep_25 > 0:
                loss_l2 = torch.mean(torch.topk(l2_error.view(-1), k=num_keep_25)[0])
            else:
                loss_l2 = torch.mean(l2_error)

            # ==========================================================
            # 🌟 3. 专杀 MaxE：极限 Top-1% 惩罚
            # 专门盯着误差最离谱的那一小撮点，给予极端的 L1 惩罚
            # ==========================================================
            num_keep_1 = max(1, int(l1_error.shape[0] * 0.01)) # 前 1% 的点
            loss_max_e = torch.mean(torch.topk(l1_error.view(-1), k=num_keep_1)[0])

            # ==========================================================
            # 🌟 4. 融合：按科学权重组装目标
            # ==========================================================
            # 权重设计哲学：
            # 1.0 * loss_l1: 维持微观锐利，保证 MAE
            # 0.5 * loss_l2: 压低 RMSE，消除中等偏大的误差
            # 1.0 * loss_max_e: 像锤子一样砸平 30 米高的那几个 MaxE 刺头
            loss_z_final = 1.0 *loss_l1 + 0.5 * loss_l2 + 1.0 * loss_max_e

            # ==========================================================
            # 🌟 5. 法向量约束 (Normal constraint) - 使用直接预测替代autograd
            # ==========================================================
            loss_normal = torch.tensor(0.0, device=pz.device)
            if (self.normal_weight > 0
                    and pred_normal is not None
                    and "query_normal_gt" in input_dict):

                gt_normal = input_dict["query_normal_gt"]  # (Q, 3)

                # Direct cosine similarity between predicted and GT normals
                cos_sim = F.cosine_similarity(pred_normal, gt_normal, dim=-1)
                raw_normal_loss = 1.0 - cos_sim  # (Q,)

                # OHEM on normal loss (挖掘法向量误差最大的区域进行重点惩罚)
                num_keep_n = int(raw_normal_loss.shape[0] * self.ohem_ratio)
                if num_keep_n > 0:
                    loss_normal = torch.mean(torch.topk(raw_normal_loss, k=num_keep_n)[0])
                else:
                    loss_normal = torch.mean(raw_normal_loss)

            loss_final = loss_z_final + self.normal_weight * loss_normal

            monitor_mae = torch.mean(l1_error).detach()
            monitor_rmse = torch.sqrt(torch.mean(l2_error)).detach()
            monitor_maxe = torch.max(l1_error).detach()

            return dict(
                loss=loss_final,
                l1_ohem=loss_l1.detach(),
                l2=loss_l2.detach(),
                maxe_penalty=loss_max_e.detach(),
                normal=loss_normal.detach(),
                m_mae=monitor_mae,
                m_rmse=monitor_rmse,
                m_maxe=monitor_maxe,
            )

        # ---- Dual-branch loss ----
        pred_base, pred_detail = head_output
        query_gt = input_dict["query_gt"]
        query_gt_low = input_dict["query_gt_low"]

        if query_gt.dim() == 1:
            query_gt = query_gt.unsqueeze(-1)
        if query_gt_low.dim() == 1:
            query_gt_low = query_gt_low.unsqueeze(-1)
        pb = pred_base.unsqueeze(-1) if pred_base.dim() == 1 else pred_base
        pd = pred_detail.unsqueeze(-1) if pred_detail.dim() == 1 else pred_detail

        loss_base = F.smooth_l1_loss(pb, query_gt_low, reduction="mean", beta=1.0)
        loss_final = F.smooth_l1_loss(
            pb.detach() + pd, query_gt, reduction="mean", beta=0.1,
        )
        loss_reg = torch.mean(torch.abs(pd)) * self.reg_weight

        loss = loss_base + loss_final + loss_reg
        return dict(
            loss=loss,
            loss_base=loss_base,
            loss_final=loss_final,
            loss_reg=loss_reg,
        )

    # ------------------------------------------------------------------
    # Standard forward (used by DefaultTrainer.run_step)
    # ------------------------------------------------------------------
    def forward(self, input_dict):
        """Standard forward for train / eval.

        Train:
            Returns ``dict(loss=..., loss_base=..., loss_final=...,
            loss_reg=...)``.
        Eval with query:
            Returns ``dict(cnf_pred=..., loss=..., ...)``.
        Eval without query:
            Returns ``dict(support_feat=..., support_coord=...)``.
        """
        support_feat, support_coord = self._run_backbone(input_dict)
        support_offset = input_dict.get("offset")

        # Extract segment labels for Dual-KNN head (no hard filtering)
        support_segment = input_dict.get("segment", None)

        if self.training:
            query_coord = input_dict["query_coord"]
            # NOTE: No longer need requires_grad_(True) for normal loss
            # Normals are now predicted directly by a dedicated head branch
            head_output = self.head(
                support_coord, support_feat, query_coord,
                support_offset=support_offset,
                query_offset=input_dict.get("query_offset"),
                support_segment=support_segment,
            )
            return self.compute_loss(head_output, input_dict)

        # ---- Eval / Test ----
        if "query_coord" in input_dict:
            query_coord = input_dict["query_coord"]
            head_output = self.head(
                support_coord, support_feat, query_coord,
                support_offset=support_offset,
                query_offset=input_dict.get("query_offset"),
                support_segment=support_segment,
            )
            if isinstance(head_output, tuple):
                pred_final = head_output[0] + head_output[1]
            else:
                pred_final = head_output
            result = dict(cnf_pred=pred_final)

            # Compute loss when GT available (validation)
            has_gt = "query_gt" in input_dict
            if has_gt:
                loss_dict = self.compute_loss(head_output, input_dict)
                result.update(loss_dict)
            return result

        # No query → return features for CnfTester
        return dict(support_feat=support_feat, support_coord=support_coord)
