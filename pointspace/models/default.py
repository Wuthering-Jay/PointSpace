import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
import torch_cluster
# from peft import LoraConfig, get_peft_model
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

            # 辅助损失
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

            # 设置各 stage 的权重比例
            if aux_weights is not None:
                assert len(aux_weights) == len(aux_channels), (
                    f"aux_weights 长度 ({len(aux_weights)}) 必须与 "
                    f"aux_channels 长度 ({len(aux_channels)}) 一致"
                )
                self.aux_weights = aux_weights
            else:
                # 默认权重均为 1.0
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

        # =====================================================================
        # 混合深监督: 处理辅助输出
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

                # 使用 pointops.interpolation 将低分辨率特征插值到原始分辨率
                with torch.amp.autocast('cuda', enabled=False):
                    interpolated_feat = self.pointops.interpolation(
                        aux_coord.float(), origin_coord.float(), aux_feat.float(), aux_offset, origin_offset
                    )

                # 通过辅助头生成 logits
                aux_logits = self.aux_heads[i](interpolated_feat)
                aux_logits_list.append(aux_logits)

        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            
            # 辅助损失
            if len(aux_logits_list) > 0 and self.aux_criteria is not None:
                aux_loss = 0.0
                for i, aux_logits in enumerate(aux_logits_list):
                    stage_loss = self.aux_criteria(aux_logits, input_dict["segment"])
                    # 应用各 stage 的权重比例
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


# @MODELS.register_module()
# class DefaultLORASegmentorV2(nn.Module):
#     def __init__(
#         self,
#         num_classes,
#         backbone_out_channels,
#         backbone=None,
#         criteria=None,
#         freeze_backbone=False,
#         use_lora=False,
#         lora_r=8,
#         lora_alpha=16,
#         lora_dropout=0.1,
#         backbone_path=None,
#         keywords=None,
#         replacements=None,
#     ):
#         super().__init__()
#         self.seg_head = (
#             nn.Linear(backbone_out_channels, num_classes)
#             if num_classes > 0
#             else nn.Identity()
#         )
#         self.keywords = keywords
#         self.replacements = replacements
#         self.backbone = build_model(backbone)
#         backbone_weight = torch.load(
#             backbone_path,
#             map_location=lambda storage, loc: storage.cuda(),
#         )
#         self.backbone_load(backbone_weight)

#         self.criteria = build_criteria(criteria)
#         self.freeze_backbone = freeze_backbone
#         self.use_lora = use_lora

#         if self.use_lora:
#             lora_config = LoraConfig(
#                 r=lora_r,
#                 lora_alpha=lora_alpha,
#                 target_modules=["qkv"],
#                 # target_modules=["query", "value"],
#                 lora_dropout=lora_dropout,
#                 bias="none",
#             )
#             self.backbone.enc = get_peft_model(self.backbone.enc, lora_config)

#         if self.freeze_backbone:
#             for p in self.backbone.parameters():
#                 p.requires_grad = False
#         if self.use_lora:
#             for name, param in self.backbone.named_parameters():
#                 if "lora_" in name:
#                     param.requires_grad = True
#         self.backbone.enc.print_trainable_parameters()

#     def backbone_load(self, checkpoint):
#         weight = OrderedDict()
#         for key, value in checkpoint["state_dict"].items():
#             if not key.startswith("module."):
#                 key = "module." + key  # xxx.xxx -> module.xxx.xxx
#             # Now all keys contain "module." no matter DDP or not.
#             if self.keywords in key:
#                 key = key.replace(self.keywords, self.replacements)
#             key = key[7:]  # module.xxx.xxx -> xxx.xxx
#             if key.startswith("backbone."):
#                 key = key[9:]
#             weight[key] = value
#         load_state_info = self.backbone.load_state_dict(weight, strict=False)
#         print(f"Missing keys: {load_state_info[0]}")
#         print(f"Unexpected keys: {load_state_info[1]}")

#     def forward(self, input_dict, return_point=False):
#         point = Point(input_dict)
#         if self.freeze_backbone and not self.use_lora:
#             with torch.no_grad():
#                 point = self.backbone(point)
#         else:
#             point = self.backbone(point)

#         if isinstance(point, Point):
#             while "pooling_parent" in point.keys():
#                 assert "pooling_inverse" in point.keys()
#                 parent = point.pop("pooling_parent")
#                 inverse = point.pop("pooling_inverse")
#                 parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
#                 point = parent
#             feat = point.feat
#         else:
#             feat = point

#         seg_logits = self.seg_head(feat)
#         return_dict = dict()
#         if return_point:
#             return_dict["point"] = point

#         if self.training:
#             loss = self.criteria(seg_logits, input_dict["segment"])
#             return_dict["loss"] = loss
#         elif "segment" in input_dict.keys():
#             loss = self.criteria(seg_logits, input_dict["segment"])
#             return_dict["loss"] = loss
#             return_dict["seg_logits"] = seg_logits
#         else:
#             return_dict["seg_logits"] = seg_logits
#         return return_dict


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
class DeepLASegmentor(nn.Module):
    """DeepLANet 专用 Segmentor，支持混合深监督 (Hybrid Deep Supervision)。

    基于 DefaultSegmentorV2，增加以下功能:
    1. 收集 backbone 的 aux_outputs (各 Encoder 阶段的中间特征)
    2. 使用 pointops.interpolation 将低分辨率中间特征插值到原始分辨率
    3. 动态辅助头 (Auxiliary Heads): 为每个 Encoder 阶段生成带 Dropout 的分类头
    4. 双轨损失机制: criteria (主损失) + aux_criteria (辅助损失)
    5. 超点一致性损失: sp_criteria (Superpoint Consistency Loss，仅训练时生效)

    Args:
        num_classes (int): 分类类别数
        backbone_out_channels (int): backbone 输出特征维度
        backbone (dict): backbone 配置
        criteria (list[dict]): 主损失配置 (如 CE + Lovasz)
        aux_criteria (list[dict]): 辅助损失配置 (如纯 CE)
        aux_channels (list[int]): 各 Encoder 阶段的输出通道数，用于创建辅助头
        aux_dropout (float): 辅助头的 Dropout 概率
        aux_weights (tuple[float]): 各 stage 的辅助损失权重比例，如 (0.1, 0.2, 0.3, 0.4)
                                     如果为 None，默认所有 stage 权重相等 (1.0)
        sp_criteria (list[dict]): 超点一致性损失配置 (如 SuperpointConsistencyLoss)
                                  仅在训练时生效，验证和测试时自动跳过
        freeze_backbone (bool): 是否冻结 backbone 参数
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

        # 主分割头
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

        # 冻结 backbone
        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # 动态创建辅助头
        # aux_channels 应与 backbone 的 enc_channels 对应
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

            # 设置各 stage 的权重比例
            if aux_weights is not None:
                assert len(aux_weights) == len(aux_channels), (
                    f"aux_weights 长度 ({len(aux_weights)}) 必须与 "
                    f"aux_channels 长度 ({len(aux_channels)}) 一致"
                )
                self.aux_weights = aux_weights
            else:
                # 默认权重均为 1.0
                self.aux_weights = tuple([1.0] * len(aux_channels))

    def forward(self, input_dict, return_point=False):
        point = Point(input_dict)

        # 保存原始坐标和 offset，用于辅助特征插值
        origin_coord = point.coord.clone()
        origin_offset = point.offset.clone()

        # Backbone 前向
        point = self.backbone(point)

        # 处理 pooling_parent 层级 (与 DefaultSegmentorV2 一致)
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

        # 主分割头
        seg_logits = self.seg_head(feat)

        return_dict = dict()
        if return_point:
            return_dict["point"] = point

        # =====================================================================
        # 混合深监督: 处理辅助输出
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

                # 使用 pointops.interpolation 将低分辨率特征插值到原始分辨率
                # interpolation(src_coord, dst_coord, src_feat, src_offset, dst_offset)
                # 必须在 FP32 下执行，否则 1/dist 的操作在 float16 (AMP) 下极易产生 inf/nan
                import torch
                with torch.amp.autocast('cuda', enabled=False):
                    interpolated_feat = self.pointops.interpolation(
                        aux_coord.float(), origin_coord.float(), aux_feat.float(), aux_offset, origin_offset
                    )

                # 通过辅助头生成 logits
                aux_logits = self.aux_heads[i](interpolated_feat)
                aux_logits_list.append(aux_logits)

        # =====================================================================
        # 计算损失
        # =====================================================================
        if self.training:
            # 主损失
            loss = self.criteria(seg_logits, input_dict["segment"])

            # 辅助损失（只在配置时才计算和显示）
            if len(aux_logits_list) > 0 and self.aux_criteria is not None:
                aux_loss = 0.0
                for i, aux_logits in enumerate(aux_logits_list):
                    stage_loss = self.aux_criteria(aux_logits, input_dict["segment"])
                    # 应用各 stage 的权重比例
                    weighted_loss = stage_loss * self.aux_weights[i]
                    aux_loss = aux_loss + weighted_loss
                # aux_criteria 的 loss_weight 已在 build_criteria 中处理
                loss = loss + aux_loss
                return_dict["l_aux"] = aux_loss  # 只在配置了 aux_criteria 时才添加到日志

            return_dict["loss"] = loss

        elif "segment" in input_dict.keys():
            # Eval 模式: 只计算主损失
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
            return_dict["seg_logits"] = seg_logits

        else:
            # Test 模式
            return_dict["seg_logits"] = seg_logits

        return return_dict
    

@MODELS.register_module()
class DeepLNLSegmentor(nn.Module):
    """
    DeepLANet 噪声标签学习 (LNL) 终极 Segmentor: ICL-DeepLA (V4.0 Spatial-Uncertainty Aware Edition)
    
    核心创新与 V4.0 升级:
    1. 打破跳跃连接悖论: 严格在 Encoder 内部计算跨层一致性。
    2. 平滑退火 (Smooth Transition): alpha 随 epoch 线性预热，防止 Loss 休克。
    3. 空间一致性 (Spatial Consistency): [V4新增] 利用 KNN 进行置信度空间平滑，完美保护大类边界难例。
    4. 预测不确定性 (Predictive Uncertainty): [V4新增] 计算 HDS 多尺度预测的信息熵，严格防止“毒性原型”注入。
    5. 类别自适应阈值与邻域投票: [V4新增] PSSM 打捞不仅依靠自适应阈值，还强制要求局部邻域达成共识，秒杀孤立噪点。
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
        
        # --- ICL-DeepLA V4.0 专属核心超参数 ---
        shallow_stage=2,      # 物理 Stage 2 (提供纯净局部几何先验)
        bottleneck_stage=4,   # 物理 Stage 4 (Encoder最深处，提供纯净全局语义)
        max_alpha=2.0,        # CDCS 散度衰减的最大系数
        warmup_epochs=10,     # 前 N 个 epoch 为纯热身期 (alpha=0)
        rampup_epochs=15,     # 热身期结束后，alpha 爬升的过渡期轮数
        base_tau_pseudo=0.90, # 基础伪标签阈值 (结合自适应机制)
        pseudo_weight=0.1,    # PSSM 伪标签损失权重
        ignore_index=-1,      # 数据集中的未分类/忽略标签 ID
        
        # --- V4.0 空间与不确定性参数 ---
        k_neighbors=16,       # KNN 空间平滑的邻居数量
        uncertainty_th=0.2,   # 预测不确定性(信息熵)的容忍阈值，越低越严格
    ):
        super().__init__()
        import pointops
        self.pointops = pointops
        self.num_classes = num_classes

        # LNL 专属参数赋值
        self.shallow_stage = shallow_stage
        self.bottleneck_stage = bottleneck_stage
        self.max_alpha = max_alpha
        self.warmup_epochs = warmup_epochs
        self.rampup_epochs = rampup_epochs
        self.base_tau_pseudo = base_tau_pseudo
        self.pseudo_weight = pseudo_weight
        self.ignore_index = ignore_index
        self.k_neighbors = k_neighbors
        self.uncertainty_th = uncertainty_th

        # ==========================================================
        # 1. 基础网络与损失模块构建
        # ==========================================================
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.aux_criteria = build_criteria(aux_criteria) if aux_criteria else None
        
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.seg_head = nn.Linear(backbone_out_channels, num_classes) if num_classes > 0 else nn.Identity()

        self.aux_heads = None
        self.aux_weights = None
        if aux_channels is not None and len(aux_channels) > 0:
            self.aux_heads = nn.ModuleList()
            for ch in aux_channels:
                self.aux_heads.append(nn.Sequential(
                    nn.Dropout(p=aux_dropout),
                    nn.Linear(ch, num_classes)
                ))
            
            if aux_weights is not None:
                assert len(aux_weights) == len(aux_channels)
                self.aux_weights = aux_weights
            else:
                self.aux_weights = tuple([1.0] * len(aux_channels))

        # ==========================================================
        # 2. PSSM 缓冲注册 (不参与梯度回传)
        # ==========================================================
        self.register_buffer('prototypes', torch.zeros(num_classes, backbone_out_channels))
        self.register_buffer('class_confidences', torch.ones(num_classes))

    def get_dynamic_alpha(self, current_epoch):
        """计算平滑过渡的 Alpha 值，防止机制启动瞬间网络崩溃"""
        if current_epoch < self.warmup_epochs:
            return 0.0
        elif current_epoch >= self.warmup_epochs + self.rampup_epochs:
            return self.max_alpha
        else:
            ratio = (current_epoch - self.warmup_epochs) / self.rampup_epochs
            return self.max_alpha * ratio

    def forward(self, input_dict, return_point=False):
        point = Point(input_dict)
        origin_coord = point.coord.clone()
        origin_offset = point.offset.clone()

        point = self.backbone(point)

        # 剥离 Pooling 层级获取解码器最终特征
        if isinstance(point, Point):
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            final_feat = point.feat
        else:
            final_feat = point

        seg_logits = self.seg_head(final_feat)

        return_dict = dict()
        if return_point:
            return_dict["point"] = point

        # =====================================================================
        # Eval / Test 模式: 跳过 LNL，执行常规推理
        # =====================================================================
        if not self.training:
            return_dict["seg_logits"] = seg_logits
            if "segment" in input_dict.keys():
                return_dict["loss"] = self.criteria(seg_logits, input_dict["segment"])
            return return_dict

        # =====================================================================
        # Train 模式: ICL-DeepLA V4.0 核心抗噪计算
        # =====================================================================
        target = input_dict["segment"]
        valid_mask = (target != self.ignore_index)
        
        from pointspace.engines.hooks.misc import RuntimeInfoHook
        current_epoch = RuntimeInfoHook.state.get("epoch", 0)

        # -----------------------------------------------------------
        # [步骤 A]: 提取 Encoder 所有辅助分支预测
        # -----------------------------------------------------------
        aux_logits_list = []
        if self.aux_heads and hasattr(point, "aux_outputs"):
            for i, aux_points in enumerate(point.aux_outputs[:len(self.aux_heads)]):
                aux_coord, aux_feat, aux_offset = aux_points
                # fp32 interpolation to avoid nan in amp
                import torch
                with torch.amp.autocast('cuda', enabled=False):
                    interpolated_feat = self.pointops.interpolation(
                        aux_coord.float(), origin_coord.float(), aux_feat.float(), aux_offset, origin_offset
                    )
                aux_logits_list.append(self.aux_heads[i](interpolated_feat))


        robust_weight = torch.ones_like(target, dtype=torch.float32)
        clean_target = target.clone()
        pseudo_loss = (seg_logits.sum() * 0.0) # DDP 兼容预埋梯度
        
        current_alpha = self.get_dynamic_alpha(current_epoch)

        if current_alpha > 0:
            # =======================================================
            # [步骤 B]: CDCS 与 空间一致性平滑 (Spatial Consistency)
            # =======================================================
            shallow_idx, bottleneck_idx = self.shallow_stage - 1, self.bottleneck_stage - 1
            
            if len(aux_logits_list) > bottleneck_idx:
                shallow_logits = aux_logits_list[shallow_idx]
                bottleneck_logits = aux_logits_list[bottleneck_idx]
                
                P_shallow = F.softmax(shallow_logits.detach(), dim=-1)
                kl_div = F.kl_div(F.log_softmax(bottleneck_logits, dim=-1), P_shallow, reduction='none').sum(dim=-1)
                kl_div = torch.clamp(kl_div, max=5.0)
                
                # 原始独立置信度
                raw_weight = torch.exp(-current_alpha * kl_div)
                
                # 【V4.0 核心】: 引入 KNN 进行局部空间平滑，完美挽救物理边界难例
                if self.k_neighbors > 0:
                    knn_idx, _ = self.pointops.knn_query(16, origin_coord.float(), origin_offset.int(), origin_coord.float(), origin_offset.int())
                    knn_idx = knn_idx.long()
                    
                    local_weights = raw_weight[knn_idx] # [N, K]
                    smoothed_weight = local_weights.mean(dim=1)
                    # 融合: 当前点权重与局部邻域权重 1:1 混合
                    robust_weight = 0.5 * raw_weight + 0.5 * smoothed_weight
                else:
                    robust_weight = raw_weight

            # 极端恶劣的错标(<0.3)将被物理截断，其余靠标准 Loss 继续学习
            noise_mask = (robust_weight < 0.3) & valid_mask
            clean_target[noise_mask] = self.ignore_index

            # =======================================================
            # [步骤 C]: 多尺度预测不确定性计算 (Predictive Uncertainty)
            # =======================================================
            uncertainty = torch.ones_like(target, dtype=torch.float32)
            if len(aux_logits_list) > 0:
                probs = [F.softmax(logits.detach(), dim=-1) for logits in aux_logits_list]
                mean_prob = sum(probs) / len(probs) # [N, C]
                # 计算信息熵 (加 1e-6 防溢出)并归一化
                entropy = -torch.sum(mean_prob * torch.log(mean_prob + 1e-6), dim=-1)
                uncertainty = entropy / math.log(self.num_classes)

            # 更新类别平均置信度
            with torch.no_grad():
                if valid_mask.any():
                    P_deep_max = F.softmax(seg_logits.detach(), dim=-1).max(dim=-1)[0]
                    for c in range(self.num_classes):
                        c_mask = (target == c) & valid_mask
                        if c_mask.any():
                            self.class_confidences[c] = 0.99 * self.class_confidences[c] + 0.01 * P_deep_max[c_mask].mean()

            # =======================================================
            # [步骤 D]: PSSM 原型挖掘与自适应邻域打捞
            # =======================================================
            pred_labels = seg_logits.argmax(dim=-1)
            
            # D.1 极致纯净的原型更新: 
            # 必须满足深浅共识高(>0.8) + 预测正确 + 【多尺度不确定性极低】(拒绝毒性注入)
            clean_proto_mask = valid_mask & (robust_weight > 0.8) & (pred_labels == target) & (uncertainty < self.uncertainty_th)
            
            if clean_proto_mask.any():
                clean_feats = final_feat[clean_proto_mask].detach()
                clean_targets = target[clean_proto_mask]
                
                for c in range(self.num_classes):
                    c_mask = (clean_targets == c)
                    if c_mask.any():
                        c_feat_mean = clean_feats[c_mask].mean(dim=0)
                        if self.prototypes[c].abs().sum() == 0:
                            self.prototypes[c] = c_feat_mean
                        else:
                            self.prototypes[c] = 0.99 * self.prototypes[c] + 0.01 * c_feat_mean

            # -----------------------------------------------------------
            # [步骤 D/E.2]: 从 ignore_index 点池中打捞长尾地物
            # -----------------------------------------------------------
            unclass_mask = (target == self.ignore_index)
            if unclass_mask.any() and self.prototypes.abs().sum() > 0:
                unclass_feats = final_feat[unclass_mask]
                norm_feats = F.normalize(unclass_feats, p=2, dim=-1)
                norm_protos = F.normalize(self.prototypes, p=2, dim=-1)
                
                # 计算余弦相似度矩阵 [M, C]
                sim_matrix = torch.mm(norm_feats, norm_protos.t())
                max_sim, pseudo_labels = sim_matrix.max(dim=-1) # pseudo_labels 长度为 M (例如 8813)
                
                # [自适应阈值过滤]
                adaptive_thresholds = self.base_tau_pseudo * self.class_confidences
                point_thresholds = adaptive_thresholds[pseudo_labels] 
                confident_mask = max_sim > point_thresholds
                
                # ==========================================================
                # 🚀 V4.0 空间一致性校验 (Spatial Consistency Check)
                # ==========================================================
                if confident_mask.any():
                    # 1. 获取全局的硬预测 (长度 N = 98103)
                    full_preds = seg_logits.argmax(dim=-1)
                    
                    # 2. 专门提取这 M 个未分类点的 16 个邻居索引 [M, 16]
                    unclass_knn_idx = knn_idx[unclass_mask]
                    
                    # 3. 获取这 16 个邻居在全局地图中的预测类别 [M, 16]
                    neighbor_preds = full_preds[unclass_knn_idx]
                    
                    # 4. 统计 16 个邻居中，有几个和当前拟定的 pseudo_labels 一致？
                    # pseudo_labels.unsqueeze(1) 变成 [M, 1] 以便触发广播机制
                    match_count = (neighbor_preds == pseudo_labels.unsqueeze(1)).sum(dim=1) # [M]
                    
                    # 5. 终极防线：不仅要余弦相似度高，且邻居中至少有 3 个点支持它！(防止孤立噪点)
                    confident_mask = confident_mask & (match_count >= 3)
                # ==========================================================

                # 如果校验后依然有幸存的真实长尾点，则计算 Loss
                if confident_mask.any():
                    p_logits = seg_logits[unclass_mask][confident_mask]
                    p_targets = pseudo_labels[confident_mask]
                    pseudo_loss = F.cross_entropy(p_logits, p_targets) * self.pseudo_weight

                if confident_mask.any():
                    p_logits = seg_logits[unclass_mask][confident_mask]
                    p_targets = pseudo_labels[confident_mask]
                    pseudo_loss = F.cross_entropy(p_logits, p_targets) * self.pseudo_weight

        # =====================================================================
        # 3. 极度优雅的 Loss 派发
        # =====================================================================
        main_loss = self.criteria(seg_logits, clean_target)
        
        aux_loss = (seg_logits.sum() * 0.0)
        if len(aux_logits_list) > 0 and self.aux_criteria is not None:
            for i, aux_logits in enumerate(aux_logits_list):
                stage_loss = self.aux_criteria(aux_logits, clean_target)
                aux_loss = aux_loss + stage_loss * self.aux_weights[i]

        total_loss = main_loss + aux_loss + pseudo_loss
        
        return_dict["loss"] = total_loss
        return_dict["l_aux"] = aux_loss
        return_dict["l_pseudo"] = pseudo_loss
        
        current_epoch = RuntimeInfoHook.state.get("epoch", 0)
        global_step = RuntimeInfoHook.state.get("global_step", 0)
        
        # 只在度过热身期，且达到指定间隔时触发，防止硬盘爆炸
        if current_epoch >= self.warmup_epochs and global_step % 35 == 0:
            import os
            import numpy as np
            
            dump_dir = "debug_dumps"
            os.makedirs(dump_dir, exist_ok=True)
            
            # 1. 基础信息导出
            coord_np = origin_coord.detach().cpu().numpy()
            target_np = target.detach().cpu().numpy()
            pred_np = seg_logits.argmax(dim=-1).detach().cpu().numpy()
            
            # 2. CDCS 抗噪信息导出
            weight_np = robust_weight.detach().cpu().numpy()
            # 注意 kl_div 之前可能被 clamp 过，这里导出真实的
            kl_np = kl_div.detach().cpu().numpy() if 'kl_div' in locals() else np.zeros_like(weight_np)
            
            # 3. PSSM 伪标签信息导出 (需要把局部子集映射回全局 N 长度)
            full_pseudo = np.full_like(target_np, -1) # 默认全为 -1
            if 'unclass_mask' in locals() and 'confident_mask' in locals() and confident_mask.any():
                # 极其严谨的全局索引映射
                unclass_indices = torch.nonzero(unclass_mask, as_tuple=True)[0]
                confident_global_indices = unclass_indices[confident_mask]
                full_pseudo[confident_global_indices.cpu().numpy()] = p_targets.cpu().numpy()
                
            # 保存为 npz 压缩包
            dump_path = os.path.join(dump_dir, f"epoch_{current_epoch}_step_{global_step}.npz")
            np.savez(dump_path, 
                     coord=coord_np, 
                     target=target_np, 
                     pred=pred_np, 
                     weight=weight_np, 
                     kl=kl_np,
                     pseudo=full_pseudo)
            # print(f"\n[Diagnostic Probe] Saved intermediate tensors to {dump_path}")

        return return_dict

# @MODELS.register_module()
# class DeepLNLSegmentor(nn.Module):
#     """
#     DeepLANet 噪声标签学习 (LNL) 终极 Segmentor: ICL-DeepLA
    
#     核心创新:
#     1. 打破跳跃连接悖论: 严格在 Encoder 内部 (Bottleneck vs Shallow) 计算跨层一致性。
#     2. 标签靶向净化: 动态生成抗噪掩码，为下游标准 Loss 提供 "clean_target"，彻底解耦。
#     3. 混合深监督 (HDS) 免疫: 所有的辅助探头同样使用 clean_target，防止噪声污染深层特征。
#     4. 原型自相似挖掘 (PSSM): 从未分类点中动态打捞长尾地物。
#     """

#     def __init__(
#         self,
#         num_classes,
#         backbone_out_channels,
#         backbone=None,
#         criteria=None,
#         aux_criteria=None,
#         aux_channels=None,
#         aux_dropout=0.1,
#         aux_weights=None,
#         freeze_backbone=False,
        
#         # --- ICL-DeepLA 专属核心超参数 ---
#         shallow_stage=2,      # 物理 Stage 2 (提供纯净局部几何先验)
#         bottleneck_stage=4,   # 物理 Stage 4 (Encoder最深处，提供纯净全局语义)
#         alpha=2.0,            # CDCS 散度衰减系数 (2.0为经验甜点值)
#         tau_pseudo=0.85,      # PSSM 伪标签打捞的余弦相似度阈值
#         pseudo_weight=0.1,    # PSSM 伪标签损失权重 (软正则化)
#         ignore_index=-1,     # 数据集中的未分类/忽略标签 ID (支持设置为 -1)
#         warmup_epochs=10,     # 前 N 个 epoch 跳过 LNL 相关逻辑，直接使用原始标签训练
#     ):
#         super().__init__()
#         import pointops
#         self.pointops = pointops
#         self.num_classes = num_classes

#         # LNL 专属参数赋值
#         self.shallow_stage = shallow_stage
#         self.bottleneck_stage = bottleneck_stage
#         self.alpha = alpha
#         self.tau_pseudo = tau_pseudo
#         self.pseudo_weight = pseudo_weight
#         self.ignore_index = ignore_index
#         self.warmup_epochs = warmup_epochs

#         # ==========================================================
#         # 1. 基础网络与损失模块构建
#         # ==========================================================
#         self.backbone = build_model(backbone)
#         self.criteria = build_criteria(criteria)
#         self.aux_criteria = build_criteria(aux_criteria) if aux_criteria else None
        
#         if freeze_backbone:
#             for p in self.backbone.parameters():
#                 p.requires_grad = False

#         # 主分割头 (接在 Decoder 最终输出上)
#         self.seg_head = nn.Linear(backbone_out_channels, num_classes) if num_classes > 0 else nn.Identity()

#         # 动态创建 Encoder 的辅助探头 (混合深监督 HDS)
#         self.aux_heads = None
#         self.aux_weights = None
#         if aux_channels is not None and len(aux_channels) > 0:
#             self.aux_heads = nn.ModuleList()
#             for ch in aux_channels:
#                 self.aux_heads.append(nn.Sequential(
#                     nn.Dropout(p=aux_dropout),
#                     nn.Linear(ch, num_classes)
#                 ))
            
#             if aux_weights is not None:
#                 assert len(aux_weights) == len(aux_channels)
#                 self.aux_weights = aux_weights
#             else:
#                 self.aux_weights = tuple([1.0] * len(aux_channels))

#         # ==========================================================
#         # 2. PSSM 注册特征原型缓冲 (不参与梯度回传)
#         # ==========================================================
#         self.register_buffer('prototypes', torch.zeros(num_classes, backbone_out_channels))


#     def forward(self, input_dict, return_point=False):
#         point = Point(input_dict)
#         origin_coord = point.coord.clone()
#         origin_offset = point.offset.clone()

#         # [主干前向传播]
#         point = self.backbone(point)

#         # 剥离 Pooling 层级获取解码器最终特征
#         if isinstance(point, Point):
#             while "pooling_parent" in point.keys():
#                 assert "pooling_inverse" in point.keys()
#                 parent = point.pop("pooling_parent")
#                 inverse = point.pop("pooling_inverse")
#                 parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
#                 point = parent
#             final_feat = point.feat
#         else:
#             final_feat = point

#         # 主预测输出 (Decoder Logits)
#         seg_logits = self.seg_head(final_feat)

#         return_dict = dict()
#         if return_point:
#             return_dict["point"] = point

#         # =====================================================================
#         # Eval / Test 模式: 跳过 LNL，执行常规推理
#         # =====================================================================
#         if not self.training:
#             return_dict["seg_logits"] = seg_logits
#             if "segment" in input_dict.keys():
#                 return_dict["loss"] = self.criteria(seg_logits, input_dict["segment"])
#             return return_dict

#         # =====================================================================
#         # Train 模式: ICL-DeepLA 内生抗噪计算
#         # =====================================================================
#         target = input_dict["segment"]
#         valid_mask = (target != self.ignore_index)
        
#         # 获取最新的 RuntimeInfoHook.state，如果未注入则给个默认保护
#         from pointspace.engines.hooks.misc import RuntimeInfoHook
#         current_epoch = RuntimeInfoHook.state.get("epoch", 0)

#         # -----------------------------------------------------------
#         # [步骤 A]: 提取 Encoder 所有辅助分支预测 (并插值回原始分辨率)
#         # -----------------------------------------------------------
#         aux_logits_list = []
#         if self.aux_heads and hasattr(point, "aux_outputs"):
#             for i, aux_points in enumerate(point.aux_outputs[:len(self.aux_heads)]):
#                 aux_coord, aux_feat, aux_offset = aux_points
#                 interpolated_feat = self.pointops.interpolation(
#                     aux_coord, origin_coord, aux_feat, aux_offset, origin_offset
#                 )
#                 aux_logits_list.append(self.aux_heads[i](interpolated_feat))


#         robust_weight = torch.ones_like(target, dtype=torch.float32)
#         clean_target = target.clone()
#         pseudo_loss = torch.tensor(0.0, device=seg_logits.device)
        
#         # 只有过了 warmup_epochs 后，才执行噪声过滤与打捞 (步骤 BCD)
#         if current_epoch >= self.warmup_epochs:
#             # -----------------------------------------------------------
#             # [步骤 B]: CDCS (跨深度一致性) —— 打破跳跃连接悖论
#             # -----------------------------------------------------------
#             shallow_idx = self.shallow_stage - 1       # 默认 1 (Stage 2)
#             bottleneck_idx = self.bottleneck_stage - 1 # 默认 3 (Stage 4)
            
#             if len(aux_logits_list) > bottleneck_idx:
#                 shallow_logits = aux_logits_list[shallow_idx]
#                 bottleneck_logits = aux_logits_list[bottleneck_idx]
                
#                 # Encoder最深处语义(被测绘标签误导) vs Encoder浅层几何(纯净)
#                 P_shallow = F.softmax(shallow_logits.detach(), dim=-1) # 必须阻断梯度!
#                 kl_div = F.kl_div(
#                     F.log_softmax(bottleneck_logits, dim=-1), 
#                     P_shallow, 
#                     reduction='none'
#                 ).sum(dim=-1)
#                 kl_div = torch.clamp(kl_div, max=5.0)
#                 robust_weight = torch.exp(-self.alpha * kl_div)

#             # -----------------------------------------------------------
#             # [步骤 C]: 动态 Target 净化 (为 Loss 解耦)
#             # -----------------------------------------------------------
#             # KL 散度极大(权重<0.3)的明确错标点，将其强行丢入 ignore_index 排污槽
#             noise_mask = (robust_weight < 0.3) & valid_mask
#             clean_target[noise_mask] = self.ignore_index

#             # -----------------------------------------------------------
#             # [步骤 D]: PSSM (原型自相似挖掘) —— 解决长尾漏标
#             # -----------------------------------------------------------
#             # D.1 动态 EMA 更新类原型 (仅使用置信度 > 0.8 的绝对干净点)
#             clean_proto_mask = valid_mask & (robust_weight > 0.8)
#             if clean_proto_mask.any():
#                 clean_feats = final_feat[clean_proto_mask].detach()
#                 clean_targets = target[clean_proto_mask]
                
#                 for c in range(self.num_classes):
#                     c_mask = (clean_targets == c)
#                     if c_mask.any():
#                         c_feat_mean = clean_feats[c_mask].mean(dim=0)
#                         if self.prototypes[c].sum() == 0:
#                             self.prototypes[c] = c_feat_mean
#                         else:
#                             self.prototypes[c] = 0.99 * self.prototypes[c] + 0.01 * c_feat_mean

#             # D.2 从 ignore_index 点池中打捞高置信度长尾地物
#             unclass_mask = (target == self.ignore_index)
#             # 注意: 确保原型已被初始化 (非全零) 才开始打捞
#             if unclass_mask.any() and self.prototypes.abs().sum() > 0:
#                 unclass_feats = final_feat[unclass_mask]
#                 norm_feats = F.normalize(unclass_feats, p=2, dim=-1)
#                 norm_protos = F.normalize(self.prototypes, p=2, dim=-1)
                
#                 # 计算余弦相似度矩阵 [N_unclass, C]
#                 sim_matrix = torch.mm(norm_feats, norm_protos.t())
#                 max_sim, pseudo_labels = sim_matrix.max(dim=-1)
                
#                 confident_mask = max_sim > self.tau_pseudo
#                 if confident_mask.any():
#                     p_logits = seg_logits[unclass_mask][confident_mask]
#                     p_targets = pseudo_labels[confident_mask]
#                     # 单独计算打捞补偿损失
#                     pseudo_loss = F.cross_entropy(p_logits, p_targets) * self.pseudo_weight

#         # =====================================================================
#         # 3. 极度优雅的 Loss 派发 (完全不污染 criteria 的内部代码)
#         # =====================================================================
#         # 主损失 (传入被净化过的 clean_target)
#         main_loss = self.criteria(seg_logits, clean_target)
        
#         # 辅助损失 (HDS 分支全部受到抗噪掩码的保护)
#         aux_loss = torch.tensor(0.0, device=seg_logits.device)
#         if len(aux_logits_list) > 0 and self.aux_criteria is not None:
#             for i, aux_logits in enumerate(aux_logits_list):
#                 stage_loss = self.aux_criteria(aux_logits, clean_target)
#                 aux_loss = aux_loss + stage_loss * self.aux_weights[i]

#         total_loss = main_loss + aux_loss + pseudo_loss
        
#         return_dict["loss"] = total_loss
#         return_dict["l_aux"] = aux_loss
#         return_dict["l_pseudo"] = pseudo_loss
        
#         return return_dict


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
