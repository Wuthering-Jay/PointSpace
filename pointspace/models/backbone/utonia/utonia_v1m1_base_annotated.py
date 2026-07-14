"""
Utonia V1M1

Author: Yujia Zhang (yujia.zhang.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

# -----------------------------------------------------------------------------
# 中文注释版说明
# -----------------------------------------------------------------------------
# 本文件是 `utonia_v1m1_base.py` 的中文注释版副本。
#
# Utonia-v1m1 的核心训练思想可以概括为：
# 1. 使用一个 3D student backbone 学习点云多视图表征；
# 2. 使用一个 3D teacher backbone 作为 EMA/离线 teacher，产生稳定目标分布；
# 3. 对 global view 做 mask prediction，让 student 从被遮挡输入预测 teacher 表征；
# 4. 对 local view 做 unmask prediction，让局部视图对齐主要 global view；
# 5. 可选地使用 frozen 2D encoder，如 DINOv2/SigLIP/RADIO，将 3D 特征对齐到图像 patch 特征；
# 6. 多个 loss 通过配置中的权重组合，最终得到预训练目标。
#
# 这个副本只增加解释性注释，不应该改变任何可执行逻辑。

from itertools import chain
from packaging import version
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch_scatter
import torchvision.transforms
from timm.layers import trunc_normal_
from torch.nn.utils import weight_norm
from transformers import AutoConfig, ViTModel, ViTConfig
from transformers import AutoModel, AutoProcessor
from copy import deepcopy

import pointops
from pointspace.models.utils.structure import Point
from pointspace.models.builder import MODELS, build_model
from pointspace.models.modules import PointModel
from pointspace.models.utils import (
    offset2batch,
    offset2bincount,
    batch2offset,
    bincount2offset,
)
from pointspace.utils.comm import get_world_size, all_gather
from pointspace.utils.scheduler import CosineScheduler


# -----------------------------------------------------------------------------
# OnlineCluster
# -----------------------------------------------------------------------------
# 这是 Utonia 中 student/teacher head 的基础模块。
#
# 它的功能类似 DINO/iBOT/SwAV 系列方法里的 projection head + prototype head：
# - 先把 backbone 输出特征投影到 embedding 空间；
# - 再做 L2 normalize；
# - 最后用一组 prototype 计算相似度 logits。
#
# 输出不是分类标签，而是“每个点/token 属于各 prototype 的相似度分布”。
# teacher 分支会通过 sinkhorn_knopp 把 logits 转成平衡 assignment；
# student 分支用 softmax 后去匹配 teacher assignment。
class OnlineCluster(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels=4096,
        embed_channels=512,
        num_prototypes=4096,
        enable_mlp=True,
    ):
        super().__init__()

        # enable_mlp=True 时使用两层 MLP projection：
        # in_channels -> hidden_channels -> embed_channels。
        # student/teacher 的 3D head 默认使用 MLP；
        # enc2d teacher head 会关闭 MLP，只保留 prototype。
        if enable_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.GELU(),
                nn.Linear(hidden_channels, embed_channels),
            )
        self.apply(self._init_weights)

        # prototype 使用 weight normalization。
        # 这里兼容 PyTorch 2.1 前后的 API 差异：
        # - torch >= 2.1 使用 torch.nn.utils.parametrizations.weight_norm；
        # - 旧版本使用 torch.nn.utils.weight_norm。
        #
        # prototype 的范数参数被固定为 1，只训练方向。
        # 这样 prototype 更接近单位球面上的聚类中心。
        if version.parse(torch.__version__) >= version.parse("2.1.0"):
            self.prototype = torch.nn.utils.parametrizations.weight_norm(
                nn.Linear(embed_channels, num_prototypes, bias=False)
            )
            self.prototype.parametrizations.weight.original0.data.fill_(1)
            self.prototype.parametrizations.weight.original0.requires_grad = False

        else:
            self.prototype = torch.nn.utils.weight_norm(
                nn.Linear(embed_channels, num_prototypes, bias=False)
            )
            self.prototype.weight_g.data.fill_(1)
            self.prototype.weight_g.requires_grad = False

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, feat):
        # 输入 feat 形状通常为 [num_points_or_tokens, channels]。
        # 如果存在 MLP，先投影到 prototype 使用的 embedding 空间。
        if hasattr(self, "mlp"):
            feat = self.mlp(feat)

        # 按最后一维做 L2 normalize。
        # eps 在 fp16 下稍大，避免半精度数值不稳定。
        eps = 1e-6 if feat.dtype == torch.float16 else 1e-12
        feat = nn.functional.normalize(feat, dim=-1, p=2, eps=eps)

        # 与 prototype 计算相似度，输出 shape:
        # [num_points_or_tokens, num_prototypes]。
        similarity = self.prototype(feat)
        return similarity


# -----------------------------------------------------------------------------
# Utonia-v1m1 主模型
# -----------------------------------------------------------------------------
# 该类是预训练时真正使用的模型封装。
#
# 它内部同时维护：
# - self.student：可训练的 3D backbone 和 head；
# - self.teacher：不直接反传的 teacher backbone/head，通过 EMA 或离线权重提供目标；
# - self.enc2d_model：冻结的 2D 图像 encoder，用于 2D-3D 对齐；
# - 多个 scheduler：mask size、mask ratio、teacher temperature、EMA momentum。
#
# 注意：这里的 backbone 通常来自配置中的 `PT-v3m3`。
@MODELS.register_module("Utonia-v1m1")
class Utonia(PointModel):
    def __init__(
        self,
        image_weight_name,
        image_weight_path,
        backbone,
        head_in_channels,
        backbone_out_channels,
        embedding_channels,
        patch_w,
        patch_h,
        student_pretrained_path=None,
        teacher_pretrained_path=None,
        student_pretrained=False,
        head_hidden_channels=4096,
        head_embed_channels=512,
        head_num_prototypes=4096,
        enc2d_head_in_channels=384,
        enc2d_head_hidden_channels=4096,
        enc2d_head_embed_channels=256,
        enc2d_head_num_prototypes=384,
        teacher_custom=None,
        num_global_view=2,
        num_local_view=4,
        mask_size_start=5,
        mask_size_base=20,
        mask_size_warmup_ratio=0.05,
        mask_ratio_start=0.3,
        mask_ratio_base=0.7,
        mask_ratio_warmup_ratio=0.05,
        mask_jitter=None,
        teacher_temp_start=0.04,
        teacher_temp_base=0.07,
        teacher_temp_warmup_ratio=0.05,
        student_temp=0.1,
        mask_loss_weight=2 / 10,
        roll_mask_loss_weight=2 / 10,
        unmask_loss_weight=4 / 10,
        enc2d_loss_weight=2 / 10,
        momentum_base=0.996,
        momentum_final=1,
        match_max_k=8,
        match_max_r=0.08,
        up_cast_level=2,
        enc2d_upcast_level=4,
        enc2d_cos_shift=True,
        sonata_model_type="offline",
    ):
        super(Utonia, self).__init__()

        # teacher backbone 的来源模式：
        # - online：teacher backbone 初始化为 student 的拷贝，并在训练中 EMA 更新；
        # - offline：teacher backbone 从 teacher_pretrained_path 加载，训练中不更新 backbone。
        assert sonata_model_type in ["online", "offline"]

        # 各个 loss 的权重。
        # 最终 total loss = 各子 loss * 对应权重 后求和。
        self.mask_loss_weight = mask_loss_weight
        self.roll_mask_loss_weight = roll_mask_loss_weight
        self.unmask_loss_weight = unmask_loss_weight
        self.enc2d_loss_weight = enc2d_loss_weight

        # 一个 batch 中每个样本会产生多少 global/local view。
        # 配置中通常是 2 个 global view 和 4 个 local view。
        self.num_global_view = num_global_view
        self.num_local_view = num_local_view

        # masking and scheduler
        # mask_size 控制 mask patch 的空间尺寸，会在训练开始后由 CosineScheduler 动态更新。
        # 初期较小，warmup 后变为 base 值。
        self.mask_size = mask_size_start
        self.mask_size_start = mask_size_start
        self.mask_size_base = mask_size_base
        self.mask_size_warmup_ratio = mask_size_warmup_ratio
        self.mask_size_scheduler = None

        # mask_ratio 控制有多少空间 patch 被 mask。
        # 初始值与目标值也通过 scheduler 平滑过渡。
        self.mask_ratio = mask_ratio_start
        self.mask_ratio_start = mask_ratio_start
        self.mask_ratio_base = mask_ratio_base
        self.mask_ratio_warmup_ratio = mask_ratio_warmup_ratio
        self.mask_ratio_scheduler = None

        # mask_jitter 不为 None 时，被 mask 点的坐标会加入随机扰动。
        # 这样 student 不能简单依赖原始精确坐标。
        self.mask_jitter = mask_jitter

        # temperature and scheduler
        # teacher_temp 用于 teacher logits -> balanced assignment 的温度。
        # 温度越低，teacher 分布越尖锐。
        self.teacher_temp = teacher_temp_start
        self.teacher_temp_start = teacher_temp_start
        self.teacher_temp_base = teacher_temp_base
        self.teacher_temp_warmup_ratio = teacher_temp_warmup_ratio
        self.teacher_temp_scheduler = None

        # student_temp 用于 student logits 的 softmax 温度。
        self.student_temp = student_temp

        # momentum and scheduler
        # EMA 动量。越接近 1，teacher 更新越慢。
        # momentum_final=1 表示训练末期 teacher 基本冻结。
        self.momentum = momentum_base
        self.momentum_base = momentum_base
        self.momentum_final = momentum_final
        self.momentum_scheduler = None

        # dynamic matching
        # 跨视图点匹配的限制：
        # - match_max_k 当前 match_neighbour 固定查询 k=1，参数保留给可能的扩展；
        # - match_max_r 是最近邻距离阈值，超过该半径的点对会被丢弃。
        self.match_max_k = match_max_k
        self.match_max_r = match_max_r

        # up cast level
        # up_cast_level 控制从 backbone 最深层向上恢复多少级特征。
        # backbone 的 GridPooling 会保留 pooling_parent/pooling_inverse，
        # up_cast 会把深层特征按 inverse 拼回父层，并与父层 feat concat。
        self.up_cast_level = up_cast_level
        self.enc2d_upcast_level = enc2d_upcast_level

        # one of unmask, mask, roll mask loss enable
        assert (
            unmask_loss_weight
            + mask_loss_weight
            + roll_mask_loss_weight
            + enc2d_loss_weight
            > 0
        )
        # roll mask loss need more than one global view
        # roll_mask_loss 需要至少两个 global view，因为它会把 global view 两两交换作为目标。
        assert num_global_view > 1 or roll_mask_loss_weight == 0
        # current roll mask only support two global views
        # 当前 roll_point 的实现只支持 1 或 2 个 global view。
        assert num_global_view == 1 or num_global_view == 2

        student_model_dict = dict()
        teacher_model_dict = dict()
        if teacher_custom is None:
            teacher_custom = {}

        # 构建 student backbone。
        # student 是反向传播更新的主模型。
        student_backbone = build_model(backbone)
        if student_pretrained_path != None:
            print("Load pretrained student model")
            student_backbone = self.load_sonata(
                student_backbone, path=student_pretrained_path
            )

        # turn off parameters like drop path for teacher model
        # 用 teacher_custom 覆盖 backbone 配置。
        # 典型用法是把 teacher 的 drop_path/dropout 关掉，使 teacher 目标更稳定。
        # 注意：这里直接 backbone.update(...)，会原地修改传入的 backbone dict。
        backbone.update(teacher_custom)

        # 构建 teacher backbone。
        # offline 模式会从 teacher_pretrained_path 加载；
        # online 模式稍后会用 student 权重初始化。
        teacher_backbone = build_model(backbone)
        if sonata_model_type == "offline":
            teacher_backbone = self.load_sonata(
                teacher_backbone, path=teacher_pretrained_path
            )
        student_model_dict["backbone"] = student_backbone
        teacher_model_dict["backbone"] = teacher_backbone

        if self.enc2d_loss_weight > 0:
            # 启用 2D-3D 对齐 loss 时，加载冻结的 2D 图像 encoder。
            # patch_h/patch_w 需要与数据增强后的图像 patch 网格一致。
            self.patch_h = patch_h
            self.patch_w = patch_w
            self.image_weight_name = image_weight_name

            # Load Model
            self.enc2d_model = self.load_enc2d(image_weight_name, image_weight_path)

            # 2D encoder 只作为 teacher/target，不参与训练。
            self.enc2d_model.requires_grad_(False)
            self._num_channels = enc2d_head_in_channels

            # 将 3D backbone 拼接/上采样后的通道投影到 2D feature 通道数。
            self.patch_proj = torch.nn.Linear(backbone_out_channels, self._num_channels)

            # enc2d_head_student/teacher 当前主要用于 prototype 同步，
            # 但 forward 中实际 2D-3D loss 使用的是 cosine similarity，
            # 而不是 enc2d_head 的 sinkhorn 目标。
            enc2d_head = partial(
                OnlineCluster,
                in_channels=enc2d_head_in_channels,
                hidden_channels=enc2d_head_hidden_channels,
                embed_channels=enc2d_head_in_channels,
                num_prototypes=enc2d_head_num_prototypes,
            )
            enc2d_head_ = partial(
                OnlineCluster,
                in_channels=enc2d_head_in_channels,
                hidden_channels=enc2d_head_hidden_channels,
                embed_channels=enc2d_head_in_channels,
                num_prototypes=enc2d_head_num_prototypes,
                enable_mlp=False,
            )

            self.enc2d_head_student = enc2d_head()
            self.enc2d_head_teacher = enc2d_head_()
            self.enc2d_head_student.prototype.load_state_dict(
                self.enc2d_head_teacher.prototype.state_dict()
            )
            for p in self.enc2d_head_teacher.parameters():
                p.requires_grad = False

        head = partial(
            OnlineCluster,
            in_channels=head_in_channels,
            hidden_channels=head_hidden_channels,
            embed_channels=head_embed_channels,
            num_prototypes=head_num_prototypes,
        )

        # 根据 loss 权重按需创建 head。
        # mask_head：用于 masked global view 预测 teacher global view；
        # unmask_head：用于 local view 预测 principal global view。
        if self.mask_loss_weight > 0 or self.roll_mask_loss_weight > 0:
            student_model_dict["mask_head"] = head()
            teacher_model_dict["mask_head"] = head()
        if self.unmask_loss_weight > 0:
            student_model_dict["unmask_head"] = head()
            teacher_model_dict["unmask_head"] = head()
        if (
            self.enc2d_loss_weight > 0
            and self.unmask_loss_weight
            + self.mask_loss_weight
            + self.roll_mask_loss_weight
            == 0
        ):
            student_model_dict["unmask_head"] = head()
            teacher_model_dict["unmask_head"] = head()

        # student/teacher 都是 ModuleDict，便于统一访问 backbone/head。
        self.student = nn.ModuleDict(student_model_dict)
        self.teacher = nn.ModuleDict(teacher_model_dict)

        # teacher head 初始化为 student head 的拷贝。
        for k, v in self.student.items():
            if "head" in k:
                self.teacher[k].load_state_dict(self.student[k].state_dict())

        # online 模式下 teacher backbone 也初始化为 student backbone。
        if sonata_model_type == "online":
            self.teacher.backbone.load_state_dict(self.student.backbone.state_dict())

        # teacher 所有参数不直接反传，只通过 EMA 或初始化权重更新。
        for n, p in self.teacher.named_parameters():
            p.requires_grad = False

        self.enc2d_cos_shift = enc2d_cos_shift
        self.sonata_model_type = sonata_model_type

    def load_enc2d(self, model_name, model_weight):
        # 加载 2D foundation model。
        # trust_remote_code=True 允许 HuggingFace 模型仓库自定义实现，
        # 因此 image_weight_path 可以指向 DINOv2/SigLIP/RADIO 等不同模型。
        model = AutoModel.from_pretrained(model_weight, trust_remote_code=True)
        return model.eval()

    def load_sonata(self, model, path):
        # 加载 Sonata/PT backbone 权重。
        # 兼容两种 checkpoint 格式：
        # 1. checkpoint["state_dict"] 中带 module.student.backbone. 前缀；
        # 2. 直接是模型 state_dict。
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage.cuda())
        weight = {}
        whether_weight = False
        if "state_dict" in checkpoint.keys():
            checkpoint = checkpoint["state_dict"]
            for key, value in checkpoint.items():
                if "module.student.backbone." in key:
                    whether_weight = True
                    key = key.replace("module.student.backbone.", "module.")
                    key = key[7:]  # module.xxx.xxx -> xxx.xxx
                    weight[key] = value
        if whether_weight:
            load_state_info = model.load_state_dict(weight)
        else:
            load_state_info = model.load_state_dict(checkpoint)
        print(f"Missing keys: {load_state_info[0]}")
        print(f"Unexpected keys: {load_state_info[1]}")
        return model

    @torch.no_grad()
    def ENC2D_forward(self, x):
        # 冻结 2D encoder 的前向。
        # 输入 x 通常是经过 ImgAugmentation 和 ImageNet normalize 后的图像 batch。
        # 输出 features 统一整理成 [num_images, patch_h * patch_w, channels]。

        # RADIO
        if "radio" in self.image_weight_name:
            summary, features = self.enc2d_model(x)
            features = features.reshape(
                -1, self.patch_h * self.patch_w, self._num_channels
            )
            return features
        # SigLIPv2
        if hasattr(self.enc2d_model, "vision_model"):
            outputs = self.enc2d_model.vision_model(x)
            features = outputs.last_hidden_state
        # DINOv2.5
        else:
            outputs = self.enc2d_model(x)
            features = outputs.last_hidden_state[:, -self.patch_h * self.patch_w :, :]
        return features

    def before_train(self):
        # make ModelHook after CheckPointLoader
        # 这个方法一般由 ModelHook 在训练开始前调用。
        # 它根据 scheduler.total_steps 初始化 Utonia 内部的动态调度器。
        total_steps = self.trainer.cfg.scheduler.total_steps
        curr_step = self.trainer.start_epoch * len(self.trainer.train_loader)
        # mask size scheduler
        self.mask_size_scheduler = CosineScheduler(
            start_value=self.mask_size_start,
            base_value=self.mask_size_base,
            final_value=self.mask_size_base,
            warmup_iters=int(total_steps * self.mask_size_warmup_ratio),
            total_iters=total_steps,
        )
        self.mask_size_scheduler.iter = curr_step

        # mask ratio scheduler
        self.mask_ratio_scheduler = CosineScheduler(
            start_value=self.mask_ratio_start,
            base_value=self.mask_ratio_base,
            final_value=self.mask_ratio_base,
            warmup_iters=int(total_steps * self.mask_ratio_warmup_ratio),
            total_iters=total_steps,
        )
        self.mask_ratio_scheduler.iter = curr_step

        # teacher temperature scheduler
        self.teacher_temp_scheduler = CosineScheduler(
            start_value=self.teacher_temp_start,
            base_value=self.teacher_temp_base,
            final_value=self.teacher_temp_base,
            warmup_iters=int(total_steps * self.teacher_temp_warmup_ratio),
            total_iters=total_steps,
        )
        self.teacher_temp_scheduler.iter = curr_step

        # momentum scheduler
        self.momentum_scheduler = CosineScheduler(
            base_value=self.momentum_base,
            final_value=self.momentum_final,
            total_iters=total_steps,
        )
        self.momentum_scheduler.iter = curr_step

    def before_step(self):
        # update parameters from schedulers
        # 每个 iteration 开始前更新动态训练参数。
        # 这些值不是 optimizer 的学习率，而是 Utonia 自监督目标的内部状态。
        self.mask_size = self.mask_size_scheduler.step()
        self.mask_ratio = self.mask_ratio_scheduler.step()
        self.teacher_temp = self.teacher_temp_scheduler.step()
        self.momentum = self.momentum_scheduler.step()

        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar(
                "params/mask_size",
                self.mask_size,
                self.mask_size_scheduler.iter,
            )
            self.trainer.writer.add_scalar(
                "params/mask_ratio",
                self.mask_ratio,
                self.mask_ratio_scheduler.iter,
            )
            self.trainer.writer.add_scalar(
                "params/teacher_temp",
                self.teacher_temp,
                self.teacher_temp_scheduler.iter,
            )
            self.trainer.writer.add_scalar(
                "params/momentum",
                self.momentum,
                self.momentum_scheduler.iter,
            )

    def after_step(self):
        # pass
        # EMA update teacher
        # 每个 iteration 结束后更新 teacher。
        # teacher = m * teacher + (1 - m) * student。
        # 这样 teacher 的目标分布比 student 更平滑、更稳定。
        with torch.no_grad():
            m = self.momentum

            # online 模式下，teacher backbone 也跟随 student backbone EMA 更新。
            if self.sonata_model_type == "online":
                student_param_list = list(self.student.backbone.parameters())
                teacher_param_list = list(self.teacher.backbone.parameters())
                torch._foreach_mul_(teacher_param_list, m)
                torch._foreach_add_(teacher_param_list, student_param_list, alpha=1 - m)

            # head 始终通过 EMA 更新。
            student_param_list = [
                p for n, p in self.student.named_parameters() if "head" in n
            ]
            teacher_param_list = [
                p for n, p in self.teacher.named_parameters() if "head" in n
            ]
            torch._foreach_mul_(teacher_param_list, m)
            torch._foreach_add_(teacher_param_list, student_param_list, alpha=1 - m)

            if self.enc2d_loss_weight > 0:
                # enc2d teacher head 的 prototype 直接拷贝 student prototype。
                # 注意这里是 copy，不是 EMA。
                enc2d_student_param_list = [
                    p
                    for n, p in self.enc2d_head_student.named_parameters()
                    if "prototype" in n
                ]
                enc2d_teacher_param_list = [
                    p
                    for n, p in self.enc2d_head_teacher.named_parameters()
                    if "prototype" in n
                ]
                torch._foreach_copy_(enc2d_teacher_param_list, enc2d_student_param_list)

    @staticmethod
    def sinkhorn_knopp(feat, temp, num_iter=3):
        # 将 teacher logits 转成 balanced assignment。
        #
        # 输入：
        # - feat: [N, K]，N 为点/token 数，K 为 prototype 数；
        # - temp: teacher temperature。
        #
        # 输出：
        # - [N, K]，每个样本对 prototype 的软分配。
        #
        # 该实现会在分布式训练中 all_reduce 行/总和，使所有 GPU 上的 prototype 分配更均衡。
        feat = feat.float()
        q = torch.exp(feat / temp).t()
        n = sum(all_gather(q.shape[1]))  # number of samples to assign
        k = q.shape[0]  # number of prototypes

        # make the matrix sums to 1
        sum_q = q.sum()
        if get_world_size() > 1:
            dist.all_reduce(sum_q)
        q = q / sum_q

        for i in range(num_iter):
            # normalize each row: total weight per prototype must be 1/k
            q_row_sum = q.sum(dim=1, keepdim=True)
            if get_world_size() > 1:
                dist.all_reduce(q_row_sum)
            q = q / q_row_sum / k

            # normalize each column: total weight per sample must be 1/n
            q = q / q.sum(dim=0, keepdim=True) / n

        q *= n  # the columns must sum to 1 so that Q is an assignment
        return q.t()

    def generate_mask(self, coord, offset, grid_size):
        # 根据点坐标生成空间 mask。
        #
        # 步骤：
        # 1. 由 offset 得到每个点所属 batch；
        # 2. 按当前 mask_size * grid_size 把点云划分为空间 patch；
        # 3. 随机选择 mask_ratio 比例的 patch；
        # 4. 属于这些 patch 的点被标记为 mask。
        batch = offset2batch(offset)
        mask_size = self.mask_size * grid_size
        mask_ratio = self.mask_ratio

        # Grouping points with grid patch
        min_coord = torch_scatter.segment_coo(coord, batch, reduce="min")
        grid_coord = ((coord - min_coord[batch]) // mask_size).int()
        grid_coord = torch.cat([batch.unsqueeze(-1), grid_coord], dim=-1)
        unique, point_cluster, counts = torch.unique(
            grid_coord, dim=0, sorted=True, return_inverse=True, return_counts=True
        )
        patch_num = unique.shape[0]
        mask_patch_num = int(patch_num * mask_ratio)
        patch_index = torch.randperm(patch_num, device=coord.device)
        mask_patch_index = patch_index[:mask_patch_num]
        point_mask = torch.isin(point_cluster, mask_patch_index)
        return point_mask, point_cluster

    @torch.no_grad()
    def match_neighbour(
        self,
        view1_coord,
        view1_offset,
        view2_coord,
        view2_offset,
    ):
        # 在两个视图之间做最近邻点匹配。
        #
        # view1 是 prediction 侧，view2 是 target 侧。
        # pointops.knn_query(k=1) 为 view1 中每个点找 view2 中最近点；
        # 距离超过 match_max_r 的匹配会被过滤掉。
        index2, distance = pointops.knn_query(
            1,
            view2_coord.float(),
            view2_offset.int(),
            view1_coord.float(),
            view1_offset.int(),
        )
        index1 = torch.arange(
            index2.shape[0], device=index2.device, dtype=torch.long
        ).unsqueeze(-1)
        index = torch.cat([index1, index2], dim=-1)[
            distance.squeeze(-1) < self.match_max_r
        ]
        return index

    @torch.no_grad()
    def roll_point(self, point):
        # 将 global view 两两交换，用于 roll_mask_loss。
        #
        # 输入 view 顺序假设为：
        # [pc1_view0, pc1_view1, pc2_view0, pc2_view1, ...]
        #
        # 输出变为：
        # [pc1_view1, pc1_view0, pc2_view1, pc2_view0, ...]
        #
        # 这样 masked student 的某个 global view 可以预测另一个 global view 的 teacher 目标。
        n = self.num_global_view
        # [pc1, pc1', pc2, pc2'] -> [pc1', pc1, pc2', pc2], only support num_global_view == 2
        bs = len(point.offset) // self.num_global_view
        data_dict = {}
        for key in point.keys():
            if key in ["feat", "coord", "origin_coord", "batch"]:
                value = point[key].split(offset2bincount(point.offset).tolist())
                value = chain(*[value[n * b : n * (b + 1)][::-1] for b in range(bs)])
                if key == "batch":
                    value = [torch.ones_like(v) * i for i, v in enumerate(value)]
                data_dict[key] = torch.cat(list(value), dim=0)
        return Point(data_dict)

    def up_cast(self, point, upcast_level=None):
        # 沿 backbone 的 pooling 记录向上恢复特征分辨率。
        #
        # Point Transformer backbone 在 GridPooling 时，如果 traceable=True，
        # 会在 point 中留下：
        # - pooling_parent: 下采样前的父 Point；
        # - pooling_inverse: 父点到当前 pooled 点的映射。
        #
        # up_cast 的每一级都会：
        # 1. 取出 parent 和 inverse；
        # 2. 将当前深层特征 point.feat[inverse] 映射回 parent 点；
        # 3. 与 parent.feat 拼接；
        # 4. 返回更高分辨率的 parent。
        #
        # 因此 upcast_level 越大，输出点越接近原始分辨率，通道也会因为 concat 变宽。
        if upcast_level is None:
            upcast_level = self.up_cast_level
        else:
            upcast_level = upcast_level
        for _ in range(upcast_level):
            assert "pooling_parent" in point.keys()
            assert "pooling_inverse" in point.keys()
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = parent
        return point

    @staticmethod
    def _resolve_grid_size(grid_size):
        # 单数据集/纯 3D 管线里 grid_size 可能是 float；
        # 官方多视图管线或 ToTensor 后也可能是 Tensor/list。
        # 训练逻辑只需要一个标量，因此这里统一取出标量值。
        if torch.is_tensor(grid_size):
            return grid_size.reshape(-1)[0]
        if isinstance(grid_size, (list, tuple)):
            if len(grid_size) == 0:
                raise ValueError("grid_size cannot be an empty sequence.")
            return Utonia._resolve_grid_size(grid_size[0])
        return grid_size

    @staticmethod
    def pool_corr(point, correspondence):
        # 将 2D-3D correspondence 从原始点层级同步池化到当前 point 层级。
        #
        # correspondence 通常形状类似 [num_points, img_num, 2]，
        # 其中最后一维是图像 patch 坐标 [row, col]；
        # [-1, -1] 表示该点在对应图像中不可见或无效。
        #
        # backbone 下采样后，一个 pooled 点对应多个父点。
        # 这里会沿 pooling 记录逐级聚合 correspondence：
        # - 如果 cluster 中没有有效 correspondence，则保持 -1；
        # - 如果有多个有效 correspondence，则取平均坐标。
        inverse_list = []
        idx_ptr_list = []
        point_feat = dict(offset=point.offset, feat=point.feat)
        while "pooling_parent" in point.keys():
            assert "pooling_parent" in point.keys()
            assert "pooling_inverse" in point.keys()
            assert "idx_ptr" in point.keys()
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            idx_ptr = point.pop("idx_ptr")
            inverse_list.append(inverse)
            idx_ptr_list.append(idx_ptr)
            point = parent
        inverse_list.reverse()
        idx_ptr_list.reverse()
        for inverse, idx_ptr in zip(inverse_list, idx_ptr_list):
            _, indices = torch.sort(inverse)
            img_num = correspondence.shape[1]
            correspondence_all = []
            if img_num == 0:
                correspondence_all = -torch.ones((idx_ptr.shape[0] - 1, 0, 2)).cuda()
            else:
                for img_id in range(img_num):
                    mask = torch.all(
                        correspondence[:, img_id] != torch.tensor([-1, -1]).cuda(),
                        dim=1,
                    ).float()
                    counts = torch_scatter.segment_csr(
                        mask[indices], idx_ptr, reduce="sum"
                    )
                    counts[counts == 0] = 100000
                    correspondence_img = deepcopy(correspondence[:, img_id])
                    correspondence_img[correspondence_img == -1] = 0
                    mask_sum = torch_scatter.segment_csr(
                        correspondence_img[indices], idx_ptr, reduce="sum"
                    )
                    mask_sum = mask_sum / counts.unsqueeze(1)
                    mask_sum[counts == 100000] = -1
                    correspondence_all.append(mask_sum)
                correspondence_all = torch.stack(correspondence_all, dim=1)
            correspondence = correspondence_all
        point_feat["correspondence"] = correspondence
        point_feat = Point(point_feat)
        return point_feat

    def forward(self, data_dict, return_point=False):
        # Utonia 前向有两种模式：
        #
        # 1. return_point=True：
        #    只跑 teacher backbone，并返回 up_cast 后的 point 特征。
        #    这通常用于提取特征，而不是训练 loss。
        #
        # 2. return_point=False：
        #    执行完整预训练流程，返回包含 total loss 和各子 loss 的字典。
        if return_point:
            point = self.teacher.backbone(data_dict)
            for _ in range(self.up_cast_level):
                assert "pooling_parent" in point.keys()
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            return dict(point=point)

        # prepare global_point, mask_global_point, local_point
        # 这里先在 no_grad 下构造输入 Point，并计算 teacher 目标。
        # teacher 分支不参与反传，因此可以节省显存。
        with torch.no_grad():
            grid_size = self._resolve_grid_size(data_dict["grid_size"])
            # global_point & masking
            # global_point 是未 mask 的全局视图，用于 teacher 产生目标。
            global_point = Point(
                feat=data_dict["global_feat"],
                coord=data_dict["global_coord"],
                origin_coord=data_dict["global_origin_coord"],
                offset=data_dict["global_offset"],
                grid_size=grid_size,
            )

            global_mask, global_cluster = self.generate_mask(
                global_point.coord, global_point.offset, global_point.grid_size
            )
            mask_global_coord = global_point.coord.clone().detach()

            # 对被 mask 的点坐标加入 jitter。
            # 这样即使 mask token 替换了特征，坐标也不会完全暴露原始局部结构。
            if self.mask_jitter is not None:
                mask_global_coord[global_mask] += torch.clip(
                    torch.randn_like(mask_global_coord[global_mask]).mul(
                        self.mask_jitter * grid_size
                    ),
                    max=(self.mask_jitter * grid_size) * 2,
                )

            # mask_global_point 是 student 的 masked global 输入。
            # 其中 mask 字段会被 backbone 的 Embedding 使用，将对应 feat 替换为 mask_token。
            mask_global_point = Point(
                feat=data_dict["global_feat"],
                coord=mask_global_coord,
                origin_coord=data_dict["global_origin_coord"],
                mask=global_mask,
                offset=data_dict["global_offset"],
                grid_size=grid_size,
            )
            # 只有启用 2D-3D 对齐时才需要 global_correspondence。
            # 无影像数据集（如只有 xyz/echo 的 DALES）可设置 enc2d_loss_weight=0，
            # 此时不要求 data_dict 中存在 global_correspondence/images/img_num。
            major_view_correspondence = (
                data_dict["global_correspondence"]
                if self.enc2d_loss_weight > 0
                else None
            )

            # local point & matching
            # local_point 是 student 的 local view 输入，用于 unmask loss。
            local_point = Point(
                feat=data_dict["local_feat"],
                coord=data_dict["local_coord"],
                origin_coord=data_dict["local_origin_coord"],
                offset=data_dict["local_offset"],
                grid_size=grid_size,
            )

            # create result dictionary for return
            result_dict = dict(loss=[])
            # teacher forward
            # teacher 对未 mask 的 global view 提取特征，并 up_cast 到需要的分辨率。
            global_point_ = self.teacher.backbone(global_point)
            global_point_ = self.up_cast(global_point_)
            # teacher head forward
            # only use one shared head for both mask and unmask
            # priority: mask (global) > unmask (local)
            # 如果启用了 mask/roll mask，就用 mask_head 产生 teacher prototype logits；
            # 否则用 unmask_head。
            if self.mask_loss_weight > 0 or self.roll_mask_loss_weight > 0:
                global_point_.feat = self.teacher.mask_head(global_point_.feat)
            else:
                global_point_.feat = self.teacher.unmask_head(global_point_.feat)

        if self.mask_loss_weight > 0 or self.roll_mask_loss_weight > 0:
            # student forward
            # student 处理 masked global view。
            mask_global_point_ = self.student.backbone(mask_global_point)
            mask_global_point_ = self.up_cast(mask_global_point_)
            mask_pred_sim = self.student.mask_head(mask_global_point_.feat)

            if self.mask_loss_weight > 0:
                with torch.no_grad():
                    # 将 student masked view 的点，与 teacher unmasked global view 的点做最近邻匹配。
                    match_index = self.match_neighbour(
                        mask_global_point_.origin_coord,
                        mask_global_point_.offset,
                        global_point_.origin_coord,
                        global_point_.offset,
                    )
                    # teacher forward
                    # 取匹配到的 teacher logits，通过 Sinkhorn 得到 balanced target。
                    mask_target_sim = self.sinkhorn_knopp(
                        global_point_.feat[match_index[:, 1]],
                        self.teacher_temp,
                    )

                # loss
                # 对 student logits 做 temperature softmax，
                # 与 teacher balanced assignment 做交叉熵。
                mask_loss = -torch.sum(
                    mask_target_sim
                    * F.log_softmax(
                        mask_pred_sim[match_index[:, 0]] / self.student_temp, dim=-1
                    ),
                    dim=-1,
                )

                mask_loss = torch_scatter.segment_coo(
                    mask_loss,
                    index=mask_global_point_.batch[match_index[:, 0]],
                    reduce="mean",
                ).mean()
                result_dict["mask_loss"] = mask_loss
                result_dict["loss"].append(mask_loss * self.mask_loss_weight)

            if self.roll_mask_loss_weight > 0:
                # roll mask loss 使用另一个 global view 的 teacher 输出作为目标。
                roll_global_point_ = self.roll_point(global_point_)
                with torch.no_grad():
                    # match index for pred and roll target
                    match_index = self.match_neighbour(
                        mask_global_point_.origin_coord,
                        mask_global_point_.offset,
                        roll_global_point_.origin_coord,
                        roll_global_point_.offset,
                    )
                    # teacher forward
                    roll_mask_target_sim = self.sinkhorn_knopp(
                        roll_global_point_.feat[match_index[:, 1]],
                        self.teacher_temp,
                    )

                roll_mask_loss = -torch.sum(
                    roll_mask_target_sim
                    * F.log_softmax(
                        mask_pred_sim[match_index[:, 0]] / self.student_temp, dim=-1
                    ),
                    dim=-1,
                )
                roll_mask_loss = torch_scatter.segment_coo(
                    roll_mask_loss,
                    index=mask_global_point_.batch[match_index[:, 0]],
                    reduce="mean",
                ).mean()
                result_dict["roll_mask_loss"] = roll_mask_loss
                result_dict["loss"].append(roll_mask_loss * self.roll_mask_loss_weight)
        if self.unmask_loss_weight > 0:
            # student forward
            # student 处理 local view，并预测主要 global view 的 teacher 目标。
            local_point_ = self.student.backbone(local_point)
            local_point_ = self.up_cast(local_point_)
            unmask_pred_sim = self.student.unmask_head(local_point_.feat)
            with torch.no_grad():
                # 只使用每个样本的第 0 个 global view 作为 principal view。
                principal_view_mask = global_point_.batch % self.num_global_view == 0
                principal_view_batch = (
                    global_point_.batch[principal_view_mask] // self.num_global_view
                )
                # local view 的 offset 这里取 self.num_local_view - 1 :: self.num_local_view，
                # 等价于按每个样本的 local view 组来构造匹配 batch 边界。
                match_index = self.match_neighbour(
                    local_point_.origin_coord,
                    local_point_.offset[self.num_local_view - 1 :: self.num_local_view],
                    global_point_.origin_coord[principal_view_mask],
                    batch2offset(principal_view_batch),
                )
                # teacher forward
                unmask_target_sim = self.sinkhorn_knopp(
                    global_point_.feat[principal_view_mask][match_index[:, 1]],
                    self.teacher_temp,
                )
            # loss
            unmask_loss = -torch.sum(
                unmask_target_sim
                * F.log_softmax(
                    unmask_pred_sim[match_index[:, 0]] / self.student_temp, dim=-1
                ),
                dim=-1,
            )
            unmask_loss = torch_scatter.segment_coo(
                unmask_loss,
                index=local_point_.batch[match_index[:, 0]],
                reduce="mean",
            ).mean()
            result_dict["unmask_loss"] = unmask_loss
            result_dict["loss"].append(unmask_loss * self.unmask_loss_weight)
        if self.enc2d_loss_weight > 0:
            # 2D-3D 对齐 loss。
            # 目标：让可投影到图像 patch 的 3D 点特征，与 frozen 2D encoder 的 patch 特征接近。

            # 如果前面没有算过 masked student 分支，则这里补算一次。
            # 注意原逻辑是 `mask_loss_weight == 0 or roll_mask_loss_weight == 0`，
            # 因此只要二者之一为 0 就会重算。
            if self.mask_loss_weight == 0 or self.roll_mask_loss_weight == 0:
                mask_global_point_ = self.student.backbone(mask_global_point)
                mask_global_point_ = self.up_cast(mask_global_point_)

            # enc2d 对齐可能需要比 SSL loss 更高分辨率的 3D 特征，
            # 因此继续 up_cast 到 enc2d_upcast_level。
            mask_global_point_enc2d = self.up_cast(
                mask_global_point_,
                upcast_level=self.enc2d_upcast_level - self.up_cast_level,
            )

            # 将 correspondence 同步到当前 3D 特征层级。
            to_feature = self.pool_corr(
                mask_global_point_enc2d, major_view_correspondence
            )
            data_dict_global_offset = torch.cat(
                [torch.tensor([0]).cuda(), to_feature["offset"]], dim=0
            )
            enc2d_count = (
                data_dict_global_offset[
                    1 : len(data_dict_global_offset) : self.num_global_view
                ]
                - data_dict_global_offset[
                    0 : len(data_dict_global_offset) - 1 : self.num_global_view
                ]
            )
            enc2d_offset = torch.cat(
                [torch.tensor([0]).cuda(), torch.cumsum(enc2d_count, dim=0)]
            )
            enc2d_mask = torch.cat(
                [
                    torch.arange(0, c, device=enc2d_count.device)
                    + data_dict_global_offset[i * self.num_global_view]
                    for i, c in enumerate(enc2d_count)
                ],
                dim=0,
            )

            offset_points_3d = enc2d_offset[1:]
            batch_points_3d = offset2batch(offset_points_3d)
            imgs = data_dict["images"]
            feature3d = to_feature["feat"][enc2d_mask]
            enc2d_global_mask = global_mask[enc2d_mask]
            correspondence = to_feature["correspondence"][enc2d_mask]
            v0 = correspondence.shape[1]

            # mask 表示每个 3D 点在每个视角图像中是否有有效 patch 对应。
            mask = torch.any(correspondence != torch.tensor([-1, -1]).cuda(), dim=2)
            enc2d_global_mask = enc2d_global_mask.unsqueeze(1).expand(-1, v0)
            valid_index = torch.where(mask)  # 0: 3d points index, 1: view index

            bincount_img_num = data_dict["img_num"]
            offset_img_num = bincount2offset(bincount_img_num)
            total_img_num = offset_img_num[-1]

            if total_img_num > 0:
                # expand
                with torch.no_grad():
                    # frozen 2D encoder 提取每张图的 patch feature。
                    feature2d = self.ENC2D_forward(imgs)
                    feature2d = feature2d.contiguous().view(-1, feature2d.shape[-1])
                    feature2d_mask = feature2d

                offset_img_num = torch.cat([torch.tensor([0]).cuda(), offset_img_num])[
                    :-1
                ]
                batch_index = batch_points_3d[valid_index[0]]
                batch_img_num = offset_img_num[batch_index]

                feature3d_mask = feature3d[valid_index[0]]

                feature_index = torch.cat(
                    [
                        batch_img_num.unsqueeze(-1),
                        valid_index[1].unsqueeze(-1),
                        correspondence[valid_index],
                    ],
                    dim=-1,
                )
                feature_index = feature_index.long()

                # 把 [batch 内图像起始编号, view_id, patch_row, patch_col]
                # 展平成 feature2d 的一维 patch 索引。
                feature_index = (
                    feature_index[:, 0] * self.patch_h * self.patch_w
                    + feature_index[:, 1] * self.patch_h * self.patch_w
                    + feature_index[:, 2] * self.patch_w
                    + feature_index[:, 3]
                )

                feature_index = feature_index.long()

                # 多个 3D 点可能投影到同一个 2D patch。
                # 这里先按 patch index 对 3D 特征求均值，再投影到 2D 通道数。
                feature3d_mask = torch_scatter.scatter_mean(
                    feature3d_mask, feature_index, dim=0, dim_size=feature2d.shape[0]
                )
                feature3d_mask = self.patch_proj(feature3d_mask)
                feature_index = torch.unique(feature_index)
                feature2d_mask = feature2d_mask[feature_index]
                feature3d_mask = feature3d_mask[feature_index]

                if self.enc2d_cos_shift:
                    # 去掉每个 token 自身的均值，相当于做一个简单的中心化，
                    # 再计算 cosine similarity。
                    feature2d_mask = feature2d_mask - feature2d_mask.mean(
                        dim=-1, keepdim=True
                    )
                    feature3d_mask = feature3d_mask - feature3d_mask.mean(
                        dim=-1, keepdim=True
                    )
                cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

                # cosine loss：1 - cos，相似度越高 loss 越小。
                # 乘以 10 是尺度放大，让该项与其他 SSL loss 数值量级更接近。
                loss = (1 - cos(feature2d_mask, feature3d_mask)).mean() * 10

                result_dict["enc2d_loss"] = loss
                result_dict["loss"].append(loss * self.enc2d_loss_weight)
                del (
                    feature2d,
                    feature3d,
                    feature2d_mask,
                    feature3d_mask,
                    correspondence,
                    feature_index,
                )
            elif (
                self.mask_loss_weight
                + self.unmask_loss_weight
                + self.roll_mask_loss_weight
                > 0
            ):
                # 如果当前 batch 没有图像，无法计算 enc2d_loss。
                # 为了保持 loss 字段存在且训练流程不中断，
                # 用已有 SSL loss 的加权平均作为 enc2d_loss 的替代值。
                result_ssl_loss = sum(result_dict["loss"]) / (
                    self.mask_loss_weight
                    + self.unmask_loss_weight
                    + self.roll_mask_loss_weight
                )
                result_dict["enc2d_loss"] = result_ssl_loss
                result_dict["loss"].append(result_ssl_loss * self.enc2d_loss_weight)
        result_dict["loss"] = sum(result_dict["loss"])

        if get_world_size() > 1:
            # 分布式训练下对各 loss 做平均，保证日志和返回值是全局平均。
            for loss_id, loss in result_dict.items():
                dist.all_reduce(loss, op=dist.ReduceOp.AVG)
        return result_dict
