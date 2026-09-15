"""Memory-efficient PTv3m1-compatible encoder for VPSD.

PT-v3m4 keeps the PTv3m1 CPE + serialized-attention + MLP principle while
reducing training activation memory. It preserves all serialization orders;
each block independently samples one order on every forward instead of binding
depth ``i`` to ``i % num_orders``. PDNorm is intentionally disabled because
this variant targets single-dataset training.
"""

import math

import torch
import torch.nn as nn
import torch_scatter
from addict import Dict
from torch.utils.checkpoint import checkpoint

from pointspace.models.builder import MODELS
from pointspace.models.modules import PointModule
from pointspace.models.utils.misc import offset2bincount
from pointspace.models.utils.structure import Point

from ..vpsd.hierarchy import build_encoder_hierarchy
from . import point_transformer_v3m1_base as ptv3m1


def _branch_point(metadata, feature, sparse_template=None):
    data = dict(metadata)
    data["feat"] = feature
    if sparse_template is not None:
        data["sparse_conv_feat"] = sparse_template.replace_feature(feature)
    return Point(data)


def clone_hierarchy_for_decoder(hierarchy, detach_stages=()):
    """Clone Point containers and route decoder gradients by encoder stage.

    Tensor metadata and serialization tensors are shared read-only. Feature
    tensors from selected stages are detached, while decoder parameters remain
    fully trainable. Rebuilding ``pooling_parent`` on the cloned containers
    lets the standard PTv3 decoder consume the hierarchy without popping or
    otherwise mutating the encoder trace used by VPSD.
    """

    detach_stages = {int(stage) for stage in detach_stages}
    num_stages = len(hierarchy)
    invalid = detach_stages.difference(range(num_stages))
    if invalid:
        raise ValueError(f"Invalid decoder detach stages: {sorted(invalid)}")

    cloned = []
    for stage, level in enumerate(hierarchy):
        source = level.point if hasattr(level, "point") else level
        feature = source.feat.detach() if stage in detach_stages else source.feat
        data = {
            key: value
            for key, value in source.items()
            if key
            not in {
                "feat",
                "sparse_conv_feat",
                "pooling_parent",
                "unpooling_parent",
            }
        }
        data["feat"] = feature
        if "sparse_conv_feat" in source.keys():
            data["sparse_conv_feat"] = source.sparse_conv_feat.replace_feature(
                feature
            )
        if stage > 0:
            data["pooling_parent"] = cloned[-1]
        cloned.append(Point(data))
    return tuple(cloned)


class LightweightSerializedAttention(ptv3m1.SerializedAttention):
    """PTv3 attention that gathers ordered input before QKV projection."""

    def __init__(self, source):
        nn.Module.__init__(self)
        self.channels = source.channels
        self.num_heads = source.num_heads
        self.scale = source.scale
        self.order_index = source.order_index
        self.upcast_attention = source.upcast_attention
        self.upcast_softmax = source.upcast_softmax
        self.enable_rpe = source.enable_rpe
        self.enable_flash = source.enable_flash
        self.patch_size = source.patch_size
        if hasattr(source, "patch_size_max"):
            self.patch_size_max = source.patch_size_max
        self.attn_drop = source.attn_drop
        self.qkv = source.qkv
        self.proj = source.proj
        self.proj_drop = source.proj_drop
        self.softmax = source.softmax
        self.rpe = source.rpe

    def sample_order_index(self, point, random_order):
        num_orders = int(point.serialized_order.shape[0])
        if num_orders <= 0:
            raise RuntimeError("serialized_order must contain at least one order")
        if random_order:
            return int(torch.randint(num_orders, size=()).item())
        return int(self.order_index % num_orders)

    def prepare(self, point):
        if not self.enable_flash:
            self.patch_size = min(
                offset2bincount(point.offset).min().tolist(),
                self.patch_size_max,
            )
        return self.get_padding_and_inverse(point)

    def forward_feature(self, point, feature, order_index):
        h = self.num_heads
        patch_size = self.patch_size
        channels = self.channels
        pad, unpad, cu_seqlens = self.get_padding_and_inverse(point)
        order = point.serialized_order[order_index][pad]
        inverse = unpad[point.serialized_inverse[order_index]]

        # PTv3m1 projects all N points and then gathers [N_pad, 3C]. Gathering
        # [N_pad, C] first removes the simultaneous full and ordered QKV buffers.
        qkv = self.qkv(feature[order])
        if not self.enable_flash:
            q, k, v = (
                qkv.reshape(-1, patch_size, 3, h, channels // h)
                .permute(2, 0, 3, 1, 4)
                .unbind(dim=0)
            )
            if self.upcast_attention:
                q = q.float()
                k = k.float()
            attn = (q * self.scale) @ k.transpose(-2, -1)
            if self.enable_rpe:
                attn = attn + self.rpe(self.get_rel_pos(point, order))
            if self.upcast_softmax:
                attn = attn.float()
            attn = self.softmax(attn)
            attn = self.attn_drop(attn).to(qkv.dtype)
            output = (attn @ v).transpose(1, 2).reshape(-1, channels)
        else:
            output = ptv3m1.flash_attn.flash_attn_varlen_qkvpacked_func(
                qkv.to(torch.bfloat16).reshape(-1, 3, h, channels // h),
                cu_seqlens,
                max_seqlen=self.patch_size,
                dropout_p=self.attn_drop if self.training else 0,
                softmax_scale=self.scale,
            ).reshape(-1, channels)
            output = output.to(qkv.dtype)
        output = self.proj(output[inverse])
        return self.proj_drop(output)


class LightweightBlock(PointModule):
    """Checkpointable PTv3 block with independently sampled order."""

    def __init__(
        self,
        source,
        checkpoint_cpe=True,
        checkpoint_attention=True,
        checkpoint_mlp=True,
        random_order=True,
    ):
        super().__init__()
        self.channels = source.channels
        self.pre_norm = source.pre_norm
        self.cpe = source.cpe
        self.norm1 = source.norm1
        self.attn = LightweightSerializedAttention(source.attn)
        self.norm2 = source.norm2
        self.mlp = source.mlp
        self.drop_path = source.drop_path
        self.checkpoint_cpe = bool(checkpoint_cpe)
        self.checkpoint_attention = bool(checkpoint_attention)
        self.checkpoint_mlp = bool(checkpoint_mlp)
        self.random_order = bool(random_order)

    @staticmethod
    def _use_checkpoint(enabled, module):
        return enabled and module.training and torch.is_grad_enabled()

    def forward(self, point):
        metadata = {
            key: value
            for key, value in point.items()
            if key not in {"feat", "sparse_conv_feat"}
        }
        sparse_template = point.sparse_conv_feat

        def cpe_branch(feature):
            local = _branch_point(metadata, feature, sparse_template)
            return self.cpe(local).feat

        shortcut = point.feat
        if self._use_checkpoint(self.checkpoint_cpe, self):
            branch = checkpoint(
                cpe_branch,
                point.feat,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            branch = cpe_branch(point.feat)
        point.feat = shortcut + branch

        # Cache discrete padding outside checkpoint and sample exactly once so
        # backward recomputation uses the same order as forward.
        self.attn.prepare(point)
        order_index = self.attn.sample_order_index(point, self.random_order)
        attention_metadata = {
            key: value
            for key, value in point.items()
            if key not in {"feat", "sparse_conv_feat"}
        }

        def attention_branch(feature):
            local = _branch_point(attention_metadata, feature)
            if self.pre_norm:
                local = self.norm1(local)
            local.feat = self.attn.forward_feature(local, local.feat, order_index)
            return self.drop_path(local).feat

        shortcut = point.feat
        if self._use_checkpoint(self.checkpoint_attention, self):
            branch = checkpoint(
                attention_branch,
                point.feat,
                use_reentrant=False,
                preserve_rng_state=True,
            )
        else:
            branch = attention_branch(point.feat)
        point.feat = shortcut + branch
        if not self.pre_norm:
            point = self.norm1(point)

        mlp_metadata = {
            key: value
            for key, value in point.items()
            if key not in {"feat", "sparse_conv_feat"}
        }

        def mlp_branch(feature):
            local = _branch_point(mlp_metadata, feature)
            if self.pre_norm:
                local = self.norm2(local)
            local = self.mlp(local)
            return self.drop_path(local).feat

        shortcut = point.feat
        if self._use_checkpoint(self.checkpoint_mlp, self):
            branch = checkpoint(
                mlp_branch,
                point.feat,
                use_reentrant=False,
                preserve_rng_state=True,
            )
        else:
            branch = mlp_branch(point.feat)
        point.feat = shortcut + branch
        if not self.pre_norm:
            point = self.norm2(point)
        point.sparse_conv_feat = sparse_template.replace_feature(point.feat)
        return point


class LightweightSerializedPooling(PointModule):
    """Serialized pooling that gathers before channel projection."""

    def __init__(self, source):
        super().__init__()
        self.in_channels = source.in_channels
        self.out_channels = source.out_channels
        self.stride = source.stride
        self.reduce = source.reduce
        self.shuffle_orders = source.shuffle_orders
        self.traceable = source.traceable
        self.proj = source.proj
        if hasattr(source, "norm"):
            self.norm = source.norm
        if hasattr(source, "act"):
            self.act = source.act

    def forward(self, point):
        pooling_depth = (math.ceil(self.stride) - 1).bit_length()
        if pooling_depth > point.serialized_depth:
            pooling_depth = 0
        code = point.serialized_code >> pooling_depth * 3
        _, cluster, counts = torch.unique(
            code[0], sorted=True, return_inverse=True, return_counts=True
        )
        _, indices = torch.sort(cluster)
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        head_indices = indices[idx_ptr[:-1]]
        code = code[:, head_indices]
        order = torch.argsort(code)
        inverse = torch.zeros_like(order).scatter_(
            dim=1,
            index=order,
            src=torch.arange(code.shape[1], device=order.device).repeat(
                code.shape[0], 1
            ),
        )
        if self.shuffle_orders:
            permutation = torch.randperm(code.shape[0])
            code = code[permutation]
            order = order[permutation]
            inverse = inverse[permutation]

        point_dict = Dict(
            feat=torch_scatter.segment_csr(
                self.proj(point.feat[indices]), idx_ptr, reduce=self.reduce
            ),
            coord=torch_scatter.segment_csr(
                point.coord[indices], idx_ptr, reduce="mean"
            ),
            grid_coord=point.grid_coord[head_indices] >> pooling_depth,
            serialized_code=code,
            serialized_order=order,
            serialized_inverse=inverse,
            serialized_depth=point.serialized_depth - pooling_depth,
            batch=point.batch[head_indices],
        )
        for key in ("condition", "context"):
            if key in point.keys():
                point_dict[key] = point[key]
        if self.traceable:
            point_dict["pooling_inverse"] = cluster
            point_dict["pooling_parent"] = point
        output = Point(point_dict)
        if hasattr(self, "norm"):
            output = self.norm(output)
        if hasattr(self, "act"):
            output = self.act(output)
        output.sparsify()
        return output


@MODELS.register_module("PT-v3m4")
class LightweightPointTransformerV3(ptv3m1.PointTransformerV3):
    """PTv3m1-compatible, VPSD-traceable, activation-light encoder."""

    def __init__(
        self,
        *args,
        checkpoint_cpe=True,
        checkpoint_attention=True,
        checkpoint_mlp=True,
        random_order=True,
        **kwargs,
    ):
        traceable = bool(kwargs.pop("traceable", True))
        mask_token = bool(kwargs.pop("mask_token", False))
        if not traceable:
            raise ValueError("PT-v3m4 requires traceable=True for VPSD")
        if mask_token:
            raise ValueError("Lightweight PT-v3m4 does not support mask_token")
        # PT-v3m4 intentionally has no PDNorm mode. Silently consume legacy
        # config fields so old single-dataset configs remain loadable.
        for key in (
            "pdnorm_bn",
            "pdnorm_ln",
            "pdnorm_decouple",
            "pdnorm_adaptive",
            "pdnorm_affine",
            "pdnorm_conditions",
        ):
            kwargs.pop(key, None)
        super().__init__(
            *args,
            pdnorm_bn=False,
            pdnorm_ln=False,
            **kwargs,
        )
        branches = [self.enc]
        if not self.enc_mode:
            branches.append(self.dec)
        for branch in branches:
            for stage in branch._modules.values():
                for name, module in list(stage._modules.items()):
                    if isinstance(module, ptv3m1.SerializedPooling):
                        stage._modules[name] = LightweightSerializedPooling(module)
                    elif isinstance(module, ptv3m1.Block):
                        stage._modules[name] = LightweightBlock(
                            module,
                            checkpoint_cpe=checkpoint_cpe,
                            checkpoint_attention=checkpoint_attention,
                            checkpoint_mlp=checkpoint_mlp,
                            random_order=random_order,
                        )

    def encode(self, data_dict):
        """Run the standard PTv3 stem/encoder and preserve its hierarchy."""

        point = Point(data_dict)
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()
        point = self.embedding(point)
        point = self.enc(point)
        hierarchy = build_encoder_hierarchy(
            point, expected_levels=self.num_stages
        )
        return point, hierarchy

    def decode_hierarchy(self, hierarchy, detach_stages=()):
        """Run the ordinary PTv3 decoder on a non-mutating hierarchy clone."""

        if self.enc_mode:
            raise RuntimeError("Encoder-only PT-v3m4 has no decoder")
        decoder_levels = clone_hierarchy_for_decoder(
            hierarchy, detach_stages=detach_stages
        )
        return self.dec(decoder_levels[-1])

    def forward(
        self,
        data_dict,
        return_hierarchy=False,
        decoder_detach_stages=(),
    ):
        point, hierarchy = self.encode(data_dict)
        if not self.enc_mode:
            point = self.decode_hierarchy(
                hierarchy, detach_stages=decoder_detach_stages
            )
        elif decoder_detach_stages:
            raise ValueError(
                "decoder_detach_stages is only valid when enc_mode=False"
            )
        if return_hierarchy:
            return point, hierarchy
        return point
