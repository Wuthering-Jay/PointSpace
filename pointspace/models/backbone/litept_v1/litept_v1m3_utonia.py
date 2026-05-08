"""
LitePT v2m3 Utonia

LitePT v1m1 backbone with self-supervised hooks aligned with PT-v3m3.
"""

import math
from functools import partial

import torch
import torch.nn as nn
import spconv.pytorch as spconv
from timm.layers import DropPath
from torch.nn.init import trunc_normal_

try:
    import flash_attn
except ImportError:
    flash_attn = None

from pointspace.models.builder import MODELS
from pointspace.models.modules import PointModule, PointSequential
from pointspace.models.utils.misc import offset2bincount
from pointspace.models.utils.structure import Point

from .litept_v1m1 import GridPooling, GridUnpooling, PointROPE
from .litept_v1m2_sonata import Embedding


class PointROPEAttention(PointModule):
    def __init__(
        self,
        channels,
        num_heads,
        patch_size,
        rope_freq,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        order_index=0,
        enable_flash=True,
        shift_coords=None,
        jitter_coords=None,
        rescale_coords=None,
    ):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.scale = qk_scale or (channels // num_heads) ** -0.5
        self.order_index = order_index
        self.enable_flash = enable_flash
        if enable_flash:
            assert flash_attn is not None, "Make sure flash_attn is installed."
            self.patch_size = patch_size
            self.attn_drop = attn_drop
        else:
            self.patch_size_max = patch_size
            self.patch_size = 0
            self.attn_drop = torch.nn.Dropout(attn_drop)
        self.enable_rope = rope_freq is not None and rope_freq is not False
        self.shift_coords = shift_coords
        self.jitter_coords = jitter_coords
        self.rescale_coords = rescale_coords

        self.qkv = torch.nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.proj = torch.nn.Linear(channels, channels)
        self.proj_drop = torch.nn.Dropout(proj_drop)
        self.softmax = torch.nn.Softmax(dim=-1)

        self.rope = PointROPE(freq=rope_freq) if self.enable_rope else None

    @torch.no_grad()
    def get_padding_and_inverse(self, point):
        pad_key = "pad"
        unpad_key = "unpad"
        cu_seqlens_key = "cu_seqlens_key"
        if (
            pad_key not in point.keys()
            or unpad_key not in point.keys()
            or cu_seqlens_key not in point.keys()
        ):
            offset = point.offset
            bincount = offset2bincount(offset)
            bincount_pad = (
                torch.div(
                    bincount + self.patch_size - 1,
                    self.patch_size,
                    rounding_mode="trunc",
                )
                * self.patch_size
            )
            mask_pad = bincount > self.patch_size
            bincount_pad = ~mask_pad * bincount + mask_pad * bincount_pad
            _offset = nn.functional.pad(offset, (1, 0))
            _offset_pad = nn.functional.pad(torch.cumsum(bincount_pad, dim=0), (1, 0))
            pad = torch.arange(_offset_pad[-1], device=offset.device)
            unpad = torch.arange(_offset[-1], device=offset.device)
            cu_seqlens = []
            for i in range(len(offset)):
                unpad[_offset[i] : _offset[i + 1]] += _offset_pad[i] - _offset[i]
                if bincount[i] != bincount_pad[i]:
                    pad[
                        _offset_pad[i + 1]
                        - self.patch_size
                        + (bincount[i] % self.patch_size) : _offset_pad[i + 1]
                    ] = pad[
                        _offset_pad[i + 1]
                        - 2 * self.patch_size
                        + (bincount[i] % self.patch_size) : _offset_pad[i + 1]
                        - self.patch_size
                    ]
                pad[_offset_pad[i] : _offset_pad[i + 1]] -= _offset_pad[i] - _offset[i]
                cu_seqlens.append(
                    torch.arange(
                        _offset_pad[i],
                        _offset_pad[i + 1],
                        step=self.patch_size,
                        dtype=torch.int32,
                        device=offset.device,
                    )
                )
            point[pad_key] = pad
            point[unpad_key] = unpad
            point[cu_seqlens_key] = nn.functional.pad(
                torch.concat(cu_seqlens), (0, 1), value=_offset_pad[-1]
            )
        return point[pad_key], point[unpad_key], point[cu_seqlens_key]

    def _build_rope_pos(self, point, order):
        pos = point.grid_coord[order].float()
        if self.training:
            dd = {"device": pos.device, "dtype": pos.dtype}
            if self.shift_coords is not None and self.shift_coords > 0:
                pos = pos + torch.empty(3, **dd).uniform_(
                    -self.shift_coords, self.shift_coords
                ).unsqueeze(0)
            if self.jitter_coords is not None and self.jitter_coords > 1.0:
                jitter_max = math.log(self.jitter_coords)
                jitter_xyz = torch.empty(3, **dd).uniform_(
                    -jitter_max, jitter_max
                ).exp()
                pos = pos * jitter_xyz.unsqueeze(0)
            if self.rescale_coords is not None and self.rescale_coords > 1.0:
                rescale_max = math.log(self.rescale_coords)
                rescale_val = torch.empty(1, **dd).uniform_(
                    -rescale_max, rescale_max
                ).exp()
                pos = pos * rescale_val

        k = self.patch_size
        if pos.shape[0] % k == 0:
            pos_patch = pos.reshape(-1, k, 3)
            pos_patch = pos_patch - pos_patch.amin(dim=1, keepdim=True)
            pos = pos_patch.reshape(-1, 3)
        return pos.round().to(torch.int64).reshape(1, -1, 3).contiguous()

    def forward(self, point):
        if not self.enable_flash:
            self.patch_size = min(
                offset2bincount(point.offset).min().tolist(), self.patch_size_max
            )

        h = self.num_heads
        c = self.channels
        k_patch = self.patch_size

        pad, unpad, cu_seqlens = self.get_padding_and_inverse(point)
        order = point.serialized_order[self.order_index][pad]
        inverse = unpad[point.serialized_inverse[self.order_index]]

        qkv = self.qkv(point.feat)[order]
        qkv_dtype = qkv.dtype
        q, k, v = qkv.half().chunk(3, dim=-1)
        q = q.reshape(-1, h, c // h).transpose(0, 1)[None]
        k = k.reshape(-1, h, c // h).transpose(0, 1)[None]

        if self.enable_rope:
            pos = self._build_rope_pos(point, order)
            q = self.rope(q.float(), pos).to(q.dtype)
            k = self.rope(k.float(), pos).to(k.dtype)

        if not self.enable_flash:
            q = q.squeeze(0).transpose(0, 1).reshape(-1, k_patch, h, c // h).permute(
                0, 2, 1, 3
            )
            k = k.squeeze(0).transpose(0, 1).reshape(-1, k_patch, h, c // h).permute(
                0, 2, 1, 3
            )
            v = v.reshape(-1, k_patch, h, c // h).permute(0, 2, 1, 3)
            attn = (q.float() * self.scale) @ k.float().transpose(-2, -1)
            attn = self.softmax(attn.float())
            attn = self.attn_drop(attn).to(v.dtype)
            feat = (attn @ v).transpose(1, 2).reshape(-1, c)
        else:
            qkv_rotated = torch.stack(
                [
                    q.squeeze(0).transpose(0, 1),
                    k.squeeze(0).transpose(0, 1),
                    v.reshape(-1, h, c // h),
                ],
                dim=1,
            )

            feat = flash_attn.flash_attn_varlen_qkvpacked_func(
                qkv_rotated,
                cu_seqlens,
                max_seqlen=self.patch_size,
                dropout_p=self.attn_drop if self.training else 0,
                softmax_scale=self.scale,
            ).reshape(-1, c)

        feat = feat.to(qkv_dtype)
        feat = feat[inverse]
        feat = self.proj(feat)
        feat = self.proj_drop(feat)
        point.feat = feat
        return point


class MLP(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels=None,
        out_channels=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(PointModule):
    def __init__(
        self,
        channels,
        num_heads,
        patch_size=48,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        pre_norm=True,
        order_index=0,
        cpe_indice_key=None,
        enable_conv=True,
        enable_attn=True,
        rope_freq=100.0,
        enable_flash=True,
        shift_coords=None,
        jitter_coords=None,
        rescale_coords=None,
    ):
        super().__init__()
        self.channels = channels
        self.pre_norm = pre_norm
        self.enable_conv = enable_conv
        self.enable_attn = enable_attn

        if self.enable_conv:
            self.conv = PointSequential(
                spconv.SubMConv3d(
                    channels,
                    channels,
                    kernel_size=3,
                    bias=True,
                    indice_key=cpe_indice_key,
                ),
                nn.Linear(channels, channels),
                norm_layer(channels),
            )
        else:
            self.norm0 = PointSequential(norm_layer(channels))

        if self.enable_attn:
            self.norm1 = PointSequential(norm_layer(channels))
            self.attn = PointROPEAttention(
                channels=channels,
                patch_size=patch_size,
                rope_freq=rope_freq,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                order_index=order_index,
                enable_flash=enable_flash,
                shift_coords=shift_coords,
                jitter_coords=jitter_coords,
                rescale_coords=rescale_coords,
            )
            self.norm2 = PointSequential(norm_layer(channels))
            self.mlp = PointSequential(
                MLP(
                    in_channels=channels,
                    hidden_channels=int(channels * mlp_ratio),
                    out_channels=channels,
                    act_layer=act_layer,
                    drop=proj_drop,
                )
            )
            self.drop_path = PointSequential(
                DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
            )

    def forward(self, point: Point):
        if self.enable_conv:
            shortcut = point.feat
            point = self.conv(point)
            point.feat = shortcut + point.feat
        else:
            point = self.norm0(point)

        if self.enable_attn:
            shortcut = point.feat
            if self.pre_norm:
                point = self.norm1(point)
            point = self.drop_path(self.attn(point))
            point.feat = shortcut + point.feat
            if not self.pre_norm:
                point = self.norm1(point)

            shortcut = point.feat
            if self.pre_norm:
                point = self.norm2(point)
            point = self.drop_path(self.mlp(point))
            point.feat = shortcut + point.feat
            if not self.pre_norm:
                point = self.norm2(point)

        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point


@MODELS.register_module("LitePT-v1m3")
class LitePT(PointModule):
    def __init__(
        self,
        in_channels=4,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(36, 72, 144, 252, 504),
        enc_num_head=(2, 4, 8, 14, 28),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        enc_conv=(True, True, True, False, False),
        enc_attn=(False, False, False, True, True),
        enc_rope_freq=(100.0, 100.0, 100.0, 100.0, 100.0),
        dec_depths=(0, 0, 0, 0),
        dec_channels=(72, 72, 144, 252),
        dec_num_head=(4, 4, 8, 14),
        dec_patch_size=(1024, 1024, 1024, 1024),
        dec_conv=(False, False, False, False),
        dec_attn=(False, False, False, False),
        dec_rope_freq=(100.0, 100.0, 100.0, 100.0),
        dec_rope_enable=True,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        shuffle_orders=True,
        enable_flash=True,
        traceable=True,
        mask_token=False,
        enc_mode=False,
        freeze_encoder=False,
        shift_coords=None,
        jitter_coords=None,
        rescale_coords=None,
    ):
        super().__init__()
        self.num_stages = len(enc_depths)
        self.order = [order] if isinstance(order, str) else order
        self.enc_mode = enc_mode
        self.shuffle_orders = shuffle_orders
        self.freeze_encoder = freeze_encoder

        self.enc_conv = enc_conv
        self.enc_attn = enc_attn
        self.dec_conv = dec_conv
        self.dec_attn = dec_attn

        assert self.num_stages == len(stride) + 1
        assert self.num_stages == len(enc_depths)
        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(enc_num_head)
        assert self.num_stages == len(enc_patch_size)
        assert self.enc_mode or self.num_stages == len(dec_depths) + 1
        assert self.enc_mode or self.num_stages == len(dec_channels) + 1
        assert self.enc_mode or self.num_stages == len(dec_num_head) + 1
        assert self.enc_mode or self.num_stages == len(dec_patch_size) + 1

        bn_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        ln_layer = nn.LayerNorm
        act_layer = nn.GELU

        self.embedding = Embedding(
            in_channels=in_channels,
            embed_channels=enc_channels[0],
            norm_layer=bn_layer,
            act_layer=act_layer,
            mask_token=mask_token,
        )

        enc_drop_path = [
            x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))
        ]
        self.enc = PointSequential()
        for s in range(self.num_stages):
            enc_drop_path_ = enc_drop_path[
                sum(enc_depths[:s]) : sum(enc_depths[: s + 1])
            ]
            enc = PointSequential()
            if s > 0:
                enc.add(
                    GridPooling(
                        in_channels=enc_channels[s - 1],
                        out_channels=enc_channels[s],
                        stride=stride[s - 1],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                        re_serialization=enc_attn[s],
                        serialization_order=self.order,
                    ),
                    name="down",
                )
            for i in range(enc_depths[s]):
                enc.add(
                    Block(
                        channels=enc_channels[s],
                        num_heads=enc_num_head[s],
                        patch_size=enc_patch_size[s],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=enc_drop_path_[i],
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        pre_norm=pre_norm,
                        order_index=i % len(self.order),
                        cpe_indice_key=f"stage{s}",
                        enable_conv=enc_conv[s],
                        enable_attn=enc_attn[s],
                        rope_freq=enc_rope_freq[s],
                        enable_flash=enable_flash,
                        shift_coords=shift_coords,
                        jitter_coords=jitter_coords,
                        rescale_coords=rescale_coords,
                    ),
                    name=f"block{i}",
                )
            if len(enc) != 0:
                self.enc.add(module=enc, name=f"enc{s}")

        if not self.enc_mode:
            dec_drop_path = [
                x.item() for x in torch.linspace(0, drop_path, sum(dec_depths))
            ]
            self.dec = PointSequential()
            dec_channels = list(dec_channels) + [enc_channels[-1]]
            for s in reversed(range(self.num_stages - 1)):
                dec_drop_path_ = dec_drop_path[
                    sum(dec_depths[:s]) : sum(dec_depths[: s + 1])
                ]
                dec_drop_path_.reverse()
                dec = PointSequential()
                dec.add(
                    GridUnpooling(
                        in_channels=dec_channels[s + 1],
                        skip_channels=enc_channels[s],
                        out_channels=dec_channels[s],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                        traceable=traceable,
                    ),
                    name="up",
                )
                for i in range(dec_depths[s]):
                    dec.add(
                        Block(
                            channels=dec_channels[s],
                            num_heads=dec_num_head[s],
                            patch_size=dec_patch_size[s],
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            attn_drop=attn_drop,
                            proj_drop=proj_drop,
                            drop_path=dec_drop_path_[i],
                            norm_layer=ln_layer,
                            act_layer=act_layer,
                            pre_norm=pre_norm,
                            order_index=i % len(self.order),
                            cpe_indice_key=f"stage{s}",
                            enable_conv=dec_conv[s],
                            enable_attn=dec_attn[s],
                            rope_freq=dec_rope_freq[s] if dec_rope_enable else None,
                            enable_flash=enable_flash,
                            shift_coords=shift_coords,
                            jitter_coords=jitter_coords,
                            rescale_coords=rescale_coords,
                        ),
                        name=f"block{i}",
                    )
                self.dec.add(module=dec, name=f"dec{s}")

        if self.freeze_encoder:
            for p in self.embedding.parameters():
                p.requires_grad = False
            for p in self.enc.parameters():
                p.requires_grad = False

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, spconv.SubMConv3d):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, data_dict):
        point = Point(data_dict)
        if self.enc_attn[0]:
            point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()

        point = self.embedding(point)
        point = self.enc(point)
        if not self.enc_mode:
            point = self.dec(point)
        return point
