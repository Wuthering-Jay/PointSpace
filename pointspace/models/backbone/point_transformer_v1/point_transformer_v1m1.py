"""
Point Transformer V1 Backbone

Backbone only:
  - input: data_dict or Point
  - output: Point, feat is the final decoder feature
"""

from copy import deepcopy

import einops
import pointops
import torch
import torch.nn as nn
from timm.layers import DropPath
from torch.utils.checkpoint import checkpoint

from pointspace.models.builder import MODELS
from pointspace.models.modules import PointModule
from pointspace.models.utils import offset2bincount
from pointspace.models.utils.structure import Point
from pointspace.models.point_transformer.utils import LayerNorm1d


class PointBatchNorm(nn.Module):
    """BatchNorm1d that accepts [N, C] or [N, L, C]."""

    def __init__(self, embed_channels):
        super().__init__()
        self.norm = nn.BatchNorm1d(embed_channels)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.dim() == 3:
            return (
                self.norm(input.transpose(1, 2).contiguous())
                .transpose(1, 2)
                .contiguous()
            )
        if input.dim() == 2:
            return self.norm(input)
        raise NotImplementedError


class PointTransformerLayer(nn.Module):
    def __init__(self, in_planes, out_planes, share_planes=8, nsample=16, attn_drop=0.0):
        super().__init__()
        self.mid_planes = out_planes
        self.out_planes = out_planes
        self.share_planes = share_planes
        self.nsample = nsample
        assert out_planes % share_planes == 0

        self.linear_q = nn.Linear(in_planes, self.mid_planes)
        self.linear_k = nn.Linear(in_planes, self.mid_planes)
        self.linear_v = nn.Linear(in_planes, out_planes)
        self.linear_p = nn.Sequential(
            nn.Linear(3, 3),
            LayerNorm1d(3),
            nn.ReLU(inplace=True),
            nn.Linear(3, out_planes),
        )
        self.linear_w = nn.Sequential(
            LayerNorm1d(self.mid_planes),
            nn.ReLU(inplace=True),
            nn.Linear(self.mid_planes, out_planes // share_planes),
            LayerNorm1d(out_planes // share_planes),
            nn.ReLU(inplace=True),
            nn.Linear(out_planes // share_planes, out_planes // share_planes),
        )
        self.softmax = nn.Softmax(dim=1)
        self.attn_drop = nn.Dropout(attn_drop) if attn_drop > 0 else nn.Identity()

    def forward(self, p, x, o) -> torch.Tensor:
        x_q, x_k, x_v = self.linear_q(x), self.linear_k(x), self.linear_v(x)
        x_k, idx = pointops.knn_query_and_group(
            x_k,
            p,
            o,
            new_xyz=p,
            new_offset=o,
            nsample=self.nsample,
            with_xyz=True,
        )
        x_v, _ = pointops.knn_query_and_group(
            x_v,
            p,
            o,
            new_xyz=p,
            new_offset=o,
            idx=idx,
            nsample=self.nsample,
            with_xyz=False,
        )
        p_r, x_k = x_k[:, :, 0:3], x_k[:, :, 3:]
        p_r = self.linear_p(p_r)
        r_qk = x_k - x_q.unsqueeze(1) + einops.reduce(
            p_r, "n ns (i j) -> n ns j", reduction="sum", j=self.mid_planes
        )
        w = self.linear_w(r_qk)
        w = self.attn_drop(self.softmax(w))
        x = torch.einsum(
            "n t s i, n t i -> n s i",
            einops.rearrange(x_v + p_r, "n ns (s i) -> n ns s i", s=self.share_planes),
            w,
        )
        x = einops.rearrange(x, "n s i -> n (s i)")
        return x


class TransitionDown(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, nsample=16):
        super().__init__()
        self.stride, self.nsample = stride, nsample
        if stride != 1:
            self.linear = nn.Linear(3 + in_planes, out_planes, bias=False)
            self.pool = nn.MaxPool1d(nsample)
        else:
            self.linear = nn.Linear(in_planes, out_planes, bias=False)
        self.bn = nn.BatchNorm1d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo):
        p, x, o = pxo
        if self.stride != 1:
            counts = offset2bincount(o)
            new_counts = []
            total = 0
            for i in range(counts.shape[0]):
                total += counts[i].item() // self.stride
                new_counts.append(total)
            n_o = torch.tensor(new_counts, device=o.device, dtype=torch.int32)
            idx = pointops.farthest_point_sampling(p, o, n_o)
            n_p = p[idx.long(), :]
            x, _ = pointops.knn_query_and_group(
                x,
                p,
                offset=o,
                new_xyz=n_p,
                new_offset=n_o,
                nsample=self.nsample,
                with_xyz=True,
            )
            x = self.relu(self.bn(self.linear(x).transpose(1, 2).contiguous()))
            x = self.pool(x).squeeze(-1)
            p, o = n_p, n_o
        else:
            x = self.relu(self.bn(self.linear(x)))
        return [p, x, o]


class TransitionUp(nn.Module):
    def __init__(self, in_planes, out_planes=None):
        super().__init__()
        if out_planes is None:
            self.linear1 = nn.Sequential(
                nn.Linear(2 * in_planes, in_planes),
                nn.BatchNorm1d(in_planes),
                nn.ReLU(inplace=True),
            )
            self.linear2 = nn.Sequential(
                nn.Linear(in_planes, in_planes),
                nn.ReLU(inplace=True),
            )
        else:
            self.linear1 = nn.Sequential(
                nn.Linear(out_planes, out_planes),
                nn.BatchNorm1d(out_planes),
                nn.ReLU(inplace=True),
            )
            self.linear2 = nn.Sequential(
                nn.Linear(in_planes, out_planes),
                nn.BatchNorm1d(out_planes),
                nn.ReLU(inplace=True),
            )

    def forward(self, pxo1, pxo2=None):
        if pxo2 is None:
            _, x, o = pxo1
            x_tmp = []
            for i in range(o.shape[0]):
                if i == 0:
                    s_i, e_i, cnt = 0, int(o[0].item()), int(o[0].item())
                else:
                    s_i, e_i, cnt = (
                        int(o[i - 1].item()),
                        int(o[i].item()),
                        int((o[i] - o[i - 1]).item()),
                    )
                x_b = x[s_i:e_i, :]
                x_b = torch.cat(
                    (x_b, self.linear2(x_b.sum(0, True) / cnt).repeat(cnt, 1)), 1
                )
                x_tmp.append(x_b)
            x = torch.cat(x_tmp, 0)
            x = self.linear1(x)
        else:
            p1, x1, o1 = pxo1
            p2, x2, o2 = pxo2
            x = self.linear1(x1) + pointops.interpolation(
                p2, p1, self.linear2(x2), o2, o1
            )
        return x


class Bottleneck(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_planes,
        planes,
        share_planes=8,
        nsample=16,
        attn_drop=0.0,
        drop_path=0.0,
        enable_checkpoint=False,
    ):
        super().__init__()
        self.linear1 = nn.Linear(in_planes, planes, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.transformer = PointTransformerLayer(
            planes, planes, share_planes, nsample, attn_drop=attn_drop
        )
        self.bn2 = nn.BatchNorm1d(planes)
        self.linear3 = nn.Linear(planes, planes * self.expansion, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.enable_checkpoint = enable_checkpoint
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, pxo):
        p, x, o = pxo
        identity = x
        x = self.relu(self.bn1(self.linear1(x)))
        x = (
            self.transformer(p, x, o)
            if not self.enable_checkpoint
            else checkpoint(self.transformer, p, x, o)
        )
        x = self.relu(self.bn2(x))
        x = self.bn3(self.linear3(x))
        x = identity + self.drop_path(x)
        x = self.relu(x)
        return [p, x, o]


class BlockSequence(nn.Module):
    def __init__(
        self,
        depth,
        embed_channels,
        share_planes=8,
        nsample=16,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        enable_checkpoint=False,
    ):
        super().__init__()
        if isinstance(drop_path_rate, list):
            drop_path_rates = drop_path_rate
            assert len(drop_path_rates) == depth
        elif isinstance(drop_path_rate, float):
            drop_path_rates = [deepcopy(drop_path_rate) for _ in range(depth)]
        else:
            drop_path_rates = [0.0 for _ in range(depth)]

        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(
                Bottleneck(
                    embed_channels,
                    embed_channels,
                    share_planes=share_planes,
                    nsample=nsample,
                    attn_drop=attn_drop_rate,
                    drop_path=drop_path_rates[i],
                    enable_checkpoint=enable_checkpoint,
                )
            )

    def forward(self, points):
        for block in self.blocks:
            points = block(points)
        return points


class Encoder(nn.Module):
    def __init__(
        self,
        depth,
        in_channels,
        embed_channels,
        share_planes=8,
        stride=1,
        nsample=16,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        enable_checkpoint=False,
    ):
        super().__init__()
        self.down = TransitionDown(
            in_planes=in_channels,
            out_planes=embed_channels,
            stride=stride,
            nsample=nsample,
        )
        self.blocks = BlockSequence(
            depth=depth,
            embed_channels=embed_channels,
            share_planes=share_planes,
            nsample=nsample,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            enable_checkpoint=enable_checkpoint,
        )

    def forward(self, points):
        points = self.down(points)
        return self.blocks(points)


class Decoder(nn.Module):
    def __init__(
        self,
        depth,
        in_channels,
        embed_channels,
        share_planes=8,
        nsample=16,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        enable_checkpoint=False,
        is_head=False,
    ):
        super().__init__()
        self.is_head = is_head
        self.up = TransitionUp(in_channels, None if is_head else embed_channels)
        self.blocks = BlockSequence(
            depth=depth,
            embed_channels=embed_channels,
            share_planes=share_planes,
            nsample=nsample,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            enable_checkpoint=enable_checkpoint,
        )

    def forward(self, points, skip_points=None):
        if self.is_head:
            p, _, o = points
            x = self.up(points)
            points = [p, x, o]
        else:
            p, _, o = skip_points
            x = self.up(skip_points, points)
            points = [p, x, o]
        return self.blocks(points)


@MODELS.register_module(name="PT-v1m2")
class PointTransformerV1(PointModule):
    """
    Point Transformer V1 backbone.

    input: dict or Point with keys {"coord", "feat", "offset"}
    output: Point, feat is the final decoder feature
    """

    def __init__(
        self,
        in_channels,
        enc_depths=(1, 1, 1, 1, 1),
        enc_channels=(32, 64, 128, 256, 512),
        enc_strides=(1, 4, 4, 4, 4),
        enc_nsamples=(8, 16, 16, 16, 16),
        dec_depths=(1, 1, 1, 1, 1),
        dec_channels=(32, 64, 128, 256, 512),
        dec_nsamples=(8, 16, 16, 16, 16),
        share_planes=8,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        enable_checkpoint=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_stages = len(enc_depths)
        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(enc_strides)
        assert self.num_stages == len(enc_nsamples)
        assert self.num_stages == len(dec_depths)
        assert self.num_stages == len(dec_channels)
        assert self.num_stages == len(dec_nsamples)

        enc_dp_rates = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(enc_depths))
        ]
        dec_dp_rates = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(dec_depths))
        ]

        self.enc_stages = nn.ModuleList()
        self.dec_stages = nn.ModuleList()
        for i in range(self.num_stages):
            self.enc_stages.append(
                Encoder(
                    depth=enc_depths[i],
                    in_channels=in_channels if i == 0 else enc_channels[i - 1],
                    embed_channels=enc_channels[i],
                    share_planes=share_planes,
                    stride=enc_strides[i],
                    nsample=enc_nsamples[i],
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=enc_dp_rates[
                        sum(enc_depths[:i]) : sum(enc_depths[: i + 1])
                    ],
                    enable_checkpoint=enable_checkpoint,
                )
            )

        for i in range(self.num_stages - 1, -1, -1):
            if i == self.num_stages - 1:
                self.dec_stages.append(
                    Decoder(
                        depth=dec_depths[i],
                        in_channels=enc_channels[-1],
                        embed_channels=dec_channels[i],
                        share_planes=share_planes,
                        nsample=dec_nsamples[i],
                        attn_drop_rate=attn_drop_rate,
                        drop_path_rate=dec_dp_rates[
                            sum(dec_depths[:i]) : sum(dec_depths[: i + 1])
                        ],
                        enable_checkpoint=enable_checkpoint,
                        is_head=True,
                    )
                )
            else:
                self.dec_stages.append(
                    Decoder(
                        depth=dec_depths[i],
                        in_channels=dec_channels[i + 1],
                        embed_channels=dec_channels[i],
                        share_planes=share_planes,
                        nsample=dec_nsamples[i],
                        attn_drop_rate=attn_drop_rate,
                        drop_path_rate=dec_dp_rates[
                            sum(dec_depths[:i]) : sum(dec_depths[: i + 1])
                        ],
                        enable_checkpoint=enable_checkpoint,
                        is_head=False,
                    )
                )


    def forward(self, data_dict):
        if not isinstance(data_dict, Point):
            point = Point(data_dict)
        else:
            point = data_dict

        coord = point.coord
        feat = point.feat
        offset = point.offset.int()
        points = [coord, feat, offset]

        skips = [[points]]
        for i in range(self.num_stages):
            points = self.enc_stages[i](points)
            skips[-1].append(None)
            skips.append([points])

        points = skips.pop(-1)[0]
        for i, dec_stage in enumerate(self.dec_stages):
            if dec_stage.is_head:
                points = dec_stage(points)
            else:
                skip_points = skips.pop(-1)[0]
                points = dec_stage(points, skip_points)

        coord, feat, offset = points
        point.feat = feat
        return point
