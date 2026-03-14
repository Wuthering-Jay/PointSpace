"""
Point Transformer V2 Mode 4

Backbone only (符合 DefaultSegmentorV2 规范):
  - 输入: data_dict 或 Point 对象
  - 输出: Point 对象 (feat 为解码器最终输出特征)
"""

from copy import deepcopy
import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn.pool import voxel_grid
from torch_scatter import segment_csr

import einops
from timm.layers import DropPath
import pointops

from pointspace.models.builder import MODELS
from pointspace.models.utils import offset2batch, batch2offset
from pointspace.models.utils.structure import Point
from pointspace.models.modules import PointModule


class PointBatchNorm(nn.Module):
    """
    Batch Normalization for Point Clouds data in shape of [B*N, C], [B*N, L, C]
    对形状为[n, c], [n, l, c]的点云数据进行批量归一化
    """

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
        elif input.dim() == 2:
            return self.norm(input)
        else:
            raise NotImplementedError
        
        
class PointNetLayer(nn.Module):
    def __init__(self, in_channels, out_channels, k_neighbors=16):
        """
        使用 Linear 层和 PointBatchNorm 的 PointNet 层实现
        
        参数:
            in_channels: 输入特征维度 c1
            out_channels: 输出特征维度 c2
            k_neighbors: KNN近邻数
        """
        super().__init__()
        self.k_neighbors = k_neighbors
        
        # 计算中间层维度
        mid_channels = max(in_channels, out_channels // 2)
        
        # 输入特征维度调整（如果包含xyz坐标）
        mlp_in_channels = in_channels
        
        # 定义共享MLP网络（使用Linear层）
        self.shared_mlp = nn.Sequential(
            nn.Linear(mlp_in_channels, mid_channels),
            PointBatchNorm(mid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, out_channels),
            PointBatchNorm(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, points):
        coord, feat, offset = points
        
        # KNN 查询不需要梯度
        with torch.no_grad():
            reference_index, _ = pointops.knn_query(self.k_neighbors, coord, offset)
        
        grouped_features = pointops.grouping(reference_index, feat, coord, with_xyz=False)
        n_points = grouped_features.shape[0]
        grouped_features = grouped_features.reshape(-1, grouped_features.shape[-1])
        out = self.shared_mlp(grouped_features)
        out = out.reshape(n_points, self.k_neighbors, -1)
        out = out.max(dim=1)[0]
        
        return out


class GroupedVectorAttention(nn.Module):
    """
    分组向量注意力机制
    Args:
        embed_channesl: 输入输出维度
        groups: 分组数量
        attn_drop_rate: drop比例
        qkv_bias: 无用
        pe_multiplier: 位置编码乘性因子
        pe_bias: 位置编码偏置因子
    """
    def __init__(
        self,
        embed_channels,
        groups,
        attn_drop_rate=0.0,
        qkv_bias=True,
        pe_multiplier=False,
        pe_bias=True,
    ):
        super(GroupedVectorAttention, self).__init__()
        self.embed_channels = embed_channels
        self.groups = groups
        assert embed_channels % groups == 0
        self.attn_drop_rate = attn_drop_rate
        self.qkv_bias = qkv_bias
        self.pe_multiplier = pe_multiplier
        self.pe_bias = pe_bias

        self.linear_q = nn.Sequential(
            nn.Linear(embed_channels, embed_channels, bias=qkv_bias),
            PointBatchNorm(embed_channels),
            nn.ReLU(inplace=True),
        )
        self.linear_k = nn.Sequential(
            nn.Linear(embed_channels, embed_channels, bias=qkv_bias),
            PointBatchNorm(embed_channels),
            nn.ReLU(inplace=True),
        )

        self.linear_v = nn.Linear(embed_channels, embed_channels, bias=qkv_bias)

        if self.pe_multiplier:
            self.linear_p_multiplier = nn.Sequential(
                nn.Linear(3, embed_channels),
                PointBatchNorm(embed_channels),
                nn.ReLU(inplace=True),
                nn.Linear(embed_channels, embed_channels),
            )
        if self.pe_bias:
            self.linear_p_bias = nn.Sequential(
                nn.Linear(3, embed_channels),
                PointBatchNorm(embed_channels),
                nn.ReLU(inplace=True),
                nn.Linear(embed_channels, embed_channels),
            )
        self.weight_encoding = nn.Sequential(
            nn.Linear(embed_channels, groups),
            PointBatchNorm(groups),
            nn.ReLU(inplace=True),
            nn.Linear(groups, groups),
        )
        self.softmax = nn.Softmax(dim=1)
        self.attn_drop = nn.Dropout(attn_drop_rate)

    def forward(self, feat, coord, reference_index):
        """
        input: feat: [n, c], coord: [n, 3], reference_index: [n, k]
        output: feat: [n, c]
        """
        query, key, value = (
            self.linear_q(feat),
            self.linear_k(feat),
            self.linear_v(feat),
        )
        
        # grouping 操作使用 detach 后的 reference_index
        key = pointops.grouping(reference_index.detach(), key, coord, with_xyz=True)
        value = pointops.grouping(reference_index.detach(), value, coord, with_xyz=False)
        pos, key = key[:, :, 0:3], key[:, :, 3:] # [n, k, 3], [n, k, c]
        relation_qk = key - query.unsqueeze(1) # [n, k ,c], 邻域内与中心点的相对位置, 用于相对位置编码
        if self.pe_multiplier: # 乘性因子
            pem = self.linear_p_multiplier(pos) # [n, k, c]
            relation_qk = relation_qk * pem # [n, k, c]
        if self.pe_bias: # 偏置因子
            peb = self.linear_p_bias(pos) # [n, k, c]
            relation_qk = relation_qk + peb # [n, k, c]
            value = value + peb # [n, k, c]

        weight = self.weight_encoding(relation_qk) # [n, k, g]
        weight = self.attn_drop(self.softmax(weight)) # [n, k, g]

        # mask 操作不需要梯度
        with torch.no_grad():
            mask = torch.sign(reference_index + 1) # [n, k], 无效邻域点标记为0
        
        weight = torch.einsum("n s g, n s -> n s g", weight, mask) # [n, k, g]
        value = einops.rearrange(value, "n ns (g i) -> n ns g i", g=self.groups) # [n, k, g, i]
        feat = torch.einsum("n s g i, n s g -> n g i", value, weight) # [n, g, i]
        feat = einops.rearrange(feat, "n g i -> n (g i)") # [n, c]
        return feat # [n, c]


class Block(nn.Module):
    """
    网络模块单位，结合GVA和BottleNeck对pxo进行处理，不改变数据维度
    Args:
        embed_channesl: 输入输出维度
        groups: 分组数量
        qkv_bias: 无用
        pe_multiplier: 位置编码乘性因子
        pe_bias: 位置编码偏置因子
        attn_drop_rate: drop比例
        drop_path_rate: BottleNeck的drop比例
        enable_checkpoint: checkpoint机制，以时间换空间
    """
    def __init__(
        self,
        embed_channels,
        groups,
        qkv_bias=True,
        pe_multiplier=False,
        pe_bias=True,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        enable_checkpoint=False,
    ):
        super(Block, self).__init__()
        self.attn = GroupedVectorAttention(
            embed_channels=embed_channels,
            groups=groups,
            qkv_bias=qkv_bias,
            attn_drop_rate=attn_drop_rate,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
        )
        self.fc1 = nn.Linear(embed_channels, embed_channels, bias=False)
        self.fc3 = nn.Linear(embed_channels, embed_channels, bias=False)
        self.norm1 = PointBatchNorm(embed_channels)
        self.norm2 = PointBatchNorm(embed_channels)
        self.norm3 = PointBatchNorm(embed_channels)
        self.act = nn.ReLU(inplace=True)
        self.enable_checkpoint = enable_checkpoint
        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )

    def forward(self, points, reference_index):
        """
        input: points: [pxo], [[n,3],[n,c],[b]], reference_index: [n, k]
        output: [pxo], [[n,3],[n,c],[b]], 不改变维度
        """
        coord, feat, offset = points # [n,3], [n,c], [b]
        identity = feat # [n, c]
        feat = self.act(self.norm1(self.fc1(feat))) # [n, c]
        feat = (
            self.attn(feat, coord, reference_index)
            if not self.enable_checkpoint # checkpoint机制，时间换空间，梯度等部分参数不保留，在反向传播时重新计算
            else checkpoint(self.attn, feat, coord, reference_index)
        ) # [n, c]
        feat = self.act(self.norm2(feat)) # [n, c]
        feat = self.norm3(self.fc3(feat)) # [n, c]
        feat = identity + self.drop_path(feat) # [n, c], bottleneck设计
        feat = self.act(feat) # [n, c]
        return [coord, feat, offset] # [[n,3],[n,c],[b]]


class BlockSequence(nn.Module):
    def __init__(
        self,
        depth,
        embed_channels,
        groups,
        neighbours=16,
        qkv_bias=True,
        pe_multiplier=False,
        pe_bias=True,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        enable_checkpoint=False,
    ):
        super(BlockSequence, self).__init__()

        # 确保 drop_path_rates 为 list
        if isinstance(drop_path_rate, list):
            drop_path_rates = drop_path_rate
            assert len(drop_path_rates) == depth
        elif isinstance(drop_path_rate, float):
            drop_path_rates = [deepcopy(drop_path_rate) for _ in range(depth)]
        else:
            drop_path_rates = [0.0 for _ in range(depth)]

        self.neighbours = neighbours
        self.blocks = nn.ModuleList()
        # 多个 Block 堆叠
        for i in range(depth):
            block = Block(
                embed_channels=embed_channels,
                groups=groups,
                qkv_bias=qkv_bias,
                pe_multiplier=pe_multiplier,
                pe_bias=pe_bias,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rates[i],
                enable_checkpoint=enable_checkpoint,
            )
            self.blocks.append(block)

    def forward(self, points):
        coord, feat, offset = points 
        with torch.no_grad():
            reference_index, _ = pointops.knn_query(self.neighbours, coord, offset)
        
        for block in self.blocks:
            points = block(points, reference_index)
        return points


class GridPool(nn.Module):
    """
    Partition-based Pooling (Grid Pooling)
    格网池化，基于体素划分进行池化下采样，体素内坐标平均池化，特征最大池化，得到新的pxo，同时输出体素索引
    Args:
        in_channels: 输入维度
        out_channels: 输出维度
        grid_size: 体素大小
        bias: fc层偏置
    """

    def __init__(self, in_channels, out_channels, grid_size, bias=False):
        super(GridPool, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grid_size = grid_size

        self.fc = nn.Linear(in_channels, out_channels, bias=bias)
        self.norm = PointBatchNorm(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, points, start=None):
        """
        input: points: [pxo], [[n,3],[n,c],[b]], start: [b, 3]
        output: points: [pxo], [[v,3],[v,c],[b]], cluster: [n]
        """
        coord, feat, offset = points # [n, 3] [n, c] [b]
        batch = offset2batch(offset) # [b] -> [n]
        feat = self.act(self.norm(self.fc(feat))) # [n, c]
        
        # 这些操作不需要梯度
        with torch.no_grad():
            start = (
                segment_csr(
                    coord,
                    torch.cat([batch.new_zeros(1), torch.cumsum(batch.bincount(), dim=0)]),
                    reduce="min",
                )
                if start is None
                else start
            )
            cluster = voxel_grid(
                pos=coord - start[batch], size=self.grid_size, batch=batch, start=0
            )
            unique, cluster, counts = torch.unique(
                cluster, sorted=True, return_inverse=True, return_counts=True
            )
            _, sorted_cluster_indices = torch.sort(cluster)
            idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        
        # 使用 detach 后的索引
        coord = segment_csr(coord[sorted_cluster_indices], idx_ptr, reduce="mean")
        feat = segment_csr(feat[sorted_cluster_indices], idx_ptr, reduce="max")
        batch = batch[idx_ptr[:-1]]
        offset = batch2offset(batch)
        return [coord, feat, offset], cluster.detach()  # cluster 不需要梯度


class UnpoolWithSkip(nn.Module):
    """
    Map Unpooling with skip connection
    带有跳跃连接的上采样
    Args:
        in_channels: 输入维度
        out_channels: 输出维度
        skip_channels: 跳跃连接维度
        bias: fc层偏置
        skip: 是否使用跳跃连接
        backend: 上采样方式，'map' or 'interp'
    """

    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        bias=True,
        skip=True,
        backend="map",
    ):
        super(UnpoolWithSkip, self).__init__()
        self.in_channels = in_channels
        self.skip_channels = skip_channels
        self.out_channels = out_channels
        self.skip = skip
        self.backend = backend
        assert self.backend in ["map", "interp"]

        self.proj = nn.Sequential(
            nn.Linear(in_channels, out_channels, bias=bias),
            PointBatchNorm(out_channels),
            nn.ReLU(inplace=True),
        )
        self.proj_skip = nn.Sequential(
            nn.Linear(skip_channels, out_channels, bias=bias),
            PointBatchNorm(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, points, skip_points, cluster=None):
        """
        input: points: [pxo], [[n,3],[n,c],[b]], skip_points: [pxo], [[ns,3],[ns,c],[b]], cluster: [ns]
        output: points: [pxo], [[ns,3],[ns,c],[b]]
        """
        coord, feat, offset = points # [n, 3] [n, c] [b]
        skip_coord, skip_feat, skip_offset = skip_points # [ns, 3] [ns, c] [b]
        if self.backend == "map" and cluster is not None:
            feat = self.proj(feat)[cluster] # [n, c] -> [ns, c], 投影上采样
        else:
            feat = pointops.interpolation(
                coord, skip_coord, self.proj(feat), offset, skip_offset
            ) # [n, c] -> [ns, c], 插值上采样
        if self.skip: # 跳跃连接，特征融合
            feat = feat + self.proj_skip(skip_feat) # [ns, c]
        return [skip_coord, feat, skip_offset] # [ns, 3] [ns, c] [b]


class Encoder(nn.Module):
    """
    Encoder for Point Transformer V2, 先进行格网池化, 再进行BlockSequence处理
    Args:
        depth: 编码器深度
        in_channels: 输入维度
        embed_channels: 输出维度
        groups: 分组数量
        grid_size: 体素大小
        neighbours: 邻域大小
        qkv_bias: 无用
        pe_multiplier: 位置编码乘性因子
        pe_bias: 位置编码偏置因子
        attn_drop_rate: drop比例
        drop_path_rate: BottleNeck的drop比例
        enable_checkpoint: checkpoint机制，以时间换空间
    """
    def __init__(
        self,
        depth,
        in_channels,
        embed_channels,
        groups,
        grid_size=None,
        neighbours=16,
        qkv_bias=True,
        pe_multiplier=False,
        pe_bias=True,
        attn_drop_rate=None,
        drop_path_rate=None,
        enable_checkpoint=False,
    ):
        super(Encoder, self).__init__()

        self.down = GridPool(
            in_channels=in_channels,
            out_channels=embed_channels,
            grid_size=grid_size,
        )

        self.blocks = BlockSequence(
            depth=depth,
            embed_channels=embed_channels,
            groups=groups,
            neighbours=neighbours,
            qkv_bias=qkv_bias,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
            attn_drop_rate=attn_drop_rate if attn_drop_rate is not None else 0.0,
            drop_path_rate=drop_path_rate if drop_path_rate is not None else 0.0,
            enable_checkpoint=enable_checkpoint,
        )

    def forward(self, points):
        """
        input: points: [pxo], [[n,3],[n,c],[b]]
        output: points: [pxo], [[ns,3],[ns,c],[b]], cluster: [n]
        """
        points, cluster = self.down(points)
        return self.blocks(points), cluster


class Decoder(nn.Module):
    """
    Decoder for Point Transformer V2, 先进行上采样, 再进行BlockSequence处理
    Args:
        in_channels: 输入维度
        skip_channels: 跳跃连接维度
        embed_channels: 输出维度
        groups: 分组数量
        depth: 解码器深度
        neighbours: 邻域大小
        qkv_bias: 无用
        pe_multiplier: 位置编码乘性因子
        pe_bias: 位置编码偏置因子
        attn_drop_rate: drop比例
        drop_path_rate: BottleNeck的drop比例
        enable_checkpoint: checkpoint机制，以时间换空间
        unpool_backend: 上采样方式，'map' or 'interp'
    """
    def __init__(
        self,
        in_channels,
        skip_channels,
        embed_channels,
        groups,
        depth,
        neighbours=16,
        qkv_bias=True,
        pe_multiplier=False,
        pe_bias=True,
        attn_drop_rate=None,
        drop_path_rate=None,
        enable_checkpoint=False,
        unpool_backend="map",
    ):
        super(Decoder, self).__init__()

        self.up = UnpoolWithSkip(
            in_channels=in_channels,
            out_channels=embed_channels,
            skip_channels=skip_channels,
            backend=unpool_backend,
        )

        self.blocks = BlockSequence(
            depth=depth,
            embed_channels=embed_channels,
            groups=groups,
            neighbours=neighbours,
            qkv_bias=qkv_bias,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
            attn_drop_rate=attn_drop_rate if attn_drop_rate is not None else 0.0,
            drop_path_rate=drop_path_rate if drop_path_rate is not None else 0.0,
            enable_checkpoint=enable_checkpoint,
        )

    def forward(self, points, skip_points, cluster):
        """
        input: points: [pxo], [[ns,3],[ns,c],[b]], skip_points: [pxo], [[n,3],[n,c],[b]], cluster: [n]
        output: points: [pxo], [[n,3],[n,c],[b]]
        """
        points = self.up(points, skip_points, cluster)
        return self.blocks(points)


class GVAPatchEmbed(nn.Module):
    """
    Patch Embedding for Point Transformer V2 (CNF variant)

    支持可选的 LAS 语义类别 Embedding 融合：将点云分类标签通过
    ``nn.Embedding`` 查表后与原始特征拼接，再送入投影层。

    Args:
        depth: 编码器深度
        in_channels: 输入维度
        embed_channels: 输出维度
        groups: 分组数量
        neighbours: 邻域大小
        qkv_bias: 无用
        pe_multiplier: 位置编码乘性因子
        pe_bias: 位置编码偏置因子
        attn_drop_rate: drop比例
        drop_path_rate: BottleNeck的drop比例
        enable_checkpoint: checkpoint机制，以时间换空间
        use_cls_embed: 是否启用 LAS 语义类别 Embedding
        num_classes: Embedding 查找表大小
        cls_embed_dim: 类别嵌入维度
    """
    def __init__(
        self,
        depth,
        in_channels,
        embed_channels,
        groups,
        neighbours=16,
        qkv_bias=True,
        pe_multiplier=False,
        pe_bias=True,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        enable_checkpoint=False,
        use_cls_embed=True,
        num_classes=32,
        cls_embed_dim=16,
    ):
        super(GVAPatchEmbed, self).__init__()
        self.in_channels = in_channels
        self.mid_channels = embed_channels // 2
        self.embed_channels = embed_channels
        self.use_cls_embed = use_cls_embed
        self.num_classes = num_classes

        if use_cls_embed:
            self.cls_embedding = nn.Embedding(num_classes, cls_embed_dim)
            actual_in = in_channels + cls_embed_dim
        else:
            actual_in = in_channels

        self.proj = nn.Sequential(
            nn.Linear(actual_in, self.mid_channels, bias=False),
            PointBatchNorm(self.mid_channels),
            nn.ReLU(inplace=True),
        )
        self.pointnet = PointNetLayer(actual_in, embed_channels - self.mid_channels, neighbours)

        self.blocks = BlockSequence(
            depth=depth,
            embed_channels=embed_channels,
            groups=groups,
            neighbours=neighbours,
            qkv_bias=qkv_bias,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            enable_checkpoint=enable_checkpoint,
        )

    def forward(self, points):
        """
        input:
            如果 use_cls_embed=True:  points = [coord, feat, offset, cls_label]
            如果 use_cls_embed=False: points = [coord, feat, offset]
        output: points: [coord, feat, offset]
        """
        if self.use_cls_embed:
            coord, feat, offset, cls_label = points
            cls_label = torch.clamp(cls_label.long(), min=0, max=self.num_classes - 1)
            cls_feat = self.cls_embedding(cls_label)  # (N, cls_embed_dim)
            feat = torch.cat([feat, cls_feat], dim=-1)  # (N, in_channels + cls_embed_dim)
        else:
            coord, feat, offset = points

        feat1 = self.proj(feat)
        feat2 = self.pointnet([coord, feat, offset])
        feat = torch.cat([feat1, feat2], dim=1)
        return self.blocks([coord, feat, offset])


@MODELS.register_module("PT-v2m5")
class PointTransformerV2(PointModule):
    """
    Point Transformer V2 Backbone (CNF variant)

    作为纯 backbone 使用，不包含 seg_head。
    输入: data_dict (dict) 或 Point 对象，需包含 coord, feat, offset。
    输出: Point 对象，feat 为解码器最终输出特征，维度 dec_channels[0]。

    支持可选的 LAS 语义类别 Embedding (``use_cls_embed``)：
    从 ``data_dict["segment"]`` 提取逐点类别标签，通过
    ``nn.Embedding`` 映射为可学习特征后与原始特征拼接。

    Args:
        in_channels: 输入特征维度
        use_cls_embed: 是否启用 LAS 语义类别 Embedding
        num_classes: Embedding 查找表大小
        cls_embed_dim: 类别嵌入维度
        (其余参数同 PT-v2m4)
    """
    def __init__(
        self,
        in_channels,
        patch_embed_depth=1,
        patch_embed_channels=48,
        patch_embed_groups=6,
        patch_embed_neighbours=8,
        enc_depths=(2, 2, 6, 2),
        enc_channels=(96, 192, 384, 512),
        enc_groups=(12, 24, 48, 64),
        enc_neighbours=(16, 16, 16, 16),
        dec_depths=(1, 1, 1, 1),
        dec_channels=(48, 96, 192, 384),
        dec_groups=(6, 12, 24, 48),
        dec_neighbours=(16, 16, 16, 16),
        grid_sizes=(0.06, 0.12, 0.24, 0.48),
        attn_qkv_bias=True,
        pe_multiplier=False,
        pe_bias=True,
        attn_drop_rate=0.0,
        drop_path_rate=0,
        enable_checkpoint=False,
        unpool_backend="map",
        use_cls_embed=True,
        num_classes=32,
        cls_embed_dim=16,
    ):
        super(PointTransformerV2, self).__init__()
        self.in_channels = in_channels
        self.use_cls_embed = use_cls_embed
        self.num_stages = len(enc_depths)
        assert self.num_stages == len(dec_depths)
        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(dec_channels)
        assert self.num_stages == len(enc_groups)
        assert self.num_stages == len(dec_groups)
        assert self.num_stages == len(enc_neighbours)
        assert self.num_stages == len(dec_neighbours)
        assert self.num_stages == len(grid_sizes)
        # 点云嵌入层
        self.patch_embed = GVAPatchEmbed(
            in_channels=in_channels,
            embed_channels=patch_embed_channels,
            groups=patch_embed_groups,
            depth=patch_embed_depth,
            neighbours=patch_embed_neighbours,
            qkv_bias=attn_qkv_bias,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
            attn_drop_rate=attn_drop_rate,
            enable_checkpoint=enable_checkpoint,
            use_cls_embed=use_cls_embed,
            num_classes=num_classes,
            cls_embed_dim=cls_embed_dim,
        )
        # bottleneck的drop率逐渐提高
        enc_dp_rates = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(enc_depths))
        ]
        dec_dp_rates = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(dec_depths))
        ]
        # 前一层的输出维度作为下一层的输入维度
        enc_channels = [patch_embed_channels] + list(enc_channels) # [48, 96, 192, 384, 512]
        dec_channels = list(dec_channels) + [enc_channels[-1]] # [48, 96, 192, 384, 512]
        # 编码器与解码器
        self.enc_stages = nn.ModuleList()
        self.dec_stages = nn.ModuleList()
        for i in range(self.num_stages):
            enc = Encoder(
                depth=enc_depths[i],
                in_channels=enc_channels[i],
                embed_channels=enc_channels[i + 1],
                groups=enc_groups[i],
                grid_size=grid_sizes[i],
                neighbours=enc_neighbours[i],
                qkv_bias=attn_qkv_bias,
                pe_multiplier=pe_multiplier,
                pe_bias=pe_bias,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=enc_dp_rates[
                    sum(enc_depths[:i]) : sum(enc_depths[: i + 1])
                ],
                enable_checkpoint=enable_checkpoint,
            )
            dec = Decoder(
                depth=dec_depths[i],
                in_channels=dec_channels[i + 1],
                skip_channels=enc_channels[i],
                embed_channels=dec_channels[i],
                groups=dec_groups[i],
                neighbours=dec_neighbours[i],
                qkv_bias=attn_qkv_bias,
                pe_multiplier=pe_multiplier,
                pe_bias=pe_bias,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dec_dp_rates[
                    sum(dec_depths[:i]) : sum(dec_depths[: i + 1])
                ],
                enable_checkpoint=enable_checkpoint,
                unpool_backend=unpool_backend,
            )
            self.enc_stages.append(enc)
            self.dec_stages.append(dec)

    def forward(self, data_dict):
        """
        input: data_dict (dict 或 Point): 需包含 "coord" [n, 3], "feat" [n, c], "offset" [b]
        output: Point 对象, feat 为解码器最终输出特征 [n, dec_channels[0]]
        """
        # 兼容 dict 和 Point 两种输入
        if not isinstance(data_dict, Point):
            point = Point(data_dict)
        else:
            point = data_dict

        coord = point.coord
        feat = point.feat
        offset = point.offset.int()

        # 提取逐点类别标签 (可选)
        if self.use_cls_embed:
            if "segment" in point.keys() and point["segment"] is not None:
                cls_label = point["segment"]
            else:
                cls_label = torch.zeros(coord.shape[0], dtype=torch.long,
                                        device=coord.device)
            if not torch.is_tensor(cls_label):
                cls_label = torch.as_tensor(cls_label, device=coord.device)
            if cls_label.dim() > 1:
                cls_label = cls_label.view(-1)
            points = [coord, feat, offset, cls_label]
        else:
            points = [coord, feat, offset]
        points = self.patch_embed(points)
        skips = [[points]]  # 便于添加cluster
        for i in range(self.num_stages):
            points, cluster = self.enc_stages[i](points)
            skips[-1].append(cluster)  # record grid cluster of pooling
            skips.append([points])  # record points info of current stage
        # 取出最后一层的点云信息
        points = skips.pop(-1)[0]  # unpooling points info in the last enc stage
        for i in reversed(range(self.num_stages)):
            skip_points, cluster = skips.pop(-1)
            points = self.dec_stages[i](points, skip_points, cluster)
        coord, feat, offset = points

        # 将输出包装为 Point 对象返回
        point.feat = feat
        return point
