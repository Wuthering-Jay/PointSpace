"""
DeepLANet V1 Backbone

Backbone only (符合 DefaultSegmentorV2 规范):
  - 输入: data_dict 或 Point 对象
  - 输出: Point 对象 (feat 为解码器最终输出特征)

主要特点:
  1. 使用 LocalFeatureAggregation (LFA) 替代 GroupedVectorAttention
  2. Stage-Level Position Embedding: 每个 Stage 只计算一次位置编码，极大节约计算量
"""

from copy import deepcopy
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn.pool import voxel_grid
from torch_scatter import segment_csr

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


def compute_stage_positional_encoding(coord, reference_index, dist):
    """
    阶段级位置编码计算（Stage-Level Position Embedding）

    核心创新: 在每个 Stage 开始时计算一次位置编码，所有 Block 共享，
    避免每层重复计算，极大节约计算量。

    位置编码包含:
    - 中心点坐标 (3)
    - 邻域点坐标 (3)
    - 相对位置 (3)
    - 距离 (1)
    共 10 维

    Args:
        coord: [n, 3] 点云坐标
        reference_index: [n, k] 邻域索引
        dist: [n, k] 邻域距离

    Returns:
        pos_encoding: [n, k, 10] 位置编码特征
    """
    n, k = reference_index.shape

    # 获取邻域坐标: [n, k, 3]
    neighbor_coord = pointops.grouping(reference_index, coord, coord, with_xyz=False)

    # 中心点坐标扩展: [n, 3] -> [n, k, 3]
    center_coord = coord.unsqueeze(1).expand(-1, k, -1)

    # 相对位置: [n, k, 3]
    relative_pos = center_coord - neighbor_coord

    # 距离: [n, k] -> [n, k, 1]
    dist_feat = dist.unsqueeze(-1)

    # 拼接位置编码: [n, k, 10]
    pos_encoding = torch.cat([
        center_coord,      # [n, k, 3]
        neighbor_coord,    # [n, k, 3]
        relative_pos,      # [n, k, 3]
        dist_feat,         # [n, k, 1]
    ], dim=-1)

    return pos_encoding


class StagePositionalEncoding(nn.Module):
    """
    阶段级位置编码器（Stage-Level Positional Encoding）

    **彻底优化**: 在 Stage 级别一次性将 10 维位置编码映射为高维特征 [n, k, d]
    所有 Block 共享这个高维 PE，避免每个 Block 重复计算 MLP

    这是对原始实现的关键改进:
    - 原来: 每个 Block 的 LSE 都执行 MLP(10->d)，重复计算 depth 次
    - 现在: Stage 级别执行一次 MLP(10->d)，所有 Block 共享

    Args:
        d: 输出位置特征维度
    """

    def __init__(self, d):
        super(StagePositionalEncoding, self).__init__()
        # 把原本在每层里的 MLP 提上来！在整个 Stage 只算一次
        self.mlp = nn.Sequential(
            nn.Linear(10, d),
            PointBatchNorm(d),
            nn.ReLU(inplace=True),
        )

    def forward(self, coord, reference_index, dist):
        """
        Args:
            coord: [n, 3] 点云坐标
            reference_index: [n, k] 邻域索引
            dist: [n, k] 邻域距离

        Returns:
            pe: [n, k, d] 高维位置特征，所有 Block 共享
        """
        # 计算 10 维原始位置编码
        pos_encoding = compute_stage_positional_encoding(coord, reference_index, dist)  # [n, k, 10]

        n, k, _ = pos_encoding.shape

        # 算一次高维 PE: [n, k, 10] -> [n, k, d]
        pe = self.mlp(pos_encoding.reshape(-1, 10)).reshape(n, k, -1)

        return pe


class LocalSpatialEncoding(nn.Module):
    """
    局部空间编码模块（极简版，无 MLP）

    **彻底优化**: 删除内部 MLP，直接使用 Stage 级别预计算的高维 PE
    - 原来: 每次调用都执行 MLP(10->d)
    - 现在: 直接使用预计算的 [n, k, d] PE

    无参数，纯特征拼接！
    """

    def __init__(self):
        super(LocalSpatialEncoding, self).__init__()
        # 删掉 __init__ 里的 MLP，无参数！

    def forward(self, pe, feat, coord, reference_index):
        """
        Args:
            pe: [n, k, d] 预计算的高维位置特征（Stage级别）
            feat: [n, d] 点云特征
            coord: [n, 3] 点云坐标（用于获取邻域特征）
            reference_index: [n, k] 邻域索引

        Returns:
            [n, k, 2*d] 拼接后的特征 (PE + 邻域特征)
        """
        # pe 已经是 [n, k, d] 了，直接用！
        # 获取邻域特征: [n, k, d]
        neighbor_feat = pointops.grouping(reference_index, feat, coord, with_xyz=False)

        # 拼接 PE 和邻域特征: [n, k, 2*d]
        output = torch.cat([pe, neighbor_feat], dim=-1)

        return output


class AttentivePooling(nn.Module):
    """
    注意力池化模块
    使用注意力机制对邻域特征进行加权聚合

    Args:
        in_channels: 输入特征维度
        out_channels: 输出特征维度
    """

    def __init__(self, in_channels, out_channels):
        super(AttentivePooling, self).__init__()

        # 注意力分数计算
        self.score_fn = nn.Sequential(
            nn.Linear(in_channels, in_channels, bias=False),
            nn.Softmax(dim=1),  # 在邻域维度上进行softmax
        )

        # 输出投影
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            PointBatchNorm(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, reference_index):
        """
        Args:
            x: [n, k, d] 邻域特征
            reference_index: [n, k] 邻域索引, 用于生成mask

        Returns:
            [n, out_channels] 池化后的特征
        """
        # 计算注意力分数: [n, k, d]
        scores = self.score_fn(x)  # [n, k, d]

        # 生成mask: 无效邻域点标记为0
        with torch.no_grad():
            mask = torch.sign(reference_index + 1).unsqueeze(-1)  # [n, k, 1]

        # 应用mask
        scores = scores * mask  # [n, k, d]

        # 加权求和: [n, k, d] -> [n, d]
        pooled = torch.sum(scores * x, dim=1)  # [n, d]

        # 输出投影: [n, d] -> [n, out_channels]
        return self.mlp(pooled)


class LocalFeatureAggregation(nn.Module):
    """
    局部特征聚合模块 (Local Feature Aggregation)
    基于 Stage-Level Position Embedding 彻底优化

    核心设计:
    1. mlp1: 特征降维 d_in -> d_out//2
    2. lse1 + pool1: 使用预计算的高维 PE [n,k,d] + 注意力池化
    3. lse2 + pool2: 再次使用高维 PE + 注意力池化
    4. mlp2 + shortcut: 特征升维 + 残差连接

    彻底优化: PE 的 MLP(10->d) 在 Stage 级别计算一次，LFA 直接使用高维 PE

    Args:
        d_in: 输入特征维度
        d_out: 输出特征维度 (实际输出为 2*d_out)
        num_neighbors: 邻域点数量
    """

    def __init__(self, d_in, d_out, num_neighbors):
        super(LocalFeatureAggregation, self).__init__()
        self.num_neighbors = num_neighbors
        self.d_out = d_out

        # 输入特征降维
        self.mlp1 = nn.Sequential(
            nn.Linear(d_in, d_out // 2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 输出特征升维
        self.mlp2 = nn.Linear(d_out, 2 * d_out)

        # Shortcut连接
        self.shortcut = nn.Sequential(
            nn.Linear(d_in, 2 * d_out),
            PointBatchNorm(2 * d_out),
        )

        # 两组局部空间编码模块（无参数，直接使用预计算的高维 PE）
        self.lse1 = LocalSpatialEncoding()
        self.lse2 = LocalSpatialEncoding()

        # 两组注意力池化模块
        # lse1 输出维度 2 * (d_out//2) = d_out, pool1 输出 d_out//2
        self.pool1 = AttentivePooling(d_out, d_out // 2)
        # lse2 输出维度 2 * (d_out//2) = d_out, pool2 输出 d_out
        self.pool2 = AttentivePooling(d_out, d_out)

        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, coord, feat, pe, reference_index):
        """
        Args:
            coord: [n, 3] 点云坐标
            feat: [n, d_in] 点云特征
            pe: [n, k, d_out//2] 预计算的高维位置特征（Stage级别）
            reference_index: [n, k] 邻域索引

        Returns:
            [n, 2*d_out] 输出特征
        """
        # 计算shortcut
        shortcut = self.shortcut(feat)  # [n, 2*d_out]

        # 第一阶段: 降维
        x = self.mlp1(feat)  # [n, d_out//2]

        # 第二阶段: 使用预计算的高维 PE + 注意力池化
        x = self.lse1(pe, x, coord, reference_index)  # [n, k, d_out]
        x = self.pool1(x, reference_index)  # [n, d_out//2]

        # 第三阶段: 再次使用高维 PE + 注意力池化
        x = self.lse2(pe, x, coord, reference_index)  # [n, k, d_out]
        x = self.pool2(x, reference_index)  # [n, d_out]

        # 第四阶段: 升维 + 残差连接
        x = self.mlp2(x)  # [n, 2*d_out]
        x = self.lrelu(x + shortcut)  # [n, 2*d_out]

        return x


class Block(nn.Module):
    """
    网络模块单位，结合LFA和BottleNeck对pxo进行处理
    使用 LocalFeatureAggregation 替代 GroupedVectorAttention
    基于 Stage-Level Position Embedding 优化

    Args:
        embed_channels: 输入输出维度
        num_neighbors: 邻域点数量
        drop_path_rate: BottleNeck的drop比例
        enable_checkpoint: checkpoint机制，以时间换空间
    """

    def __init__(
        self,
        embed_channels,
        num_neighbors=16,
        drop_path_rate=0.0,
        enable_checkpoint=False,
    ):
        super(Block, self).__init__()

        # LFA 输入 embed_channels, 输出 embed_channels (内部使用 d_out = embed_channels // 2)
        # 实际 LFA 输出为 2 * (embed_channels // 2) = embed_channels
        self.lfa = LocalFeatureAggregation(
            d_in=embed_channels,
            d_out=embed_channels // 2,  # 实际输出 2 * d_out = embed_channels
            num_neighbors=num_neighbors,
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

    def forward(self, points, pe, reference_index):
        """
        Args:
            points: [pxo], [[n,3],[n,c],[b]]
            pe: [n, k, d] 预计算的高维位置特征（Stage级别）
            reference_index: [n, k] 邻域索引

        Returns:
            [pxo], [[n,3],[n,c],[b]], 不改变维度
        """
        coord, feat, offset = points  # [n,3], [n,c], [b]
        identity = feat  # [n, c]
        feat = self.act(self.norm1(self.fc1(feat)))  # [n, c]

        # 使用 LFA，传入预计算的高维 PE
        feat = (
            self.lfa(coord, feat, pe, reference_index)
            if not self.enable_checkpoint
            else checkpoint(self.lfa, coord, feat, pe, reference_index, use_reentrant=False)
        )  # [n, c]

        feat = self.act(self.norm2(feat))  # [n, c]
        feat = self.norm3(self.fc3(feat))  # [n, c]
        feat = identity + self.drop_path(feat)  # [n, c], bottleneck设计
        feat = self.act(feat)  # [n, c]
        return [coord, feat, offset]  # [[n,3],[n,c],[b]]


class BlockSequence(nn.Module):
    """
    Block序列，多个Block堆叠
    基于 Stage-Level Position Embedding 彻底优化:
    1. 在 Stage 开始时计算一次 KNN 和 10 维位置编码
    2. 通过 StagePositionalEncoding 一次性映射为高维 PE [n, k, d]
    3. 所有 Block 共享高维 PE，避免每层重复计算 MLP

    Args:
        depth: Block数量
        embed_channels: 特征维度
        neighbours: 邻域点数量
        drop_path_rate: BottleNeck的drop比例
        enable_checkpoint: checkpoint机制
    """

    def __init__(
        self,
        depth,
        embed_channels,
        neighbours=16,
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

        # Stage-Level PE 编码器: [n, k, 10] -> [n, k, d_pe]
        # LFA 内部: d_out = embed_channels // 2, mlp1 输出 d_out // 2 = embed_channels // 4
        # LSE 拼接 pe 和 neighbor_feat，输出维度 d_pe + embed_channels/4
        # AttentivePooling 期望输入 d_out = embed_channels // 2
        # 所以 d_pe = embed_channels // 4
        d_pe = embed_channels // 4
        self.stage_pe = StagePositionalEncoding(d_pe)

        self.blocks = nn.ModuleList()

        # 多个 Block 堆叠
        for i in range(depth):
            block = Block(
                embed_channels=embed_channels,
                num_neighbors=neighbours,
                drop_path_rate=drop_path_rates[i],
                enable_checkpoint=enable_checkpoint,
            )
            self.blocks.append(block)

    def forward(self, points):
        """
        Stage-Level Position Embedding 彻底优化:
        1. 计算一次 KNN
        2. 通过 StagePositionalEncoding 一次性获得高维 PE
        3. 所有 Block 共享高维 PE

        Args:
            points: [pxo], [[n,3],[n,c],[b]]

        Returns:
            [pxo], [[n,3],[n,c],[b]]
        """
        coord, feat, offset = points

        # Stage级别: 计算一次 KNN 查询
        with torch.no_grad():
            reference_index, dist = pointops.knn_query(self.neighbours, coord, offset)

        # 整个 Stage 只发生 1 次 [N, K, 10] 到 [N, K, d] 的映射
        pe = self.stage_pe(coord, reference_index, dist)  # [n, k, d]

        # 所有 Block 共享同一个高维 PE
        for block in self.blocks:
            points = block(points, pe, reference_index)

        return points


class GridPool(nn.Module):
    """
    Partition-based Pooling (Grid Pooling)
    格网池化，基于体素划分进行池化下采样

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
        Args:
            points: [pxo], [[n,3],[n,c],[b]]
            start: [b, 3]

        Returns:
            points: [pxo], [[v,3],[v,c],[b]]
            cluster: [n]
        """
        coord, feat, offset = points
        batch = offset2batch(offset)
        feat = self.act(self.norm(self.fc(feat)))

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

        coord = segment_csr(coord[sorted_cluster_indices], idx_ptr, reduce="mean")
        feat = segment_csr(feat[sorted_cluster_indices], idx_ptr, reduce="max")
        batch = batch[idx_ptr[:-1]]
        offset = batch2offset(batch)
        return [coord, feat, offset], cluster.detach()


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
        Args:
            points: [pxo], [[n,3],[n,c],[b]]
            skip_points: [pxo], [[ns,3],[ns,c],[b]]
            cluster: [ns]

        Returns:
            points: [pxo], [[ns,3],[ns,c],[b]]
        """
        coord, feat, offset = points
        skip_coord, skip_feat, skip_offset = skip_points

        if self.backend == "map" and cluster is not None:
            feat = self.proj(feat)[cluster]
        else:
            feat = pointops.interpolation(
                coord, skip_coord, self.proj(feat), offset, skip_offset
            )

        if self.skip:
            feat = feat + self.proj_skip(skip_feat)

        return [skip_coord, feat, skip_offset]


class Encoder(nn.Module):
    """
    DeepLANet Encoder, 先进行格网池化, 再进行BlockSequence处理

    Args:
        depth: 编码器深度
        in_channels: 输入维度
        embed_channels: 输出维度
        grid_size: 体素大小
        neighbours: 邻域大小
        drop_path_rate: BottleNeck的drop比例
        enable_checkpoint: checkpoint机制
    """

    def __init__(
        self,
        depth,
        in_channels,
        embed_channels,
        grid_size=None,
        neighbours=16,
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
            neighbours=neighbours,
            drop_path_rate=drop_path_rate if drop_path_rate is not None else 0.0,
            enable_checkpoint=enable_checkpoint,
        )

    def forward(self, points):
        """
        Args:
            points: [pxo], [[n,3],[n,c],[b]]

        Returns:
            points: [pxo], [[ns,3],[ns,c],[b]]
            cluster: [n]
        """
        points, cluster = self.down(points)
        return self.blocks(points), cluster


class Decoder(nn.Module):
    """
    DeepLANet Decoder, 先进行上采样, 再进行BlockSequence处理

    Args:
        in_channels: 输入维度
        skip_channels: 跳跃连接维度
        embed_channels: 输出维度
        depth: 解码器深度
        neighbours: 邻域大小
        drop_path_rate: BottleNeck的drop比例
        enable_checkpoint: checkpoint机制
        unpool_backend: 上采样方式
    """

    def __init__(
        self,
        in_channels,
        skip_channels,
        embed_channels,
        depth,
        neighbours=16,
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
            neighbours=neighbours,
            drop_path_rate=drop_path_rate if drop_path_rate is not None else 0.0,
            enable_checkpoint=enable_checkpoint,
        )

    def forward(self, points, skip_points, cluster):
        """
        Args:
            points: [pxo], [[ns,3],[ns,c],[b]]
            skip_points: [pxo], [[n,3],[n,c],[b]]
            cluster: [n]

        Returns:
            points: [pxo], [[n,3],[n,c],[b]]
        """
        points = self.up(points, skip_points, cluster)
        return self.blocks(points)


class LFAPatchEmbed(nn.Module):
    """
    Patch Embedding for DeepLANet using LocalFeatureAggregation
    基于 Stage-Level Position Embedding 彻底优化

    核心优化:
    - 使用 StagePositionalEncoding 在 PatchEmbed 阶段一次性计算高维 PE
    - lfa_embed 直接使用高维 PE [n, k, d]，不再内部计算 MLP

    Args:
        depth: 编码器深度
        in_channels: 输入维度
        embed_channels: 输出维度
        neighbours: 邻域大小
        drop_path_rate: BottleNeck的drop比例
        enable_checkpoint: checkpoint机制
    """

    def __init__(
        self,
        depth,
        in_channels,
        embed_channels,
        neighbours=16,
        drop_path_rate=0.0,
        enable_checkpoint=False,
    ):
        super(LFAPatchEmbed, self).__init__()
        self.in_channels = in_channels
        self.embed_channels = embed_channels
        self.neighbours = neighbours

        # LFA 内部: d_out = embed_channels // 2, mlp1 输出 d_out // 2 = embed_channels // 4
        # LSE 拼接 pe 和 neighbor_feat，输出维度 d_pe + embed_channels/4
        # AttentivePooling 期望输入 d_out = embed_channels // 2
        # 所以 d_pe = embed_channels // 4
        d_pe = embed_channels // 4

        # Stage-Level PE 编码器: [n, k, 10] -> [n, k, d_pe]
        # 这里单独设置一个 StagePositionalEncoding，因为 lfa_embed 是独立的一层
        self.stage_pe = StagePositionalEncoding(d_pe)

        # 使用 LFA 进行初始特征提取
        # LFA 输入 in_channels, d_out = embed_channels // 2, 输出 embed_channels
        self.lfa_embed = LocalFeatureAggregation(
            d_in=in_channels,
            d_out=embed_channels // 2,
            num_neighbors=neighbours,
        )

        self.blocks = BlockSequence(
            depth=depth,
            embed_channels=embed_channels,
            neighbours=neighbours,
            drop_path_rate=drop_path_rate,
            enable_checkpoint=enable_checkpoint,
        )

    def forward(self, points):
        """
        Stage-Level Position Embedding 彻底优化:
        1. 计算一次 KNN
        2. 通过 StagePositionalEncoding 一次性获得高维 PE
        3. lfa_embed 直接使用高维 PE

        Args:
            points: [pxo], [[n,3],[n,c],[b]]

        Returns:
            points: [pxo], [[n,3],[n,c],[b]]
        """
        coord, feat, offset = points

        # Stage级别: 计算一次 KNN 查询
        with torch.no_grad():
            reference_index, dist = pointops.knn_query(self.neighbours, coord, offset)

        # 一次性计算高维 PE: [n, k, d_pe]
        pe = self.stage_pe(coord, reference_index, dist)

        # LFA 嵌入使用高维 PE
        feat = self.lfa_embed(coord, feat, pe, reference_index)  # [n, embed_channels]

        return self.blocks([coord, feat, offset])


@MODELS.register_module("DeepLANet-v1")
@MODELS.register_module("DeepLANet-V1")
class DeepLANetV1(PointModule):
    """
    DeepLANet V1 Backbone

    使用 LocalFeatureAggregation 替代 GroupedVectorAttention 的点云处理backbone

    作为纯 backbone 使用，不包含 seg_head。
    输入: data_dict (dict) 或 Point 对象，需包含 coord, feat, offset。
    输出: Point 对象，feat 为解码器最终输出特征，维度 dec_channels[0]。

    Args:
        in_channels: 输入特征维度
        patch_embed_depth: Patch Embedding深度
        patch_embed_channels: Patch Embedding输出维度
        patch_embed_neighbours: Patch Embedding邻域大小
        enc_depths: 编码器深度
        enc_channels: 编码器输出维度
        enc_neighbours: 编码器邻域大小
        dec_depths: 解码器深度
        dec_channels: 解码器输出维度
        dec_neighbours: 解码器邻域大小
        grid_sizes: 体素大小
        drop_path_rate: BottleNeck的drop比例
        enable_checkpoint: checkpoint机制
        unpool_backend: 上采样方式
    """

    def __init__(
        self,
        in_channels,
        patch_embed_depth=1,
        patch_embed_channels=48,
        patch_embed_neighbours=8,
        enc_depths=(2, 2, 6, 2),
        enc_channels=(96, 192, 384, 512),
        enc_neighbours=(16, 16, 16, 16),
        dec_depths=(1, 1, 1, 1),
        dec_channels=(48, 96, 192, 384),
        dec_neighbours=(16, 16, 16, 16),
        grid_sizes=(0.06, 0.12, 0.24, 0.48),
        drop_path_rate=0,
        enable_checkpoint=False,
        unpool_backend="map",
    ):
        super(DeepLANetV1, self).__init__()
        self.in_channels = in_channels
        self.num_stages = len(enc_depths)

        assert self.num_stages == len(dec_depths)
        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(dec_channels)
        assert self.num_stages == len(enc_neighbours)
        assert self.num_stages == len(dec_neighbours)
        assert self.num_stages == len(grid_sizes)

        # 点云嵌入层
        self.patch_embed = LFAPatchEmbed(
            in_channels=in_channels,
            embed_channels=patch_embed_channels,
            depth=patch_embed_depth,
            neighbours=patch_embed_neighbours,
            enable_checkpoint=enable_checkpoint,
        )

        # drop率逐渐提高
        enc_dp_rates = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(enc_depths))
        ]
        dec_dp_rates = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(dec_depths))
        ]

        # 前一层的输出维度作为下一层的输入维度
        enc_channels = [patch_embed_channels] + list(enc_channels)
        dec_channels = list(dec_channels) + [enc_channels[-1]]

        # 编码器与解码器
        self.enc_stages = nn.ModuleList()
        self.dec_stages = nn.ModuleList()

        for i in range(self.num_stages):
            enc = Encoder(
                depth=enc_depths[i],
                in_channels=enc_channels[i],
                embed_channels=enc_channels[i + 1],
                grid_size=grid_sizes[i],
                neighbours=enc_neighbours[i],
                drop_path_rate=enc_dp_rates[
                    sum(enc_depths[:i]): sum(enc_depths[:i + 1])
                ],
                enable_checkpoint=enable_checkpoint,
            )
            dec = Decoder(
                depth=dec_depths[i],
                in_channels=dec_channels[i + 1],
                skip_channels=enc_channels[i],
                embed_channels=dec_channels[i],
                neighbours=dec_neighbours[i],
                drop_path_rate=dec_dp_rates[
                    sum(dec_depths[:i]): sum(dec_depths[:i + 1])
                ],
                enable_checkpoint=enable_checkpoint,
                unpool_backend=unpool_backend,
            )
            self.enc_stages.append(enc)
            self.dec_stages.append(dec)

    def forward(self, data_dict):
        """
        Args:
            data_dict (dict 或 Point): 需包含 "coord" [n, 3], "feat" [n, c], "offset" [b]

        Returns:
            Point 对象, feat 为解码器最终输出特征 [n, dec_channels[0]]
        """
        # 兼容 dict 和 Point 两种输入
        if not isinstance(data_dict, Point):
            point = Point(data_dict)
        else:
            point = data_dict

        coord = point.coord
        feat = point.feat
        offset = point.offset.int()

        points = [coord, feat, offset]
        points = self.patch_embed(points)

        skips = [[points]]
        for i in range(self.num_stages):
            points, cluster = self.enc_stages[i](points)
            skips[-1].append(cluster)
            skips.append([points])

        points = skips.pop(-1)[0]
        for i in reversed(range(self.num_stages)):
            skip_points, cluster = skips.pop(-1)
            points = self.dec_stages[i](points, skip_points, cluster)

        coord, feat, offset = points

        point.feat = feat
        return point
