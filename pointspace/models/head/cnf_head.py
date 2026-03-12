"""Dual-branch Conditional Neural Field (CNF) head.

Components
----------
* :class:`RelativeFourierEncoding`
    High-frequency Fourier encoder for relative coordinates.
* :class:`DualBranchCNFHead`
    Asymmetric dual-stream decoding:
      - Base stream: Linear PE + IDW anchor → low-frequency trend
      - Detail stream: Fourier PE + relative_z → high-frequency residual
"""

import math

import torch
import torch.nn as nn
import torch_cluster

from pointspace.models.utils import offset2batch
from ..builder import MODELS


# ──────────────────────────────────────────────────────────────────────────────
# Sub-modules
# ──────────────────────────────────────────────────────────────────────────────


class RelativeFourierEncoding(nn.Module):
    """High-frequency Fourier amplifier for small relative coordinates.

    Maps each scalar component *v* of the input to
    ``[sin(v·π·2⁰), cos(v·π·2⁰), …, sin(v·π·2^(L-1)), cos(v·π·2^(L-1))]``.

    For typical local distances in [-2 m, 2 m], the highest frequency
    ``2^(num_freqs-1)`` is sufficient to capture centimetre-level variation.

    Args:
        in_dim (int): Input coordinate dimensionality (2 for XY).
        num_freqs (int): Number of frequency octaves *L*.
    """

    def __init__(self, in_dim=2, num_freqs=6):
        super().__init__()
        self.in_dim = in_dim
        self.num_freqs = num_freqs
        freq_bands = 2.0 ** torch.linspace(0, num_freqs - 1, num_freqs)
        self.register_buffer("freq_bands", freq_bands)

    @property
    def output_dim(self):
        return self.in_dim * self.num_freqs * 2

    def forward(self, x):
        """
        Args:
            x: ``(*, in_dim)`` relative coordinates.
        Returns:
            ``(*, in_dim * num_freqs * 2)`` Fourier features.
        """
        encoded = []
        for freq in self.freq_bands:
            encoded.append(torch.sin(x * freq * math.pi))
            encoded.append(torch.cos(x * freq * math.pi))
        return torch.cat(encoded, dim=-1)


# ──────────────────────────────────────────────────────────────────────────────
# Main head
# ──────────────────────────────────────────────────────────────────────────────


@MODELS.register_module()
class DualBranchCNFHead(nn.Module):
    """Asymmetric dual-stream CNF head for terrain surface reconstruction.

    **Feature bridge** — Physical XY-distance KNN finds K nearest support
    points per query.  Inverse-distance weighting (IDW) produces a smooth
    ``local_z_anchor`` (physical height datum).

    **Base stream (low-frequency)**::

        relative_xy → Linear PE → concat(backbone_feat) → fuse → max-pool
        → mlp_base → base_z_residual
        base_z = local_z_anchor + base_z_residual

    **Detail stream (high-frequency)**::

        relative_xy → RelativeFourierEncoding
        relative_z  = support_z − local_z_anchor       (physical micro-relief)
        concat(backbone_feat, fourier_feat, relative_z) → fuse → max-pool
        → mlp_detail → delta_z

    **Final output**::

        pred_z = base_z + delta_z

    All activations are Softplus(β=100) for C² smoothness.

    Args:
        backbone_out_channels (int): Feature dim from backbone (*C*).
        query_dim (int): Coordinate dim for queries (2 = XY).
        num_targets (int): Output values per query (1 = scalar z).
        k_neighbors (int): KNN neighbours.
        hidden_dim (int): Hidden dimension for MLPs.
        num_freqs (int): Fourier PE octaves for detail branch.
        base_hidden_dims (list[int]): Base MLP hidden sizes.
        detail_hidden_dims (list[int]): Detail MLP hidden sizes.
    """

    def __init__(
        self,
        backbone_out_channels=64,
        query_dim=2,
        num_targets=1,
        k_neighbors=16,
        hidden_dim=256,
        num_freqs=6,
        base_hidden_dims=None,
        detail_hidden_dims=None,
    ):
        super().__init__()
        if base_hidden_dims is None:
            base_hidden_dims = [128, 64]
        if detail_hidden_dims is None:
            detail_hidden_dims = [128, 64]

        self.query_dim = query_dim
        self.num_targets = num_targets
        self.k_neighbors = k_neighbors

        beta = 100  # Softplus sharpness — near-ReLU shape, still C²

        # ---- Base stream: Linear PE on relative_xy ----
        self.pe_base = nn.Sequential(
            nn.Linear(query_dim, 32),
            nn.Softplus(beta=beta),
            nn.Linear(32, 64),
        )
        self.fuse_base = nn.Sequential(
            nn.Linear(backbone_out_channels + 64, hidden_dim),
            nn.Softplus(beta=beta),
        )
        layers_b = []
        d = hidden_dim
        for h in base_hidden_dims:
            layers_b.extend([nn.Linear(d, h), nn.Softplus(beta=beta)])
            d = h
        layers_b.append(nn.Linear(d, num_targets))
        self.mlp_base = nn.Sequential(*layers_b)

        # ---- Detail stream: Fourier PE + relative_z ----
        self.rfe_detail = RelativeFourierEncoding(
            in_dim=query_dim, num_freqs=num_freqs,
        )
        fourier_dim = self.rfe_detail.output_dim  # in_dim * num_freqs * 2
        # fuse: backbone_feat + fourier_feat + relative_z (1)
        self.fuse_detail = nn.Sequential(
            nn.Linear(backbone_out_channels + fourier_dim + 1, hidden_dim),
            nn.Softplus(beta=beta),
        )
        layers_d = []
        d = hidden_dim
        for h in detail_hidden_dims:
            layers_d.extend([nn.Linear(d, h), nn.Softplus(beta=beta)])
            d = h
        layers_d.append(nn.Linear(d, num_targets))
        self.mlp_detail = nn.Sequential(*layers_d)

    def forward(self, support_coord, support_feat, query_coord,
                support_offset=None, query_offset=None):
        """
        Args:
            support_coord: (N, 3) support point positions (x, y, z).
            support_feat: (N, C) backbone per-point features.
            query_coord: (Q, query_dim) query positions (x, y).
            support_offset / query_offset: optional (B,) batch boundaries.

        Returns:
            (pred_base, pred_detail) — each (Q,) or (Q, num_targets).
        """
        qd = self.query_dim
        K = self.k_neighbors

        s_xy = support_coord[:, :qd].contiguous()
        q_xy = query_coord[:, :qd].contiguous()

        # Batch indices
        if support_offset is not None and query_offset is not None:
            batch_s = offset2batch(support_offset)
            batch_q = offset2batch(query_offset)
        else:
            batch_s = torch.zeros(s_xy.shape[0], dtype=torch.long, device=s_xy.device)
            batch_q = torch.zeros(q_xy.shape[0], dtype=torch.long, device=q_xy.device)

        # 1. KNN in physical XY space
        assign = torch_cluster.knn(
            x=s_xy, y=q_xy, k=K,
            batch_x=batch_s, batch_y=batch_q,
        )
        q_row, s_col = assign[0], assign[1]

        Q = q_xy.shape[0]

        # 2. Gather neighbour data → (Q, K, ...)
        grouped_coords = support_coord[s_col].view(Q, K, -1)       # (Q, K, 3)
        grouped_feats = support_feat[s_col].view(Q, K, -1)         # (Q, K, C)

        # 3. Relative XY: neighbour − query
        relative_xy = grouped_coords[:, :, :qd] - q_xy.unsqueeze(1)  # (Q, K, qd)

        # 4. IDW anchor: smooth physical height datum
        dist_sq = torch.sum(relative_xy ** 2, dim=-1)                # (Q, K)
        weights = 1.0 / (dist_sq + 1e-6)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        local_z_anchor = torch.sum(
            grouped_coords[:, :, 2] * weights, dim=-1, keepdim=True
        )  # (Q, 1)

        # ── Base stream ──────────────────────────────────────────────
        pe_b = self.pe_base(relative_xy)                              # (Q, K, 64)
        feat_b = torch.cat([grouped_feats, pe_b], dim=-1)            # (Q, K, C+64)
        feat_b = self.fuse_base(feat_b)                               # (Q, K, H)
        q_feat_base = torch.max(feat_b, dim=1)[0]                    # (Q, H)

        base_z_residual = self.mlp_base(q_feat_base)                 # (Q, num_targets)
        pred_base = local_z_anchor + base_z_residual                  # (Q, num_targets)

        # ── Detail stream ────────────────────────────────────────────
        pe_d = self.rfe_detail(relative_xy)                           # (Q, K, fourier_dim)
        relative_z = grouped_coords[:, :, 2:3] - local_z_anchor.unsqueeze(1)  # (Q, K, 1)

        feat_d = torch.cat([grouped_feats, pe_d, relative_z], dim=-1)
        feat_d = self.fuse_detail(feat_d)                             # (Q, K, H)
        q_feat_detail = torch.max(feat_d, dim=1)[0]                  # (Q, H)

        pred_detail = self.mlp_detail(q_feat_detail)                  # (Q, num_targets)

        # Squeeze scalar output
        if self.num_targets == 1:
            pred_base = pred_base.squeeze(-1)
            pred_detail = pred_detail.squeeze(-1)

        return pred_base, pred_detail


# ──────────────────────────────────────────────────────────────────────────────
# SingleBranchCNFHead
# ──────────────────────────────────────────────────────────────────────────────


@MODELS.register_module()
class SingleBranchCNFHead(nn.Module):
    """Unified single-stream CNF head for terrain surface reconstruction.

    Combines all techniques from the dual-branch design into one fused
    representation:

    1. **KNN** in physical XY → gather K nearest support neighbours.
    2. **IDW anchor** → smooth local height datum ``z_anchor``.
    3. **Linear PE** on relative XY (learnable low-freq embedding).
    4. **Fourier PE** on relative XY (high-freq positional encoding).
    5. **Relative Z** = ``support_z - z_anchor`` (local micro-relief).
    6. **Concatenate** backbone features + linear PE + Fourier PE +
       relative_z → fuse → max-pool → MLP → residual.
    7. ``pred_z = z_anchor + residual``.

    All activations are Softplus(β=100) for C² smoothness.

    Args:
        backbone_out_channels (int): Feature dim from backbone (*C*).
        query_dim (int): Coordinate dim for queries (2 = XY).
        num_targets (int): Output values per query (1 = scalar z).
        k_neighbors (int): KNN neighbours.
        hidden_dim (int): Hidden dimension after fusion.
        num_freqs (int): Fourier PE octaves.
        mlp_hidden_dims (list[int]): MLP hidden layer sizes.
    """

    def __init__(
        self,
        backbone_out_channels=64,
        query_dim=2,
        num_targets=1,
        k_neighbors=16,
        hidden_dim=256,
        num_freqs=6,
        mlp_hidden_dims=None,
    ):
        super().__init__()
        if mlp_hidden_dims is None:
            mlp_hidden_dims = [128, 64]

        self.query_dim = query_dim
        self.num_targets = num_targets
        self.k_neighbors = k_neighbors

        beta = 100  # Softplus sharpness

        # ---- Position encodings ----
        # Linear PE (low-frequency learnable)
        self.pe_linear = nn.Sequential(
            nn.Linear(query_dim, 32),
            nn.Softplus(beta=beta),
            nn.Linear(32, 64),
        )
        # Fourier PE (high-frequency)
        self.rfe = RelativeFourierEncoding(in_dim=query_dim, num_freqs=num_freqs)
        fourier_dim = self.rfe.output_dim  # in_dim * num_freqs * 2

        # ---- Fusion: backbone_feat + linear_pe + fourier_pe + relative_z + local_z_std(1)
        fuse_in = backbone_out_channels + 64 + fourier_dim + 2

        # Value 分支: 提取高维特征内容
        self.value_proj = nn.Sequential(
            nn.Linear(fuse_in, hidden_dim),
            nn.Softplus(beta=beta),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # Attention 分支: 评估邻居重要性 (发言权重)
        self.attn_proj = nn.Sequential(
            nn.Linear(fuse_in, hidden_dim // 2),
            nn.Softplus(beta=beta),
            nn.Linear(hidden_dim // 2, 1)
        )

        # ---- MLP → residual ----
        layers = []
        d = hidden_dim
        for h in mlp_hidden_dims:
            layers.extend([nn.Linear(d, h), nn.Softplus(beta=beta)])
            d = h
        layers.append(nn.Linear(d, num_targets))
        self.mlp = nn.Sequential(*layers)

    def forward(self, support_coord, support_feat, query_coord,
                support_offset=None, query_offset=None):
        """
        Args:
            support_coord: (N, 3) support point positions (x, y, z).
            support_feat: (N, C) backbone per-point features.
            query_coord: (Q, query_dim) query positions (x, y).
            support_offset / query_offset: optional (B,) batch boundaries.

        Returns:
            pred_z: (Q,) or (Q, num_targets) predicted values.
        """
        qd = self.query_dim
        K = self.k_neighbors

        s_xy = support_coord[:, :qd].contiguous()
        q_xy = query_coord[:, :qd].contiguous()

        # Batch indices
        if support_offset is not None and query_offset is not None:
            batch_s = offset2batch(support_offset)
            batch_q = offset2batch(query_offset)
        else:
            batch_s = torch.zeros(s_xy.shape[0], dtype=torch.long, device=s_xy.device)
            batch_q = torch.zeros(q_xy.shape[0], dtype=torch.long, device=q_xy.device)

        # 1. KNN in physical XY space
        assign = torch_cluster.knn(
            x=s_xy, y=q_xy, k=K,
            batch_x=batch_s, batch_y=batch_q,
        )
        q_row, s_col = assign[0], assign[1]
        Q = q_xy.shape[0]

        # 2. Gather neighbour data → (Q, K, ...)
        grouped_coords = support_coord[s_col].view(Q, K, -1)       # (Q, K, 3)
        grouped_feats = support_feat[s_col].view(Q, K, -1)         # (Q, K, C)

        # 3. Relative XY: neighbour − query
        relative_xy = grouped_coords[:, :, :qd] - q_xy.unsqueeze(1)  # (Q, K, qd)

        # 4. IDW anchor: smooth physical height datum
        dist_sq = torch.sum(relative_xy ** 2, dim=-1)                # (Q, K)
        weights = 1.0 / (dist_sq + 1e-6)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        local_z_anchor = torch.sum(
            grouped_coords[:, :, 2] * weights, dim=-1, keepdim=True
        )  # (Q, 1)

        # 5. Position encodings
        pe_lin = self.pe_linear(relative_xy)                          # (Q, K, 64)
        pe_four = self.rfe(relative_xy)                               # (Q, K, fourier_dim)

        # 6. Relative Z (local micro-relief)
        relative_z = grouped_coords[:, :, 2:3] - local_z_anchor.unsqueeze(1)  # (Q, K, 1)

        # 🌟 地形粗糙度探雷器 (防 NaN 处理)
        local_z_std = torch.std(grouped_coords[:, :, 2], dim=1, unbiased=False, keepdim=True) 
        local_z_std = local_z_std.unsqueeze(1).expand(-1, K, -1)      # (Q, K, 1)

        # 7. 🌟 交叉注意力无损融合 (取代 MaxPool)
        feat_cat = torch.cat([grouped_feats, pe_lin, pe_four, relative_z, local_z_std], dim=-1)
        
        value = self.value_proj(feat_cat)           # (Q, K, H)
        attn_logits = self.attn_proj(feat_cat)      # (Q, K, 1)
        
        attn_weights = torch.softmax(attn_logits, dim=1)  # 邻居权重归一化 (Q, K, 1)
        q_feat = torch.sum(value * attn_weights, dim=1)   # 加权求和融合 -> (Q, H)

        residual = self.mlp(q_feat)                                   # (Q, num_targets)
        pred_z = local_z_anchor + residual                            # (Q, num_targets)

        # Squeeze scalar output
        if self.num_targets == 1:
            pred_z = pred_z.squeeze(-1)
            local_z_anchor = local_z_anchor.squeeze(-1)

        # Training: also return IDW anchor for terrain-complexity weighting
        if self.training:
            return pred_z, local_z_anchor
        return pred_z
