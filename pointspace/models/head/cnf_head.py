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

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
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
# CrossGroupedVectorAttention
# ──────────────────────────────────────────────────────────────────────────────


class CrossGroupedVectorAttention(nn.Module):
    """Grouped vector attention from query → key/value neighbors.

    Uses positional multiplier and bias (linear_p_multiplier / linear_p_bias)
    to modulate relation_qk and value by 3-D relative position.

    Args:
        q_channels: Feature dims of query (Q, q_channels).
        kv_channels: Feature dims of key/value neighbors (Q, K, kv_channels).
        embed_channels: Internal attention dimension. Must be divisible by groups.
        groups: Number of attention heads.
        attn_drop_rate: Dropout rate on attention weights.
    """

    def __init__(self, q_channels, kv_channels, embed_channels, groups,
                 attn_drop_rate=0.0):
        super().__init__()
        self.groups = groups
        assert embed_channels % groups == 0, (
            f"embed_channels ({embed_channels}) must be divisible by groups ({groups})"
        )

        self.linear_q = nn.Sequential(
            nn.Linear(q_channels, embed_channels),
            nn.LayerNorm(embed_channels),
            nn.Softplus(beta=100),
        )
        self.linear_k = nn.Sequential(
            nn.Linear(kv_channels, embed_channels),
            nn.LayerNorm(embed_channels),
            nn.Softplus(beta=100),
        )
        self.linear_v = nn.Linear(kv_channels, embed_channels)

        self.linear_p_multiplier = nn.Sequential(
            nn.Linear(3, embed_channels),
            nn.LayerNorm(embed_channels),
            nn.Softplus(beta=100),
            nn.Linear(embed_channels, embed_channels),
        )
        self.linear_p_bias = nn.Sequential(
            nn.Linear(3, embed_channels),
            nn.LayerNorm(embed_channels),
            nn.Softplus(beta=100),
            nn.Linear(embed_channels, embed_channels),
        )

        self.weight_encoding = nn.Sequential(
            nn.Linear(embed_channels, groups),
            nn.LayerNorm(groups),
            nn.Softplus(beta=100),
            nn.Linear(groups, groups),
        )
        self.softmax = nn.Softmax(dim=1)
        self.attn_drop = nn.Dropout(attn_drop_rate)

    def forward(self, q_feat, grouped_kv_feat, relative_pos):
        """
        Args:
            q_feat: (Q, q_channels) query features.
            grouped_kv_feat: (Q, K, kv_channels) neighbor key/value features.
            relative_pos: (Q, K, 3) 3-D relative position (neighbor − query).

        Returns:
            (Q, embed_channels) attended features.
        """
        query = self.linear_q(q_feat)            # (Q, C)
        key = self.linear_k(grouped_kv_feat)     # (Q, K, C)
        value = self.linear_v(grouped_kv_feat)   # (Q, K, C)

        relation_qk = key - query.unsqueeze(1)   # (Q, K, C)
        pem = self.linear_p_multiplier(relative_pos)   # (Q, K, C)
        peb = self.linear_p_bias(relative_pos)         # (Q, K, C)

        relation_qk = relation_qk * pem + peb    # (Q, K, C)
        value = value + peb                      # (Q, K, C)

        weight = self.weight_encoding(relation_qk)   # (Q, K, G)
        weight = self.attn_drop(self.softmax(weight))  # (Q, K, G)

        value = einops.rearrange(value, "q k (g i) -> q k g i", g=self.groups)
        feat = torch.einsum("q k g i, q k g -> q g i", value, weight)
        feat = einops.rearrange(feat, "q g i -> q (g i)")  # (Q, C)
        return feat


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
                support_offset=None, query_offset=None,
                support_segment=None):
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
# SingleBranchCNFHead  (Cross-GVA + FiLM + RBF Gaussian ground anchor)
# ──────────────────────────────────────────────────────────────────────────────


@MODELS.register_module()
class SingleBranchCNFHead(nn.Module):
    """Dual-KNN CNF head: Cross-GVA + FiLM modulation + learnable RBF anchor.

    Architecture overview:

    **Ground Branch** (wider KNN, K_g = K × 3):
      - RBF Gaussian kernel (learnable σ) instead of IDW
      - Outputs z_anchor (height datum) + feat_anchor (local feature mean)
      - Computes macro-terrain stats: z_range, z_skew, local_z_std

    **Semantic Branch** (K nearest overall):
      - Gathers all-class neighbours
      - Concatenates explicit class embedding + Z-Fourier PE

    **Fusion (Cross-GVA + FiLM)**:
      - CrossGroupedVectorAttention(q=feat_anchor, kv=semantic_neighbors)
        with 3-D relative position modulation
      - FiLM (Frequency-based Input Layerwise Modulation): γ·F + β
        conditioned on macro terrain stats (range, skew, std)
      - MLP → residual on z_anchor → pred_z

    **No-segment fallback**:
      When ``support_segment`` is None or contains no ground-class points,
      uses the lowest 2–7 % of z values as a robust pseudo-ground set
      (strips the very lowest percentile to exclude noise/water returns).

    Args:
        backbone_out_channels: Feature dim from backbone (*C*).
        query_dim: Coordinate dim for queries (2 = XY).
        num_targets: Output values per query (1 = scalar z).
        k_neighbors: KNN neighbours for semantic branch (K). Ground branch
            uses 3K to capture wider terrain context.
        hidden_dim: Internal attention dimension and FiLM dimension.
        z_num_freqs: Fourier PE octaves for Z-axis height difference.
        mlp_hidden_dims: MLP hidden layer sizes.
        ground_class: Ground class label in ``segment``.
        num_classes: Total semantic classes (for class embedding table).
        class_embed_dim: Dimensionality of class embedding.
        attn_groups: Number of attention head groups in Cross-GVA.
            Must divide hidden_dim.
        predict_normals: If True, add a normal_mlp to predict surface normals.
            Training returns (pred_z, z_anchor, pred_normal) instead of 2-tuple.
    """

    def __init__(
        self,
        backbone_out_channels=64,
        query_dim=2,
        num_targets=1,
        k_neighbors=16,
        hidden_dim=256,
        z_num_freqs=4,
        mlp_hidden_dims=None,
        ground_class=2,
        num_classes=20,
        class_embed_dim=16,
        attn_groups=4,
        predict_normals=False,
    ):
        super().__init__()
        if mlp_hidden_dims is None:
            mlp_hidden_dims = [128, 64]

        self.query_dim = query_dim
        self.num_targets = num_targets
        self.k_neighbors = k_neighbors
        self.ground_class = ground_class

        # Learnable Gaussian kernel bandwidth (RBF)
        self.sigma = nn.Parameter(torch.tensor([1.0]))

        self.class_embed = nn.Embedding(num_classes, class_embed_dim)
        self.z_rfe = RelativeFourierEncoding(in_dim=1, num_freqs=z_num_freqs)
        z_fourier_dim = self.z_rfe.output_dim

        # Cross-GVA: q ← feat_anchor (C), kv ← [feat, class_embed, z_pe]
        kv_channels = backbone_out_channels + class_embed_dim + z_fourier_dim
        self.cross_gva = CrossGroupedVectorAttention(
            q_channels=backbone_out_channels,
            kv_channels=kv_channels,
            embed_channels=hidden_dim,
            groups=attn_groups,
        )

        # FiLM conditioning network (input: range, skew, std — 3 scalars)
        self.film_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.Softplus(beta=100),
            nn.Linear(hidden_dim, hidden_dim * 2),  # outputs γ and β
        )
        # Identity-mapping init: γ → 1, β → 0 so early training is stable
        nn.init.zeros_(self.film_mlp[2].weight)
        nn.init.zeros_(self.film_mlp[2].bias)
        self.film_mlp[2].bias.data[:hidden_dim] = 1.0  # front half = γ

        layers = []
        d = hidden_dim
        for h in mlp_hidden_dims:
            layers.extend([nn.Linear(d, h), nn.Softplus(beta=100)])
            d = h
        layers.append(nn.Linear(d, num_targets))
        self.mlp = nn.Sequential(*layers)

        # Normal prediction head (optional)
        self.predict_normals = predict_normals
        if predict_normals:
            self.normal_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Softplus(beta=100),
                nn.Linear(hidden_dim // 2, 3),
            )
            # Initialize for upward-facing default normal (0, 0, 1)
            nn.init.normal_(self.normal_mlp[0].weight, std=0.01)
            nn.init.zeros_(self.normal_mlp[0].bias)
            nn.init.normal_(self.normal_mlp[2].weight, std=0.01)
            nn.init.zeros_(self.normal_mlp[2].bias)
            self.normal_mlp[2].bias.data[2] = 1.0

    # ------------------------------------------------------------------
    # KNN helper: sparse assign → dense (Q, K) index matrix
    # ------------------------------------------------------------------
    @staticmethod
    def _to_dense(q_row, s_col, Q, K, device):
        counts = torch.bincount(q_row, minlength=Q)[:Q]
        if (counts == K).all():
            return s_col.reshape(Q, K)
        dense = torch.zeros((Q, K), dtype=torch.long, device=device)
        ptr = torch.cat([counts.new_zeros(1), counts.cumsum(0)])[:-1]
        for k in range(K):
            valid = counts > k
            dense[valid, k] = s_col[ptr[valid] + k]
            mask = (~valid) & (counts > 0)
            if mask.any():
                dense[mask, k] = s_col[ptr[mask]]
        return dense

    def forward(self, support_coord, support_feat, query_coord,
                support_offset=None, query_offset=None,
                support_segment=None):
        """
        Args:
            support_coord: (N, 3) support point positions (x, y, z).
            support_feat:  (N, C) backbone per-point features.
            query_coord:   (Q, query_dim) query positions (x, y).
            support_offset / query_offset: optional (B,) batch boundaries.
            support_segment: (N,) or (N, 1) integer class labels.
        """
        qd = self.query_dim
        K = self.k_neighbors
        K_g = K * 3  # wider ground-branch receptive field
        Q = query_coord.shape[0]

        # ------------------------------------------------------------------
        # 0. Ground mask: class-based or robust 2–7 % lowest-z fallback
        # ------------------------------------------------------------------
        if support_segment is not None:
            ground_mask = (support_segment.squeeze() == self.ground_class)
            if not ground_mask.any():
                # No class-2 points → take robust lowest-z slice
                z_vals = support_coord[:, 2]
                n = z_vals.shape[0]
                i0 = max(1, int(0.02 * n))
                i1 = max(i0 + 1, int(0.07 * n))
                sorted_idx = torch.argsort(z_vals)
                ground_mask = torch.zeros_like(ground_mask)
                ground_mask[sorted_idx[i0:i1]] = True
        else:
            z_vals = support_coord[:, 2]
            n = z_vals.shape[0]
            i0 = max(1, int(0.02 * n))
            i1 = max(i0 + 1, int(0.07 * n))
            sorted_idx = torch.argsort(z_vals)
            ground_mask = torch.zeros(
                support_coord.shape[0], dtype=torch.bool,
                device=support_coord.device,
            )
            ground_mask[sorted_idx[i0:i1]] = True

        q_xy = query_coord[:, :qd].contiguous()
        s_coord_g = support_coord[ground_mask]
        s_feat_g = support_feat[ground_mask]
        s_xy_g = s_coord_g[:, :qd].contiguous()
        s_xy_full = support_coord[:, :qd].contiguous()

        if support_offset is not None and query_offset is not None:
            batch_s_full = offset2batch(support_offset)
            batch_s_g = batch_s_full[ground_mask]
            batch_q = offset2batch(query_offset)
        else:
            batch_s_full = torch.zeros(
                support_coord.shape[0], dtype=torch.long,
                device=support_coord.device,
            )
            batch_s_g = torch.zeros(
                s_coord_g.shape[0], dtype=torch.long,
                device=s_coord_g.device,
            )
            batch_q = torch.zeros(Q, dtype=torch.long, device=q_xy.device)

        # ------------------------------------------------------------------
        # 1. Ground Branch: wide-KNN + learnable RBF anchor
        # ------------------------------------------------------------------
        assign_g = torch_cluster.knn(
            x=s_xy_g, y=q_xy, k=K_g,
            batch_x=batch_s_g, batch_y=batch_q,
        )
        s_col_g_dense = self._to_dense(assign_g[0], assign_g[1], Q, K_g, q_xy.device)

        grouped_coords_g = s_coord_g[s_col_g_dense]   # (Q, K_g, 3)
        grouped_feats_g = s_feat_g[s_col_g_dense]     # (Q, K_g, C)

        relative_xy_g = grouped_coords_g[:, :, :qd] - q_xy.unsqueeze(1)
        dist_sq_g = torch.sum(relative_xy_g ** 2, dim=-1)             # (Q, K_g)

        # Learnable Gaussian (RBF) kernel weights
        logits_g = -dist_sq_g / (2 * self.sigma ** 2 + 1e-6)
        weights_g = torch.nn.functional.softmax(logits_g, dim=-1) # [Q, K_g]

        local_feat_anchor = torch.bmm(weights_g.unsqueeze(1), grouped_feats_g).squeeze(1)
        local_z_anchor = torch.bmm(weights_g.unsqueeze(1), grouped_coords_g[:, :, 2:3]).squeeze(1) # (Q, 1)

        # Macro terrain statistics for FiLM conditioning
        macro_z_max = grouped_coords_g[:, :, 2].max(dim=1, keepdim=True).values
        macro_z_min = grouped_coords_g[:, :, 2].min(dim=1, keepdim=True).values
        macro_z_range = macro_z_max - macro_z_min                              # (Q, 1)
        macro_z_skew = grouped_coords_g[:, :, 2].mean(dim=1, keepdim=True) - local_z_anchor  # (Q, 1)

        # ------------------------------------------------------------------
        # 2. Semantic Branch: full-class KNN + class embed + Z Fourier PE
        # ------------------------------------------------------------------
        assign_f = torch_cluster.knn(
            x=s_xy_full, y=q_xy, k=K,
            batch_x=batch_s_full, batch_y=batch_q,
        )
        s_col_f_dense = self._to_dense(assign_f[0], assign_f[1], Q, K, q_xy.device)

        grouped_coords_f = support_coord[s_col_f_dense]   # (Q, K, 3)
        grouped_feats_f = support_feat[s_col_f_dense]     # (Q, K, C)

        if support_segment is not None:
            grouped_seg = support_segment.squeeze()[s_col_f_dense]  # (Q, K)
        else:
            grouped_seg = torch.full(
                (Q, K), self.ground_class, dtype=torch.long,
                device=q_xy.device,
            )
        grouped_class_embed = self.class_embed(grouped_seg)  # (Q, K, D_cls)

        # Z-axis relative Fourier encoding
        relative_z_f = grouped_coords_f[:, :, 2:3] - local_z_anchor.unsqueeze(1)
        pe_z_four = self.z_rfe(relative_z_f)            # (Q, K, z_fourier_dim)

        local_z_std = torch.std(
            grouped_coords_f[:, :, 2], dim=1, unbiased=False, keepdim=True
        )  # (Q, 1)

        # 3-D relative position for Cross-GVA position modulation
        query_xyz = torch.cat([q_xy, local_z_anchor], dim=-1)          # (Q, 3)
        relative_pos_f = grouped_coords_f - query_xyz.unsqueeze(1)     # (Q, K, 3)

        # ------------------------------------------------------------------
        # 3. Cross-GVA fusion + FiLM modulation
        # ------------------------------------------------------------------
        neighbor_kv = torch.cat(
            [grouped_feats_f, grouped_class_embed, pe_z_four], dim=-1
        )  # (Q, K, kv_channels)

        F_query = self.cross_gva(
            q_feat=local_feat_anchor,
            grouped_kv_feat=neighbor_kv,
            relative_pos=relative_pos_f,
        )  # (Q, hidden_dim)

        S_terrain = torch.cat(
            [macro_z_range, macro_z_skew, local_z_std], dim=-1
        )  # (Q, 3)
        film_params = self.film_mlp(S_terrain)                # (Q, hidden_dim * 2)
        gamma, beta = torch.chunk(film_params, 2, dim=-1)     # each (Q, hidden_dim)

        F_modulated = gamma * F_query + beta                  # (Q, hidden_dim)
        residual = self.mlp(F_modulated)                      # (Q, num_targets)
        pred_z = local_z_anchor + residual                    # (Q, num_targets)

        # Predict normals if enabled
        pred_normal = None
        if self.predict_normals:
            raw_normal = self.normal_mlp(F_modulated)         # (Q, 3)
            pred_normal = F.normalize(raw_normal, p=2, dim=-1)  # unit normal

        if self.num_targets == 1:
            pred_z = pred_z.squeeze(-1)
            local_z_anchor = local_z_anchor.squeeze(-1)

        if self.training:
            if self.predict_normals:
                return pred_z, local_z_anchor, pred_normal    # 3-tuple
            return pred_z, local_z_anchor                     # 2-tuple
        return pred_z
