"""Fusion modules for combining video features and graph features.

Four strategies (select via config ``fusion_type``):
  1. ``VideoGraphFusion``       -- concat + MLP  (baseline)
  2. ``CrossAttentionFusion``   -- temporal cross-attention (high capacity,
                                   prone to overfitting on small datasets)
  3. ``GatedFusion``            -- graph features produce a sigmoid gate on
                                   video features (position-wise, low param count,
                                   recommended for small datasets)
  4. ``BalancedFusion``         -- projects both branches to the same dimension,
                                   L2-normalizes, then combines with a learnable
                                   mixing weight so neither branch dominates.

All include a learnable team-side embedding (home / away).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# 1. Concat + MLP  (original baseline)
# ======================================================================

class VideoGraphFusion(nn.Module):
    """Fuse video and graph features via concat -> MLP.

    Args:
        video_dim:  dimension of per-window video features  (e.g. 576).
        graph_dim:  dimension of per-window graph features  (e.g. 128).
        output_dim: fused feature dimension fed to the neck (e.g. 576).
        dropout:    dropout rate in the projection MLP.
        use_team_embed: if True, add a learnable home/away embedding (dim = graph_dim).
    """

    def __init__(
        self,
        video_dim=576,
        graph_dim=128,
        output_dim=576,
        dropout=0.3,
        use_team_embed=True,
    ):
        super().__init__()
        self.use_team_embed = use_team_embed
        if use_team_embed:
            self.team_embed = nn.Embedding(2, graph_dim)

        concat_dim = video_dim + graph_dim
        self.mlp = nn.Sequential(
            nn.Linear(concat_dim, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, video_feats, graph_feats, team_side_ids=None):
        """
        Args:
            video_feats:   [B, T, Dv]
            graph_feats:   [B, T, Dg]
            team_side_ids: [B] long tensor, 0=home 1=away (optional)
        Returns:
            fused: [B, T, D_out]
        """
        if self.use_team_embed and team_side_ids is not None:
            te = self.team_embed(team_side_ids).unsqueeze(1)
            graph_feats = graph_feats + te

        x = torch.cat([video_feats, graph_feats], dim=-1)
        return self.mlp(x)


# ======================================================================
# 2. Cross-Attention Fusion  (high capacity -- use with caution)
# ======================================================================

class CrossAttentionFusion(nn.Module):
    """Fuse video and graph features via multi-head cross-attention.

    WARNING: This does T x T temporal cross-attention which is very
    high-capacity and will overfit on small datasets (< 500 samples).
    Prefer ``GatedFusion`` for small datasets.
    """

    def __init__(
        self,
        video_dim=576,
        graph_dim=128,
        output_dim=576,
        num_heads=4,
        dropout=0.3,
        use_team_embed=True,
    ):
        super().__init__()
        self.video_dim = video_dim
        self.graph_dim = graph_dim
        self.num_heads = num_heads
        self.use_team_embed = use_team_embed

        if use_team_embed:
            self.team_embed = nn.Embedding(2, graph_dim)

        self.d_attn = video_dim
        assert self.d_attn % num_heads == 0
        self.head_dim = self.d_attn // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(video_dim, self.d_attn)
        self.k_proj = nn.Linear(graph_dim, self.d_attn)
        self.v_proj = nn.Linear(graph_dim, self.d_attn)
        self.out_proj = nn.Linear(self.d_attn, video_dim)
        self.attn_drop = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(video_dim)
        self.norm_kv = nn.LayerNorm(graph_dim)
        self.norm2 = nn.LayerNorm(video_dim)

        self.ffn = nn.Sequential(
            nn.Linear(video_dim, video_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(video_dim * 2, video_dim),
            nn.Dropout(dropout),
        )

        self.final_proj = (
            nn.Linear(video_dim, output_dim)
            if output_dim != video_dim
            else nn.Identity()
        )

    def forward(self, video_feats, graph_feats, team_side_ids=None):
        if self.use_team_embed and team_side_ids is not None:
            te = self.team_embed(team_side_ids).unsqueeze(1)
            graph_feats = graph_feats + te

        B, T, _ = video_feats.shape

        q = self.q_proj(self.norm1(video_feats))
        kv_in = self.norm_kv(graph_feats)
        k = self.k_proj(kv_in)
        v = self.v_proj(kv_in)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, T, self.d_attn)
        out = self.out_proj(out)

        x = video_feats + out
        x = x + self.ffn(self.norm2(x))
        return self.final_proj(x)


# ======================================================================
# 3. Gated Fusion  (recommended for small datasets)
# ======================================================================

class GatedFusion(nn.Module):
    """Graph-conditioned gating on video features (FiLM-style).

    At each temporal position independently:
      1.  Project graph features to produce a *gate* (sigmoid) and
          a *shift* (bias) in the video feature space.
      2.  Modulate video features:  out = video * gate + shift
      3.  Pass through a small MLP to produce the final output.

    This is position-wise (no T x T attention), so it has very few
    learnable parameters and generalises well on small datasets.

    Args:
        video_dim:      dimension of video features   (e.g. 576).
        graph_dim:      dimension of graph features   (e.g. 128).
        output_dim:     output dimension (usually == video_dim).
        dropout:        dropout rate.
        use_team_embed: add learnable home/away embedding to graph tokens.
    """

    def __init__(
        self,
        video_dim=576,
        graph_dim=128,
        output_dim=576,
        dropout=0.3,
        use_team_embed=True,
    ):
        super().__init__()
        self.use_team_embed = use_team_embed
        if use_team_embed:
            self.team_embed = nn.Embedding(2, graph_dim)

        # Graph -> gate (sigmoid) and shift (additive bias) in video space
        self.gate_proj = nn.Sequential(
            nn.Linear(graph_dim, video_dim),
            nn.Sigmoid(),
        )
        self.shift_proj = nn.Linear(graph_dim, video_dim)

        self.norm_v = nn.LayerNorm(video_dim)
        self.norm_g = nn.LayerNorm(graph_dim)

        # Small refinement MLP after modulation
        self.mlp = nn.Sequential(
            nn.Linear(video_dim, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, video_feats, graph_feats, team_side_ids=None):
        """
        Args:
            video_feats:   [B, T, Dv]
            graph_feats:   [B, T, Dg]
            team_side_ids: [B] long, 0=home 1=away (optional)
        Returns:
            fused: [B, T, D_out]
        """
        if self.use_team_embed and team_side_ids is not None:
            te = self.team_embed(team_side_ids).unsqueeze(1)
            graph_feats = graph_feats + te

        g = self.norm_g(graph_feats)                            # [B, T, Dg]
        gate = self.gate_proj(g)                                # [B, T, Dv]  in (0,1)
        shift = self.shift_proj(g)                              # [B, T, Dv]

        v = self.norm_v(video_feats)                            # [B, T, Dv]
        modulated = v * gate + shift                            # [B, T, Dv]

        # residual: keep original video signal dominant
        x = video_feats + modulated                             # [B, T, Dv]
        return self.mlp(x)                                      # [B, T, D_out]


# ======================================================================
# 4. Balanced Fusion  (equal branch representation)
# ======================================================================

class BalancedFusion(nn.Module):
    """Project both branches to the same dimension, normalize, then mix.

    Solves the problem where concat fusion lets the higher-dimensional
    branch (video 576-d) dominate over the lower-dimensional branch
    (graph 128-d).

    Pipeline:
      1. Project video (Dv -> D_mid) and graph (Dg -> D_mid) separately.
      2. L2-normalize both projections so they have equal magnitude.
      3. Combine via learnable mixing weight:
             fused = alpha * video_proj + (1-alpha) * graph_proj
         where alpha = sigmoid(learnable_logit), initialized at 0.5.
      4. Refinement MLP -> output_dim.

    Args:
        video_dim:      input video feature dimension  (e.g. 576).
        graph_dim:      input graph feature dimension   (e.g. 128).
        output_dim:     output fused dimension          (e.g. 576).
        mid_dim:        shared projection dimension     (e.g. 256).
        dropout:        dropout rate.
        init_alpha:     initial mixing weight for video branch (0-1).
                        0.5 = equal; lower = more graph influence.
        use_team_embed: add learnable home/away embedding.
    """

    def __init__(
        self,
        video_dim=576,
        graph_dim=128,
        output_dim=576,
        mid_dim=256,
        dropout=0.3,
        init_alpha=0.5,
        use_team_embed=True,
    ):
        super().__init__()
        self.use_team_embed = use_team_embed
        if use_team_embed:
            self.team_embed = nn.Embedding(2, graph_dim)

        self.video_proj = nn.Sequential(
            nn.Linear(video_dim, mid_dim),
            nn.GELU(),
            nn.LayerNorm(mid_dim),
        )
        self.graph_proj = nn.Sequential(
            nn.Linear(graph_dim, mid_dim),
            nn.GELU(),
            nn.LayerNorm(mid_dim),
        )

        # Learnable logit: sigmoid(logit) = alpha (video weight)
        init_logit = math.log(init_alpha / (1.0 - init_alpha + 1e-8))
        self.mix_logit = nn.Parameter(torch.tensor(init_logit))

        self.mlp = nn.Sequential(
            nn.Linear(mid_dim, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, video_feats, graph_feats, team_side_ids=None):
        """
        Args:
            video_feats:   [B, T, Dv]
            graph_feats:   [B, T, Dg]
            team_side_ids: [B] long tensor, 0=home 1=away (optional)
        Returns:
            fused: [B, T, D_out]
        """
        if self.use_team_embed and team_side_ids is not None:
            te = self.team_embed(team_side_ids).unsqueeze(1)
            graph_feats = graph_feats + te

        v = self.video_proj(video_feats)                       # [B, T, D_mid]
        g = self.graph_proj(graph_feats)                       # [B, T, D_mid]

        # L2-normalize so both branches have equal magnitude
        v = F.normalize(v, dim=-1)
        g = F.normalize(g, dim=-1)

        alpha = torch.sigmoid(self.mix_logit)                  # scalar in (0, 1)
        fused = alpha * v + (1.0 - alpha) * g                 # [B, T, D_mid]

        return self.mlp(fused)                                 # [B, T, D_out]


# ======================================================================
# 5. Graph-Base Gated Residual Fusion (graph main, video residual)
# ======================================================================

class GraphResidualGatedFusion(nn.Module):
    """Graph-centric fusion with a small gated video residual.

    Formulation:
        v' = video_proj(v)
        alpha = sigmoid(Wg * g + Wv * v' + b)
        h = g + lambda * alpha * v'

    This keeps graph features as the base signal while allowing video to
    provide a controlled correction.
    """

    def __init__(
        self,
        video_dim=576,
        graph_dim=128,
        output_dim=128,
        dropout=0.1,
        residual_scale=0.1,
        use_team_embed=True,
    ):
        super().__init__()
        self.use_team_embed = use_team_embed
        self.residual_scale = float(residual_scale)
        if use_team_embed:
            self.team_embed = nn.Embedding(2, graph_dim)

        # v' = phi(v): MLP(video_dim -> graph_dim)
        self.video_proj = nn.Sequential(
            nn.Linear(video_dim, graph_dim),
            nn.GELU(),
            nn.LayerNorm(graph_dim),
            nn.Dropout(dropout),
        )

        # alpha = sigmoid(Wg g + Wv v' + b)
        self.gate_g = nn.Linear(graph_dim, graph_dim)
        self.gate_v = nn.Linear(graph_dim, graph_dim)
        self.gate_bias = nn.Parameter(torch.zeros(graph_dim))

        # Optional output projection (identity when output_dim==graph_dim)
        self.out_proj = (
            nn.Sequential(
                nn.LayerNorm(graph_dim),
                nn.Linear(graph_dim, output_dim),
            )
            if output_dim != graph_dim
            else nn.Identity()
        )

    def forward(self, video_feats, graph_feats, team_side_ids=None):
        if self.use_team_embed and team_side_ids is not None:
            te = self.team_embed(team_side_ids).unsqueeze(1)  # [B,1,Dg]
            graph_feats = graph_feats + te

        v_proj = self.video_proj(video_feats)  # [B,T,Dg]
        alpha = torch.sigmoid(
            self.gate_g(graph_feats) + self.gate_v(v_proj) + self.gate_bias
        )  # [B,T,Dg]

        fused = graph_feats + self.residual_scale * alpha * v_proj
        return self.out_proj(fused)
