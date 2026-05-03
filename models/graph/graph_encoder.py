"""Spatio-temporal graph encoder for team player formations.

Pipeline per temporal window
    raw bbox features -> BoxToNodeEncoder -> GNN layers -> node pooling -> projection

Default graph topology:
    spatial edges:  fully-connected within each frame (all teammates)
    temporal edges: bidirectional between consecutive frames

Optional sparse topology:
    spatial edges: per-frame kNN in normalized (cx, cy)
    temporal edges: nearest-neighbor matching between adjacent frames

Uses PyTorch Geometric (GATConv + graph-level pooling) when available;
falls back to multi-head self-attention over nodes otherwise.
"""

import math
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATConv, global_mean_pool
    from torch_geometric.data import Data, Batch
    _HAS_PYG = True
except ImportError:
    _HAS_PYG = False

from .box_encoder import BoxToNodeEncoder


def has_pyg():
    return _HAS_PYG


# -----------------------------------------------------------------------
# Image feature encoder for per-player DINOv3 crops
# -----------------------------------------------------------------------

class ImageNodeEncoder(nn.Module):
    """Project high-dimensional image features (e.g. 768-d DINOv3) to the
    GNN hidden dimension via a two-layer MLP with LayerNorm."""

    def __init__(self, input_dim: int = 768, hidden_dim: int = 256, output_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
            nn.GELU(),
        )

    def forward(self, x):
        """x: [*, input_dim] -> [*, output_dim]"""
        return self.mlp(x)


class GatedNodeFusion(nn.Module):
    """Fuse position-based and image-based node embeddings via learned gating.

    For each node independently:
        gate = sigmoid(W [pos_embed || img_embed] + b)
        fused = gate * pos_embed + (1 - gate) * img_embed
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.gate_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid(),
        )

    def forward(self, pos_embed, img_embed):
        """pos_embed, img_embed: [*, D] -> fused: [*, D]"""
        combined = torch.cat([pos_embed, img_embed], dim=-1)
        gate = self.gate_proj(combined)
        return gate * pos_embed + (1.0 - gate) * img_embed


def _build_dense_spatio_temporal_edges(frame_ids, num_nodes):
    """Build edge_index [2, E] for spatial + temporal edges in one window.
    """
    if num_nodes == 0:
        return torch.zeros(2, 0, dtype=torch.long)

    num_frames = int(frame_ids.max().item()) + 1
    frame_nodes = [[] for _ in range(num_frames)]
    for n in range(num_nodes):
        frame_nodes[frame_ids[n].item()].append(n)

    src = []
    dst = []

    for nodes in frame_nodes:
        for i in nodes:
            for j in nodes:
                if i != j:
                    src.append(i)
                    dst.append(j)

    for f in range(num_frames - 1):
        for i in frame_nodes[f]:
            for j in frame_nodes[f + 1]:
                src.append(i); dst.append(j)
                src.append(j); dst.append(i)

    if not src:
        return torch.zeros(2, 0, dtype=torch.long)
    return torch.tensor([src, dst], dtype=torch.long)


def _pairwise_sq_dists(points_a, points_b):
    diff = points_a.unsqueeze(1) - points_b.unsqueeze(0)
    return (diff * diff).sum(dim=-1)


def _build_sparse_spatial_edges(frame_nodes, positions, spatial_k):
    edges = set()
    k = max(int(spatial_k), 0)
    if k <= 0:
        return edges

    for nodes in frame_nodes:
        num_frame_nodes = len(nodes)
        if num_frame_nodes <= 1:
            continue

        frame_pos = positions[nodes]
        dists = _pairwise_sq_dists(frame_pos, frame_pos)
        dists.fill_diagonal_(float("inf"))
        k_eff = min(k, num_frame_nodes - 1)
        nn_idx = torch.topk(dists, k=k_eff, dim=1, largest=False).indices
        for src_local, nbrs in enumerate(nn_idx):
            src = nodes[src_local]
            for dst_local in nbrs.tolist():
                edges.add((src, nodes[dst_local]))
    return edges


def _build_sparse_temporal_edges(frame_nodes, positions, temporal_k):
    edges = set()
    k = max(int(temporal_k), 0)
    if k <= 0 or len(frame_nodes) <= 1:
        return edges

    for f in range(len(frame_nodes) - 1):
        src_nodes = frame_nodes[f]
        dst_nodes = frame_nodes[f + 1]
        if not src_nodes or not dst_nodes:
            continue

        src_pos = positions[src_nodes]
        dst_pos = positions[dst_nodes]
        dists = _pairwise_sq_dists(src_pos, dst_pos)
        k_eff = min(k, len(dst_nodes))
        nn_idx = torch.topk(dists, k=k_eff, dim=1, largest=False).indices
        for src_local, nbrs in enumerate(nn_idx):
            src = src_nodes[src_local]
            for dst_local in nbrs.tolist():
                dst = dst_nodes[dst_local]
                edges.add((src, dst))
                edges.add((dst, src))
    return edges


def build_spatio_temporal_edges(
    frame_ids,
    num_nodes,
    node_positions=None,
    edge_style="dense_full",
    spatial_k=3,
    temporal_k=1,
):
    """Build edge_index [2, E] for one window."""
    if edge_style == "dense_full":
        return _build_dense_spatio_temporal_edges(frame_ids, num_nodes)

    if num_nodes == 0:
        return torch.zeros(2, 0, dtype=torch.long)
    if node_positions is None:
        raise ValueError("node_positions is required when edge_style != 'dense_full'")

    num_frames = int(frame_ids.max().item()) + 1
    frame_nodes = [[] for _ in range(num_frames)]
    for n in range(num_nodes):
        frame_nodes[frame_ids[n].item()].append(n)

    positions = node_positions[:num_nodes].float().cpu()
    edges = set()
    edges.update(_build_sparse_spatial_edges(frame_nodes, positions, spatial_k))
    edges.update(_build_sparse_temporal_edges(frame_nodes, positions, temporal_k))

    if not edges:
        return torch.zeros(2, 0, dtype=torch.long)

    edge_index = torch.tensor(sorted(edges), dtype=torch.long).t().contiguous()
    return edge_index


# -----------------------------------------------------------------------
# Pure-PyTorch multi-head self-attention block (fallback when PyG absent)
# -----------------------------------------------------------------------

class _MaskedMultiHeadAttention(nn.Module):
    """Standard multi-head self-attention with a boolean node mask."""

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = math.sqrt(self.head_dim)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        """
        Args:
            x:    [B, N, D]
            mask: [B, N] bool  True = valid node
        Returns:
            out:  [B, N, D]
        """
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)       # [3, B, H, N, Dk]
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # [B, H, N, N]

        # mask: [B, N] -> [B, 1, 1, N]  invalid positions get -inf
        attn_mask = (~mask).unsqueeze(1).unsqueeze(2)              # [B, 1, 1, N]
        attn = attn.masked_fill(attn_mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = attn.masked_fill(attn.isnan(), 0.0)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)              # [B, H, N, Dk]
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.out_proj(out)


class _FallbackGNNLayer(nn.Module):
    """One transformer-style block: self-attention + FFN, with node mask."""

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.attn = _MaskedMultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x


class _AttentionPool(nn.Module):
    """Learnable attention pooling over nodes inside one graph/window."""

    def __init__(self, hidden_dim, num_pool_heads=1, combine_type="mean"):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_pool_heads = int(num_pool_heads)
        self.combine_type = str(combine_type)
        if self.num_pool_heads <= 0:
            raise ValueError(f"num_pool_heads must be positive, got {self.num_pool_heads}")
        if self.combine_type not in {"mean", "linear"}:
            raise ValueError(f"Unsupported combine_type: {self.combine_type}")

        self.score_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.Tanh(),
                    nn.Linear(hidden_dim, 1),
                )
                for _ in range(self.num_pool_heads)
            ]
        )
        self.combine_proj = (
            nn.Linear(hidden_dim * self.num_pool_heads, hidden_dim)
            if self.combine_type == "linear" and self.num_pool_heads > 1
            else None
        )

    def _combine_heads(self, pooled_heads):
        if self.num_pool_heads == 1:
            return pooled_heads[0]
        if self.combine_type == "mean":
            return torch.stack(pooled_heads, dim=0).mean(dim=0)
        return self.combine_proj(torch.cat(pooled_heads, dim=-1))

    def forward_masked(self, x, mask, return_info=False):
        """Pool padded node tensors with a boolean validity mask."""
        pooled_heads = []
        weight_heads = []
        for score in self.score_heads:
            logits = score(x).squeeze(-1)                   # [BT, N]
            logits = logits.masked_fill(~mask, float("-inf"))
            weights = torch.softmax(logits, dim=1)
            weights = weights.masked_fill(weights.isnan(), 0.0)
            weight_heads.append(weights)
            pooled_heads.append((x * weights.unsqueeze(-1)).sum(dim=1))  # [BT, H]
        pooled = self._combine_heads(pooled_heads)
        if not return_info:
            return pooled
        return pooled, {
            "weights": torch.stack(weight_heads, dim=0),   # [H_pool, BT, N]
            "valid_counts": mask.sum(dim=1),               # [BT]
        }

    def forward_segmented(self, x, ptr, return_info=False):
        """Pool concatenated node tensors using PyG batch ptr offsets."""
        pooled_heads = []
        padded_weight_heads = []
        valid_counts = [(end - start) for start, end in zip(ptr[:-1].tolist(), ptr[1:].tolist())]
        max_nodes = max(valid_counts) if valid_counts else 0
        for score in self.score_heads:
            pooled = []
            padded_weights = []
            for start, end in zip(ptr[:-1].tolist(), ptr[1:].tolist()):
                graph_x = x[start:end]
                logits = score(graph_x).squeeze(-1)         # [num_nodes]
                weights = torch.softmax(logits, dim=0)
                pooled.append((graph_x * weights.unsqueeze(-1)).sum(dim=0))
                if return_info:
                    padded = torch.zeros(max_nodes, device=weights.device, dtype=weights.dtype)
                    padded[:weights.shape[0]] = weights
                    padded_weights.append(padded)
            pooled_heads.append(torch.stack(pooled, dim=0))  # [num_graphs, H]
            if return_info:
                padded_weight_heads.append(torch.stack(padded_weights, dim=0))
        pooled = self._combine_heads(pooled_heads)
        if not return_info:
            return pooled
        return pooled, {
            "weights": torch.stack(padded_weight_heads, dim=0),  # [H_pool, num_graphs, max_nodes]
            "valid_counts": torch.tensor(valid_counts, device=x.device, dtype=torch.long),
        }


# -----------------------------------------------------------------------

class SpatioTemporalGraphEncoder(nn.Module):
    """Encode player bounding-box windows into per-window feature vectors.

    Args:
        node_input_dim:  raw feature dim per node (default 10).
        node_hidden_dim: output of BoxToNodeEncoder.
        gnn_hidden_dim:  hidden width of GNN layers (also GAT head_dim * heads).
        output_dim:      final graph-feature dimension per window.
        num_gnn_layers:  depth of the GNN stack.
        num_heads:       number of GAT attention heads (PyG path only).
        dropout:         dropout rate inside GNN layers.
    """

    def __init__(
        self,
        node_input_dim=10,
        node_hidden_dim=64,
        gnn_hidden_dim=128,
        output_dim=128,
        num_gnn_layers=2,
        num_heads=4,
        dropout=0.1,
        cache_edges=True,
        edge_cache_size=4096,
        edge_style="dense_full",
        spatial_k=3,
        temporal_k=1,
        pool_type="mean",
        pool_num_heads=1,
        pool_combine_type="mean",
        # Image feature parameters
        use_image_features=False,
        image_feat_dim=768,
        image_node_hidden_dim=256,
        image_node_fusion="gated",
        use_gat=True,
    ):
        super().__init__()
        self.output_dim = output_dim
        # `use_gat=False` forces the transformer-style fallback even when
        # PyTorch Geometric is available. This lets us A/B test GATConv vs
        # masked multi-head self-attention without uninstalling PyG.
        self.use_gat = bool(use_gat)
        self.use_pyg = _HAS_PYG and self.use_gat
        self.edge_style = str(edge_style)
        self.spatial_k = int(spatial_k)
        self.temporal_k = int(temporal_k)
        self.pool_type = str(pool_type)
        if self.pool_type not in {"mean", "attn", "residual_attn"}:
            raise ValueError(f"Unsupported pool_type: {self.pool_type}")
        self.pool_num_heads = int(pool_num_heads)
        self.pool_combine_type = str(pool_combine_type)
        self.cache_edges = bool(cache_edges) and self.edge_style == "dense_full"
        self.edge_cache_size = int(edge_cache_size)
        self._edge_index_cache = OrderedDict()
        self.latest_pool_info = None
        self.node_encoder = BoxToNodeEncoder(node_input_dim, node_hidden_dim, gnn_hidden_dim)

        # -- optional image feature branch ------------------------------------
        self.use_image_features = bool(use_image_features)
        self.image_node_fusion_type = str(image_node_fusion)
        if self.use_image_features:
            self.image_encoder = ImageNodeEncoder(
                input_dim=int(image_feat_dim),
                hidden_dim=int(image_node_hidden_dim),
                output_dim=gnn_hidden_dim,
            )
            if self.image_node_fusion_type == "gated":
                self.node_fusion = GatedNodeFusion(gnn_hidden_dim)
            elif self.image_node_fusion_type == "add":
                self.node_fusion = None
            else:
                raise ValueError(f"Unsupported image_node_fusion: {self.image_node_fusion_type}")

        if self.use_pyg:
            self.gnn_layers = nn.ModuleList()
            self.norms = nn.ModuleList()
            for i in range(num_gnn_layers):
                in_dim = gnn_hidden_dim
                self.gnn_layers.append(
                    GATConv(
                        in_dim,
                        gnn_hidden_dim // num_heads,
                        heads=num_heads,
                        dropout=dropout,
                        concat=True,
                    )
                )
                self.norms.append(nn.LayerNorm(gnn_hidden_dim))
        else:
            self.gnn_layers = nn.ModuleList()
            for _ in range(num_gnn_layers):
                self.gnn_layers.append(
                    _FallbackGNNLayer(gnn_hidden_dim, num_heads, dropout)
                )

        self.pool = (
            _AttentionPool(
                gnn_hidden_dim,
                num_pool_heads=self.pool_num_heads,
                combine_type=self.pool_combine_type,
            )
            if self.pool_type in {"attn", "residual_attn"}
            else None
        )
        self.output_proj = nn.Linear(gnn_hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    # ------------------------------------------------------------------
    def _pool_segmented_mean(self, x, batch_index):
        return global_mean_pool(x, batch_index)

    def _pool_masked_mean(self, x, mask):
        mask_f = mask.unsqueeze(-1).float()
        x = x * mask_f
        return x.sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)

    def _pool_segmented(self, x, ptr, batch_index):
        mean_pooled = self._pool_segmented_mean(x, batch_index)
        self.latest_pool_info = None
        if self.pool_type == "mean":
            return mean_pooled

        attn_pooled, pool_info = self.pool.forward_segmented(x, ptr, return_info=True)
        pool_info["mean_pooled"] = mean_pooled.detach().cpu()
        pool_info["attn_pooled"] = attn_pooled.detach().cpu()
        self.latest_pool_info = {
            key: value.detach().cpu() if torch.is_tensor(value) else value
            for key, value in pool_info.items()
        }
        if self.pool_type == "attn":
            return attn_pooled
        return mean_pooled + attn_pooled

    def _pool_masked(self, x, mask):
        mean_pooled = self._pool_masked_mean(x, mask)
        self.latest_pool_info = None
        if self.pool_type == "mean":
            return mean_pooled

        attn_pooled, pool_info = self.pool.forward_masked(x, mask, return_info=True)
        pool_info["mean_pooled"] = mean_pooled.detach().cpu()
        pool_info["attn_pooled"] = attn_pooled.detach().cpu()
        self.latest_pool_info = {
            key: value.detach().cpu() if torch.is_tensor(value) else value
            for key, value in pool_info.items()
        }
        if self.pool_type == "attn":
            return attn_pooled
        return mean_pooled + attn_pooled

    def consume_latest_pool_info(self):
        pool_info = self.latest_pool_info
        self.latest_pool_info = None
        return pool_info

    # ------------------------------------------------------------------
    def forward(self, graph_nodes, graph_mask, graph_frame_ids,
                graph_image_feats=None):
        """
        Args:
            graph_nodes:       [B, T, N_max, D_pos]  padded raw node features
            graph_mask:        [B, T, N_max]          bool - True = valid node
            graph_frame_ids:   [B, T, N_max]          long - frame offset (0..7)
            graph_image_feats: [B, T, N_max, D_img]   (optional) DINOv3 features

        Returns:
            graph_feats: [B, T, output_dim]
        """
        B, T, N, D = graph_nodes.shape
        device = graph_nodes.device

        if self.use_pyg:
            return self._forward_pyg(
                graph_nodes, graph_mask, graph_frame_ids,
                B, T, N, D, device, graph_image_feats,
            )
        return self._forward_fallback(
            graph_nodes, graph_mask, B, T, N, D, device, graph_image_feats,
        )

    def _get_cached_edge_index(self, frame_ids, node_positions, valid, device):
        if valid <= 0:
            return torch.zeros(2, 0, dtype=torch.long, device=device)

        frame_ids_cpu = frame_ids[:valid].detach().to("cpu")
        cache_key = tuple(int(x) for x in frame_ids_cpu.tolist())
        if self.cache_edges:
            cached = self._edge_index_cache.get(cache_key)
            if cached is not None:
                self._edge_index_cache.move_to_end(cache_key)
                return cached.to(device=device, non_blocking=True)

        edge_index = build_spatio_temporal_edges(
            frame_ids_cpu,
            valid,
            node_positions=node_positions[:valid].detach(),
            edge_style=self.edge_style,
            spatial_k=self.spatial_k,
            temporal_k=self.temporal_k,
        )
        if self.cache_edges:
            self._edge_index_cache[cache_key] = edge_index
            if len(self._edge_index_cache) > self.edge_cache_size:
                self._edge_index_cache.popitem(last=False)
        return edge_index.to(device=device, non_blocking=True)

    # ------------------------------------------------------------------
    def _forward_pyg(self, graph_nodes, graph_mask, graph_frame_ids,
                     B, T, N, D, device, graph_image_feats=None):
        has_img = self.use_image_features and graph_image_feats is not None
        graphs = []
        for b in range(B):
            for t in range(T):
                m = graph_mask[b, t]          # [N]
                valid = int(m.sum().item())
                if valid == 0:
                    x = torch.zeros(1, D, device=device)
                    ei = torch.zeros(2, 0, dtype=torch.long, device=device)
                    data_obj = Data(x=x, edge_index=ei)
                    if has_img:
                        data_obj.x_img = torch.zeros(
                            1, graph_image_feats.shape[-1], device=device,
                        )
                else:
                    x = graph_nodes[b, t, m]  # [valid, D]
                    fids = graph_frame_ids[b, t, m]
                    pos = x[:, :2]
                    ei = self._get_cached_edge_index(fids, pos, valid, device)
                    data_obj = Data(x=x, edge_index=ei)
                    if has_img:
                        data_obj.x_img = graph_image_feats[b, t, m]
                graphs.append(data_obj)

        batch = Batch.from_data_list(graphs)
        x_pos = self.node_encoder(batch.x)                   # [total_nodes, H]

        if has_img:
            x_img = self.image_encoder(batch.x_img)           # [total_nodes, H]
            if self.node_fusion is not None:
                x = self.node_fusion(x_pos, x_img)
            else:
                x = x_pos + x_img
        else:
            x = x_pos

        for gnn, norm in zip(self.gnn_layers, self.norms):
            x = gnn(x, batch.edge_index)
            x = norm(x)
            x = torch.relu(x)
            x = self.dropout(x)

        pooled = self._pool_segmented(x, batch.ptr, batch.batch)  # [B*T, gnn_hidden]
        out = self.output_proj(pooled)                        # [B*T, output_dim]
        return out.view(B, T, -1)

    # ------------------------------------------------------------------
    def _forward_fallback(self, graph_nodes, graph_mask, B, T, N, D, device,
                          graph_image_feats=None):
        """Multi-head self-attention over nodes when PyG is not installed."""
        has_img = self.use_image_features and graph_image_feats is not None
        flat_nodes = graph_nodes.view(B * T, N, D)           # [BT, N, D_pos]
        flat_mask = graph_mask.view(B * T, N)                # [BT, N]

        x_pos = self.node_encoder(flat_nodes)                # [BT, N, H]

        if has_img:
            flat_img = graph_image_feats.view(B * T, N, -1)  # [BT, N, D_img]
            x_img = self.image_encoder(flat_img)             # [BT, N, H]
            if self.node_fusion is not None:
                x = self.node_fusion(x_pos, x_img)
            else:
                x = x_pos + x_img
        else:
            x = x_pos

        for layer in self.gnn_layers:
            x = layer(x, flat_mask)                          # [BT, N, H]

        pooled = self._pool_masked(x, flat_mask)              # [BT, H]
        out = self.output_proj(pooled)                       # [BT, output_dim]
        return out.view(B, T, -1)
