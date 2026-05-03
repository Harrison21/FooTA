"""Multimodal AQA model that fuses video features with team-graph features.

When *graph_nodes* is supplied the forward pass runs:
    video_feats --+
                  +-- fusion --> neck --> head --> predictions
    graph_feats --+

When *graph_nodes* is None the model degrades to the video-only path:
    video_feats -----------------> neck --> head --> predictions
"""

import torch
import torch.nn as nn

from models.graph import SpatioTemporalGraphEncoder
from models.fusion import VideoGraphFusion


class MultimodalAQAModel(nn.Module):
    """Wrapper that owns graph_encoder + fusion and delegates to neck + head.

    Args:
        graph_encoder: :class:`SpatioTemporalGraphEncoder` instance.
        fusion:        :class:`VideoGraphFusion` instance.
        neck:          TQN (or compatible) temporal aggregation module.
        head:          Evaluator_weighted (or compatible) prediction head.
    """

    def __init__(self, graph_encoder, fusion, neck, head):
        super().__init__()
        self.graph_encoder = graph_encoder
        self.fusion = fusion
        self.neck = neck
        self.head = head

    def forward(
        self,
        video_feats,
        graph_nodes=None,
        graph_mask=None,
        graph_frame_ids=None,
        train=False,
    ):
        """
        Args:
            video_feats:     [B, T, Dv]
            graph_nodes:     [B, T, N, 10]   (optional)
            graph_mask:      [B, T, N]        (optional)
            graph_frame_ids: [B, T, N]        (optional)
            train:           passed to neck.

        Returns:
            probs:     [B] or [B, output_dim]
            weight:    from head
            means:     from head
            var:       from head (always None in current head)
            attn:      (self_maps, cross_maps, memorys) from neck
        """
        if graph_nodes is not None:
            graph_feats = self.graph_encoder(
                graph_nodes, graph_mask, graph_frame_ids,
            )                                                  # [B, T, Dg]
            fused = self.fusion(video_feats, graph_feats)      # [B, T, D_out]
        else:
            fused = video_feats                                # [B, T, Dv]

        tgt_weight, attn = self.neck(fused, train=train)       # neck -> head
        probs, weight, means, var = self.head(tgt_weight)
        return probs, weight, means, var, attn
