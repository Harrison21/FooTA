
def build_backbone(cfg):
    
    backbone = cfg.backbone
    
    if backbone == "i3d":
        from models.backbone.i3d import I3D
        return I3D().cuda()
    elif backbone == "vivit":
        from models.backbone.vivit import ViViT
        return ViViT(224, 16, 100, 103).cuda()
    else:
        raise ValueError(f'Unsupported dataset name [{backbone}]')

def build_neck(cfg):
    
    neck = cfg.neck
    feat_dim = cfg.get('feature_dim', 1024)
    
    if neck == "TQN":
        from models.neck.TQN import TQN
        tqn_dropout = float(cfg.get("tqn_dropout", 0.5))
        return TQN(feat_dim,cfg.q_number,cfg.query_var,cfg.pe,N=cfg.num_layers,max_len=cfg.max_len,dropout=tqn_dropout).cuda()
    elif neck == "TQT":
        from models.neck.TQT import ActionDecoder
        return ActionDecoder().cuda()
    else:
        raise ValueError(f'Unsupported dataset name [{neck}]')

def build_head(cfg):
    
    head = cfg.head
    feat_dim = cfg.get('feature_dim', 1024)
    output_dim = cfg.get('output_dim', 1)
    
    from models.head.evaluator import Evaluator_weighted
    return Evaluator_weighted(input_dim=feat_dim, output_dim=output_dim).cuda()


def build_graph_encoder(cfg):
    """Build :class:`SpatioTemporalGraphEncoder` from config."""
    from models.graph.graph_encoder import SpatioTemporalGraphEncoder
    return SpatioTemporalGraphEncoder(
        node_input_dim=int(cfg.get("graph_node_input_dim", 10)),
        node_hidden_dim=int(cfg.get("graph_node_hidden_dim", 64)),
        gnn_hidden_dim=int(cfg.get("graph_gnn_hidden_dim", 128)),
        output_dim=int(cfg.get("graph_output_dim", 128)),
        num_gnn_layers=int(cfg.get("graph_num_gnn_layers", 2)),
        num_heads=int(cfg.get("graph_num_heads", 4)),
        dropout=float(cfg.get("graph_dropout", 0.1)),
        cache_edges=bool(cfg.get("graph_cache_edges", True)),
        edge_cache_size=int(cfg.get("graph_edge_cache_size", 4096)),
        edge_style=str(cfg.get("graph_edge_style", "dense_full")),
        spatial_k=int(cfg.get("graph_spatial_k", 3)),
        temporal_k=int(cfg.get("graph_temporal_k", 1)),
        pool_type=str(cfg.get("graph_pool_type", "mean")),
        pool_num_heads=int(cfg.get("graph_pool_num_heads", 1)),
        pool_combine_type=str(cfg.get("graph_pool_combine_type", "mean")),
        use_image_features=bool(cfg.get("use_image_features", False)),
        image_feat_dim=int(cfg.get("image_feat_dim", 768)),
        image_node_hidden_dim=int(cfg.get("image_node_hidden_dim", 256)),
        image_node_fusion=str(cfg.get("image_node_fusion", "gated")),
        use_gat=bool(cfg.get("graph_use_gat", True)),
    ).cuda()


def build_fusion(cfg):
    """Build a fusion module from config.

    ``fusion_type`` in config selects the strategy:
        "concat"           -> VideoGraphFusion     (concat + MLP)
        "cross_attention"  -> CrossAttentionFusion  (temporal cross-attn)
        "gated"            -> GatedFusion           (FiLM-style gating)
    """
    fusion_type = str(cfg.get("fusion_type", "concat"))
    video_dim = int(cfg.get("feature_dim", 576))
    graph_dim = int(cfg.get("graph_output_dim", 128))
    output_dim = int(cfg.get("fusion_output_dim", video_dim))
    dropout = float(cfg.get("fusion_dropout", 0.3))

    if fusion_type == "cross_attention":
        from models.fusion import CrossAttentionFusion
        num_heads = int(cfg.get("fusion_num_heads", 4))
        return CrossAttentionFusion(
            video_dim=video_dim,
            graph_dim=graph_dim,
            output_dim=output_dim,
            num_heads=num_heads,
            dropout=dropout,
        ).cuda()
    elif fusion_type == "gated":
        from models.fusion import GatedFusion
        return GatedFusion(
            video_dim=video_dim,
            graph_dim=graph_dim,
            output_dim=output_dim,
            dropout=dropout,
        ).cuda()
    elif fusion_type == "balanced":
        from models.fusion import BalancedFusion
        mid_dim = int(cfg.get("fusion_mid_dim", 256))
        init_alpha = float(cfg.get("fusion_init_alpha", 0.5))
        return BalancedFusion(
            video_dim=video_dim,
            graph_dim=graph_dim,
            output_dim=output_dim,
            mid_dim=mid_dim,
            dropout=dropout,
            init_alpha=init_alpha,
        ).cuda()
    elif fusion_type == "graph_residual_gated":
        from models.fusion import GraphResidualGatedFusion
        # This fusion includes its own video projection: video_feature_dim -> graph_dim.
        video_in_dim = int(cfg.get("video_feature_dim", cfg.get("feature_dim", 576)))
        residual_scale = float(cfg.get("fusion_residual_scale", 0.1))
        return GraphResidualGatedFusion(
            video_dim=video_in_dim,
            graph_dim=graph_dim,
            output_dim=output_dim,
            dropout=dropout,
            residual_scale=residual_scale,
        ).cuda()
    else:
        from models.fusion import VideoGraphFusion
        return VideoGraphFusion(
            video_dim=video_dim,
            graph_dim=graph_dim,
            output_dim=output_dim,
            dropout=dropout,
        ).cuda()
