import os
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import wandb

from utils.utils import parse_args, init_seed, init_gpu, get_logger
from dataloader import build_dataloader
from models import build_backbone, build_neck, build_head, build_graph_encoder, build_fusion
from run import run
from loss import AutomaticWeightedLoss
from torchinfo import summary


def init_wandb(cfg):
    if not bool(cfg.get("use_wandb", False)):
        return None

    os.makedirs("exp", exist_ok=True)
    cfg_dict = cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg)
    run_name = cfg.get(
        "wandb_run_name",
        f"{cfg.dataset_name}_{cfg.backbone}_{cfg.neck}_seed{cfg.seed}",
    )
    return wandb.init(
        project=cfg.get("wandb_project", "FootballAQA"),
        entity=cfg.get("wandb_entity", None),
        name=run_name,
        mode=cfg.get("wandb_mode", "online"),
        dir=cfg.get("wandb_dir", "exp"),
        config=cfg_dict,
        reinit=True,
    )


def main(cfg):

    # necessary modules
    init_seed(cfg)
    multi_gpu, device = init_gpu(cfg)

    # dataloader
    data_loaders = build_dataloader(cfg)

    # network
    use_video = bool(cfg.get("use_video_branch", True))
    backbone = build_backbone(cfg) if use_video else None
    neck = build_neck(cfg)
    head = build_head(cfg)
    video_projector = None

    # -- optional graph branch ---------------------------------------------
    use_graph = bool(cfg.get("use_graph_branch", False))
    # Output-ensemble mode (deprecated path kept for compatibility): combines
    # graph and video branch predictions with a fixed weight.
    late_fusion = bool(cfg.get("late_fusion", False))
    # Graph-only prediction + video auxiliary training loss.
    # Final / inference output = graph output.
    # Training:  L_total = L_graph + lambda_aux * L_video
    graph_aux_video = bool(cfg.get("graph_aux_video", False))
    # graph_aux_video takes priority over late_fusion - no output ensemble.
    if graph_aux_video and late_fusion:
        print("[Config] graph_aux_video=True overrides late_fusion=True; "
              "disabling output-level fusion.")
        late_fusion = False
    graph_encoder = None
    fusion = None
    video_neck = None
    video_head = None
    if use_graph:
        graph_encoder = build_graph_encoder(cfg)
        use_image = bool(cfg.get("use_image_features", False))
        if use_video:
            if graph_aux_video:
                # Independent video neck + head used only for an auxiliary
                # loss signal. Graph branch remains the sole final predictor.
                video_neck = build_neck(cfg)
                video_head = build_head(cfg)
                print(
                    f"[GraphAuxVideo] Enabled: lambda_aux="
                    f"{float(cfg.get('lambda_aux', 0.1)):.4f}, "
                    f"graph_dim={graph_encoder.output_dim}, "
                    f"video neck/head built from feature_dim="
                    f"{int(cfg.get('feature_dim', 576))}. "
                    f"Inference uses graph output only; no feature fusion, "
                    f"no output ensemble."
                )
            elif late_fusion:
                # Dedicated video-side neck + head (no feature-level fusion).
                video_neck = build_neck(cfg)
                video_head = build_head(cfg)
                print(
                    f"[LateFusion] Enabled: lambda_fuse="
                    f"{float(cfg.get('lambda_fuse', 0.05)):.4f}, "
                    f"graph_dim={graph_encoder.output_dim}, "
                    f"video neck/head built from feature_dim="
                    f"{int(cfg.get('feature_dim', 576))}"
                )
            else:
                fusion = build_fusion(cfg)
                print(f"[Multimodal] Graph encoder output_dim={graph_encoder.output_dim}, "
                      f"Fusion output_dim={cfg.get('fusion_output_dim', cfg.feature_dim)}")
        else:
            print(f"[GraphOnly] Graph encoder output_dim={graph_encoder.output_dim}")
        if use_image:
            print(f"[ImageFeatures] DINOv3 image features enabled: "
                  f"dim={cfg.get('image_feat_dim', 768)}, "
                  f"fusion={cfg.get('image_node_fusion', 'gated')}")

    # Optional explicit video projection (e.g. 576 -> 128) before fusion/neck.
    video_feat_dim = int(cfg.get("video_feature_dim", cfg.get("feature_dim", 576)))
    model_feat_dim = int(cfg.get("feature_dim", 576))
    use_video_projector = bool(cfg.get("video_project_before_fusion", True))
    if (not late_fusion and not graph_aux_video) and str(cfg.get("fusion_type", "concat")) == "graph_residual_gated":
        # This fusion has its own built-in video projection (video_feature_dim -> graph_dim).
        use_video_projector = False
    if late_fusion or graph_aux_video:
        # Both modes feed the video branch directly into its own neck built
        # with feature_dim, so video features must first be projected to
        # feature_dim.
        use_video_projector = True
    if use_video and use_video_projector and video_feat_dim != model_feat_dim:
        video_projector = nn.Sequential(
            nn.Linear(video_feat_dim, model_feat_dim),
            nn.GELU(),
            nn.LayerNorm(model_feat_dim),
        ).cuda()
        print(
            f"[VideoProjector] Enabled explicit projection: "
            f"{video_feat_dim} -> {model_feat_dim}"
        )

    if multi_gpu:
        if backbone is not None:
            backbone = nn.DataParallel(backbone)
        head = nn.DataParallel(head)
        neck = nn.DataParallel(neck)
        if graph_encoder is not None:
            graph_encoder = nn.DataParallel(graph_encoder)
        if fusion is not None:
            fusion = nn.DataParallel(fusion)
        if video_projector is not None:
            video_projector = nn.DataParallel(video_projector)
        if video_neck is not None:
            video_neck = nn.DataParallel(video_neck)
        if video_head is not None:
            video_head = nn.DataParallel(video_head)

    if cfg.get('load_from', None) and os.path.exists(cfg.load_from):
        ckpt = torch.load(cfg.load_from)
        neck.load_state_dict(ckpt["neck"])
        head.load_state_dict(ckpt["evaluator"])
        if use_graph and "graph_encoder" in ckpt:
            graph_encoder.load_state_dict(ckpt["graph_encoder"])
        if use_graph and "fusion" in ckpt:
            fusion.load_state_dict(ckpt["fusion"])
        if video_projector is not None and "video_projector" in ckpt:
            video_projector.load_state_dict(ckpt["video_projector"])

    # loss function
    mse = nn.MSELoss()
    kld = nn.KLDivLoss(reduction='sum')
    awl = AutomaticWeightedLoss(2, main_task_weight=5.0)

    # optimizer and scheduler
    param_groups = [
        {'params': neck.parameters(), 'lr': cfg.lr},
        {'params': head.parameters(), 'lr': cfg.lr},
        {'params': awl.parameters(), 'lr': cfg.lr},
    ]
    if backbone is not None:
        param_groups.insert(0, {'params': backbone.parameters(), 'lr': cfg.lr})
    if graph_encoder is not None:
        param_groups.append({'params': graph_encoder.parameters(), 'lr': cfg.lr})
    if fusion is not None:
        param_groups.append({'params': fusion.parameters(), 'lr': cfg.lr})
    if video_projector is not None:
        param_groups.append({'params': video_projector.parameters(), 'lr': cfg.lr})
    if video_neck is not None:
        param_groups.append({'params': video_neck.parameters(), 'lr': cfg.lr})
    if video_head is not None:
        param_groups.append({'params': video_head.parameters(), 'lr': cfg.lr})

    optimizer = torch.optim.Adam(param_groups, lr=cfg.lr, weight_decay=1e-5)
    lr_step = int(cfg.get("lr_step_size", 400))
    lr_gam = float(cfg.get("lr_gamma", 0.1))
    scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_gam)

    # log and wandb
    custom_log_name = cfg.get("log_name", "")
    if custom_log_name:
        base_logger = get_logger(f'exp/{custom_log_name}.log', "EXP1")
    else:
        base_logger = get_logger(f'exp/{cfg.seed}_{cfg.dataset_name}_{cfg.label}_{cfg.att_loss}_{cfg.query_var}_{cfg.pe}.log', "EXP1")
    wandb_run = init_wandb(cfg)

    network = [
        backbone, neck, head, graph_encoder, fusion, video_projector,
        video_neck, video_head,
    ]
    if multi_gpu:
        network = [
            backbone, neck, head, graph_encoder, fusion, video_projector,
            video_neck, video_head,
        ]
    try:
        run(cfg, base_logger, network, data_loaders, kld, mse, optimizer, scheduler, awl, wandb_run=wandb_run)
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == '__main__':
    cfg = parse_args()
    print(f"PE: {cfg.pe}, Query Var: {cfg.query_var}, Att Loss: {cfg.att_loss}, Dino Loss: {cfg.dino_loss}")
    print(f"Video branch: {cfg.get('use_video_branch', True)}")
    print(f"Graph branch: {cfg.get('use_graph_branch', False)}")
    print(f"Image features: {cfg.get('use_image_features', False)}")
    main(cfg)
