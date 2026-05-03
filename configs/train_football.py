seed = 42
multi_gpu = "0"

# --- dataset config -------------------------------------------------------
dataset_name = "football"

# Features, match stats, and bbox JSONs now live together under FootballAQA.
football_data_root = "../VideoMamba/FootballAQA"
football_leagues = [
    "Premier League",
    "LaLiga",
    "Bundesliga",
    "Ligue_1",
    "england_epl",
    "france_league",
]

football_dirs = [f"{football_data_root}/{league}" for league in football_leagues]
football_dir = football_dirs[0]

# Labels are resolved from the same match folders as the feature files.
football_meta_dirs = football_dirs

football_rounds = ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th"]

football_random_split = True
football_test_ratio = 0.2
football_split_seed = 42
football_score_strat_bins = 3

debug_dump_test_preds = True
debug_dump_every_n_epochs = 1
debug_dump_dir = "exp/v19_graph_only_mask_bbox_nogat/debug_test_preds"

train_rounds = ["1st", "2nd", "3rd", "4th", "5th", "6th"]
test_rounds = ["7th"]

football_max_clips = 256

label = "overall"
football_target_mode = "avg"

# --- dataloader config -----------------------------------------------------
subset = 0
bs_train = 8
bs_test = 8
num_workers = 4
prefetch_factor = 2
persistent_workers = True
football_cache_features = True
football_cache_graph_data = True
football_precompute_graph_data = True
football_mmap_features = False

# --- network config --------------------------------------------------------
backbone = "vivit"
use_video_branch = False
use_graph_branch = True

neck = "TQN"
head = "weighted"

feature_dim = 128
# Input dimensionality of pre-extracted video .npy features.
# Keep this at 576 for ViViT pooled features, while feature_dim stays 128
# for graph branch / neck / head.
video_feature_dim = 576
# For graph_residual_gated fusion, projection is inside fusion itself.
video_project_before_fusion = False
output_dim = 1

q_number = 256
query_var = 2
pe = "query_pe"
att_loss = True
dino_loss = True
dino_noise_std = 0.01
max_len = 256

num_layers = 2

# --- training config -------------------------------------------------------
split_feats = True
epoch_num = 400
lr = 1e-4
cudnn_deterministic = False
cudnn_benchmark = True

use_wandb = True
wandb_project = "FootballAQA"
wandb_run_name = "exp/v19_graph_only_mask_bbox_nogat"
wandb_mode = "online"

# --- checkpointing ---------------------------------------------------------
# `ckpt_name` sets the filename used when saving a new best checkpoint.
# `.pt` is appended automatically if missing.
log_name = "v19_graph_only_mask_bbox_nogat"
ckpt_dir = "ckpts"
ckpt_name = "football_graph_only_mask_bbox_nogat"

lr_step_size = 2000
lr_gamma = 0.5

# --- graph branch config ---------------------------------------------------
bbox_root = football_data_root

# Simulate partial camera coverage by masking one random side of bboxes
# at load time. One side is sampled per training sample from:
#   ["left", "right", "top", "bottom"].
football_camera_mask_enable = False
football_camera_mask_prob = 1.0
football_camera_mask_apply_on = "train"  # "train", "test", or "all"
football_camera_mask_sides = ["left", "right", "top", "bottom"]
football_camera_mask_ratio_min = 0.25
football_camera_mask_ratio_max = 0.40

graph_num_frames = 8
graph_stride = 4
graph_max_nodes_per_window = 120

# Opponent context features: node_input_dim = 14 (base 10 + opp 4)
#   dist_to_nearest_opp, rel_cx_opp, rel_cy_opp, n_opps_within_r
graph_use_opp_context = True
graph_opp_context_radius = 0.3   # normalised pitch coords (~1/5 of pitch width)
graph_node_input_dim = 14        # must match: 10 + 4 if use_opp_context else 10

graph_node_hidden_dim = 64
graph_gnn_hidden_dim = 128
graph_output_dim = 128
graph_num_gnn_layers = 2
graph_num_heads = 4
graph_dropout = 0.1
# When True (default), the graph encoder uses PyG GATConv layers if PyG is
# installed. Set to False to force the transformer-style fallback
# (masked multi-head self-attention over nodes) even when PyG is available.
graph_use_gat = False
graph_cache_edges = False
graph_edge_cache_size = 4096
graph_edge_style = "sparse_knn"
graph_spatial_k = 3
graph_temporal_k = 1
graph_pool_type = "residual_attn"
graph_pool_num_heads = 2
graph_pool_combine_type = "mean"

# --- graph pool wandb visualization ---------------------------------------
graph_pool_log_wandb = True
graph_pool_log_split = "test"
graph_pool_log_every_n_epochs = 1
graph_pool_log_max_windows = 4
graph_pool_log_max_nodes = 32

# Fusion module is unused in bbox-only mode, but kept for config completeness.
fusion_type = "graph_residual_gated"
fusion_output_dim = 128
fusion_mid_dim = 128
fusion_init_alpha = 0.01
fusion_dropout = 0.1
fusion_residual_scale = 0.1

# --- late (output-level) fusion -------------------------------------------
# When `late_fusion = True`, the graph and video branches run through
# independent neck+head stacks and only the final outputs are mixed with a
# fixed (non-learnable) weight. `feature_level` fusion above is bypassed.
#
#   regression:     y      = (1 - lambda_fuse) * y_graph      + lambda_fuse * y_video
#   logits/class.:  logits = (1 - lambda_fuse) * logits_graph + lambda_fuse * logits_video
#
# Defaults keep graph dominant (lambda_fuse=0.05).
late_fusion = False
lambda_fuse = 0.05

# --- graph-only prediction + video auxiliary loss -------------------------
# When `graph_aux_video = True`:
#   - Graph and video branches are independent (no feature fusion, no
#     output ensemble, no video->graph interaction).
#   - Final / inference output = graph output only.
#   - Video branch contributes only to the training loss:
#         L_total = L_graph + lambda_aux * L_video
#     where L_* is MSE for regression (output_dim==1) or cross-entropy
#     for logits/classification (output_dim>1).
# Overrides `late_fusion` when both are set.
graph_aux_video = False
lambda_aux = 0.0
