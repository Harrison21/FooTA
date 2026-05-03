import argparse
import csv
import glob
import json
import os
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from mmengine.config import Config

from loss import cal_spearmanr_rl2
from models.graph.box_encoder import compute_box_features_for_window
from models.graph.graph_encoder import SpatioTemporalGraphEncoder
from models.head.evaluator import Evaluator_weighted
from models.neck.TQN import TQN


@dataclass
class RealWorldSample:
    match_id: str
    match_dir: str
    stats_path: str
    bbox_path: str
    home_team: str
    away_team: str
    home_score: float
    away_score: float


def parse_args():
    parser = argparse.ArgumentParser(
        description="Zero-shot graph-only inference on real-world FootballAQA data."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional experiment config used to build the graph model and resolve checkpoint.",
    )
    parser.add_argument("--data-root", default="real_word_data")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint path. If omitted, falls back to cfg.load_from when --config is set.",
    )
    parser.add_argument("--output", default="exp/real_world_zero_shot/preds.csv")
    parser.add_argument(
        "--protocol",
        choices=("sampled", "indexed"),
        default="sampled",
        help=(
            "sampled treats extracted frames as a contiguous sequence. indexed uses "
            "raw frame_idx indexing and is mainly a temporal-sampling diagnostic."
        ),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-clips", type=int, default=256)
    parser.add_argument(
        "--window-sampling",
        choices=("uniform", "balanced"),
        default=None,
        help=(
            "uniform keeps the original evenly-spaced temporal sampling. balanced "
            "splits time into even bins and selects the most home/away-balanced "
            "window inside each bin."
        ),
    )
    parser.add_argument(
        "--balanced-min-side-boxes",
        type=int,
        default=None,
        help=(
            "Minimum total boxes required for each side in a candidate balanced "
            "window. If too few windows match, the selector falls back to all windows."
        ),
    )
    parser.add_argument("--graph-num-frames", type=int, default=None)
    parser.add_argument("--graph-stride", type=int, default=None)
    parser.add_argument("--graph-max-nodes", type=int, default=None)
    parser.add_argument("--opp-context-radius", type=float, default=None)
    parser.add_argument(
        "--graph-motion-dt-normalize",
        dest="graph_motion_dt_normalize",
        action="store_true",
        default=None,
        help="Normalize graph motion features by true sampled-frame dt.",
    )
    parser.add_argument(
        "--no-graph-motion-dt-normalize",
        dest="graph_motion_dt_normalize",
        action="store_false",
        help="Disable graph motion dt normalization even if config enables it.",
    )
    parser.add_argument(
        "--graph-motion-dt-mode",
        choices=("all", "dxdy"),
        default=None,
        help="Which motion feature dims to scale by true dt when dt normalization is enabled.",
    )
    parser.add_argument("--skip-pitch-projected", action="store_true", default=True)
    parser.add_argument(
        "--include-pitch-projected",
        action="store_false",
        dest="skip_pitch_projected",
        help="Also run *_pitch_projected.json files if present.",
    )
    args = parser.parse_args()
    cfg = Config.fromfile(args.config) if args.config else None
    apply_config_defaults(args, cfg)
    return args, cfg


def _resolve_arg(args, cfg, arg_name, cfg_name, fallback):
    value = getattr(args, arg_name, None)
    if value is not None:
        return value
    if cfg is not None and cfg_name in cfg:
        return cfg.get(cfg_name)
    return fallback


def apply_config_defaults(args, cfg):
    args.checkpoint = _resolve_arg(
        args,
        cfg,
        "checkpoint",
        "load_from",
        "ckpts/i3d_football_42_overall_True_2_query_pe_2.pt",
    )
    args.window_sampling = _resolve_arg(
        args, cfg, "window_sampling", "real_world_window_sampling", "uniform"
    )
    args.balanced_min_side_boxes = int(
        _resolve_arg(
            args,
            cfg,
            "balanced_min_side_boxes",
            "real_world_balanced_min_side_boxes",
            1,
        )
    )
    args.graph_num_frames = int(
        _resolve_arg(args, cfg, "graph_num_frames", "graph_num_frames", 8)
    )
    args.graph_stride = int(
        _resolve_arg(args, cfg, "graph_stride", "graph_stride", 4)
    )
    args.graph_max_nodes = int(
        _resolve_arg(
            args, cfg, "graph_max_nodes", "graph_max_nodes_per_window", 120
        )
    )
    args.opp_context_radius = float(
        _resolve_arg(
            args,
            cfg,
            "opp_context_radius",
            "graph_opp_context_radius",
            0.2,
        )
    )
    args.max_clips = int(
        _resolve_arg(args, cfg, "max_clips", "football_max_clips", args.max_clips)
    )
    args.graph_motion_dt_normalize = bool(
        _resolve_arg(
            args,
            cfg,
            "graph_motion_dt_normalize",
            "graph_motion_dt_normalize",
            False,
        )
    )
    args.graph_motion_dt_mode = str(
        _resolve_arg(
            args,
            cfg,
            "graph_motion_dt_mode",
            "graph_motion_dt_mode",
            "all",
        )
    ).lower()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def discover_samples(data_root, skip_pitch_projected=True):
    samples = []
    for match_dir in sorted(glob.glob(os.path.join(data_root, "*"))):
        if not os.path.isdir(match_dir):
            continue
        match_id = os.path.basename(match_dir)
        stats_path = os.path.join(match_dir, f"{match_id}.json")
        if not os.path.exists(stats_path):
            stats_candidates = [
                p
                for p in glob.glob(os.path.join(match_dir, "*.json"))
                if not p.endswith("_rough_player_positions.json")
                and not p.endswith("_pitch_projected.json")
            ]
            if not stats_candidates:
                print(f"[RealWorld] Skip {match_id}: missing stats json")
                continue
            stats_path = sorted(stats_candidates)[0]

        bbox_candidates = sorted(
            glob.glob(os.path.join(match_dir, "*_full_rough_player_positions*.json"))
        )
        if skip_pitch_projected:
            bbox_candidates = [
                p for p in bbox_candidates if not p.endswith("_pitch_projected.json")
            ]
        if not bbox_candidates:
            print(f"[RealWorld] Skip {match_id}: missing full bbox json")
            continue

        stats = load_json(stats_path)
        match = stats.get("match", {})
        perf = stats.get("performance_scores", {})
        home_team = match.get("home_team")
        away_team = match.get("away_team")
        if not home_team or not away_team:
            teams = sorted(perf.keys())
            if len(teams) < 2:
                print(f"[RealWorld] Skip {match_id}: invalid teams")
                continue
            home_team, away_team = teams[:2]
        if home_team not in perf or away_team not in perf:
            print(f"[RealWorld] Skip {match_id}: missing performance_scores teams")
            continue

        for bbox_path in bbox_candidates:
            samples.append(
                RealWorldSample(
                    match_id=match_id,
                    match_dir=match_dir,
                    stats_path=stats_path,
                    bbox_path=bbox_path,
                    home_team=home_team,
                    away_team=away_team,
                    home_score=float(perf[home_team].get("score_0_100", 50.0)),
                    away_score=float(perf[away_team].get("score_0_100", 50.0)),
                )
            )
    return samples


def parse_bbox_sampled(bbox_path):
    data = load_json(bbox_path)
    frames = data.get("frames", [])
    total = len(frames)
    home = [None] * total
    away = [None] * total
    for idx, frame in enumerate(frames):
        home_boxes = []
        away_boxes = []
        for player in frame.get("players", []):
            bbox = player.get("bbox")
            if bbox is None or len(bbox) != 4:
                continue
            if player.get("team") == "home":
                home_boxes.append(bbox)
            elif player.get("team") == "away":
                away_boxes.append(bbox)
        if home_boxes:
            home[idx] = np.asarray(home_boxes, dtype=np.float32)
        if away_boxes:
            away[idx] = np.asarray(away_boxes, dtype=np.float32)
    return data, home, away


def parse_bbox_indexed(bbox_path):
    data = load_json(bbox_path)
    total = int(data.get("total_frames", 0))
    home = [None] * total
    away = [None] * total
    for frame in data.get("frames", []):
        fidx = int(frame.get("frame_idx", -1))
        if fidx < 0 or fidx >= total:
            continue
        home_boxes = []
        away_boxes = []
        for player in frame.get("players", []):
            bbox = player.get("bbox")
            if bbox is None or len(bbox) != 4:
                continue
            if player.get("team") == "home":
                home_boxes.append(bbox)
            elif player.get("team") == "away":
                away_boxes.append(bbox)
        if home_boxes:
            home[fidx] = np.asarray(home_boxes, dtype=np.float32)
        if away_boxes:
            away[fidx] = np.asarray(away_boxes, dtype=np.float32)
    return data, home, away


def resolve_motion_time_delta(bbox_meta):
    extract_fps = bbox_meta.get("extract_fps", None)
    if extract_fps is not None and float(extract_fps) > 0.0:
        return 1.0 / float(extract_fps)
    fps = float(bbox_meta.get("fps", 30.0) or 30.0)
    extract_stride = float(bbox_meta.get("extract_stride", 1.0) or 1.0)
    if fps <= 0.0:
        return 1.0
    return extract_stride / fps


def build_graph_arrays(
    team_bboxes,
    opp_bboxes,
    frame_width,
    frame_height,
    args,
    motion_time_delta=None,
):
    nf = int(args.graph_num_frames)
    stride = int(args.graph_stride)
    num_windows = max(0, (len(team_bboxes) - nf) // stride + 1)
    node_dim = 14
    nodes = np.zeros(
        (num_windows, args.graph_max_nodes, node_dim), dtype=np.float32
    )
    mask = np.zeros((num_windows, args.graph_max_nodes), dtype=np.bool_)
    frame_ids = np.zeros((num_windows, args.graph_max_nodes), dtype=np.int64)

    for win_idx in range(num_windows):
        start = win_idx * stride
        window_frames = list(range(start, start + nf))
        n, m, fid = compute_box_features_for_window(
            team_bboxes,
            window_frames,
            frame_width,
            frame_height,
            max_nodes=args.graph_max_nodes,
            opp_bboxes_by_frame=opp_bboxes,
            opp_context_radius=args.opp_context_radius,
            motion_time_delta=motion_time_delta,
            motion_dt_mode=args.graph_motion_dt_mode,
        )
        nodes[win_idx] = n
        mask[win_idx] = m
        frame_ids[win_idx] = fid

    return nodes, mask, frame_ids


def _count_boxes(boxes_by_frame, frame_idx):
    if frame_idx >= len(boxes_by_frame):
        return 0
    boxes = boxes_by_frame[frame_idx]
    if boxes is None:
        return 0
    return int(len(boxes))


def compute_window_balance(home_bboxes, away_bboxes, args):
    nf = int(args.graph_num_frames)
    stride = int(args.graph_stride)
    total_frames = min(len(home_bboxes), len(away_bboxes))
    num_windows = max(0, (total_frames - nf) // stride + 1)
    stats = np.zeros((num_windows, 3), dtype=np.float32)
    for win_idx in range(num_windows):
        start = win_idx * stride
        home_count = 0
        away_count = 0
        for frame_idx in range(start, start + nf):
            home_count += _count_boxes(home_bboxes, frame_idx)
            away_count += _count_boxes(away_bboxes, frame_idx)
        total = home_count + away_count
        diff = abs(home_count - away_count)
        ratio = diff / float(max(total, 1))
        stats[win_idx] = (home_count, away_count, ratio)
    return stats


def select_window_indices(home_bboxes, away_bboxes, max_clips, args):
    stats = compute_window_balance(home_bboxes, away_bboxes, args)
    num_windows = int(stats.shape[0])
    if num_windows == 0:
        return np.zeros((0,), dtype=int), stats
    if num_windows <= max_clips:
        return np.arange(num_windows, dtype=int), stats
    if args.window_sampling == "uniform":
        return np.linspace(0, num_windows - 1, max_clips, dtype=int), stats

    min_side = int(args.balanced_min_side_boxes)
    selected = []
    bin_edges = np.linspace(0, num_windows, max_clips + 1, dtype=int)
    for bin_idx in range(max_clips):
        start = int(bin_edges[bin_idx])
        end = int(bin_edges[bin_idx + 1])
        if end <= start:
            end = min(start + 1, num_windows)
        candidates = np.arange(start, end, dtype=int)
        eligible = candidates[
            (stats[candidates, 0] >= min_side) & (stats[candidates, 1] >= min_side)
        ]
        if eligible.size == 0:
            eligible = candidates
        total_counts = stats[eligible, 0] + stats[eligible, 1]
        order = np.lexsort((-total_counts, stats[eligible, 2]))
        selected.append(int(eligible[order[0]]))
    return np.asarray(selected, dtype=int), stats


def select_or_pad_time(nodes, mask, frame_ids, max_clips, indices=None):
    num_windows = int(nodes.shape[0])
    if indices is not None:
        nodes = nodes[indices]
        mask = mask[indices]
        frame_ids = frame_ids[indices]
    elif num_windows > max_clips:
        indices = np.linspace(0, num_windows - 1, max_clips, dtype=int)
        nodes = nodes[indices]
        mask = mask[indices]
        frame_ids = frame_ids[indices]
    elif num_windows < max_clips:
        pad_t = max_clips - num_windows
        nodes = np.pad(nodes, ((0, pad_t), (0, 0), (0, 0)))
        mask = np.pad(mask, ((0, pad_t), (0, 0)))
        frame_ids = np.pad(frame_ids, ((0, pad_t), (0, 0)))
    if nodes.shape[0] < max_clips:
        pad_t = max_clips - int(nodes.shape[0])
        nodes = np.pad(nodes, ((0, pad_t), (0, 0), (0, 0)))
        mask = np.pad(mask, ((0, pad_t), (0, 0)))
        frame_ids = np.pad(frame_ids, ((0, pad_t), (0, 0)))
    return nodes, mask, frame_ids, num_windows


def build_models(device, cfg=None):
    graph_cfg = cfg or {}
    node_input_dim = int(graph_cfg.get("graph_node_input_dim", 14))
    graph_output_dim = int(graph_cfg.get("graph_output_dim", 128))
    graph_encoder = SpatioTemporalGraphEncoder(
        node_input_dim=node_input_dim,
        node_hidden_dim=int(graph_cfg.get("graph_node_hidden_dim", 64)),
        gnn_hidden_dim=int(graph_cfg.get("graph_gnn_hidden_dim", 128)),
        output_dim=graph_output_dim,
        num_gnn_layers=int(graph_cfg.get("graph_num_gnn_layers", 2)),
        num_heads=int(graph_cfg.get("graph_num_heads", 4)),
        dropout=float(graph_cfg.get("graph_dropout", 0.1)),
        cache_edges=bool(graph_cfg.get("graph_cache_edges", False)),
        edge_cache_size=int(graph_cfg.get("graph_edge_cache_size", 4096)),
        edge_style=str(graph_cfg.get("graph_edge_style", "sparse_knn")),
        spatial_k=int(graph_cfg.get("graph_spatial_k", 3)),
        temporal_k=int(graph_cfg.get("graph_temporal_k", 1)),
        pool_type=str(graph_cfg.get("graph_pool_type", "residual_attn")),
        pool_num_heads=int(graph_cfg.get("graph_pool_num_heads", 2)),
        pool_combine_type=str(graph_cfg.get("graph_pool_combine_type", "mean")),
    ).to(device)
    neck = TQN(
        graph_output_dim,
        int(graph_cfg.get("q_number", 256)),
        int(graph_cfg.get("query_var", 2)),
        str(graph_cfg.get("pe", "query_pe")),
        N=int(graph_cfg.get("num_layers", 2)),
        max_len=int(graph_cfg.get("max_len", 256)),
    ).to(device)
    head = Evaluator_weighted(
        input_dim=graph_output_dim,
        output_dim=int(graph_cfg.get("output_dim", 1)),
    ).to(device)
    return graph_encoder, neck, head


def load_checkpoint(path, graph_encoder, neck, head, device):
    ckpt = torch.load(path, map_location=device)
    if "graph_encoder" not in ckpt:
        raise KeyError("Checkpoint missing required key: graph_encoder")
    if "neck" not in ckpt:
        raise KeyError("Checkpoint missing required key: neck")

    head_state = ckpt.get("head")
    if head_state is None:
        head_state = ckpt.get("evaluator")
    if head_state is None:
        raise KeyError("Checkpoint missing required key: head/evaluator")

    graph_encoder.load_state_dict(ckpt["graph_encoder"])
    neck.load_state_dict(ckpt["neck"])
    head.load_state_dict(head_state)
    return ckpt


@torch.no_grad()
def predict_one(nodes, mask, frame_ids, graph_encoder, neck, head, device):
    graph_encoder.eval()
    neck.eval()
    head.eval()
    g_nodes = torch.from_numpy(nodes).unsqueeze(0).to(device)
    g_mask = torch.from_numpy(mask).unsqueeze(0).bool().to(device)
    g_fids = torch.from_numpy(frame_ids).unsqueeze(0).long().to(device)
    graph_feats = graph_encoder(g_nodes, g_mask, g_fids)
    tgt_weight, _ = neck(graph_feats, train=False)
    pred, _, _, _ = head(tgt_weight)
    pred_norm = float(pred.reshape(-1)[0].detach().cpu().item())
    return pred_norm


def summarise_graph(mask):
    valid_per_window = mask.sum(axis=1).astype(np.float32)
    if valid_per_window.size == 0:
        return 0.0, 1.0, 0
    empty_ratio = float((valid_per_window == 0).mean())
    return float(valid_per_window.mean()), empty_ratio, int(valid_per_window.size)


def summarise_balance(stats, indices):
    if stats.size == 0:
        return 0.0, 0.0, 0.0
    selected = stats[indices] if indices is not None and len(indices) > 0 else stats
    if selected.size == 0:
        return 0.0, 0.0, 0.0
    return (
        float(selected[:, 0].mean()),
        float(selected[:, 1].mean()),
        float(selected[:, 2].mean()),
    )


def calc_metrics(rows):
    preds = [r["pred_0_100"] for r in rows]
    trues = [r["true_0_100"] for r in rows]
    if not preds:
        return {}
    rho, _, rl2 = cal_spearmanr_rl2(preds, trues)
    pred_arr = np.asarray(preds, dtype=np.float32)
    true_arr = np.asarray(trues, dtype=np.float32)
    return {
        "team_srcc": float(rho),
        "team_rl2": float(rl2),
        "team_mae": float(np.abs(pred_arr - true_arr).mean()),
        "team_mse": float(((pred_arr - true_arr) ** 2).mean()),
        "pred_mean": float(pred_arr.mean()),
        "pred_std": float(pred_arr.std()),
        "true_mean": float(true_arr.mean()),
        "true_std": float(true_arr.std()),
    }


def calc_match_metrics(rows):
    by_match = {}
    for row in rows:
        by_match.setdefault(row["match_id"], {})[row["side"]] = row

    avg_preds = []
    avg_trues = []
    winner_hits = []
    for sides in by_match.values():
        if "home" not in sides or "away" not in sides:
            continue
        hp = sides["home"]["pred_0_100"]
        ap = sides["away"]["pred_0_100"]
        ht = sides["home"]["true_0_100"]
        at = sides["away"]["true_0_100"]
        avg_preds.append((hp + ap) / 2.0)
        avg_trues.append((ht + at) / 2.0)
        winner_hits.append(float((hp >= ap) == (ht >= at)))

    metrics = {}
    if len(avg_preds) >= 2:
        avg_rho, _, avg_rl2 = cal_spearmanr_rl2(avg_preds, avg_trues)
        metrics.update(
            {
                "match_avg_srcc": float(avg_rho),
                "match_avg_rl2": float(avg_rl2),
            }
        )
    if winner_hits:
        metrics["winner_side_acc"] = float(np.mean(winner_hits))
    return metrics


def build_match_rows(rows):
    by_match = {}
    for row in rows:
        by_match.setdefault(row["match_id"], {})[row["side"]] = row

    match_rows = []
    for match_id in sorted(by_match):
        sides = by_match[match_id]
        if "home" not in sides or "away" not in sides:
            continue
        home = sides["home"]
        away = sides["away"]
        home_pred = float(home["pred_0_100"])
        away_pred = float(away["pred_0_100"])
        home_true = float(home["true_0_100"])
        away_true = float(away["true_0_100"])
        match_rows.append(
            {
                "match_id": match_id,
                "home_team": home["team"],
                "away_team": away["team"],
                "home_pred": home_pred,
                "home_true": home_true,
                "away_pred": away_pred,
                "away_true": away_true,
                "avg_pred": (home_pred + away_pred) / 2.0,
                "avg_true": (home_true + away_true) / 2.0,
                "winner_pred": "home" if home_pred >= away_pred else "away",
                "winner_true": "home" if home_true >= away_true else "away",
            }
        )
    return match_rows


def _fmt_value(value):
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def print_table(title, rows, columns):
    print(f"[RealWorld] {title}:")
    if not rows:
        print("  (empty)")
        return

    widths = {}
    for key, header in columns:
        widths[key] = max(len(header), max(len(_fmt_value(row.get(key, ""))) for row in rows))

    header_line = "  " + " | ".join(header.ljust(widths[key]) for key, header in columns)
    divider_line = "  " + "-+-".join("-" * widths[key] for key, _ in columns)
    print(header_line)
    print(divider_line)
    for row in rows:
        print(
            "  "
            + " | ".join(_fmt_value(row.get(key, "")).ljust(widths[key]) for key, _ in columns)
        )


def print_metric_block(title, metrics, keys):
    print(f"[RealWorld] {title}:")
    for key in keys:
        if key in metrics:
            print(f"  {key}: {metrics[key]:.6f}")


def save_scatter_plot(path, x, y, title, xlabel, ylabel):
    if len(x) == 0 or len(y) == 0:
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x, y, alpha=0.8, s=40)

    xy_min = min(float(np.min(x)), float(np.min(y)))
    xy_max = max(float(np.max(x)), float(np.max(y)))
    if xy_min == xy_max:
        pad = 1.0
    else:
        pad = max(1.0, (xy_max - xy_min) * 0.05)
    lo = xy_min - pad
    hi = xy_max + pad
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.0, color="gray")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_csv(path, rows):
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fieldnames = [
        "match_id",
        "protocol",
        "side",
        "team",
        "pred_norm",
        "pred_0_100",
        "true_0_100",
        "error_0_100",
        "valid_windows",
        "selected_windows",
        "mean_valid_nodes",
        "empty_window_ratio",
        "mean_home_boxes_per_window",
        "mean_away_boxes_per_window",
        "mean_home_away_imbalance",
        "bbox_path",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    args, cfg = parse_args()
    samples = discover_samples(args.data_root, args.skip_pitch_projected)
    if not samples:
        raise RuntimeError(f"No real-world samples found under {args.data_root}")

    device = torch.device(args.device)
    graph_encoder, neck, head = build_models(device, cfg=cfg)
    ckpt = load_checkpoint(args.checkpoint, graph_encoder, neck, head, device)
    print(
        f"[RealWorld] Loaded checkpoint {args.checkpoint} "
        f"(epoch={ckpt.get('epoch', 'unknown')}, rho_best={ckpt.get('rho_best', 'unknown')})"
    )
    if cfg is not None:
        print(f"[RealWorld] Built model from config {args.config}")
    print(
        f"[RealWorld] graph_motion_dt_normalize={args.graph_motion_dt_normalize} "
        f"mode={args.graph_motion_dt_mode if args.graph_motion_dt_normalize else 'off'}"
    )

    rows = []
    parse_bbox = parse_bbox_sampled if args.protocol == "sampled" else parse_bbox_indexed
    for sample in samples:
        bbox_meta, home_bboxes, away_bboxes = parse_bbox(sample.bbox_path)
        frame_width = int(bbox_meta["frame_width"])
        frame_height = int(bbox_meta["frame_height"])
        motion_dt = None
        if args.graph_motion_dt_normalize:
            motion_dt = resolve_motion_time_delta(bbox_meta)
        window_indices, balance_stats = select_window_indices(
            home_bboxes, away_bboxes, args.max_clips, args
        )
        mean_home_boxes, mean_away_boxes, mean_imbalance = summarise_balance(
            balance_stats, window_indices
        )
        side_specs = [
            ("home", sample.home_team, sample.home_score, home_bboxes, away_bboxes),
            ("away", sample.away_team, sample.away_score, away_bboxes, home_bboxes),
        ]

        for side, team, true_score, team_bboxes, opp_bboxes in side_specs:
            nodes, mask, frame_ids = build_graph_arrays(
                team_bboxes,
                opp_bboxes,
                frame_width,
                frame_height,
                args,
                motion_time_delta=motion_dt,
            )
            raw_windows = int(nodes.shape[0])
            time_indices = window_indices if raw_windows > args.max_clips else None
            nodes, mask, frame_ids, valid_windows = select_or_pad_time(
                nodes,
                mask,
                frame_ids,
                args.max_clips,
                indices=time_indices,
            )
            mean_nodes, empty_ratio, _ = summarise_graph(mask)
            pred_norm = predict_one(
                nodes, mask, frame_ids, graph_encoder, neck, head, device
            )
            pred_100 = pred_norm * 100.0
            row = {
                "match_id": sample.match_id,
                "protocol": args.protocol,
                "side": side,
                "team": team,
                "pred_norm": pred_norm,
                "pred_0_100": pred_100,
                "true_0_100": true_score,
                "error_0_100": pred_100 - true_score,
                "valid_windows": valid_windows,
                "selected_windows": int(min(len(window_indices), args.max_clips)),
                "mean_valid_nodes": mean_nodes,
                "empty_window_ratio": empty_ratio,
                "mean_home_boxes_per_window": mean_home_boxes,
                "mean_away_boxes_per_window": mean_away_boxes,
                "mean_home_away_imbalance": mean_imbalance,
                "bbox_path": sample.bbox_path,
            }
            rows.append(row)
            print(
                f"[RealWorld] {sample.match_id} {side:<4} {team:<16} "
                f"pred={pred_100:6.2f} true={true_score:6.2f} "
                f"windows={valid_windows}->{row['selected_windows']} "
                f"nodes={mean_nodes:.2f} empty={empty_ratio:.3f} "
                f"balance={mean_home_boxes:.1f}/{mean_away_boxes:.1f} "
                f"dt={motion_dt if motion_dt is not None else 'off'} "
                f"imb={mean_imbalance:.3f}"
            )

    write_csv(args.output, rows)
    metrics = calc_metrics(rows)
    metrics.update(calc_match_metrics(rows))
    metrics_path = os.path.splitext(args.output)[0] + "_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)

    team_rows = sorted(
        rows,
        key=lambda r: (r["match_id"], 0 if r["side"] == "home" else 1, r["team"]),
    )
    match_rows = build_match_rows(rows)

    output_root = os.path.splitext(args.output)[0]
    team_plot_path = output_root + "_team_scatter.png"
    match_plot_path = output_root + "_match_avg_scatter.png"
    save_scatter_plot(
        team_plot_path,
        [row["true_0_100"] for row in team_rows],
        [row["pred_0_100"] for row in team_rows],
        "Real-World Team Scores: Prediction vs Ground Truth",
        "True score (0-100)",
        "Pred score (0-100)",
    )
    save_scatter_plot(
        match_plot_path,
        [row["avg_true"] for row in match_rows],
        [row["avg_pred"] for row in match_rows],
        "Real-World Match Average: Prediction vs Ground Truth",
        "True match average (0-100)",
        "Pred match average (0-100)",
    )

    print(f"[RealWorld] Wrote predictions: {args.output}")
    print(f"[RealWorld] Wrote metrics: {metrics_path}")
    print(f"[RealWorld] Wrote team plot: {team_plot_path}")
    print(f"[RealWorld] Wrote match plot: {match_plot_path}")
    print_table(
        "Match Rows",
        match_rows,
        [
            ("match_id", "match_id"),
            ("home_team", "home_team"),
            ("away_team", "away_team"),
            ("home_pred", "home_pred"),
            ("home_true", "home_true"),
            ("away_pred", "away_pred"),
            ("away_true", "away_true"),
            ("avg_pred", "avg_pred"),
            ("avg_true", "avg_true"),
            ("winner_pred", "winner_pred"),
            ("winner_true", "winner_true"),
        ],
    )
    print_metric_block(
        "Match Metrics",
        metrics,
        ["match_avg_srcc", "match_avg_rl2", "winner_side_acc"],
    )
    print_metric_block(
        "Team Metrics",
        metrics,
        ["team_srcc", "team_rl2", "team_mae", "team_mse"],
    )
    print_metric_block(
        "Distribution",
        metrics,
        ["pred_mean", "pred_std", "true_mean", "true_std"],
    )


if __name__ == "__main__":
    main()
