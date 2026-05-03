"""Zero-shot video-only inference on real-world FootballAQA matches.

Companion to ``real_world_infer.py`` (which targets the graph / bbox
checkpoints). This script targets the ROI-mask **video-only** checkpoint
trained with ``configs/train_football_roi_mask_only.py``: TQN neck +
``Evaluator_weighted`` head operating directly on pre-extracted 576-D ViViT
features (no backbone forward pass, no graph branch, no bbox parsing).

Pipeline:
1. Discover one prediction sample per ``.npy`` file under ``--data-root`` match
   folders, or under optional ``--features-dir`` folders.
2. Mirror the dataloader's video-only test-time path (centre crop / zero pad to
   ``--max-clips``) before forwarding through the neck + head.
3. Optionally read per-team ground-truth scores from ``--data-root`` (matched
   by the feature stem) so we can report SRCC / RL2 / MAE / MSE against the
   ``avg`` target the checkpoint was trained on.
"""

import argparse
import csv
import glob
import json
import os
import re
from dataclasses import dataclass
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from mmengine.config import Config

from loss import cal_spearmanr_rl2
from models.head.evaluator import Evaluator_weighted
from models.neck.TQN import TQN


DEFAULT_DATA_ROOTS = (
    "/home2/zvnm27/PHD/VideoMamba/FootballAQA/england_epl",
    "/home2/zvnm27/PHD/VideoMamba/FootballAQA/france_league",
)
DEFAULT_CONFIG = "configs/train_football_roi_mask_only.py"
DEFAULT_CHECKPOINT = "ckpts/football_roi_mask_only.pt"


@dataclass
class RealWorldVideoSample:
    match_id: str
    feature_path: str
    stats_path: Optional[str]
    home_team: Optional[str]
    away_team: Optional[str]
    home_score: Optional[float]
    away_score: Optional[float]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Zero-shot video-only inference on real-world FootballAQA data "
            "using the ROI-mask checkpoint."
        )
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="Experiment config used to resolve neck/head dims and defaults.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=f"Checkpoint path. Defaults to cfg.load_from or {DEFAULT_CHECKPOINT}.",
    )
    parser.add_argument(
        "--data-root",
        action="append",
        default=None,
        help=(
            "Directory with match folders containing bbox JSONs, stats JSONs, "
            "and per-match ViViT feature .npy files. Can be passed multiple "
            "times. Defaults to the England EPL and France league folders."
        ),
    )
    parser.add_argument(
        "--features-dir",
        action="append",
        default=None,
        help=(
            "Optional legacy directory holding per-match ViViT feature .npy "
            "files. Can be passed multiple times."
        ),
    )
    parser.add_argument(
        "--output",
        default="exp/real_world_roi_mask/preds.csv",
        help="CSV destination for per-match predictions.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--max-clips",
        type=int,
        default=None,
        help="Sequence length fed to the neck. Defaults to cfg.football_max_clips.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config not found: {args.config}")
    cfg = Config.fromfile(args.config)
    apply_config_defaults(args, cfg)
    args.data_roots = _normalize_dirs(
        args.data_root if args.data_root is not None else DEFAULT_DATA_ROOTS,
        "data root",
    )
    args.features_dirs = _normalize_dirs(args.features_dir or [], "features dir")
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
        args, cfg, "checkpoint", "load_from", DEFAULT_CHECKPOINT
    )
    args.max_clips = int(
        _resolve_arg(args, cfg, "max_clips", "football_max_clips", 256)
    )


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_dirs(paths, label):
    dirs = []
    for path in paths:
        if not path:
            continue
        norm = os.path.abspath(os.path.expanduser(path))
        if not os.path.isdir(norm):
            raise FileNotFoundError(f"{label} not found: {norm}")
        dirs.append(norm)
    return dirs


# ---------------------------------------------------------------------------
# Sample discovery
# ---------------------------------------------------------------------------
def _candidate_stats_paths(data_root, match_id, feature_path=None):
    """Return likely locations for a per-match stats JSON, in priority order."""
    candidates = []
    if feature_path:
        feature_dir = os.path.dirname(feature_path)
        candidates.append(os.path.join(feature_dir, f"{match_id}.json"))
        for cand in glob.glob(os.path.join(feature_dir, "*.json")):
            if cand.endswith("_rough_player_positions.json"):
                continue
            if cand.endswith("_pitch_projected.json"):
                continue
            candidates.append(cand)
    if data_root:
        candidates.extend(
            [
                os.path.join(data_root, match_id, f"{match_id}.json"),
                os.path.join(data_root, f"{match_id}.json"),
            ]
        )
        for cand in glob.glob(os.path.join(data_root, match_id, "*.json")):
            if cand.endswith("_rough_player_positions.json"):
                continue
            if cand.endswith("_pitch_projected.json"):
                continue
            candidates.append(cand)
    seen = set()
    unique = []
    for cand in candidates:
        if cand in seen:
            continue
        seen.add(cand)
        if os.path.exists(cand):
            unique.append(cand)
    return unique


_TEAM_SLUG_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_(.+?)(?:-premier-league.*)?$")


def _slug_to_team_guess(stem):
    """Best-effort split of ``YYYY-MM-DD_<home>-<away>-<league>...`` into teams.

    Used only as a fallback for log lines / CSV columns when no stats JSON is
    present. Predictions never depend on this guess.
    """
    match = _TEAM_SLUG_RE.match(stem)
    if not match:
        return None, None
    body = match.group(1)
    parts = body.split("-")
    if len(parts) < 2:
        return None, None
    mid = len(parts) // 2
    home = "-".join(parts[:mid])
    away = "-".join(parts[mid:])
    return home or None, away or None


def _resolve_labels(stats_path, fallback_home, fallback_away):
    """Pull (home_team, away_team, home_score, away_score) from a stats JSON.

    Returns ``(None, None, None, None)`` if the JSON cannot be parsed in the
    expected schema. Score fields use the ``performance_scores[*].score_0_100``
    layout used by the original real-world dataset.
    """
    if stats_path is None:
        return fallback_home, fallback_away, None, None
    try:
        stats = load_json(stats_path)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[RealWorld] Skip stats {stats_path}: {exc}")
        return fallback_home, fallback_away, None, None

    match = stats.get("match", {}) or {}
    perf = stats.get("performance_scores", {}) or {}
    home_team = match.get("home_team") or fallback_home
    away_team = match.get("away_team") or fallback_away
    if not home_team or not away_team:
        teams = sorted(perf.keys())
        if len(teams) >= 2:
            home_team = home_team or teams[0]
            away_team = away_team or teams[1]
    if home_team in perf and away_team in perf:
        home_score = float(perf[home_team].get("score_0_100", float("nan")))
        away_score = float(perf[away_team].get("score_0_100", float("nan")))
    else:
        home_score = away_score = None
    return home_team, away_team, home_score, away_score


def _feature_paths_under(root):
    """Find feature arrays in either a flat directory or one-level match folders."""
    return sorted(
        glob.glob(os.path.join(root, "*.npy"))
        + glob.glob(os.path.join(root, "*", "*.npy"))
    )


def _is_relative_to(path, root):
    try:
        return os.path.commonpath([path, root]) == root
    except ValueError:
        return False


def _collect_feature_records(features_dirs, data_roots):
    records = {}
    for feature_dir in features_dirs:
        for feat_path in _feature_paths_under(feature_dir):
            abs_path = os.path.abspath(feat_path)
            records.setdefault(abs_path, None)

    for data_root in data_roots:
        for feat_path in _feature_paths_under(data_root):
            abs_path = os.path.abspath(feat_path)
            # Prefer the data root associated with the feature path so labels
            # resolve from the same match folder as the bbox-derived files.
            records[abs_path] = data_root

    return [(path, records[path]) for path in sorted(records)]


def discover_samples(features_dirs, data_roots):
    feature_records = _collect_feature_records(features_dirs, data_roots)
    if not feature_records:
        raise FileNotFoundError(
            "No .npy feature files found under data roots or feature dirs: "
            f"data_roots={data_roots}, features_dirs={features_dirs}"
        )

    samples = []
    for feat_path, matched_data_root in feature_records:
        match_id = os.path.splitext(os.path.basename(feat_path))[0]
        slug_home, slug_away = _slug_to_team_guess(match_id)

        stats_candidates = []
        stats_roots = (
            [matched_data_root]
            if matched_data_root is not None
            else [
                data_root
                for data_root in data_roots
                if _is_relative_to(os.path.dirname(feat_path), data_root)
            ] or data_roots
        )
        for data_root in stats_roots:
            stats_candidates.extend(
                _candidate_stats_paths(data_root, match_id, feat_path)
            )
        stats_path = stats_candidates[0] if stats_candidates else None
        home_team, away_team, home_score, away_score = _resolve_labels(
            stats_path, slug_home, slug_away
        )

        samples.append(
            RealWorldVideoSample(
                match_id=match_id,
                feature_path=feat_path,
                stats_path=stats_path,
                home_team=home_team,
                away_team=away_team,
                home_score=home_score,
                away_score=away_score,
            )
        )
    return samples


# ---------------------------------------------------------------------------
# Feature loading + temporal shaping
# ---------------------------------------------------------------------------
def load_and_shape_features(feature_path, max_clips, expected_dim):
    """Mirror ``Football_Dataset._getitem_video_only`` for test-time inference.

    Long sequences are centre-cropped (matching ``self.subset != 'train'`` in
    the dataloader) and short sequences are zero-padded so the neck always
    sees a fixed temporal length.
    """
    feats = np.load(feature_path, mmap_mode="r")
    feats = np.asarray(feats, dtype=np.float32)

    if feats.ndim != 2 or feats.shape[1] != expected_dim:
        raise ValueError(
            f"Unexpected feature shape {feats.shape} (expected [T, {expected_dim}]) "
            f"for {feature_path}"
        )

    raw_clips = int(feats.shape[0])
    if raw_clips > max_clips:
        st = (raw_clips - max_clips) // 2
        feats = feats[st: st + max_clips]
    elif raw_clips < max_clips:
        pad = np.zeros((max_clips - raw_clips, feats.shape[1]), dtype=np.float32)
        feats = np.concatenate([feats, pad], axis=0)

    feats = np.ascontiguousarray(feats, dtype=np.float32)
    return feats, raw_clips


# ---------------------------------------------------------------------------
# Model build / load / predict
# ---------------------------------------------------------------------------
def build_models(device, cfg):
    feat_dim = int(cfg.get("feature_dim", 576))
    neck = TQN(
        feat_dim,
        int(cfg.get("q_number", 256)),
        int(cfg.get("query_var", 2)),
        str(cfg.get("pe", "query_pe")),
        N=int(cfg.get("num_layers", 2)),
        max_len=int(cfg.get("max_len", 256)),
        dropout=float(cfg.get("tqn_dropout", 0.5)),
    ).to(device)
    head = Evaluator_weighted(
        input_dim=feat_dim,
        output_dim=int(cfg.get("output_dim", 1)),
    ).to(device)
    return neck, head


def _strip_module_prefix(state_dict):
    """Drop the ``module.`` prefix added by ``nn.DataParallel`` if present."""
    if not state_dict:
        return state_dict
    if not all(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k[len("module."):]: v for k, v in state_dict.items()}


def load_checkpoint(path, neck, head, device):
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device)

    if "neck" not in ckpt:
        raise KeyError("Checkpoint missing required key: neck")
    head_state = ckpt.get("head", ckpt.get("evaluator"))
    if head_state is None:
        raise KeyError("Checkpoint missing required key: head/evaluator")

    neck.load_state_dict(_strip_module_prefix(ckpt["neck"]))
    head.load_state_dict(_strip_module_prefix(head_state))
    return ckpt


@torch.no_grad()
def predict_one(feats_tensor, neck, head, device):
    neck.eval()
    head.eval()
    clip_feats = feats_tensor.unsqueeze(0).to(device)  # [1, T, D]
    tgt_weight, _ = neck(clip_feats, train=False)
    pred, _, _, _ = head(tgt_weight)
    pred_norm = float(pred.reshape(-1)[0].detach().cpu().item())
    return pred_norm


# ---------------------------------------------------------------------------
# Metrics + reporting
# ---------------------------------------------------------------------------
def calc_metrics(rows):
    scored = [
        r for r in rows
        if isinstance(r.get("true_avg_0_100"), (int, float))
        and not (isinstance(r["true_avg_0_100"], float) and np.isnan(r["true_avg_0_100"]))
    ]
    if not scored:
        return {}
    preds = [float(r["pred_avg_0_100"]) for r in scored]
    trues = [float(r["true_avg_0_100"]) for r in scored]
    pred_arr = np.asarray(preds, dtype=np.float32)
    true_arr = np.asarray(trues, dtype=np.float32)

    metrics = {
        "match_count_with_labels": len(scored),
        "match_avg_mae": float(np.abs(pred_arr - true_arr).mean()),
        "match_avg_mse": float(((pred_arr - true_arr) ** 2).mean()),
        "pred_mean": float(pred_arr.mean()),
        "pred_std": float(pred_arr.std()),
        "true_mean": float(true_arr.mean()),
        "true_std": float(true_arr.std()),
    }
    if len(preds) >= 2:
        rho, _, rl2 = cal_spearmanr_rl2(preds, trues)
        metrics["match_avg_srcc"] = float(rho)
        metrics["match_avg_rl2"] = float(rl2)
    return metrics


def write_csv(path, rows):
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fieldnames = [
        "match_id",
        "home_team",
        "away_team",
        "pred_norm",
        "pred_avg_0_100",
        "home_true_0_100",
        "away_true_0_100",
        "true_avg_0_100",
        "error_avg_0_100",
        "raw_clips",
        "used_clips",
        "feature_path",
        "stats_path",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


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


def _fmt_value(value):
    if value is None or value == "":
        return "-"
    if isinstance(value, float):
        if np.isnan(value):
            return "-"
        return f"{value:.4f}"
    return str(value)


def print_table(title, rows, columns):
    print(f"[RealWorld] {title}:")
    if not rows:
        print("  (empty)")
        return
    widths = {
        key: max(
            len(header),
            max(len(_fmt_value(row.get(key, ""))) for row in rows),
        )
        for key, header in columns
    }
    header_line = "  " + " | ".join(
        header.ljust(widths[key]) for key, header in columns
    )
    divider_line = "  " + "-+-".join("-" * widths[key] for key, _ in columns)
    print(header_line)
    print(divider_line)
    for row in rows:
        print(
            "  "
            + " | ".join(
                _fmt_value(row.get(key, "")).ljust(widths[key])
                for key, _ in columns
            )
        )


def print_metric_block(title, metrics, keys):
    if not metrics:
        return
    print(f"[RealWorld] {title}:")
    for key in keys:
        if key not in metrics:
            continue
        value = metrics[key]
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")


def _fmt_paths(paths):
    return ", ".join(paths) if paths else "-"


def main():
    args, cfg = parse_args()
    samples = discover_samples(args.features_dirs, args.data_roots)

    expected_dim = int(cfg.get("video_feature_dim", cfg.get("feature_dim", 576)))
    device = torch.device(args.device)
    neck, head = build_models(device, cfg)
    ckpt = load_checkpoint(args.checkpoint, neck, head, device)
    print(
        f"[RealWorld] Loaded checkpoint {args.checkpoint} "
        f"(epoch={ckpt.get('epoch', 'unknown')}, "
        f"rho_best={ckpt.get('rho_best', 'unknown')})"
    )
    print(
        f"[RealWorld] Config={args.config} | feature_dim={expected_dim} | "
        f"max_clips={args.max_clips}"
    )
    print(f"[RealWorld] data_roots={_fmt_paths(args.data_roots)}")
    print(f"[RealWorld] features_dirs={_fmt_paths(args.features_dirs)}")

    rows = []
    for sample in samples:
        feats_np, raw_clips = load_and_shape_features(
            sample.feature_path, args.max_clips, expected_dim
        )
        feats_tensor = torch.from_numpy(feats_np)
        pred_norm = predict_one(feats_tensor, neck, head, device)
        pred_100 = pred_norm * 100.0

        if sample.home_score is not None and sample.away_score is not None:
            true_avg = (sample.home_score + sample.away_score) / 2.0
            error = pred_100 - true_avg
        else:
            true_avg = None
            error = None

        used_clips = min(raw_clips, args.max_clips)
        row = {
            "match_id": sample.match_id,
            "home_team": sample.home_team or "",
            "away_team": sample.away_team or "",
            "pred_norm": pred_norm,
            "pred_avg_0_100": pred_100,
            "home_true_0_100": (
                sample.home_score if sample.home_score is not None else ""
            ),
            "away_true_0_100": (
                sample.away_score if sample.away_score is not None else ""
            ),
            "true_avg_0_100": true_avg if true_avg is not None else "",
            "error_avg_0_100": error if error is not None else "",
            "raw_clips": raw_clips,
            "used_clips": used_clips,
            "feature_path": sample.feature_path,
            "stats_path": sample.stats_path or "",
        }
        rows.append(row)

        if true_avg is None:
            print(
                f"[RealWorld] {sample.match_id} pred={pred_100:6.2f} "
                f"true=N/A clips={raw_clips}->{used_clips}"
            )
        else:
            print(
                f"[RealWorld] {sample.match_id} pred={pred_100:6.2f} "
                f"true_avg={true_avg:6.2f} err={error:+6.2f} "
                f"home={sample.home_score:.2f}/away={sample.away_score:.2f} "
                f"clips={raw_clips}->{used_clips}"
            )

    write_csv(args.output, rows)
    metrics = calc_metrics(rows)
    metrics_path = os.path.splitext(args.output)[0] + "_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)

    scored_rows = [
        r for r in rows
        if isinstance(r.get("true_avg_0_100"), (int, float))
    ]
    output_root = os.path.splitext(args.output)[0]
    scatter_path = output_root + "_match_avg_scatter.png"
    if scored_rows:
        save_scatter_plot(
            scatter_path,
            [float(r["true_avg_0_100"]) for r in scored_rows],
            [float(r["pred_avg_0_100"]) for r in scored_rows],
            "Real-World Match Average: ROI-Mask Video-Only Prediction vs GT",
            "True match average (0-100)",
            "Pred match average (0-100)",
        )

    print(f"[RealWorld] Wrote predictions: {args.output}")
    print(f"[RealWorld] Wrote metrics: {metrics_path}")
    if scored_rows:
        print(f"[RealWorld] Wrote scatter plot: {scatter_path}")

    print_table(
        "Match Rows",
        rows,
        [
            ("match_id", "match_id"),
            ("home_team", "home_team"),
            ("away_team", "away_team"),
            ("pred_avg_0_100", "pred_avg"),
            ("home_true_0_100", "home_true"),
            ("away_true_0_100", "away_true"),
            ("true_avg_0_100", "true_avg"),
            ("error_avg_0_100", "error_avg"),
            ("raw_clips", "raw_clips"),
            ("used_clips", "used_clips"),
        ],
    )
    print_metric_block(
        "Match Metrics",
        metrics,
        [
            "match_count_with_labels",
            "match_avg_srcc",
            "match_avg_rl2",
            "match_avg_mae",
            "match_avg_mse",
        ],
    )
    print_metric_block(
        "Distribution",
        metrics,
        ["pred_mean", "pred_std", "true_mean", "true_std"],
    )


if __name__ == "__main__":
    main()
