import argparse
import copy
import csv
import json
import os
import random

import numpy as np
import torch
from mmengine.config import Config
from torch import nn
from torch.utils.data import DataLoader, Dataset

from real_world_infer import (
    apply_config_defaults,
    build_models,
    calc_metrics,
    discover_samples,
    load_checkpoint,
    parse_bbox_indexed,
    parse_bbox_sampled,
    print_metric_block,
    print_table,
    resolve_motion_time_delta,
    save_scatter_plot,
    select_or_pad_time,
    select_window_indices,
    summarise_balance,
    summarise_graph,
    build_graph_arrays,
)


class RealWorldTeamDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return {
            "nodes": torch.from_numpy(row["nodes"]),
            "mask": torch.from_numpy(row["mask"]),
            "frame_ids": torch.from_numpy(row["frame_ids"]),
            "score": torch.tensor(row["true_0_100"] / 100.0, dtype=torch.float32),
            "match_id": row["match_id"],
            "side": row["side"],
            "team": row["team"],
            "bbox_path": row["bbox_path"],
            "group_id": row["group_id"],
        }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Few-shot real-world fine-tuning: each fold uses one real-world unit "
            "for training and evaluates on the remaining units."
        )
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--data-root", default="real_word_data")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default="exp/real_world_ft")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--protocol", choices=("sampled", "indexed"), default="sampled")
    parser.add_argument("--cv-unit", choices=("match", "bbox"), default="match")
    parser.add_argument("--max-clips", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-folds", type=int, default=5)
    parser.add_argument("--save-fold-checkpoints", action="store_true")
    parser.add_argument("--window-sampling", choices=("uniform", "balanced"), default=None)
    parser.add_argument("--balanced-min-side-boxes", type=int, default=None)
    parser.add_argument("--graph-num-frames", type=int, default=None)
    parser.add_argument("--graph-stride", type=int, default=None)
    parser.add_argument("--graph-max-nodes", type=int, default=None)
    parser.add_argument("--opp-context-radius", type=float, default=None)
    parser.add_argument(
        "--graph-motion-dt-normalize",
        dest="graph_motion_dt_normalize",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--no-graph-motion-dt-normalize",
        dest="graph_motion_dt_normalize",
        action="store_false",
    )
    parser.add_argument(
        "--graph-motion-dt-mode",
        choices=("all", "dxdy"),
        default=None,
    )
    parser.add_argument("--skip-pitch-projected", action="store_true", default=True)
    parser.add_argument(
        "--include-pitch-projected",
        action="store_false",
        dest="skip_pitch_projected",
    )
    args = parser.parse_args()
    cfg = Config.fromfile(args.config) if args.config else None
    apply_config_defaults(args, cfg)
    return args, cfg


def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_group_id(sample, cv_unit):
    if cv_unit == "bbox":
        return sample.bbox_path
    return sample.match_id


def prepare_rows(samples, args):
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
            rows.append(
                {
                    "match_id": sample.match_id,
                    "group_id": build_group_id(sample, args.cv_unit),
                    "protocol": args.protocol,
                    "side": side,
                    "team": team,
                    "true_0_100": float(true_score),
                    "valid_windows": valid_windows,
                    "selected_windows": int(min(len(window_indices), args.max_clips)),
                    "mean_valid_nodes": float(mean_nodes),
                    "empty_window_ratio": float(empty_ratio),
                    "mean_home_boxes_per_window": float(mean_home_boxes),
                    "mean_away_boxes_per_window": float(mean_away_boxes),
                    "mean_home_away_imbalance": float(mean_imbalance),
                    "bbox_path": sample.bbox_path,
                    "nodes": np.ascontiguousarray(nodes, dtype=np.float32),
                    "mask": np.ascontiguousarray(mask, dtype=np.bool_),
                    "frame_ids": np.ascontiguousarray(frame_ids, dtype=np.int64),
                }
            )
    return rows


def build_optimizer(args, graph_encoder, neck, head):
    params = list(graph_encoder.parameters()) + list(neck.parameters()) + list(head.parameters())
    return torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)


def forward_batch(batch, graph_encoder, neck, head, device):
    nodes = batch["nodes"].to(device=device, dtype=torch.float32)
    mask = batch["mask"].to(device=device, dtype=torch.bool)
    frame_ids = batch["frame_ids"].to(device=device, dtype=torch.long)
    graph_feats = graph_encoder(nodes, mask, frame_ids)
    tgt_weight, _ = neck(graph_feats, train=False)
    pred, _, _, _ = head(tgt_weight)
    return pred.reshape(-1)


def train_one_fold(train_rows, args, cfg):
    device = torch.device(args.device)
    graph_encoder, neck, head = build_models(device, cfg=cfg)
    try:
        ckpt = load_checkpoint(args.checkpoint, graph_encoder, neck, head, device)
    except RuntimeError as exc:
        raise RuntimeError(
            "Failed to load checkpoint into the current graph model. "
            "This usually means the checkpoint architecture does not match the "
            "current environment/config. In this repo, the current no-PyG setup "
            "is compatible with checkpoints such as `ckpts/g2_radius_0p30.pt`, "
            "`ckpts/g2_radius_0p30_motion_dt_norm.pt`, and "
            "`ckpts/g2_radius_0p30_motion_dt_norm_dxdy.pt`."
        ) from exc

    train_loader = DataLoader(
        RealWorldTeamDataset(train_rows),
        batch_size=min(args.batch_size, len(train_rows)),
        shuffle=True,
        num_workers=0,
    )
    optimizer = build_optimizer(args, graph_encoder, neck, head)
    criterion = nn.MSELoss()

    best_state = None
    best_loss = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        graph_encoder.train()
        neck.train()
        head.train()
        loss_sum = 0.0
        item_count = 0
        for batch in train_loader:
            optimizer.zero_grad()
            pred = forward_batch(batch, graph_encoder, neck, head, device)
            target = batch["score"].to(device=device, dtype=torch.float32)
            loss = criterion(pred, target)
            loss.backward()
            if args.grad_clip is not None and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(graph_encoder.parameters()) + list(neck.parameters()) + list(head.parameters()),
                    args.grad_clip,
                )
            optimizer.step()

            batch_size = int(target.shape[0])
            loss_sum += float(loss.item()) * batch_size
            item_count += batch_size

        epoch_loss = loss_sum / max(item_count, 1)
        history.append({"epoch": epoch, "train_mse": epoch_loss})
        if epoch_loss <= best_loss:
            best_loss = epoch_loss
            best_state = {
                "graph_encoder": copy.deepcopy(graph_encoder.state_dict()),
                "neck": copy.deepcopy(neck.state_dict()),
                "head": copy.deepcopy(head.state_dict()),
            }

    if best_state is not None:
        graph_encoder.load_state_dict(best_state["graph_encoder"])
        neck.load_state_dict(best_state["neck"])
        head.load_state_dict(best_state["head"])

    return graph_encoder, neck, head, ckpt, history, best_loss


@torch.no_grad()
def evaluate_rows(rows, graph_encoder, neck, head, args, fold_name, split_name):
    device = torch.device(args.device)
    graph_encoder.eval()
    neck.eval()
    head.eval()
    out_rows = []
    for row in rows:
        batch = {
            "nodes": torch.from_numpy(row["nodes"]).unsqueeze(0),
            "mask": torch.from_numpy(row["mask"]).unsqueeze(0),
            "frame_ids": torch.from_numpy(row["frame_ids"]).unsqueeze(0),
        }
        pred_norm = float(forward_batch(batch, graph_encoder, neck, head, device)[0].cpu().item())
        pred_100 = pred_norm * 100.0
        out = {
            "fold": fold_name,
            "split": split_name,
            "train_group": fold_name,
            "group_id": row["group_id"],
            "match_id": row["match_id"],
            "protocol": row["protocol"],
            "side": row["side"],
            "team": row["team"],
            "pred_norm": pred_norm,
            "pred_0_100": pred_100,
            "true_0_100": row["true_0_100"],
            "error_0_100": pred_100 - row["true_0_100"],
            "valid_windows": row["valid_windows"],
            "selected_windows": row["selected_windows"],
            "mean_valid_nodes": row["mean_valid_nodes"],
            "empty_window_ratio": row["empty_window_ratio"],
            "mean_home_boxes_per_window": row["mean_home_boxes_per_window"],
            "mean_away_boxes_per_window": row["mean_away_boxes_per_window"],
            "mean_home_away_imbalance": row["mean_home_away_imbalance"],
            "bbox_path": row["bbox_path"],
        }
        out_rows.append(out)
    return out_rows


def calc_fold_summary(train_group, train_rows, test_rows, best_loss, history):
    summary = {
        "train_group": train_group,
        "num_train_rows": len(train_rows),
        "num_test_rows": len(test_rows),
        "best_train_mse": float(best_loss),
        "epochs": len(history),
    }
    if train_rows:
        train_metrics = calc_metrics(train_rows)
        train_metrics.update(calc_repeated_match_metrics(train_rows))
        summary.update({f"train_{k}": v for k, v in train_metrics.items()})
    if test_rows:
        test_metrics = calc_metrics(test_rows)
        test_metrics.update(calc_repeated_match_metrics(test_rows))
        summary.update({f"test_{k}": v for k, v in test_metrics.items()})
    return summary


def write_fold_summary_csv(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_prediction_rows(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "fold",
        "split",
        "train_group",
        "group_id",
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
        writer.writerows(rows)


def write_match_rows(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "fold",
        "match_id",
        "home_team",
        "away_team",
        "home_pred",
        "home_true",
        "away_pred",
        "away_true",
        "avg_pred",
        "avg_true",
        "winner_pred",
        "winner_true",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def calc_repeated_match_metrics(rows):
    by_match = {}
    for row in rows:
        key = (row["fold"], row["match_id"])
        by_match.setdefault(key, {})[row["side"]] = row

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
        from loss import cal_spearmanr_rl2

        avg_rho, _, avg_rl2 = cal_spearmanr_rl2(avg_preds, avg_trues)
        metrics["match_avg_srcc"] = float(avg_rho)
        metrics["match_avg_rl2"] = float(avg_rl2)
    if winner_hits:
        metrics["winner_side_acc"] = float(np.mean(winner_hits))
    return metrics


def build_repeated_match_rows(rows):
    by_match = {}
    for row in rows:
        key = (row["fold"], row["match_id"])
        by_match.setdefault(key, {})[row["side"]] = row

    match_rows = []
    for (fold, match_id), sides in sorted(by_match.items()):
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
                "fold": fold,
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


def aggregate_repeat_metrics(test_rows, fold_summaries):
    metrics = {}
    if test_rows:
        metrics["all_fold_test_predictions"] = calc_metrics(test_rows)
        metrics["all_fold_test_predictions"].update(calc_repeated_match_metrics(test_rows))

    numeric_keys = sorted(
        {
            key
            for row in fold_summaries
            for key, value in row.items()
            if isinstance(value, (float, int)) and key not in {"epochs", "num_train_rows", "num_test_rows"}
        }
    )
    mean_metrics = {}
    for key in numeric_keys:
        values = [float(row[key]) for row in fold_summaries if key in row]
        if values:
            mean_metrics[key] = float(np.mean(values))
    metrics["mean_over_folds"] = mean_metrics
    return metrics


def maybe_save_fold_checkpoint(path, graph_encoder, neck, head, base_ckpt, args, train_group):
    state = {
        "graph_encoder": graph_encoder.state_dict(),
        "neck": neck.state_dict(),
        "head": head.state_dict(),
        "base_checkpoint": args.checkpoint,
        "base_epoch": base_ckpt.get("epoch", None),
        "base_rho_best": base_ckpt.get("rho_best", None),
        "train_group": train_group,
    }
    torch.save(state, path)


def main():
    args, cfg = parse_args()
    init_seed(args.seed)

    samples = discover_samples(args.data_root, args.skip_pitch_projected)
    if not samples:
        raise RuntimeError(f"No real-world samples found under {args.data_root}")
    if args.checkpoint is None:
        raise ValueError("`--checkpoint` is required for fine-tuning.")

    prepared_rows = prepare_rows(samples, args)
    group_ids = sorted({row["group_id"] for row in prepared_rows})
    if len(group_ids) < 2:
        raise RuntimeError(
            f"Need at least 2 {args.cv_unit} groups for cross-validation, got {len(group_ids)}."
        )
    if args.max_folds is not None:
        group_ids = group_ids[: args.max_folds]

    os.makedirs(args.output_dir, exist_ok=True)
    all_train_rows = []
    all_test_rows = []
    fold_summaries = []

    print(
        f"[RealWorldFT] Prepared {len(prepared_rows)} team rows from {len(samples)} bbox samples. "
        f"cv_unit={args.cv_unit}, folds={len(group_ids)}, checkpoint={args.checkpoint}"
    )

    for fold_idx, train_group in enumerate(group_ids, start=1):
        train_rows_src = [row for row in prepared_rows if row["group_id"] == train_group]
        test_rows_src = [row for row in prepared_rows if row["group_id"] != train_group]
        if not train_rows_src or not test_rows_src:
            print(f"[RealWorldFT] Skip fold {train_group}: empty train/test split")
            continue

        print(
            f"[RealWorldFT] Fold {fold_idx}/{len(group_ids)} train_group={train_group} "
            f"train_rows={len(train_rows_src)} test_rows={len(test_rows_src)}"
        )
        graph_encoder, neck, head, base_ckpt, history, best_loss = train_one_fold(
            train_rows_src, args, cfg
        )
        train_rows = evaluate_rows(
            train_rows_src, graph_encoder, neck, head, args, train_group, "train"
        )
        test_rows = evaluate_rows(
            test_rows_src, graph_encoder, neck, head, args, train_group, "test"
        )
        all_train_rows.extend(train_rows)
        all_test_rows.extend(test_rows)

        fold_dir = os.path.join(args.output_dir, f"fold_{fold_idx:02d}")
        os.makedirs(fold_dir, exist_ok=True)
        write_prediction_rows(os.path.join(fold_dir, "train_preds.csv"), train_rows)
        write_prediction_rows(os.path.join(fold_dir, "test_preds.csv"), test_rows)
        write_match_rows(
            os.path.join(fold_dir, "train_match_preds.csv"),
            build_repeated_match_rows(train_rows),
        )
        write_match_rows(
            os.path.join(fold_dir, "test_match_preds.csv"),
            build_repeated_match_rows(test_rows),
        )
        with open(os.path.join(fold_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        if args.save_fold_checkpoints:
            maybe_save_fold_checkpoint(
                os.path.join(fold_dir, "finetuned.pt"),
                graph_encoder,
                neck,
                head,
                base_ckpt,
                args,
                train_group,
            )

        summary = calc_fold_summary(train_group, train_rows, test_rows, best_loss, history)
        fold_summaries.append(summary)
        with open(os.path.join(fold_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)

        print_metric_block(
            f"Fold {fold_idx} Test Metrics",
            {k[5:]: v for k, v in summary.items() if k.startswith("test_")},
            [
                "match_avg_srcc",
                "match_avg_rl2",
                "winner_side_acc",
                "team_srcc",
                "team_rl2",
                "team_mae",
                "team_mse",
            ],
        )

    aggregate = aggregate_repeat_metrics(all_test_rows, fold_summaries)
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2, sort_keys=True)
    write_fold_summary_csv(os.path.join(args.output_dir, "fold_summary.csv"), fold_summaries)

    all_rows_path = os.path.join(args.output_dir, "all_test_preds.csv")
    write_prediction_rows(all_rows_path, all_test_rows)
    match_rows = build_repeated_match_rows(all_test_rows)
    write_match_rows(os.path.join(args.output_dir, "all_test_match_preds.csv"), match_rows)

    output_root = os.path.join(args.output_dir, "all_test")
    save_scatter_plot(
        output_root + "_team_scatter.png",
        [row["true_0_100"] for row in all_test_rows],
        [row["pred_0_100"] for row in all_test_rows],
        "Real-World Fine-Tune CV: Prediction vs Ground Truth",
        "True score (0-100)",
        "Pred score (0-100)",
    )
    save_scatter_plot(
        output_root + "_match_avg_scatter.png",
        [row["avg_true"] for row in match_rows],
        [row["avg_pred"] for row in match_rows],
        "Real-World Fine-Tune CV: Match Avg Prediction vs Ground Truth",
        "True avg score (0-100)",
        "Pred avg score (0-100)",
    )

    print_table(
        "Fold Summary",
        fold_summaries,
        [
            ("train_group", "train_group"),
            ("num_train_rows", "num_train"),
            ("num_test_rows", "num_test"),
            ("best_train_mse", "best_train_mse"),
            ("test_match_avg_srcc", "test_match_srcc"),
            ("test_match_avg_rl2", "test_match_rl2"),
            ("test_team_srcc", "test_team_srcc"),
            ("test_team_rl2", "test_team_rl2"),
            ("test_team_mae", "test_team_mae"),
            ("test_winner_side_acc", "test_winner_acc"),
        ],
    )
    print_table(
        "All Test Match Rows",
        match_rows,
        [
            ("fold", "fold"),
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

    overall = aggregate.get("all_fold_test_predictions", {})
    print_metric_block(
        "Overall Repeated-Test Metrics",
        overall,
        [
            "match_avg_srcc",
            "match_avg_rl2",
            "winner_side_acc",
            "team_srcc",
            "team_rl2",
            "team_mae",
            "team_mse",
        ],
    )
    print(f"[RealWorldFT] Wrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
