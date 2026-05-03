import argparse
import glob
import json
import os
from functools import lru_cache

import cv2
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


TEAM_COLORS = {
    "home": (35, 220, 80),
    "away": (255, 80, 80),
    "goalkeeper": (255, 210, 40),
    "referee": (60, 160, 255),
    "unknown": (220, 220, 220),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Gradio viewer for real-world FootballAQA data.")
    parser.add_argument("--data-root", default="real_word_data")
    parser.add_argument("--game-root", default="features")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _video_label(path):
    base = os.path.splitext(os.path.basename(path))[0]
    if base.endswith("_1"):
        return "1st half"
    if base.endswith("_2"):
        return "2nd half"
    return base


def scan_real_world_data(data_root):
    matches = {}
    for match_dir in sorted(glob.glob(os.path.join(data_root, "*"))):
        if not os.path.isdir(match_dir):
            continue
        match_id = os.path.basename(match_dir)
        stats_path = os.path.join(match_dir, f"{match_id}.json")
        if not os.path.exists(stats_path):
            stats_path = None

        clips = []
        for video_path in sorted(glob.glob(os.path.join(match_dir, "*.mp4"))):
            base = os.path.splitext(os.path.basename(video_path))[0]
            bbox_path = os.path.join(match_dir, f"{base}_rough_player_positions.json")
            if not os.path.exists(bbox_path):
                continue
            clips.append(
                {
                    "label": _video_label(video_path),
                    "video_path": video_path,
                    "bbox_path": bbox_path,
                }
            )

        if clips:
            matches[match_id] = {
                "match_id": match_id,
                "match_dir": match_dir,
                "stats_path": stats_path,
                "clips": clips,
            }
    return matches


def scan_game_bbox_files(game_root):
    bbox_paths = sorted(glob.glob(os.path.join(game_root, "*", "*_rough_player_positions.json")))
    items = {}
    for path in bbox_paths:
        rel = os.path.relpath(path, game_root)
        label = rel[: -len("_rough_player_positions.json")]
        items[label] = path
    return items


@lru_cache(maxsize=32)
def cached_json(path):
    return load_json(path)


def get_clip(matches, match_id, clip_label):
    if not match_id or match_id not in matches:
        return None
    for clip in matches[match_id]["clips"]:
        if clip["label"] == clip_label:
            return clip
    return matches[match_id]["clips"][0] if matches[match_id]["clips"] else None


def get_frame_info(bbox_data, sampled_index):
    frames = bbox_data.get("frames", [])
    if not frames:
        return None
    sampled_index = int(np.clip(sampled_index, 0, len(frames) - 1))
    return frames[sampled_index]


def read_video_frame(video_path, frame_idx):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def draw_box(image, bbox, color, label, show_labels, thickness):
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    h, w = image.shape[:2]
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w - 1, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h - 1, y2))
    cv2.rectangle(image, (x1, y1), (x2, y2), color, int(thickness))
    if show_labels and label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.45
        text_thick = 1
        (tw, th), _ = cv2.getTextSize(label, font, scale, text_thick)
        y_text = max(0, y1 - th - 6)
        cv2.rectangle(image, (x1, y_text), (x1 + tw + 6, y_text + th + 6), color, -1)
        cv2.putText(
            image,
            label,
            (x1 + 3, y_text + th + 2),
            font,
            scale,
            (0, 0, 0),
            text_thick,
            cv2.LINE_AA,
        )


def draw_detections(frame_rgb, frame_info, visible_layers, conf_threshold, show_labels, thickness):
    image = frame_rgb.copy()
    counts = {"home": 0, "away": 0, "goalkeeper": 0, "referee": 0, "unknown": 0}

    if "players" in visible_layers:
        for player in frame_info.get("players", []):
            conf = float(player.get("conf", 1.0))
            if conf < conf_threshold:
                continue
            team = player.get("team", "unknown")
            if team not in {"home", "away"}:
                team = "unknown"
            counts[team] += 1
            label = f"{team} {conf:.2f}"
            draw_box(image, player.get("bbox", [0, 0, 0, 0]), TEAM_COLORS[team], label, show_labels, thickness)

    if "goalkeepers" in visible_layers:
        for player in frame_info.get("goalkeepers", []):
            conf = float(player.get("conf", 1.0))
            if conf < conf_threshold:
                continue
            counts["goalkeeper"] += 1
            draw_box(
                image,
                player.get("bbox", [0, 0, 0, 0]),
                TEAM_COLORS["goalkeeper"],
                f"GK {conf:.2f}",
                show_labels,
                thickness,
            )

    if "referees" in visible_layers:
        for player in frame_info.get("referees", []):
            conf = float(player.get("conf", 1.0))
            if conf < conf_threshold:
                continue
            counts["referee"] += 1
            draw_box(
                image,
                player.get("bbox", [0, 0, 0, 0]),
                TEAM_COLORS["referee"],
                f"REF {conf:.2f}",
                show_labels,
                thickness,
            )

    return image, counts


def match_summary(matches, match_id):
    if not match_id or match_id not in matches:
        return "No match selected."
    match = matches[match_id]
    lines = [f"match_id: {match_id}", f"clips: {len(match['clips'])}"]
    if match.get("stats_path") and os.path.exists(match["stats_path"]):
        stats = cached_json(match["stats_path"])
        info = stats.get("match", {})
        lines.append(f"teams: {info.get('home_team', '?')} vs {info.get('away_team', '?')}")
        score = info.get("score", {})
        if score:
            lines.append(f"score: {score.get('home', '?')} - {score.get('away', '?')}")
        perf = stats.get("performance_scores", {})
        for team, values in perf.items():
            lines.append(
                f"{team}: overall={values.get('score_0_100', '?')}, "
                f"attack={values.get('attack', '?')}, "
                f"control={values.get('control', '?')}, "
                f"defense={values.get('defense', '?')}"
            )
    return "\n".join(lines)


def render_frame(
    matches,
    match_id,
    clip_label,
    sampled_index,
    visible_layers,
    conf_threshold,
    show_labels,
    thickness,
):
    clip = get_clip(matches, match_id, clip_label)
    if clip is None:
        return None, "No clip selected.", []

    bbox_data = cached_json(clip["bbox_path"])
    frame_info = get_frame_info(bbox_data, sampled_index)
    if frame_info is None:
        return None, "No bbox frames found.", []

    raw_frame_idx = int(frame_info.get("frame_idx", 0))
    frame_rgb = read_video_frame(clip["video_path"], raw_frame_idx)
    if frame_rgb is None:
        return None, f"Failed to read video frame {raw_frame_idx}.", []

    image, counts = draw_detections(
        frame_rgb,
        frame_info,
        visible_layers or [],
        float(conf_threshold),
        bool(show_labels),
        int(thickness),
    )
    timestamp = frame_info.get("timestamp")
    if timestamp is None:
        fps = float(bbox_data.get("fps", 30.0) or 30.0)
        timestamp = raw_frame_idx / fps

    info = {
        "match": match_id,
        "clip": clip_label,
        "sampled_index": int(sampled_index),
        "raw_frame_idx": raw_frame_idx,
        "timestamp_sec": round(float(timestamp), 3),
        "extract_stride": bbox_data.get("extract_stride"),
        "extract_fps": bbox_data.get("extract_fps"),
        "visible_counts": counts,
        "video": clip["video_path"],
        "bbox": clip["bbox_path"],
    }
    rows = [[key, value] for key, value in counts.items()]
    return Image.fromarray(image), json.dumps(info, indent=2), rows


def update_match(matches, match_id):
    if not match_id or match_id not in matches:
        return gr.update(choices=[], value=None), gr.update(maximum=0, value=0), None, "No match selected."
    clips = matches[match_id]["clips"]
    labels = [clip["label"] for clip in clips]
    first_label = labels[0] if labels else None
    max_index = 0
    video_path = None
    if first_label is not None:
        clip = clips[0]
        bbox_data = cached_json(clip["bbox_path"])
        max_index = max(0, len(bbox_data.get("frames", [])) - 1)
        video_path = clip["video_path"]
    return (
        gr.update(choices=labels, value=first_label),
        gr.update(minimum=0, maximum=max_index, value=0, step=1),
        video_path,
        match_summary(matches, match_id),
    )


def update_clip(matches, match_id, clip_label):
    clip = get_clip(matches, match_id, clip_label)
    if clip is None:
        return gr.update(maximum=0, value=0), None
    bbox_data = cached_json(clip["bbox_path"])
    max_index = max(0, len(bbox_data.get("frames", [])) - 1)
    return gr.update(minimum=0, maximum=max_index, value=0, step=1), clip["video_path"]


def shift_frame(value, delta, maximum):
    value = int(value or 0) + int(delta)
    maximum = int(maximum or 0)
    return max(0, min(maximum, value))


def _frame_team_counts(frame, conf_threshold=0.0):
    home = 0
    away = 0
    for player in frame.get("players", []):
        if float(player.get("conf", 1.0)) < conf_threshold:
            continue
        if player.get("team") == "home":
            home += 1
        elif player.get("team") == "away":
            away += 1
    return home, away


@lru_cache(maxsize=512)
def compute_window_count_series(bbox_path, window_frames=8, stride=4, conf_threshold=0.0):
    bbox_data = cached_json(bbox_path)
    frames = bbox_data.get("frames", [])
    if len(frames) < window_frames:
        return {
            "x": np.asarray([], dtype=np.float32),
            "home": np.asarray([], dtype=np.float32),
            "away": np.asarray([], dtype=np.float32),
            "home_ratio": np.asarray([], dtype=np.float32),
            "away_ratio": np.asarray([], dtype=np.float32),
            "total": np.asarray([], dtype=np.float32),
            "summary": {},
        }

    fps = float(bbox_data.get("fps", 30.0) or 30.0)
    xs = []
    home_counts = []
    away_counts = []
    for start in range(0, len(frames) - window_frames + 1, stride):
        window = frames[start : start + window_frames]
        home = 0
        away = 0
        for frame in window:
            h, a = _frame_team_counts(frame, conf_threshold)
            home += h
            away += a
        mid = window[len(window) // 2]
        timestamp = mid.get("timestamp")
        if timestamp is None:
            timestamp = float(mid.get("frame_idx", start)) / fps
        xs.append(float(timestamp))
        home_counts.append(float(home))
        away_counts.append(float(away))

    x = np.asarray(xs, dtype=np.float32)
    home = np.asarray(home_counts, dtype=np.float32)
    away = np.asarray(away_counts, dtype=np.float32)
    total = home + away
    home_ratio = np.divide(home, total, out=np.zeros_like(home), where=total > 0)
    away_ratio = np.divide(away, total, out=np.zeros_like(away), where=total > 0)
    non_empty = total > 0
    summary = {
        "windows": int(total.shape[0]),
        "mean_home_nodes": float(home.mean()) if home.size else 0.0,
        "mean_away_nodes": float(away.mean()) if away.size else 0.0,
        "mean_total_nodes": float(total.mean()) if total.size else 0.0,
        "empty_window_ratio": float((total == 0).mean()) if total.size else 0.0,
        "mean_home_ratio": float(home_ratio[non_empty].mean()) if non_empty.any() else 0.0,
        "mean_away_ratio": float(away_ratio[non_empty].mean()) if non_empty.any() else 0.0,
        "extract_stride": bbox_data.get("extract_stride"),
        "extract_fps": bbox_data.get("extract_fps"),
        "bbox_path": bbox_path,
    }
    return {
        "x": x,
        "home": home,
        "away": away,
        "home_ratio": home_ratio,
        "away_ratio": away_ratio,
        "total": total,
        "summary": summary,
    }


def _plot_series(ax, series, prefix, count_kind, linestyle="-"):
    x = series["x"]
    if x.size == 0:
        return
    if count_kind == "total":
        ax.plot(x, series["total"], linestyle=linestyle, label=f"{prefix} total")
    elif count_kind == "home/away":
        ax.plot(x, series["home"], linestyle=linestyle, label=f"{prefix} home")
        ax.plot(x, series["away"], linestyle=linestyle, label=f"{prefix} away")
    else:
        ax.plot(x, series["home_ratio"], linestyle=linestyle, label=f"{prefix} home ratio")
        ax.plot(x, series["away_ratio"], linestyle=linestyle, label=f"{prefix} away ratio")


def render_window_stats(
    matches,
    game_items,
    real_match_id,
    real_clip_label,
    game_label,
    window_frames,
    stride,
    conf_threshold,
    count_kind,
):
    real_clip = get_clip(matches, real_match_id, real_clip_label)
    if real_clip is None:
        return None, "No real-world clip selected."
    if not game_label or game_label not in game_items:
        return None, "No game bbox selected."

    real_series = compute_window_count_series(
        real_clip["bbox_path"], int(window_frames), int(stride), float(conf_threshold)
    )
    game_series = compute_window_count_series(
        game_items[game_label], int(window_frames), int(stride), float(conf_threshold)
    )

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=False)
    _plot_series(axes[0], real_series, "real", count_kind, "-")
    _plot_series(axes[0], game_series, "game", count_kind, "--")
    axes[0].set_title(f"Window {count_kind}: real-world vs game data")
    axes[0].set_xlabel("time (seconds)")
    axes[0].set_ylabel("ratio" if count_kind == "ratio" else "detections per window")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best")

    if real_series["x"].size:
        axes[1].plot(real_series["x"], real_series["total"], label="real total", color="tab:blue")
    if game_series["x"].size:
        axes[1].plot(game_series["x"], game_series["total"], label="game total", color="tab:orange")
    axes[1].set_title("Total home+away detections per window")
    axes[1].set_xlabel("time (seconds)")
    axes[1].set_ylabel("detections per window")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="best")
    fig.tight_layout()

    summary = {
        "window_frames": int(window_frames),
        "stride": int(stride),
        "conf_threshold": float(conf_threshold),
        "real": real_series["summary"],
        "game": game_series["summary"],
    }
    return fig, json.dumps(summary, indent=2)


def _all_real_clip_paths(matches):
    paths = []
    for match_id in sorted(matches.keys()):
        for clip in matches[match_id]["clips"]:
            paths.append((f"{match_id}/{clip['label']}", clip["bbox_path"]))
    return paths


def _summarize_group(dataset, side, node_values, ratio_values, total_values, num_sources):
    nodes = np.concatenate(node_values) if node_values else np.asarray([], dtype=np.float32)
    ratios = np.concatenate(ratio_values) if ratio_values else np.asarray([], dtype=np.float32)
    totals = np.concatenate(total_values) if total_values else np.asarray([], dtype=np.float32)
    non_empty_ratios = ratios[totals > 0] if ratios.size and totals.size == ratios.size else ratios
    return {
        "dataset": dataset,
        "side": side,
        "sources": int(num_sources),
        "windows": int(nodes.shape[0]),
        "mean_nodes_per_window": float(nodes.mean()) if nodes.size else 0.0,
        "std_nodes_per_window": float(nodes.std()) if nodes.size else 0.0,
        "mean_ratio": float(non_empty_ratios.mean()) if non_empty_ratios.size else 0.0,
        "std_ratio": float(non_empty_ratios.std()) if non_empty_ratios.size else 0.0,
        "empty_window_ratio": float((totals == 0).mean()) if totals.size else 0.0,
    }


def aggregate_dataset_window_stats(dataset, path_items, window_frames, stride, conf_threshold):
    grouped_nodes = {"home": [], "away": []}
    grouped_ratios = {"home": [], "away": []}
    grouped_totals = {"home": [], "away": []}
    used_sources = 0
    for _, bbox_path in path_items:
        series = compute_window_count_series(
            bbox_path, int(window_frames), int(stride), float(conf_threshold)
        )
        if series["total"].size == 0:
            continue
        used_sources += 1
        grouped_nodes["home"].append(series["home"])
        grouped_nodes["away"].append(series["away"])
        grouped_ratios["home"].append(series["home_ratio"])
        grouped_ratios["away"].append(series["away_ratio"])
        grouped_totals["home"].append(series["total"])
        grouped_totals["away"].append(series["total"])

    return [
        _summarize_group(
            dataset,
            side,
            grouped_nodes[side],
            grouped_ratios[side],
            grouped_totals[side],
            used_sources,
        )
        for side in ("home", "away")
    ]


def render_aggregate_window_stats(matches, game_items, window_frames, stride, conf_threshold):
    real_rows = aggregate_dataset_window_stats(
        "real", _all_real_clip_paths(matches), window_frames, stride, conf_threshold
    )
    game_rows = aggregate_dataset_window_stats(
        "game", sorted(game_items.items()), window_frames, stride, conf_threshold
    )
    rows = real_rows + game_rows

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    labels = [f"{row['dataset']}/{row['side']}" for row in rows]
    node_values = [row["mean_nodes_per_window"] for row in rows]
    ratio_values = [row["mean_ratio"] for row in rows]
    colors = ["#2ca02c", "#d62728", "#98df8a", "#ff9896"]

    axes[0].bar(labels, node_values, color=colors)
    axes[0].set_title("Mean detections per graph window")
    axes[0].set_ylabel("detections per window")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].grid(True, axis="y", alpha=0.25)

    axes[1].bar(labels, ratio_values, color=colors)
    axes[1].set_title("Mean side ratio per graph window")
    axes[1].set_ylabel("side / (home + away)")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].grid(True, axis="y", alpha=0.25)
    fig.tight_layout()

    table = [
        [
            row["dataset"],
            row["side"],
            row["sources"],
            row["windows"],
            row["mean_nodes_per_window"],
            row["std_nodes_per_window"],
            row["mean_ratio"],
            row["std_ratio"],
            row["empty_window_ratio"],
        ]
        for row in rows
    ]
    summary = {
        "window_frames": int(window_frames),
        "stride": int(stride),
        "conf_threshold": float(conf_threshold),
        "groups": rows,
    }
    return fig, table, json.dumps(summary, indent=2)


def _imbalance_from_series(series):
    total = series["total"]
    diff = np.abs(series["home"] - series["away"])
    return np.divide(diff, total, out=np.ones_like(diff), where=total > 0)


def _select_global_balanced_windows(series, max_clips, min_side_boxes):
    num_windows = int(series["total"].shape[0])
    if num_windows == 0:
        return np.asarray([], dtype=int)
    if num_windows <= max_clips:
        return np.arange(num_windows, dtype=int)

    home = series["home"]
    away = series["away"]
    total = series["total"]
    imbalance = _imbalance_from_series(series)
    candidates = np.where((home >= min_side_boxes) & (away >= min_side_boxes))[0]
    if candidates.size < max_clips:
        candidates = np.arange(num_windows, dtype=int)
    order = np.lexsort((-total[candidates], imbalance[candidates]))
    return np.sort(candidates[order[:max_clips]]).astype(int)


def _select_time_bin_balanced_windows(series, max_clips, min_side_boxes):
    num_windows = int(series["total"].shape[0])
    if num_windows == 0:
        return np.asarray([], dtype=int)
    if num_windows <= max_clips:
        return np.arange(num_windows, dtype=int)

    home = series["home"]
    away = series["away"]
    total = series["total"]
    imbalance = _imbalance_from_series(series)
    selected = []
    bin_edges = np.linspace(0, num_windows, max_clips + 1, dtype=int)
    for bin_idx in range(max_clips):
        start = int(bin_edges[bin_idx])
        end = int(bin_edges[bin_idx + 1])
        if end <= start:
            end = min(start + 1, num_windows)
        candidates = np.arange(start, end, dtype=int)
        eligible = candidates[
            (home[candidates] >= min_side_boxes)
            & (away[candidates] >= min_side_boxes)
        ]
        if eligible.size == 0:
            eligible = candidates
        order = np.lexsort((-total[eligible], imbalance[eligible]))
        selected.append(int(eligible[order[0]]))
    return np.asarray(selected, dtype=int)


def _sampling_summary(name, series, indices):
    x = series["x"][indices] if indices.size else np.asarray([], dtype=np.float32)
    home = series["home"][indices] if indices.size else np.asarray([], dtype=np.float32)
    away = series["away"][indices] if indices.size else np.asarray([], dtype=np.float32)
    total = series["total"][indices] if indices.size else np.asarray([], dtype=np.float32)
    imbalance = _imbalance_from_series(series)[indices] if indices.size else np.asarray([], dtype=np.float32)
    gaps = np.diff(x) if x.size > 1 else np.asarray([], dtype=np.float32)
    return {
        "method": name,
        "selected_windows": int(indices.size),
        "first_time_sec": float(x.min()) if x.size else 0.0,
        "last_time_sec": float(x.max()) if x.size else 0.0,
        "time_span_sec": float(x.max() - x.min()) if x.size else 0.0,
        "mean_time_gap_sec": float(gaps.mean()) if gaps.size else 0.0,
        "std_time_gap_sec": float(gaps.std()) if gaps.size else 0.0,
        "mean_home_boxes": float(home.mean()) if home.size else 0.0,
        "mean_away_boxes": float(away.mean()) if away.size else 0.0,
        "mean_total_boxes": float(total.mean()) if total.size else 0.0,
        "mean_abs_home_away_diff": float(np.abs(home - away).mean()) if home.size else 0.0,
        "mean_imbalance": float(imbalance.mean()) if imbalance.size else 0.0,
        "std_imbalance": float(imbalance.std()) if imbalance.size else 0.0,
    }


def render_sampling_timeline(
    matches,
    real_match_id,
    real_clip_label,
    window_frames,
    stride,
    conf_threshold,
    max_clips,
    min_side_boxes,
):
    real_clip = get_clip(matches, real_match_id, real_clip_label)
    if real_clip is None:
        return None, [], "No real-world clip selected."

    series = compute_window_count_series(
        real_clip["bbox_path"], int(window_frames), int(stride), float(conf_threshold)
    )
    if series["x"].size == 0:
        return None, [], "No windows available for this clip."

    max_clips = max(1, int(max_clips))
    min_side_boxes = max(0, int(min_side_boxes))
    global_idx = _select_global_balanced_windows(series, max_clips, min_side_boxes)
    timebin_idx = _select_time_bin_balanced_windows(series, max_clips, min_side_boxes)
    imbalance = _imbalance_from_series(series)

    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    axes[0].plot(series["x"], series["home"], label="home boxes", color="tab:green", alpha=0.85)
    axes[0].plot(series["x"], series["away"], label="away boxes", color="tab:red", alpha=0.85)
    axes[0].scatter(
        series["x"][global_idx],
        series["home"][global_idx],
        s=18,
        marker="x",
        color="tab:blue",
        label="global balanced samples",
        zorder=3,
    )
    axes[0].scatter(
        series["x"][timebin_idx],
        series["away"][timebin_idx],
        s=18,
        marker="o",
        facecolors="none",
        edgecolors="tab:purple",
        label="time-bin balanced samples",
        zorder=3,
    )
    axes[0].set_title("Home/Away bbox counts and sampled windows")
    axes[0].set_ylabel("boxes per window")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best")

    axes[1].plot(series["x"], imbalance, color="tab:gray", label="imbalance")
    axes[1].scatter(
        series["x"][global_idx],
        imbalance[global_idx],
        s=24,
        marker="x",
        color="tab:blue",
        label="global balanced",
        zorder=3,
    )
    axes[1].scatter(
        series["x"][timebin_idx],
        imbalance[timebin_idx],
        s=24,
        marker="o",
        facecolors="none",
        edgecolors="tab:purple",
        label="time-bin balanced",
        zorder=3,
    )
    axes[1].set_title("Window imbalance = |home - away| / (home + away)")
    axes[1].set_ylabel("imbalance")
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="best")

    axes[2].scatter(series["x"][global_idx], np.ones_like(global_idx), s=20, marker="|", color="tab:blue")
    axes[2].scatter(series["x"][timebin_idx], np.zeros_like(timebin_idx), s=20, marker="|", color="tab:purple")
    axes[2].set_yticks([0, 1])
    axes[2].set_yticklabels(["time-bin balanced", "global balanced"])
    axes[2].set_xlabel("time (seconds)")
    axes[2].set_title("Sampled windows on the timeline")
    axes[2].grid(True, axis="x", alpha=0.25)
    fig.tight_layout()

    summaries = [
        _sampling_summary("global_balanced", series, global_idx),
        _sampling_summary("time_bin_balanced", series, timebin_idx),
    ]
    table = [
        [
            row["method"],
            row["selected_windows"],
            row["first_time_sec"],
            row["last_time_sec"],
            row["time_span_sec"],
            row["mean_time_gap_sec"],
            row["std_time_gap_sec"],
            row["mean_home_boxes"],
            row["mean_away_boxes"],
            row["mean_imbalance"],
            row["std_imbalance"],
        ]
        for row in summaries
    ]
    summary = {
        "window_frames": int(window_frames),
        "stride": int(stride),
        "conf_threshold": float(conf_threshold),
        "max_clips": max_clips,
        "min_side_boxes": min_side_boxes,
        "all_windows": int(series["x"].size),
        "overlap_selected_windows": int(np.intersect1d(global_idx, timebin_idx).size),
        "bbox_path": real_clip["bbox_path"],
        "methods": summaries,
    }
    return fig, table, json.dumps(summary, indent=2)


def build_app(matches, game_items):
    match_ids = sorted(matches.keys())
    default_match = match_ids[0] if match_ids else None
    default_clips = [clip["label"] for clip in matches[default_match]["clips"]] if default_match else []
    default_clip = default_clips[0] if default_clips else None
    default_max = 0
    default_video = None
    if default_match and default_clip:
        clip = get_clip(matches, default_match, default_clip)
        bbox_data = cached_json(clip["bbox_path"])
        default_max = max(0, len(bbox_data.get("frames", [])) - 1)
        default_video = clip["video_path"]

    game_labels = sorted(game_items.keys())
    default_game = game_labels[0] if game_labels else None

    with gr.Blocks(title="Real-World FootballAQA Viewer") as demo:
        gr.Markdown("# Real-World FootballAQA Viewer")
        with gr.Tab("Frame Viewer"):
            with gr.Row():
                with gr.Column(scale=1):
                    match_dd = gr.Dropdown(match_ids, value=default_match, label="Match")
                    clip_dd = gr.Dropdown(default_clips, value=default_clip, label="Clip")
                    video = gr.Video(value=default_video, label="Source video", height=220)
                    frame_slider = gr.Slider(
                        minimum=0,
                        maximum=default_max,
                        value=0,
                        step=1,
                        label="Sampled bbox frame index",
                    )
                    with gr.Row():
                        prev_btn = gr.Button("Prev")
                        next_btn = gr.Button("Next")
                    layers = gr.CheckboxGroup(
                        choices=["players", "goalkeepers", "referees"],
                        value=["players", "referees"],
                        label="Layers",
                    )
                    conf = gr.Slider(0.0, 1.0, value=0.0, step=0.01, label="Confidence threshold")
                    show_labels = gr.Checkbox(value=True, label="Show labels")
                    thickness = gr.Slider(1, 6, value=2, step=1, label="Box thickness")
                    match_text = gr.Textbox(
                        value=match_summary(matches, default_match),
                        label="Match labels / scores",
                        lines=8,
                    )
                with gr.Column(scale=2):
                    image = gr.Image(label="Frame with boxes", type="pil", height=620)
                    info = gr.Code(label="Frame metadata", language="json")
                    counts = gr.Dataframe(headers=["class", "count"], label="Visible counts")

        with gr.Tab("Window Counts"):
            with gr.Row():
                with gr.Column(scale=1):
                    stats_real_match = gr.Dropdown(match_ids, value=default_match, label="Real-world match")
                    stats_real_clip = gr.Dropdown(default_clips, value=default_clip, label="Real-world clip")
                    stats_game = gr.Dropdown(game_labels, value=default_game, label="Game-data bbox")
                    stats_window = gr.Slider(1, 64, value=8, step=1, label="Window frames")
                    stats_stride = gr.Slider(1, 64, value=4, step=1, label="Window stride")
                    stats_conf = gr.Slider(0.0, 1.0, value=0.0, step=0.01, label="Confidence threshold")
                    stats_kind = gr.Radio(
                        choices=["home/away", "total", "ratio"],
                        value="home/away",
                        label="Y value",
                    )
                    stats_btn = gr.Button("Update stats")
                    stats_summary = gr.Code(label="Window stats summary", language="json")
                with gr.Column(scale=2):
                    stats_plot = gr.Plot(label="Window count / ratio over time")
            gr.Markdown("## Aggregate all clips")
            with gr.Row():
                aggregate_btn = gr.Button("Aggregate all real/game windows")
            with gr.Row():
                aggregate_plot = gr.Plot(label="Aggregate real/home/away vs game/home/away")
                aggregate_table = gr.Dataframe(
                    headers=[
                        "dataset",
                        "side",
                        "sources",
                        "windows",
                        "mean_nodes_per_window",
                        "std_nodes_per_window",
                        "mean_ratio",
                        "std_ratio",
                        "empty_window_ratio",
                    ],
                    label="Aggregate means",
                )
            aggregate_summary = gr.Code(label="Aggregate summary", language="json")

        max_state = gr.State(default_max)

        def on_render(match_id, clip_label, frame_idx, layers_value, conf_value, labels_value, thick_value):
            return render_frame(
                matches,
                match_id,
                clip_label,
                frame_idx,
                layers_value,
                conf_value,
                labels_value,
                thick_value,
            )

        controls = [match_dd, clip_dd, frame_slider, layers, conf, show_labels, thickness]
        outputs = [image, info, counts]

        def on_match(match_id):
            clip_update, slider_update, video_path, summary = update_match(matches, match_id)
            max_value = slider_update["maximum"] if isinstance(slider_update, dict) else 0
            return clip_update, slider_update, video_path, summary, max_value

        def on_clip(match_id, clip_label):
            slider_update, video_path = update_clip(matches, match_id, clip_label)
            max_value = slider_update["maximum"] if isinstance(slider_update, dict) else 0
            return slider_update, video_path, max_value

        match_dd.change(
            on_match,
            inputs=[match_dd],
            outputs=[clip_dd, frame_slider, video, match_text, max_state],
        ).then(on_render, inputs=controls, outputs=outputs)

        clip_dd.change(
            on_clip,
            inputs=[match_dd, clip_dd],
            outputs=[frame_slider, video, max_state],
        ).then(on_render, inputs=controls, outputs=outputs)

        for component in [frame_slider, layers, conf, show_labels, thickness]:
            component.change(on_render, inputs=controls, outputs=outputs)

        prev_btn.click(
            lambda value, maximum: shift_frame(value, -1, maximum),
            inputs=[frame_slider, max_state],
            outputs=[frame_slider],
        ).then(on_render, inputs=controls, outputs=outputs)

        next_btn.click(
            lambda value, maximum: shift_frame(value, 1, maximum),
            inputs=[frame_slider, max_state],
            outputs=[frame_slider],
        ).then(on_render, inputs=controls, outputs=outputs)

        demo.load(on_render, inputs=controls, outputs=outputs)

        def on_stats_match(match_id):
            clip_update, _, _, _ = update_match(matches, match_id)
            return clip_update

        stats_real_match.change(
            on_stats_match,
            inputs=[stats_real_match],
            outputs=[stats_real_clip],
        )

        stats_inputs = [
            stats_real_match,
            stats_real_clip,
            stats_game,
            stats_window,
            stats_stride,
            stats_conf,
            stats_kind,
        ]

        def on_stats(real_match_id, real_clip_label, game_label, window_frames, stride, conf_value, kind):
            return render_window_stats(
                matches,
                game_items,
                real_match_id,
                real_clip_label,
                game_label,
                window_frames,
                stride,
                conf_value,
                kind,
            )

        stats_btn.click(on_stats, inputs=stats_inputs, outputs=[stats_plot, stats_summary])
        demo.load(on_stats, inputs=stats_inputs, outputs=[stats_plot, stats_summary])

        def on_aggregate(window_frames, stride, conf_value):
            return render_aggregate_window_stats(
                matches,
                game_items,
                window_frames,
                stride,
                conf_value,
            )

        aggregate_inputs = [stats_window, stats_stride, stats_conf]
        aggregate_outputs = [aggregate_plot, aggregate_table, aggregate_summary]
        aggregate_btn.click(on_aggregate, inputs=aggregate_inputs, outputs=aggregate_outputs)
        demo.load(on_aggregate, inputs=aggregate_inputs, outputs=aggregate_outputs)

        with gr.Tab("Sampling Timeline"):
            with gr.Row():
                with gr.Column(scale=1):
                    sampling_match = gr.Dropdown(match_ids, value=default_match, label="Real-world match")
                    sampling_clip = gr.Dropdown(default_clips, value=default_clip, label="Real-world clip")
                    sampling_window = gr.Slider(1, 64, value=8, step=1, label="Window frames")
                    sampling_stride = gr.Slider(1, 64, value=4, step=1, label="Window stride")
                    sampling_conf = gr.Slider(0.0, 1.0, value=0.0, step=0.01, label="Confidence threshold")
                    sampling_max_clips = gr.Slider(1, 512, value=256, step=1, label="Max sampled windows")
                    sampling_min_side = gr.Slider(0, 64, value=1, step=1, label="Min boxes per side")
                    sampling_btn = gr.Button("Update sampling timeline")
                    sampling_summary = gr.Code(label="Sampling summary", language="json")
                with gr.Column(scale=2):
                    sampling_plot = gr.Plot(label="Sampling methods over time")
                    sampling_table = gr.Dataframe(
                        headers=[
                            "method",
                            "selected_windows",
                            "first_time_sec",
                            "last_time_sec",
                            "time_span_sec",
                            "mean_time_gap_sec",
                            "std_time_gap_sec",
                            "mean_home_boxes",
                            "mean_away_boxes",
                            "mean_imbalance",
                            "std_imbalance",
                        ],
                        label="Sampling comparison",
                    )

        def on_sampling_match(match_id):
            clip_update, _, _, _ = update_match(matches, match_id)
            return clip_update

        sampling_match.change(
            on_sampling_match,
            inputs=[sampling_match],
            outputs=[sampling_clip],
        )

        sampling_inputs = [
            sampling_match,
            sampling_clip,
            sampling_window,
            sampling_stride,
            sampling_conf,
            sampling_max_clips,
            sampling_min_side,
        ]

        def on_sampling(
            real_match_id,
            real_clip_label,
            window_frames,
            stride,
            conf_value,
            max_clips,
            min_side_boxes,
        ):
            return render_sampling_timeline(
                matches,
                real_match_id,
                real_clip_label,
                window_frames,
                stride,
                conf_value,
                max_clips,
                min_side_boxes,
            )

        sampling_outputs = [sampling_plot, sampling_table, sampling_summary]
        sampling_btn.click(on_sampling, inputs=sampling_inputs, outputs=sampling_outputs)
        demo.load(on_sampling, inputs=sampling_inputs, outputs=sampling_outputs)

    return demo


def main():
    args = parse_args()
    matches = scan_real_world_data(args.data_root)
    if not matches:
        raise RuntimeError(f"No match clips found under {args.data_root}")
    game_items = scan_game_bbox_files(args.game_root)
    if not game_items:
        raise RuntimeError(f"No game bbox files found under {args.game_root}")
    app = build_app(matches, game_items)
    app.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
