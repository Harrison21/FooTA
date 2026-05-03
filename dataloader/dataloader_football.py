import os
import json
import random
import torch
import numpy as np
from utils.dataloader_utils import worker_init_fn, get_video_trans


class Football_Dataset(torch.utils.data.Dataset):
    """
    FootballAQA Dataset - supports both video-only and multimodal (video + graph) modes.

    Data layout:
        football_dir/
            1st/
                1st_Arsenal_vs_Sunderland.npy
                1st_Arsenal_vs_Sunderland_team_stats.json
                1st_Arsenal_vs_Sunderland_rough_player_positions.json
            ...

        or, for real-world match folders:
        football_dir/
            2023-08-11_ogc-nice-lille-osc/
                2023-08-11_ogc-nice-lille-osc.npy
                2023-08-11_ogc-nice-lille-osc.json
                ogc-nice-lille-osc_full_rough_player_positions.json

    When ``use_graph_branch=True`` the loader reads bounding-box JSON files
    from the same FootballAQA tree (``bbox_root``), splits each match into
    home-team and away-team samples, and returns per-window graph data aligned
    with the pre-extracted video features.

    Args:
        args: config object with at least:
            - football_dir: root path to the league folder
            - train_rounds: list of round names for training, e.g. ["1st","2nd","3rd","4th","5th"]
            - test_rounds:  list of round names for testing,  e.g. ["6th","7th"]
            - football_max_clips: int, pad/truncate feature sequence length
            - label: str, which score to predict: "overall", "attack", "control", "defense"
        subset: "train" or "test"
        transform: video transforms (unused here, kept for interface compatibility)
    """

    ROUND_NAMES = ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th"]

    def __init__(self, args, subset, transform):
        self.subset = subset
        self.transforms = transform
        self.data_dirs = list(args.get("football_dirs", []))
        if not self.data_dirs:
            self.data_dirs = [args.football_dir]
        self.meta_dirs = list(args.get("football_meta_dirs", self.data_dirs))
        if not self.meta_dirs:
            self.meta_dirs = list(self.data_dirs)
        self.max_clips = args.football_max_clips
        self.label_target = args.label  # "overall", "attack", "control", "defense"
        # "pair" -> [home, away], "avg" -> mean(home, away),
        # "home" -> home only, "away" -> away only, "diff" -> home-away.
        self.target_mode = args.get("football_target_mode", "pair")
        self.round_filter = set(args.get("football_rounds", self.ROUND_NAMES))
        self.random_split = args.get("football_random_split", False)
        self.test_ratio = float(args.get("football_test_ratio", 0.3))
        self.split_seed = int(args.get("football_split_seed", args.get("seed", 42)))
        self.score_strat_bins = int(args.get("football_score_strat_bins", 3))

        # -- graph branch config -----------------------------------------------
        self.use_graph = bool(args.get("use_graph_branch", False))
        self.bbox_root = str(args.get("bbox_root", ""))
        self.graph_num_frames = int(args.get("graph_num_frames", 8))
        self.graph_stride = int(args.get("graph_stride", 4))
        self.graph_max_nodes = int(args.get("graph_max_nodes_per_window", 120))
        self.graph_use_opp_context = bool(args.get("graph_use_opp_context", False))
        self.graph_opp_context_radius = float(args.get("graph_opp_context_radius", 0.2))
        self.graph_motion_dt_normalize = bool(args.get("graph_motion_dt_normalize", False))
        self.graph_motion_dt_mode = str(args.get("graph_motion_dt_mode", "all")).lower()
        _raw_feats = args.get("graph_opp_context_features", None)
        self.graph_opp_context_features = tuple(_raw_feats) if _raw_feats is not None else None
        self.cache_features = bool(args.get("football_cache_features", True))
        self.cache_graph_data = bool(args.get("football_cache_graph_data", True))
        self.precompute_graph_data = bool(
            args.get("football_precompute_graph_data", self.use_graph and self.cache_graph_data)
        )
        self.use_mmap_features = bool(args.get("football_mmap_features", not self.cache_features))
        # Camera-view simulation: randomly mask one side (left/right/top/bottom)
        # to mimic partial pitch visibility in real broadcasts.
        self.camera_mask_enable = bool(args.get("football_camera_mask_enable", False))
        self.camera_mask_prob = float(args.get("football_camera_mask_prob", 1.0))
        self.camera_mask_apply_on = str(args.get("football_camera_mask_apply_on", "train"))
        self.camera_mask_sides = tuple(
            args.get("football_camera_mask_sides", ["left", "right", "top", "bottom"])
        )
        self.camera_mask_ratio_min = float(args.get("football_camera_mask_ratio_min", 0.25))
        self.camera_mask_ratio_max = float(args.get("football_camera_mask_ratio_max", 0.40))
        self.camera_mask_active = (
            self.camera_mask_enable
            and (
                self.camera_mask_apply_on == "all"
                or self.camera_mask_apply_on == self.subset
                or (self.camera_mask_apply_on == "train" and self.subset == "train")
            )
        )
        if self.camera_mask_active and (self.cache_graph_data or self.precompute_graph_data):
            print(
                "[FootballAQA] camera mask augmentation active; disabling "
                "football_cache_graph_data and football_precompute_graph_data for this split."
            )
            self.cache_graph_data = False
            self.precompute_graph_data = False

        # -- image (DINOv3 crop) feature config --------------------------------
        self.use_image_features = bool(args.get("use_image_features", False))
        self.image_feat_dim = int(args.get("image_feat_dim", 768))
        self.image_match_threshold = float(args.get("image_match_threshold", 0.05))
        self.dinov3_feat_root = str(args.get("dinov3_feat_root", args.get("bbox_root", "")))
        self.dinov3_cache_size = int(args.get("dinov3_cache_size", 16))

        self._bbox_map = {}   # video_name -> bbox json path
        self._bbox_cache = {} # video_name -> parsed bbox data
        self._dinov3_map = {} # video_name -> dinov3 npz path
        self._dinov3_cache = {} # video_name -> indexed dinov3 data (LRU, bounded)
        self._dinov3_cache_order = []  # LRU eviction order
        self._feature_cache = {}
        self._feature_len_cache = {}
        self._graph_cache = {}
        self._json_index_cache = {}

        # -- scan game samples -------------------------------------------------
        if self.random_split:
            all_samples = self._scan_rounds(self.ROUND_NAMES)
            train_samples, test_samples = self._stratified_split_by_league(all_samples)
            game_samples = train_samples if subset == "train" else test_samples
        else:
            if subset == "train":
                round_names = list(args.train_rounds)
            else:
                round_names = list(args.test_rounds)
            game_samples = self._scan_rounds(round_names)

        # -- graph: scan bbox files, preload, and expand to team samples --
        if self.use_graph:
            self._bbox_map = self._scan_bbox_files()
            self._preload_bbox_data(game_samples)
            if self.use_image_features:
                self._dinov3_map = self._scan_dinov3_files()
            self.samples = self._expand_to_team_samples(game_samples)
            if self.precompute_graph_data:
                self._precompute_graph_cache()
        else:
            self.samples = game_samples

        # Compute score statistics for denormalization later
        all_scores = self._collect_all_scores()
        self.score_min = min(all_scores) if all_scores else 0
        self.score_max = max(all_scores) if all_scores else 1

    # ------------------------------------------------------------------
    # Score collection helpers
    # ------------------------------------------------------------------
    def _collect_all_scores(self):
        if self.use_graph:
            return [s["score_value"] for s in self.samples]
        return [score for s in self.samples for score in s["scores"]]

    # ------------------------------------------------------------------
    # Bounding-box discovery & loading
    # ------------------------------------------------------------------
    @staticmethod
    def _bbox_priority(path):
        name = os.path.basename(path)
        if "_full_rough_player_positions.json" in name:
            return 3
        return 1

    @staticmethod
    def _bbox_aliases(path, video_name):
        aliases = {video_name}
        parent = os.path.basename(os.path.dirname(path))
        if parent and parent not in Football_Dataset.ROUND_NAMES:
            aliases.add(parent)

        for suffix in ("_full", "_1", "_2"):
            if video_name.endswith(suffix):
                aliases.add(video_name[: -len(suffix)])
        return aliases

    def _add_bbox_file(self, bbox_map, bbox_priority, path, video_name):
        priority = self._bbox_priority(path)
        for key in self._bbox_aliases(path, video_name):
            if not key:
                continue
            if priority >= bbox_priority.get(key, -1):
                bbox_map[key] = path
                bbox_priority[key] = priority

    def _scan_bbox_files(self):
        """Scan ``bbox_root`` for all ``*_rough_player_positions.json`` files.

        Supports both flat layout (bbox_root/League/file.json) and nested
        layout (bbox_root/League/Round/file.json).

        Returns:
            dict mapping video_name -> json_path
        """
        bbox_map = {}
        bbox_priority = {}
        if not self.bbox_root or not os.path.isdir(self.bbox_root):
            print(f"[FootballAQA] Warning: bbox_root not found: {self.bbox_root}")
            return bbox_map

        suffix = "_rough_player_positions.json"
        for league_dir in sorted(os.listdir(self.bbox_root)):
            league_path = os.path.join(self.bbox_root, league_dir)
            if not os.path.isdir(league_path):
                continue
            for entry in sorted(os.listdir(league_path)):
                entry_path = os.path.join(league_path, entry)
                if entry.endswith(suffix):
                    video_name = entry[: -len(suffix)]
                    self._add_bbox_file(bbox_map, bbox_priority, entry_path, video_name)
                elif os.path.isdir(entry_path):
                    for fname in sorted(os.listdir(entry_path)):
                        if fname.endswith(suffix):
                            video_name = fname[: -len(suffix)]
                            self._add_bbox_file(
                                bbox_map,
                                bbox_priority,
                                os.path.join(entry_path, fname),
                                video_name,
                            )
        print(f"[FootballAQA] Found {len(bbox_map)} bbox files under {self.bbox_root}")
        return bbox_map

    def _preload_bbox_data(self, game_samples):
        """Parse bbox JSONs for all matched games and cache per-frame bbox lists."""
        loaded = 0
        for sample in game_samples:
            video_name = self._video_name_from_npy(sample["npy_path"])
            if video_name in self._bbox_cache:
                continue
            bbox_path = self._bbox_map.get(video_name)
            if bbox_path is None:
                continue
            try:
                parsed = self._parse_bbox_json(bbox_path)
                self._bbox_cache[video_name] = parsed
                loaded += 1
            except Exception as e:
                print(f"[FootballAQA] Warning: failed to load {bbox_path}: {e}")
        print(f"[FootballAQA] Loaded bbox data for {loaded} games "
              f"({len(self._bbox_cache)} cached total)")

    # ------------------------------------------------------------------
    # DINOv3 image feature discovery & loading
    # ------------------------------------------------------------------
    def _scan_dinov3_files(self):
        """Scan ``dinov3_feat_root`` for ``*_dinov3_bbox_feats.npz`` files."""
        feat_map = {}
        root = self.dinov3_feat_root
        if not root or not os.path.isdir(root):
            print(f"[FootballAQA] Warning: dinov3_feat_root not found: {root}")
            return feat_map

        suffix = "_dinov3_bbox_feats.npz"
        for dirpath, _dirnames, filenames in os.walk(root):
            for fname in filenames:
                if fname.endswith(suffix):
                    video_name = fname[: -len(suffix)]
                    feat_map[video_name] = os.path.join(dirpath, fname)
        print(f"[FootballAQA] Found {len(feat_map)} DINOv3 feature files under {root}")
        return feat_map

    def _get_dinov3_index(self, video_name):
        """Lazy-load a single match's DINOv3 index with bounded LRU cache."""
        cached = self._dinov3_cache.get(video_name)
        if cached is not None:
            # Move to end (most recently used)
            try:
                self._dinov3_cache_order.remove(video_name)
            except ValueError:
                pass
            self._dinov3_cache_order.append(video_name)
            return cached

        npz_path = self._dinov3_map.get(video_name)
        if npz_path is None:
            return {}

        try:
            parsed = self._parse_dinov3_npz(npz_path)
        except Exception as e:
            print(f"[FootballAQA] Warning: failed to load DINOv3 {npz_path}: {e}")
            return {}

        # Evict oldest entries if cache is full
        while len(self._dinov3_cache) >= self.dinov3_cache_size:
            if self._dinov3_cache_order:
                oldest = self._dinov3_cache_order.pop(0)
                self._dinov3_cache.pop(oldest, None)
            else:
                break

        self._dinov3_cache[video_name] = parsed
        self._dinov3_cache_order.append(video_name)
        return parsed

    def _parse_dinov3_npz(self, npz_path):
        """Parse a DINOv3 npz into a per-frame-per-team spatial index.

        Features are kept as float16 in the cache to save memory (~2x);
        conversion to float32 happens at alignment time.

        Returns:
            dict mapping (frame_idx, team_id) -> {
                "centers_px": np.float32 [K, 2],
                "feats":      np.float16 [K, D],
            }
        """
        data = np.load(npz_path, allow_pickle=False)
        feats = np.asarray(data["feat"], dtype=np.float16)
        frame_idxs = np.asarray(data["frame_idx"], dtype=np.int32)
        team_ids = np.asarray(data["team_id"], dtype=np.uint8)
        bboxes = np.asarray(data["bbox"], dtype=np.float32)

        if feats.ndim != 2 or feats.shape[0] == 0:
            return {}

        # Group by (frame_idx, team_id) using vectorised ops
        keys = frame_idxs.astype(np.int64) * 2 + team_ids.astype(np.int64)
        cx_px = (bboxes[:, 0] + bboxes[:, 2]) / 2.0
        cy_px = (bboxes[:, 1] + bboxes[:, 3]) / 2.0
        centers = np.stack([cx_px, cy_px], axis=1)  # [N, 2]

        unique_keys = np.unique(keys)
        result = {}
        for uk in unique_keys:
            sel = keys == uk
            frame_idx = int(uk // 2)
            team_id = int(uk % 2)
            result[(frame_idx, team_id)] = {
                "centers_px": np.ascontiguousarray(centers[sel], dtype=np.float32),
                "feats": np.ascontiguousarray(feats[sel]),  # stays float16
            }
        return result

    def _align_image_features_for_window(
        self, dinov3_index, team_side, window_frame_indices,
        nodes, mask, frame_ids, frame_width, frame_height,
    ):
        """Align DINOv3 features with graph nodes for one temporal window.

        For each valid graph node, find the DINOv3 crop whose bbox center is
        closest (in normalised coords). Returns a feature matrix aligned with
        the node ordering.

        Returns:
            image_feats: np.float32 [max_nodes, image_feat_dim]
        """
        max_nodes = nodes.shape[0]
        feat_dim = self.image_feat_dim
        image_feats = np.zeros((max_nodes, feat_dim), dtype=np.float32)
        if not dinov3_index:
            return image_feats

        team_id = 0 if team_side == "home" else 1
        threshold = self.image_match_threshold

        for node_idx in range(max_nodes):
            if not mask[node_idx]:
                continue
            f_offset = int(frame_ids[node_idx])
            if f_offset < 0 or f_offset >= len(window_frame_indices):
                continue
            abs_frame = window_frame_indices[f_offset]
            entry = dinov3_index.get((abs_frame, team_id))
            if entry is None:
                continue

            node_cx = nodes[node_idx, 0]
            node_cy = nodes[node_idx, 1]
            centers_px = entry["centers_px"]
            centers_norm_x = centers_px[:, 0] / frame_width
            centers_norm_y = centers_px[:, 1] / frame_height
            dx = centers_norm_x - node_cx
            dy = centers_norm_y - node_cy
            dists = dx * dx + dy * dy
            best_idx = int(np.argmin(dists))
            if dists[best_idx] < threshold * threshold:
                image_feats[node_idx] = entry["feats"][best_idx].astype(np.float32)

        return image_feats

    @staticmethod
    def _video_name_from_npy(npy_path: str) -> str:
        return os.path.splitext(os.path.basename(npy_path))[0]

    @staticmethod
    def _parse_bbox_json(json_path: str) -> dict:
        """Parse a player-positions JSON into compact per-frame bbox lists.

        Returns dict with keys:
            home_bboxes:  list[np.ndarray | None] indexed by frame_idx
            away_bboxes:  list[np.ndarray | None]
            frame_width, frame_height, total_frames: int
        """
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        fw = int(data["frame_width"])
        fh = int(data["frame_height"])
        total = int(data["total_frames"])
        fps = float(data.get("fps", 30.0) or 30.0)
        extract_stride = float(data.get("extract_stride", 1.0) or 1.0)
        extract_fps = data.get("extract_fps", None)
        extract_fps = float(extract_fps) if extract_fps not in (None, 0, 0.0) else None

        home = [None] * total
        away = [None] * total

        for frame in data.get("frames", []):
            fidx = int(frame["frame_idx"])
            if fidx < 0 or fidx >= total:
                continue
            h_list = []
            a_list = []
            for p in frame.get("players", []):
                bbox = p.get("bbox")
                if bbox is None or len(bbox) != 4:
                    continue
                team = p.get("team", "")
                if team == "home":
                    h_list.append(bbox)
                elif team == "away":
                    a_list.append(bbox)
            if h_list:
                home[fidx] = np.array(h_list, dtype=np.float32)
            if a_list:
                away[fidx] = np.array(a_list, dtype=np.float32)

        return {
            "home_bboxes": home,
            "away_bboxes": away,
            "frame_width": fw,
            "frame_height": fh,
            "total_frames": total,
            "fps": fps,
            "extract_stride": extract_stride,
            "extract_fps": extract_fps,
        }

    @staticmethod
    def _resolve_motion_time_delta(bbox_data: dict) -> float:
        extract_fps = bbox_data.get("extract_fps", None)
        if extract_fps is not None and extract_fps > 0.0:
            return 1.0 / extract_fps
        fps = float(bbox_data.get("fps", 30.0) or 30.0)
        extract_stride = float(bbox_data.get("extract_stride", 1.0) or 1.0)
        if fps <= 0.0:
            return 1.0
        return extract_stride / fps

    # ------------------------------------------------------------------
    # Home / away expansion
    # ------------------------------------------------------------------
    def _expand_to_team_samples(self, game_samples):
        """Each game -> two team-level samples (home + away)."""
        team_samples = []
        for game in game_samples:
            video_name = self._video_name_from_npy(game["npy_path"])
            has_bbox = video_name in self._bbox_cache

            for side_idx, side in enumerate(("home", "away")):
                team_name = game["teams"][side_idx]
                score_value = float(game["scores"][side_idx])
                team_samples.append({
                    "npy_path": game["npy_path"],
                    "score_value": score_value,
                    "team_name": team_name,
                    "team_side": side,
                    "game_id": game["game_id"],
                    "json_path": game.get("json_path"),
                    "league": game.get("league", "unknown"),
                    "video_name": video_name,
                    "has_bbox": has_bbox,
                })
        return team_samples

    # ------------------------------------------------------------------
    # Graph windowing
    # ------------------------------------------------------------------
    def _compute_graph_data(self, sample, num_clips_npy, mask_side=None, mask_ratio=0.0):
        """Compute per-window graph arrays for one team sample.

        Returns:
            nodes:      np.float32  [num_clips, max_nodes, node_input_dim]
            mask:       np.bool_    [num_clips, max_nodes]
            frame_ids:  np.int64    [num_clips, max_nodes]
            num_clips:  int  (aligned clip count)
            img_feats:  np.float32  [num_clips, max_nodes, image_feat_dim]
                        (only when use_image_features=True, else None)
        """
        from models.graph.box_encoder import compute_box_features_for_window, opp_feature_dim

        use_opp = bool(getattr(self, "graph_use_opp_context", False))
        opp_radius = float(getattr(self, "graph_opp_context_radius", 0.2))
        opp_feats_sel = getattr(self, "graph_opp_context_features", None)
        node_dim = 10 + (opp_feature_dim(opp_feats_sel) if use_opp else 0)

        video_name = sample["video_name"]
        cache_key = (
            video_name,
            sample["team_side"],
            num_clips_npy,
            use_opp,
            opp_feats_sel,
            self.graph_motion_dt_normalize,
            self.use_image_features,
            mask_side,
            round(float(mask_ratio), 4),
        )
        if self.cache_graph_data and cache_key in self._graph_cache:
            return self._graph_cache[cache_key]
        bbox_data = self._bbox_cache.get(video_name)

        if bbox_data is None:
            nc = num_clips_npy
            img_feats_empty = (
                np.zeros((nc, self.graph_max_nodes, self.image_feat_dim), dtype=np.float32)
                if self.use_image_features else None
            )
            output = (
                np.zeros((nc, self.graph_max_nodes, node_dim), dtype=np.float32),
                np.zeros((nc, self.graph_max_nodes), dtype=np.bool_),
                np.zeros((nc, self.graph_max_nodes), dtype=np.int64),
                nc,
                img_feats_empty,
            )
            if self.cache_graph_data:
                self._graph_cache[cache_key] = output
            return output

        team_bboxes = (bbox_data["home_bboxes"] if sample["team_side"] == "home"
                       else bbox_data["away_bboxes"])
        opp_bboxes = (bbox_data["away_bboxes"] if sample["team_side"] == "home"
                      else bbox_data["home_bboxes"]) if use_opp else None
        fw = bbox_data["frame_width"]
        fh = bbox_data["frame_height"]
        total_frames = bbox_data["total_frames"]
        motion_dt = None
        if self.graph_motion_dt_normalize:
            motion_dt = self._resolve_motion_time_delta(bbox_data)

        nf = self.graph_num_frames
        st = self.graph_stride
        num_clips_bbox = max(0, (total_frames - nf) // st + 1)
        num_clips = min(num_clips_npy, num_clips_bbox)

        all_nodes = np.zeros((num_clips, self.graph_max_nodes, node_dim), dtype=np.float32)
        all_mask = np.zeros((num_clips, self.graph_max_nodes), dtype=np.bool_)
        all_fids = np.zeros((num_clips, self.graph_max_nodes), dtype=np.int64)

        dinov3_index = (
            self._get_dinov3_index(video_name)
            if self.use_image_features else None
        )
        all_img_feats = (
            np.zeros((num_clips, self.graph_max_nodes, self.image_feat_dim), dtype=np.float32)
            if self.use_image_features else None
        )

        use_side_mask = mask_side is not None and mask_ratio > 0.0
        for clip_idx in range(num_clips):
            start = clip_idx * st
            window_frames = list(range(start, start + nf))
            if use_side_mask:
                masked_team = [
                    self._apply_side_mask_to_frame_bboxes(
                        team_bboxes[fidx] if fidx < len(team_bboxes) else None,
                        mask_side,
                        mask_ratio,
                        fw,
                        fh,
                    )
                    for fidx in window_frames
                ]
                masked_opp = None
                if use_opp and opp_bboxes is not None:
                    masked_opp = [
                        self._apply_side_mask_to_frame_bboxes(
                            opp_bboxes[fidx] if fidx < len(opp_bboxes) else None,
                            mask_side,
                            mask_ratio,
                            fw,
                            fh,
                        )
                        for fidx in window_frames
                    ]
                local_frames = list(range(len(window_frames)))
                n, m, fid = compute_box_features_for_window(
                    masked_team, local_frames, fw, fh, self.graph_max_nodes,
                    opp_bboxes_by_frame=masked_opp,
                    opp_context_radius=opp_radius,
                    opp_context_features=opp_feats_sel,
                    motion_time_delta=motion_dt,
                    motion_dt_mode=self.graph_motion_dt_mode,
                )
            else:
                n, m, fid = compute_box_features_for_window(
                    team_bboxes, window_frames, fw, fh, self.graph_max_nodes,
                    opp_bboxes_by_frame=opp_bboxes,
                    opp_context_radius=opp_radius,
                    opp_context_features=opp_feats_sel,
                    motion_time_delta=motion_dt,
                    motion_dt_mode=self.graph_motion_dt_mode,
                )
            all_nodes[clip_idx] = n
            all_mask[clip_idx] = m
            all_fids[clip_idx] = fid

            if self.use_image_features:
                all_img_feats[clip_idx] = self._align_image_features_for_window(
                    dinov3_index, sample["team_side"], window_frames,
                    n, m, fid, fw, fh,
                )

        output = (all_nodes, all_mask, all_fids, num_clips, all_img_feats)
        if self.cache_graph_data:
            self._graph_cache[cache_key] = output
        return output

    def _sample_camera_mask(self):
        """Sample one side and one ratio for camera-view masking."""
        if not self.camera_mask_active:
            return None, 0.0
        if not self.camera_mask_sides:
            return None, 0.0
        if random.random() > max(0.0, min(1.0, self.camera_mask_prob)):
            return None, 0.0

        side = random.choice(self.camera_mask_sides)
        lo = max(0.0, min(1.0, self.camera_mask_ratio_min))
        hi = max(0.0, min(1.0, self.camera_mask_ratio_max))
        if hi < lo:
            lo, hi = hi, lo
        ratio = random.uniform(lo, hi) if hi > lo else lo
        return side, ratio

    @staticmethod
    def _apply_side_mask_to_frame_bboxes(bboxes, side, ratio, frame_width, frame_height):
        """Filter one frame's bboxes by masking one border side."""
        if bboxes is None or len(bboxes) == 0:
            return None
        if side is None or ratio <= 0.0:
            return bboxes

        arr = np.asarray(bboxes, dtype=np.float32)
        cx = (arr[:, 0] + arr[:, 2]) / 2.0
        cy = (arr[:, 1] + arr[:, 3]) / 2.0

        if side == "left":
            keep = cx >= (frame_width * ratio)
        elif side == "right":
            keep = cx <= (frame_width * (1.0 - ratio))
        elif side == "top":
            keep = cy >= (frame_height * ratio)
        elif side == "bottom":
            keep = cy <= (frame_height * (1.0 - ratio))
        else:
            keep = np.ones(arr.shape[0], dtype=np.bool_)

        if not np.any(keep):
            return None
        return np.ascontiguousarray(arr[keep], dtype=np.float32)

    def _load_feature_array(self, feat_path):
        if self.cache_features and feat_path in self._feature_cache:
            return self._feature_cache[feat_path]

        load_kwargs = {}
        if self.use_mmap_features and not self.cache_features:
            load_kwargs["mmap_mode"] = "r"
        feats = np.load(feat_path, **load_kwargs)
        feats = np.asarray(feats, dtype=np.float32)
        self._feature_len_cache[feat_path] = int(feats.shape[0])
        if self.cache_features:
            self._feature_cache[feat_path] = feats
        return feats

    def _get_num_clips(self, feat_path):
        cached = self._feature_len_cache.get(feat_path)
        if cached is not None:
            return cached
        load_kwargs = {"mmap_mode": "r"} if self.use_mmap_features else {}
        feats = np.load(feat_path, **load_kwargs)
        num_clips = int(feats.shape[0])
        self._feature_len_cache[feat_path] = num_clips
        return num_clips

    def _precompute_graph_cache(self):
        if not self.samples:
            return

        cached_keys = len(self._graph_cache)
        for sample in self.samples:
            feat_path = sample["npy_path"]
            num_clips_npy = self._get_num_clips(feat_path)
            self._compute_graph_data(sample, num_clips_npy)
        print(
            f"[FootballAQA] Precomputed graph windows for {len(self.samples)} team samples "
            f"({len(self._graph_cache) - cached_keys} new cache entries)"
        )

    # ------------------------------------------------------------------
    # Original scan / build / split helpers  (unchanged logic)
    # ------------------------------------------------------------------
    def _get_meta_dir_for_data_dir(self, data_idx):
        if data_idx < len(self.meta_dirs):
            return self.meta_dirs[data_idx]
        return self.meta_dirs[-1] if self.meta_dirs else self.data_dirs[data_idx]

    @staticmethod
    def _is_stats_json_name(fname):
        if not fname.endswith(".json"):
            return False
        if fname.endswith("_rough_player_positions.json"):
            return False
        if fname.endswith("_pitch_projected.json"):
            return False
        return True

    @staticmethod
    def _stats_game_key(fname):
        if fname.endswith("_team_stats.json"):
            return fname[: -len("_team_stats.json")]
        return os.path.splitext(fname)[0]

    def _get_json_index(self, meta_dir):
        """Build and cache game_key -> stats JSON mapping for one league."""
        if meta_dir in self._json_index_cache:
            return self._json_index_cache[meta_dir]

        index = {}
        if not meta_dir or not os.path.isdir(meta_dir):
            self._json_index_cache[meta_dir] = index
            return index

        for dirpath, _dirnames, filenames in os.walk(meta_dir):
            for fname in filenames:
                if not self._is_stats_json_name(fname):
                    continue
                game_key = self._stats_game_key(fname)
                full_path = os.path.join(dirpath, fname)
                if game_key not in index:
                    index[game_key] = full_path

        self._json_index_cache[meta_dir] = index
        return index

    def _scan_rounds(self, round_names):
        """Collect one sample per game from either nested or flat folder layout."""
        samples = []
        for data_idx, data_dir in enumerate(self.data_dirs):
            league_name = os.path.basename(os.path.normpath(data_dir))
            if not os.path.isdir(data_dir):
                print(f"[FootballAQA] Warning: league directory not found: {data_dir}")
                continue

            meta_dir = self._get_meta_dir_for_data_dir(data_idx)
            json_index = self._get_json_index(meta_dir)

            flat_samples = self._scan_flat_league_dir(
                data_dir=data_dir,
                league_name=league_name,
                round_names=round_names,
                json_index=json_index,
            )
            if flat_samples:
                samples.extend(flat_samples)
                continue

            match_folder_samples = self._scan_match_folder_league_dir(
                data_dir=data_dir,
                league_name=league_name,
                round_names=round_names,
                json_index=json_index,
            )
            if match_folder_samples:
                samples.extend(match_folder_samples)
                continue

            for rnd in round_names:
                if rnd not in self.round_filter:
                    continue
                rnd_path = os.path.join(data_dir, rnd)
                if not os.path.isdir(rnd_path):
                    continue

                # New dataset layout:
                #   League/Round/*.npy + *_team_stats.json
                # (no per-game subfolders)
                round_file_samples = self._scan_flat_round_dir(
                    rnd_path=rnd_path,
                    league_name=league_name,
                    round_name=rnd,
                    json_index=json_index,
                )
                if round_file_samples:
                    samples.extend(round_file_samples)
                    continue

                for game_folder in sorted(os.listdir(rnd_path)):
                    game_path = os.path.join(rnd_path, game_folder)
                    if not os.path.isdir(game_path):
                        continue

                    npy_file = None
                    for fname in os.listdir(game_path):
                        if fname.endswith(".npy"):
                            npy_file = os.path.join(game_path, fname)

                    game_key = game_folder
                    json_file = json_index.get(game_key)
                    if json_file is None:
                        for fname in os.listdir(game_path):
                            if self._is_stats_json_name(fname):
                                json_file = os.path.join(game_path, fname)
                                break

                    if npy_file is None or json_file is None:
                        print(
                            f"[FootballAQA] Warning: missing .npy or .json for {game_path} "
                            f"(meta_dir={meta_dir})"
                        )
                        continue

                    sample = self._build_sample(
                        league_name=league_name,
                        round_name=rnd,
                        npy_file=npy_file,
                        json_file=json_file,
                        game_key=game_key,
                    )
                    if sample is not None:
                        samples.append(sample)
        return samples

    def _scan_match_folder_league_dir(self, data_dir, league_name, round_names, json_index):
        """Scan ``League/<match_id>/<match_id>.npy`` real-world folders."""
        samples = []
        scan_unrounded = self.random_split or set(round_names) == set(self.ROUND_NAMES)
        if not scan_unrounded:
            return samples

        for match_folder in sorted(os.listdir(data_dir)):
            match_path = os.path.join(data_dir, match_folder)
            if not os.path.isdir(match_path):
                continue
            if match_folder in self.ROUND_NAMES:
                continue

            npy_map = {}
            json_file = None
            for fname in sorted(os.listdir(match_path)):
                full_path = os.path.join(match_path, fname)
                if not os.path.isfile(full_path):
                    continue
                if fname.endswith(".npy"):
                    npy_map[os.path.splitext(fname)[0]] = full_path
                elif self._is_stats_json_name(fname):
                    json_file = full_path

            for game_key, npy_file in sorted(npy_map.items()):
                sample_json = json_index.get(game_key) or json_file
                if sample_json is None:
                    print(f"[FootballAQA] Warning: missing stats JSON for {npy_file}")
                    continue
                sample = self._build_sample(
                    league_name=league_name,
                    round_name=self._infer_round_from_name(game_key),
                    npy_file=npy_file,
                    json_file=sample_json,
                    game_key=game_key,
                )
                if sample is not None:
                    samples.append(sample)
        return samples

    def _scan_flat_round_dir(self, rnd_path, league_name, round_name, json_index):
        """Scan one round directory containing files directly (no subfolders)."""
        samples = []
        files = sorted(os.listdir(rnd_path))
        npy_map = {}

        for fname in files:
            full_path = os.path.join(rnd_path, fname)
            if not os.path.isfile(full_path):
                continue
            if fname.endswith(".npy"):
                npy_map[os.path.splitext(fname)[0]] = full_path

        for game_key, npy_file in sorted(npy_map.items()):
            json_file = json_index.get(game_key)
            if json_file is None:
                continue
            sample = self._build_sample(
                league_name=league_name,
                round_name=round_name,
                npy_file=npy_file,
                json_file=json_file,
                game_key=game_key,
            )
            if sample is not None:
                samples.append(sample)
        return samples

    def _scan_flat_league_dir(self, data_dir, league_name, round_names, json_index):
        samples = []
        allowed_rounds = set(round_names) & self.round_filter
        files = sorted(os.listdir(data_dir))
        npy_map = {}

        for fname in files:
            full_path = os.path.join(data_dir, fname)
            if not os.path.isfile(full_path):
                continue
            if fname.endswith(".npy"):
                npy_map[os.path.splitext(fname)[0]] = full_path

        common_keys = sorted(set(npy_map.keys()) & set(json_index.keys()))
        if not common_keys:
            return samples

        for game_key in common_keys:
            inferred_round = self._infer_round_from_name(game_key)
            if inferred_round is not None and inferred_round not in allowed_rounds:
                continue

            sample = self._build_sample(
                league_name=league_name,
                round_name=inferred_round,
                npy_file=npy_map[game_key],
                json_file=json_index[game_key],
                game_key=game_key,
            )
            if sample is not None:
                samples.append(sample)
        return samples

    def _infer_round_from_name(self, text):
        for rnd in self.ROUND_NAMES:
            if text.startswith(f"{rnd}_"):
                return rnd
        return None

    def _build_sample(self, league_name, round_name, npy_file, json_file, game_key):
        with open(json_file, "r", encoding="utf-8") as f:
            meta = json.load(f)

        perf = meta.get("performance_scores", {})
        match_info = meta.get("match", {})
        home_team = match_info.get("home_team")
        away_team = match_info.get("away_team")

        if home_team in perf and away_team in perf:
            team_order = [home_team, away_team]
        else:
            team_order = sorted(list(perf.keys()))
            if len(team_order) < 2:
                print(f"[FootballAQA] Warning: invalid performance_scores in {json_file}")
                return None
            team_order = team_order[:2]

        game_scores = []
        for team_name in team_order:
            team_scores = perf.get(team_name, {})
            if self.label_target == "overall":
                score = team_scores.get("score_0_100", 50.0)
            else:
                score = team_scores.get(self.label_target, 0.5)
            game_scores.append(float(score))

        round_part = round_name if round_name is not None else "unknown_round"
        game_id = f"{league_name}_{round_part}_{game_key}"
        return {
            "npy_path": npy_file,
            "scores": game_scores,
            "teams": team_order,
            "game_id": game_id,
            "json_path": json_file,
            "league": league_name,
        }

    # ------------------------------------------------------------------
    # Stratified splitting  (unchanged logic)
    # ------------------------------------------------------------------
    def _stratified_split_by_league(self, all_samples):
        if not all_samples:
            return [], []

        rng = random.Random(self.split_seed)
        by_league = {}
        for sample in all_samples:
            league = sample.get("league", "unknown")
            by_league.setdefault(league, []).append(sample)

        train_samples = []
        test_samples = []

        for league in sorted(by_league.keys()):
            league_samples = list(by_league[league])
            n = len(league_samples)
            if n <= 1:
                train_samples.extend(league_samples)
                continue

            test_count = int(round(n * self.test_ratio))
            test_count = min(max(test_count, 1), n - 1)

            strata = self._build_score_strata(league_samples)
            strata_keys = sorted(strata.keys())
            test_per_stratum = self._allocate_test_counts(strata, test_count, n)

            for key in strata_keys:
                group = list(strata[key])
                rng.shuffle(group)
                k = test_per_stratum.get(key, 0)
                test_samples.extend(group[:k])
                train_samples.extend(group[k:])

        rng.shuffle(train_samples)
        rng.shuffle(test_samples)
        return train_samples, test_samples

    def _build_score_strata(self, league_samples):
        if not league_samples:
            return {}

        num_bins = max(self.score_strat_bins, 1)
        means = []
        diffs = []
        for sample in league_samples:
            s1, s2 = sample["scores"][0], sample["scores"][1]
            means.append((s1 + s2) / 2.0)
            diffs.append(abs(s1 - s2))

        mean_edges = self._quantile_edges(means, num_bins)
        diff_edges = self._quantile_edges(diffs, num_bins)

        strata = {}
        for sample in league_samples:
            s1, s2 = sample["scores"][0], sample["scores"][1]
            mean_val = (s1 + s2) / 2.0
            diff_val = abs(s1 - s2)
            mean_bin = self._bin_value(mean_val, mean_edges)
            diff_bin = self._bin_value(diff_val, diff_edges)
            key = (diff_bin, mean_bin)
            strata.setdefault(key, []).append(sample)
        return strata

    def _quantile_edges(self, values, num_bins):
        if num_bins <= 1 or len(values) <= 1:
            return []
        edges = np.quantile(values, [i / num_bins for i in range(1, num_bins)])
        return [float(x) for x in np.asarray(edges).tolist()]

    def _bin_value(self, value, edges):
        idx = 0
        while idx < len(edges) and value > edges[idx]:
            idx += 1
        return idx

    def _allocate_test_counts(self, strata, target_total, league_total):
        keys = sorted(strata.keys())
        if target_total <= 0 or not keys:
            return {k: 0 for k in keys}

        base = {}
        fracs = []
        allocated = 0
        for key in keys:
            size = len(strata[key])
            exact = (size * target_total) / float(league_total)
            take = int(np.floor(exact))
            take = min(max(take, 0), size)
            base[key] = take
            allocated += take
            fracs.append((exact - take, key))

        remaining = target_total - allocated
        if remaining > 0:
            for _, key in sorted(fracs, key=lambda x: x[0], reverse=True):
                if remaining <= 0:
                    break
                if base[key] < len(strata[key]):
                    base[key] += 1
                    remaining -= 1

        if remaining > 0:
            for key in keys:
                if remaining <= 0:
                    break
                room = len(strata[key]) - base[key]
                if room > 0:
                    add = min(room, remaining)
                    base[key] += add
                    remaining -= add

        return base

    # ------------------------------------------------------------------
    # Printing helpers
    # ------------------------------------------------------------------
    def print_selected_games(self):
        by_league = {}
        for sample in self.samples:
            league = sample.get("league", "unknown")
            by_league.setdefault(league, []).append(sample)

        print(f"[FootballAQA] Selected {self.subset} samples: {len(self.samples)}"
              f"{' (team-level)' if self.use_graph else ''}")
        for league in sorted(by_league.keys()):
            league_samples = sorted(by_league[league],
                                    key=lambda s: s.get("game_id", ""))
            print(f"[FootballAQA]   {league}: {len(league_samples)}")
            for sample in league_samples:
                if self.use_graph:
                    print(
                        f"[FootballAQA]     - {sample.get('game_id', '?')} | "
                        f"{sample['team_side']}: {sample['team_name']} "
                        f"score={sample['score_value']:.2f} "
                        f"bbox={'Y' if sample.get('has_bbox') else 'N'}"
                    )
                else:
                    teams = sample.get("teams", ["team_a", "team_b"])
                    scores = sample.get("scores", [0.0, 0.0])
                    avg_score = (float(scores[0]) + float(scores[1])) / 2.0
                    print(
                        f"[FootballAQA]     - {sample.get('game_id', 'unknown_game')} | "
                        f"{teams[0]}={scores[0]:.2f}, {teams[1]}={scores[1]:.2f}, avg={avg_score:.2f}"
                    )

    def print_score_distribution(self):
        by_league = {}
        for sample in self.samples:
            league = sample.get("league", "unknown")
            by_league.setdefault(league, []).append(sample)

        def _calc_stats(samples):
            if self.use_graph:
                team_scores = np.array(
                    [float(s["score_value"]) for s in samples], dtype=np.float32
                )
                if team_scores.size == 0:
                    return None
                return {
                    "team_mean": float(team_scores.mean()),
                    "team_std": float(team_scores.std()),
                }
            team_scores = []
            avg_scores = []
            score_gaps = []
            for s in samples:
                a, b = float(s["scores"][0]), float(s["scores"][1])
                team_scores.extend([a, b])
                avg_scores.append((a + b) / 2.0)
                score_gaps.append(abs(a - b))
            if not team_scores:
                return None
            team_scores = np.array(team_scores, dtype=np.float32)
            avg_scores = np.array(avg_scores, dtype=np.float32)
            score_gaps = np.array(score_gaps, dtype=np.float32)
            return {
                "team_mean": float(team_scores.mean()),
                "team_std": float(team_scores.std()),
                "avg_mean": float(avg_scores.mean()),
                "avg_std": float(avg_scores.std()),
                "gap_mean": float(score_gaps.mean()),
                "gap_std": float(score_gaps.std()),
            }

        global_stats = _calc_stats(self.samples)
        if global_stats is None:
            return

        if self.use_graph:
            print(
                f"[FootballAQA] {self.subset} score stats (team-level) | "
                f"mean={global_stats['team_mean']:.2f}, std={global_stats['team_std']:.2f}"
            )
        else:
            print(
                f"[FootballAQA] {self.subset} score stats (global) | "
                f"team_mean={global_stats['team_mean']:.2f}, team_std={global_stats['team_std']:.2f}, "
                f"avg_match_mean={global_stats['avg_mean']:.2f}, avg_match_std={global_stats['avg_std']:.2f}, "
                f"gap_mean={global_stats['gap_mean']:.2f}, gap_std={global_stats['gap_std']:.2f}"
            )
        for league in sorted(by_league.keys()):
            stats = _calc_stats(by_league[league])
            if stats is None:
                continue
            if self.use_graph:
                print(
                    f"[FootballAQA]   {self.subset}:{league} | "
                    f"mean={stats['team_mean']:.2f}, std={stats['team_std']:.2f}"
                )
            else:
                print(
                    f"[FootballAQA]   {self.subset}:{league} | "
                    f"team_mean={stats['team_mean']:.2f}, team_std={stats['team_std']:.2f}, "
                    f"avg_match_mean={stats['avg_mean']:.2f}, avg_match_std={stats['avg_std']:.2f}, "
                    f"gap_mean={stats['gap_mean']:.2f}, gap_std={stats['gap_std']:.2f}"
                )

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------
    def __getitem__(self, index):
        sample = self.samples[index]

        # Load pre-extracted video features -> (num_clips, feat_dim)
        feat_path = sample["npy_path"]
        feats = self._load_feature_array(feat_path)

        if self.use_graph:
            return self._getitem_graph(sample, feats)
        return self._getitem_video_only(sample, feats)

    # -- video-only path (original behaviour) ----------------------------
    def _getitem_video_only(self, sample, feats):
        if len(feats) > self.max_clips:
            if self.subset == "train":
                st = np.random.randint(0, len(feats) - self.max_clips)
            else:
                st = (len(feats) - self.max_clips) // 2
            feats = feats[st: st + self.max_clips]
        elif len(feats) < self.max_clips:
            pad = np.zeros((self.max_clips - len(feats), feats.shape[1]), dtype=np.float32)
            feats = np.concatenate([feats, pad], axis=0)

        feats = np.ascontiguousarray(feats, dtype=np.float32)
        data = {}
        data["feats"] = torch.from_numpy(feats)
        data["video"] = torch.tensor(0)

        if self.label_target == "overall":
            norm_scores = np.array(sample["scores"], dtype=np.float32) / 100.0
        else:
            norm_scores = np.array(sample["scores"], dtype=np.float32)

        if self.target_mode == "avg":
            data["score"] = torch.tensor(float(norm_scores.mean()), dtype=torch.float32)
        elif self.target_mode == "home":
            data["score"] = torch.tensor(float(norm_scores[0]), dtype=torch.float32)
        elif self.target_mode == "away":
            data["score"] = torch.tensor(float(norm_scores[1]), dtype=torch.float32)
        elif self.target_mode == "diff":
            data["score"] = torch.tensor(float(norm_scores[0] - norm_scores[1]), dtype=torch.float32)
        else:
            data["score"] = torch.from_numpy(norm_scores.astype(np.float32, copy=False))

        clip_info = f"{sample['game_id']}_{sample['teams'][0]}_vs_{sample['teams'][1]}"
        return data, clip_info

    # -- multimodal path  --------------------------------------------------
    def _getitem_graph(self, sample, feats):
        num_clips_npy = len(feats)
        mask_side, mask_ratio = self._sample_camera_mask()

        graph_nodes, graph_mask, graph_fids, num_clips_bbox, graph_img_feats = \
            self._compute_graph_data(
                sample,
                num_clips_npy,
                mask_side=mask_side,
                mask_ratio=mask_ratio,
            )

        # Align video and graph to the same temporal length
        num_clips = min(num_clips_npy, num_clips_bbox)
        feats = feats[:num_clips]
        graph_nodes = graph_nodes[:num_clips]
        graph_mask = graph_mask[:num_clips]
        graph_fids = graph_fids[:num_clips]
        if graph_img_feats is not None:
            graph_img_feats = graph_img_feats[:num_clips]

        # Temporal subsampling: if sequence is much longer than max_clips,
        # uniformly subsample instead of cropping a contiguous window.
        # This preserves coverage of the whole match.
        if num_clips > self.max_clips:
            if self.subset == "train" and num_clips > self.max_clips * 2:
                offset = np.random.randint(0, max(1, num_clips - self.max_clips))
                indices = np.linspace(offset, num_clips - 1, self.max_clips, dtype=int)
            elif num_clips > self.max_clips * 2:
                indices = np.linspace(0, num_clips - 1, self.max_clips, dtype=int)
            else:
                if self.subset == "train":
                    st = np.random.randint(0, num_clips - self.max_clips)
                else:
                    st = (num_clips - self.max_clips) // 2
                indices = np.arange(st, st + self.max_clips)

            feats = feats[indices]
            graph_nodes = graph_nodes[indices]
            graph_mask = graph_mask[indices]
            graph_fids = graph_fids[indices]
            if graph_img_feats is not None:
                graph_img_feats = graph_img_feats[indices]
        elif num_clips < self.max_clips:
            pad_t = self.max_clips - num_clips
            feats = np.pad(feats, ((0, pad_t), (0, 0)))
            graph_nodes = np.pad(graph_nodes, ((0, pad_t), (0, 0), (0, 0)))
            graph_mask = np.pad(graph_mask, ((0, pad_t), (0, 0)))
            graph_fids = np.pad(graph_fids, ((0, pad_t), (0, 0)))
            if graph_img_feats is not None:
                graph_img_feats = np.pad(graph_img_feats, ((0, pad_t), (0, 0), (0, 0)))

        score_raw = float(sample["score_value"])
        if self.label_target == "overall":
            score = np.float32(score_raw / 100.0)
        else:
            score = np.float32(score_raw)

        team_side_id = np.int64(0 if sample["team_side"] == "home" else 1)

        data = {
            "feats": torch.from_numpy(np.ascontiguousarray(feats, dtype=np.float32)),
            "video": torch.tensor(0),
            "score": torch.tensor(score, dtype=torch.float32),
            "graph_nodes": torch.from_numpy(np.ascontiguousarray(graph_nodes, dtype=np.float32)),
            "graph_mask": torch.from_numpy(np.ascontiguousarray(graph_mask, dtype=np.bool_)),
            "graph_frame_ids": torch.from_numpy(np.ascontiguousarray(graph_fids, dtype=np.int64)),
            "team_side_id": torch.tensor(team_side_id, dtype=torch.long),
        }
        if graph_img_feats is not None:
            data["graph_image_feats"] = torch.from_numpy(
                np.ascontiguousarray(graph_img_feats, dtype=np.float32)
            )

        clip_info = f"{sample['game_id']}_{sample['team_side']}_{sample['team_name']}"
        return data, clip_info

    def __len__(self):
        return len(self.samples)


def get_dataloader(args):
    data_loaders = {}
    train_trans, test_trans = get_video_trans()

    data_loader_train = Football_Dataset(args, "train", train_trans)
    data_loader_test = Football_Dataset(args, "test", test_trans)

    print(f"[FootballAQA] Train samples: {len(data_loader_train)}, "
          f"Test samples: {len(data_loader_test)}")
    data_loader_train.print_score_distribution()
    data_loader_test.print_score_distribution()
    data_loader_test.print_selected_games()

    if args.subset > 1:
        subset = list(range(0, len(data_loader_train), args.subset))
        data_loader_train = torch.utils.data.Subset(data_loader_train, subset)

    common_loader_kwargs = {
        "num_workers": int(args.num_workers),
        "pin_memory": True,
        "persistent_workers": bool(args.get("persistent_workers", int(args.num_workers) > 0)),
    }
    if int(args.num_workers) > 0:
        common_loader_kwargs["prefetch_factor"] = int(args.get("prefetch_factor", 4))

    train_dataloader = torch.utils.data.DataLoader(
        data_loader_train,
        batch_size=args.bs_train,
        shuffle=True,
        worker_init_fn=worker_init_fn,
        **common_loader_kwargs,
    )
    test_dataloader = torch.utils.data.DataLoader(
        data_loader_test,
        batch_size=args.bs_test,
        shuffle=False,
        **common_loader_kwargs,
    )
    data_loaders["train"] = train_dataloader
    data_loaders["test"] = test_dataloader

    return data_loaders
