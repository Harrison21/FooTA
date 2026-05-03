"""Box-to-node feature encoder for player bounding boxes.

Converts raw [x1, y1, x2, y2] detections into 10-dim node vectors:
    [cx, cy, w, h, dx, dy, dw, dh, area, aspect_ratio]
and projects them through a small MLP.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional

# Opponent-context feature registry
# Each key maps to the number of scalar dimensions it contributes.
_OPP_FEAT_DIMS = {"dist": 1, "rel_pos": 2, "count": 1}
_ALL_OPP_FEATURES = ("dist", "rel_pos", "count")


def opp_feature_dim(features=None):
    """Return total scalar dimensions for the given opponent-context feature subset.

    Args:
        features: iterable of feature names (subset of ``_ALL_OPP_FEATURES``),
                  or ``None`` to use all features (4-D).
    """
    sel = _ALL_OPP_FEATURES if features is None else features
    return sum(_OPP_FEAT_DIMS[f] for f in _ALL_OPP_FEATURES if f in frozenset(sel))


class BoxToNodeEncoder(nn.Module):
    """Two-layer MLP that lifts raw 10-dim bbox features to a richer space."""

    def __init__(self, input_dim: int = 10, hidden_dim: int = 64, output_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """x: [*, input_dim] -> [*, output_dim]"""
        return self.mlp(x)


# ---------------------------------------------------------------------------
# Standalone numpy helper used by the dataset (runs on CPU, no grad)
# ---------------------------------------------------------------------------

def compute_box_features_for_window(
    team_bboxes_by_frame,  # list-of-frames: each is np.ndarray (N_f, 4) or None
    window_frame_indices,  # list[int] of length num_frames (e.g. 8)
    frame_width: int,
    frame_height: int,
    max_nodes: int = 120,
    opp_bboxes_by_frame=None,   # opponent bboxes (same format); enables extra features
    opp_context_radius: float = 0.2,  # normalised pitch radius for counting opponents
    opp_context_features=None,  # subset of ("dist","rel_pos","count"); None = all
    motion_time_delta: Optional[float] = None,  # seconds between successive sampled frames
    motion_dt_mode: Optional[str] = None,  # None/"all"/"dxdy"
):
    """Compute padded node features, mask, and frame-ids for one temporal window.

    When *opp_bboxes_by_frame* is provided each node gets 4 extra opponent-context
    features appended, making the feature vector 14-D instead of 10-D:

        Base (10-D):
            cx, cy, w, h, Δcx, Δcy, Δw, Δh, area, ar

        Opponent context (up to 4-D, controlled by opp_context_features):
            dist  (1-D): dist_to_nearest_opp — Euclidean distance in normalised (cx,cy) space
            rel_pos (2-D): rel_cx_opp, rel_cy_opp — signed relative position to nearest opp
            count (1-D): n_opps_within_r — count of opponents within opp_context_radius

    Returns:
        nodes:     np.float32  [max_nodes, 10 + opp_dim]
        mask:      np.bool_    [max_nodes]
        frame_ids: np.int64    [max_nodes]
    """
    opp_feats_set = frozenset(
        _ALL_OPP_FEATURES if opp_context_features is None else opp_context_features
    )
    use_opp = opp_bboxes_by_frame is not None and len(opp_feats_set) > 0
    feat_dim = 10 + (opp_feature_dim(opp_feats_set) if use_opp else 0)
    eps = 1e-6

    nodes = np.zeros((max_nodes, feat_dim), dtype=np.float32)
    mask = np.zeros(max_nodes, dtype=np.bool_)
    frame_ids = np.zeros(max_nodes, dtype=np.int64)

    node_idx = 0
    prev_positions = []  # [(cx, cy, w, h)] from previous frame

    for f_offset, frame_idx in enumerate(window_frame_indices):
        bboxes = team_bboxes_by_frame[frame_idx] if frame_idx < len(team_bboxes_by_frame) else None
        if bboxes is None or len(bboxes) == 0:
            prev_positions = []
            continue

        # Pre-compute normalised opponent positions for this frame
        if use_opp:
            opp_raw = (opp_bboxes_by_frame[frame_idx]
                       if frame_idx < len(opp_bboxes_by_frame) else None)
            if opp_raw is not None and len(opp_raw) > 0:
                opp_cx = ((opp_raw[:, 0] + opp_raw[:, 2]) / 2.0) / frame_width
                opp_cy = ((opp_raw[:, 1] + opp_raw[:, 3]) / 2.0) / frame_height
                opp_pos = np.stack([opp_cx, opp_cy], axis=1)  # [N_opp, 2]
            else:
                opp_pos = None
        else:
            opp_pos = None

        current_positions = []
        for bbox in bboxes:
            if node_idx >= max_nodes:
                break
            x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
            cx = (x1 + x2) / 2.0 / frame_width
            cy = (y1 + y2) / 2.0 / frame_height
            w = (x2 - x1) / frame_width
            h = (y2 - y1) / frame_height
            area = w * h
            ar = w / (h + eps)

            dx, dy, dw, dh = 0.0, 0.0, 0.0, 0.0
            if f_offset > 0 and prev_positions:
                dists = [(cx - p[0]) ** 2 + (cy - p[1]) ** 2 for p in prev_positions]
                ni = int(np.argmin(dists))
                dx = cx - prev_positions[ni][0]
                dy = cy - prev_positions[ni][1]
                dw = w - prev_positions[ni][2]
                dh = h - prev_positions[ni][3]
                if motion_time_delta is not None and motion_time_delta > 0.0:
                    inv_dt = 1.0 / motion_time_delta
                    mode = "all" if motion_dt_mode is None else str(motion_dt_mode).lower()
                    if mode in ("all", "dxdy"):
                        dx *= inv_dt
                        dy *= inv_dt
                    if mode == "all":
                        dw *= inv_dt
                        dh *= inv_dt

            base = [cx, cy, w, h, dx, dy, dw, dh, area, ar]

            if use_opp:
                if opp_pos is not None:
                    diff = opp_pos - np.array([cx, cy])          # [N_opp, 2]
                    sq_dists = (diff ** 2).sum(axis=1)           # [N_opp]
                    nn_idx = int(np.argmin(sq_dists))
                    dist_nn = float(np.sqrt(sq_dists[nn_idx]))
                    rel_cx = float(diff[nn_idx, 0])
                    rel_cy = float(diff[nn_idx, 1])
                    n_within = int((sq_dists < opp_context_radius ** 2).sum())
                else:
                    dist_nn, rel_cx, rel_cy, n_within = 1.0, 0.0, 0.0, 0
                opp_feats = []
                if "dist" in opp_feats_set:
                    opp_feats.append(dist_nn)
                if "rel_pos" in opp_feats_set:
                    opp_feats.extend([rel_cx, rel_cy])
                if "count" in opp_feats_set:
                    opp_feats.append(float(n_within))
            else:
                opp_feats = []

            nodes[node_idx] = base + opp_feats
            mask[node_idx] = True
            frame_ids[node_idx] = f_offset
            current_positions.append((cx, cy, w, h))
            node_idx += 1

        prev_positions = current_positions

    return nodes, mask, frame_ids
