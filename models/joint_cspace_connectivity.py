"""
Joint-time C-space connectivity on [T×W] grids (Scheme 2).

Monotone forward/backward propagation respects time index: lateral moves only
when advancing t (matches dist-field planner in joint_vit_field_planner).
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch


def passable_from_joint_grid(
    grid: Dict,
    *,
    coll_threshold: float = 0.85,
    sing_threshold: float = 0.95,
) -> np.ndarray:
    """Binary [T, W] passable from rasterized joint-time grid."""
    coll = np.asarray(grid.get("soft_cost"), dtype=np.float32)
    if coll.ndim == 4:
        coll = coll[0, 0]
    elif coll.ndim == 3:
        coll = coll[0]
    hard = np.asarray(grid["hard_obstacle"], dtype=np.float32)
    if hard.ndim == 4:
        hard = hard[0, 0]
    elif hard.ndim == 3:
        hard = hard[0]
    H, W = hard.shape
    passable = (hard < coll_threshold) & (coll < coll_threshold)
    if "collision_row" in grid:
        # Prefer per-cell collision if soft_cost was aggregated oddly.
        passable = passable.copy()
    return passable.astype(bool)


def grow_joint_time_monotone_forward(
    passable: np.ndarray,
    start: Tuple[int, int],
    *,
    lateral: int = 1,
) -> np.ndarray:
    """Forward reachable set from start with moves (t,w) -> (t+1, w±lateral)."""
    H, W = passable.shape
    conn = np.zeros((H, W), dtype=bool)
    sr, sc = int(start[0]), int(start[1])
    sr = max(0, min(H - 1, sr))
    sc = max(0, min(W - 1, sc))
    conn[sr, sc] = True
    for t in range(H - 1):
        for w in range(W):
            if not conn[t, w]:
                continue
            for dw in range(-lateral, lateral + 1):
                nw = w + dw
                if 0 <= nw < W and passable[t + 1, nw]:
                    conn[t + 1, nw] = True
    return conn


def grow_joint_time_monotone_backward(
    passable: np.ndarray,
    end: Tuple[int, int],
    *,
    lateral: int = 1,
) -> np.ndarray:
    """Backward reachable-to-end set with moves (t+1,w) -> (t, w±lateral)."""
    H, W = passable.shape
    conn = np.zeros((H, W), dtype=bool)
    er, ec = int(end[0]), int(end[1])
    er = max(0, min(H - 1, er))
    ec = max(0, min(W - 1, ec))
    conn[er, ec] = True
    for t in range(H - 2, -1, -1):
        for w in range(W):
            for dw in range(-lateral, lateral + 1):
                nw = w + dw
                if 0 <= nw < W and conn[t + 1, nw] and passable[t + 1, nw]:
                    conn[t, w] = True
                    break
    return conn


def build_joint_monotone_connectivity(
    passable: np.ndarray,
    start: Tuple[int, int],
    end: Tuple[int, int],
    *,
    lateral: int = 1,
) -> Dict[str, np.ndarray]:
    """GT monotone start/end intersection on joint-time grid."""
    passable = np.asarray(passable, dtype=bool).copy()
    sr, sc = start
    er, ec = end
    passable[sr, sc] = True
    passable[er, ec] = True
    start_conn = grow_joint_time_monotone_forward(passable, start, lateral=lateral)
    end_conn = grow_joint_time_monotone_backward(passable, end, lateral=lateral)
    intersection = start_conn & end_conn
    return {
        "passable": passable.astype(np.float32),
        "start_connected": start_conn.astype(np.float32),
        "end_connected": end_conn.astype(np.float32),
        "connected_intersection": intersection.astype(np.float32),
        "goal_reachable": float(intersection[er, ec] > 0.5),
    }


def connectivity_labels_from_aligner(
    fields: Dict,
    grid: Dict,
    *,
    lateral: int = 1,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Build GT [T, W] connectivity intersection from tier-5 aligner output.

    Uses geometric passable from grid hard/soft costs (oracle), not ViT prediction.
    """
    traversable = fields["traversable"][0, 0].detach().cpu().numpy()
    hard = fields["hard_obstacle"][0, 0].detach().cpu().numpy()
    nav = fields["navigation_cost"][0, 0].detach().cpu().numpy()
    passable = (traversable > 0.45) & (hard < 0.55) & (nav < 1.0e5)
    if "soft_cost" in grid:
        sc = grid["soft_cost"]
        if hasattr(sc, "detach"):
            sc = sc[0, 0].detach().cpu().numpy()
        elif np.asarray(sc).ndim >= 2:
            sc = np.asarray(sc)[0, 0] if np.asarray(sc).ndim == 4 else np.asarray(sc)[0]
        passable = passable & (sc < 0.85)

    start = tuple(grid["start"])
    end = tuple(grid["end"])
    conn = build_joint_monotone_connectivity(passable, start, end, lateral=lateral)
    meta = {
        "goal_reachable_monotone": conn["goal_reachable"],
        "intersection_rate": float(conn["connected_intersection"].mean()),
        "passable_rate": float(passable.mean()),
    }
    return conn["connected_intersection"].astype(np.float32), meta


def mask_passable_with_connectivity(
    passable: np.ndarray,
    connectivity: np.ndarray,
    *,
    threshold: float = 0.5,
    protect_endpoints: bool = True,
    start: Optional[Tuple[int, int]] = None,
    end: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Restrict search passable mask to predicted/GT connected region."""
    out = passable.astype(bool).copy()
    conn = np.asarray(connectivity, dtype=np.float32) >= threshold
    if conn.shape != out.shape:
        if conn.ndim == 2 and out.ndim == 3:
            conn = conn.reshape(out.shape)
        elif conn.size == out.size:
            conn = conn.reshape(out.shape)
    out &= conn
    if protect_endpoints and start is not None:
        sr, sc = start
        out[sr, sc] = True
    if protect_endpoints and end is not None:
        er, ec = end
        out[er, ec] = True
    return out


def passable_3d_from_fields(fields: Dict, grid: Dict) -> np.ndarray:
    """[T, K, K] passable from aligner + multibasis grid."""
    from models.unified_aligner import joint_search_grid_from_fields

    passable, _, _, _, _ = joint_search_grid_from_fields(fields, grid)
    k = int(grid.get("k_side", int(np.sqrt(passable.shape[1]))))
    return passable.reshape(passable.shape[0], k, k)


def grow_joint_time_monotone_forward_3d(
    passable: np.ndarray,
    start: Tuple[int, int, int],
    *,
    lateral: int = 1,
) -> np.ndarray:
    """Forward reachable on [T,K,K] with 2D lateral moves at t+1."""
    T, K, _ = passable.shape
    conn = np.zeros((T, K, K), dtype=bool)
    st, sy, sx = start
    st = max(0, min(T - 1, int(st)))
    sy = max(0, min(K - 1, int(sy)))
    sx = max(0, min(K - 1, int(sx)))
    conn[st, sy, sx] = True
    for t in range(T - 1):
        for iy in range(K):
            for ix in range(K):
                if not conn[t, iy, ix]:
                    continue
                for dy in range(-lateral, lateral + 1):
                    for dx in range(-lateral, lateral + 1):
                        ny, nx = iy + dy, ix + dx
                        if 0 <= ny < K and 0 <= nx < K and passable[t + 1, ny, nx]:
                            conn[t + 1, ny, nx] = True
    return conn


def grow_joint_time_monotone_backward_3d(
    passable: np.ndarray,
    end: Tuple[int, int, int],
    *,
    lateral: int = 1,
) -> np.ndarray:
    T, K, _ = passable.shape
    conn = np.zeros((T, K, K), dtype=bool)
    et, ey, ex = end
    et = max(0, min(T - 1, int(et)))
    ey = max(0, min(K - 1, int(ey)))
    ex = max(0, min(K - 1, int(ex)))
    conn[et, ey, ex] = True
    for t in range(T - 2, -1, -1):
        for iy in range(K):
            for ix in range(K):
                for dy in range(-lateral, lateral + 1):
                    for dx in range(-lateral, lateral + 1):
                        ny, nx = iy + dy, ix + dx
                        if 0 <= ny < K and 0 <= nx < K and conn[t + 1, ny, nx] and passable[t + 1, ny, nx]:
                            conn[t, iy, ix] = True
                            break
                    if conn[t, iy, ix]:
                        break
    return conn


def build_joint_monotone_connectivity_3d(
    passable: np.ndarray,
    start_flat: Tuple[int, int],
    end_flat: Tuple[int, int],
    k_side: int,
    *,
    lateral: int = 1,
) -> Dict[str, np.ndarray]:
    """GT monotone intersection on [T,K,K]. start/end are (t, flat)."""
    sr, sf = start_flat
    er, ef = end_flat
    sy, sx = divmod(int(sf), k_side)
    ey, ex = divmod(int(ef), k_side)
    passable = np.asarray(passable, dtype=bool).copy()
    passable[sr, sy, sx] = True
    passable[er, ey, ex] = True
    start_conn = grow_joint_time_monotone_forward_3d(passable, (sr, sy, sx), lateral=lateral)
    end_conn = grow_joint_time_monotone_backward_3d(passable, (er, ey, ex), lateral=lateral)
    intersection = start_conn & end_conn
    return {
        "start_connected": start_conn.astype(np.float32),
        "end_connected": end_conn.astype(np.float32),
        "connected_intersection": intersection.astype(np.float32),
        "connected_intersection_flat": intersection.reshape(intersection.shape[0], -1).astype(np.float32),
        "goal_reachable": float(intersection[er, ey, ex]),
    }


def connectivity_labels_from_multibasis(fields: Dict, grid: Dict) -> Tuple[np.ndarray, Dict[str, float]]:
    """GT connectivity on [T,K,K] flattened to [T,K*K]."""
    passable3 = passable_3d_from_fields(fields, grid)
    k = int(grid["k_side"])
    conn = build_joint_monotone_connectivity_3d(passable3, grid["start"], grid["end"], k)
    meta = {
        "goal_reachable_monotone": conn["goal_reachable"],
        "intersection_rate": float(conn["connected_intersection"].mean()),
        "passable_rate": float(passable3.mean()),
        "multibasis": True,
    }
    return conn["connected_intersection_flat"].astype(np.float32), meta


def connectivity_bce_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    pos_weight: float = 3.0,
) -> torch.Tensor:
    """BCE on [B,1,T,W] or [B,1,H,W] connectivity logits/probabilities."""
    if pred.shape != target.shape:
        pred = torch.nn.functional.interpolate(pred, size=target.shape[-2:], mode="bilinear", align_corners=False)
    pw = torch.tensor([pos_weight], device=pred.device, dtype=pred.dtype)
    return torch.nn.functional.binary_cross_entropy_with_logits(pred, target, pos_weight=pw)
