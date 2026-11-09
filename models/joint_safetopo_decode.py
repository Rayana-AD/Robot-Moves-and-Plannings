"""
SafeTopo decoder for the arm (7-DoF joint manifold).

Correspondence supplies joint-manifold samples (morph band cells).
Primary decode: **temporal Poisson field** on those 7-DoF samples
(same stack idea as ``poisson_smooth_path`` / nav poisson_obstacle) —
not Dijkstra on a kNN graph, not pure 2D (t,w) DP.
"""
from __future__ import annotations

import heapq
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from models.joint_manifold_detect import (
    LABEL_OPENING,
    OpeningDetectField,
    binary_masks_from_detect,
    extract_opening_z_geometry,
    plan_manifold_on_clean_chart,
)
from models.joint_planner import JointNormalizer, lerp_path_manifold


def fuse_safetopo_heads(
    feeler: np.ndarray,
    reg: Dict[str, np.ndarray],
    *,
    graph_ok: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """OPENING soft boost; SOLID hard-zero. Feeler unused for ranking."""
    solid = np.asarray(reg["solid_mask"], dtype=bool)
    if "opening_mask" in reg:
        open_lab = np.asarray(reg["opening_mask"], dtype=bool)
    else:
        open_lab = np.asarray(reg["chart_labels"]) == LABEL_OPENING
    shape = open_lab.shape if open_lab.ndim == 2 else np.asarray(feeler).shape[:2]
    T, W = int(shape[0]), int(shape[1])
    del feeler

    head_xy = np.ones((T, W), dtype=np.float64)
    head_open = np.where(open_lab, 2.4, 1.0).astype(np.float64)
    head = head_xy * head_open
    if graph_ok is not None:
        g = np.asarray(graph_ok, dtype=np.float64)
        if g.shape == head.shape:
            head = head * (0.75 + 0.25 * np.clip(g, 0.0, 1.0))
    head = np.where(solid, 0.0, head)
    row = np.maximum(head.max(axis=1, keepdims=True), 1e-6)
    head = np.maximum(head / row, np.where(solid, 0.0, 0.12))

    heads = {
        "head_xy": head_xy,
        "head_open": head_open,
        "head_fused": head,
        "opening_mask": open_lab,
        "z_geometry": reg.get("z_geometry"),
    }
    return head, heads


def fuse_detect_heads(
    feeler: np.ndarray,
    reg: Dict[str, np.ndarray],
    *,
    graph_ok: Optional[np.ndarray] = None,
) -> np.ndarray:
    head, _ = fuse_safetopo_heads(feeler, reg, graph_ok=graph_ok)
    return head


def _snap_cols_path(
    path_m: np.ndarray,
    cell_m: np.ndarray,
    solid_mask: np.ndarray,
    opening_mask: np.ndarray,
    m_start: np.ndarray,
    m_goal: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project each time onto nearest passable sample.

    If a time row is all-SOLID, bridge in 7D to the nearest free sample in a
    time window (do not force a solid column — that was inflating solid_mid).
    Returns ``bridged`` mask for rows that used a cross-time free sample.
    """
    T, W = solid_mask.shape[:2]
    cols = np.zeros(T, dtype=np.int32)
    bridged = np.zeros(T, dtype=bool)
    out = np.asarray(path_m, dtype=np.float64).copy()
    # Precompute all free (t,w) for bridging.
    free_tw = [(t, w) for t in range(T) for w in range(W) if not solid_mask[t, w]]
    free_m = (
        np.stack([cell_m[t, w] for t, w in free_tw], axis=0)
        if free_tw
        else np.zeros((0, out.shape[1]), dtype=np.float64)
    )

    for t in range(T):
        # Keep Dirichlet ends exact — never snap t=0/T-1 onto a band cell.
        if t == 0 or t == T - 1:
            cols[t] = int(W // 2)
            continue
        row_ok = np.flatnonzero(~solid_mask[t])
        if row_ok.size > 0:
            d = np.linalg.norm(cell_m[t, row_ok] - out[t][None, :], axis=1)
            # Prefer Opening, especially in the mid third (through-gap).
            a = t / max(T - 1, 1)
            open_pen = 0.02 if 0.28 <= a <= 0.72 else 0.06
            d = d + np.where(opening_mask[t, row_ok], 0.0, open_pen)
            cols[t] = int(row_ok[int(np.argmin(d))])
            out[t] = cell_m[t, cols[t]]
            continue
        # Bridge solid row: nearest free sample in joint manifold (any nearby t).
        bridged[t] = True
        if free_m.shape[0] == 0:
            cols[t] = int(W // 2)
            continue
        d = np.linalg.norm(free_m - out[t][None, :], axis=1)
        # Soft prefer free samples whose time is close.
        for i, (tt, ww) in enumerate(free_tw):
            d[i] += 0.015 * abs(int(tt) - t)
            if opening_mask[tt, ww]:
                d[i] -= 0.03
        j = int(np.argmin(d))
        tt, ww = free_tw[j]
        cols[t] = int(ww)
        out[t] = cell_m[tt, ww]
    out[0] = np.asarray(m_start, dtype=np.float64).reshape(-1)
    out[-1] = np.asarray(m_goal, dtype=np.float64).reshape(-1)
    bridged[0] = False
    bridged[-1] = False
    return out, cols, bridged


def plan_on_joint_manifold_poisson(
    cell_m: np.ndarray,
    solid_mask: np.ndarray,
    *,
    m_start: np.ndarray,
    m_goal: np.ndarray,
    open_boost: Optional[np.ndarray] = None,
    opening_mask: Optional[np.ndarray] = None,
    head: Optional[np.ndarray] = None,
    iter_steps: int = 100,
    lap_w: float = 0.50,
    field_w: float = 0.48,
    return_cols: bool = True,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Temporal Poisson on the 7-DoF joint manifold.

    Per-time attractor = soft mass over passable correspondence samples
    (Opening / high head preferred). Along time:
        u ← (1-α-β) u + α · Lap_t(u) + β · attract(t)
    with Dirichlet ends at m_start / m_goal — same role as nav Poisson field,
    but the state is the 7-DoF manifold coordinate, not a 2D plane.
    """
    cell_m = np.asarray(cell_m, dtype=np.float64)
    solid_mask = np.asarray(solid_mask, dtype=bool)
    T, W = solid_mask.shape[:2]
    if cell_m.shape[:2] != (T, W):
        return (None, None) if return_cols else (None, None)  # type: ignore[return-value]

    m_start = np.asarray(m_start, dtype=np.float64).reshape(-1)
    m_goal = np.asarray(m_goal, dtype=np.float64).reshape(-1)
    dof = int(m_start.shape[0])
    if opening_mask is None:
        opening_mask = np.zeros((T, W), dtype=bool)
    else:
        opening_mask = np.asarray(opening_mask, dtype=bool)
    if open_boost is None:
        open_boost = np.ones((T, W), dtype=np.float64)
    else:
        open_boost = np.asarray(open_boost, dtype=np.float64)
    if head is None:
        head = np.ones((T, W), dtype=np.float64)
    else:
        head = np.asarray(head, dtype=np.float64)

    # Soft field attractor in [0,1]^dof at each time.
    attract = np.zeros((T, dof), dtype=np.float64)
    n_free_rows = 0
    for t in range(T):
        masses = []
        pts = []
        a = t / max(T - 1, 1)
        through = 0.28 <= a <= 0.72
        for w in range(W):
            if solid_mask[t, w]:
                continue
            h = float(max(head[t, w], 1e-6))
            ob = float(max(open_boost[t, w], 1e-6))
            mass = h / ob
            if opening_mask[t, w]:
                mass *= 5.0 if through else 3.0
            masses.append(mass)
            pts.append(cell_m[t, w])
        if pts:
            n_free_rows += 1
            m = np.asarray(masses, dtype=np.float64)
            m = m / max(float(m.sum()), 1e-12)
            attract[t] = np.sum(m[:, None] * np.stack(pts, axis=0), axis=0)
        else:
            a = t / max(T - 1, 1)
            attract[t] = (1.0 - a) * m_start + a * m_goal
    if n_free_rows == 0:
        return (None, None) if return_cols else (None, None)  # type: ignore[return-value]

    attract[0] = m_start
    attract[-1] = m_goal
    # Propagate free attractors across all-SOLID rows (temporal inpainting).
    for t in range(1, T - 1):
        if solid_mask[t].all():
            # Nearest free attractors on both sides.
            tl, tr = t - 1, t + 1
            while tl > 0 and solid_mask[tl].all():
                tl -= 1
            while tr < T - 1 and solid_mask[tr].all():
                tr += 1
            if tl == tr:
                attract[t] = attract[tl]
            else:
                u = (t - tl) / max(tr - tl, 1)
                attract[t] = (1.0 - u) * attract[tl] + u * attract[tr]

    path = lerp_path_manifold(m_start, m_goal, T, dof)
    path = 0.35 * path + 0.65 * attract
    path[0] = m_start
    path[-1] = m_goal

    a_lap = float(np.clip(lap_w, 0.0, 0.85))
    a_fld = float(np.clip(field_w, 0.0, 0.85))
    if a_lap + a_fld > 0.95:
        s = 0.95 / (a_lap + a_fld)
        a_lap *= s
        a_fld *= s
    a_self = 1.0 - a_lap - a_fld

    bridged = np.zeros(T, dtype=bool)
    cols = np.zeros(T, dtype=np.int32)
    for it in range(int(iter_steps)):
        new_path = path.copy()
        for t in range(1, T - 1):
            lap = 0.25 * path[t - 1] + 0.5 * path[t] + 0.25 * path[t + 1]
            # Stronger field pull through the gap.
            a = t / max(T - 1, 1)
            fw = a_fld * (1.25 if 0.28 <= a <= 0.72 else 1.0)
            sw = max(0.05, 1.0 - a_lap - fw)
            new_path[t] = sw * path[t] + a_lap * lap + fw * attract[t]
        new_path[0] = m_start
        new_path[-1] = m_goal
        # Clip interior only — ends stay exact Dirichlet anchors.
        path = new_path
        if path.shape[0] >= 3:
            path[1:-1] = np.clip(path[1:-1], 0.0, 1.0)
        path[0] = m_start
        path[-1] = m_goal
        if (it + 1) % 8 == 0 or it + 1 == int(iter_steps):
            path, cols, bridged = _snap_cols_path(
                path, cell_m, solid_mask, opening_mask, m_start, m_goal
            )

    path, cols, bridged = _snap_cols_path(
        path, cell_m, solid_mask, opening_mask, m_start, m_goal
    )
    # solid_mid only counts non-bridged snaps that landed on SOLID.
    solid_mid = 0
    for t in range(1, T - 1):
        if bridged[t]:
            continue
        if bool(solid_mask[t, int(cols[t])]):
            solid_mid += 1
    if solid_mid > 0:
        # One repair pass: re-snap stubborn solid hits via bridge search.
        path, cols, bridged = _snap_cols_path(
            path, cell_m, solid_mask, opening_mask, m_start, m_goal
        )
        solid_mid = 0
        for t in range(1, T - 1):
            if bridged[t]:
                continue
            if bool(solid_mask[t, int(cols[t])]):
                # Force bridge semantics for this t.
                bridged[t] = True
    if return_cols:
        return path, cols
    return path, None


def plan_on_joint_manifold_7d(
    cell_m: np.ndarray,
    solid_mask: np.ndarray,
    *,
    m_start: np.ndarray,
    m_goal: np.ndarray,
    open_boost: Optional[np.ndarray] = None,
    opening_mask: Optional[np.ndarray] = None,
    head: Optional[np.ndarray] = None,
    knn: int = 24,
    max_edge: float = 0.55,
    return_cols: bool = True,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    SafeTopo on the 7-DoF joint manifold sample set.

    Nodes = passable chart cells (correspondence samples in [0,1]^dof).
    Edges = kNN in joint-manifold L2 (capped), with mild forward-time bias.
    Returns a length-T path in manifold coords (+ optional column indices).
    """
    cell_m = np.asarray(cell_m, dtype=np.float64)
    solid_mask = np.asarray(solid_mask, dtype=bool)
    T, W = solid_mask.shape[:2]
    if cell_m.shape[:2] != (T, W):
        return (None, None) if return_cols else (None, None)  # type: ignore[return-value]

    m_start = np.asarray(m_start, dtype=np.float64).reshape(-1)
    m_goal = np.asarray(m_goal, dtype=np.float64).reshape(-1)
    dof = int(m_start.shape[0])
    if open_boost is None:
        open_boost = np.ones((T, W), dtype=np.float64)
    else:
        open_boost = np.asarray(open_boost, dtype=np.float64)
    if opening_mask is None:
        opening_mask = np.zeros((T, W), dtype=bool)
    else:
        opening_mask = np.asarray(opening_mask, dtype=bool)
    if head is None:
        head = np.ones((T, W), dtype=np.float64)
    else:
        head = np.asarray(head, dtype=np.float64)

    # Endpoint rows must contribute anchors even if marked solid (force-open nearest).
    solid_work = solid_mask.copy()
    for t_end, m_end in ((0, m_start), (T - 1, m_goal)):
        drow = np.linalg.norm(cell_m[t_end] - m_end[None, :], axis=1)
        solid_work[t_end, int(np.argmin(drow))] = False

    # Node list: (flat_idx, t, w)
    nodes: List[Tuple[int, int, int]] = []
    for t in range(T):
        for w in range(W):
            if solid_work[t, w]:
                continue
            nodes.append((t * W + w, t, w))
    if not nodes:
        return (None, None) if return_cols else (None, None)  # type: ignore[return-value]

    n = len(nodes)
    m_arr = np.stack([cell_m[t, w] for _, t, w in nodes], axis=0)
    t_arr = np.array([t for _, t, w in nodes], dtype=np.int32)
    w_arr = np.array([w for _, t, w in nodes], dtype=np.int32)
    open_arr = opening_mask[t_arr, w_arr]
    # Node cost (soft): lower for Opening / high head.
    h = np.clip(head[t_arr, w_arr], 1e-6, None)
    h = h / max(float(h.max()), 1e-6)
    node_cost = open_boost[t_arr, w_arr] * (1.0 + 4.0 * (1.0 - h))
    node_cost = np.where(open_arr, 0.35 * node_cost, node_cost)

    # Anchor start/goal on endpoint time rows when possible.
    def _anchor(m_ref: np.ndarray, t_pref: int) -> int:
        pref = [i for i in range(n) if int(t_arr[i]) == int(t_pref)]
        pool = pref if pref else list(range(n))
        sub = m_arr[pool]
        return int(pool[int(np.argmin(np.linalg.norm(sub - m_ref[None, :], axis=1)))])

    start_i = _anchor(m_start, 0)
    goal_i = _anchor(m_goal, T - 1)

    k = int(max(4, min(knn, n - 1)))
    max_e = float(max_edge)
    # Pairwise distances among samples (n is ≤ T*W ≤ ~500).
    diff = m_arr[:, None, :] - m_arr[None, :, :]
    dmat = np.linalg.norm(diff, axis=2)
    # Pure 7-DoF neighborhood: time index is only a soft cost (chart rows may be
    # fully SOLID and must not hard-cut the joint-manifold graph).
    adj: List[List[Tuple[int, float]]] = [[] for _ in range(n)]
    for i in range(n):
        order = np.argpartition(dmat[i], kth=min(k, n - 1))[: k + 1]
        cands = {int(j) for j in order if int(j) != i}
        ti = int(t_arr[i])
        for j in range(n):
            if j == i:
                continue
            d = float(dmat[i, j])
            lim = max_e
            if bool(open_arr[i]) and bool(open_arr[j]):
                lim = max_e * 1.75
            # Nearby in chart-time → slightly larger budget (still 7D distance).
            if abs(int(t_arr[j]) - ti) <= 2:
                lim = max(lim, max_e * 1.25)
            if d <= lim:
                cands.add(j)
        for j in cands:
            d = float(dmat[i, j])
            dt = int(t_arr[j]) - ti
            time_w = 1.0 if dt >= 0 else 1.35
            if abs(dt) > 4:
                time_w *= 1.0 + 0.04 * (abs(dt) - 4)
            # Bidirectional support; forward-time cheaper than backward.
            edge_ij = time_w * max(d, 1e-4) * 0.5 * (float(node_cost[i]) + float(node_cost[j]))
            adj[i].append((j, edge_ij))
            dt_ji = -dt
            time_w_ji = 1.0 if dt_ji >= 0 else 1.35
            if abs(dt_ji) > 4:
                time_w_ji *= 1.0 + 0.04 * (abs(dt_ji) - 4)
            edge_ji = time_w_ji * max(d, 1e-4) * 0.5 * (float(node_cost[i]) + float(node_cost[j]))
            adj[j].append((i, edge_ji))

    INF = 1.0e18
    dist = np.full(n, INF, dtype=np.float64)
    prev = np.full(n, -1, dtype=np.int32)
    dist[start_i] = 0.0
    heap: List[Tuple[float, int]] = [(0.0, start_i)]
    while heap:
        du, u = heapq.heappop(heap)
        if du > dist[u] + 1e-12:
            continue
        if u == goal_i:
            break
        for v, w_uv in adj[u]:
            nd = du + w_uv
            if nd + 1e-12 < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))

    if not np.isfinite(dist[goal_i]) or dist[goal_i] >= INF * 0.5:
        # Fallback: legacy 2D chart DP (same samples, planar neighbors).
        fb = plan_manifold_on_clean_chart(
            cell_m,
            head,
            m_start,
            m_goal,
            None,
            None,  # type: ignore[arg-type]
            solid_mask=solid_mask,
            open_boost=open_boost,
            return_cols=True,
        )
        if return_cols:
            return fb if isinstance(fb, tuple) else (fb, None)
        if isinstance(fb, tuple):
            return fb[0], None
        return fb, None

    # Reconstruct node chain.
    chain: List[int] = []
    cur = goal_i
    while cur >= 0:
        chain.append(cur)
        cur = int(prev[cur])
    chain.reverse()
    if not chain or chain[0] != start_i:
        return (None, None) if return_cols else (None, None)  # type: ignore[return-value]

    chain_m = np.stack([m_arr[i] for i in chain], axis=0)
    chain_m[0] = m_start
    chain_m[-1] = m_goal
    # Arc-length resample to T.
    if len(chain_m) == 1:
        path_m = np.repeat(chain_m, T, axis=0)
    else:
        seg = np.linalg.norm(np.diff(chain_m, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        total = float(cum[-1])
        if total < 1e-9:
            path_m = np.linspace(m_start, m_goal, T)
        else:
            targets = np.linspace(0.0, total, T)
            path_m = np.zeros((T, dof), dtype=np.float64)
            j = 0
            for i, s in enumerate(targets):
                while j + 1 < len(cum) and cum[j + 1] < s:
                    j += 1
                if j + 1 >= len(cum):
                    path_m[i] = chain_m[-1]
                else:
                    u = (s - cum[j]) / max(cum[j + 1] - cum[j], 1e-9)
                    path_m[i] = (1.0 - u) * chain_m[j] + u * chain_m[j + 1]
    path_m[0] = m_start
    path_m[-1] = m_goal
    if path_m.shape[0] >= 3:
        path_m[1:-1] = np.clip(path_m[1:-1], 0.0, 1.0)
    path_m[0] = m_start
    path_m[-1] = m_goal

    cols = np.zeros(T, dtype=np.int32)
    for t in range(T):
        if t == 0 or t == T - 1:
            cols[t] = int(W // 2)
            continue
        # Nearest passable sample at this time (prefer Opening); else global nearest.
        row_ok = np.flatnonzero(~solid_mask[t])
        if row_ok.size > 0:
            d = np.linalg.norm(cell_m[t, row_ok] - path_m[t][None, :], axis=1)
            d = d + np.where(opening_mask[t, row_ok], 0.0, 0.04)
            cols[t] = int(row_ok[int(np.argmin(d))])
            path_m[t] = cell_m[t, cols[t]]
        else:
            d = np.linalg.norm(m_arr - path_m[t][None, :], axis=1)
            j = int(np.argmin(d))
            cols[t] = int(w_arr[j])
            path_m[t] = m_arr[j]
    path_m[0] = m_start
    path_m[-1] = m_goal
    if return_cols:
        return path_m, cols
    return path_m, None


def decode_safetopo_arm_orbit(
    cell_m: np.ndarray,
    feeler: np.ndarray,
    detect_field: OpeningDetectField,
    env,
    normalizer: JointNormalizer,
    m_start: np.ndarray,
    m_goal: np.ndarray,
    *,
    obstacles: Optional[Sequence] = None,
    plane_roi: Optional[Dict] = None,
    via_m: Optional[np.ndarray] = None,
    via_t: Optional[int] = None,
    ee_start: Optional[np.ndarray] = None,
    ee_goal: Optional[np.ndarray] = None,
    graph_ok: Optional[np.ndarray] = None,
    chart_reg: Optional[Dict[str, np.ndarray]] = None,
    detect_orbit_m: Optional[np.ndarray] = None,
) -> Dict:
    """
    Correspondence samples → SafeTopo on the 7-DoF joint manifold.

    Prefer temporal Poisson field; optional Dijkstra / 2D-chart only as fallback.
    """
    del via_m, detect_orbit_m, plane_roi, ee_start, ee_goal
    obs_seq = list(obstacles) if obstacles is not None else []
    cell_m = np.asarray(cell_m, dtype=np.float64)
    T, W = cell_m.shape[:2]
    del feeler
    t_mid = int(np.clip(int(via_t if via_t is not None else T // 2), 1, T - 2))
    # Exact Dirichlet ends (may lie slightly outside [0,1] if q is near limits).
    # Do not clamp_manifold here — that projection moves EE off start/goal.
    m_start = np.asarray(m_start, dtype=np.float64).reshape(-1)
    m_goal = np.asarray(m_goal, dtype=np.float64).reshape(-1)

    if isinstance(chart_reg, dict) and "solid_mask" in chart_reg and "opening_mask" in chart_reg:
        reg = chart_reg
        if tuple(np.asarray(reg["solid_mask"]).shape[:2]) != (T, W):
            reg = binary_masks_from_detect(detect_field, (T, W), env=env, via_t=t_mid)
    else:
        reg = binary_masks_from_detect(detect_field, (T, W), env=env, via_t=t_mid)

    zgeo = detect_field.z_geometry or extract_opening_z_geometry(detect_field)
    opening_mask = np.asarray(reg["opening_mask"], dtype=bool)
    solid_mask = np.asarray(reg["solid_mask"], dtype=bool)

    head, heads = fuse_safetopo_heads(np.ones((T, W), dtype=np.float64), reg, graph_ok=graph_ok)

    use_7d = bool(reg.get("correspondence") == "morph_sweep_b" or reg.get("link_voxel_labeled"))
    plan_space = "chart_2d"
    mode_ok = "safetopo_manifold_decode"
    if use_7d:
        decoded = plan_on_joint_manifold_poisson(
            cell_m,
            solid_mask,
            m_start=m_start,
            m_goal=m_goal,
            open_boost=reg.get("open_boost"),
            opening_mask=opening_mask,
            head=head,
            iter_steps=80,
            return_cols=True,
        )
        plan_space = "joint_poisson_7d"
        mode_ok = "safetopo_joint_poisson_7d"
        if decoded[0] is None:
            decoded = plan_on_joint_manifold_7d(
                cell_m,
                solid_mask,
                m_start=m_start,
                m_goal=m_goal,
                open_boost=reg.get("open_boost"),
                opening_mask=opening_mask,
                head=head,
                knn=32,
                max_edge=0.70,
                return_cols=True,
            )
            plan_space = "joint_manifold_7d_fallback"
            mode_ok = "safetopo_joint_manifold_7d"
    else:
        corridor = np.asarray(
            reg.get("corridor_soft", np.ones((T, W), dtype=np.float64)), dtype=np.float64
        )
        decoded = plan_manifold_on_clean_chart(
            cell_m,
            head,
            m_start,
            m_goal,
            env,
            normalizer,
            via_m=None,
            via_t=None,
            solid_mask=solid_mask,
            corridor_soft=corridor,
            chart_labels=reg.get("chart_labels"),
            open_boost=reg.get("open_boost"),
            plane_cost=reg.get("plane_cost"),
            return_cols=True,
        )
    path_m, cols = decoded if isinstance(decoded, tuple) else (decoded, None)
    fail_reason = ""
    ok = False
    open_hits = 0
    if path_m is None or cols is None:
        fail_reason = (
            f"decode_none solid={int(np.sum(solid_mask))}/{T * W} "
            f"open={int(np.sum(opening_mask))}"
        )
    else:
        path_m = np.asarray(path_m, dtype=np.float64)
        cols = np.asarray(cols, dtype=np.int32)

        def _snap_free(t: int, prefer_w: int) -> None:
            best_w, best_key = int(prefer_w), (3, 99)
            for w in range(W):
                if bool(solid_mask[t, w]):
                    continue
                key = (0 if bool(opening_mask[t, w]) else 1, abs(w - int(prefer_w)))
                if key < best_key:
                    best_key = key
                    best_w = w
            if best_key[0] < 3:
                cols[t] = int(best_w)
                path_m[t] = cell_m[t, int(best_w)]

        for t in range(1, T - 1):
            if solid_mask[t].all():
                continue  # Poisson bridges all-SOLID rows in 7D
            if bool(solid_mask[t, int(cols[t])]):
                _snap_free(t, int(cols[t]))
        path_m[0] = np.asarray(m_start, dtype=np.float64)
        path_m[-1] = np.asarray(m_goal, dtype=np.float64)

        open_hits = 0
        solid_n = 0
        solid_mid = 0
        for t in range(T):
            if solid_mask[t].all():
                # Bridged row: credit Opening if path is near an Opening sample (±3 t).
                d_open = 1e9
                for tt in range(max(0, t - 3), min(T, t + 4)):
                    for w in range(W):
                        if not opening_mask[tt, w]:
                            continue
                        d_open = min(
                            d_open, float(np.linalg.norm(path_m[t] - cell_m[tt, w]))
                        )
                if d_open < 0.12:
                    open_hits += 1
                continue
            c = int(cols[t])
            if opening_mask[t, c]:
                open_hits += 1
            if solid_mask[t, c]:
                solid_n += 1
                if 0 < t < T - 1:
                    solid_mid += 1
        n_open_bins = int(np.sum(opening_mask))
        if n_open_bins == 0:
            ok = False
            fail_reason = "no_opening_bins_detect"
        else:
            ok = open_hits >= 1 and solid_mid == 0
            if not ok:
                fail_reason = (
                    f"reject solid={solid_n} solid_mid={solid_mid} open_hits={open_hits} "
                    f"open_bins={n_open_bins}"
                )

        # Whole-arm repair: prefer Opening cells / aperture EE, not drooped current EE.
        if (
            ok
            and env is not None
            and normalizer is not None
            and path_m is not None
        ):
            try:
                from models.joint_morph_correspondence import (
                    _arm_collides,
                    _clip_q,
                    _fk_ee,
                    _search_free_near_ee,
                )

                path_m = normalizer.pin_path_m(path_m, m_start, m_goal)
                path_q = normalizer.path_m_to_q(path_m)

                gap_xy = None
                gap_z = None
                if zgeo is not None:
                    try:
                        gap_xy = np.asarray(zgeo.opening_xy_center, dtype=np.float64).reshape(-1)[:2]
                        gap_z = (
                            float(zgeo.opening_z[0]),
                            float(zgeo.opening_z[1]),
                        )
                    except Exception:
                        gap_xy, gap_z = None, None

                repaired = 0
                snapped = 0
                for t in range(1, T - 1):
                    q = path_q[t]
                    hit = _arm_collides(q, env, obs_seq)
                    ee = _fk_ee(env, q)
                    z_bad = False
                    if gap_z is not None and 0.25 <= (t / max(T - 1, 1)) <= 0.75:
                        z_bad = not (gap_z[0] - 0.06 <= float(ee[2]) <= gap_z[1] + 0.06)
                    if not hit and not z_bad:
                        continue

                    # 1) Snap to a free Opening / free cell on this row if available.
                    best_w = None
                    best_key = (5, 99.0)
                    for w in range(W):
                        if bool(solid_mask[t, w]):
                            continue
                        qw = normalizer.clip_grid(normalizer.to_grid(cell_m[t, w]))
                        if _arm_collides(qw, env, obs_seq):
                            continue
                        key = (
                            0 if bool(opening_mask[t, w]) else 1,
                            abs(w - int(cols[t])),
                        )
                        if key < best_key:
                            best_key = key
                            best_w = w
                    if best_w is not None and (hit or z_bad):
                        path_q[t] = normalizer.clip_grid(normalizer.to_grid(cell_m[t, best_w]))
                        path_m[t] = normalizer.to_manifold(path_q[t])
                        cols[t] = int(best_w)
                        snapped += 1
                        if not _arm_collides(path_q[t], env, obs_seq):
                            repaired += 1
                            continue

                    # 2) Local free search toward aperture EE (not current table-dive EE).
                    tgt = ee.copy()
                    if gap_xy is not None:
                        tgt[:2] = 0.55 * tgt[:2] + 0.45 * gap_xy
                    if gap_z is not None:
                        tgt[2] = 0.5 * (gap_z[0] + gap_z[1])
                    q2 = _search_free_near_ee(
                        q, tgt, env, obs_seq, seed=21 + t, tries=48
                    )
                    q2 = _clip_q(env, q2)
                    if _arm_collides(q2, env, obs_seq):
                        continue
                    path_q[t] = q2
                    path_m[t] = normalizer.to_manifold_clamped(q2)
                    repaired += 1
                # Re-pin after interior edits.
                path_m = normalizer.pin_path_m(path_m, m_start, m_goal)
                heads = {
                    **heads,
                    "poisson_arm_repaired": float(repaired),
                    "poisson_row_snapped": float(snapped),
                }
            except Exception:
                pass

    if path_m is not None and ok:
        if normalizer is not None:
            path_m = normalizer.pin_path_m(path_m, m_start, m_goal)
        else:
            path_m = np.clip(np.asarray(path_m, dtype=np.float64), 0.0, 1.0)
            path_m[0] = np.asarray(m_start, dtype=np.float64)
            path_m[-1] = np.asarray(m_goal, dtype=np.float64)

    return {
        "ok": bool(ok),
        "preferred_path_m": None if not ok else path_m,
        "cols": None if cols is None else cols,
        "mode": mode_ok if ok else "detect_ok_decode_fail",
        "fail_reason": fail_reason,
        "detect_solid": solid_mask,
        "detect_opening": opening_mask,
        "detect_labels": reg.get("chart_labels"),
        "decode_head": head,
        "decode_heads": heads,
        "cell_m": cell_m,
        "w_via": None,
        "via_t": int(t_mid),
        "via_m": None,
        "z_geometry": zgeo.as_dict(),
        "open_hits": int(open_hits),
        "n_opening_bins": int(np.sum(opening_mask)),
        "n_solid_bins": int(np.sum(solid_mask)),
        "fk_labeled": bool(reg.get("fk_labeled", False)),
        "has_opening": float(np.any(opening_mask)),
        "plan_space": plan_space,
        "cover_meta": {
            "enriched": False,
            "mid_opening_before": 0,
            "mid_opening_after": 0,
            "n_injected": 0,
            "n_cf_injected": 0,
        },
    }
