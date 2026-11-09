"""
Nav depth floor oracle — train/eval GT and collision judge only.

NOT used by SafeTopo Q/K inject (`build_patch_topology_tokens` in safe_topo_qkv.py).
Configure via env NAV_FLOOR_ORACLE (median | plane | plane_v2 | plane_v3 | rgbd_pixel | rgbd_sheet).

Visual audit: scripts/audit_nav_oracle_labels.py
Eval S/G review: scripts/review_nav_eval_manifest.py
"""
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# NYU Depth v2 Kinect intrinsics at native 640x480 (Silberman et al.).
NYU_INTRINSICS_640x480 = {"fx": 518.857901, "fy": 519.469611, "cx": 325.582449, "cy": 253.736166}

# Kinect no-return. 2-D depth renders often paint this (and far Z) black.
# Black is missing / into-the-page range — never an obstacle by itself.
DEPTH_VALID_LO = 0.06


def render_depth_surface_rgb(
    z: np.ndarray,
    *,
    lo: float = 0.4,
    hi: float = 8.0,
) -> np.ndarray:
    """
    Unified 2-D coloring of the Kinect range image (the depth *surface*).

    Measured: turbo-like near=warm, far=cool. Far stays blue — never crushed to black.
    Unmeasured (z <= DEPTH_VALID_LO): mid-gray. That hole is not an obstacle cue.
    """
    z = np.asarray(z, dtype=np.float32)
    measured = z > DEPTH_VALID_LO
    t = np.zeros_like(z)
    t[measured] = np.clip((z[measured] - lo) / max(1e-3, hi - lo), 0.0, 1.0)
    try:
        from matplotlib import colormaps

        out = colormaps["turbo_r"](t)[..., :3].astype(np.float32)
    except Exception:
        r = np.clip(1.0 - 2.5 * t, 0, 1)
        g = np.clip(1.0 - 2.5 * np.abs(t - 0.4), 0, 1)
        b = np.clip(2.5 * (t - 0.6), 0, 1)
        out = np.stack([r, g, b], axis=-1).astype(np.float32)
    out[~measured] = np.array([0.45, 0.45, 0.45], dtype=np.float32)
    return out


def _protrude_from_floor_np(z: np.ndarray, valid: np.ndarray, sheet: np.ndarray) -> np.ndarray:
    """True where a measured point sits closer than the column's floor sheet (sticks up)."""
    H, W = z.shape
    col_z = np.full(W, np.nan, dtype=np.float32)
    for c in range(W):
        m = sheet[:, c]
        if m.any():
            col_z[c] = float(np.median(z[m, c]))
    good = np.isfinite(col_z)
    if good.any():
        fill = float(np.median(col_z[good]))
        col_z[~good] = fill
    else:
        col_z[:] = float(np.median(z[valid])) if valid.any() else 2.0
    floor_z = col_z[np.newaxis, :]
    margin = np.maximum(0.08, 0.028 * np.maximum(z, floor_z))
    return valid & (~sheet) & ((floor_z - z) > margin)


def floor_oracle_mode() -> str:
    """'median' (legacy range-band heuristic) or 'plane' (geometric plane fit). Env NAV_FLOOR_ORACLE."""
    import os

    mode = os.environ.get("NAV_FLOOR_ORACLE", "median").strip().lower()
    return mode if mode in ("median", "plane", "plane_v2", "plane_v3", "rgbd_pixel", "rgbd_sheet") else "median"


def _backproject(depth: torch.Tensor, intrinsics: Optional[Dict[str, float]] = None) -> torch.Tensor:
    """depth [B,1,H,W] (m) → points [B,H,W,3] camera frame (x right, y down, z forward)."""
    B, _, H, W = depth.shape
    k = intrinsics or NYU_INTRINSICS_640x480
    sx, sy = W / 640.0, H / 480.0
    fx, fy, cx, cy = k["fx"] * sx, k["fy"] * sy, k["cx"] * sx, k["cy"] * sy
    v = torch.arange(H, device=depth.device, dtype=depth.dtype).view(1, H, 1).expand(B, H, W)
    u = torch.arange(W, device=depth.device, dtype=depth.dtype).view(1, 1, W).expand(B, H, W)
    z = depth[:, 0]
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return torch.stack([x, y, z], dim=-1)


def fit_floor_plane(
    depth: torch.Tensor,
    *,
    floor_band_start: float = 0.5,
    n_hyp: int = 256,
    max_points: int = 4000,
    inlier_thresh: float = 0.04,
    max_tilt_deg: float = 35.0,
    min_depth: float = 0.3,
    max_depth: float = 8.0,
    support_ratio: float = 0.25,
    max_below_frac: float = 0.03,
    seed: int = 0,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    RANSAC plane fit on lower-band depth points. Camera "up" is -y.

    Returns (normal [B,3] pointing up, offset [B]) so that height = n·P + d, or (None, None)
    when no valid plane is found for the batch (caller falls back to median heuristic).
    """
    B, _, H, W = depth.shape
    pts = _backproject(depth)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    up = torch.tensor([0.0, -1.0, 0.0], device=depth.device, dtype=depth.dtype)
    cos_max = math.cos(math.radians(max_tilt_deg))
    normals, offsets = [], []
    for b in range(B):
        z = depth[b, 0]
        band = torch.zeros_like(z, dtype=torch.bool)
        band[int(H * floor_band_start) :, :] = True
        valid = band & (z > min_depth) & (z < max_depth)
        P = pts[b][valid]
        if P.shape[0] < 50:
            return None, None
        if P.shape[0] > max_points:
            idx = torch.randperm(P.shape[0], generator=gen)[:max_points].to(P.device)
            P = P[idx]
        N = P.shape[0]
        tri = torch.randint(0, N, (n_hyp, 3), generator=gen).to(P.device)
        p0, p1, p2 = P[tri[:, 0]], P[tri[:, 1]], P[tri[:, 2]]
        n = torch.cross(p1 - p0, p2 - p0, dim=-1)
        n = n / (n.norm(dim=-1, keepdim=True) + 1e-9)
        # orient toward camera-up
        sign = torch.sign((n @ up).clamp(min=-1, max=1))
        sign[sign == 0] = 1.0
        n = n * sign.unsqueeze(-1)
        tilt_ok = (n @ up) >= cos_max
        if tilt_ok.sum() == 0:
            return None, None
        n = n[tilt_ok]
        p0 = p0[tilt_ok]
        d = -(n * p0).sum(-1)  # plane: n·x + d = 0
        dist = (n @ P.T) + d.unsqueeze(-1)  # [n_hyp, N]  (+ = above plane)
        inl = (dist.abs() < inlier_thresh)
        counts = inl.sum(-1)
        top = int(counts.max())
        if top < max(30, int(0.05 * N)):
            return None, None
        # Floor = lowest well-supported horizontal plane. Tables/beds/counters often have more
        # inliers than the floor, so among candidates with >= support_ratio of the best count and
        # few points clearly *below* them, choose the one farthest below the camera (largest d).
        below = (dist < -3 * inlier_thresh).float().mean(-1)
        cand = (counts >= max(30, int(support_ratio * top))) & (below < max_below_frac)
        if cand.sum() == 0:
            cand = counts >= max(30, int(support_ratio * top))
        d_masked = torch.where(cand, d, torch.full_like(d, -1e9))
        best = int(torch.argmax(d_masked))
        # least-squares refine on inliers
        Q = P[inl[best]]
        c = Q.mean(0)
        _, _, Vt = torch.linalg.svd(Q - c, full_matrices=False)
        n_ref = Vt[-1]
        if (n_ref @ up) < 0:
            n_ref = -n_ref
        if (n_ref @ up) < cos_max:
            n_ref = n[best]
        d_ref = -(n_ref @ c)
        normals.append(n_ref)
        offsets.append(d_ref)
    return torch.stack(normals), torch.stack(offsets)


def floor_plane_fields(
    depth: torch.Tensor,
    *,
    floor_band_start: float = 0.5,
    floor_margin: float = 0.10,
    obstacle_height: float = 0.15,
    floor_temperature: float = 40.0,
    height_temperature: float = 30.0,
) -> Optional[Dict[str, torch.Tensor]]:
    """
    Geometric floor oracle: height above a RANSAC-fit floor plane (metres).

    floor_prob high where |height| < floor_margin; height_obstacle high where height > obstacle_height.
    Invalid depth (<= 0.06 m) is treated as obstacle. Returns None if plane fit fails.
    Thresholds (0.10 / 0.15 m) were tuned by visual audit on NYU hold-out frames: Kinect noise plus
    residual plane tilt gives ~±0.09 m on true floor at the near/far image borders.
    """
    n, d = fit_floor_plane(depth, floor_band_start=floor_band_start)
    if n is None:
        return None
    pts = _backproject(depth)  # [B,H,W,3]
    height = (pts * n.view(-1, 1, 1, 3)).sum(-1) + d.view(-1, 1, 1)  # [B,H,W], + above floor
    height = height.unsqueeze(1)
    valid = depth > 0.06
    floor_prob = torch.sigmoid((floor_margin - height.abs()) * floor_temperature) * valid
    height_obstacle = torch.sigmoid((height - obstacle_height) * height_temperature)
    height_obstacle = torch.where(valid, height_obstacle, torch.ones_like(height_obstacle))
    floor_depth = depth[:, :, int(depth.shape[2] * floor_band_start) :, :].flatten(2).median(dim=-1).values.view(-1, 1, 1, 1)
    return {
        "floor_prob": floor_prob,
        "height_obstacle": height_obstacle,
        "floor_depth": floor_depth,
        "relative_height": height,
        "plane_normal": n,
        "plane_offset": d,
    }


# ---------------------------------------------------------------------------------------------
# Plane oracle v2 (2026-09 visual audit of 200 NYU frames, docs/nav_oracle_label_audit.md)
#
# Failure modes of v1 that v2 addresses:
#   A. wrong plane (table / sofa / bed picked when floor has < 25 % of the best support)
#      -> lower support ratio + camera-height prior (Kinect held 0.7-2.2 m above floor)
#   B. roll / pitch extrapolation error (inliers only in the near band -> far/side floor reads
#      as +0.1..0.3 m and becomes a false obstacle) -> iterative re-fit on the whole image
#   C. Kinect depth noise grows ~ z^2 (Khoshelham & Elberink 2012) -> depth-adaptive tolerance
#   D. NYU border band (Eigen crop 45/41 px @640x480) has inpainted, unreliable depth
#      -> border pixels inherit the nearest interior label (no new obstacle/free is invented there)
# ---------------------------------------------------------------------------------------------

# Eigen crop [45:471, 41:601] expressed as fractions of the image (top, bottom, left, right).
NYU_BORDER_FRAC = (45 / 480.0, 1.0 - 471 / 480.0, 41 / 640.0, 1.0 - 601 / 640.0)


def kinect_depth_tolerance(z: torch.Tensor, base: float, k: float = 0.003, cap_extra: float = 0.12) -> torch.Tensor:
    """base + k*z^2, capped at base + cap_extra (metres). Random Kinect error ~ 1.4e-3 z^2 (1 sigma)."""
    return base + torch.clamp(k * z * z, max=cap_extra)


def fit_floor_plane_v2(
    depth: torch.Tensor,
    *,
    floor_band_start: float = 0.45,
    n_hyp: int = 512,
    max_points: int = 4000,
    inlier_base: float = 0.04,
    max_tilt_deg: float = 35.0,
    min_depth: float = 0.3,
    max_depth: float = 8.0,
    support_ratio: float = 0.10,
    max_below_frac: float = 0.03,
    cam_height_range: Tuple[float, float] = (0.7, 2.2),
    refine_iters: int = 3,
    seed: int = 0,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Dict[str, float]]:
    """
    RANSAC floor plane with (i) lowest-supported-plane selection under a camera-height prior and
    (ii) whole-image iterative least-squares refinement with depth-adaptive inlier tolerance.

    Returns (normal [B,3] up, offset [B], info). info["cam_height"] = offset (camera above floor),
    info["floor_conf"] in [0,1] is low when the chosen plane violates the camera-height prior.
    """
    B, _, H, W = depth.shape
    pts = _backproject(depth)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    up = torch.tensor([0.0, -1.0, 0.0], device=depth.device, dtype=depth.dtype)
    cos_max = math.cos(math.radians(max_tilt_deg))
    normals, offsets = [], []
    info: Dict[str, float] = {"cam_height": 0.0, "floor_conf": 0.0, "n_inliers": 0.0, "support_frac": 0.0}
    for b in range(B):
        z = depth[b, 0]
        valid_all = (z > min_depth) & (z < max_depth)
        band = torch.zeros_like(z, dtype=torch.bool)
        band[int(H * floor_band_start) :, :] = True
        P_all = pts[b][valid_all]
        Z_all = z[valid_all]
        P = pts[b][band & valid_all]
        if P.shape[0] < 50:
            return None, None, info
        if P.shape[0] > max_points:
            idx = torch.randperm(P.shape[0], generator=gen)[:max_points].to(P.device)
            P = P[idx]
        N = P.shape[0]
        thr_h = kinect_depth_tolerance(P[:, 2], inlier_base, k=0.002, cap_extra=0.04)  # [N]
        tri = torch.randint(0, N, (n_hyp, 3), generator=gen).to(P.device)
        p0, p1, p2 = P[tri[:, 0]], P[tri[:, 1]], P[tri[:, 2]]
        n = torch.cross(p1 - p0, p2 - p0, dim=-1)
        n = n / (n.norm(dim=-1, keepdim=True) + 1e-9)
        sign = torch.sign((n @ up).clamp(min=-1, max=1))
        sign[sign == 0] = 1.0
        n = n * sign.unsqueeze(-1)
        tilt_ok = (n @ up) >= cos_max
        if tilt_ok.sum() == 0:
            return None, None, info
        n, p0 = n[tilt_ok], p0[tilt_ok]
        d = -(n * p0).sum(-1)
        dist = (n @ P.T) + d.unsqueeze(-1)  # [n_hyp, N], + above plane
        inl = dist.abs() < thr_h.unsqueeze(0)
        counts = inl.sum(-1)
        top = int(counts.max())
        if top < max(30, int(0.03 * N)):
            return None, None, info
        below = (dist < -0.12).float().mean(-1)
        supported = counts >= max(60, int(support_ratio * top))
        in_prior = (d >= cam_height_range[0]) & (d <= cam_height_range[1])
        cand = supported & (below < max_below_frac) & in_prior
        conf = 1.0
        if cand.sum() == 0:
            cand = supported & (below < max_below_frac)
            conf = 0.5
        if cand.sum() == 0:
            cand = supported
            conf = 0.25
        d_masked = torch.where(cand, d, torch.full_like(d, -1e9))
        best = int(torch.argmax(d_masked))
        n_cur, d_cur = n[best], d[best]

        # Coarse-to-fine whole-image refinement (fixes roll/pitch extrapolation without letting
        # far furniture bases pull the plane): tolerance shrinks 3x -> 1x over the iterations,
        # only rows below 0.3H are used, and any step that drifts the normal by > 6 deg from the
        # RANSAC pick or pushes the camera height outside the prior is rejected.
        n_pick, d_pick = n_cur.clone(), d_cur.clone()
        rows = torch.arange(H, device=depth.device).view(H, 1).expand(H, W)[valid_all]
        low = rows >= int(0.3 * H)
        P_ref, Z_ref = P_all[low], Z_all[low]
        thr_base = kinect_depth_tolerance(Z_ref, inlier_base, k=0.003, cap_extra=0.08)
        n_inl = 0
        scales = [3.0, 2.0, 1.0][-max(1, refine_iters) :] if refine_iters <= 3 else [3.0, 2.0] + [1.0] * (refine_iters - 2)
        cos_drift = math.cos(math.radians(6.0))
        for s in scales:
            h_all = (P_ref @ n_cur) + d_cur
            inl_all = h_all.abs() < thr_base * s
            cnt = int(inl_all.sum())
            if cnt < 50:
                break
            Q = P_ref[inl_all]
            c = Q.mean(0)
            _, _, Vt = torch.linalg.svd(Q - c, full_matrices=False)
            n_ref = Vt[-1]
            if (n_ref @ up) < 0:
                n_ref = -n_ref
            d_ref = -(n_ref @ c)
            if (n_ref @ up) < cos_max or (n_ref @ n_pick) < cos_drift:
                break
            if conf >= 1.0 and not (cam_height_range[0] <= float(d_ref) <= cam_height_range[1]):
                break
            n_cur, d_cur, n_inl = n_ref, d_ref, cnt
        cam_h = float(d_cur)
        if not (cam_height_range[0] <= cam_h <= cam_height_range[1]):
            conf = min(conf, 0.5)
        normals.append(n_cur)
        offsets.append(d_cur)
        info = {
            "cam_height": cam_h,
            "floor_conf": conf,
            "n_inliers": float(n_inl),
            "support_frac": float(n_inl / max(1, P_all.shape[0])),
        }
    return torch.stack(normals), torch.stack(offsets), info


def _inherit_border(x: torch.Tensor, frac: Tuple[float, float, float, float] = NYU_BORDER_FRAC, top: bool = False) -> torch.Tensor:
    """Replace the unreliable NYU border band of x [B,1,H,W] by the nearest interior row/column."""
    B, C, H, W = x.shape
    t = int(round(frac[0] * H)) if top else 0
    bt = max(1, int(round(frac[1] * H)))
    l = int(round(frac[2] * W))
    r = int(round(frac[3] * W))
    y = x.clone()
    if l > 0:
        y[:, :, :, :l] = y[:, :, :, l : l + 1]
    if r > 0:
        y[:, :, :, W - r :] = y[:, :, :, W - r - 1 : W - r]
    if bt > 0:
        y[:, :, H - bt :, :] = y[:, :, H - bt - 1 : H - bt, :]
    if t > 0:
        y[:, :, :t, :] = y[:, :, t : t + 1, :]
    return y


def floor_plane_fields_v2(
    depth: torch.Tensor,
    *,
    floor_margin: float = 0.10,
    obstacle_height: float = 0.15,
    noise_k: float = 0.004,
    noise_cap: float = 0.12,
    floor_temperature: float = 40.0,
    height_temperature: float = 30.0,
    inherit_border: bool = True,
) -> Optional[Dict[str, torch.Tensor]]:
    """
    Plane oracle v2: `fit_floor_plane_v2` + depth-adaptive margins + NYU border inheritance.

    floor_margin(z) = 0.10 + 0.004 z^2 (cap +0.12): floor if |h| < margin.
    obstacle_height(z) = 0.15 + 0.004 z^2 (cap +0.12): obstacle if h > that.
    Between the two (per pixel) the label is neither confidently floor nor obstacle (soft).
    """
    n, d, info = fit_floor_plane_v2(depth)
    if n is None:
        return None
    pts = _backproject(depth)
    height = ((pts * n.view(-1, 1, 1, 3)).sum(-1) + d.view(-1, 1, 1)).unsqueeze(1)
    z = depth
    valid = z > 0.06
    margin = kinect_depth_tolerance(z, floor_margin, k=noise_k, cap_extra=noise_cap)
    obst = kinect_depth_tolerance(z, obstacle_height, k=noise_k, cap_extra=noise_cap)
    floor_prob = torch.sigmoid((margin - height.abs()) * floor_temperature) * valid
    height_obstacle = torch.sigmoid((height - obst) * height_temperature)
    height_obstacle = torch.where(valid, height_obstacle, torch.ones_like(height_obstacle))
    if inherit_border:
        floor_prob = _inherit_border(floor_prob)
        height_obstacle = _inherit_border(height_obstacle)
    floor_depth = depth[:, :, int(depth.shape[2] * 0.5) :, :].flatten(2).median(dim=-1).values.view(-1, 1, 1, 1)
    return {
        "floor_prob": floor_prob,
        "height_obstacle": height_obstacle,
        "floor_depth": floor_depth,
        "relative_height": height,
        "plane_normal": n,
        "plane_offset": d,
        "plane_info": info,
    }


def _column_bottom_depth_ref(z: np.ndarray, valid: np.ndarray, *, row_frac: float = 0.82, n_rows: int = 12) -> np.ndarray:
    """Per-column median depth from the bottom strip (local floor range reference)."""
    H, W = z.shape
    r0 = max(0, int(row_frac * H) - n_rows)
    ref = np.full(W, np.nan, dtype=np.float32)
    for c in range(W):
        col = z[r0:H, c]
        colv = valid[r0:H, c]
        if colv.any():
            ref[c] = float(np.median(col[colv]))
    good = np.isfinite(ref)
    if good.any():
        fill = float(np.median(ref[good]))
        ref[~good] = fill
    else:
        ref[:] = float(np.median(z[valid])) if valid.any() else 2.0
    return ref


def _local_depth_floor_np(
    z: np.ndarray,
    valid: np.ndarray,
    *,
    row_start_frac: float = 0.48,
    rel_tol: float = 0.045,
    abs_tol: float = 0.11,
) -> np.ndarray:
    """
    Floor sheet from per-column bottom depth reference + row-wise depth band.

    Complements global plane when tilt fit drifts (00001 corridor, 00025 dining room).
    """
    H, W = z.shape
    ref = _column_bottom_depth_ref(z, valid)
    out = np.zeros((H, W), dtype=bool)
    r0 = int(row_start_frac * H)
    for r in range(r0, H):
        row_frac = (r - r0) / max(1, H - 1 - r0)
        z_ref = ref * (1.0 + 0.18 * row_frac)
        tol = np.maximum(abs_tol, rel_tol * z_ref)
        out[r] = valid[r] & (z[r] <= z_ref + tol)
    return out


def _depth_continuity_floor_np(
    z: np.ndarray,
    valid: np.ndarray,
    *,
    seed_row_frac: float = 0.86,
    rel_tol: float = 0.026,
    abs_tol: float = 0.075,
    max_depth: float = 9.0,
) -> np.ndarray:
    """
    Recover a floor sheet by BFS from the bottom image band on depth continuity alone.

    Stops at depth jumps (table tops, walls) but reconnects far floor that a global plane
    mis-labels as +height (corridor / dining-room cases like 00001, 00006).
    """
    H, W = z.shape
    out = np.zeros((H, W), dtype=bool)
    visited = np.zeros((H, W), dtype=bool)
    seed_r = int(seed_row_frac * H)
    stack: List[Tuple[int, int]] = []
    for r in range(seed_r, H):
        for c in range(W):
            if valid[r, c] and z[r, c] < max_depth:
                stack.append((r, c))
                visited[r, c] = True
    if not stack:
        return out
    while stack:
        r, c = stack.pop()
        out[r, c] = True
        z0 = z[r, c]
        tol = max(abs_tol, rel_tol * z0)
        for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= H or nc < 0 or nc >= W or visited[nr, nc]:
                continue
            if not valid[nr, nc] or z[nr, nc] >= max_depth:
                continue
            if abs(z[nr, nc] - z0) <= tol:
                visited[nr, nc] = True
                stack.append((nr, nc))
    return out


def floor_plane_fields_v3(
    depth: torch.Tensor,
    *,
    floor_margin: float = 0.10,
    obstacle_height: float = 0.15,
    noise_k: float = 0.004,
    noise_cap: float = 0.14,
    floor_temperature: float = 40.0,
    height_temperature: float = 30.0,
    inherit_border: bool = True,
    recover_hi: float = 0.38,
    recover_lo: float = 0.06,
    drift_median_h: float = 0.32,
) -> Optional[Dict[str, torch.Tensor]]:
    """
    Plane oracle v3: v2 + depth-continuity floor recovery for mis-labelled far floor.

    Pixels on the continuity sheet with plane height in (recover_lo, recover_hi) are
    re-labelled as floor (typical +0.10..+0.30 m false obstacles on distant tiles).

    When the lower-half plane height median exceeds `drift_median_h`, also trust the
    local depth-reference floor sheet (column bottom depth + perspective slack).
    """
    base = floor_plane_fields_v2(
        depth,
        floor_margin=floor_margin,
        obstacle_height=obstacle_height,
        noise_k=noise_k,
        noise_cap=noise_cap,
        floor_temperature=floor_temperature,
        height_temperature=height_temperature,
        inherit_border=False,
    )
    if base is None:
        return None
    z = depth[:, 0]
    valid = z > 0.06
    height = base["relative_height"]
    margin = kinect_depth_tolerance(z, floor_margin, k=noise_k, cap_extra=noise_cap)
    floor_prob = base["floor_prob"].clone()
    height_obstacle = base["height_obstacle"].clone()
    cont_list = []
    local_list = []
    B = depth.shape[0]
    for b in range(B):
        zb = z[b].detach().cpu().numpy()
        vb = valid[b].detach().cpu().numpy()
        cont = _depth_continuity_floor_np(zb, vb)
        local = _local_depth_floor_np(zb, vb)
        cont_t = torch.from_numpy(cont).to(depth.device, dtype=torch.bool)
        local_t = torch.from_numpy(local).to(depth.device, dtype=torch.bool)
        cont_list.append(cont_t)
        local_list.append(local_t)
        h = height[b, 0]
        lower = h[int(h.shape[0] * 0.55) :]
        drift = float(torch.median(h[valid[b]].abs()).item()) if valid[b].any() else 0.0
        plane_recover = cont_t & (h > recover_lo) & (h < recover_hi) & valid[b]
        if drift > drift_median_h:
            depth_recover = local_t & valid[b] & (h > recover_lo)
            recover = plane_recover | depth_recover
        else:
            recover = plane_recover
        floor_prob[b, 0][recover] = torch.maximum(floor_prob[b, 0][recover], torch.tensor(0.92, device=depth.device))
        height_obstacle[b, 0][recover] = torch.minimum(height_obstacle[b, 0][recover], torch.tensor(0.08, device=depth.device))
    continuity = torch.stack(cont_list).unsqueeze(1)
    local_floor = torch.stack(local_list).unsqueeze(1)
    drift_flags = []
    for b in range(B):
        hb = height[b, 0]
        drift_flags.append(
            float(torch.median(hb[valid[b]].abs()).item()) > drift_median_h if valid[b].any() else 0.0
        )
    if inherit_border:
        floor_prob = _inherit_border(floor_prob)
        height_obstacle = _inherit_border(height_obstacle)
    out = dict(base)
    out["floor_prob"] = floor_prob
    out["height_obstacle"] = height_obstacle
    out["depth_continuity_floor"] = continuity.float()
    out["local_depth_floor"] = local_floor.float()
    out["drift_floor_sheet"] = local_floor.float()
    out["plane_drift"] = torch.tensor(drift_flags, device=depth.device, dtype=depth.dtype).view(-1, 1, 1, 1)
    return out


def _depth_step_obstacle_np(
    z: np.ndarray,
    h: np.ndarray,
    valid: np.ndarray,
    *,
    h_max: float = 0.13,
    step_rel: float = 0.032,
    step_abs: float = 0.050,
) -> np.ndarray:
    """Depth discontinuity at near-floor height → thin vertical obstacles (chair legs)."""
    H, W = z.shape
    out = np.zeros((H, W), dtype=bool)
    near_floor = valid & (np.abs(h) < h_max)
    for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        z_n = np.roll(z, -dr, axis=0) if dr else np.roll(z, -dc, axis=1)
        v_n = np.roll(valid, -dr, axis=0) if dr else np.roll(valid, -dc, axis=1)
        step = np.abs(z - z_n)
        tol = np.maximum(step_abs, step_rel * np.maximum(z, z_n))
        out |= near_floor & valid & v_n & (step > tol)
    return out


def _rgb_vertical_obstacle_np(
    rgb: np.ndarray,
    near_floor: np.ndarray,
    *,
    min_grad: float = 0.065,
    dark_thresh: float = 0.58,
) -> np.ndarray:
    """Dark vertical RGB structure on the floor band (chair/table legs). rgb [H,W,3] in 0..1."""
    gray = rgb.mean(axis=-1)
    gy = np.zeros_like(gray)
    gy[1:-1, :] = np.abs(gray[2:, :] - gray[:-2, :]) * 0.5
    gx = np.zeros_like(gray)
    gx[:, 1:-1] = np.abs(gray[:, 2:] - gray[:, :-2]) * 0.5
    vertical = (gy > min_grad) & (gy > gx * 1.15)
    dark = gray < dark_thresh
    return near_floor & vertical & dark


def floor_rgbd_pixel_fields(
    depth: torch.Tensor,
    rgb: Optional[torch.Tensor] = None,
    *,
    floor_margin: float = 0.10,
    obstacle_height: float = 0.15,
    noise_k: float = 0.004,
    noise_cap: float = 0.12,
    floor_temperature: float = 40.0,
    height_temperature: float = 30.0,
    inherit_border: bool = True,
    rgb_obstacle_weight: float = 0.88,
    depth_step_weight: float = 0.92,
    near_floor_h: float = 0.12,
) -> Optional[Dict[str, torch.Tensor]]:
    """
    RGB-D pixel oracle: plane_v2 depth core + local depth steps + RGB vertical leg cues.

    Adds obstacles only (never global floor recovery like plane_v3). RGB is required for full
    mode; without rgb, depth-step cues still apply.
    """
    base = floor_plane_fields_v2(
        depth,
        floor_margin=floor_margin,
        obstacle_height=obstacle_height,
        noise_k=noise_k,
        noise_cap=noise_cap,
        floor_temperature=floor_temperature,
        height_temperature=height_temperature,
        inherit_border=False,
    )
    if base is None:
        return None
    floor_prob = base["floor_prob"].clone()
    height_obstacle = base["height_obstacle"].clone()
    z = depth[:, 0]
    valid = z > 0.06
    height = base["relative_height"]
    rgb_disc_list = []
    depth_disc_list = []
    B = depth.shape[0]
    for b in range(B):
        zb = z[b].detach().cpu().numpy()
        hb = height[b, 0].detach().cpu().numpy()
        vb = valid[b].detach().cpu().numpy()
        near = vb & (np.abs(hb) < near_floor_h)
        d_disc = _depth_step_obstacle_np(zb, hb, vb)
        if rgb is not None:
            if rgb.dim() != 4 or rgb.shape[1] != 3:
                raise ValueError("rgb must be [B, 3, H, W]")
            rgb_np = rgb[b].permute(1, 2, 0).detach().cpu().numpy()
            rgb_np = np.clip(rgb_np, 0.0, 1.0)
            r_disc = _rgb_vertical_obstacle_np(rgb_np, near)
        else:
            r_disc = np.zeros_like(d_disc, dtype=bool)
        depth_disc_list.append(d_disc)
        rgb_disc_list.append(r_disc)
        # Require both depth step AND RGB vertical cue to avoid breaking narrow floor corridors.
        extra = d_disc & r_disc
        if extra.any():
            extra_t = torch.from_numpy(extra).to(depth.device)
            height_obstacle[b, 0] = torch.where(
                extra_t,
                torch.maximum(height_obstacle[b, 0], torch.tensor(0.90, device=depth.device)),
                height_obstacle[b, 0],
            )
            floor_prob[b, 0] = torch.where(
                extra_t,
                torch.minimum(floor_prob[b, 0], torch.tensor(0.10, device=depth.device)),
                floor_prob[b, 0],
            )
    if inherit_border:
        floor_prob = _inherit_border(floor_prob)
        height_obstacle = _inherit_border(height_obstacle)
    out = dict(base)
    out["floor_prob"] = floor_prob
    out["height_obstacle"] = height_obstacle
    out["rgb_obstacle"] = torch.stack([torch.from_numpy(x.astype(np.float32)) for x in rgb_disc_list]).unsqueeze(1).to(depth.device)
    out["depth_step_obstacle"] = torch.stack([torch.from_numpy(x.astype(np.float32)) for x in depth_disc_list]).unsqueeze(1).to(depth.device)
    return out


def _rgb_luma_ab(rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cheap opponent colour (luma + a/b) for floor matching. rgb [H,W,3] in 0..1."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    a = r - g
    bb = 0.5 * (r + g) - b
    return luma, a, bb


def _color_stats(rgb: np.ndarray, mask: np.ndarray) -> Optional[Dict[str, float]]:
    if mask.sum() < 20:
        return None
    luma, a, bb = _rgb_luma_ab(rgb)
    return {
        "luma": float(np.median(luma[mask])),
        "a": float(np.median(a[mask])),
        "b": float(np.median(bb[mask])),
        "luma_s": max(0.055, float(np.std(luma[mask]))),
        "a_s": max(0.035, float(np.std(a[mask]))),
        "b_s": max(0.035, float(np.std(bb[mask]))),
    }


def _color_residual(rgb: np.ndarray, stats: Dict[str, float]) -> np.ndarray:
    luma, a, bb = _rgb_luma_ab(rgb)
    return np.sqrt(
        ((luma - stats["luma"]) / stats["luma_s"]) ** 2
        + ((a - stats["a"]) / stats["a_s"]) ** 2
        + ((bb - stats["b"]) / stats["b_s"]) ** 2
    ).astype(np.float32)


def _dilate_bool(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    H, W = mask.shape
    out = mask.copy()
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            if dr * dr + dc * dc > radius * radius:
                continue
            src_r0, src_r1 = max(0, -dr), min(H, H - dr)
            src_c0, src_c1 = max(0, -dc), min(W, W - dc)
            dst_r0, dst_r1 = max(0, dr), min(H, H + dr)
            dst_c0, dst_c1 = max(0, dc), min(W, W + dc)
            sl = mask[src_r0:src_r1, src_c0:src_c1]
            out[dst_r0:dst_r1, dst_c0:dst_c1] |= sl
    return out


def _vertical_close_bool(mask: np.ndarray, k: int = 9) -> np.ndarray:
    """Close thin vertical gaps (chair legs) along columns."""
    if k <= 1:
        return mask
    H, W = mask.shape
    ker = np.ones(k, dtype=np.float32)
    out = np.zeros_like(mask)
    for c in range(W):
        col = mask[:, c].astype(np.float32)
        dil = np.convolve(col, ker, mode="same") > 0.5
        ero = np.convolve(dil.astype(np.float32), ker, mode="same") >= (k - 1e-3)
        out[:, c] = ero
    return out


def _keep_thin_runs(mask: np.ndarray, max_width: int = 6) -> np.ndarray:
    """Keep only narrow horizontal runs so dark floors / rugs are not treated as legs."""
    H, W = mask.shape
    out = np.zeros_like(mask)
    for r in range(H):
        c = 0
        while c < W:
            if not mask[r, c]:
                c += 1
                continue
            c1 = c
            while c1 < W and mask[r, c1]:
                c1 += 1
            if (c1 - c) <= max_width:
                out[r, c:c1] = True
            c = c1
    return out


def _fill_closed_rgb_contours(
    rgb: np.ndarray,
    sheet: np.ndarray,
    *,
    edge_thresh: float = 0.085,
    max_frac: float = 0.16,
    min_px: int = 80,
) -> np.ndarray:
    """Fill compact interiors of strong RGB contours that sit on the floor sheet (glass tables)."""
    H, W = sheet.shape
    gray = rgb.mean(axis=-1)
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = np.abs(gray[:, 2:] - gray[:, :-2]) * 0.5
    gy[1:-1, :] = np.abs(gray[2:, :] - gray[:-2, :]) * 0.5
    edge = (np.hypot(gx, gy) > edge_thresh) & _dilate_bool(sheet, 4)
    wall = _dilate_bool(edge, 1)
    # Flood from the image border through non-edge pixels → unenclosed background.
    bg = np.zeros((H, W), dtype=bool)
    stack: List[Tuple[int, int]] = []
    for r in range(H):
        for c in (0, W - 1):
            if not wall[r, c]:
                bg[r, c] = True
                stack.append((r, c))
    for c in range(W):
        for r in (0, H - 1):
            if not wall[r, c] and not bg[r, c]:
                bg[r, c] = True
                stack.append((r, c))
    while stack:
        r, c = stack.pop()
        for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and (not wall[nr, nc]) and (not bg[nr, nc]):
                bg[nr, nc] = True
                stack.append((nr, nc))
    interior = (~wall) & (~bg) & sheet
    if interior.sum() < min_px or interior.mean() > max_frac:
        return np.zeros((H, W), dtype=bool)
    return interior | (wall & sheet & _dilate_bool(interior, 2))


def _grow_floor_sheet_np(
    z: np.ndarray,
    valid: np.ndarray,
    *,
    seed_row_frac: float = 0.80,
    rel_tol: float = 0.034,
    abs_tol: float = 0.10,
) -> np.ndarray:
    """Depth-only floor sheet: BFS from the bottom band. Stop at closer surfaces.

    Far / into-the-page samples stay on the sheet if depth is continuous. Do not
    treat missing (black) pixels as walls.
    """
    H, W = z.shape
    seed_r = int(seed_row_frac * H)
    seed = valid.copy()
    seed[:seed_r] = False
    visited = np.zeros((H, W), dtype=bool)
    out = np.zeros((H, W), dtype=bool)
    stack: List[Tuple[int, int]] = []
    for r in range(seed_r, H):
        for c in range(W):
            if seed[r, c]:
                visited[r, c] = True
                out[r, c] = True
                stack.append((r, c))
    if not stack:
        return out
    while stack:
        r, c = stack.pop()
        z0 = z[r, c]
        tol = max(abs_tol, rel_tol * z0)
        closer = max(0.045, 0.016 * z0)
        for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= H or nc < 0 or nc >= W or visited[nr, nc]:
                continue
            if not valid[nr, nc]:
                continue
            zn = z[nr, nc]
            if abs(zn - z0) > tol:
                continue
            if (z0 - zn) > closer:
                continue
            visited[nr, nc] = True
            out[nr, nc] = True
            stack.append((nr, nc))
    return out


def _box_sum(x: np.ndarray, k: int) -> np.ndarray:
    """Inclusive box sum via integral image. x is 2-D float32."""
    pad = k // 2
    p = np.pad(x, pad, mode="edge")
    c = np.pad(p, ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)
    return c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]


def _local_ncc(a: np.ndarray, b: np.ndarray, k: int = 7) -> np.ndarray:
    """Per-pixel cosine of mean-centered k×k patches (normalized cross-correlation)."""
    a = a.astype(np.float32, copy=False)
    b = b.astype(np.float32, copy=False)
    n = float(k * k)
    sa, sb = _box_sum(a, k), _box_sum(b, k)
    sa2, sb2, sab = _box_sum(a * a, k), _box_sum(b * b, k), _box_sum(a * b, k)
    ma, mb = sa / n, sb / n
    num = sab - n * ma * mb
    den = np.sqrt(np.maximum((sa2 - n * ma * ma) * (sb2 - n * mb * mb), 0.0))
    out = np.zeros_like(a, dtype=np.float32)
    good = den > 1e-6
    out[good] = num[good] / den[good]
    return np.clip(out, -1.0, 1.0)


def _rgbd_local_cosine(
    rgb: np.ndarray,
    z: np.ndarray,
    valid: np.ndarray,
    *,
    win: int = 7,
) -> np.ndarray:
    """
    Per-pixel cosine match between RGB and the 2-D depth surface.

    `cos_surf`: local shape of luma vs raw depth (the rendered range image).
    `cos_grad`: local RGB edges vs depth edges. Reflections / white glare have
    RGB structure without a matching depth ridge → low score; chair legs agree.
    """
    luma = rgb.mean(axis=-1).astype(np.float32)
    zz = z.astype(np.float32)
    zz = np.where(valid, zz, np.median(zz[valid]) if valid.any() else 2.0)
    gx = np.zeros_like(luma)
    gy = np.zeros_like(luma)
    zx = np.zeros_like(zz)
    zy = np.zeros_like(zz)
    gx[:, 1:-1] = 0.5 * np.abs(luma[:, 2:] - luma[:, :-2])
    gy[1:-1, :] = 0.5 * np.abs(luma[2:, :] - luma[:-2, :])
    zx[:, 1:-1] = 0.5 * np.abs(zz[:, 2:] - zz[:, :-2])
    zy[1:-1, :] = 0.5 * np.abs(zz[2:, :] - zz[:-2, :])
    cos_surf = _local_ncc(luma, zz, k=win)
    cos_grad = _local_ncc(np.hypot(gx, gy), np.hypot(zx, zy), k=win)
    agree = 0.45 * cos_surf + 0.55 * cos_grad
    agree = np.where(valid, agree, 0.0)
    return agree.astype(np.float32)


def _rgb_complete_broken_legs(
    rgb: np.ndarray,
    z: np.ndarray,
    valid: np.ndarray,
    sheet: np.ndarray,
    agree: Optional[np.ndarray] = None,
    *,
    agree_min: float = 0.22,
) -> np.ndarray:
    """RGB stitches thin legs only where RGB-D local cosine says the two views agree."""
    if agree is None:
        agree = _rgbd_local_cosine(rgb, z, valid)
    near = _dilate_bool(sheet, 3)
    d_step = _depth_step_obstacle_np(z, np.zeros_like(z), valid, h_max=1.0, step_rel=0.028, step_abs=0.045)
    # Parquet / glare depth noise is a step without RGB-D agreement — drop it.
    depth_frag = _keep_thin_runs(d_step & near & (agree >= agree_min), max_width=8)
    if not depth_frag.any():
        return np.zeros_like(sheet)
    luma = rgb.mean(axis=-1)
    # White / specular RGB is not a leg unless depth also agrees.
    rgb_v = _rgb_vertical_obstacle_np(rgb, _dilate_bool(depth_frag, 5), min_grad=0.070, dark_thresh=0.45)
    rgb_v &= (agree >= agree_min) & (luma < 0.82)
    legs = depth_frag | (rgb_v & _dilate_bool(depth_frag, 4))
    legs = _vertical_close_bool(legs, k=13)
    legs = _keep_thin_runs(legs, max_width=7)
    return _dilate_bool(legs, 1) & near


def _min_blob_np(mask: np.ndarray, min_px: int = 12) -> np.ndarray:
    """Drop connected components smaller than `min_px` (floor speckle from thin fill)."""
    if min_px <= 1 or not mask.any():
        return mask
    H, W = mask.shape
    out = np.zeros_like(mask)
    seen = np.zeros_like(mask)
    for r0 in range(H):
        for c0 in range(W):
            if not mask[r0, c0] or seen[r0, c0]:
                continue
            stack = [(r0, c0)]
            seen[r0, c0] = True
            blob: List[Tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                blob.append((r, c))
                for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            if len(blob) >= min_px:
                for r, c in blob:
                    out[r, c] = True
    return out


def _local_far_depth_np(z: np.ndarray, valid: np.ndarray, *, hy: int = 5, wx: int = 2, q: float = 75.0) -> np.ndarray:
    """Local far-depth envelope (percentile of a (2hy+1)×(2wx+1) window)."""
    H, W = z.shape
    zf = np.where(valid, z, np.nan).astype(np.float32)
    pad = np.pad(zf, ((hy, hy), (wx, wx)), mode="edge")
    try:
        from numpy.lib.stride_tricks import sliding_window_view

        win = sliding_window_view(pad, (2 * hy + 1, 2 * wx + 1))
        with np.errstate(all="ignore"):
            out = np.nanpercentile(win, q, axis=(-1, -2)).astype(np.float32)
    except Exception:
        out = np.zeros_like(z, dtype=np.float32)
        for r in range(H):
            r0, r1 = r, r + 2 * hy + 1
            for c in range(W):
                sl = pad[r0:r1, c : c + 2 * wx + 1]
                vals = sl[np.isfinite(sl)]
                out[r, c] = float(np.percentile(vals, q)) if vals.size else z[r, c]
    fill = float(np.nanmedian(zf)) if np.isfinite(zf).any() else 2.0
    out[~np.isfinite(out)] = fill
    return out


def _local_protrude_np(z: np.ndarray, valid: np.ndarray, *, hy: int = 5, wx: int = 2) -> np.ndarray:
    """True where a pixel is closer than the local far-depth envelope (sticks up from nearby floor)."""
    floor_z = _local_far_depth_np(z, valid, hy=hy, wx=wx, q=75.0)
    margin = np.maximum(0.09, 0.032 * np.maximum(z, floor_z))
    return valid & ((floor_z - z) > margin)


def _rgbd_cosine_complete_np(
    rgb: np.ndarray,
    z: np.ndarray,
    h: np.ndarray,
    valid: np.ndarray,
    sheet: np.ndarray,
    agree: np.ndarray,
    already: Optional[np.ndarray] = None,
    *,
    agree_seed: float = 0.32,
    agree_grow: float = 0.32,
) -> np.ndarray:
    """
    Cosine-gated thin-leg completion.

    Seeds = depth steps on the rim of already-high furniture whose RGB and depth
    agree. Floor-tile grout (00001 aisle, 00103) has steps + cosine but is not
    next to furniture, so it is not a seed. Grow is dark vertical RGB from those
    seeds only; wide carpets are dropped by `_keep_thin_runs`.
    """
    near = _dilate_bool(sheet, 2)
    core = already if already is not None else ~sheet
    rim = _dilate_bool(core, 4) & sheet
    d_step = _depth_step_obstacle_np(z, h, valid, h_max=0.22, step_rel=0.040, step_abs=0.060)
    seeds = d_step & rim & (agree >= agree_seed) & valid
    if not seeds.any():
        return np.zeros_like(sheet)
    luma = rgb.mean(axis=-1)
    rgb_v = _rgb_vertical_obstacle_np(rgb, _dilate_bool(seeds, 5), min_grad=0.070, dark_thresh=0.42)
    rgb_v &= (agree >= agree_grow) & (luma < 0.55)
    legs = seeds | rgb_v
    legs = _vertical_close_bool(legs, k=7)
    legs = _keep_thin_runs(legs, max_width=5)
    return legs & near


def _rgb_drop_legs_np(
    rgb: np.ndarray,
    already: np.ndarray,
    sheet: np.ndarray,
    agree: np.ndarray,
    protrude: Optional[np.ndarray] = None,
    *,
    max_down: int = 72,
    color_resid: float = 2.2,
) -> np.ndarray:
    """
    Drop thin legs from the underside of *protruding* furniture onto the floor sheet.

    Seeds are the bottom edge of already-high pixels that locally stick up (chair /
    table, not a tilted wall). Depth often reads floor under the shank (cosine low);
    colour residual vs the floor sheet plus `_keep_thin_runs` paints the missing
    column and refuses wide carpet / shadow.
    """
    H, W = already.shape
    luma = rgb.mean(axis=-1)
    stats = _color_stats(rgb, sheet)
    resid = _color_residual(rgb, stats) if stats is not None else np.zeros((H, W), dtype=np.float32)
    below = np.zeros_like(already)
    below[:-1] = already[1:]
    bottom = already & ~below
    if protrude is not None:
        bottom = bottom & protrude
    out = np.zeros_like(already)
    min_r = int(0.28 * H)
    for c in range(W):
        rows = np.where(bottom[:, c])[0]
        if rows.size == 0:
            continue
        r0 = int(rows.max())
        if r0 < min_r:
            continue
        r = r0 + 1
        n = 0
        while r < H and n < max_down:
            if already[r, c] or out[r, c]:
                r += 1
                n += 1
                continue
            # Darkness alone is not enough (00001 lower-left shadow). Require
            # a colour mismatch vs the floor sheet or RGB-D cosine agreement.
            not_floor = resid[r, c] >= color_resid
            cosine_hit = agree[r, c] >= 0.32
            if not (not_floor or cosine_hit):
                break
            if luma[r, c] >= 0.72:
                break
            out[r, c] = True
            r += 1
            n += 1
    return _keep_thin_runs(out, max_width=5)


def _rgb_edge_fill_np(
    rgb: np.ndarray,
    already: np.ndarray,
    agree: np.ndarray,
    height: np.ndarray,
    *,
    agree_min: float = 0.22,
    min_height: float = 0.0,
) -> np.ndarray:
    """1-pixel RGB edge fill on the border of current obstacles. Height is optional."""
    del height
    ring = _dilate_bool(already, 1) & ~already
    if not ring.any():
        return np.zeros_like(already)
    luma = rgb.mean(axis=-1)
    rgb_v = _rgb_vertical_obstacle_np(rgb, ring, min_grad=0.070, dark_thresh=0.45)
    return rgb_v & (agree >= agree_min) & (luma < 0.82)


def floor_rgbd_sheet_fields(
    depth: torch.Tensor,
    rgb: Optional[torch.Tensor] = None,
    *,
    inherit_border: bool = True,
    obstacle_height: float = 0.15,
) -> Optional[Dict[str, torch.Tensor]]:
    """
    Pixel oracle: raw depth height + tilt veto + RGB-D cosine / colour leg completion.

    1. Measured pixel with plane height above the Kinect noise floor → obstacle.
    2. Unmeasured (black / into-the-page) is not an obstacle.
    3. Lower-image plane hits that do *not* locally protrude (carpet, tilt residual,
       image-border floor) are vetoed. Walls / pictures stay via the upper-image plane.
    4. RGB-D cosine completes thin legs where depth and RGB agree; colour-drop
       paints shanks that Kinect reads as floor (cosine low). No sheet-BFS judge.
    """
    base = floor_plane_fields_v2(depth, inherit_border=False)
    if base is None:
        return None
    z = depth[:, 0]
    valid = z > DEPTH_VALID_LO
    h = base["relative_height"][:, 0]
    obst_tol = kinect_depth_tolerance(z, obstacle_height)
    obst = valid & (h > obst_tol)
    thin_list = []
    agree_list = []
    B, _, H, W = depth.shape
    for b in range(B):
        zb = z[b].detach().cpu().numpy()
        vb = valid[b].detach().cpu().numpy()
        hb = h[b].detach().cpu().numpy()
        already = obst[b].detach().cpu().numpy()
        prot = _local_protrude_np(zb, vb)
        rows = np.arange(H)[:, None]
        band = vb & (rows >= int(0.55 * H))
        drift = float(np.median(np.abs(hb[band]))) if band.any() else 0.0
        # Severe leftover tilt (00001 LL, 00025 carpet): clear non-protruding
        # lower-image plane hits. Mild frames only clear h < 0.40 m so chairs
        # stay and the frozen 28-frame S/G remain connected.
        if drift > 0.28:
            veto = already & (~prot) & (rows >= int(0.38 * H))
        else:
            veto = already & (~prot) & (hb < 0.40)
        already = already & ~veto
        thin = np.zeros_like(already)
        agree = np.zeros(already.shape, dtype=np.float32)
        if rgb is not None:
            if rgb.dim() != 4 or rgb.shape[1] != 3:
                raise ValueError("rgb must be [B, 3, H, W]")
            rgb_np = np.clip(rgb[b].permute(1, 2, 0).detach().cpu().numpy(), 0.0, 1.0)
            agree = _rgbd_local_cosine(rgb_np, zb, vb)
            sheet = vb & ~already
            thin = _rgbd_cosine_complete_np(rgb_np, zb, hb, vb, sheet, agree, already)
            thin = thin | _rgb_edge_fill_np(rgb_np, already, agree, hb, min_height=0.0)
            thin = _min_blob_np(thin, min_px=16)
        obst[b] = torch.from_numpy(already | thin).to(obst.device)
        thin_list.append(thin.astype(np.float32))
        agree_list.append(agree.astype(np.float32))
    if inherit_border:
        obst = _inherit_border(obst.unsqueeze(1).float())[:, 0] > 0.5
    floor_prob = torch.where(obst.unsqueeze(1), torch.full_like(h, 0.08).unsqueeze(1), torch.full_like(h, 0.92).unsqueeze(1))
    height_obstacle = torch.where(obst.unsqueeze(1), torch.full_like(h, 0.94).unsqueeze(1), torch.full_like(h, 0.06).unsqueeze(1))
    out = dict(base)
    out["floor_prob"] = floor_prob
    out["height_obstacle"] = height_obstacle
    out["thin_obstacle"] = torch.stack([torch.from_numpy(x) for x in thin_list]).unsqueeze(1).to(depth.device)
    out["rgbd_cosine"] = torch.stack([torch.from_numpy(x) for x in agree_list]).unsqueeze(1).to(depth.device)
    out["skip_morph"] = torch.ones(B, 1, 1, 1, device=depth.device)
    return out


def floor_height_fields(
    depth: torch.Tensor,
    floor_band_start: float = 0.55,
    floor_temperature: float = 12.0,
    height_temperature: float = 16.0,
    floor_margin_ratio: float = 0.08,
    mode: Optional[str] = None,
    rgb: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    Estimate floor likelihood and height obstacle from depth.

    mode='median' (legacy): range-band heuristic — |depth − median(lower band)|. NOT a true floor
    detector (walls at the same range read as floor). mode='plane': geometric plane fit
    (`floor_plane_fields`), falling back to median if the fit fails. Default from env NAV_FLOOR_ORACLE.
    """
    if depth.dim() != 4 or depth.shape[1] != 1:
        raise ValueError("depth must have shape [B, 1, H, W]")
    mode = mode or floor_oracle_mode()
    if mode == "rgbd_sheet":
        out = floor_rgbd_sheet_fields(depth, rgb)
        if out is not None:
            return out
    elif mode == "rgbd_pixel":
        out = floor_rgbd_pixel_fields(depth, rgb)
        if out is not None:
            return out
    elif mode == "plane_v3":
        out = floor_plane_fields_v3(depth)
        if out is not None:
            return out
    elif mode == "plane_v2":
        out = floor_plane_fields_v2(depth)
        if out is not None:
            return out
    elif mode == "plane":
        out = floor_plane_fields(depth)
        if out is not None:
            return out
    _, _, H, _ = depth.shape
    lower = depth[:, :, int(H * floor_band_start) :, :]
    floor_depth = lower.flatten(2).median(dim=-1).values.view(depth.shape[0], 1, 1, 1)
    rel_height = depth - floor_depth
    margin = (floor_depth.abs().clamp_min(1.0) * floor_margin_ratio).clamp_min(0.05)
    floor_prob = torch.sigmoid((margin - rel_height.abs()) * floor_temperature)
    height_obstacle = torch.sigmoid((rel_height.abs() - margin) * height_temperature)
    return {
        "floor_prob": floor_prob,
        "height_obstacle": height_obstacle,
        "floor_depth": floor_depth,
        "relative_height": rel_height,
    }
