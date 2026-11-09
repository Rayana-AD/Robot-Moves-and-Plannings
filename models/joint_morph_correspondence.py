"""
Scheme B: start–goal guided morph sweep = sole manifold↔voxel correspondence.

1) Opening-centered manifold search: find whole-arm-free q with EE in the slit.
2) Build morph band; inject Opening seeds into the mid third.
3) Color band (whole-arm collision + distal/EE Opening); proximal SOLID voxels
   alone do not veto an EE-in-slit free config.
4) SafeTopo temporal Poisson bridges m_start → Opening seeds → m_goal on the
   7-DoF joint-manifold samples (not a pure 2D (t,w) DP).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple  # Sequence used by obstacles

import numpy as np

from models.joint_manifold_detect import (
    LABEL_FREE,
    LABEL_OPENING,
    LABEL_SOLID,
    OpeningDetectField,
    binary_masks_from_detect,
    extract_opening_z_geometry,
)
from models.joint_planner import JointNormalizer


def clear_morph_query_caches(env) -> None:
    """Drop FK/collision caches (call after base move or scene change)."""
    for attr in ("_morph_fk_cache", "_morph_coll_cache"):
        if hasattr(env, attr):
            try:
                getattr(env, attr).clear()
            except Exception:
                setattr(env, attr, {})


def _q_cache_key(q: np.ndarray) -> Tuple[float, ...]:
    return tuple(np.round(np.asarray(q, dtype=np.float64).reshape(-1), 5).tolist())


def _fk_ee(env, q: np.ndarray) -> np.ndarray:
    cache = getattr(env, "_morph_fk_cache", None)
    if cache is None:
        cache = {}
        try:
            env._morph_fk_cache = cache  # type: ignore[attr-defined]
        except Exception:
            pass
    key = _q_cache_key(q)
    hit = cache.get(key)
    if hit is not None:
        return np.asarray(hit, dtype=np.float64).copy()
    ee = np.asarray(env.forward_kinematics(q), dtype=np.float64).reshape(-1)[:3].copy()
    cache[key] = ee
    return ee.copy()


def _clip_q(env, q: np.ndarray) -> np.ndarray:
    """Keep joints inside env limits so manifold↔grid stays invertible."""
    q = np.asarray(q, dtype=np.float64).reshape(-1).copy()
    lo = getattr(env, "joint_min", None)
    hi = getattr(env, "joint_max", None)
    if lo is None or hi is None:
        return q
    return np.clip(q, np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64))


def refine_q_toward_ee(
    q0: np.ndarray,
    target_ee: np.ndarray,
    env,
    *,
    steps: int = 8,
    step_scale: float = 0.12,
) -> np.ndarray:
    """Coordinate-descent joint nudge toward EE target (obstacles ignored)."""
    q = _clip_q(env, q0)
    tgt = np.asarray(target_ee, dtype=np.float64)[:3]
    best = q.copy()
    best_d = float(np.linalg.norm(_fk_ee(env, best) - tgt))
    scale = float(step_scale)
    for _ in range(int(steps)):
        improved = False
        for j in range(q.shape[0]):
            for s in (-scale, scale, -0.5 * scale, 0.5 * scale):
                cand = q.copy()
                cand[j] = cand[j] + s
                cand = _clip_q(env, cand)
                d = float(np.linalg.norm(_fk_ee(env, cand) - tgt))
                if d + 1e-5 < best_d:
                    best, best_d, q = cand, d, cand
                    improved = True
        if not improved:
            scale *= 0.55
            if scale < 5e-4:
                break
    return _clip_q(env, best)


def _gap_anchor(env, detect_field: Optional[OpeningDetectField]) -> np.ndarray:
    """Opening xyz for the workspace guide (Detect via / plane / opening_spec)."""
    if detect_field is not None and detect_field.via_ee is not None:
        return np.asarray(detect_field.via_ee, dtype=np.float64).reshape(-1)[:3].copy()
    if detect_field is not None and detect_field.z_geometry is not None:
        zgeo = detect_field.z_geometry
        return np.array(
            [
                float(zgeo.opening_xy_center[0]),
                float(zgeo.opening_xy_center[1]),
                0.5 * (float(zgeo.opening_z[0]) + float(zgeo.opening_z[1])),
            ],
            dtype=np.float64,
        )
    if detect_field is not None and isinstance(detect_field.plane_roi, dict):
        roi = detect_field.plane_roi
        if float(roi.get("ok", 0.0)) >= 0.5:
            cx, cy = [float(v) for v in roi["center_xy"]]
            z = 0.55
            spec = getattr(env, "_opening_spec", None)
            if isinstance(spec, dict) and spec.get("gap_center") is not None:
                z = float(np.asarray(spec["gap_center"], dtype=np.float64).reshape(-1)[2])
            return np.array([cx, cy, z], dtype=np.float64)
    spec = getattr(env, "_opening_spec", None)
    if isinstance(spec, dict) and spec.get("gap_center") is not None:
        return np.asarray(spec["gap_center"], dtype=np.float64).reshape(-1)[:3].copy()
    return np.array([0.45, 0.0, 0.45], dtype=np.float64)


def guide_polyline(
    ee_s: np.ndarray,
    ee_g: np.ndarray,
    gap_center: np.ndarray,
    *,
    T: int,
    opening_z: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """Workspace guide: start EE → opening → goal EE (z held in aperture band)."""
    gap = np.asarray(gap_center, dtype=np.float64)[:3].copy()
    z_lo = z_hi = None
    if opening_z is not None:
        z_lo, z_hi = float(opening_z[0]), float(opening_z[1])
        gap[2] = float(np.clip(gap[2], z_lo, z_hi))
    else:
        # Prefer Detect gap height; only gently blend if EE already near aperture.
        z_ee = 0.5 * (float(ee_s[2]) + float(ee_g[2]))
        if abs(z_ee - float(gap[2])) < 0.18:
            gap[2] = 0.85 * float(gap[2]) + 0.15 * z_ee
    ee_s = np.asarray(ee_s, dtype=np.float64)[:3].copy()
    ee_g = np.asarray(ee_g, dtype=np.float64)[:3].copy()
    out = np.zeros((T, 3), dtype=np.float64)
    for t in range(T):
        a = t / max(T - 1, 1)
        if a < 0.5:
            u = a / 0.5
            out[t] = (1.0 - u) * ee_s + u * gap
        else:
            u = (a - 0.5) / 0.5
            out[t] = (1.0 - u) * gap + u * ee_g
        # Hold mid guide inside aperture height when Detect z is known.
        if z_lo is not None and z_hi is not None and 0.30 <= a <= 0.70:
            out[t, 2] = float(np.clip(out[t, 2], z_lo, z_hi))
    return out


def _sample_joint_cloud(
    qs: np.ndarray,
    qg: np.ndarray,
    env,
    guide: np.ndarray,
    *,
    seed: int = 15,
    cheap: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compact joint cloud near LERP / endpoints + a few EE-pulled seeds on the guide."""
    qs = np.asarray(qs, dtype=np.float64).reshape(-1)
    qg = np.asarray(qg, dtype=np.float64).reshape(-1)
    dof = int(qs.shape[0])
    rng = np.random.default_rng(int(seed))
    cloud: List[np.ndarray] = []
    n_lerp = 20 if cheap else 48
    n_jitter = 12 if cheap else 36
    n_end = 12 if cheap else 40
    for a in np.linspace(0.0, 1.0, n_lerp):
        cloud.append((1.0 - a) * qs + a * qg)
    for a in np.linspace(0.0, 1.0, n_jitter):
        base = (1.0 - a) * qs + a * qg
        cloud.append(base + rng.normal(0.0, 0.16, size=dof))
    for _ in range(n_end):
        cloud.append(qs + rng.normal(0.0, 0.22, size=dof))
        cloud.append(qg + rng.normal(0.0, 0.22, size=dof))
    # Sparse EE-pulled seeds (through-gap critical); keep cheap.
    stride = max(1, len(guide) // (4 if cheap else 8))
    refine_n = 4 if cheap else 8
    for tgt in guide[::stride]:
        for a in ((0.5,) if cheap else (0.4, 0.55)):
            q0 = (1.0 - a) * qs + a * qg
            cloud.append(refine_q_toward_ee(q0, tgt, env, steps=refine_n, step_scale=0.14))
    if not cheap:
        # Mid-guide: denser redundancy samples so the free funnel can appear on the spine.
        mid0, mid1 = max(1, len(guide) // 3), max(2, (2 * len(guide)) // 3)
        for tgt in guide[mid0:mid1:max(1, (mid1 - mid0) // 4)]:
            for a in np.linspace(0.35, 0.65, 5):
                q0 = (1.0 - a) * qs + a * qg + rng.normal(0.0, 0.28, size=dof)
                cloud.append(refine_q_toward_ee(q0, tgt, env, steps=10, step_scale=0.13))
    q_arr = np.stack(cloud, axis=0)
    ee_arr = np.stack([_fk_ee(env, q) for q in q_arr], axis=0)
    return q_arr, ee_arr


def _arm_collides(q: np.ndarray, env, obstacles: Sequence, *, thr: float = 0.5) -> bool:
    from models.joint_planner import collision_cost_q

    cache = getattr(env, "_morph_coll_cache", None)
    if cache is None:
        cache = {}
        try:
            env._morph_coll_cache = cache  # type: ignore[attr-defined]
        except Exception:
            pass
    key = _q_cache_key(q)
    hit = cache.get(key)
    if hit is not None:
        return bool(hit)
    if env.check_self_collision(q):
        cache[key] = True
        return True
    coll = float(collision_cost_q(q, env, list(obstacles))) >= float(thr)
    cache[key] = bool(coll)
    return bool(coll)


def _search_free_near_ee(
    q0: np.ndarray,
    tgt_ee: np.ndarray,
    env,
    obstacles: Sequence,
    *,
    seed: int,
    tries: int = 28,
    refine_steps: int = 5,
) -> np.ndarray:
    """Local search: keep EE near target, prefer whole-arm free."""
    rng = np.random.default_rng(int(seed))
    q0 = _clip_q(env, q0)
    tgt = np.asarray(tgt_ee, dtype=np.float64)[:3]
    best = refine_q_toward_ee(q0, tgt, env, steps=min(8, refine_steps + 3), step_scale=0.12)

    def _key(q: np.ndarray) -> Tuple[int, float]:
        d = float(np.linalg.norm(_fk_ee(env, q) - tgt))
        free = 0 if not _arm_collides(q, env, obstacles) else 1
        return (free, d)

    best_key = _key(best)
    if best_key[0] == 0 and best_key[1] < 0.08:
        return _clip_q(env, best)
    for k in range(int(tries)):
        scale = 0.08 + 0.22 * (k / max(tries, 1))
        cand = _clip_q(env, best + rng.normal(0.0, scale, size=q0.shape[0]))
        cand = refine_q_toward_ee(cand, tgt, env, steps=int(refine_steps), step_scale=0.10)
        key = _key(cand)
        if key < best_key:
            best, best_key = cand, key
            if key[0] == 0 and key[1] < 0.06:
                break
    return _clip_q(env, best)


def greedy_open_morph(
    qs: np.ndarray,
    qg: np.ndarray,
    env,
    gap_center: np.ndarray,
    *,
    T: int = 48,
    seed: int = 15,
    obstacles: Optional[Sequence] = None,
    opening_z: Optional[Tuple[float, float]] = None,
    cheap: bool = False,
) -> np.ndarray:
    """Start/goal anchors → guide through gap → prefer whole-arm-free joints."""
    qs = np.asarray(qs, dtype=np.float64).reshape(-1)
    qg = np.asarray(qg, dtype=np.float64).reshape(-1)
    obs = list(obstacles) if obstacles is not None else []
    ee_s = _fk_ee(env, qs)
    ee_g = _fk_ee(env, qg)
    guide = guide_polyline(ee_s, ee_g, gap_center, T=T, opening_z=opening_z)
    q_cloud, ee_cloud = _sample_joint_cloud(qs, qg, env, guide, seed=seed, cheap=cheap)
    # Precompute collision flags for cloud (once).
    cloud_hit = np.array(
        [_arm_collides(q, env, obs) for q in q_cloud], dtype=bool
    )
    path = np.zeros((T, qs.shape[0]), dtype=np.float64)
    for t in range(T):
        tgt = guide[t]
        a_t = t / max(T - 1, 1)
        through = 0.28 <= a_t <= 0.72
        d = np.linalg.norm(ee_cloud - tgt[None, :], axis=1)
        score = d.copy()
        if t > 0:
            score = score + 0.08 * np.linalg.norm(q_cloud - path[t - 1][None, :], axis=1)
        # Strongly prefer whole-arm-free cloud members (harder near the slit).
        score = score + np.where(cloud_hit, 6.0 if through else 4.0, 0.0)
        i = int(np.argmin(score))
        q_seed = refine_q_toward_ee(
            q_cloud[i], tgt, env, steps=(3 if cheap else 6), step_scale=0.11
        )
        if _arm_collides(q_seed, env, obs):
            q_seed = _search_free_near_ee(
                q_seed,
                tgt,
                env,
                obs,
                seed=seed + t,
                tries=(12 if cheap else (52 if through else 28)),
                refine_steps=(3 if cheap else 5),
            )
        # If still hitting in the slit, try the best free cloud member near tgt.
        if through and _arm_collides(q_seed, env, obs) and not cheap:
            free_ix = np.flatnonzero(~cloud_hit)
            if free_ix.size > 0:
                j = int(free_ix[np.argmin(d[free_ix])])
                cand = refine_q_toward_ee(q_cloud[j], tgt, env, steps=8, step_scale=0.12)
                if not _arm_collides(cand, env, obs):
                    q_seed = cand
                else:
                    q_seed = _search_free_near_ee(
                        cand, tgt, env, obs, seed=seed + 100 + t, tries=40
                    )
        path[t] = _clip_q(env, q_seed)
    path[0] = qs
    path[-1] = qg
    # Mid-third collision-only rescue toward aperture EE (do not disturb free spine).
    gap = np.asarray(gap_center, dtype=np.float64)[:3]
    for t in range(T):
        a_t = t / max(T - 1, 1)
        if not (0.28 <= a_t <= 0.72):
            continue
        if not _arm_collides(path[t], env, obs):
            continue
        tgt = guide[t].copy()
        tgt[:2] = 0.70 * tgt[:2] + 0.30 * gap[:2]
        if opening_z is not None:
            tgt[2] = float(
                np.clip(0.5 * (opening_z[0] + opening_z[1]), opening_z[0], opening_z[1])
            )
        path[t] = _search_free_near_ee(
            path[t], tgt, env, obs, seed=seed + 200 + t, tries=56
        )
    path[0] = qs
    path[-1] = qg
    return path


def _slit_basis_xyz(
    gap_center: Optional[np.ndarray],
    detect_field: Optional[OpeningDetectField] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Workspace basis at the opening: along-slit, through-wall, up.

    door_frame / Y-slit in X-wall → along≈Y, through≈X. Prefer Detect half-extents
    when available so lateral columns slide inside the aperture, not joint-space noise.
    """
    along = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    thru = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    zgeo = None if detect_field is None else getattr(detect_field, "z_geometry", None)
    if zgeo is not None:
        try:
            hx = float(np.asarray(zgeo.opening_xy_half, dtype=np.float64).reshape(-1)[0])
            hy = float(np.asarray(zgeo.opening_xy_half, dtype=np.float64).reshape(-1)[1])
            if hx > hy + 1e-6:
                along = np.array([1.0, 0.0, 0.0], dtype=np.float64)
                thru = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        except Exception:
            pass
    del gap_center
    return along, thru, up


def lateral_band_q(
    path_q: np.ndarray,
    env=None,
    *,
    W: int = 11,
    lat_rad: float = 0.14,
    obstacles: Optional[Sequence] = None,
    gap_center: Optional[np.ndarray] = None,
    detect_field: Optional[OpeningDetectField] = None,
    cheap: bool = False,
    cheap_strict: bool = False,
) -> np.ndarray:
    """
    (T,W,dof) band by **workspace lateral** along the slit (not joint-space ±e0/e1).

    Joint-space side offsets shove EE off the aperture and out of the whole-arm
    free funnel. Here each column targets spine_EE + α·along_slit (+ small z),
    then IK-nudge / free search so cells stay in the through-gap corridor.
    ``lat_rad`` is max EE lateral displacement in meters along the slit.
    """
    path_q = np.asarray(path_q, dtype=np.float64)
    T, dof = path_q.shape
    W = max(int(W), 1)
    mid = W // 2
    obs = list(obstacles) if obstacles is not None else []
    if env is None:
        # No FK available — fall back to tiny joint jitter (legacy).
        cell_q = np.zeros((T, W, dof), dtype=np.float64)
        for t in range(T):
            for w in range(W):
                off = 0.0 if W == 1 else 0.04 * (w - mid) / max(mid, 1)
                q = path_q[t].copy()
                q[min(1, dof - 1)] += off
                cell_q[t, w] = q
        return cell_q

    along, thru, up = _slit_basis_xyz(gap_center, detect_field)
    # Cap lateral by Detect half-width when known (stay inside aperture).
    lat_m = float(lat_rad)
    zgeo = None if detect_field is None else getattr(detect_field, "z_geometry", None)
    if zgeo is not None:
        try:
            half = np.asarray(zgeo.opening_xy_half, dtype=np.float64).reshape(-1)
            slit_half = float(max(half[0], half[1]))
            lat_m = float(min(lat_m, max(0.06, 0.85 * slit_half)))
        except Exception:
            pass

    cell_q = np.zeros((T, W, dof), dtype=np.float64)
    # Cheap path: only mid ±k columns get workspace IK; outer cols use joint jitter.
    lat_span = 3 if (cheap and cheap_strict) else 2
    w_budget = (
        set(range(max(0, mid - lat_span), min(W, mid + lat_span + 1)))
        if cheap
        else set(range(W))
    )
    ik_steps = 4 if (cheap and cheap_strict) else (3 if cheap else 6)
    ik_tries = 10 if (cheap and cheap_strict) else (6 if cheap else 16)
    ik_refine = 4 if (cheap and cheap_strict) else (3 if cheap else 5)
    for t in range(T):
        q_spine = path_q[t]
        ee0 = _fk_ee(env, q_spine)
        # Mid third: pure along-slit. Approach/exit: smaller lateral + slight thru.
        a_t = t / max(T - 1, 1)
        mid_w = 1.0 if 0.28 <= a_t <= 0.72 else 0.45
        for w in range(W):
            if W == 1 or w == mid:
                cell_q[t, w] = q_spine
                continue
            alpha = float(w - mid) / float(max(mid, 1))  # [-1, 1]
            if w not in w_budget:
                qw = q_spine.copy()
                qw[min(1, dof - 1)] += 0.04 * alpha
                cell_q[t, w] = _clip_q(env, qw)
                continue
            # Alternate small z / thru so neighboring columns explore redundancy.
            z_off = 0.04 * alpha * (1.0 if (w % 2 == 0) else -0.5)
            thru_off = 0.03 * alpha * (0.0 if mid_w > 0.9 else 0.6)
            tgt = (
                ee0
                + (mid_w * lat_m * alpha) * along
                + thru_off * thru
                + z_off * up
            )
            q = refine_q_toward_ee(q_spine, tgt, env, steps=ik_steps, step_scale=0.12)
            if _arm_collides(q, env, obs):
                q = _search_free_near_ee(
                    q,
                    tgt,
                    env,
                    obs,
                    seed=15 + 17 * t + w,
                    tries=ik_tries,
                    refine_steps=ik_refine,
                )
            # Keep mid free if we found nothing better than spine.
            if _arm_collides(q, env, obs) and not _arm_collides(q_spine, env, obs):
                q = q_spine.copy()
            cell_q[t, w] = q
        cell_q[t, mid] = q_spine
    return cell_q


def cell_ee_from_q(env, cell_q: np.ndarray) -> np.ndarray:
    T, W, _ = cell_q.shape
    out = np.zeros((T, W, 3), dtype=np.float64)
    for t in range(T):
        for w in range(W):
            out[t, w] = _fk_ee(env, cell_q[t, w])
    return out


def _ee_in_opening_geometry(
    ee: np.ndarray,
    detect_field: OpeningDetectField,
    *,
    xy_scale: float = 1.05,
) -> bool:
    """True if EE lies in Detect Opening AABB (geometric slit), not link-SOLID veto."""
    zgeo = detect_field.z_geometry or extract_opening_z_geometry(detect_field)
    if zgeo is None:
        return int(detect_field.label_at_ee(ee)) == LABEL_OPENING
    p = np.asarray(ee, dtype=np.float64).reshape(-1)[:3]
    c = np.asarray(zgeo.opening_xy_center, dtype=np.float64).reshape(-1)[:2]
    h = np.asarray(zgeo.opening_xy_half, dtype=np.float64).reshape(-1)[:2] * float(xy_scale)
    z0, z1 = float(zgeo.opening_z[0]), float(zgeo.opening_z[1])
    if not (z0 - 0.02 <= float(p[2]) <= z1 + 0.02):
        return False
    return bool(abs(p[0] - c[0]) <= h[0] and abs(p[1] - c[1]) <= h[1])


def _ee_near_bottleneck(
    ee: np.ndarray,
    detect_field: OpeningDetectField,
    env,
    *,
    radius_m: float = 0.11,
) -> bool:
    """
    Task-general bottleneck test: Opening AABB when present, else ball around
    geometric gap (arch underside, bugtrap exit, chicane bend, …).
    """
    if _ee_in_opening_geometry(ee, detect_field):
        return True
    # Soft OPENING voxel at EE.
    try:
        if int(detect_field.label_at_ee(ee)) == LABEL_OPENING:
            return True
    except Exception:
        pass
    gap = _gap_anchor(env, detect_field)
    return float(np.linalg.norm(np.asarray(ee, dtype=np.float64).reshape(-1)[:3] - gap)) <= float(
        radius_m
    )


def search_opening_manifold_seeds(
    env,
    detect_field: OpeningDetectField,
    normalizer: JointNormalizer,
    *,
    q_hint: Optional[np.ndarray] = None,
    obstacles: Optional[Sequence] = None,
    seed: int = 15,
    n_tries: int = 72,
    max_keep: int = 3,
    min_keep: int = 1,
) -> Dict[str, Any]:
    """
    Opening-centered 7-DoF search around the slit geometric center.

    Accept whole-arm-free q whose EE is inside the Opening AABB (primary),
    optionally with distal Opening voxels. Proximal SOLID voxels alone do not
    reject — that was wiping the free funnel when the shoulder grazes the frame hull.

    Early-stop: once ``min_keep`` seeds are found, remaining outer tries use a
    cheap local budget; stop entirely at ``max_keep`` (default 3 — enough for
    mid-band injection without burning a full 140×48 collision budget).
    """
    obs = list(obstacles) if obstacles is not None else []
    gap = _gap_anchor(env, detect_field)
    zgeo = detect_field.z_geometry or extract_opening_z_geometry(detect_field)
    if zgeo is not None:
        cx, cy = float(zgeo.opening_xy_center[0]), float(zgeo.opening_xy_center[1])
        hx, hy = float(zgeo.opening_xy_half[0]), float(zgeo.opening_xy_half[1])
        z0, z1 = float(zgeo.opening_z[0]), float(zgeo.opening_z[1])
    else:
        cx, cy, cz = float(gap[0]), float(gap[1]), float(gap[2])
        hx, hy, z0, z1 = 0.04, 0.12, cz - 0.08, cz + 0.08
    lo = np.asarray(normalizer.joint_min, dtype=np.float64)
    hi = np.asarray(normalizer.joint_max, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    hint = None if q_hint is None else np.asarray(q_hint, dtype=np.float64).reshape(-1)
    seeds: List[np.ndarray] = []
    seed_ee: List[np.ndarray] = []
    n_eval = 0
    # After min_keep hit, only a short outer-loop tail for diversity.
    post_hit_budget = 12
    stop_after_i: Optional[int] = None

    def _accept(q: np.ndarray) -> bool:
        if _arm_collides(q, env, obs):
            return False
        ee = _fk_ee(env, q)
        # Door slit OR family gap / exit / arch bottleneck.
        return _ee_near_bottleneck(ee, detect_field, env, radius_m=0.11)

    for i in range(int(n_tries)):
        if stop_after_i is not None and i >= int(stop_after_i):
            break
        if len(seeds) >= int(max_keep):
            break
        # Sample EE targets in the bottleneck volume (AABB or gap ball).
        if zgeo is not None:
            tgt = np.array(
                [
                    cx + rng.uniform(-0.85 * hx, 0.85 * hx),
                    cy + rng.uniform(-0.85 * hy, 0.85 * hy),
                    rng.uniform(z0 + 0.015, z1 - 0.015),
                ],
                dtype=np.float64,
            )
        else:
            # No Detect AABB: isotropic jitter around gap_anchor (bugtrap/arch/chicane).
            rad = 0.08
            tgt = gap + rng.uniform(-rad, rad, size=3)
            tgt[2] = float(np.clip(tgt[2], gap[2] - 0.06, gap[2] + 0.06))
        if hint is not None and i % 3 == 0:
            q0 = hint + rng.normal(0.0, 0.22, size=hint.shape[0])
        elif hint is not None and i % 3 == 1:
            q0 = 0.5 * hint + 0.5 * (lo + rng.random(lo.shape[0]) * (hi - lo))
        else:
            q0 = lo + rng.random(lo.shape[0]) * (hi - lo)
        q0 = _clip_q(env, q0)
        have = len(seeds)
        # Coarse→fine: full refine only before the first accepted seed.
        if have == 0:
            refine_n, local_tries, local_ref = 12, 24, 5
        else:
            refine_n, local_tries, local_ref = 6, 10, 3
        q = refine_q_toward_ee(q0, tgt, env, steps=refine_n, step_scale=0.16)
        n_eval += 1
        # Fast path: accept without local search when already good.
        if not _accept(q):
            q = _search_free_near_ee(
                q, tgt, env, obs, seed=seed + 17 * i, tries=local_tries, refine_steps=local_ref
            )
            if not _accept(q):
                continue
        ee = _fk_ee(env, q)
        # Dedup in joint space.
        if seeds and min(float(np.linalg.norm(q - s)) for s in seeds) < 0.12:
            continue
        seeds.append(q.copy())
        seed_ee.append(ee.copy())
        if len(seeds) >= int(min_keep) and stop_after_i is None:
            stop_after_i = i + 1 + int(post_hit_budget)

    return {
        "q_seeds": np.stack(seeds, axis=0) if seeds else np.zeros((0, int(normalizer.dof))),
        "ee_seeds": np.stack(seed_ee, axis=0) if seed_ee else np.zeros((0, 3)),
        "n_seeds": int(len(seeds)),
        "n_evals": int(n_eval),
        "gap_anchor": gap.tolist(),
    }


def inject_opening_seeds_into_band(
    cell_q: np.ndarray,
    cell_ee: np.ndarray,
    path_q: np.ndarray,
    q_seeds: np.ndarray,
    env,
    *,
    mid_frac: Tuple[float, float] = (0.32, 0.68),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Overwrite mid-third spine/band columns with Opening manifold seeds."""
    cell_q = np.asarray(cell_q, dtype=np.float64).copy()
    cell_ee = np.asarray(cell_ee, dtype=np.float64).copy()
    path_q = np.asarray(path_q, dtype=np.float64).copy()
    q_seeds = np.asarray(q_seeds, dtype=np.float64)
    if q_seeds.ndim != 2 or q_seeds.shape[0] == 0:
        return cell_q, cell_ee, path_q
    T, W, _ = cell_q.shape
    mid = W // 2
    t0 = int(np.clip(mid_frac[0] * (T - 1), 1, T - 2))
    t1 = int(np.clip(mid_frac[1] * (T - 1), t0 + 1, T - 1))
    n = int(q_seeds.shape[0])
    for k, t in enumerate(range(t0, t1 + 1)):
        q = q_seeds[k % n]
        path_q[t] = q
        cell_q[t, mid] = q
        cell_ee[t, mid] = _fk_ee(env, q)
        # Neighbors: small joint jitter around the seed (redundancy columns).
        for w in range(W):
            if w == mid:
                continue
            off = 0.04 * (w - mid) / max(mid, 1)
            qw = q.copy()
            qw[min(1, qw.shape[0] - 1)] += off
            qw = _clip_q(env, qw)
            cell_q[t, w] = qw
            cell_ee[t, w] = _fk_ee(env, qw)
    return cell_q, cell_ee, path_q


def color_band_whole_arm(
    cell_q: np.ndarray,
    cell_ee: np.ndarray,
    detect_field: OpeningDetectField,
    env,
    *,
    obstacles: Optional[Sequence] = None,
    coll_threshold: float = 0.5,
    cheap: bool = False,
) -> Dict[str, Any]:
    """
    Paint morph band on the **7-DoF joint manifold samples**.

    Hard SOLID: whole-arm mesh / self-collision, or distal keypoints in SOLID.
    OPENING: not colliding and (distal Opening voxels OR EE in Opening AABB).
    Proximal-only SOLID voxels no longer wipe Opening when EE is in the slit.
    """
    from models.joint_planner import collision_cost_q

    cell_q = np.asarray(cell_q, dtype=np.float64)
    cell_ee = np.asarray(cell_ee, dtype=np.float64)
    T, W = int(cell_q.shape[0]), int(cell_q.shape[1])
    obs = list(obstacles) if obstacles is not None else []

    base = binary_masks_from_detect(
        detect_field, (T, W), env=env, cell_xyz=cell_ee
    )
    chart_labels = np.full((T, W), LABEL_FREE, dtype=np.int8)
    solid_mask = np.zeros((T, W), dtype=bool)
    opening_mask = np.zeros((T, W), dtype=bool)
    open_boost = np.ones((T, W), dtype=np.float64)
    open_score = np.zeros((T, W), dtype=np.float64)
    plane_cost = np.asarray(base["plane_cost"], dtype=np.float64)
    corridor_soft = np.asarray(base["corridor_soft"], dtype=np.float64)
    prob = np.ones((T, W), dtype=np.float64)
    n_arm_hit = 0
    n_link_solid = 0
    has_link_api = (not cheap) and callable(getattr(env, "link_positions", None))

    for t in range(T):
        for w in range(W):
            q = cell_q[t, w]
            ee = cell_ee[t, w]
            if cheap:
                arm_hit = _arm_collides(q, env, obs, thr=float(coll_threshold))
            else:
                arm_hit = bool(env.check_self_collision(q)) or (
                    float(collision_cost_q(q, env, obs)) >= float(coll_threshold)
                )
            link_xyz = [ee]
            if has_link_api:
                try:
                    link_xyz = list(env.link_positions(q))
                except Exception:
                    link_xyz = [ee]
            labs = [int(detect_field.label_at_ee(p)) for p in link_xyz]
            distal = labs[-3:] if len(labs) >= 3 else labs
            distal_solid = any(lab == LABEL_SOLID for lab in distal)
            distal_open = any(lab == LABEL_OPENING for lab in distal)
            any_solid = any(lab == LABEL_SOLID for lab in labs)
            if any_solid:
                n_link_solid += 1
            ee_geo = _ee_in_opening_geometry(ee, detect_field)

            # Mesh collision always SOLID. Distal SOLID always SOLID.
            # Proximal-only SOLID is ignored when EE sits in the geometric slit.
            hard_solid = bool(arm_hit or distal_solid or (any_solid and not ee_geo))
            if hard_solid:
                if arm_hit:
                    n_arm_hit += 1
                chart_labels[t, w] = LABEL_SOLID
                solid_mask[t, w] = True
                opening_mask[t, w] = False
                open_boost[t, w] = 1.0
                open_score[t, w] = 0.0
                corridor_soft[t, w] = 1.35
                prob[t, w] = 0.0
                continue

            if distal_open or ee_geo:
                chart_labels[t, w] = LABEL_OPENING
                opening_mask[t, w] = True
                open_boost[t, w] = 0.20
                open_score[t, w] = 1.0
                corridor_soft[t, w] = 1.0
                prob[t, w] = 2.2
            else:
                chart_labels[t, w] = LABEL_FREE
                open_boost[t, w] = 1.0
                open_score[t, w] = 0.0
                corridor_soft[t, w] = float(base["corridor_soft"][t, w])
                prob[t, w] = 1.0

    row_max = np.maximum(prob.max(axis=1, keepdims=True), 1e-6)
    prob = prob / row_max
    opening_bin = opening_mask.astype(np.float64)
    zgeo = detect_field.z_geometry or extract_opening_z_geometry(detect_field)
    n_open_vox = int(np.sum(np.asarray(detect_field.labels) == LABEL_OPENING))
    return {
        "chart_labels": chart_labels,
        "solid_mask": solid_mask,
        "opening_mask": opening_mask,
        "link_hit": solid_mask.copy(),
        "open_boost": open_boost,
        "plane_cost": plane_cost,
        "corridor_soft": corridor_soft,
        "open_score": open_score,
        "intersect_mask": opening_mask,
        "union_mask": opening_mask,
        "prob": prob,
        "ee_xyz": cell_ee,
        "opening_z_head": opening_bin,
        "opening_xy_head": opening_bin,
        "via_z_head": opening_bin,
        "via_xy_head": opening_bin,
        "z_geometry": zgeo.as_dict(),
        "n_opening_bins": int(np.sum(opening_mask)),
        "n_solid_bins": int(np.sum(solid_mask)),
        "n_opening_voxels": int(n_open_vox),
        "n_arm_collision_bins": int(n_arm_hit),
        "n_link_voxel_solid_bins": int(n_link_solid),
        "voxel_res_m": float(detect_field.res),
        "clearance_m": float(detect_field.clearance_m),
        "fk_labeled": True,
        "whole_arm_labeled": True,
        "link_voxel_labeled": True,
        "dual_field_xyz": False,
        "correspondence": "morph_sweep_b",
    }


def _lerp_spine_q(qs: np.ndarray, qg: np.ndarray, T: int) -> np.ndarray:
    """Cheap joint LERP spine (no collision search)."""
    qs = np.asarray(qs, dtype=np.float64).reshape(-1)
    qg = np.asarray(qg, dtype=np.float64).reshape(-1)
    out = np.zeros((int(T), qs.shape[0]), dtype=np.float64)
    for t in range(int(T)):
        a = t / max(int(T) - 1, 1)
        out[t] = (1.0 - a) * qs + a * qg
    return out


def _piecewise_lerp_spine_q(q_anchors: Sequence[np.ndarray], T: int) -> np.ndarray:
    """Concatenate joint LERPs through waypoint ``q`` anchors."""
    anchors = [np.asarray(q, dtype=np.float64).reshape(-1) for q in q_anchors]
    if len(anchors) < 2:
        return _lerp_spine_q(anchors[0], anchors[-1], T)
    n_seg = len(anchors) - 1
    seg_lens = [max(int(T) // n_seg, 1)] * n_seg
    seg_lens[-1] = int(T) - sum(seg_lens[:-1])
    out = np.zeros((int(T), anchors[0].shape[0]), dtype=np.float64)
    ti = 0
    for seg_i in range(n_seg):
        qa, qb = anchors[seg_i], anchors[seg_i + 1]
        seg_t = max(int(seg_lens[seg_i]), 1)
        for k in range(seg_t):
            if ti >= int(T):
                break
            a = k / max(seg_t - 1, 1)
            out[ti] = (1.0 - a) * qa + a * qb
            ti += 1
    out[-1] = anchors[-1]
    return out


def _chicane_spine_q(
    qs: np.ndarray,
    qg: np.ndarray,
    env,
    normalizer: JointNormalizer,
    scene,
    *,
    T: int,
    seed: int,
) -> np.ndarray:
    """S-bend piecewise spine: start → 3 corridor gaps → goal (EE-guided)."""
    from models.joint_visible_passage import (
        chicane_corridor_gaps_xy,
        task_ee_anchors,
        _task_ee_z,
    )

    qs = np.asarray(qs, dtype=np.float64).reshape(-1)
    qg = np.asarray(qg, dtype=np.float64).reshape(-1)
    ee_s, ee_g, _, _ = task_ee_anchors(scene, placement="diagonal")
    z = float(_task_ee_z(scene))
    gaps = chicane_corridor_gaps_xy(scene)
    ee_wps = [ee_s]
    for gx, gy in gaps:
        ee_wps.append(np.array([gx, gy, z], dtype=np.float64))
    ee_wps.append(ee_g)
    q_anchors = [qs]
    q_hint = qs.copy()
    for ee_wp in ee_wps[1:-1]:
        q_hint = refine_q_toward_ee(q_hint, ee_wp, env, steps=22, step_scale=0.14)
        q_hint = normalizer.clip_grid(q_hint)
        q_anchors.append(q_hint.copy())
    q_anchors.append(qg)
    return _piecewise_lerp_spine_q(q_anchors, int(T))


def build_morph_sweep_chart(
    q_start: np.ndarray,
    q_goal: np.ndarray,
    env,
    normalizer: JointNormalizer,
    detect_field: OpeningDetectField,
    *,
    T: Optional[int] = None,
    W: int = 11,
    lat_rad: float = 0.14,
    seed: int = 15,
    obstacles: Optional[Sequence] = None,
    cheap: bool = False,
    cheap_strict: bool = False,
    joint_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Full scheme-B chart: morph spine → lateral band → whole-arm 有/无 → cell_m.

    When ``cheap=True`` and Opening seeds are found, skip heavy ``greedy_open_morph``
    and use a LERP spine + seed injection (collision still used for seeds/audit).
    ``cheap_strict`` keeps the cheap spine but raises opening-seed / lateral IK budgets
    (sid15-style backup, not the multi-minute ``greedy_open_morph`` path).

    Returns dict with ``cell_m``, ``chart_reg``, ``path_q``, ``cell_ee``, ``meta``.
    """
    qs = np.asarray(q_start, dtype=np.float64).reshape(-1)
    qg = np.asarray(q_goal, dtype=np.float64).reshape(-1)
    if T is None:
        T = 40
    T = int(max(T, 4))
    W = int(max(W, 1))
    # Fresh FK/collision caches for this morph build (same base/geometry).
    clear_morph_query_caches(env)

    gap = _gap_anchor(env, detect_field)
    obs = list(obstacles) if obstacles is not None else []
    opening_z = None
    zgeo = detect_field.z_geometry or extract_opening_z_geometry(detect_field)
    if zgeo is not None:
        opening_z = (float(zgeo.opening_z[0]), float(zgeo.opening_z[1]))

    # 1) Opening-centered manifold seeds (geometric slit + whole-arm free).
    # Early-stop: 1 seed is enough for the mid funnel; keep up to 3 for diversity.
    # Do not weaken n_tries in cheap mode — seed hit rate dominates success.
    seed_tries = 120 if (cheap and cheap_strict) else 72
    seed_keep = (2 if cheap_strict else 1) if cheap else 3
    seed_pack = search_opening_manifold_seeds(
        env,
        detect_field,
        normalizer,
        q_hint=0.5 * (qs + qg),
        obstacles=obs,
        seed=seed,
        n_tries=int(seed_tries),
        max_keep=int(seed_keep),
        min_keep=1,
    )
    # Alternate RNG if the first pass misses (matches setback rebuild seed+1).
    if int(seed_pack.get("n_seeds", 0)) == 0 and cheap:
        alt_tries = 96 if cheap_strict else 72
        alt_keep = 2 if cheap_strict else 1
        for alt_off in ((1, 2) if cheap_strict else (1,)):
            alt = search_opening_manifold_seeds(
                env,
                detect_field,
                normalizer,
                q_hint=0.5 * (qs + qg),
                obstacles=obs,
                seed=seed + int(alt_off),
                n_tries=int(alt_tries),
                max_keep=int(alt_keep),
                min_keep=1,
            )
            if int(alt.get("n_seeds", 0)) > 0:
                seed_pack = {
                    **alt,
                    "n_evals": int(seed_pack.get("n_evals", 0)) + int(alt.get("n_evals", 0)),
                }
                break
    q_seeds = np.asarray(seed_pack["q_seeds"], dtype=np.float64)

    chicane_scene = None
    fam = str((joint_meta or {}).get("obstacle_family_name") or "")
    if fam == "corridor_chicane" and joint_meta is not None:
        from models.joint_visible_passage import rebuild_arm_fit_sealed_scene

        chicane_scene = rebuild_arm_fit_sealed_scene(
            joint_meta, keep_gap_center=True, seed=int(seed),
        )

    # 2) Morph spine; if seeds exist under cheap mode, LERP + pin (skip heavy morph).
    spine_mode = "greedy_open_morph"
    if cheap and q_seeds.shape[0] > 0:
        if chicane_scene is not None:
            path_q = _chicane_spine_q(
                qs, qg, env, normalizer, chicane_scene, T=int(T), seed=int(seed),
            )
            spine_mode = "chicane_piecewise_lerp"
        else:
            path_q = _lerp_spine_q(qs, qg, T)
            spine_mode = "lerp_plus_opening_seed"
    elif cheap and q_seeds.shape[0] == 0:
        if chicane_scene is not None:
            path_q = _chicane_spine_q(
                qs, qg, env, normalizer, chicane_scene, T=int(T), seed=int(seed),
            )
            spine_mode = "chicane_piecewise_lerp"
        else:
            path_q = _lerp_spine_q(qs, qg, T)
            spine_mode = "lerp_no_seed_fallback"
    else:
        path_q = greedy_open_morph(
            qs,
            qg,
            env,
            gap,
            T=T,
            seed=seed,
            obstacles=obs,
            opening_z=opening_z,
            cheap=cheap,
        )
    if q_seeds.shape[0] > 0:
        t_mid = T // 2
        path_q[t_mid] = q_seeds[0]
        # Blend neighbors toward the seed so Poisson has a smooth funnel.
        for dt in range(1, max(2, T // 8)):
            a = 1.0 - dt / max(T // 8, 1)
            if t_mid - dt > 0:
                path_q[t_mid - dt] = (1.0 - 0.55 * a) * path_q[t_mid - dt] + (0.55 * a) * q_seeds[0]
            if t_mid + dt < T - 1:
                path_q[t_mid + dt] = (1.0 - 0.55 * a) * path_q[t_mid + dt] + (0.55 * a) * q_seeds[
                    min(dt, q_seeds.shape[0] - 1)
                ]

    cell_q = lateral_band_q(
        path_q,
        env,
        W=W,
        lat_rad=float(lat_rad),
        obstacles=obs,
        gap_center=gap,
        detect_field=detect_field,
        cheap=cheap,
        cheap_strict=cheap_strict,
    )
    cell_ee = cell_ee_from_q(env, cell_q)
    if q_seeds.shape[0] > 0:
        cell_q, cell_ee, path_q = inject_opening_seeds_into_band(
            cell_q, cell_ee, path_q, q_seeds, env
        )

    cell_m = np.zeros((T, W, int(normalizer.dof)), dtype=np.float64)
    for t in range(T):
        for w in range(W):
            # Interior samples: clamp manifold into [0,1]. Ends use exact qs/qg below.
            if t == 0 or t == T - 1:
                cell_m[t, w] = normalizer.to_manifold(cell_q[t, w])
            else:
                cell_m[t, w] = normalizer.to_manifold_clamped(cell_q[t, w])
    # Exact start/goal anchors (do not limit-project — that moves EE off the SG marks).
    m_s = normalizer.to_manifold(qs)
    m_g = normalizer.to_manifold(qg)
    mid = W // 2
    cell_m[0, mid] = m_s
    cell_m[-1, mid] = m_g
    cell_q[0, mid] = qs
    cell_q[-1, mid] = qg
    path_q[0], path_q[-1] = qs, qg
    cell_ee[0, mid] = _fk_ee(env, qs)
    cell_ee[-1, mid] = _fk_ee(env, qg)
    # Refresh manifold coords after seed injection / endpoint pin.
    for t in range(T):
        for w in range(W):
            if t == 0 or t == T - 1:
                cell_m[t, w] = normalizer.to_manifold(cell_q[t, w])
            else:
                cell_m[t, w] = normalizer.to_manifold_clamped(cell_q[t, w])
    cell_m[0, mid] = m_s
    cell_m[-1, mid] = m_g

    chart_reg = color_band_whole_arm(
        cell_q, cell_ee, detect_field, env, obstacles=obstacles, cheap=cheap
    )

    zgeo = detect_field.z_geometry or extract_opening_z_geometry(detect_field)
    mid_ee = cell_ee[T // 2, mid]
    mid_lab = int(chart_reg["chart_labels"][T // 2, mid])
    meta = {
        "correspondence": "morph_sweep_b",
        "labeling": "whole_arm+distal_or_ee_opening_geo",
        "lateral": "workspace_slit",
        "plan_space": "joint_poisson_7d",
        "opening_seed_n": int(seed_pack.get("n_seeds", 0)),
        "opening_seed_evals": int(seed_pack.get("n_evals", 0)),
        "opening_q_seeds": (
            np.asarray(seed_pack["q_seeds"], dtype=np.float64).tolist()
            if int(seed_pack.get("n_seeds", 0)) > 0
            else []
        ),
        "opening_ee_seeds": (
            np.asarray(seed_pack["ee_seeds"], dtype=np.float64).tolist()
            if int(seed_pack.get("n_seeds", 0)) > 0
            else []
        ),
        "gap_anchor": gap.tolist(),
        "mid_ee": mid_ee.tolist(),
        "mid_label": {LABEL_SOLID: "solid", LABEL_OPENING: "opening", LABEL_FREE: "free"}.get(
            mid_lab, str(mid_lab)
        ),
        "n_opening_bins": int(np.sum(chart_reg["opening_mask"])),
        "n_solid_bins": int(np.sum(chart_reg["solid_mask"])),
        "n_arm_collision_bins": int(chart_reg.get("n_arm_collision_bins", 0)),
        "T": T,
        "W": W,
        "lat_rad": float(lat_rad),
        "cheap": bool(cheap),
        "cheap_strict": bool(cheap_strict),
        "opening_seed_tries": int(seed_tries),
        "spine_mode": spine_mode,
        "z_geometry": None if zgeo is None else zgeo.as_dict(),
    }
    return {
        "cell_m": cell_m,
        "cell_q": cell_q,
        "cell_ee": cell_ee,
        "path_q": path_q,
        "chart_reg": chart_reg,
        "meta": meta,
    }


# ---------------------------------------------------------------------------
# Base-pose feedback when scheme B finds the slit too narrow for whole-arm
# ---------------------------------------------------------------------------


def morph_slit_too_narrow(chart_reg: Dict[str, Any], meta: Dict[str, Any]) -> bool:
    """True if morph band has no usable through-gap 「无」 for whole-arm planning."""
    n_open = int(meta.get("n_opening_bins", chart_reg.get("n_opening_bins", 0)) or 0)
    n_solid = int(meta.get("n_solid_bins", chart_reg.get("n_solid_bins", 0)) or 0)
    n_arm = int(meta.get("n_arm_collision_bins", chart_reg.get("n_arm_collision_bins", 0)) or 0)
    labs = np.asarray(chart_reg.get("chart_labels"), dtype=np.int8)
    if labs.ndim != 2:
        return n_open == 0
    T, W = labs.shape
    mid = W // 2
    t0, t1 = max(1, T // 3), max(2, (2 * T) // 3)
    spine_mid = labs[t0:t1, mid]
    mid_empty = int(np.sum((spine_mid == LABEL_FREE) | (spine_mid == LABEL_OPENING)))
    mid_open = int(np.sum(spine_mid == LABEL_OPENING))
    # Blocked: no Opening on band, or mid-spine has no empty bins.
    if n_open == 0:
        return True
    if mid_empty == 0:
        return True
    if mid_open == 0 and n_arm >= int(0.55 * T * W):
        return True
    if n_solid >= int(0.75 * T * W):
        return True
    return False


def _read_base_pose(env) -> np.ndarray:
    pos, orn = env.p.getBasePositionAndOrientation(env.robot)
    yaw = float(env.p.getEulerFromQuaternion(orn)[2])
    return np.array([float(pos[0]), float(pos[1]), yaw], dtype=np.float64)


def propose_one_base_setback(
    base_xy_yaw: np.ndarray,
    gap_center: np.ndarray,
    *,
    standoff_m: float = 0.28,
) -> np.ndarray:
    """
    One geometric micro-adjust from B's 'slit too narrow' signal.

    Stay on the same side of the door as the current park; pull back to a
    shoulder-clear standoff; keep Y on the slit center.

    Keep the original yaw — a hard re-yaw re-frames the whole arm and makes
    cheap EE-anchor reseed fail, which then poisons the full joint trajectory
    (looks like the base nudge was weighted onto every waypoint).
    """
    b0 = np.asarray(base_xy_yaw, dtype=np.float64).reshape(-1)
    gap = np.asarray(gap_center, dtype=np.float64).reshape(-1)[:3]
    d = b0[:2] - gap[:2]
    n = float(np.linalg.norm(d))
    if n < 1e-6:
        dir_xy = np.array([1.0, 0.0], dtype=np.float64)
    else:
        dir_xy = d / n
    # Align laterally to slit center; only the radial standoff changes.
    xy = gap[:2] + dir_xy * float(standoff_m)
    yaw = float(b0[2])  # preserve park heading
    return np.array([float(xy[0]), float(xy[1]), yaw], dtype=np.float64)


def _reseed_sg_pair_cheap(
    env,
    ee_s_world: np.ndarray,
    ee_g_world: np.ndarray,
    q_hint_s: np.ndarray,
    q_hint_g: np.ndarray,
    *,
    seed: int = 15,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """After one base nudge: keep world EE anchors, light free search (no visible-pair hammer)."""

    def _recover(q_hint: np.ndarray, ee_tgt: np.ndarray, s: int) -> np.ndarray:
        q0 = _clip_q(env, q_hint)
        q = _search_free_near_ee(q0, ee_tgt, env, [], seed=s, tries=40)
        # Only accept limit projection if world EE stays near the anchor.
        qc = _clip_q(env, q)
        if float(np.linalg.norm(_fk_ee(env, qc) - ee_tgt)) <= 0.06:
            return qc
        return q

    qs = _recover(q_hint_s, ee_s_world, seed)
    qg = _recover(q_hint_g, ee_g_world, seed + 1)
    return qs, qg, {"reseed": "ee_anchor"}


def build_morph_sweep_chart_with_base_feedback(
    q_start: np.ndarray,
    q_goal: np.ndarray,
    env,
    normalizer: JointNormalizer,
    detect_field: OpeningDetectField,
    *,
    T: Optional[int] = None,
    W: int = 11,
    lat_rad: float = 0.14,
    seed: int = 15,
    obstacles: Optional[Sequence] = None,
    joint_meta: Optional[Dict[str, Any]] = None,
    max_base_attempts: int = 0,
    standoff_m: float = 0.28,
    preemptive_standoff: bool = False,
    cheap: bool = False,
    cheap_strict: bool = False,
) -> Dict[str, Any]:
    """
    Scheme B + optional base feedback.

    Default: no reactive setback (``max_base_attempts=0``). Prefer
    ``preemptive_standoff=True`` to pull back once *before* the first morph
    (one morph pass), instead of fail→rebuild.

    If the band still shows the slit too narrow and ``max_base_attempts>=1``,
    apply one geometric setback, cheap-reseed start/goal, rebuild morph once.
    """
    qs = np.asarray(q_start, dtype=np.float64).reshape(-1)
    qg = np.asarray(q_goal, dtype=np.float64).reshape(-1)
    obs = list(obstacles) if obstacles is not None else []
    base0 = _read_base_pose(env)
    ee_s0 = _fk_ee(env, qs)
    ee_g0 = _fk_ee(env, qg)
    gap = _gap_anchor(env, detect_field)

    applied = False
    park = base0.copy()
    preemptive_meta: Dict[str, Any] = {"reseed": "unchanged"}
    qs0, qg0 = qs.copy(), qg.copy()

    # Optional: one standoff *before* morph so we never pay a double rebuild.
    # Use seed+1 for the subsequent morph — same RNG as the successful reactive rebuild.
    morph_seed = int(seed)
    if bool(preemptive_standoff):
        park_try = propose_one_base_setback(base0, gap, standoff_m=float(standoff_m))
        if float(np.linalg.norm(park_try[:2] - base0[:2])) >= 0.04:
            env.set_base_pose(float(park_try[0]), float(park_try[1]), float(park_try[2]))
            clear_morph_query_caches(env)
            qs_try, qg_try, rmeta = _reseed_sg_pair_cheap(
                env, ee_s0, ee_g0, qs0, qg0, seed=seed + 17,
            )
            ds = float(np.linalg.norm(_fk_ee(env, qs_try) - ee_s0))
            dg = float(np.linalg.norm(_fk_ee(env, qg_try) - ee_g0))
            if max(ds, dg) > 0.10:
                env.set_base_pose(float(base0[0]), float(base0[1]), float(base0[2]))
                clear_morph_query_caches(env)
                qs, qg = qs0, qg0
                park = base0.copy()
                preemptive_meta = {
                    **rmeta,
                    "reseed": "preemptive_rejected_ee_drift",
                    "ee_err_s": ds,
                    "ee_err_g": dg,
                }
            else:
                qs, qg = qs_try, qg_try
                park = park_try
                applied = True
                morph_seed = int(seed) + 1
                preemptive_meta = {
                    **rmeta,
                    "reseed": "preemptive_standoff",
                    "ee_err_s": ds,
                    "ee_err_g": dg,
                    "standoff_m": float(standoff_m),
                }

    # Pass 0: morph at (possibly adjusted) base.
    morph = build_morph_sweep_chart(
        qs,
        qg,
        env,
        normalizer,
        detect_field,
        T=T,
        W=W,
        lat_rad=lat_rad,
        seed=morph_seed,
        obstacles=obs,
        cheap=cheap,
        cheap_strict=cheap_strict,
        joint_meta=joint_meta,
    )
    reg = morph["chart_reg"]
    meta = dict(morph["meta"])
    blocked = morph_slit_too_narrow(reg, meta)
    log: List[Dict[str, Any]] = [
        {
            "attempt": 0,
            "base_xy_yaw": park.tolist(),
            "blocked": bool(blocked),
            "n_opening_bins": int(meta.get("n_opening_bins", 0)),
            "n_solid_bins": int(meta.get("n_solid_bins", 0)),
            "n_arm_collision_bins": int(meta.get("n_arm_collision_bins", 0)),
            "mid_label": meta.get("mid_label"),
            "reseed": preemptive_meta,
            "preemptive_standoff": bool(preemptive_standoff),
        }
    ]

    # Reactive setback (off by default): at most one geometric nudge + rebuild.
    morph0 = morph
    if blocked and int(max_base_attempts) >= 1 and not applied:
        park_try = propose_one_base_setback(base0, gap, standoff_m=float(standoff_m))
        if float(np.linalg.norm(park_try[:2] - base0[:2])) >= 0.04:
            env.set_base_pose(float(park_try[0]), float(park_try[1]), float(park_try[2]))
            clear_morph_query_caches(env)
            qs_try, qg_try, rmeta = _reseed_sg_pair_cheap(
                env, ee_s0, ee_g0, qs0, qg0, seed=seed + 17,
            )
            ds = float(np.linalg.norm(_fk_ee(env, qs_try) - ee_s0))
            dg = float(np.linalg.norm(_fk_ee(env, qg_try) - ee_g0))
            # If world EE anchors are lost, revert — do not poison the full orbit.
            if max(ds, dg) > 0.10:
                env.set_base_pose(float(base0[0]), float(base0[1]), float(base0[2]))
                clear_morph_query_caches(env)
                qs, qg = qs0, qg0
                morph = morph0
                reg = morph["chart_reg"]
                meta = dict(morph["meta"])
                blocked = morph_slit_too_narrow(reg, meta)
                park = base0.copy()
                applied = False
                log.append(
                    {
                        "attempt": 1,
                        "base_xy_yaw": park_try.tolist(),
                        "blocked": True,
                        "reverted": True,
                        "ee_err_s": ds,
                        "ee_err_g": dg,
                        "reseed": {**rmeta, "reseed": "rejected_ee_drift"},
                        "standoff_m": float(standoff_m),
                    }
                )
            else:
                qs, qg = qs_try, qg_try
                morph = build_morph_sweep_chart(
                    qs,
                    qg,
                    env,
                    normalizer,
                    detect_field,
                    T=T,
                    W=W,
                    lat_rad=lat_rad,
                    seed=seed + 1,
                    obstacles=obs,
                    cheap=cheap,
                    cheap_strict=cheap_strict,
                    joint_meta=joint_meta,
                )
                reg = morph["chart_reg"]
                meta = dict(morph["meta"])
                blocked = morph_slit_too_narrow(reg, meta)
                park = park_try
                applied = True
                log.append(
                    {
                        "attempt": 1,
                        "base_xy_yaw": park.tolist(),
                        "blocked": bool(blocked),
                        "reverted": False,
                        "ee_err_s": ds,
                        "ee_err_g": dg,
                        "n_opening_bins": int(meta.get("n_opening_bins", 0)),
                        "n_solid_bins": int(meta.get("n_solid_bins", 0)),
                        "n_arm_collision_bins": int(meta.get("n_arm_collision_bins", 0)),
                        "mid_label": meta.get("mid_label"),
                        "reseed": rmeta,
                        "standoff_m": float(standoff_m),
                    }
                )

    meta = {
        **meta,
        "base_feedback_applied": bool(applied),
        "base_feedback_attempt": 1 if applied else 0,
        "base_xy_yaw": park.tolist(),
        "slit_blocked": bool(blocked),
        "preemptive_standoff": bool(preemptive_standoff),
        "cheap": bool(cheap),
        "cheap_strict": bool(cheap_strict),
        "max_base_attempts": int(max_base_attempts),
    }
    out = {
        **morph,
        "q_start": qs.copy(),
        "q_goal": qg.copy(),
        "base_xy_yaw": park.copy(),
        "base_feedback_log": log,
        "base_feedback_applied": bool(applied),
        "meta": meta,
    }
    env.set_base_pose(float(park[0]), float(park[1]), float(park[2]))
    try:
        env._morph_base_feedback = log  # type: ignore[attr-defined]
        env._morph_sweep_chart = out  # type: ignore[attr-defined]
    except Exception:
        pass
    return out
