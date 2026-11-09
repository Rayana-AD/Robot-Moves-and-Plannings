"""
Nav patch connectivity from depth (global passable + component structure).

Unlike arm morph (random-free-motion chart → manifold → grid), nav uses a **fixed
image patch grid** on the floor plane. GT is deterministic: floor_height_fields +
morphological close — **no start/end sampling** in the primary protocol.

Eval: rgb_only model passable field vs depth-oracle patch passable + component edges.
No path planning, hard_coll, or goal-in-component metrics.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.joint_cspace_connectivity import connectivity_bce_loss
from models.nav_floor_oracle import floor_height_fields
from models.safe_topo_qkv import (
    _expand_topology_features,
    _morphological_close,
    build_connectivity_fields,
    sharpen_obstacle_mask,
)


class GridToLogitField(nn.Module):
    """ViT patch tokens → dense logit field [B, 1, H, W]. Local copy — do not import joint_vit_bridge (PyBullet)."""

    def __init__(self, vit_hidden_dim: int = 768, patch_size: int = 4):
        super().__init__()
        self.patch_size = patch_size
        self.decoder = nn.Sequential(
            nn.Linear(vit_hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, patch_size * patch_size),
        )

    def forward(self, vit_patch_tokens: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, _N, D = vit_patch_tokens.shape
        ps = self.patch_size
        ph, pw = H // ps, W // ps
        patches = vit_patch_tokens[:, 1 : 1 + ph * pw]
        x = self.decoder(patches)
        x = x.view(B, ph, pw, ps, ps).permute(0, 1, 3, 2, 4).contiguous()
        x = x.view(B, 1, ph * ps, pw * ps)
        if x.shape[2] != H or x.shape[3] != W:
            x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
        return x


def traversable_from_depth(
    depth: torch.Tensor,
    *,
    rgb: Optional[torch.Tensor] = None,
    floor_band_start: float = 0.55,
    morph_close_radius: int = 2,
    frame_id: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    """Shared depth-oracle traversable recipe (pixel [B,1,H,W]).

    `rgb` [B,3,H,W] in 0..1 enables NAV_FLOOR_ORACLE=rgbd_pixel fusion in `floor_height_fields`.
    `frame_id` enables per-frame manual corrections from NAV_ORACLE_OVERRIDES
    (models.nav_oracle_overrides); applied after the morphological close.
    """
    height = floor_height_fields(depth, floor_band_start=floor_band_start, rgb=rgb)
    traversable = torch.clamp(
        height["floor_prob"] * (1.0 - height["height_obstacle"]),
        0.0,
        1.0,
    )
    skip_morph = height.get("skip_morph")
    if skip_morph is None or not bool((skip_morph > 0.5).any()):
        sheet = height.get("drift_floor_sheet")
        drift = height.get("plane_drift")
        morph_r = morph_close_radius
        if drift is not None and bool((drift > 0.5).any()):
            morph_r = max(morph_r, 4)
        if sheet is not None and bool((drift > 0.5).any()):
            on_sheet = sheet > 0.5
            traversable = torch.where(on_sheet, torch.maximum(traversable, torch.full_like(traversable, 0.92)), traversable)
        if morph_r > 0:
            traversable = _morphological_close(traversable, radius=morph_r)
    thin = height.get("thin_obstacle")
    if thin is not None:
        traversable = torch.where(thin > 0.5, torch.zeros_like(traversable), traversable)
    if frame_id is not None:
        from models.nav_oracle_overrides import apply_traversable_overrides

        traversable = apply_traversable_overrides(traversable, frame_id)
    return {
        "traversable": traversable,
        "floor_prob": height["floor_prob"],
        "height_obstacle": height["height_obstacle"],
    }


def _crop_to_patch_grid(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    _, _, H, W = x.shape
    ph, pw = H // patch_size, W // patch_size
    return x[:, :, : ph * patch_size, : pw * patch_size]


def label_connected_components_2d(passable: np.ndarray) -> np.ndarray:
    """4-neighbor connected component ids; 0 = impassable."""
    passable = np.asarray(passable, dtype=bool)
    H, W = passable.shape
    labels = np.zeros((H, W), dtype=np.int32)
    cur = 0
    for r in range(H):
        for c in range(W):
            if not passable[r, c] or labels[r, c]:
                continue
            cur += 1
            stack = [(r, c)]
            labels[r, c] = cur
            while stack:
                rr, cc = stack.pop()
                for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W and passable[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = cur
                        stack.append((nr, nc))
    return labels


def build_nav_patch_global_gt(
    depth: torch.Tensor,
    *,
    rgb: Optional[torch.Tensor] = None,
    patch_size: int = 16,
    floor_band_start: float = 0.55,
    morph_close_radius: int = 2,
    frame_id: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    """
    Global patch GT from depth — no start/end, no flood from endpoints.

    Returns patch_passable, patch_component_id, patch_component_size_norm, pixel traversable.
    """
    if depth.dim() != 4 or depth.shape[1] != 1:
        raise ValueError("depth must be [B, 1, H, W]")

    fields = traversable_from_depth(
        depth,
        rgb=rgb,
        floor_band_start=floor_band_start,
        morph_close_radius=morph_close_radius,
        frame_id=frame_id,
    )
    traversable = _crop_to_patch_grid(fields["traversable"], patch_size)
    floor_prob = _crop_to_patch_grid(fields["floor_prob"], patch_size)
    height_obstacle = _crop_to_patch_grid(fields["height_obstacle"], patch_size)

    B, _, H, W = traversable.shape
    ph, pw = H // patch_size, W // patch_size

    patch_passable_list = []
    patch_comp_list = []
    patch_size_norm_list = []
    num_components_list = []

    for b in range(B):
        passable_px = (traversable[b, 0].detach().cpu().numpy() > 0.5)
        passable_patch = (
            F.avg_pool2d(traversable[b : b + 1], kernel_size=patch_size, stride=patch_size) > 0.5
        )
        comp_px = label_connected_components_2d(passable_px)
        comp_patch = np.zeros((ph, pw), dtype=np.int32)
        size_norm = np.zeros((ph, pw), dtype=np.float32)
        for pi in range(ph):
            for pj in range(pw):
                block = comp_px[pi * patch_size : (pi + 1) * patch_size, pj * patch_size : (pj + 1) * patch_size]
                block_pass = passable_px[pi * patch_size : (pi + 1) * patch_size, pj * patch_size : (pj + 1) * patch_size]
                if not block_pass.any():
                    comp_patch[pi, pj] = 0
                    continue
                ids, counts = np.unique(block[block > 0], return_counts=True)
                cid = int(ids[counts.argmax()])
                comp_patch[pi, pj] = cid
                size_norm[pi, pj] = float((comp_px == cid).sum()) / max(1, passable_px.sum())
        patch_passable_list.append(passable_patch)
        patch_comp_list.append(torch.from_numpy(comp_patch).unsqueeze(0).unsqueeze(0).float())
        patch_size_norm_list.append(torch.from_numpy(size_norm).unsqueeze(0).unsqueeze(0).float())
        num_components_list.append(int(comp_px.max()))

    patch_passable = torch.cat(patch_passable_list, dim=0).float()
    patch_component = torch.cat(patch_comp_list, dim=0)
    patch_component_size_norm = torch.cat(patch_size_norm_list, dim=0)

    return {
        "traversable": traversable,
        "floor_prob": floor_prob,
        "height_obstacle": height_obstacle,
        "patch_passable": patch_passable,
        "patch_component": patch_component,
        "patch_component_size_norm": patch_component_size_norm,
        "num_components": num_components_list,
        "patch_hw": (ph, pw),
    }


def build_patch_global_topology_tokens(
    gt: Dict[str, torch.Tensor],
    *,
    patch_size: int = 16,
    topo_dim: int = 32,
    include_cls: bool = True,
) -> torch.Tensor:
    """
    Global topology tokens (no start/end channels).

    Base 8: passable, component_size_norm, row, col, floor, height, boundary_h, boundary_v.
    """
    patch_pass = gt["patch_passable"]
    size_norm = gt["patch_component_size_norm"]
    B, _, ph, pw = patch_pass.shape
    device, dtype = patch_pass.device, patch_pass.dtype

    comp = gt["patch_component"][0, 0].detach().cpu().numpy().astype(np.int32)
    pass_np = patch_pass[0, 0].detach().cpu().numpy() > 0.5
    boundary_h = np.zeros((ph, pw), dtype=np.float32)
    boundary_v = np.zeros((ph, pw), dtype=np.float32)
    for i in range(ph):
        for j in range(pw):
            if not pass_np[i, j]:
                continue
            if j + 1 < pw and pass_np[i, j + 1] and comp[i, j] != comp[i, j + 1]:
                boundary_h[i, j] = 1.0
                boundary_h[i, j + 1] = 1.0
            if i + 1 < ph and pass_np[i + 1, j] and comp[i, j] != comp[i + 1, j]:
                boundary_v[i, j] = 1.0
                boundary_v[i + 1, j] = 1.0
    boundary_h = torch.from_numpy(boundary_h).to(device=device, dtype=dtype).view(1, 1, ph, pw).expand(B, 1, ph, pw)
    boundary_v = torch.from_numpy(boundary_v).to(device=device, dtype=dtype).view(1, 1, ph, pw).expand(B, 1, ph, pw)

    row = torch.linspace(-1.0, 1.0, ph, device=device, dtype=dtype).view(1, 1, ph, 1).expand(B, 1, ph, pw)
    col = torch.linspace(-1.0, 1.0, pw, device=device, dtype=dtype).view(1, 1, 1, pw).expand(B, 1, ph, pw)
    floor_p = F.avg_pool2d(gt["floor_prob"], kernel_size=patch_size, stride=patch_size)
    height_o = F.avg_pool2d(gt["height_obstacle"], kernel_size=patch_size, stride=patch_size)

    topo_map = torch.cat(
        [
            patch_pass * 2.0 - 1.0,
            size_norm * 2.0 - 1.0,
            row,
            col,
            floor_p * 2.0 - 1.0,
            height_o * 2.0 - 1.0,
            boundary_h * 2.0 - 1.0,
            boundary_v * 2.0 - 1.0,
        ],
        dim=1,
    )
    tokens = topo_map.flatten(2).transpose(1, 2).contiguous()
    tokens = _expand_topology_features(tokens, topo_dim)
    if include_cls:
        cls = torch.zeros(B, 1, topo_dim, device=device, dtype=dtype)
        tokens = torch.cat([cls, tokens], dim=1)
    return tokens


def model_patch_passable_from_obstacle(
    obstacle: torch.Tensor,
    *,
    patch_size: int = 16,
    threshold: float = 0.58,
    temperature: float = 20.0,
) -> torch.Tensor:
    """ViT obstacle [B,1,H,W] → patch passable [B,1,ph,pw]."""
    gated = sharpen_obstacle_mask(obstacle, threshold=threshold, temperature=temperature)
    traversable = 1.0 - gated
    traversable = _crop_to_patch_grid(traversable, patch_size)
    return (F.avg_pool2d(traversable, kernel_size=patch_size, stride=patch_size) > 0.5).float()


def compute_global_patch_metrics(
    oracle_passable: np.ndarray,
    pred_passable: np.ndarray,
) -> Dict[str, float]:
    """
    Global connectivity metrics (no start/end).

    Primary: patch_passable_iou, component_edge_agreement.
    """
    oracle = np.asarray(oracle_passable, dtype=bool)
    pred = np.asarray(pred_passable, dtype=bool)
    if oracle.shape != pred.shape:
        raise ValueError(f"shape mismatch {oracle.shape} vs {pred.shape}")

    inter = (oracle & pred).sum()
    union = (oracle | pred).sum()
    iou = float(inter / union) if union else 1.0
    acc = float((oracle == pred).mean())
    prec = float(inter / pred.sum()) if pred.sum() else 1.0
    rec = float(inter / oracle.sum()) if oracle.sum() else 1.0
    f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0

    oracle_comp = label_connected_components_2d(oracle)
    pred_comp = label_connected_components_2d(pred)
    n_oracle = int(oracle_comp.max())
    n_pred = int(pred_comp.max())

    agree = 0
    total = 0
    ph, pw = oracle.shape
    for i in range(ph):
        for j in range(pw):
            if j + 1 < pw and oracle[i, j] and oracle[i, j + 1]:
                total += 1
                o_same = oracle_comp[i, j] == oracle_comp[i, j + 1]
                p_same = pred_comp[i, j] == pred_comp[i, j + 1] if pred[i, j] and pred[i, j + 1] else False
                agree += int(o_same == p_same)
            if i + 1 < ph and oracle[i, j] and oracle[i + 1, j]:
                total += 1
                o_same = oracle_comp[i, j] == oracle_comp[i + 1, j]
                p_same = pred_comp[i, j] == pred_comp[i + 1, j] if pred[i, j] and pred[i + 1, j] else False
                agree += int(o_same == p_same)

    edge_agree = float(agree / total) if total else 1.0
    return {
        "patch_passable_iou": iou,
        "patch_passable_acc": acc,
        "patch_passable_f1": f1,
        "component_edge_agreement": edge_agree,
        "num_components_oracle": float(n_oracle),
        "num_components_pred": float(n_pred),
        "num_components_abs_delta": float(abs(n_oracle - n_pred)),
        "oracle_passable_rate": float(oracle.mean()),
        "pred_passable_rate": float(pred.mean()),
    }


def build_floor_inject_tokens(
    depth: torch.Tensor,
    *,
    patch_size: int = 16,
    topo_dim: int = 32,
    include_cls: bool = True,
) -> torch.Tensor:
    """Deprecated local-geometry inject (floor_prob + height_obstacle). Use build_passable_inject_tokens."""
    height = floor_height_fields(depth)
    floor_p = _crop_to_patch_grid(height["floor_prob"], patch_size)
    height_o = _crop_to_patch_grid(height["height_obstacle"], patch_size)
    floor_p = F.avg_pool2d(floor_p, kernel_size=patch_size, stride=patch_size)
    height_o = F.avg_pool2d(height_o, kernel_size=patch_size, stride=patch_size)
    topo_map = torch.cat([floor_p * 2.0 - 1.0, height_o * 2.0 - 1.0], dim=1)
    tokens = topo_map.flatten(2).transpose(1, 2).contiguous()
    tokens = _expand_topology_features(tokens, topo_dim)
    if include_cls:
        cls = torch.zeros(tokens.shape[0], 1, topo_dim, device=tokens.device, dtype=tokens.dtype)
        tokens = torch.cat([cls, tokens], dim=1)
    return tokens


def build_passable_inject_tokens(
    patch_passable: torch.Tensor,
    *,
    topo_dim: int = 32,
    include_cls: bool = True,
) -> torch.Tensor:
    """
    Runtime inject: full-image connectivity (global patch_passable), no S/G intersection.

    Input is the plane-oracle connected floor on the patch grid — the same field used
    as the train label. Do not pass flood(S,G) intersection here.
    """
    if patch_passable.dim() != 4 or patch_passable.shape[1] != 1:
        raise ValueError("patch_passable must be [B, 1, ph, pw]")
    tokens = (patch_passable * 2.0 - 1.0).flatten(2).transpose(1, 2).contiguous()
    tokens = _expand_topology_features(tokens, topo_dim)
    if include_cls:
        cls = torch.zeros(tokens.shape[0], 1, topo_dim, device=tokens.device, dtype=tokens.dtype)
        tokens = torch.cat([cls, tokens], dim=1)
    return tokens


def flood_patch_intersection(
    passable: torch.Tensor,
    start_rc: Tuple[int, int],
    end_rc: Tuple[int, int],
    *,
    patch_size: int = 16,
    connectivity_steps: int = 400,
    protect_radius: int = 0,
) -> Dict[str, torch.Tensor]:
    """Flood a (predicted or oracle) passable field from S/G → pixel + patch intersection."""
    if passable.dim() != 4:
        raise ValueError("passable must be [B,1,H,W]")
    conn = build_connectivity_fields(
        passable,
        start_rc,
        end=end_rc,
        steps=connectivity_steps,
        protect_radius=protect_radius,
    )
    inter = _crop_to_patch_grid(conn["connected_intersection"], patch_size)
    patch_inter = (F.avg_pool2d(inter, kernel_size=patch_size, stride=patch_size) > 0.5).float()
    er, ec = pixel_to_patch_rc(end_rc, patch_size)
    _, _, ph, pw = patch_inter.shape
    er = max(0, min(ph - 1, er))
    ec = max(0, min(pw - 1, ec))
    return {
        "connected_intersection": inter,
        "patch_intersection": patch_inter,
        "goal_in_patch_intersection": patch_inter[:, :, er, ec] > 0.5,
        "patch_hw": (ph, pw),
    }


def patch_binary_metrics(pred: np.ndarray, oracle: np.ndarray) -> Dict[str, float]:
    """IoU / F1 / acc on a boolean patch (or pixel) grid."""
    p = np.asarray(pred, dtype=bool)
    o = np.asarray(oracle, dtype=bool)
    if p.shape != o.shape:
        raise ValueError(f"shape mismatch {p.shape} vs {o.shape}")
    inter = int((p & o).sum())
    union = int((p | o).sum())
    pred_pos = int(p.sum())
    oracle_pos = int(o.sum())
    iou = float(inter / union) if union else 1.0
    prec = float(inter / pred_pos) if pred_pos else 1.0
    rec = float(inter / oracle_pos) if oracle_pos else 1.0
    f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return {
        "iou": iou,
        "f1": f1,
        "acc": float((p == o).mean()),
        "pred_pos": float(pred_pos),
        "oracle_pos": float(oracle_pos),
    }


def pixel_to_patch_rc(point: Tuple[int, int], patch_size: int = 16) -> Tuple[int, int]:
    row, col = int(point[0]), int(point[1])
    return row // patch_size, col // patch_size


def build_nav_patch_connectivity_gt(
    depth: torch.Tensor,
    start_rc: Tuple[int, int],
    end_rc: Tuple[int, int],
    *,
    rgb: Optional[torch.Tensor] = None,
    patch_size: int = 16,
    connectivity_steps: int = 100,
    protect_radius: int = 1,
    floor_band_start: float = 0.55,
    morph_close_radius: int = 2,
    frame_id: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    """
    Depth-oracle patch connectivity (same traversable recipe as eval depth oracle).

    Returns pixel fields plus patch-pooled:
      patch_passable, patch_start_connected, patch_end_connected,
      patch_intersection, goal_in_patch_intersection (scalar bool tensor).
    """
    if depth.dim() != 4 or depth.shape[1] != 1:
        raise ValueError("depth must be [B, 1, H, W]")

    fields = traversable_from_depth(
        depth,
        rgb=rgb,
        floor_band_start=floor_band_start,
        morph_close_radius=morph_close_radius,
        frame_id=frame_id,
    )
    traversable = fields["traversable"]

    conn = build_connectivity_fields(
        traversable,
        start_rc,
        end=end_rc,
        steps=connectivity_steps,
        protect_radius=protect_radius,
    )

    _, _, H, W = depth.shape
    ph, pw = H // patch_size, W // patch_size
    traversable = traversable[:, :, : ph * patch_size, : pw * patch_size]
    for key in ("start_connected", "end_connected", "connected_intersection", "passable"):
        conn[key] = conn[key][:, :, : ph * patch_size, : pw * patch_size]

    def pool(x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(x, kernel_size=patch_size, stride=patch_size)

    patch_passable = (pool(traversable) > 0.5).float()
    patch_start = (pool(conn["start_connected"]) > 0.5).float()
    patch_end = (pool(conn["end_connected"]) > 0.5).float()
    patch_inter = (pool(conn["connected_intersection"]) > 0.5).float()

    er, ec = pixel_to_patch_rc(end_rc, patch_size)
    er = max(0, min(ph - 1, er))
    ec = max(0, min(pw - 1, ec))
    goal_in_patch = patch_inter[:, :, er, ec] > 0.5

    return {
        "traversable": traversable,
        "start_connected": conn["start_connected"],
        "end_connected": conn["end_connected"],
        "connected_intersection": conn["connected_intersection"],
        "passable": conn["passable"],
        "floor_prob": fields["floor_prob"],
        "height_obstacle": fields["height_obstacle"],
        "patch_passable": patch_passable,
        "patch_start_connected": patch_start,
        "patch_end_connected": patch_end,
        "patch_intersection": patch_inter,
        "goal_in_patch_intersection": goal_in_patch,
        "patch_hw": (ph, pw),
    }


def build_patch_flood_topology_tokens(
    gt: Dict[str, torch.Tensor],
    *,
    patch_size: int = 16,
    topo_dim: int = 32,
    include_cls: bool = True,
) -> torch.Tensor:
    """
    Clean deterministic topology tokens from flood GT (not hand xy/grad heuristics).

    Base 8 channels per patch:
      passable, start_conn, end_conn, intersection, row_norm, col_norm,
      pooled floor_prob, pooled height_obstacle.
    """
    patch_pass = gt["patch_passable"]
    patch_start = gt["patch_start_connected"]
    patch_end = gt["patch_end_connected"]
    patch_inter = gt["patch_intersection"]
    B, _, ph, pw = patch_pass.shape
    device, dtype = patch_pass.device, patch_pass.dtype

    row = torch.linspace(-1.0, 1.0, ph, device=device, dtype=dtype).view(1, 1, ph, 1).expand(B, 1, ph, pw)
    col = torch.linspace(-1.0, 1.0, pw, device=device, dtype=dtype).view(1, 1, 1, pw).expand(B, 1, ph, pw)

    floor_p = F.avg_pool2d(gt["floor_prob"], kernel_size=patch_size, stride=patch_size)
    height_o = F.avg_pool2d(gt["height_obstacle"], kernel_size=patch_size, stride=patch_size)

    topo_map = torch.cat(
        [
            patch_pass * 2.0 - 1.0,
            patch_start * 2.0 - 1.0,
            patch_end * 2.0 - 1.0,
            patch_inter * 2.0 - 1.0,
            row,
            col,
            floor_p * 2.0 - 1.0,
            height_o * 2.0 - 1.0,
        ],
        dim=1,
    )
    tokens = topo_map.flatten(2).transpose(1, 2).contiguous()
    tokens = _expand_topology_features(tokens, topo_dim)

    if include_cls:
        cls = torch.zeros(B, 1, topo_dim, device=device, dtype=dtype)
        tokens = torch.cat([cls, tokens], dim=1)
    return tokens


class NavPatchConnectivityHead(nn.Module):
    """ViT patch tokens → patch-grid logits [B, 1, ph, pw] (passable; optional intersection)."""

    def __init__(self, vit_hidden_dim: int = 768, patch_size: int = 16, *, dual_task: bool = True):
        super().__init__()
        self.patch_size = patch_size
        self.dual_task = dual_task
        self.passable_decoder = GridToLogitField(vit_hidden_dim, patch_size)
        self.intersection_decoder = (
            GridToLogitField(vit_hidden_dim, patch_size) if dual_task else None
        )

    def forward(
        self, vit_feat: torch.Tensor, img_h: int, img_w: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        ph = img_h // self.patch_size
        pw = img_w // self.patch_size
        h, w = ph * self.patch_size, pw * self.patch_size
        passable = self.passable_decoder(vit_feat, h, w)
        if self.intersection_decoder is None:
            return passable, None
        inter = self.intersection_decoder(vit_feat, h, w)
        return passable, inter


def patch_passable_from_logits(
    logits: torch.Tensor,
    *,
    patch_size: int = 16,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Logits [B,1,H,W] → binary patch passable [B,1,ph,pw]. Pools pixel logits to patch grid."""
    prob = torch.sigmoid(logits)
    _, _, H, W = prob.shape
    if H > patch_size * 2 and W > patch_size * 2:
        prob = F.avg_pool2d(prob, kernel_size=patch_size, stride=patch_size)
    return (prob > threshold).float()


def nav_route_b_dual_loss(
    passable_logits: torch.Tensor,
    inter_logits: torch.Tensor,
    gt_passable: torch.Tensor,
    gt_intersection: torch.Tensor,
    *,
    inter_weight: float = 0.5,
    pos_weight: float = 3.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Legacy appendix: passable + flood-intersection (requires start/end GT). Route B global uses passable only."""
    loss_p, stats_p = nav_patch_connectivity_loss(passable_logits, gt_passable, pos_weight=pos_weight)
    loss_i, stats_i = nav_patch_connectivity_loss(inter_logits, gt_intersection, pos_weight=pos_weight)
    total = loss_p + inter_weight * loss_i
    return total, {
        "loss_total": float(total.detach().cpu()),
        "loss_passable": stats_p["loss_conn"],
        "loss_intersection": stats_i["loss_conn"],
        "acc_passable": stats_p["conn_acc"],
        "acc_intersection": stats_i["conn_acc"],
    }


def nav_patch_connectivity_loss(
    pred_logits: torch.Tensor,
    gt_intersection: torch.Tensor,
    *,
    pos_weight: float = 3.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    if pred_logits.shape[-2:] != gt_intersection.shape[-2:]:
        pred_logits = F.interpolate(
            pred_logits, size=gt_intersection.shape[-2:], mode="bilinear", align_corners=False
        )
    loss = connectivity_bce_loss(pred_logits, gt_intersection, pos_weight=pos_weight)
    with torch.no_grad():
        prob = torch.sigmoid(pred_logits)
        acc = ((prob > 0.5) == (gt_intersection > 0.5)).float().mean()
        pos_rate = (gt_intersection > 0.5).float().mean()
        pred_pos = (prob > 0.5).float().mean()
    return loss, {
        "loss_conn": float(loss.detach().cpu()),
        "conn_acc": float(acc.detach().cpu()),
        "gt_pos_rate": float(pos_rate.detach().cpu()),
        "pred_pos_rate": float(pred_pos.detach().cpu()),
    }
