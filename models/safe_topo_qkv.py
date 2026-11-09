"""
SafeTopoQKV attention utilities.

The module keeps the HuggingFace ViT interface unchanged. Topology tokens are
set on wrapped attention layers before a ViT forward pass, then the wrapper
replaces the last topo_dim dimensions of each head's Q/K vectors.
"""
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _normalize_channel(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    dims = tuple(range(1, x.dim()))
    x_min = x.amin(dim=dims, keepdim=True)
    x_max = x.amax(dim=dims, keepdim=True)
    return (x - x_min) / (x_max - x_min + eps) * 2.0 - 1.0


def _expand_topology_features(tokens: torch.Tensor, topo_dim: int) -> torch.Tensor:
    """
    Expand the 8 base topology channels without learnable parameters.

    For topo_dim=32 this yields base, signed magnitude, sin phase, and cos phase
    features instead of padding the extra 24 channels with zeros.
    """
    if topo_dim <= tokens.shape[-1]:
        return tokens[..., :topo_dim]

    features = [
        tokens,
        tokens.sign() * tokens.square(),
        torch.sin(math.pi * tokens),
        torch.cos(math.pi * tokens),
    ]
    expanded = torch.cat(features, dim=-1)
    while expanded.shape[-1] < topo_dim:
        expanded = torch.cat([expanded, torch.sin(math.pi * expanded)], dim=-1)
    return expanded[..., :topo_dim]


def build_patch_topology_tokens(
    depth: torch.Tensor,
    patch_size: int = 16,
    topo_dim: int = 8,
    include_cls: bool = True,
    *,
    disable_floor_obstacle: bool = False,
) -> torch.Tensor:
    """
    Build patch-level topology tokens from depth.

    Returns [B, 1 + ph*pw, topo_dim] when include_cls=True. The base channels are:
    x, y, normalized depth gradient dx/dy, cos(theta), sin(theta),
    floor connectivity proxy, obstacle proxy.
    """
    if depth.dim() != 4 or depth.shape[1] != 1:
        raise ValueError("depth must have shape [B, 1, H, W]")

    B, _, H, W = depth.shape
    device = depth.device
    dtype = depth.dtype

    depth_smooth = F.avg_pool2d(depth, kernel_size=3, stride=1, padding=1)
    grad_y = depth_smooth[:, :, 1:, :] - depth_smooth[:, :, :-1, :]
    grad_y = F.pad(grad_y, (0, 0, 0, 1))
    grad_x = depth_smooth[:, :, :, 1:] - depth_smooth[:, :, :, :-1]
    grad_x = F.pad(grad_x, (0, 1, 0, 0))
    grad_mag = torch.sqrt(grad_x.square() + grad_y.square() + 1e-8)

    # A simple floor proxy: close to the median depth in the lower image band.
    lower = depth[:, :, int(H * 0.55) :, :]
    floor_depth = lower.flatten(2).median(dim=-1).values.view(B, 1, 1, 1)
    floor = torch.exp(-torch.abs(depth - floor_depth) / (0.35 * floor_depth.abs().clamp_min(1.0)))
    obstacle = torch.clamp(_normalize_channel(grad_mag) * 0.5 + 0.5, 0.0, 1.0)

    y_coord = torch.linspace(-1, 1, H, device=device, dtype=dtype).view(1, 1, H, 1).expand(B, 1, H, W)
    x_coord = torch.linspace(-1, 1, W, device=device, dtype=dtype).view(1, 1, 1, W).expand(B, 1, H, W)
    norm_grad_x = _normalize_channel(grad_x)
    norm_grad_y = _normalize_channel(grad_y)
    theta = torch.atan2(grad_y, grad_x + 1e-8)

    floor_tok = floor * 2.0 - 1.0
    obstacle_tok = obstacle * 2.0 - 1.0
    if disable_floor_obstacle:
        floor_tok = torch.zeros_like(floor_tok)
        obstacle_tok = torch.zeros_like(obstacle_tok)
    topo_map = torch.cat(
        [
            x_coord,
            y_coord,
            norm_grad_x,
            norm_grad_y,
            torch.cos(theta),
            torch.sin(theta),
            floor_tok,
            obstacle_tok,
        ],
        dim=1,
    )

    ph, pw = H // patch_size, W // patch_size
    if ph == 0 or pw == 0:
        raise ValueError("image is smaller than one ViT patch")

    topo_map = topo_map[:, :, : ph * patch_size, : pw * patch_size]
    pooled = F.avg_pool2d(topo_map, kernel_size=patch_size, stride=patch_size)
    tokens = pooled.flatten(2).transpose(1, 2).contiguous()

    tokens = _expand_topology_features(tokens, topo_dim)

    if include_cls:
        cls = torch.zeros(B, 1, topo_dim, device=device, dtype=dtype)
        tokens = torch.cat([cls, tokens], dim=1)

    return tokens


def sharpen_obstacle_mask(
    obstacle: torch.Tensor,
    threshold: float = 0.58,
    temperature: float = 20.0,
) -> torch.Tensor:
    """Turn ambiguous obstacle probabilities around 0.5 into a sharper gate."""
    return torch.sigmoid((obstacle - threshold) * temperature)


def protect_points(mask: torch.Tensor, points: Iterable[Tuple[int, int]], radius: int = 2) -> torch.Tensor:
    """Force small neighborhoods around start/end to stay traversable."""
    if radius <= 0:
        return mask
    B, _, H, W = mask.shape
    protected = mask.clone()
    yy, xx = torch.meshgrid(
        torch.arange(H, device=mask.device),
        torch.arange(W, device=mask.device),
        indexing="ij",
    )
    disk = None
    for row, col in points:
        rr = int(max(0, min(H - 1, row)))
        cc = int(max(0, min(W - 1, col)))
        point_disk = ((yy - rr).square() + (xx - cc).square()) <= radius * radius
        disk = point_disk if disk is None else (disk | point_disk)
    if disk is not None:
        protected[:, :, disk] = 1.0
    return protected


def _morphological_close(mask: torch.Tensor, radius: int = 2) -> torch.Tensor:
    """Fill small holes/gaps in a soft binary map using max-pool dilation then erosion."""
    if radius <= 0:
        return mask
    kernel = 2 * radius + 1
    dilated = F.max_pool2d(mask, kernel_size=kernel, stride=1, padding=radius)
    eroded = 1.0 - F.max_pool2d(1.0 - dilated, kernel_size=kernel, stride=1, padding=radius)
    return eroded.clamp(0.0, 1.0)


def _grow_connected_component(
    passable: torch.Tensor,
    point: Tuple[int, int],
    steps: int,
) -> torch.Tensor:
    B, _, H, W = passable.shape
    row = int(max(0, min(H - 1, point[0])))
    col = int(max(0, min(W - 1, point[1])))
    connected = torch.zeros(B, 1, H, W, device=passable.device, dtype=passable.dtype)
    connected[:, :, row, col] = 1.0
    for _ in range(max(1, steps)):
        grown = F.max_pool2d(connected, kernel_size=3, stride=1, padding=1)
        connected = torch.maximum(connected, grown * passable)
        connected[:, :, row, col] = 1.0
    return connected


def build_connectivity_fields(
    traversable: torch.Tensor,
    start: Tuple[int, int],
    end: Optional[Tuple[int, int]] = None,
    steps: int = 160,
    protect_radius: int = 2,
) -> Dict[str, torch.Tensor]:
    """
    Build symmetric endpoint connectivity while preserving true component metrics.
    """
    if traversable.dim() != 4:
        raise ValueError("traversable must have shape [B, C, H, W]")
    if traversable.shape[1] != 1:
        traversable = traversable.mean(dim=1, keepdim=True)

    points = [start] if end is None else [start, end]
    traversable = protect_points(traversable, points, radius=protect_radius)
    passable = (traversable > 0.5).to(dtype=traversable.dtype)

    start_connected = _grow_connected_component(passable, start, steps=steps)
    if end is None:
        end_connected = torch.zeros_like(start_connected)
        connected_intersection = torch.zeros_like(start_connected)
        connected = start_connected
    else:
        end_connected = _grow_connected_component(passable, end, steps=steps)
        connected_intersection = start_connected * end_connected
        # Symmetric target handling preserves both endpoint neighborhoods for
        # Poisson tracing; intersection indicates true same-component reachability.
        connected = torch.maximum(start_connected, end_connected)

    poisson_obstacle = 1.0 - connected.clamp(0.0, 1.0)
    return {
        "poisson_obstacle": poisson_obstacle,
        "connected_traversable": connected,
        "start_connected": start_connected,
        "end_connected": end_connected,
        "connected_intersection": connected_intersection,
        "traversable": traversable,
        "passable": passable,
    }


def connected_traversable_from_obstacle(
    obstacle: torch.Tensor,
    start: Tuple[int, int],
    end: Optional[Tuple[int, int]] = None,
    threshold: float = 0.58,
    temperature: float = 20.0,
    steps: int = 160,
    protect_radius: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Backward-compatible wrapper returning Poisson obstacle, connected domain, traversable.
    """
    if obstacle.dim() != 4:
        raise ValueError("obstacle must have shape [B, C, H, W]")
    if obstacle.shape[1] != 1:
        obstacle = obstacle.mean(dim=1, keepdim=True)
    obstacle_gate = sharpen_obstacle_mask(obstacle, threshold=threshold, temperature=temperature)
    fields = build_connectivity_fields(
        1.0 - obstacle_gate,
        start,
        end=end,
        steps=steps,
        protect_radius=protect_radius,
    )
    return fields["poisson_obstacle"], fields["connected_traversable"], fields["traversable"]


# ---------------------------------------------------------------------------
# Nav floor oracle (GT + collision judge) — see models/nav_floor_oracle.py.
# SafeTopo Q/K inject uses build_patch_topology_tokens() below, not the oracle.
# ---------------------------------------------------------------------------
from models.nav_floor_oracle import floor_height_fields  # noqa: F401 re-export



def align_navigation_fields(
    semantic_obstacle: torch.Tensor,
    depth: torch.Tensor,
    start: Tuple[int, int],
    end: Optional[Tuple[int, int]] = None,
    threshold: float = 0.58,
    temperature: float = 20.0,
    connectivity_steps: int = 160,
    protect_radius: int = 2,
    semantic_weight: float = 0.65,
    height_weight: float = 0.35,
) -> Dict[str, torch.Tensor]:
    """
    Stage-1 aligner: combine semantic obstacle, depth floor/height, and connectivity.
    """
    if semantic_obstacle.shape[1] != 1:
        semantic_obstacle = semantic_obstacle.mean(dim=1, keepdim=True)
    height = floor_height_fields(depth)
    semantic_gate = sharpen_obstacle_mask(semantic_obstacle, threshold=threshold, temperature=temperature)
    obstacle_gate = torch.clamp(
        semantic_weight * semantic_gate + height_weight * height["height_obstacle"],
        0.0,
        1.0,
    )
    floor_gate = height["floor_prob"]
    hard_obstacle = torch.maximum(
        (obstacle_gate > 0.82).to(dtype=obstacle_gate.dtype),
        (height["height_obstacle"] > 0.88).to(dtype=obstacle_gate.dtype),
    )
    hard_free = protect_points(1.0 - hard_obstacle, [start] if end is None else [start, end], radius=protect_radius)
    hard_obstacle = 1.0 - hard_free
    unknown_cost = (1.0 - floor_gate).clamp(0.0, 1.0)
    soft_cost = torch.clamp(0.55 * obstacle_gate + 0.30 * height["height_obstacle"] + 0.15 * unknown_cost, 0.0, 1.0)
    soft_traversable = torch.clamp(floor_gate * (1.0 - 0.65 * obstacle_gate) * (1.0 - 0.35 * height["height_obstacle"]), 0.0, 1.0)

    from models.unified_aligner import UnifiedAlignerConfig, core_align_fields

    aligner_cfg = UnifiedAlignerConfig(
        threshold=threshold,
        temperature=temperature,
        connectivity_steps=connectivity_steps,
        protect_radius=protect_radius,
        semantic_weight=semantic_weight,
        height_weight=height_weight,
    )
    conn = core_align_fields(
        soft_cost,
        hard_obstacle,
        soft_traversable,
        start,
        end=end,
        unknown_cost=unknown_cost,
        cfg=aligner_cfg,
    )
    conn.update(
        {
            "semantic_obstacle": semantic_gate,
            "height_obstacle": height["height_obstacle"],
            "floor_prob": floor_gate,
            "floor_depth": height["floor_depth"],
            "relative_height": height["relative_height"],
            "obstacle_gate": obstacle_gate,
            "aligner_tier": 5,
        }
    )
    return conn


class SafeTopoNavigationAligner(nn.Module):
    """Module wrapper for the two-stage navigation alignment utilities."""

    def __init__(
        self,
        threshold: float = 0.58,
        temperature: float = 20.0,
        connectivity_steps: int = 160,
        protect_radius: int = 2,
        semantic_weight: float = 0.65,
        height_weight: float = 0.35,
    ):
        super().__init__()
        self.threshold = threshold
        self.temperature = temperature
        self.connectivity_steps = connectivity_steps
        self.protect_radius = protect_radius
        self.semantic_weight = semantic_weight
        self.height_weight = height_weight

    def forward(self, semantic_obstacle, depth, start, end=None):
        return align_navigation_fields(
            semantic_obstacle,
            depth,
            start,
            end=end,
            threshold=self.threshold,
            temperature=self.temperature,
            connectivity_steps=self.connectivity_steps,
            protect_radius=self.protect_radius,
            semantic_weight=self.semantic_weight,
            height_weight=self.height_weight,
        )


class SafeTopoQKVSelfAttention(nn.Module):
    """
    Drop-in wrapper for transformers.models.vit.modeling_vit.ViTSelfAttention.

    It preserves the original query/key/value projections, but overwrites a tail
    slice of each head's Q/K with topology tokens before computing QK attention.
    """

    def __init__(
        self,
        base_attention: nn.Module,
        topo_dim: int = 8,
        topo_scale: float = 1.0,
        head_mode: str = "uniform",
    ):
        super().__init__()
        self.query = base_attention.query
        self.key = base_attention.key
        self.value = base_attention.value
        self.dropout = getattr(base_attention, "dropout", nn.Dropout(getattr(base_attention, "dropout_prob", 0.0)))
        self.num_attention_heads = base_attention.num_attention_heads
        self.attention_head_size = base_attention.attention_head_size
        self.all_head_size = base_attention.all_head_size
        self.topo_dim = int(topo_dim)
        self.topo_scale = float(topo_scale)
        self.head_mode = head_mode
        if self.topo_dim <= 0 or self.topo_dim > self.attention_head_size:
            raise ValueError("topo_dim must be in [1, attention_head_size]")
        if self.head_mode not in ("uniform", "split", "joint_split"):
            raise ValueError("head_mode must be 'uniform', 'split', or 'joint_split'")
        self.topology_tokens: Optional[torch.Tensor] = None
        self.layer_role: Optional[str] = None
        # Inference ablation: split mode skips last-1/3 connectivity-head inject (train unchanged).
        self.skip_connectivity_group: bool = False

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def set_topology_tokens(self, tokens: Optional[torch.Tensor]) -> None:
        self.topology_tokens = tokens

    def _topology_for(self, hidden_states: torch.Tensor) -> Optional[torch.Tensor]:
        if self.topology_tokens is None:
            return None

        topo = self.topology_tokens.to(device=hidden_states.device, dtype=hidden_states.dtype)
        if topo.shape[0] != hidden_states.shape[0]:
            raise ValueError("topology token batch size does not match hidden states")
        if topo.shape[1] < hidden_states.shape[1]:
            pad = torch.zeros(
                topo.shape[0],
                hidden_states.shape[1] - topo.shape[1],
                topo.shape[2],
                device=topo.device,
                dtype=topo.dtype,
            )
            topo = torch.cat([topo, pad], dim=1)
        topo = topo[:, : hidden_states.shape[1], :]
        if topo.shape[-1] < self.topo_dim:
            pad = torch.zeros(*topo.shape[:-1], self.topo_dim - topo.shape[-1], device=topo.device, dtype=topo.dtype)
            topo = torch.cat([topo, pad], dim=-1)
        topo = topo[..., : self.topo_dim] * self.topo_scale
        return topo

    def _split_head_topology(self, topo: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, D = topo.shape
        H = self.num_attention_heads
        by_head = torch.zeros(B, H, N, D, device=topo.device, dtype=topo.dtype)
        active = torch.zeros(H, device=topo.device, dtype=torch.bool)

        if self.head_mode == "uniform":
            by_head[:] = topo.unsqueeze(1)
            active[:] = True
            return by_head, active

        # Split roles for ViT-Base style heads:
        # first third: visual heads, untouched
        # middle third: direction/angle topology
        # final third: connectivity/obstacle topology
        first_third = H // 3
        second_third = (2 * H) // 3

        if self.head_mode == "split":
            direction_topo = topo
            # The base 8 channels are [x, y, grad_x, grad_y, cos, sin, floor, obstacle].
            # Rolling by 6 makes connectivity channels lead the dot-product subspace.
            connectivity_topo = torch.roll(topo, shifts=-6, dims=-1)
            by_head[:, first_third:second_third] = direction_topo.unsqueeze(1)
            if not self.skip_connectivity_group:
                by_head[:, second_third:] = connectivity_topo.unsqueeze(1)
                active[first_third:] = True
            else:
                active[first_third:second_third] = True
            return by_head, active

        if self.head_mode == "joint_split":
            # Four shape groups for manipulator C-space: kinematic / coordination / safety / narrow.
            q1 = max(1, H // 4)
            q2 = max(q1 + 1, H // 2)
            q3 = max(q2 + 1, (3 * H) // 4)
            d = topo.shape[-1]
            groups = (
                topo,
                torch.roll(topo, shifts=-(d // 4), dims=-1),
                torch.roll(topo, shifts=-(d // 2), dims=-1),
                torch.roll(topo, shifts=-((3 * d) // 4), dims=-1),
            )
            by_head[:, :q1] = groups[0].unsqueeze(1)
            by_head[:, q1:q2] = groups[1].unsqueeze(1)
            by_head[:, q2:q3] = groups[2].unsqueeze(1)
            by_head[:, q3:] = groups[3].unsqueeze(1)
            active[:] = True
            return by_head, active

        raise ValueError(f"unsupported head_mode: {self.head_mode}")

    def forward(self, hidden_states, head_mask=None, output_attentions: bool = False):
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        topo = self._topology_for(hidden_states)
        if topo is not None:
            query_layer = query_layer.clone()
            key_layer = key_layer.clone()
            topo_by_head, active_heads = self._split_head_topology(topo)
            query_layer[:, active_heads, :, -self.topo_dim :] = topo_by_head[:, active_heads]
            key_layer[:, active_heads, :, -self.topo_dim :] = topo_by_head[:, active_heads]

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        if output_attentions:
            return (context_layer, attention_probs)
        return (context_layer, None)


def install_safetopo_attention(
    vit_model: nn.Module,
    num_last_layers: int = 1,
    topo_dim: int = 8,
    topo_scale: float = 1.0,
    head_mode: str = "uniform",
    layer_roles: Optional[Sequence[str]] = None,
) -> List[SafeTopoQKVSelfAttention]:
    """Replace the last num_last_layers ViT self-attention modules.

    When ``layer_roles`` is set (len == num_last_layers), each wrapper gets a
    distinct ``layer_role`` and uses uniform head routing (specialization via tokens).
    """
    if num_last_layers <= 0:
        return []
    layers = vit_model.encoder.layer
    start = max(0, len(layers) - num_last_layers)
    roles = list(layer_roles) if layer_roles is not None else [None] * num_last_layers
    if layer_roles is not None and len(roles) != num_last_layers:
        raise ValueError("layer_roles length must match num_last_layers")

    wrappers: List[SafeTopoQKVSelfAttention] = []
    for i, layer in enumerate(layers[start:]):
        current = layer.attention.attention
        role = roles[i] if i < len(roles) else None
        eff_head_mode = "uniform" if role is not None else head_mode
        if isinstance(current, SafeTopoQKVSelfAttention):
            wrapper = current
        else:
            wrapper = SafeTopoQKVSelfAttention(
                current, topo_dim=topo_dim, topo_scale=topo_scale, head_mode=eff_head_mode
            )
            layer.attention.attention = wrapper
        wrapper.head_mode = eff_head_mode
        wrapper.layer_role = role
        wrappers.append(wrapper)
    return wrappers


def set_safetopo_tokens(
    wrappers: Iterable[SafeTopoQKVSelfAttention],
    tokens: Optional[torch.Tensor],
    *,
    tokens_per_layer: Optional[Sequence[torch.Tensor]] = None,
) -> None:
    """Set topology tokens on all wrappers, or per-layer when ``tokens_per_layer`` is given."""
    wrappers = list(wrappers)
    if tokens_per_layer is not None:
        if len(tokens_per_layer) != len(wrappers):
            raise ValueError("tokens_per_layer length must match number of wrappers")
        for wrapper, tok in zip(wrappers, tokens_per_layer):
            wrapper.set_topology_tokens(tok)
        return
    for wrapper in wrappers:
        wrapper.set_topology_tokens(tokens)


# Joint-time topology: 12 base channels grouped as kinematic / coordination / safety / narrow.
JOINT_TOPO_NUM_CHANNELS = 12
JOINT_TOPO_CHANNEL_GROUPS = {
    "kinematic": (0, 6),       # q, vel, limit, cos, sin, time
    "coordination": (6, 9),    # joint index + cross-joint coupling
    "safety": (9, 11),         # scene + self collision
    "narrow": (11, 12),        # passage-band proximity
}


def _joint_coordination_maps(
    path_m: torch.Tensor,
    vel: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Cross-joint coordination: each DOF channel encodes relation to the rest of the arm.

    Returns per-[B,T,D] maps: mean deviation, velocity co-movement, cross-velocity coupling.
    """
    B, T, D = path_m.shape
    mean_q = path_m.mean(dim=-1, keepdim=True)
    coord_dev = mean_q - path_m
    vel_mean = vel.mean(dim=-1, keepdim=True)
    coord_vel_sync = vel * vel_mean
    denom = max(D - 1, 1)
    vel_other_mean = (vel.sum(dim=-1, keepdim=True) - vel) / denom
    coord_cross = vel * vel_other_mean
    return coord_dev, coord_vel_sync, coord_cross


def build_joint_topology_map(
    path_m: torch.Tensor,
    collision_cost: Optional[torch.Tensor] = None,
    singularity_cost: Optional[torch.Tensor] = None,
    self_collision_cost: Optional[torch.Tensor] = None,
    narrow_passage_cost: Optional[torch.Tensor] = None,
    patch_time: int = 4,
    patch_joint: int = 2,
) -> torch.Tensor:
    """
    Build joint-time topology tensor [B, 12, H, W] on the configuration manifold grid.

    Channels (shape groups for ``head_mode='joint_split'``):
      0-5  kinematic (q, vel, limit, cos/sin phase, time)
      6-8  coordination (joint index, mean deviation, cross-velocity)
      9-10 safety (scene obstacle, self-collision sensitivity)
      11   narrow-passage band (C-space preimage boundary)
    """
    if path_m.dim() != 3:
        raise ValueError("path_m must have shape [B, T, D]")
    B, T, D = path_m.shape
    device = path_m.device
    dtype = path_m.dtype

    vel = path_m[:, 1:, :] - path_m[:, :-1, :]
    vel = F.pad(vel, (0, 0, 0, 1))
    limit_margin = torch.minimum(path_m, 1.0 - path_m).min(dim=-1, keepdim=True).values * 2.0 - 1.0
    t_coord = torch.linspace(-1, 1, T, device=device, dtype=dtype).view(1, T, 1).expand(B, T, 1)
    j_coord = torch.linspace(-1, 1, D, device=device, dtype=dtype).view(1, 1, D).expand(B, T, D)
    theta = torch.atan2(vel, torch.ones_like(vel) * 1e-8 + path_m.roll(1, dims=-1))

    def _expand_cost(cost: Optional[torch.Tensor]) -> torch.Tensor:
        if cost is None:
            return torch.zeros(B, T, D, device=device, dtype=dtype)
        if cost.dim() == 2:
            cost = cost.unsqueeze(-1)
        return cost.expand(B, T, D)

    scene = _expand_cost(collision_cost) * 2.0 - 1.0
    self_coll = _expand_cost(self_collision_cost) * 2.0 - 1.0
    narrow = _expand_cost(narrow_passage_cost) * 2.0 - 1.0
    coord_dev, _, coord_cross = _joint_coordination_maps(path_m, vel)

    topo_map = torch.stack(
        [
            path_m,
            vel,
            limit_margin.expand(B, T, D),
            torch.cos(theta),
            torch.sin(theta),
            t_coord.expand(B, T, D),
            j_coord,
            coord_dev,
            coord_cross,
            scene,
            self_coll,
            narrow,
        ],
        dim=1,
    )

    pt, pj = patch_time, patch_joint
    pad_t = (pt - T % pt) % pt
    pad_j = (pj - D % pj) % pj
    if pad_t or pad_j:
        topo_map = F.pad(topo_map, (0, pad_j, 0, pad_t))
    return topo_map


def manifold_features_to_safetopo_tokens(
    feat: torch.Tensor,
    n_patches: int,
    topo_dim: int,
    *,
    include_cls: bool = True,
) -> torch.Tensor:
    """
    Map native C-space feature sequence [B, N, C] → ViT SafeTopo tokens [B, 1+N', topo_dim].

    Interpolation is along the sample/time axis only (no joint-index pseudo spatial axis).
    """
    if feat.dim() != 3:
        raise ValueError("feat must be [B, N, C]")
    B, N, C = feat.shape
    if n_patches <= 0:
        raise ValueError("n_patches must be positive")
    flat = feat.transpose(1, 2).contiguous()  # [B, C, N]
    seq = F.interpolate(flat, size=n_patches, mode="linear", align_corners=False)
    seq = seq.transpose(1, 2).contiguous()
    tokens = _expand_topology_features(seq, topo_dim)
    if include_cls:
        cls = torch.zeros(B, 1, topo_dim, device=tokens.device, dtype=tokens.dtype)
        tokens = torch.cat([cls, tokens], dim=1)
    return tokens


def build_joint_manifold_step_features(
    path_m: torch.Tensor,
    collision_cost: Optional[torch.Tensor] = None,
    singularity_cost: Optional[torch.Tensor] = None,
    self_collision_cost: Optional[torch.Tensor] = None,
    narrow_passage_cost: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Per-timestep native 7D C-space features [B, T, C]."""
    if path_m.dim() != 3:
        raise ValueError("path_m must be [B, T, D]")
    B, T, D = path_m.shape
    device, dtype = path_m.device, path_m.dtype

    vel = path_m[:, 1:, :] - path_m[:, :-1, :]
    vel = F.pad(vel, (0, 0, 0, 1))
    if T > 1:
        vel[:, 0] = vel[:, 1]

    limit = torch.minimum(path_m, 1.0 - path_m)
    limit_min = limit.min(dim=-1, keepdim=True).values

    def _expand_cost(cost: Optional[torch.Tensor]) -> torch.Tensor:
        if cost is None:
            return torch.zeros(B, T, 1, device=device, dtype=dtype)
        if cost.dim() == 2:
            cost = cost.unsqueeze(-1)
        if cost.shape[-1] == 1:
            return cost.expand(B, T, 1)
        return cost.mean(dim=-1, keepdim=True)

    scene = _expand_cost(collision_cost)
    self_c = _expand_cost(self_collision_cost)
    narrow = _expand_cost(narrow_passage_cost)
    if narrow_passage_cost is None and singularity_cost is not None:
        narrow = _expand_cost(singularity_cost)

    t_coord = torch.linspace(-1, 1, T, device=device, dtype=dtype).view(1, T, 1).expand(B, T, 1)
    mean_m = path_m.mean(dim=-1, keepdim=True)
    coord_dev = torch.linalg.norm(path_m - mean_m, dim=-1, keepdim=True)

    return torch.cat([path_m, vel, limit_min, scene, self_c, narrow, t_coord, coord_dev], dim=-1)


def build_joint_topology_tokens(
    path_m: torch.Tensor,
    collision_cost: Optional[torch.Tensor] = None,
    singularity_cost: Optional[torch.Tensor] = None,
    self_collision_cost: Optional[torch.Tensor] = None,
    narrow_passage_cost: Optional[torch.Tensor] = None,
    patch_time: int = 4,
    patch_joint: int = 2,
    topo_dim: int = 8,
    include_cls: bool = True,
) -> torch.Tensor:
    """
    Build joint SafeTopo tokens from native 7D C-space trajectory features.

    Tokens are indexed along time in manifold space, not the pseudo-2D (time × joint-index) chart.
    ``patch_joint`` is kept for API compatibility; n_patches still matches the ViT patch grid.
    """
    if narrow_passage_cost is None and singularity_cost is not None:
        if singularity_cost.dim() == 2:
            narrow_passage_cost = singularity_cost.unsqueeze(-1)
        else:
            narrow_passage_cost = singularity_cost

    B, T, D = path_m.shape
    feats = build_joint_manifold_step_features(
        path_m,
        collision_cost=collision_cost,
        singularity_cost=singularity_cost,
        self_collision_cost=self_collision_cost,
        narrow_passage_cost=narrow_passage_cost,
    )
    ph = max(1, T // patch_time)
    pw = max(1, D // patch_joint)
    n_patches = ph * pw
    return manifold_features_to_safetopo_tokens(
        feats, n_patches, topo_dim, include_cls=include_cls
    )
