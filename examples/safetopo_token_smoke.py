#!/usr/bin/env python3
"""Smoke test: build 32-D topology tokens from a random depth map (no ViT load)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from models.safe_topo_qkv import build_patch_topology_tokens


def main() -> None:
    depth = torch.rand(1, 1, 224, 224) * 3.0 + 0.5
    tok = build_patch_topology_tokens(depth, patch_size=16, topo_dim=32)
    print("topology tokens shape:", tuple(tok.shape))  # [B, 1+196, 32]
    assert tok.shape[-1] == 32


if __name__ == "__main__":
    main()
