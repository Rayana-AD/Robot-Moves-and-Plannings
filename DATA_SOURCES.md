# Data sources (nothing bundled in this repository)

This repository ships **code and documentation only**. Datasets, trained checkpoints, and full experiment logs are **not** uploaded here. Use the links below to obtain data legally and cite the original work.

## Navigation (SafeTopo)

| Resource | Role | How to obtain |
|----------|------|----------------|
| **NYU Depth Dataset v2** | RGB-D indoor scenes; 60-train / 28-eval split used in the paper | [https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) — also available via Hugging Face `datasets` loaders in the full research codebase |
| **ViT-Base patch16 224** | Frozen visual backbone | `google/vit-base-patch16-224` on Hugging Face (ImageNet-21k pre-training) |
| **Eval occupancy judge** | RGB-D fusion oracle for hard-collision scoring (not training labels) | Algorithm documented in `models/nav_floor_oracle.py` (`NAV_FLOOR_ORACLE=rgbd_sheet`); no third-party download required |

Training protocol (mechanism check): global passable-map BCE, **no start/goal queries** during training. See `docs/safetopo.md`.

## Manipulation (Morph)

| Resource | Role | How to obtain |
|----------|------|----------------|
| **PyBullet + Franka URDF** | Whole-arm collision and FK | `pip install pybullet`; URDF from `pybullet_data` (`franka_panda/panda.urdf`) |
| **Sealed fixture families** | Bookshelf, door, bugtrap, chicane, arch, etc. | **Procedural** boxes spawned in PyBullet (`joint_visible_passage` in the full research repo); not a separate dataset download |
| **Opening Detect field** | Workspace SOLID / OPENING / FREE voxels | Built at runtime from fixture geometry + perception; see `docs/morph.md` |

## Coupled demos

Kitchen / NYU frame **00090** coupled showcases in the paper use the same NYU RGB-D frame plus a morphed arm plan after base parking. No additional dataset beyond NYU + simulation scene parameters.

## What we deliberately omit

- Paper PDF / supplementary ZIP  
- Full `results/` trees, mined JSON records, ViT checkpoints (multi-GB)  
- Replica / Habitat assets unless you enable those optional scripts in the private research tree  

If you reproduce numbers, request checkpoint hashes or split manifests from the authors after publication.
