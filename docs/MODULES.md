# Module map and sync policy

## Included in this repository

| Path | Lines of responsibility | Standalone? |
|------|-------------------------|-------------|
| `models/safe_topo_qkv.py` | Topology tokens, ViT attention wrappers, head split | Yes (PyTorch + transformers) |
| `models/nav_patch_connectivity.py` | Passable head, patch GT helpers | Needs `nav_floor_oracle`, `joint_cspace_connectivity` |
| `models/nav_floor_oracle.py` | Depth/RGB-D floor oracle for train & eval judge | Yes (NumPy/Torch) |
| `models/joint_cspace_connectivity.py` | BCE / connectivity tensor helpers | Yes |
| `models/joint_morph_correspondence.py` | Morph chart construction | Needs PyBullet env, Detect, planner from research repo |
| `models/joint_safetopo_decode.py` | Poisson decode + column snap | Needs chart masks from correspondence |

Imports use the package name `models.*`. Either:

```bash
export PYTHONPATH="/path/to/Robot-Moves-and-Plannings"
```

or install editable:

```bash
pip install -e .   # after pyproject.toml is added in your fork
```

## Still in the private research repository

These are **not** vendored here to keep the public tree small and to avoid shipping multi-gigabyte artifacts:

- `scripts/train_nav_clean_connectivity.py`, `run_nav_clean_eval.py` — NYU 28-scene eval  
- `models/joint_visible_passage.py`, `joint_scene.py`, `coupled_vit_pipeline.py` — sealed families + coupled ViT runs  
- `results/`, checkpoints, mined replay JSON  

When the paper is public, we may tag matching releases (`v1.0-paper`) between this repo and the research monorepo.

## Sync checklist (maintainers)

1. Copy changed files from research `models/` into this repo.  
2. Run import smoke test: `python -c "from models.safe_topo_qkv import build_patch_topology_tokens"`.  
3. Refresh figures under `assets/figures/` to match `main.tex` `\includegraphics` names (PNG only, no PDF). Current body figures: `fig1-1`, `fig2_nav_clean_hug`, `fig_six_families_geom_sg_topdown_morph`, `fig_showcase_coupled_morph_00090_kitchen_v7`. Do not reuse older `fig6_six_families_joint_qualitative` / ViT Tier-5 sheets.  
4. Update `DATA_SOURCES.md` if splits or oracle flags change.
