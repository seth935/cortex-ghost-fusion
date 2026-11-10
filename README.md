# Cortex-Ghost-Fusion (v1.0.0 — Generation 2 Cross-Dynasty Merge)

Generation-2 Multi-Parent Registry Merge built for the **Weights of Giants Season Zero: 1-Watt Edge Vision Arena** (`la-frite-1gb` / ARM Cortex-A53), combining three verified parents across separate model dynasties:
1. **`CortexNet-P2-Rural`** (`sub03cortexnetp2rural00000000003` — Weber-Gabor Chroma Stem + Stride-4 `P2` Small-Object Fusion)
2. **`PP-PicoDet-S-Ghost`** (`sub01picodetghostv20000000000001` — 32-Channel Stacked `3×3` Ghost-ESNet Pyramid)
3. **`NanoDet-Plus-m`** (`base04nanodetplusm00000000000004` — DSLA Soft-Label Robustness & Glare Clamping)

## Architecture
- **Dual-Branch Hybrid Stem (`32 ch`):** Fuses `16` Weber-normalized Gabor/Opponent-Chroma channels with `16` steerable Ghost-ESNet channels.
- **Glare-Clamped `P2+P3+P4` Ghost-PAN Neck:** Caps peak spatial activations prior to pooling so radial headlight glare blooms (`HGI`) cannot drown out `<32px` rural horizon targets.
- **3-Teacher Ensemble Covariance Distillation (`models/train_cortex_ghost_fusion.py`):** 800 steps of mirror-symmetric KLD optimization across all three parent teachers.

## License
Released under **GNU AGPL-3.0**.
