# Expressive whole-body SMPL-X (body+hands+face) one-pass models — research 2026-06-29

Goal: replace NLF (SMPL-only) + separate ViTPose hand-curl hack with ONE node that
regresses full SMPL-X (body + hands + facial expression) jointly, so hands don't
"melt" and it's a single node. Constraint: must NOT require the mmcv/mmdet/mmpose
stack (not installed; fragile vs ComfyUI's torch 2.11 / py3.10).

## Disk inventory
- NONE of these models on disk. ComfyUI/models has only `nlf/` (our torchscript) + `vitpose/`.
- venv: `smplx 0.1.28` ✅, `chumpy`, `MultiScaleDeformableAttention`. NO mmcv/mmdet/mmpose/mmhuman3d.

## Findings (from repo READMEs / install scripts)

### Multi-HMR (ECCV 2024, Naver) — RECOMMENDED
- Output: full SMPL-X (body + hands + **expression**) in ONE forward pass. ✅ exactly the ask.
- Deps: plain PyTorch + DINOv2 backbone + smplx + a few pkgs. **NO mmcv/mm-stack.** Tested py3.9 / CUDA 12.1 → compatible with ComfyUI torch 2.11/py3.10 (standard ops).
- Weights: publicly downloadable (Naver Labs + HuggingFace), 6 checkpoints (e.g. multiHMR_672_L ~74ms, multiHMR_896_L ~89ms).
- Accuracy: EHF PVE ~37–42, 3DPW PVE ~90, BEDLAM-val ~57 (strong).
- License: **Naver custom, NON-COMMERCIAL / research-only** (the `Anny` checkpoint has separate terms). ⚠️ flag if commercial use needed.

### SMPLest-X (2025, SMPLCap) — best accuracy, but env-incompatible
- Output: full SMPL-X (body+hands+face). Highest benchmark accuracy in the family.
- install.sh pins **python=3.8, pytorch==1.12.0, cudatoolkit=11.3** (conda) → INCOMPATIBLE with ComfyUI's torch 2.11/py3.10; would need a separate env/subprocess. Extended from SMPLer-X (mm-stack lineage). Huge weight 8.2 GB.
- Verdict: too heavy/incompatible to run in-process.

### SMPLer-X (NeurIPS 2023), OSX (CVPR 2023), AiOS (CVPR 2024)
- All expressive whole-body SMPL-X, but all built on the **mmcv/mmdet/mmpose/mmhuman3d** stack and older torch → same env-incompatibility / fragility. Not suitable for in-process ComfyUI integration.

## Decision
**Multi-HMR** = best accuracy-vs-integration tradeoff with NO mmcv and modern-torch
compatibility. Plan: new node `image -> Multi-HMR -> SMPL-X (body+hands+expression)`
replacing NLFSMPLXEstimator + WholeBodyHandDetector; SMPLXEditor stays on top.
Caveat to confirm with user: non-commercial license.
Sources: github.com/naver/multi-hmr ; github.com/SMPLCap/SMPLest-X (scripts/install.sh).
</content>
