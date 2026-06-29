# Single-image SMPL-X / full-body HMR SOTA (researched 2026-06-26)

Question: is the current `SMPLXFit` (ClickPose COCO-17 → MotionAGFormer lift → Umeyama+IK onto SMPL-X, betas=0) the best choice, or is there SOTA/research we're missing?

## Landscape

| Method | Year/venue | Output | Predicts shape (betas)? | Keypoint/HITL prompt? | Code / license |
|---|---|---|---|---|---|
| **SAM 3D Body (3DB)** | Nov 2025, Meta | **MHR** (not SMPL-X) | Yes (`shape_params`) | **Yes — 2D keypoints + masks as prompts** (SAM-style) | github.com/facebookresearch/sam-3d-body, HF facebook/sam-3d-body-dinov3, "SAM License" |
| **SMPLest-X** | TPAMI 2025 | SMPL-X | Yes | No (feed-forward) | github.com/SMPLCap/SMPLest-X (wqyin/SMPLest-X) |
| **SMPLer-X** | NeurIPS 2023 | SMPL-X | Yes | No | github.com/SMPLCap/SMPLer-X. AGORA 107.2 NMVE, EHF 62.3 PVE |
| **Multi-HMR** | ECCV 2024, Naver | SMPL-X (multi-person, single shot) | Yes | No | arxiv 2402.14654, ViT backbone + Human Prediction Head |
| **NLF (Neural Localizer Fields)** | 2024 | SMPL-X + global pos | Yes | No | 25M frames, SOTA generalization |
| **CameraHMR** | 2025 | SMPL (body) + camera | Yes | No | strong camera/global, body-only |

## Key takeaways
- The current `SMPLXFit` is a **pragmatic, not SOTA** design. Frozen betas=0 is a hard fidelity ceiling; MotionAGFormer is a *temporal/video* lifter used off-label on single frames; 2-stage error stacking.
- Its one differentiator is **HITL editability** (ClickPose 2D fix + drag-IK editor), which feed-forward regressors lack.
- **SAM 3D Body is the paradigm match**: it is *promptable by 2D keypoints* — i.e. ClickPose's HITL-corrected keypoints become the prompt. SOTA accuracy + predicts shape. BUT: outputs **MHR, not SMPL-X** (no conversion provided), "SAM License" (commercial terms unverified), heavier deps (DINOv3).

## Two viable upgrade paths
1. **Stay SMPL-X, upgrade the init**: replace MotionAGFormer lift with a direct SMPL-X regressor (**SMPLest-X** or **Multi-HMR**) as initialization → real betas + better pose prior, then keep the IK editor on top. Keeps editability, stays in SMPL-X, moderate effort.
2. **Adopt SAM 3D Body**: keypoint-promptable = native HITL, SOTA, predicts shape — but MHR (need MHR→SMPL-X conversion or switch rig), SAM License, bigger dependency. Higher payoff, more disruptive.

## Sources
- https://ai.meta.com/research/publications/sam-3d-body-robust-full-body-human-mesh-recovery/
- https://github.com/facebookresearch/sam-3d-body
- https://huggingface.co/facebook/sam-3d-body-dinov3
- https://arxiv.org/html/2501.09782v1 (SMPLest-X)
- https://github.com/SMPLCap/SMPLer-X
- https://arxiv.org/abs/2402.14654 (Multi-HMR)
