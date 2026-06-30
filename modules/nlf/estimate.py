"""
NLF (Neural Localizer Fields, Sárándi & Pons-Moll, NeurIPS 2024) single-image
human pose -> SMPL-X parameters.

The released TorchScript model (isarandi/nlf, nlf_l_multi) is **SMPL**, not SMPL-X:
``detect_smpl_batched`` returns pose (72 = 24x3 axis-angle), betas (10), trans (3),
plus a 6890-vertex mesh and 24 joints. We map the SMPL BODY pose onto SMPL-X
(joints 1-21 share the kinematic chain), which is a one-forward-pass SOTA pose.

Caveats (surfaced, not hidden):
  - **betas do NOT transfer** (SMPL and SMPL-X use different shape bases) -> we set
    betas=0 (neutral); tune with the editor's `betas` widget, or add the official
    SMPL->SMPL-X transfer later.
  - NLF has **no hands/face** -> hands come from WholeBodyHandDetector or stay flat.
  - NLF predicts in an OpenCV camera frame (Y-down, Z-forward); we rotate 180° about
    X to put the body Y-up for the SMPL-X model / renderer.
"""

import os

import cv2
import numpy as np
import torch

DEFAULT_NLF_MODEL = "models/nlf/nlf_l_multi_0.3.2.torchscript"

# OpenCV camera (Y-down, Z-forward) -> Y-up world (rotate 180° about X).
_RFIX = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], np.float32)

_nlf_cache: dict = {}


def load_nlf(model_path: str, device: str):
    """Load (and cache) the NLF TorchScript model. Needs torchvision imported so the
    serialized ``torchvision::nms`` op resolves.

    NLF's released TorchScript is GPU-only: its exported detector bakes
    torch.device("cuda:0") into the graph, so it cannot run on CPU."""
    if not str(device).startswith("cuda"):
        raise RuntimeError(
            "NLF's released TorchScript model is GPU-only (cuda:0 is baked into the "
            "exported detector), so it cannot run on CPU. Set device=cuda on 'Load NLF' "
            "(free GPU memory if it OOMs), or use 'Full Body: Multi-HMR' which runs on CPU."
        )
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"NLF model not found: {model_path!r}. Download "
            f"nlf_l_multi_0.3.2.torchscript from "
            f"github.com/isarandi/nlf/releases into ComfyUI/models/nlf/."
        )
    key = (os.path.abspath(model_path), device)
    if key not in _nlf_cache:
        import torchvision  # noqa: F401  (registers torchvision::nms for the jit model)
        _nlf_cache[key] = torch.jit.load(model_path, map_location=device).eval()
    return _nlf_cache[key]


def _pick_person(pred):
    """Choose the largest-box detection; return (pose72, betas10, trans3, box4) numpy.
    box4 = [x0,y0,x1,y1] in image pixels."""
    # When NLF detects nobody it may omit keys entirely -> treat as "no person".
    pose_l = pred.get("pose")
    if not pose_l or pose_l[0].shape[0] == 0:
        raise RuntimeError(
            "NLF detected no person in the image. Try a clearer / more centered photo, "
            "or use 'Full Body: Multi-HMR'."
        )
    pose = pose_l[0]              # (N,72)
    betas = pred["betas"][0]     # (N,10)
    trans = pred["trans"][0]     # (N,3)
    boxes = pred["boxes"][0] if "boxes" in pred else None
    if pose.shape[0] == 1 or boxes is None:
        i = 0
    else:
        b = boxes.detach().cpu().numpy()
        areas = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        i = int(np.argmax(areas))
    box = (boxes[i, :4].detach().cpu().numpy().astype(np.float32)
           if boxes is not None else np.zeros(4, np.float32))
    return (pose[i].detach().cpu().numpy().astype(np.float32),
            betas[i].detach().cpu().numpy().astype(np.float32),
            trans[i].detach().cpu().numpy().astype(np.float32),
            box)


def estimate_smplx_params(model, image_rgb01, device, offload_after=True):
    """
    Run NLF on one image and return SMPL-X body params (Y-up).

    image_rgb01: (H,W,3) float RGB in [0,1] (a single ComfyUI IMAGE frame).
    Returns dict: global_orient(3), body_pose(63), transl(3), bbox[x0,y0,x1,y1] —
    all numpy float32. bbox is the person box in image pixels (no frame fix).
    NLF is moved back to CPU afterwards (offload_after) to free GPU for rendering.
    """
    img_u8 = np.clip(np.asarray(image_rgb01, np.float32) * 255.0, 0, 255).astype(np.uint8)
    t = torch.from_numpy(img_u8).permute(2, 0, 1).unsqueeze(0).to(device)  # (1,3,H,W) uint8
    model.to(device)
    try:
        with torch.inference_mode():
            pred = model.detect_smpl_batched(t)
        pose, _betas, trans, bbox = _pick_person(pred)
    finally:
        del t
        if offload_after:
            model.to("cpu")
            if device != "cpu" and torch.cuda.is_available():
                torch.cuda.empty_cache()

    # SMPL pose: [0:3]=global_orient, [3:66]=body joints 1-21 (shared with SMPL-X),
    # [66:72]=SMPL hand-root joints (no SMPL-X body equivalent -> dropped).
    Rg, _ = cv2.Rodrigues(pose[:3])
    g_fixed, _ = cv2.Rodrigues(_RFIX @ Rg)              # into Y-up world
    return {
        "global_orient": g_fixed.reshape(3).astype(np.float32),
        "body_pose": pose[3:66].astype(np.float32).copy(),
        "transl": (_RFIX @ trans).astype(np.float32),
        "bbox": bbox,                                  # [x0,y0,x1,y1] image px
    }
