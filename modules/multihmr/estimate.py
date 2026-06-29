"""
Multi-HMR (Naver, ECCV 2024) single-image expressive whole-body SMPL-X.

ONE forward pass -> full SMPL-X for the most prominent person: global_orient,
body_pose, left/right hand_pose, jaw_pose, betas (real shape), expression, transl.
This replaces the NLF(SMPL)+hand-curl hack so hands/face are jointly estimated.

Integration notes:
- Multi-HMR's code lives at MULTIHMR_DIR (cloned, not pip-installed). It uses
  generic top-level imports (`import utils`/`blocks`/`model`), so we add it to
  sys.path and import LAZILY (inside load) — after ComfyUI's startup imports — to
  minimise sys.modules collisions.
- It expects assets under a `models/` dir; we patch the two paths it reads:
  blocks.smpl_layer.SMPLX_DIR (parent of smplx/) and model.MEAN_PARAMS.
- The pose `rotvec` is [53,3] axis-angle = root(1)+body(21)+lhand(15)+rhand(15)+jaw(1).
- Multi-HMR predicts in an OpenCV camera frame (Y-down); we rotate 180° about X
  (as for NLF) to put global_orient/transl into the Y-up world our renderer uses.
  License: Multi-HMR weights are Naver NON-COMMERCIAL / research-only.
"""

import os
import sys

import numpy as np
import torch

MULTIHMR_DIR = "/home/wswg3/github/multi-hmr"
DEFAULT_MULTIHMR_CKPT = "models/multiHMR/multiHMR_896_L.pt"   # relative to ComfyUI CWD
_MEAN_PARAMS = "/home/wswg3/github/ComfyUI/models/smpl_mean_params.npz"

_RFIX = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], np.float32)   # OpenCV cam -> Y-up
_cache: dict = {}
_Model = None            # captured Multi-HMR Model class
_normalize_rgb = None    # captured utils.image.normalize_rgb
_get_focal = None        # captured utils.camera.get_focalLength_from_fieldOfView


def _prepare_imports(smplx_parent):
    """
    Import Multi-HMR's ``model``/``blocks``/``utils`` in ISOLATION from ComfyUI's own
    top-level ``utils``/``model``/``blocks`` packages.

    ComfyUI ships its own ``utils`` package (already in sys.modules), so importing
    Multi-HMR normally makes ``from utils import inverse_perspective_projection``
    resolve to ComfyUI's ``utils`` and fail. We temporarily evict those names from
    sys.modules, import from MULTIHMR_DIR, capture the symbols we need (they keep
    their own module globals alive via the bound references), then restore ComfyUI's
    entries so the rest of ComfyUI keeps working.
    """
    global _Model, _normalize_rgb, _get_focal
    if _Model is not None:
        return
    import importlib

    def _match(n):
        return n in ("utils", "blocks", "model") or n.startswith(("utils.", "blocks.", "model."))

    saved = {n: m for n, m in list(sys.modules.items()) if _match(n)}
    for n in saved:
        del sys.modules[n]
    sys.path.insert(0, MULTIHMR_DIR)
    try:
        _normalize_rgb = importlib.import_module("utils.image").normalize_rgb
        _get_focal = importlib.import_module("utils.camera").get_focalLength_from_fieldOfView
        sl = importlib.import_module("blocks.smpl_layer")
        sl.SMPLX_DIR = smplx_parent                  # smplx.create(SMPLX_DIR,'smplx',...)
        mh_model = importlib.import_module("model")
        mh_model.MEAN_PARAMS = _MEAN_PARAMS          # np.load at Model init (buffers overwritten by ckpt)
        _Model = mh_model.Model
    finally:
        try:
            sys.path.remove(MULTIHMR_DIR)
        except ValueError:
            pass
        for n in [n for n in list(sys.modules) if _match(n)]:
            del sys.modules[n]                       # drop Multi-HMR's entries
        sys.modules.update(saved)                    # restore ComfyUI's


def load_multihmr(ckpt_path, smplx_parent, device):
    """Load (and cache) Multi-HMR. Returns (model, img_size)."""
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"Multi-HMR checkpoint not found: {ckpt_path!r}. Download e.g. "
            f"multiHMR_896_L.pt from download.europe.naverlabs.com/ComputerVision/MultiHMR/ "
            f"into ComfyUI/models/multiHMR/."
        )
    key = (os.path.abspath(ckpt_path), device)
    if key in _cache:
        return _cache[key]
    _prepare_imports(smplx_parent)

    # weights_only=False: the ckpt stores argparse.Namespace (trusted Naver source).
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    kwargs = {k: v for k, v in vars(ckpt["args"]).items()}
    kwargs["type"] = ckpt["args"].train_return_type
    kwargs["img_size"] = ckpt["args"].img_size[0]
    model = _Model(**kwargs).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    _cache[key] = (model, int(kwargs["img_size"]))
    return _cache[key]


def _preprocess(image_rgb01, img_size, device):
    """ComfyUI IMAGE frame (H,W,3 float [0,1]) -> normalized (1,3,S,S) tensor."""
    from PIL import Image, ImageOps
    arr = (np.clip(np.asarray(image_rgb01, np.float32), 0, 1) * 255).astype(np.uint8)
    pil = Image.fromarray(arr).convert("RGB")
    pil = ImageOps.contain(pil, (img_size, img_size))      # keep aspect
    pil = ImageOps.pad(pil, size=(img_size, img_size))     # pad to square (zeros)
    x = _normalize_rgb(np.asarray(pil))                    # (3,S,S) float, ImageNet norm
    return torch.from_numpy(x).unsqueeze(0).to(device)


def _camera(img_size, device, fov=60.0):
    K = torch.eye(3)
    f = _get_focal(fov=fov, img_size=img_size)
    K[0, 0] = K[1, 1] = f
    K[0, 2] = K[1, 2] = img_size // 2
    return K.unsqueeze(0).to(device)


def _split_pose(rotvec):
    """rotvec [53,3] -> SMPL-X param vectors (numpy float32)."""
    import cv2
    g, _ = cv2.Rodrigues(_RFIX @ cv2.Rodrigues(rotvec[0])[0])   # global_orient into Y-up
    return {
        "global_orient": g.reshape(3).astype(np.float32),
        "body_pose": rotvec[1:22].reshape(63).astype(np.float32).copy(),
        "left_hand_pose": rotvec[22:37].reshape(45).astype(np.float32).copy(),
        "right_hand_pose": rotvec[37:52].reshape(45).astype(np.float32).copy(),
        "jaw_pose": rotvec[52].reshape(3).astype(np.float32).copy(),
    }


def _fit10(v):
    v = np.asarray(v, np.float32).reshape(-1)
    return v[:10].copy() if v.shape[0] >= 10 else np.pad(v, (0, 10 - v.shape[0]))


def estimate_smplx_params(model, img_size, image_rgb01, device, det_thresh=0.3):
    """Run Multi-HMR; return full SMPL-X params for the most prominent person."""
    x = _preprocess(image_rgb01, img_size, device)
    K = _camera(img_size, device)
    use_amp = str(device).startswith("cuda")                 # fp16 like the demo (saves memory)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
        humans = model(x, is_training=False, nms_kernel_size=1,
                       det_thresh=float(det_thresh), K=K)
    if not humans:
        raise RuntimeError("Multi-HMR detected no person in the image.")

    def _extent(h):
        j = h["j2d"].detach().cpu().numpy()
        return float(j[:, 1].max() - j[:, 1].min())
    h = max(humans, key=_extent)                            # most prominent person

    rot = h["rotvec"].detach().cpu().numpy().astype(np.float32)      # [53,3]
    transl = h["transl"].detach().cpu().numpy().astype(np.float32).reshape(3)
    out = _split_pose(rot)
    out["betas"] = _fit10(h["shape"].detach().cpu().numpy())
    out["expression"] = _fit10(h["expression"].detach().cpu().numpy())
    out["transl"] = (_RFIX @ transl).astype(np.float32)
    return out
