"""
WiLoR (Potamias et al. 2024) in-the-wild hand reconstruction -> SMPL-X hand pose.

YOLO detects each hand, WiLoR reconstructs MANO. We keep only the per-finger
articulation (``hand_pose``, 15 joints) and map it to SMPL-X ``left_hand_pose`` /
``right_hand_pose`` (45 = 15x3 axis-angle). The wrist global orientation is NOT
used — the wrist comes from the body estimator (NLF/Multi-HMR), so grafting the
finger pose attaches cleanly at the body's wrist (no melt).

Left hands: ViTDetDataset flips the crop so the right-hand MANO model applies, so
WiLoR's left output is in right-hand convention -> we mirror it to SMPL-X's left
convention by negating the y,z axis-angle components per joint.

WiLoR is its own `wilor` package (no bare utils/model collision), so a plain
sys.path insert is safe. MANO model files live in WILOR_DIR/mano_data.
License: WiLoR weights are CC-BY-NC-ND (non-commercial).
"""

import os
import sys

import cv2
import numpy as np
import torch

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# WiLoR source (code + model_config.yaml + mano_data). install.py clones it into
# vendor/WiLoR; override with the WILOR_DIR env var.
WILOR_DIR = os.environ.get("WILOR_DIR", os.path.join(_PKG_ROOT, "vendor", "WiLoR"))
DEFAULT_WILOR_CKPT = os.path.join(WILOR_DIR, "pretrained_models", "wilor_final.ckpt")
DEFAULT_WILOR_DETECTOR = os.path.join(WILOR_DIR, "pretrained_models", "detector.pt")
_CFG = os.path.join(WILOR_DIR, "pretrained_models", "model_config.yaml")
_MANO_DIR = os.path.join(WILOR_DIR, "mano_data")

_cache: dict = {}


def _ensure_render_stubs():
    """Stub pyrender so WiLoR's renderer modules (`import pyrender` at module scope)
    load without the optional, inference-unused renderer installed."""
    import types

    class _Dummy:
        def __getattr__(self, _n):
            return _Dummy()

        def __call__(self, *a, **k):
            return _Dummy()

    if "pyrender" not in sys.modules:
        try:
            __import__("pyrender")
        except Exception:
            m = types.ModuleType("pyrender")
            m.__getattr__ = lambda _n: _Dummy()
            sys.modules["pyrender"] = m


def load_wilor(ckpt_path=DEFAULT_WILOR_CKPT, detector_path=DEFAULT_WILOR_DETECTOR, device="cuda"):
    """Load (and cache) the WiLoR model + YOLO hand detector.

    ckpt_path / detector_path are the weights (folder_paths-resolved by the Load
    node); the model config + MANO files come from the WiLoR code clone (WILOR_DIR).
    """
    key = (ckpt_path, detector_path, device)
    if key in _cache:
        return _cache[key]
    for p in (ckpt_path, detector_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(
                f"WiLoR asset missing: {p!r}. Download detector.pt + wilor_final.ckpt from "
                f"huggingface.co/spaces/rolpotamias/WiLoR into ComfyUI/models/wilor/."
            )
    if WILOR_DIR not in sys.path:
        sys.path.insert(0, WILOR_DIR)
    _ensure_render_stubs()          # let vendor `import pyrender` succeed on fresh clones
    from wilor.models import WiLoR
    from wilor.configs import get_config
    from ultralytics import YOLO

    cfg = get_config(_CFG, update_cachedir=True)
    cfg.defrost()
    if ("vit" in cfg.MODEL.BACKBONE.TYPE) and ("BBOX_SHAPE" not in cfg.MODEL):
        cfg.MODEL.BBOX_SHAPE = [192, 256]
    if "PRETRAINED_WEIGHTS" in cfg.MODEL.BACKBONE:
        cfg.MODEL.BACKBONE.pop("PRETRAINED_WEIGHTS")
    cfg.MANO.DATA_DIR = _MANO_DIR + "/"
    cfg.MANO.MODEL_PATH = _MANO_DIR + "/"
    cfg.MANO.MEAN_PARAMS = os.path.join(_MANO_DIR, "mano_mean_params.npz")
    cfg.freeze()

    # Lightning's load_from_checkpoint can't pass weights_only=False, and the ckpt
    # pickles non-tensor globals (trusted source) that torch 2.6 rejects by default.
    # So load the Lightning ckpt manually and build the model ourselves.
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model = WiLoR(cfg, init_renderer=False)
    model.load_state_dict(state, strict=False)
    model = model.to(device).eval()
    detector = YOLO(detector_path)
    _cache[key] = (model, cfg, detector)
    return _cache[key]


def estimate_hand_pose(model, cfg, detector, image_rgb01, device, conf=0.3):
    """
    Detect + reconstruct hands; return {"left_hand_pose": (45,), "right_hand_pose": (45,)}
    (only the sides that were detected), as SMPL-X-convention axis-angle.
    """
    import roma
    from wilor.datasets.vitdet_dataset import ViTDetDataset

    bgr = cv2.cvtColor((np.clip(image_rgb01, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    det = detector(bgr, conf=conf, verbose=False, device=device)[0]
    boxes, right = [], []
    for d in det:
        b = d.boxes.data.cpu().numpy().squeeze()
        if b.size < 4:
            continue
        right.append(float(np.atleast_1d(d.boxes.cls.cpu().numpy().squeeze())[0]))
        boxes.append(b[:4].tolist())
    if not boxes:
        return {}

    ds = ViTDetDataset(cfg, bgr, np.stack(boxes), np.stack(right))
    loader = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)

    out_pose: dict = {}
    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.no_grad():
            out = model(batch)
        hp = out["pred_mano_params"]["hand_pose"]               # (B,15,3,3) rotmat
        aa = roma.rotmat_to_rotvec(hp.reshape(-1, 3, 3)).reshape(hp.shape[0], 15, 3)
        aa = aa.float().cpu().numpy()
        rights = batch["right"].cpu().numpy().reshape(-1)
        for n in range(hp.shape[0]):
            a = aa[n].copy()
            is_right = rights[n] > 0.5
            if not is_right:                                    # mirror to SMPL-X left convention
                a[:, 1] *= -1
                a[:, 2] *= -1
            key = "right_hand_pose" if is_right else "left_hand_pose"
            out_pose.setdefault(key, a.reshape(45).astype(np.float32))   # first (highest-conf) per side
    return out_pose
