"""
Face: SMIRK (Retsinas et al. 2024) expression capture -> SMPL-X jaw + expression.

SMIRK's encoder regresses FLAME expression (50) + jaw pose from a face crop. SMPL-X
shares FLAME's expression space, so we keep the first 10 expression coefficients
(SMPL-X's default expression dim, matching Multi-HMR's `_fit10`) and the jaw axis-angle
directly. Eyes / neck are not estimated. The output is a partial SMPL-X dict carrying
only the face, grafted onto the body estimator's result (fed into Body: NLF's
`smplx_face` input, same wrist-relative graft path as WiLoR hands).

SMIRK source (georgeretsi/smirk) is loaded by file path from vendor/smirk (override
with the SMIRK_DIR env var); install.py clones it. Only `src/smirk_encoder.py` is
imported, so the upstream demo/training deps are not required.
License: SMIRK code is MIT, but it drives FLAME (MPI, non-commercial) -> the resulting
pipeline is research / non-commercial use only.
"""

import importlib.util
import os
import sys

import cv2
import numpy as np
import torch

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# SMIRK source (code under src/). install.py clones it into vendor/smirk; override
# with the SMIRK_DIR env var.
SMIRK_DIR = os.environ.get("SMIRK_DIR", os.path.join(_PKG_ROOT, "vendor", "smirk"))

_cache: dict = {}


def _smirk_encoder_cls():
    """Import SmirkEncoder from the vendored source by file path — sidesteps upstream
    demo/training imports that would otherwise pull unused deps."""
    enc_path = os.path.join(SMIRK_DIR, "src", "smirk_encoder.py")
    if not os.path.isfile(enc_path):
        raise RuntimeError(
            f"SMIRK source not found at {enc_path}. Clone "
            f"https://github.com/georgeretsi/smirk into {SMIRK_DIR} (install.py does this "
            f"for you), or set the SMIRK_DIR env var to a SMIRK checkout."
        )
    if SMIRK_DIR not in sys.path:
        sys.path.insert(0, SMIRK_DIR)
    spec = importlib.util.spec_from_file_location("_smirk_encoder_isolated", enc_path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as e:
        missing = getattr(e, "name", None) or str(e)
        raise RuntimeError(
            f"SMIRK import failed: missing dependency '{missing}'. Install SMIRK's "
            f"upstream requirements (e.g. `pip install timm`) and retry. Source: {SMIRK_DIR}"
        ) from e
    cls = getattr(module, "SmirkEncoder", None)
    if cls is None:
        raise RuntimeError(
            f"{enc_path} does not define SmirkEncoder — upstream API may have changed "
            f"(see https://github.com/georgeretsi/smirk)."
        )
    return cls


def load_smirk(ckpt_path, device="cuda"):
    """Load (and cache) the SMIRK expression encoder from a checkpoint."""
    key = (os.path.abspath(ckpt_path), device)
    if key in _cache:
        return _cache[key]
    encoder = _smirk_encoder_cls()(n_exp=50, n_shape=300)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    # Upstream checkpoints bundle smirk_encoder + smirk_generator under one flat
    # state_dict; keep the encoder side only (mirrors demo.py in georgeretsi/smirk).
    enc_state = {k.replace("smirk_encoder.", "", 1): v
                 for k, v in state.items() if k.startswith("smirk_encoder.")}
    if not enc_state:                      # older dumps may already be encoder-only
        enc_state = state
    missing, unexpected = encoder.load_state_dict(enc_state, strict=False)
    if missing:
        preview = ", ".join(missing[:6])
        raise RuntimeError(
            f"SMIRK checkpoint {ckpt_path} is missing {len(missing)} required weights "
            f"(first: {preview}; unexpected: {len(unexpected)}) — likely an encoder/config "
            f"mismatch."
        )
    encoder.eval().to(device)
    _cache[key] = encoder
    return encoder


def _prep_face(rgb01):
    """HxWx3 float RGB in [0,1] -> [1,3,224,224] tensor (SMIRK's expected input size)."""
    img = np.clip(np.asarray(rgb01, np.float32), 0.0, 1.0)
    if img.shape[:2] != (224, 224):
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).contiguous()


def estimate_face_params(encoder, rgb01, device="cuda"):
    """Run SMIRK on a face crop; return {"expression": (10,), "jaw_pose": (3,)} for SMPL-X."""
    x = _prep_face(rgb01).to(device)
    with torch.no_grad():
        out = encoder(x)
    exp = out["expression_params"].detach().cpu().float().numpy().reshape(-1)
    jaw = out["jaw_params"].detach().cpu().float().numpy().reshape(-1)[:3]
    # SMPL-X shares FLAME's expression space; keep the first 10 coeffs (SMPL-X default,
    # matching Multi-HMR). Higher FLAME components have no SMPL-X counterpart.
    expr10 = np.zeros(10, np.float32)
    n = min(10, exp.shape[0])
    expr10[:n] = exp[:n]
    return {"expression": expr10, "jaw_pose": jaw.astype(np.float32).copy()}
