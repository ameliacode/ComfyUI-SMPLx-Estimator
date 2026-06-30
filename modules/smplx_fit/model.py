"""
SMPL-X body model + VPoser loaders for headless fitting.

Design notes (verified against the live env this session):
- ``smplx.create(model_path=DIR, model_type='smplx')`` appends ``smplx/SMPLX_<GENDER>.<ext>``
  to ``model_path``. So model_path must be the PARENT of the ``smplx/`` folder.
  ``_resolve_model_parent`` accepts a .npz file, the ``smplx`` dir, or the parent
  and normalises to what smplx.create wants.
- human_body_prior (VPoser) is loaded via sys.path injection, NOT pip-installed:
  its pyproject pins torch>=2.5/py>=3.11 which conflicts with the ComfyUI venv
  (torch 2.11 / py3.10). Importing by path avoids any dependency change that could
  break the co-resident ClickPose node. ``load_model`` returns a TUPLE
  (model, cfg) — it MUST be unpacked.
"""

import os
import logging

import torch

log = logging.getLogger(__name__)

# Default SMPL-X location: the standard ComfyUI models/smplx/ folder (resolved via
# folder_paths at runtime). Users drop SMPLX_<GENDER>.npz there. Falls back for
# headless test runs where ComfyUI isn't on the path.
try:
    import folder_paths as _fp
    _MODELS_DIR = _fp.models_dir
except Exception:
    _MODELS_DIR = os.environ.get(
        "COMFYUI_MODELS_DIR", os.path.expanduser("~/github/ComfyUI/models"))
DEFAULT_SMPLX_PARENT = os.path.join(_MODELS_DIR, "smplx")


def _resolve_model_parent(path: str) -> str:
    """
    Normalise a user-supplied SMPL-X path to the directory smplx.create expects.

    Accepts:
      - <parent>                       (already contains smplx/SMPLX_*.npz)  -> as-is
      - <parent>/smplx                 (the folder holding SMPLX_*.npz)      -> dirname
      - <parent>/smplx/SMPLX_NEUTRAL.npz (a model file)                      -> dirname(dirname)
    """
    p = os.path.abspath(os.path.expanduser(path))
    if os.path.isfile(p):  # a .npz / .pkl file -> parent of the 'smplx' folder
        return os.path.dirname(os.path.dirname(p))
    if os.path.basename(p.rstrip("/")) == "smplx":  # the smplx/ folder itself
        return os.path.dirname(p)
    return p  # assume it is already the parent that contains smplx/


_smplx_cache: dict = {}


def load_smplx(model_path: str, gender: str, device: str, num_betas: int = 10):
    """Load (and cache) a neutral/male/female SMPL-X model on the given device."""
    parent = _resolve_model_parent(model_path)
    key = (parent, gender, device, num_betas)
    if key in _smplx_cache:
        return _smplx_cache[key]

    import smplx

    expected = os.path.join(parent, "smplx", f"SMPLX_{gender.upper()}.npz")
    if not os.path.exists(expected):
        raise FileNotFoundError(
            f"SMPL-X model not found at {expected}. "
            f"Pass the directory that contains 'smplx/SMPLX_{gender.upper()}.npz' "
            f"(or the .npz file itself)."
        )

    # ComfyUI runs nodes under torch.inference_mode(); tensors created there are
    # "inference tensors" that cannot be used in autograd (fitting needs grad).
    # Build the model explicitly OUTSIDE inference mode so its buffers are normal.
    with torch.inference_mode(False):
        model = smplx.create(
            model_path=parent,
            model_type="smplx",
            gender=gender,
            use_pca=False,
            flat_hand_mean=True,
            num_betas=num_betas,
            ext="npz",
        ).to(device)
    model.eval()
    _smplx_cache[key] = model
    log.info("[smplx_fit] loaded SMPL-X %s from %s on %s", gender, expected, device)
    return model


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device
