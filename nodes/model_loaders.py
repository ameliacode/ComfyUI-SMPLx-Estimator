"""
Model loader nodes for the estimators.

Separates weight loading from inference (ComfyUI convention): a Load node loads
the network once and outputs a *_MODEL bundle that the estimator node consumes.
Weights are resolved through ComfyUI `folder_paths` (models/<key>/), never
hardcoded — download the weights into models/{nlf,multihmr,wilor}/ (see README).
SMPL-X / MANO are registration-walled — the loaders raise a clear error if the
SMPL-X model directory is missing.
"""

import os

import folder_paths
import torch

from ..modules.nlf.estimate import load_nlf
from ..modules.multihmr.estimate import load_multihmr
from ..modules.wilor.estimate import load_wilor, DEFAULT_WILOR_DETECTOR
from ..modules.smplx_fit.model import resolve_device, DEFAULT_SMPLX_PARENT

# ── register model folders (models/nlf, models/multihmr, models/wilor) ───────────
for _k in ("nlf", "multihmr", "wilor"):
    try:
        folder_paths.add_model_folder_path(_k, os.path.join(folder_paths.models_dir, _k))
    except Exception:
        pass


def _list(folder_key):
    try:
        return folder_paths.get_filename_list(folder_key)
    except Exception:
        return []


def _resolve(folder_key, filename):
    p = folder_paths.get_full_path(folder_key, filename)
    return p or os.path.join(folder_paths.models_dir, folder_key, filename)


def _check_smplx(smplx_model_path, gender):
    expected = os.path.join(os.path.expanduser(smplx_model_path), "smplx",
                            f"SMPLX_{gender.upper()}.npz")
    if not os.path.exists(expected):
        raise FileNotFoundError(
            f"SMPL-X model not found ({expected}). SMPL-X is registration-walled and "
            f"cannot be auto-downloaded: register at https://smpl-x.is.tue.mpg.de/ and "
            f"place SMPLX_{gender.upper()}.npz under <smplx_model_path>/smplx/."
        )


_GENDERS = ["neutral", "male", "female"]
_DEVICES = ["auto", "cuda", "cpu"]


def _oom_fallback(dev, fn):
    """Run fn(dev); on CUDA OOM (GPU busy), retry on CPU. For CPU-capable models."""
    try:
        return fn(dev)
    except torch.OutOfMemoryError:
        if dev == "cpu":
            raise
        torch.cuda.empty_cache()
        print("[model_loaders] CUDA out of memory at load — retrying on CPU (slower).")
        return fn("cpu")


class LoadNLF:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "nlf_model": (_list("nlf"),),
            "smplx_model_path": ("STRING", {"default": DEFAULT_SMPLX_PARENT}),
            "gender": (_GENDERS,),
            "device": (_DEVICES,),
        }}

    RETURN_TYPES = ("NLF_MODEL",)
    RETURN_NAMES = ("nlf_model",)
    FUNCTION = "load"
    CATEGORY = "editpose/loaders"

    def load(self, nlf_model, smplx_model_path, gender, device):
        dev = resolve_device(device)
        _check_smplx(smplx_model_path, gender)
        model = load_nlf(_resolve("nlf", nlf_model), dev)
        return ({"model": model, "smplx_parent": smplx_model_path,
                 "gender": gender, "device": dev},)


class LoadMultiHMR:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "multihmr_model": (_list("multihmr"),),
            "smplx_model_path": ("STRING", {"default": DEFAULT_SMPLX_PARENT}),
            "gender": (_GENDERS,),
            "device": (_DEVICES,),
        }}

    RETURN_TYPES = ("MULTIHMR_MODEL",)
    RETURN_NAMES = ("multihmr_model",)
    FUNCTION = "load"
    CATEGORY = "editpose/loaders"

    def load(self, multihmr_model, smplx_model_path, gender, device):
        _check_smplx(smplx_model_path, gender)
        ckpt = _resolve("multihmr", multihmr_model)

        def _do(dev):
            model, img_size = load_multihmr(ckpt, smplx_model_path, dev)
            return ({"model": model, "img_size": img_size, "smplx_parent": smplx_model_path,
                     "gender": gender, "device": dev},)
        return _oom_fallback(resolve_device(device), _do)


class LoadWiLoR:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "wilor_model": (_list("wilor"),),
            "detector": (_list("wilor"),),
            "device": (_DEVICES,),
        }}

    RETURN_TYPES = ("WILOR_MODEL",)
    RETURN_NAMES = ("wilor_model",)
    FUNCTION = "load"
    CATEGORY = "editpose/loaders"

    def load(self, wilor_model, detector, device):
        ckpt = _resolve("wilor", wilor_model)
        detp = _resolve("wilor", detector)

        def _do(dev):
            model, cfg, det = load_wilor(ckpt, detp, dev)
            return ({"model": model, "cfg": cfg, "detector": det, "device": dev},)
        return _oom_fallback(resolve_device(device), _do)
