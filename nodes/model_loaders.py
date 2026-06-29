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
        dev = resolve_device(device)
        _check_smplx(smplx_model_path, gender)
        model, img_size = load_multihmr(_resolve("multihmr", multihmr_model), smplx_model_path, dev)
        return ({"model": model, "img_size": img_size, "smplx_parent": smplx_model_path,
                 "gender": gender, "device": dev},)


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
        dev = resolve_device(device)
        model, cfg, det = load_wilor(_resolve("wilor", wilor_model), _resolve("wilor", detector), dev)
        return ({"model": model, "cfg": cfg, "detector": det, "device": dev},)
