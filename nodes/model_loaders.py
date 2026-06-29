"""
Model loader nodes (Load + Download & Load) for the estimators.

Separates weight loading from inference (ComfyUI convention): a Load node loads
the network once and outputs a *_MODEL bundle that the estimator node consumes.
Weights are resolved through ComfyUI `folder_paths` (models/<key>/), never
hardcoded. The Download & Load variants fetch from the official hosts on first
use. SMPL-X / MANO are registration-walled and cannot be auto-downloaded — the
loaders raise a clear error if the SMPL-X model directory is missing.
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

# Known weights for the Download & Load variants.
_URLS = {
    "nlf": {
        "nlf_l_multi_0.3.2.torchscript":
            "https://github.com/isarandi/nlf/releases/download/v0.3.2/nlf_l_multi_0.3.2.torchscript",
    },
    "multihmr": {
        "multiHMR_896_L.pt":
            "https://download.europe.naverlabs.com/ComputerVision/MultiHMR/multiHMR_896_L.pt",
        "multiHMR_672_L.pt":
            "https://download.europe.naverlabs.com/ComputerVision/MultiHMR/multiHMR_672_L.pt",
    },
    "wilor": {
        "wilor_final.ckpt":
            "https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/wilor_final.ckpt",
        "detector.pt":
            "https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt",
    },
}


def _list(folder_key):
    try:
        return folder_paths.get_filename_list(folder_key)
    except Exception:
        return []


def _resolve(folder_key, filename):
    p = folder_paths.get_full_path(folder_key, filename)
    return p or os.path.join(folder_paths.models_dir, folder_key, filename)


def _download(folder_key, filename):
    """Fetch a known weight into models/<key>/ if missing; return its path."""
    d = os.path.join(folder_paths.models_dir, folder_key)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, filename)
    if not os.path.isfile(path):
        url = _URLS[folder_key][filename]
        print(f"[model_loaders] downloading {url}\n  -> {path}")
        import urllib.request
        urllib.request.urlretrieve(url, path)
    return path


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


# ── NLF ──────────────────────────────────────────────────────────────────────
class _NLFLoaderBase:
    RETURN_TYPES = ("NLF_MODEL",)
    RETURN_NAMES = ("nlf_model",)
    FUNCTION = "load"
    CATEGORY = "editpose/loaders"

    def _bundle(self, ckpt_path, smplx_model_path, gender, device):
        dev = resolve_device(device)
        _check_smplx(smplx_model_path, gender)
        model = load_nlf(ckpt_path, dev)
        return ({"model": model, "smplx_parent": smplx_model_path,
                 "gender": gender, "device": dev},)


class LoadNLF(_NLFLoaderBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "nlf_model": (_list("nlf"),),
            "smplx_model_path": ("STRING", {"default": DEFAULT_SMPLX_PARENT}),
            "gender": (_GENDERS,),
            "device": (_DEVICES,),
        }}

    def load(self, nlf_model, smplx_model_path, gender, device):
        return self._bundle(_resolve("nlf", nlf_model), smplx_model_path, gender, device)


class DownloadAndLoadNLF(_NLFLoaderBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": (list(_URLS["nlf"]),),
            "smplx_model_path": ("STRING", {"default": DEFAULT_SMPLX_PARENT}),
            "gender": (_GENDERS,),
            "device": (_DEVICES,),
        }}

    def load(self, model, smplx_model_path, gender, device):
        return self._bundle(_download("nlf", model), smplx_model_path, gender, device)


# ── Multi-HMR ────────────────────────────────────────────────────────────────
class _MultiHMRLoaderBase:
    RETURN_TYPES = ("MULTIHMR_MODEL",)
    RETURN_NAMES = ("multihmr_model",)
    FUNCTION = "load"
    CATEGORY = "editpose/loaders"

    def _bundle(self, ckpt_path, smplx_model_path, gender, device):
        dev = resolve_device(device)
        _check_smplx(smplx_model_path, gender)
        model, img_size = load_multihmr(ckpt_path, smplx_model_path, dev)
        return ({"model": model, "img_size": img_size, "smplx_parent": smplx_model_path,
                 "gender": gender, "device": dev},)


class LoadMultiHMR(_MultiHMRLoaderBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "multihmr_model": (_list("multihmr"),),
            "smplx_model_path": ("STRING", {"default": DEFAULT_SMPLX_PARENT}),
            "gender": (_GENDERS,),
            "device": (_DEVICES,),
        }}

    def load(self, multihmr_model, smplx_model_path, gender, device):
        return self._bundle(_resolve("multihmr", multihmr_model), smplx_model_path, gender, device)


class DownloadAndLoadMultiHMR(_MultiHMRLoaderBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": (list(_URLS["multihmr"]),),
            "smplx_model_path": ("STRING", {"default": DEFAULT_SMPLX_PARENT}),
            "gender": (_GENDERS,),
            "device": (_DEVICES,),
        }}

    def load(self, model, smplx_model_path, gender, device):
        return self._bundle(_download("multihmr", model), smplx_model_path, gender, device)


# ── WiLoR ────────────────────────────────────────────────────────────────────
class _WiLoRLoaderBase:
    RETURN_TYPES = ("WILOR_MODEL",)
    RETURN_NAMES = ("wilor_model",)
    FUNCTION = "load"
    CATEGORY = "editpose/loaders"

    def _bundle(self, ckpt_path, detector_path, device):
        dev = resolve_device(device)
        model, cfg, detector = load_wilor(ckpt_path, detector_path, dev)
        return ({"model": model, "cfg": cfg, "detector": detector, "device": dev},)


class LoadWiLoR(_WiLoRLoaderBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "wilor_model": (_list("wilor"),),
            "detector": (_list("wilor"),),
            "device": (_DEVICES,),
        }}

    def load(self, wilor_model, detector, device):
        return self._bundle(_resolve("wilor", wilor_model), _resolve("wilor", detector), device)


class DownloadAndLoadWiLoR(_WiLoRLoaderBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"device": (_DEVICES,)}}

    def load(self, device):
        ckpt = _download("wilor", "wilor_final.ckpt")
        det = _download("wilor", "detector.pt")
        return self._bundle(ckpt, det, device)
