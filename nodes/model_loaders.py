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


def _comfy_tqdm():
    """tqdm subclass that mirrors HuggingFace download progress into ComfyUI's
    progress bar (ComfyUI-SAM3DBody style)."""
    try:
        import comfy.utils
        import tqdm as _tqdm_mod
    except ImportError:
        return None
    holder = {"pbar": None, "total": 0, "done": 0}

    class _T(_tqdm_mod.tqdm):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            if self.total and self.total > 0 and holder["pbar"] is None:
                holder["total"] = self.total
                holder["done"] = 0
                holder["pbar"] = comfy.utils.ProgressBar(self.total)

        def update(self, n=1):
            ret = super().update(n)
            if n and holder["pbar"] and holder["total"] > 0:
                holder["done"] = min(holder["done"] + n, holder["total"])
                holder["pbar"].update_absolute(holder["done"], holder["total"])
            return ret
    return _T


def _hf_download_smplx(repo_id, gender):
    """Download a SMPL-X repo from HuggingFace into models/smplx/ (snapshot_download,
    like ComfyUI-SAM3DBody) and return the parent dir so load_smplx finds
    parent/smplx/SMPLX_<GENDER>.npz. Normalizes layout + skips if already present."""
    import glob
    import shutil
    parent = os.path.join(folder_paths.models_dir, "smplx")
    dest_smplx = os.path.join(parent, "smplx")
    fname = f"SMPLX_{gender.upper()}.npz"
    target = os.path.join(dest_smplx, fname)
    if os.path.isfile(target):                      # already downloaded
        return parent

    try:
        from huggingface_hub import snapshot_download
        os.makedirs(parent, exist_ok=True)
        print(f"[Load SMPLx] downloading {repo_id} from HuggingFace -> {parent}")
        kw = {}
        tq = _comfy_tqdm()           # show download progress in ComfyUI's UI
        if tq is not None:
            kw["tqdm_class"] = tq
        snapshot_download(repo_id=repo_id, local_dir=parent,
                          allow_patterns=["*.npz", "*.pkl", "smplx/*", "SMPLX/*", "**/SMPLX_*"],
                          **kw)
    except Exception as e:
        raise RuntimeError(
            f"[Load SMPLx] HuggingFace download from {repo_id!r} failed: {e}\n"
            f"Manually place {fname} at {dest_smplx}/ (e.g. from "
            f"https://huggingface.co/{repo_id} or https://smpl-x.is.tue.mpg.de/)."
        ) from e

    if os.path.isfile(target):
        return parent
    # repo layout varies — locate the npz and normalize it into parent/smplx/
    matches = glob.glob(os.path.join(parent, "**", fname), recursive=True)
    if not matches:
        raise FileNotFoundError(
            f"[Load SMPLx] {fname} not found in HF repo {repo_id!r} after download.")
    os.makedirs(dest_smplx, exist_ok=True)
    if os.path.abspath(matches[0]) != os.path.abspath(target):
        shutil.copy(matches[0], target)
    return parent


class LoadSMPLX:
    """Load the SMPL-X body model (local folder or HuggingFace) -> SMPLX_MODEL,
    shared by the estimators (NLF/Multi-HMR) and the editor."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model_source": (["local", "huggingface"],),
            "gender": (_GENDERS,),
            "path_or_repo": ("STRING", {"default": DEFAULT_SMPLX_PARENT,
                                        "tooltip": "local: folder containing "
                                                   "smplx/SMPLX_<GENDER>.npz.  "
                                                   "huggingface: repo id hosting the SMPL-X .npz."}),
        }}

    RETURN_TYPES = ("SMPLX_MODEL",)
    RETURN_NAMES = ("smplx_model",)
    FUNCTION = "load"
    CATEGORY = "editpose/loaders"

    def load(self, model_source, gender, path_or_repo):
        src = path_or_repo.strip()
        if model_source == "huggingface":
            if not src:
                raise ValueError("model_source=huggingface but path_or_repo is empty — enter a repo id.")
            parent = _hf_download_smplx(src, gender)
        else:
            parent = os.path.expanduser(src)
        _check_smplx(parent, gender)
        return ({"model_path": parent, "gender": gender},)


class LoadNLF:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": (_list("nlf"),),
            "smplx_model": ("SMPLX_MODEL",),
            "device": (_DEVICES,),
        }}

    RETURN_TYPES = ("NLF_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "editpose/loaders"

    def load(self, model, smplx_model, device):
        dev = resolve_device(device)
        net = load_nlf(_resolve("nlf", model), dev)
        return ({"model": net, "smplx_parent": smplx_model["model_path"],
                 "gender": smplx_model["gender"], "device": dev},)


class LoadMultiHMR:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": (_list("multihmr"),),
            "smplx_model": ("SMPLX_MODEL",),
            "device": (_DEVICES,),
        }}

    RETURN_TYPES = ("MULTIHMR_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "editpose/loaders"

    def load(self, model, smplx_model, device):
        ckpt = _resolve("multihmr", model)
        parent, gender = smplx_model["model_path"], smplx_model["gender"]

        def _do(dev):
            net, img_size = load_multihmr(ckpt, parent, dev)
            return ({"model": net, "img_size": img_size, "smplx_parent": parent,
                     "gender": gender, "device": dev},)
        return _oom_fallback(resolve_device(device), _do)


class LoadWiLoR:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": (_list("wilor"),),
            "detector": (_list("wilor"),),
            "device": (_DEVICES,),
        }}

    RETURN_TYPES = ("WILOR_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "editpose/loaders"

    def load(self, model, detector, device):
        ckpt = _resolve("wilor", model)
        detp = _resolve("wilor", detector)

        def _do(dev):
            model, cfg, det = load_wilor(ckpt, detp, dev)
            return ({"model": model, "cfg": cfg, "detector": det, "device": dev},)
        return _oom_fallback(resolve_device(device), _do)
