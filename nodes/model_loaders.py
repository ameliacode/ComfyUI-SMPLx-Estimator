"""
Model loader nodes for the estimators.

Separates weight loading from inference (ComfyUI convention): a Load node loads
the network once and outputs a *_MODEL bundle that the estimator node consumes.

Every loader has the same shape (ComfyUI-SAM3DBody style):
  - model_source : "local" | "huggingface"
  - model_path   : local folder holding the weight(s)            (used when local)
  - hf_token     : HuggingFace access token for gated/private repos (used when hf)
The HuggingFace repo is the per-loader REPO_ID constant. SMPL-X / MANO are
registration-walled, so set REPO_ID + hf_token for a repo you have access to,
or use model_source=local.
"""

import os

import folder_paths
import torch

from ..modules.nlf.estimate import load_nlf
from ..modules.multihmr.estimate import load_multihmr
from ..modules.wilor.estimate import load_wilor
from ..modules.smplx_fit.model import resolve_device, DEFAULT_SMPLX_PARENT

_GENDERS = ["neutral", "male", "female"]
_DEVICES = ["auto", "cuda", "cpu"]


def _local_dir(key):
    return os.path.join(folder_paths.models_dir, key)


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


def _hf_snapshot(key, repo_id, hf_token, dest, allow_patterns=None):
    """snapshot_download repo_id into dest with a ComfyUI progress bar + token."""
    from huggingface_hub import snapshot_download
    os.makedirs(dest, exist_ok=True)
    print(f"[Load {key}] downloading {repo_id} from HuggingFace -> {dest}")
    kw = {}
    tq = _comfy_tqdm()
    if tq is not None:
        kw["tqdm_class"] = tq
    if hf_token and hf_token.strip():
        kw["token"] = hf_token.strip()
    if allow_patterns:
        kw["allow_patterns"] = allow_patterns
    try:
        snapshot_download(repo_id=repo_id, local_dir=dest, **kw)
    except Exception as e:
        raise RuntimeError(
            f"[Load {key}] HuggingFace download from {repo_id!r} failed: {e}\n"
            f"Place the files under {dest}/ manually, or use model_source=local."
        ) from e


def _resolve_weight(key, repo_id, model_source, model_path, hf_token, filename):
    """Return a local path to <filename> for a single-file model.
    local: model_path is a folder (or the file) holding <filename>.
    huggingface: snapshot_download REPO_ID into models/<key>/ and locate <filename>."""
    if model_source == "huggingface":
        dest = _local_dir(key)
        target = os.path.join(dest, filename)
        if os.path.isfile(target):                     # already downloaded
            return target
        if not repo_id:
            raise ValueError(f"[Load {key}] no HuggingFace repo configured (REPO_ID is empty). "
                             f"Use model_source=local, or set REPO_ID in model_loaders.py.")
        _hf_snapshot(key, repo_id, hf_token, dest)
        if os.path.isfile(target):
            return target
        import glob
        matches = glob.glob(os.path.join(dest, "**", filename), recursive=True)
        if matches:
            return matches[0]
        raise FileNotFoundError(
            f"[Load {key}] {filename} not found in repo {repo_id!r} after download.")
    # local
    p = os.path.expanduser((model_path or "").strip())
    if os.path.isdir(p):
        p = os.path.join(p, filename)
    if not os.path.isfile(p):
        raise FileNotFoundError(
            f"[Load {key}] weight not found: {p}. Place {filename} there, or set "
            f"model_source=huggingface.")
    return p


def _check_smplx(smplx_model_path, gender):
    expected = os.path.join(os.path.expanduser(smplx_model_path), "smplx",
                            f"SMPLX_{gender.upper()}.npz")
    if not os.path.exists(expected):
        raise FileNotFoundError(
            f"SMPL-X model not found ({expected}). SMPL-X is registration-walled: register at "
            f"https://smpl-x.is.tue.mpg.de/ and place SMPLX_{gender.upper()}.npz under "
            f"<model_path>/smplx/, or use model_source=huggingface with a repo you can access."
        )


def _resolve_smplx(repo_id, model_source, gender, model_path, hf_token):
    """Return the SMPL-X parent dir (containing smplx/SMPLX_<GENDER>.npz)."""
    if model_source == "huggingface":
        parent = _local_dir("smplx")
        dest_smplx = os.path.join(parent, "smplx")
        fname = f"SMPLX_{gender.upper()}.npz"
        target = os.path.join(dest_smplx, fname)
        if not os.path.isfile(target):
            if not repo_id:
                raise ValueError("[Load SMPLx] no HuggingFace repo configured (REPO_ID is empty). "
                                 "Use model_source=local, or set REPO_ID in model_loaders.py.")
            _hf_snapshot("SMPLx", repo_id, hf_token, parent,
                         allow_patterns=["*.npz", "*.pkl", "smplx/*", "SMPLX/*", "**/SMPLX_*"])
            if not os.path.isfile(target):             # normalize repo layout -> parent/smplx/
                import glob
                import shutil
                m = glob.glob(os.path.join(parent, "**", fname), recursive=True)
                if not m:
                    raise FileNotFoundError(
                        f"[Load SMPLx] {fname} not found in HF repo {repo_id!r} after download.")
                os.makedirs(dest_smplx, exist_ok=True)
                if os.path.abspath(m[0]) != os.path.abspath(target):
                    shutil.copy(m[0], target)
    else:
        parent = os.path.expanduser((model_path or "").strip())
    _check_smplx(parent, gender)
    return parent


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


_PATH_TIP = ("local: folder holding the weight file(s).  "
             "huggingface: leave model_path; set hf_token for gated/private repos.")


class LoadSMPLX:
    """Load the SMPL-X body model -> SMPLX_MODEL, shared by the estimators + editor."""

    REPO_ID = ""   # HF repo hosting SMPLX_<GENDER>.npz (gated -> provide hf_token)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model_source": (["local", "huggingface"],),
            "gender": (_GENDERS,),
            "model_path": ("STRING", {"default": DEFAULT_SMPLX_PARENT,
                                      "tooltip": "Folder containing smplx/SMPLX_<GENDER>.npz "
                                                 "(model_source=local)."}),
            "hf_token": ("STRING", {"default": "",
                                    "tooltip": "HuggingFace access token (model_source=huggingface)."}),
        }}

    RETURN_TYPES = ("SMPLX_MODEL",)
    RETURN_NAMES = ("smplx_model",)
    FUNCTION = "load"
    CATEGORY = "editpose/loaders"

    def load(self, model_source, gender, model_path, hf_token):
        parent = _resolve_smplx(self.REPO_ID, model_source, gender, model_path, hf_token)
        return ({"model_path": parent, "gender": gender},)


class LoadNLF:
    REPO_ID = ""   # HF repo hosting nlf_l_multi_0.3.2.torchscript
    FILENAME = "nlf_l_multi_0.3.2.torchscript"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model_source": (["local", "huggingface"],),
            "model_path": ("STRING", {"default": _local_dir("nlf"), "tooltip": _PATH_TIP}),
            "hf_token": ("STRING", {"default": "", "tooltip": "HuggingFace access token."}),
            "smplx_model": ("SMPLX_MODEL",),
            "device": (_DEVICES,),
        }}

    RETURN_TYPES = ("NLF_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "editpose/loaders"

    def load(self, model_source, model_path, hf_token, smplx_model, device):
        dev = resolve_device(device)
        ckpt = _resolve_weight("nlf", self.REPO_ID, model_source, model_path, hf_token, self.FILENAME)
        net = load_nlf(ckpt, dev)
        return ({"model": net, "smplx_parent": smplx_model["model_path"],
                 "gender": smplx_model["gender"], "device": dev},)


class LoadMultiHMR:
    REPO_ID = ""   # HF repo hosting multiHMR_896_L.pt
    FILENAME = "multiHMR_896_L.pt"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model_source": (["local", "huggingface"],),
            "model_path": ("STRING", {"default": _local_dir("multihmr"), "tooltip": _PATH_TIP}),
            "hf_token": ("STRING", {"default": "", "tooltip": "HuggingFace access token."}),
            "smplx_model": ("SMPLX_MODEL",),
            "device": (_DEVICES,),
        }}

    RETURN_TYPES = ("MULTIHMR_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "editpose/loaders"

    def load(self, model_source, model_path, hf_token, smplx_model, device):
        ckpt = _resolve_weight("multihmr", self.REPO_ID, model_source, model_path, hf_token, self.FILENAME)
        parent, gender = smplx_model["model_path"], smplx_model["gender"]

        def _do(dev):
            net, img_size = load_multihmr(ckpt, parent, dev)
            return ({"model": net, "img_size": img_size, "smplx_parent": parent,
                     "gender": gender, "device": dev},)
        return _oom_fallback(resolve_device(device), _do)


class LoadWiLoR:
    REPO_ID = ""   # HF repo hosting wilor_final.ckpt + detector.pt
    CKPT = "wilor_final.ckpt"
    DETECTOR = "detector.pt"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model_source": (["local", "huggingface"],),
            "model_path": ("STRING", {"default": _local_dir("wilor"), "tooltip": _PATH_TIP}),
            "hf_token": ("STRING", {"default": "", "tooltip": "HuggingFace access token."}),
            "device": (_DEVICES,),
        }}

    RETURN_TYPES = ("WILOR_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "editpose/loaders"

    def load(self, model_source, model_path, hf_token, device):
        ckpt = _resolve_weight("wilor", self.REPO_ID, model_source, model_path, hf_token, self.CKPT)
        detp = _resolve_weight("wilor", self.REPO_ID, model_source, model_path, hf_token, self.DETECTOR)

        def _do(dev):
            net, cfg, det = load_wilor(ckpt, detp, dev)
            return ({"model": net, "cfg": cfg, "detector": det, "device": dev},)
        return _oom_fallback(resolve_device(device), _do)
