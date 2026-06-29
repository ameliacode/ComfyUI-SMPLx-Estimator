"""
NLF single-image SMPL-X estimator (robust body / motion).

    IMAGE -> NLFSMPLXEstimator -> SMPLX -> SMPLXEditor

NLF (Neural Localizer Fields) is very robust in the wild and for global body
pose, but its released model is SMPL-only — we map its body pose to SMPL-X with
NEUTRAL shape (betas=0) and FLAT hands/face. Use this when you want the most
robust body; pose the hands in the SMPL-X Editor (finger drag / grasp), or use the
Multi-HMR estimator when you want hands + expression estimated from the image.
"""

import hashlib

import numpy as np
import torch

from ..modules.nlf.estimate import load_nlf, estimate_smplx_params, DEFAULT_NLF_MODEL
from ..modules.smplx_fit.model import load_smplx, resolve_device, DEFAULT_SMPLX_PARENT
from ..modules.smplx_fit.joint_maps import body_joint_names
from ..modules.smplx_fit.render import render_maps
from .smplx_nodes import _forward_mesh, _ground, _img


def _smplx_dict(params, gender, model_path):
    z = lambda n: np.zeros(n, np.float32)  # noqa: E731
    return {
        "global_orient": params["global_orient"],
        "body_pose": params["body_pose"],
        "betas": z(10),                            # SMPL betas don't transfer -> neutral
        "transl": params["transl"],
        "left_hand_pose": z(45), "right_hand_pose": z(45),   # NLF has no hands -> flat
        "jaw_pose": z(3), "leye_pose": z(3), "reye_pose": z(3), "expression": z(10),
        "gender": gender, "model_path": model_path,
        "joint_names": body_joint_names(), "joints_3d": np.zeros((55, 3), np.float32),
        "fit_loss": 0.0,
    }


class NLFSMPLXEstimator:
    """Robust single-image body -> SMPL-X via NLF (neutral shape, flat hands)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "nlf_model_path": ("STRING", {"default": DEFAULT_NLF_MODEL}),
                "smplx_model_path": ("STRING", {"default": DEFAULT_SMPLX_PARENT}),
                "gender": (["neutral", "male", "female"],),
                "device": (["auto", "cuda", "cpu"],),
            },
        }

    RETURN_TYPES = ("SMPLX", "IMAGE")
    RETURN_NAMES = ("smplx", "preview")
    OUTPUT_NODE = True
    FUNCTION = "estimate"
    CATEGORY = "editpose"

    @classmethod
    def IS_CHANGED(cls, image, nlf_model_path, smplx_model_path, gender, device):
        h = hashlib.sha256()
        h.update(np.asarray(image).tobytes())
        h.update(repr((nlf_model_path, smplx_model_path, gender, device)).encode())
        return h.hexdigest()

    def _run(self, run_dev, image_rgb01, nlf_model_path, smplx_model_path, gender):
        model = load_nlf(nlf_model_path, run_dev)
        params = estimate_smplx_params(model, image_rgb01, run_dev)
        if run_dev != "cpu":
            torch.cuda.empty_cache()
        print(f"[nlf] estimated SMPL-X body pose from "
              f"{image_rgb01.shape[1]}x{image_rgb01.shape[0]} on {run_dev}")
        smplx_model = load_smplx(smplx_model_path, gender, run_dev)
        smplx_dict = _smplx_dict(params, gender, smplx_model_path)
        verts, faces, joints = _forward_mesh(smplx_dict, smplx_model, run_dev)
        smplx_dict["joints_3d"] = joints
        smplx_dict, verts = _ground(smplx_dict, verts)
        pose, _, _, _ = render_maps(verts, faces, run_dev, size=512, ground=False)
        return {"ui": {}, "result": (smplx_dict, _img(pose))}

    def estimate(self, image, nlf_model_path, smplx_model_path, gender, device):
        dev = resolve_device(device)
        rgb01 = image[0].cpu().numpy().astype(np.float32)        # (H,W,3) [0,1]
        try:
            return self._run(dev, rgb01, nlf_model_path, smplx_model_path, gender)
        except torch.OutOfMemoryError:
            if dev == "cpu":
                raise
            torch.cuda.empty_cache()
            print("[nlf] CUDA out of memory (GPU likely busy) — retrying on CPU (slower).")
            return self._run("cpu", rgb01, nlf_model_path, smplx_model_path, gender)
