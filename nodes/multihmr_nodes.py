"""
Multi-HMR expressive whole-body SMPL-X estimator (single node, one forward pass).

    IMAGE -> MultiHMREstimator -> SMPLX -> SMPLXEditor

Replaces the NLF(SMPL) + WholeBodyHandDetector hack: Multi-HMR jointly regresses the
full SMPL-X (body + hands + face/expression + real betas), so hands integrate with
the body (no melt) and there is no separate hand node.

NOTE: Multi-HMR weights are under a Naver NON-COMMERCIAL / research license.
First run fetches the DINOv2 architecture via torch.hub (needs internet once).
"""

import hashlib

import numpy as np
import torch

from ..modules.multihmr.estimate import (
    load_multihmr, estimate_smplx_params, DEFAULT_MULTIHMR_CKPT,
)
from ..modules.smplx_fit.model import load_smplx, resolve_device, DEFAULT_SMPLX_PARENT
from ..modules.smplx_fit.joint_maps import body_joint_names
from ..modules.smplx_fit.render import render_maps
from .smplx_nodes import _forward_mesh, _ground, _img


def _smplx_dict(params, gender, model_path):
    z = lambda n: np.zeros(n, np.float32)  # noqa: E731
    return {
        "global_orient": params["global_orient"],
        "body_pose": params["body_pose"],
        "betas": params["betas"],                  # real predicted shape
        "transl": params["transl"],
        "left_hand_pose": params["left_hand_pose"],
        "right_hand_pose": params["right_hand_pose"],
        "jaw_pose": params["jaw_pose"],
        "leye_pose": z(3), "reye_pose": z(3),
        "expression": params["expression"],
        "gender": gender, "model_path": model_path,
        "joint_names": body_joint_names(), "joints_3d": np.zeros((55, 3), np.float32),
        "fit_loss": 0.0,
    }


class MultiHMREstimator:
    """One-pass expressive whole-body SMPL-X (body + hands + face) via Multi-HMR."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "multihmr_ckpt_path": ("STRING", {"default": DEFAULT_MULTIHMR_CKPT}),
                "smplx_model_path": ("STRING", {"default": DEFAULT_SMPLX_PARENT}),
                "gender": (["neutral", "male", "female"],),
                "device": (["auto", "cuda", "cpu"],),
                "det_thresh": ("FLOAT", {"default": 0.3, "min": 0.05, "max": 0.9, "step": 0.05,
                                         "tooltip": "Person detection threshold."}),
            },
        }

    RETURN_TYPES = ("SMPLX", "IMAGE")
    RETURN_NAMES = ("smplx", "preview")
    OUTPUT_NODE = True
    FUNCTION = "estimate"
    CATEGORY = "editpose"

    @classmethod
    def IS_CHANGED(cls, image, multihmr_ckpt_path, smplx_model_path, gender, device, det_thresh):
        h = hashlib.sha256()
        h.update(np.asarray(image).tobytes())
        h.update(repr((multihmr_ckpt_path, smplx_model_path, gender, device, det_thresh)).encode())
        return h.hexdigest()

    def estimate(self, image, multihmr_ckpt_path, smplx_model_path, gender, device, det_thresh):
        dev = resolve_device(device)
        rgb01 = image[0].cpu().numpy().astype(np.float32)        # (H,W,3) [0,1]

        model, img_size = load_multihmr(multihmr_ckpt_path, smplx_model_path, dev)
        model.to(dev)
        try:
            params = estimate_smplx_params(model, img_size, rgb01, dev, det_thresh=det_thresh)
        finally:
            if dev != "cpu":                                     # free GPU for the render
                model.to("cpu")
                torch.cuda.empty_cache()
        print("[multihmr] estimated full SMPL-X (body+hands+expression) from "
              f"{rgb01.shape[1]}x{rgb01.shape[0]}")

        smplx_model = load_smplx(smplx_model_path, gender, dev)
        smplx_dict = _smplx_dict(params, gender, smplx_model_path)
        verts, faces, joints = _forward_mesh(smplx_dict, smplx_model, dev)
        smplx_dict["joints_3d"] = joints
        smplx_dict, verts = _ground(smplx_dict, verts)
        pose, _, _, _ = render_maps(verts, faces, dev, size=512, ground=False)
        return {"ui": {}, "result": (smplx_dict, _img(pose))}
