"""
Full Body: Multi-HMR — one-pass expressive whole-body SMPL-X.

    Load Multi-HMR -> MULTIHMR_MODEL ─► Full Body: Multi-HMR (model, image) -> SMPLX

Multi-HMR jointly regresses the full SMPL-X (body + hands + face/expression + real
betas) in a single forward pass, so hands integrate with the body (no melt).

NOTE: Multi-HMR weights are Naver NON-COMMERCIAL / research.
"""

import hashlib

import numpy as np

from ..modules.multihmr.estimate import estimate_smplx_params
from ..modules.smplx_fit.model import load_smplx
from ..modules.smplx_fit.joint_maps import body_joint_names
from ..modules.smplx_fit.render import render_maps
from .smplx_nodes import _forward_mesh, _ground, _img


def _smplx_dict(params, gender, model_path):
    z = lambda n: np.zeros(n, np.float32)  # noqa: E731
    return {
        "global_orient": params["global_orient"], "body_pose": params["body_pose"],
        "betas": params["betas"], "transl": params["transl"],
        "left_hand_pose": params["left_hand_pose"], "right_hand_pose": params["right_hand_pose"],
        "jaw_pose": params["jaw_pose"], "leye_pose": z(3), "reye_pose": z(3),
        "expression": params["expression"],
        "gender": gender, "model_path": model_path,
        "joint_names": body_joint_names(), "joints_3d": np.zeros((55, 3), np.float32),
        "fit_loss": 0.0,
    }


class MultiHMREstimator:
    """One-pass expressive whole-body SMPL-X via a loaded Multi-HMR model."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MULTIHMR_MODEL",),
                "image": ("IMAGE",),
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
    def IS_CHANGED(cls, model, image, det_thresh):
        h = hashlib.sha256()
        h.update(np.asarray(image).tobytes())
        h.update(repr((model.get("smplx_parent"), model.get("gender"),
                       model.get("device"), id(model.get("model")),
                       det_thresh)).encode())
        return h.hexdigest()

    def estimate(self, model, image, det_thresh):
        b = model
        dev = b["device"]
        rgb01 = image[0].cpu().numpy().astype(np.float32)
        params = estimate_smplx_params(b["model"], b["img_size"], rgb01, dev, det_thresh=det_thresh)
        print(f"[multihmr] estimated full SMPL-X (body+hands+expression) on {dev}")
        smplx_model = load_smplx(b["smplx_parent"], b["gender"], dev)
        smplx_dict = _smplx_dict(params, b["gender"], b["smplx_parent"])
        verts, faces, joints = _forward_mesh(smplx_dict, smplx_model, dev)
        smplx_dict["joints_3d"] = joints
        smplx_dict, verts = _ground(smplx_dict, verts)
        pose, _, _, _ = render_maps(verts, faces, dev, size=512, ground=False)
        return {"ui": {}, "result": (smplx_dict, _img(pose))}
