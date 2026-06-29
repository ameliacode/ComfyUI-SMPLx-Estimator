"""
Body: NLF — robust single-image body -> SMPL-X.

    Load NLF -> NLF_MODEL ─► Body: NLF (model, image) -> SMPLX -> SMPL-X Editor

Robust body / global pose. NLF's released model is SMPL-only, so we map its body
pose to SMPL-X with NEUTRAL shape (betas=0) and FLAT hands/face. Pose the hands in
the editor, or feed a Hand: WiLoR result into `hands_from`.
"""

import hashlib

import numpy as np

from ..modules.nlf.estimate import estimate_smplx_params
from ..modules.smplx_fit.model import load_smplx
from ..modules.smplx_fit.joint_maps import body_joint_names
from ..modules.smplx_fit.render import render_maps
from .smplx_nodes import _forward_mesh, _ground, _img


def _smplx_dict(params, gender, model_path):
    z = lambda n: np.zeros(n, np.float32)  # noqa: E731
    return {
        "global_orient": params["global_orient"], "body_pose": params["body_pose"],
        "betas": z(10), "transl": params["transl"],
        "left_hand_pose": z(45), "right_hand_pose": z(45),
        "jaw_pose": z(3), "leye_pose": z(3), "reye_pose": z(3), "expression": z(10),
        "gender": gender, "model_path": model_path,
        "joint_names": body_joint_names(), "joints_3d": np.zeros((55, 3), np.float32),
        "fit_loss": 0.0,
    }


class NLFSMPLXEstimator:
    """Robust body -> SMPL-X via a loaded NLF model (neutral shape, flat hands)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "nlf_model": ("NLF_MODEL",),
                "image": ("IMAGE",),
            },
            "optional": {
                "hands_from": ("SMPLX", {
                    "tooltip": "Optional SMPL-X (e.g. from Hand: WiLoR) to graft hand pose + "
                               "jaw + expression onto NLF's body (wrist-relative — no melt)."}),
            },
        }

    RETURN_TYPES = ("SMPLX", "IMAGE")
    RETURN_NAMES = ("smplx", "preview")
    OUTPUT_NODE = True
    FUNCTION = "estimate"
    CATEGORY = "editpose"

    @classmethod
    def IS_CHANGED(cls, nlf_model, image, hands_from=None):
        h = hashlib.sha256()
        h.update(np.asarray(image).tobytes())
        h.update(repr((nlf_model.get("smplx_parent"), nlf_model.get("gender"),
                       nlf_model.get("device"), id(nlf_model.get("model")))).encode())
        if hands_from:
            for k in ("left_hand_pose", "right_hand_pose", "jaw_pose", "expression"):
                if k in hands_from:
                    h.update(np.asarray(hands_from[k], np.float32).tobytes())
        return h.hexdigest()

    def estimate(self, nlf_model, image, hands_from=None):
        b = nlf_model
        dev = b["device"]
        rgb01 = image[0].cpu().numpy().astype(np.float32)
        params = estimate_smplx_params(b["model"], rgb01, dev)
        smplx_model = load_smplx(b["smplx_parent"], b["gender"], dev)
        smplx_dict = _smplx_dict(params, b["gender"], b["smplx_parent"])
        if hands_from:                                          # graft hands + expression
            for k in ("left_hand_pose", "right_hand_pose", "jaw_pose", "expression"):
                if k in hands_from:
                    smplx_dict[k] = np.asarray(hands_from[k], np.float32).copy()
        print(f"[nlf] estimated SMPL-X body on {dev}"
              + (" + grafted hands/expression" if hands_from else ""))
        verts, faces, joints = _forward_mesh(smplx_dict, smplx_model, dev)
        smplx_dict["joints_3d"] = joints
        smplx_dict, verts = _ground(smplx_dict, verts)
        pose, _, _, _ = render_maps(verts, faces, dev, size=512, ground=False)
        return {"ui": {}, "result": (smplx_dict, _img(pose))}
