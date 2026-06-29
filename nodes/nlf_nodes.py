"""
NLF single-image SMPL-X estimator node (SOTA one-pass alternative to the
ClickPose -> MotionAGFormer -> fit chain).

    IMAGE (+ optional HAND_KEYPOINTS) -> NLFSMPLXEstimator -> SMPLX -> SMPLXEditor

NLF predicts SMPL in one forward pass; we map its body pose onto SMPL-X (see
modules/nlf/estimate.py for the caveats: betas don't transfer -> neutral shape,
editable in SMPLXEditor; hands come from the optional detector or stay flat).
"""

import hashlib

import numpy as np
import torch

from ..modules.nlf.estimate import load_nlf, estimate_smplx_params, DEFAULT_NLF_MODEL
from ..modules.smplx_fit.model import load_smplx, resolve_device, DEFAULT_SMPLX_PARENT
from ..modules.smplx_fit.joint_maps import body_joint_names
from ..modules.smplx_fit.render import render_maps
from .smplx_nodes import _forward_mesh, _ground, _img, _apply_estimated_hands


def _blank_smplx(params, gender, model_path, num_betas=10):
    z = lambda n: np.zeros(n, np.float32)  # noqa: E731
    return {
        "global_orient": params["global_orient"],
        "body_pose": params["body_pose"],
        "betas": z(num_betas),                 # SMPL betas don't transfer -> neutral
        "transl": params["transl"],
        "left_hand_pose": z(45), "right_hand_pose": z(45),
        "jaw_pose": z(3), "leye_pose": z(3), "reye_pose": z(3), "expression": z(10),
        "gender": gender, "model_path": model_path,
        "joint_names": body_joint_names(), "joints_3d": np.zeros((55, 3), np.float32),
        "fit_loss": 0.0,
    }


class NLFSMPLXEstimator:
    """One-pass SOTA SMPL-X from a single image via NLF (body pose); shape neutral."""

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
            "optional": {
                "hand_keypoints": ("HAND_KEYPOINTS", {
                    "tooltip": "From WholeBodyHandDetector — sets the SMPL-X hand pose "
                               "(NLF has no hands). Omit for flat hands."}),
            },
        }

    RETURN_TYPES = ("SMPLX", "IMAGE", "BBOX")
    RETURN_NAMES = ("smplx", "preview", "bbox")
    OUTPUT_NODE = True
    FUNCTION = "estimate"
    CATEGORY = "editpose"

    @classmethod
    def IS_CHANGED(cls, image, nlf_model_path, smplx_model_path, gender, device,
                   hand_keypoints=None):
        h = hashlib.sha256()
        h.update(np.asarray(image).tobytes())
        h.update(repr((nlf_model_path, smplx_model_path, gender, device)).encode())
        if hand_keypoints:
            h.update(repr((hand_keypoints.get("left_curls"),
                           hand_keypoints.get("right_curls"))).encode())
        return h.hexdigest()

    def estimate(self, image, nlf_model_path, smplx_model_path, gender, device,
                 hand_keypoints=None):
        dev = resolve_device(device)
        rgb01 = image[0].cpu().numpy().astype(np.float32)        # (H,W,3) [0,1]

        nlf = load_nlf(nlf_model_path, dev)
        params = estimate_smplx_params(nlf, rgb01, dev)          # NLF -> body params (Y-up)
        print(f"[nlf] estimated SMPL-X body pose from image {rgb01.shape[1]}x{rgb01.shape[0]}")

        model = load_smplx(smplx_model_path, gender, dev)
        smplx_dict = _blank_smplx(params, gender, smplx_model_path)
        smplx_dict = _apply_estimated_hands(smplx_dict, model, hand_keypoints)  # image hands

        verts, faces, joints = _forward_mesh(smplx_dict, model, dev)
        smplx_dict["joints_3d"] = joints
        smplx_dict, verts = _ground(smplx_dict, verts)          # feet on the floor
        pose, _, _, _ = render_maps(verts, faces, dev, size=512, ground=False)
        bbox = [float(x) for x in params["bbox"]]               # [x0,y0,x1,y1] image px
        return {"ui": {}, "result": (smplx_dict, _img(pose), bbox)}
