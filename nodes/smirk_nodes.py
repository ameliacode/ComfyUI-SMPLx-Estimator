"""
Face: SMIRK — dedicated expression capture -> SMPL-X jaw + expression.

    Load SMIRK -> SMIRK_MODEL ─► Face: SMIRK (model, image) -> SMPLX(face)
                                          └─► feed into Body: NLF `smplx_face`

SMIRK's encoder regresses FLAME expression + jaw pose from a face crop; we map it to
SMPL-X `expression` (10) + `jaw_pose` (3). Output is a partial SMPLX dict carrying only
the face, grafted onto the body estimator (same path as Hand: WiLoR).
License: SMIRK code is MIT but drives FLAME (MPI non-commercial) — research-only.
"""

import hashlib

import numpy as np

from ..modules.smirk.estimate import estimate_face_params


class SMIRKFaceEstimator:
    """Dedicated face expression capture -> SMPL-X jaw + expression (feeds `smplx_face`)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SMIRK_MODEL",),
                "image": ("IMAGE", {"tooltip": "Face crop (roughly centred). Resized to 224x224 "
                                               "for SMIRK. Use the head region for best results."}),
            },
        }

    RETURN_TYPES = ("SMPLX",)
    RETURN_NAMES = ("smplx_face",)
    FUNCTION = "estimate"
    CATEGORY = "SMPLx Estimator"

    @classmethod
    def IS_CHANGED(cls, model, image):
        h = hashlib.sha256()
        h.update(np.asarray(image).tobytes())
        h.update(repr((model.get("device"), id(model.get("model")))).encode())
        return h.hexdigest()

    def estimate(self, model, image):
        b = model
        dev = b["device"]
        rgb01 = image[0].cpu().numpy().astype(np.float32)
        face = estimate_face_params(b["model"], rgb01, dev)
        print(f"[smirk] estimated face (expression + jaw) on {dev}")
        return (face,)
