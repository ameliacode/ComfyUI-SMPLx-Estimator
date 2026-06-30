"""
Hand: WiLoR — dedicated SOTA hand reconstruction -> SMPL-X hand pose.

    Load WiLoR -> WILOR_MODEL ─► Hand: WiLoR (model, image) -> SMPLX(hands)
                                          └─► feed into Body: NLF `hands_from`

Detects both hands and reconstructs MANO, mapped to SMPL-X left/right_hand_pose.
Output is a partial SMPLX dict carrying only the hand pose.
License: WiLoR weights are CC-BY-NC-ND (non-commercial).
"""

import hashlib

import numpy as np

from ..modules.wilor.estimate import estimate_hand_pose


class WiLoRHandEstimator:
    """SOTA in-the-wild hand reconstruction -> SMPL-X hand pose (feeds `hands_from`)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("WILOR_MODEL",),
                "image": ("IMAGE",),
                "conf": ("FLOAT", {"default": 0.3, "min": 0.05, "max": 0.9, "step": 0.05,
                                   "tooltip": "Hand detection confidence threshold."}),
            },
        }

    RETURN_TYPES = ("SMPLX",)
    RETURN_NAMES = ("smplx_hands",)
    FUNCTION = "estimate"
    CATEGORY = "SMPLx Estimator"

    @classmethod
    def IS_CHANGED(cls, model, image, conf):
        h = hashlib.sha256()
        h.update(np.asarray(image).tobytes())
        h.update(repr((model.get("device"), id(model.get("model")), conf)).encode())
        return h.hexdigest()

    def estimate(self, model, image, conf):
        b = model
        dev = b["device"]
        rgb01 = image[0].cpu().numpy().astype(np.float32)
        hp = estimate_hand_pose(b["model"], b["cfg"], b["detector"], rgb01, dev, conf=conf)
        sides = [k.split("_")[0] for k in hp]
        print(f"[wilor] hands detected: {sides or 'none'} on {dev}")
        # only include detected sides (consumers check `key in hands_from`)
        return ({k: v for k, v in hp.items() if v is not None},)
