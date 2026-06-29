"""
WiLoR hand estimator node (dedicated SOTA hands).

    IMAGE -> WiLoRHandEstimator -> SMPLX(hands) -> (NLF / Multi-HMR) hands_from

Detects both hands and reconstructs MANO, mapped to SMPL-X left/right_hand_pose.
Output is a partial SMPLX dict carrying only the hand pose; wire it into a body
estimator's `hands_from` input to graft WiLoR's hands onto the body.

License: WiLoR weights are CC-BY-NC-ND (non-commercial).
"""

import hashlib

import numpy as np
import torch

from ..modules.wilor.estimate import load_wilor, estimate_hand_pose
from ..modules.smplx_fit.model import resolve_device


class WiLoRHandEstimator:
    """SOTA in-the-wild hand reconstruction -> SMPL-X hand pose (feeds `hands_from`)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "device": (["auto", "cuda", "cpu"],),
                "conf": ("FLOAT", {"default": 0.3, "min": 0.05, "max": 0.9, "step": 0.05,
                                   "tooltip": "Hand detection confidence threshold."}),
            },
        }

    RETURN_TYPES = ("SMPLX",)
    RETURN_NAMES = ("hands",)
    FUNCTION = "estimate"
    CATEGORY = "editpose"

    @classmethod
    def IS_CHANGED(cls, image, device, conf):
        h = hashlib.sha256()
        h.update(np.asarray(image).tobytes())
        h.update(repr((device, conf)).encode())
        return h.hexdigest()

    def _run(self, run_dev, image_rgb01, conf):
        model, cfg, detector = load_wilor(run_dev)
        hp = estimate_hand_pose(model, cfg, detector, image_rgb01, run_dev, conf=conf)
        if run_dev != "cpu":
            torch.cuda.empty_cache()
        sides = [k.split("_")[0] for k in hp]
        print(f"[wilor] hands detected: {sides or 'none'} on {run_dev}")
        # only include detected sides (consumers check `key in hands_from`)
        return ({k: v for k, v in hp.items() if v is not None},)

    def estimate(self, image, device, conf):
        dev = resolve_device(device)
        rgb01 = image[0].cpu().numpy().astype(np.float32)
        try:
            return self._run(dev, rgb01, conf)
        except torch.OutOfMemoryError:
            if dev == "cpu":
                raise
            torch.cuda.empty_cache()
            print("[wilor] CUDA out of memory — retrying on CPU (slower).")
            return self._run("cpu", rgb01, conf)
