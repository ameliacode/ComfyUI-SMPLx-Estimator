"""
Whole-body hand detector node.

    IMAGE (+ optional ClickPose POSE_KEYPOINTS for the person box)
        -> WholeBodyHandDetector -> HAND_KEYPOINTS (+ preview IMAGE)

Runs ViTPose whole-body (ONNX) and keeps the 2x21 hand keypoints, plus a
per-finger curl estimate. HAND_KEYPOINTS feeds SMPLXFit, which turns the curls
into the SMPL-X hand pose. The body path stays on ClickPose.

HAND_KEYPOINTS dict:
    {"left": (21,2), "left_scores": (21,), "left_curls": {finger: 0..1},
     "right": ...,                                       "image_size": (H,W)}
"""

import hashlib

import cv2
import numpy as np
import torch

from ..modules.wholebody.vitpose import detect_wholebody, DEFAULT_VITPOSE_ONNX
from ..modules.wholebody.hands import split_hands, estimate_finger_curls

# OpenPose-order hand bones for the preview overlay.
_HAND_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (0, 9), (9, 10), (10, 11), (11, 12),     # middle
    (0, 13), (13, 14), (14, 15), (15, 16),   # ring
    (0, 17), (17, 18), (18, 19), (19, 20),   # pinky
]


def _to_tensor(rgb_uint8: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(rgb_uint8.astype(np.float32) / 255.0).unsqueeze(0)


def _person_bbox(pose_keypoints, W, H, margin=0.35):
    """Person box [x0,y0,x1,y1] from confident COCO-17 keypoints (expanded)."""
    kp = np.asarray(pose_keypoints["keypoints"], np.float32)
    sc = np.asarray(pose_keypoints["scores"], np.float32)
    m = sc > 0.3
    if int(m.sum()) < 3:
        return [0, 0, W, H]
    pts = kp[m]
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    bw, bh = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
    return [max(0.0, x0 - bw * margin), max(0.0, y0 - bh * margin),
            min(float(W), x1 + bw * margin), min(float(H), y1 + bh * margin)]


class WholeBodyHandDetector:
    """Detect 2x21 hand keypoints (ViTPose whole-body) + per-finger curls."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "vitpose_onnx_path": ("STRING", {"default": DEFAULT_VITPOSE_ONNX}),
                "device": (["cpu", "cuda"], {
                    "tooltip": "ONNX execution device. 'cuda' needs onnxruntime-gpu with "
                               "matching CUDA libs; falls back to CPU otherwise."}),
            },
            "optional": {
                "pose_keypoints": ("POSE_KEYPOINTS", {
                    "tooltip": "ClickPose output — used to crop a tight person box. "
                               "Omit to run on the whole image."}),
                "bbox": ("BBOX", {
                    "tooltip": "Person box [x0,y0,x1,y1] (e.g. from NLF SMPL-X Estimator) "
                               "for a tight crop. Takes priority over pose_keypoints."}),
            },
        }

    RETURN_TYPES = ("HAND_KEYPOINTS", "IMAGE")
    RETURN_NAMES = ("hand_keypoints", "preview")
    OUTPUT_NODE = True
    FUNCTION = "detect"
    CATEGORY = "editpose"

    @classmethod
    def IS_CHANGED(cls, image, vitpose_onnx_path, device, pose_keypoints=None, bbox=None):
        h = hashlib.sha256()
        h.update(np.asarray(image).tobytes())
        h.update(repr((vitpose_onnx_path, device, bbox)).encode())
        if pose_keypoints is not None:
            h.update(np.asarray(pose_keypoints["keypoints"], np.float32).tobytes())
        return h.hexdigest()

    def detect(self, image, vitpose_onnx_path, device, pose_keypoints=None, bbox=None):
        rgb01 = image[0].cpu().numpy().astype(np.float32)         # (H,W,3) [0,1]
        H, W = rgb01.shape[:2]
        if bbox is not None and len(bbox) == 4:                   # explicit box wins
            bbox = [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
        elif pose_keypoints is not None:
            bbox = _person_bbox(pose_keypoints, W, H)
        else:
            bbox = [0, 0, W, H]

        kps, sc = detect_wholebody(rgb01, bbox_xyxy=bbox, onnx_path=vitpose_onnx_path,
                                   device=device)
        hands = split_hands(kps, sc)
        lk, ls = hands["left"]
        rk, rs = hands["right"]
        lc = estimate_finger_curls(lk, ls)
        rc = estimate_finger_curls(rk, rs)
        print(f"[wholebody] L hand conf={float(ls.max()):.2f} curls={ {k: round(v,2) for k,v in lc.items()} } "
              f"| R hand conf={float(rs.max()):.2f} curls={ {k: round(v,2) for k,v in rc.items()} }")

        hand_keypoints = {
            "left": lk, "left_scores": ls, "left_curls": lc,
            "right": rk, "right_scores": rs, "right_curls": rc,
            "image_size": (H, W),
        }

        vis = cv2.cvtColor((rgb01 * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        for k, s, col in ((lk, ls, (255, 120, 0)), (rk, rs, (0, 120, 255))):
            for a, b in _HAND_BONES:
                if s[a] > 0.2 and s[b] > 0.2:
                    cv2.line(vis, tuple(k[a].astype(int)), tuple(k[b].astype(int)), col, 2, cv2.LINE_AA)
            for (x, y), c in zip(k, s):
                if c > 0.2:
                    cv2.circle(vis, (int(x), int(y)), 3, (255, 255, 255), -1, cv2.LINE_AA)
        preview = _to_tensor(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        return {"ui": {}, "result": (hand_keypoints, preview)}
