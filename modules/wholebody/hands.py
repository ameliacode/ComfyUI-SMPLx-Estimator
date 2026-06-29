"""
Whole-body hand keypoints -> SMPL-X hand pose (camera-free, per-finger curl).

The 21 COCO-WholeBody hand keypoints (OpenPose hand order) give us each finger's
2D polyline. We estimate a per-finger CURL in [0,1] from how straight that
polyline is (straight => 0, folded into a fist => 1), then drive the matching
SMPL-X finger joints along the MANO PCA curl direction (component 0, the same
open<->close axis used by the manual grasp). This captures the dominant, in-plane
finger DoF without needing a camera; it does NOT recover abduction / thumb
opposition / out-of-plane pointing (that would need reprojection IK).

COCO-WholeBody hand (21), OpenPose order:
    0 wrist
    1-4 thumb, 5-8 index, 9-12 middle, 13-16 ring, 17-20 pinky  (mcp..tip)
"""

import numpy as np

# Per-finger keypoint indices into the 21-point hand (4 points: base..tip).
_FINGER_KPS = {
    "thumb":  [1, 2, 3, 4],
    "index":  [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring":   [13, 14, 15, 16],
    "pinky":  [17, 18, 19, 20],
}

# SMPL-X hand_pose (45 = 15 joints x 3) block per finger. MANO joint order is
# [index, middle, pinky, ring, thumb], 3 joints each (verified against
# smplx JOINT_NAMES for the left-hand joints; see tests).
_FINGER_BLOCK = {
    "index":  (0, 9),
    "middle": (9, 18),
    "pinky":  (18, 27),
    "ring":   (27, 36),
    "thumb":  (36, 45),
}

# Straightness (chord/path) calibration: an open relaxed finger is ~0.97,
# a fully folded finger ~0.45. Maps to curl 0..1.
_STRAIGHT_OPEN = 0.97
_STRAIGHT_FIST = 0.45
_MIN_SCORE = 0.30          # below this the finger is treated as unobserved

# Must match nodes.smplx_nodes.GRASP_SCALE (component-0 multiplier for a full fist).
GRASP_SCALE = 4.0


def split_hands(kps133, scores133):
    """Slice the 133 whole-body keypoints into left/right 21-point hands."""
    from .vitpose import LEFT_HAND, RIGHT_HAND
    return {
        "left": (np.asarray(kps133[LEFT_HAND], np.float32),
                 np.asarray(scores133[LEFT_HAND], np.float32)),
        "right": (np.asarray(kps133[RIGHT_HAND], np.float32),
                  np.asarray(scores133[RIGHT_HAND], np.float32)),
    }


def estimate_finger_curls(hand_kps, hand_scores):
    """
    Per-finger curl in [0,1] from one hand's 21 2D keypoints.

    Returns {finger: curl} for the 5 fingers; a finger whose keypoints are below
    confidence is omitted (caller leaves that finger flat).
    """
    curls = {}
    for finger, idx in _FINGER_KPS.items():
        pts = hand_kps[idx]
        sc = hand_scores[idx]
        if float(sc.min()) < _MIN_SCORE:
            continue
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)   # base->..->tip
        path = float(seg.sum())
        chord = float(np.linalg.norm(pts[-1] - pts[0]))
        if path < 1e-3:
            continue
        straightness = chord / path
        curl = (_STRAIGHT_OPEN - straightness) / (_STRAIGHT_OPEN - _STRAIGHT_FIST)
        curls[finger] = float(np.clip(curl, 0.0, 1.0))
    return curls


def curls_to_hand_pose(curls, comp0):
    """
    Per-finger curls -> 45-dim SMPL-X hand_pose.

    comp0: the hand's MANO PCA component 0 (model.np_{left,right}_hand_components[0]).
    Each finger's 9-dim block is set to curl * GRASP_SCALE * comp0[block], so the
    finger follows the natural curl direction by its estimated amount.
    """
    pose = np.zeros(45, np.float32)
    c0 = np.asarray(comp0, np.float32)
    for finger, curl in curls.items():
        s, e = _FINGER_BLOCK[finger]
        pose[s:e] = curl * GRASP_SCALE * c0[s:e]
    return pose
