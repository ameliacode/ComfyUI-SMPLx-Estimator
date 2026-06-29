"""
Gate for whole-body hand estimation (modules/wholebody/hands.py).

Two correctness guards:
 1. The 45-dim SMPL-X hand_pose finger blocks line up with the real MANO joint
    order ([index, middle, pinky, ring, thumb], 3 joints each). A wrong block
    map silently curls the wrong finger.
 2. estimate_finger_curls reads curl from 2D geometry: a straight finger -> ~0,
    a folded finger -> ~1.

Run with the ComfyUI venv python (no pytest needed):
    venv/bin/python3.10 tests/test_wholebody_hands.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.wholebody.hands import (                       # noqa: E402
    _FINGER_BLOCK, _FINGER_KPS, estimate_finger_curls, curls_to_hand_pose,
)
from modules.smplx_fit.model import load_smplx, DEFAULT_SMPLX_PARENT  # noqa: E402
from smplx.joint_names import JOINT_NAMES                   # noqa: E402


def test_finger_block_matches_mano_order():
    # SMPL-X left-hand joints 25..39 are the 15 joints of the 45-dim hand_pose.
    hand_names = [JOINT_NAMES[i].replace("left_", "") for i in range(25, 40)]
    for finger, (s, e) in _FINGER_BLOCK.items():
        joints = e - s
        assert joints == 9, f"{finger} block must be 9 dims (3 joints)"
        # every joint name in this 3-joint block must start with the finger name
        block_joints = hand_names[s // 3: e // 3]
        assert all(n.startswith(finger) for n in block_joints), \
            f"block {finger} -> {block_joints} (MANO order mismatch)"
    print(f"  [ok] finger blocks match MANO order: "
          f"{ {f: b for f, b in _FINGER_BLOCK.items()} }")


def test_curls_isolated_to_their_block():
    model = load_smplx(DEFAULT_SMPLX_PARENT, "neutral", "cpu")
    comp0 = model.np_left_hand_components[0]
    for finger, (s, e) in _FINGER_BLOCK.items():
        pose = curls_to_hand_pose({finger: 1.0}, comp0)
        nz = set(np.where(np.abs(pose) > 1e-6)[0].tolist())
        assert nz <= set(range(s, e)) and nz, f"{finger} leaked outside its block: {nz}"
    print("  [ok] per-finger curl stays inside its own 9-dim block")


def test_curl_from_2d_geometry():
    # Build a fake hand: straight index finger vs folded index finger.
    straight = np.zeros((21, 2), np.float32)
    sc = np.ones(21, np.float32)
    # index kps 5..8 along a straight line
    for n, k in enumerate(_FINGER_KPS["index"]):
        straight[k] = [n * 10.0, 0.0]
    c_straight = estimate_finger_curls(straight, sc)["index"]

    folded = straight.copy()
    # tip folds back toward the base -> short chord, long path
    folded[_FINGER_KPS["index"][3]] = [10.0, 5.0]
    c_folded = estimate_finger_curls(folded, sc)["index"]

    assert c_straight < 0.1, f"straight finger curl should be ~0, got {c_straight}"
    assert c_folded > c_straight, "folded finger must read more curl than straight"
    print(f"  [ok] curl from 2D: straight={c_straight:.2f} < folded={c_folded:.2f}")


if __name__ == "__main__":
    test_finger_block_matches_mano_order()
    test_curls_isolated_to_their_block()
    test_curl_from_2d_geometry()
    print("\nWHOLE-BODY HAND GATE PASSED")
