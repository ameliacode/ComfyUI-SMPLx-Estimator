"""
HARD GATE for the COCO-17 -> SMPL-X joint map (P0).

A left/right swap in the map is near-symmetric: it passes 2D reprojection and
similarity alignment, then the fit converges to a confidently mirrored pose. We
guard it two ways:
  1. name-level  — every left_* keypoint maps to a left_* SMPL-X joint (and right).
  2. geometric   — pose ONLY the left arm in a real SMPL-X forward pass and assert
                   the joint the map calls 'left_wrist' is the one that actually
                   moved (and the 'right_wrist' barely moved). An index swap fails
                   this even though the name check would pass.

Runnable as a plain script (no pytest dependency in the ComfyUI venv):
    venv/bin/python3.10 tests/test_joint_map.py
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from modules.smplx_fit.joint_maps import (  # noqa: E402
    COCO_JOINT_NAMES,
    COCO_TO_SMPLX_NAME,
    build_coco_to_smplx,
    body_joint_names,
)
from modules.smplx_fit.model import load_smplx, DEFAULT_SMPLX_PARENT  # noqa: E402


def test_name_pairing_no_swap():
    """Name-level: left maps to left, right to right; all names valid; nose dropped."""
    from smplx.joint_names import JOINT_NAMES

    for coco_name, smplx_name in COCO_TO_SMPLX_NAME:
        assert coco_name in COCO_JOINT_NAMES, coco_name
        assert smplx_name in JOINT_NAMES, smplx_name
        for side in ("left", "right"):
            if coco_name.startswith(side):
                assert smplx_name.startswith(side), f"SWAP {coco_name}->{smplx_name}"
    # nose/eyes/ears must NOT be in the constraint set (poor anatomical targets)
    constrained = {c for c, _ in COCO_TO_SMPLX_NAME}
    assert "nose" not in constrained and "left_eye" not in constrained
    print("  [ok] name pairing: 12 constraints, no L/R swap, face joints dropped")


def test_build_indices():
    coco_idx, smplx_idx = build_coco_to_smplx()
    assert len(coco_idx) == len(smplx_idx) == 12
    assert len(set(smplx_idx)) == 12, "duplicate SMPL-X targets"
    assert body_joint_names()[0] == "pelvis"
    print(f"  [ok] indices resolved: coco={coco_idx} smplx={smplx_idx}")


def test_geometric_no_swap():
    """
    Pose ONLY the left shoulder so the left arm swings; assert the map's
    'left_wrist' joint moved far more than its 'right_wrist' joint.
    """
    from smplx.joint_names import JOINT_NAMES

    model = load_smplx(DEFAULT_SMPLX_PARENT, "neutral", "cpu")
    coco_idx, smplx_idx = build_coco_to_smplx()
    name_to_smplx = dict(zip([c for c, _ in COCO_TO_SMPLX_NAME], smplx_idx))

    rest = model(body_pose=torch.zeros(1, 63), global_orient=torch.zeros(1, 3),
                 betas=torch.zeros(1, 10), transl=torch.zeros(1, 3)).joints[0]

    # body_pose holds joints 1..21; left_shoulder is SMPL-X joint 16 -> body_pose row 15
    bp = torch.zeros(1, 63).view(1, 21, 3)
    ls_row = JOINT_NAMES.index("left_shoulder") - 1  # -1: pelvis(0) is global_orient
    bp[0, ls_row] = torch.tensor([0.0, 0.0, 1.2])    # big rotation of the left arm
    posed = model(body_pose=bp.view(1, 63), global_orient=torch.zeros(1, 3),
                  betas=torch.zeros(1, 10), transl=torch.zeros(1, 3)).joints[0]

    lw = name_to_smplx["left_wrist"]
    rw = name_to_smplx["right_wrist"]
    moved_left = (posed[lw] - rest[lw]).norm().item()
    moved_right = (posed[rw] - rest[rw]).norm().item()
    print(f"  left_wrist moved {moved_left:.3f}m, right_wrist moved {moved_right:.3f}m")
    assert moved_left > 0.10, "left arm did not move — wrong joint indexing"
    assert moved_left > 5 * max(moved_right, 1e-6), "L/R SWAP: right wrist moved too"
    print("  [ok] geometric: posing the left arm moves the map's left_wrist (no swap)")


if __name__ == "__main__":
    test_name_pairing_no_swap()
    test_build_indices()
    test_geometric_no_swap()
    print("\nALL JOINT-MAP GATE TESTS PASSED")
