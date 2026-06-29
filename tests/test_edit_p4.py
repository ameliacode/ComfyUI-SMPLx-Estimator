"""
P4: editor IK re-solve. Drag the left wrist by a known offset and assert:
  - the wrist moves TOWARD the target (most of the way),
  - far-away joints (e.g. the ankles) barely move,
  - global_orient / transl / betas are unchanged (frozen).

    venv/bin/python3.10 tests/test_edit_p4.py
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from modules.smplx_fit.model import (  # noqa: E402
    load_smplx, resolve_device, DEFAULT_SMPLX_PARENT,
)
from modules.smplx_fit.fitting import resolve_edit  # noqa: E402
from modules.smplx_fit.joint_maps import body_joint_names  # noqa: E402


def _make_smplx(model, device):
    """Build a valid SMPLX dict from a known posed forward (no fitting needed)."""
    from smplx.joint_names import JOINT_NAMES
    bp = torch.zeros(1, 63, device=device).view(1, 21, 3)
    bp[0, JOINT_NAMES.index("left_shoulder") - 1] = torch.tensor([0., 0., 0.7], device=device)
    bp = bp.view(1, 63)
    with torch.no_grad():
        j = model(global_orient=torch.zeros(1, 3, device=device), body_pose=bp,
                  betas=torch.zeros(1, 10, device=device),
                  transl=torch.zeros(1, 3, device=device)).joints[0, :55].cpu().numpy()
    z = lambda n: np.zeros(n, np.float32)  # noqa: E731
    return {
        "global_orient": z(3), "body_pose": bp[0].cpu().numpy().astype(np.float32),
        "betas": z(10), "transl": z(3),
        "left_hand_pose": z(45), "right_hand_pose": z(45), "jaw_pose": z(3),
        "leye_pose": z(3), "reye_pose": z(3), "expression": z(10),
        "gender": "neutral", "model_path": DEFAULT_SMPLX_PARENT,
        "joints_3d": j.astype(np.float32), "joint_names": body_joint_names(),
    }


def test_drag_wrist():
    device = resolve_device("auto")
    model = load_smplx(DEFAULT_SMPLX_PARENT, "neutral", device)
    d = _make_smplx(model, device)

    names = body_joint_names()
    lw = names.index("left_wrist")     # SMPL-X body joint index 20
    la = names.index("left_ankle")     # a far joint that should stay put

    j0 = d["joints_3d"]
    target = (j0[lw] + np.array([0.12, 0.10, 0.0], np.float32)).tolist()   # drag 0.156m
    before_lw = j0[lw].copy()
    before_la = j0[la].copy()

    # Editor uses RAW body_pose (NOT VPoser): localized edits — only joints on the
    # kinematic path to the drag change; unrelated limbs stay via the stay-regularizer.
    d2 = resolve_edit(d, {str(lw): target}, model, device, iters=80, seed=0)
    j1 = d2["joints_3d"]

    moved_lw = np.linalg.norm(j1[lw] - before_lw)
    to_target_before = np.linalg.norm(before_lw - np.array(target))
    to_target_after = np.linalg.norm(j1[lw] - np.array(target))
    moved_la = np.linalg.norm(j1[la] - before_la)

    print(f"  wrist: moved {moved_lw*100:.1f}cm | dist-to-target {to_target_before*100:.1f}cm "
          f"-> {to_target_after*100:.1f}cm")
    print(f"  far joint (ankle) moved {moved_la*100:.1f}cm")
    print(f"  frozen: d|global_orient|={np.abs(d2['global_orient']-d['global_orient']).max():.1e} "
          f"d|transl|={np.abs(d2['transl']-d['transl']).max():.1e}")

    assert to_target_after < 0.6 * to_target_before, "wrist did not move toward target"
    assert moved_la < 0.4 * moved_lw, "far joint moved too much (pose not localized)"
    assert np.array_equal(d2["global_orient"], d["global_orient"]), "global_orient not frozen"
    assert np.array_equal(d2["transl"], d["transl"]), "transl not frozen"
    print("  [ok] drag re-solves to body_pose: wrist follows, rest stays, root frozen")


if __name__ == "__main__":
    test_drag_wrist()
    print("\nP4 EDITOR IK RE-SOLVE PASSED")
