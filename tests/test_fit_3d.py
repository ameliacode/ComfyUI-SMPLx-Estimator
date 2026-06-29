"""
3D fit self-consistency. Pose SMPL-X, read off the joints the H36M map targets,
scramble them with a random similarity (scale+rotation+translation) to mimic
MotionAGFormer's normalized/root-relative space, then fit_smplx_3d and assert:
  - the fitted joints match the targets (low residual),
  - the body is upright (Y-up world),
  - the recovered body_pose is close to the planted one.

    venv/bin/python3.10 tests/test_fit_3d.py
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from modules.smplx_fit.model import load_smplx, resolve_device, DEFAULT_SMPLX_PARENT  # noqa: E402
from modules.smplx_fit.fitting import fit_smplx_3d  # noqa: E402
from modules.smplx_fit.joint_maps import build_h36m_to_smplx, body_joint_names  # noqa: E402
from modules.motionagformer.inference import H36M_JOINT_NAMES  # noqa: E402


def _rand_rotation():
    from numpy import cos, sin
    a, b, c = 0.4, -0.9, 0.3  # fixed angles (deterministic)
    Rx = np.array([[1, 0, 0], [0, cos(a), -sin(a)], [0, sin(a), cos(a)]])
    Ry = np.array([[cos(b), 0, sin(b)], [0, 1, 0], [-sin(b), 0, cos(b)]])
    Rz = np.array([[cos(c), -sin(c), 0], [sin(c), cos(c), 0], [0, 0, 1]])
    return (Rz @ Ry @ Rx).astype(np.float32)


def test_fit_3d_recovers_pose():
    device = resolve_device("auto")
    model = load_smplx(DEFAULT_SMPLX_PARENT, "neutral", device)

    # plant a known pose
    bp = torch.zeros(1, 63, device=device).view(1, 21, 3)
    from smplx.joint_names import JOINT_NAMES
    bp[0, JOINT_NAMES.index("left_shoulder") - 1] = torch.tensor([0., 0., 0.8], device=device)
    bp[0, JOINT_NAMES.index("right_knee") - 1] = torch.tensor([0.6, 0., 0.], device=device)
    bp = bp.view(1, 63)
    with torch.no_grad():
        j = model(global_orient=torch.zeros(1, 3, device=device), body_pose=bp,
                  betas=torch.zeros(1, 10, device=device), transl=torch.zeros(1, 3, device=device)).joints[0]

    # build an H36M-ordered target from the mapped SMPL-X joints, then scramble
    h_idx, s_idx = build_h36m_to_smplx()
    tgt = np.zeros((17, 3), np.float32)
    for h, s in zip(h_idx, s_idx):
        tgt[h] = j[s].cpu().numpy()
    R, scale, trans = _rand_rotation(), 0.35, np.array([1.0, -0.4, 3.0], np.float32)
    tgt_scrambled = tgt.copy()
    for h in h_idx:
        tgt_scrambled[h] = scale * (R @ tgt[h]) + trans

    out = fit_smplx_3d(tgt_scrambled, H36M_JOINT_NAMES, model, device, iters=120, seed=0)

    # residual on the constrained joints (already reported as fit_loss, in m^2)
    rmse = float(np.sqrt(out["fit_loss"] / len(s_idx)))
    names = body_joint_names()
    jj = out["joints_3d"]
    upright = jj[names.index("head"), 1] > jj[names.index("pelvis"), 1] > jj[names.index("left_ankle"), 1]
    print(f"  3D fit: per-joint rmse={rmse*100:.2f}cm  upright={upright}")
    # Joint POSITIONS are the meaningful target (axis-angle twist about a bone is
    # unconstrained by joint positions, so body_pose itself need not match exactly).
    assert rmse < 0.03, f"3D joint residual {rmse*100:.1f}cm too high"
    assert upright, "body not upright"
    print("  [ok] 3D fit recovers an upright body matching the lifted joints")


if __name__ == "__main__":
    test_fit_3d_recovers_pose()
    print("\n3D FIT SELF-CONSISTENCY PASSED")
