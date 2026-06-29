"""
Umeyama similarity alignment (scale + rotation + translation).

MotionAGFormer outputs 3D joints in a normalized, root-relative space; SMPL-X is
metric. We align the lifted joints onto the SMPL-X rest-pose joints (best similarity
fit on the shared joints) so the subsequent IK runs in true SMPL-X metric space.
"""

import numpy as np


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
    """
    Best similarity (s, R, t) minimizing || s*R*src_i + t - dst_i ||^2.

    src, dst: (N, 3). Returns (s: float, R: (3,3), t: (3,)).
    Reflection is corrected so R is a proper rotation (det +1).
    """
    src = np.asarray(src, np.float64)
    dst = np.asarray(dst, np.float64)
    n = src.shape[0]
    sm, dm = src.mean(0), dst.mean(0)
    sc, dc = src - sm, dst - dm
    cov = (dc.T @ sc) / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    var_src = (sc ** 2).sum() / n
    s = float(np.trace(np.diag(D) @ S) / var_src) if (with_scale and var_src > 1e-12) else 1.0
    t = dm - s * R @ sm
    return s, R.astype(np.float32), t.astype(np.float32)


def apply_similarity(s, R, t, pts: np.ndarray) -> np.ndarray:
    """Apply (s, R, t) to points (N, 3)."""
    return (s * (np.asarray(pts, np.float32) @ R.T) + t).astype(np.float32)
