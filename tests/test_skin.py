"""
Gate for the editor's live soft-skinning weights (modules/smplx_fit/skin.py).

Asserts the 55->22 LBS fold is well-formed (top-k per vertex over body joints,
rows normalised) AND that hand mass really lands on the wrists (a geometric
check that the kinematic fold went to the right ancestor, not just "some" joint).
Run with the ComfyUI venv python (no pytest needed):
    venv/bin/python3.10 tests/test_skin.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.smplx_fit.model import load_smplx, DEFAULT_SMPLX_PARENT  # noqa: E402
from modules.smplx_fit.skin import body_skin_weights                  # noqa: E402
from modules.smplx_fit.joint_maps import body_joint_names, NUM_BODY_JOINTS  # noqa: E402


def main():
    model = load_smplx(DEFAULT_SMPLX_PARENT, "neutral", "cpu")
    skin = body_skin_weights(model, topk=4)
    idx, w = skin["indices"], skin["weights"]
    V = idx.shape[0]

    # ── well-formedness ───────────────────────────────────────────────────────
    assert idx.shape == w.shape == (V, 4), f"bad shape {idx.shape} {w.shape}"
    assert idx.min() >= 0 and idx.max() < NUM_BODY_JOINTS, \
        f"joint idx out of body range: [{idx.min()},{idx.max()}]"
    rs = w.sum(1)
    assert np.allclose(rs, 1.0, atol=1e-4), f"rows not normalised: [{rs.min()},{rs.max()}]"
    assert (w >= -1e-6).all(), "negative weights"
    print(f"  [ok] {V} verts, top-4 over 22 body joints, rows sum to 1")

    # ── geometric fold: hand mass -> wrists ───────────────────────────────────
    names = body_joint_names()
    lw, rw = names.index("left_wrist"), names.index("right_wrist")
    # dominant joint per vertex
    dom = idx[np.arange(V), w.argmax(1)]
    n_lw = int((dom == lw).sum())
    n_rw = int((dom == rw).sum())
    assert n_lw > 50 and n_rw > 50, \
        f"hand verts did not fold onto wrists (lw={n_lw}, rw={n_rw})"
    # no vertex should be dominated by a non-existent/face-only joint (all < 22 already)
    print(f"  [ok] hand mass folded to wrists: left_wrist={n_lw} verts, right_wrist={n_rw} verts")

    print("\nSKIN-WEIGHT REDUCTION GATE PASSED")


if __name__ == "__main__":
    main()
