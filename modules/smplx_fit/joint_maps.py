"""
Joint correspondence maps for SMPL-X fitting.

CRITICAL GUARDRAIL: every correspondence is declared BY NAME-STRING, never by a
hard-coded index. COCO-17, the (non-standard) MotionAGFormer H36M order, and
SMPL-X's 55-joint order all differ, and a left/right swap is near-symmetric — it
passes 2D reprojection and similarity alignment silently, then the fit converges
to a confidently MIRRORED pose. Indices are resolved at import time from
``smplx.joint_names.JOINT_NAMES`` and COCO names, and asserted by the test suite
(tests/test_joint_map.py) including an asymmetric-pose fixture.
"""

# COCO-17 keypoint names, in ClickPose / COCO output order (matches
# modules/clickpose/inference.py COCO_JOINT_NAMES).
COCO_JOINT_NAMES = [
    "nose",            # 0
    "left_eye",        # 1
    "right_eye",       # 2
    "left_ear",        # 3
    "right_ear",       # 4
    "left_shoulder",   # 5
    "right_shoulder",  # 6
    "left_elbow",      # 7
    "right_elbow",     # 8
    "left_wrist",      # 9
    "right_wrist",     # 10
    "left_hip",        # 11
    "right_hip",       # 12
    "left_knee",       # 13
    "right_knee",      # 14
    "left_ankle",      # 15
    "right_ankle",     # 16
]

# COCO body joint -> SMPL-X joint, BY NAME. 12 reliable position constraints.
# nose/eyes/ears are intentionally DROPPED (weight 0): the COCO nose sits ~10-15cm
# in front of the SMPL-X 'head' joint, so pinning it tilts the neck. Head/neck/
# collars/spine are left to the kinematic chain + VPoser prior.
COCO_TO_SMPLX_NAME = [
    ("left_shoulder",  "left_shoulder"),
    ("right_shoulder", "right_shoulder"),
    ("left_elbow",     "left_elbow"),
    ("right_elbow",    "right_elbow"),
    ("left_wrist",     "left_wrist"),
    ("right_wrist",    "right_wrist"),
    ("left_hip",       "left_hip"),
    ("right_hip",      "right_hip"),
    ("left_knee",      "left_knee"),
    ("right_knee",     "right_knee"),
    ("left_ankle",     "left_ankle"),
    ("right_ankle",    "right_ankle"),
]

# MotionAGFormer H36M joint -> SMPL-X joint, BY NAME (the fit now lifts to 3D and
# fits to these joints). H36M uses different names for the same anatomy
# (sacrum/left_foot/left_hand/center_head/neck_base), so pairs are explicit.
# The H36M order is NON-STANDARD (modules/motionagformer/inference.py) — sourcing
# by name (not index) is what prevents a silent left/right swap.
H36M_TO_SMPLX_NAME = [
    ("sacrum",         "pelvis"),
    ("left_hip",       "left_hip"),
    ("left_knee",      "left_knee"),
    ("left_foot",      "left_ankle"),
    ("right_hip",      "right_hip"),
    ("right_knee",     "right_knee"),
    ("right_foot",     "right_ankle"),
    ("left_shoulder",  "left_shoulder"),
    ("left_elbow",     "left_elbow"),
    ("left_hand",      "left_wrist"),
    ("right_shoulder", "right_shoulder"),
    ("right_elbow",    "right_elbow"),
    ("right_hand",     "right_wrist"),
    ("center_head",    "head"),
    ("neck_base",      "neck"),
]

# The 22 SMPL-X body joints (pelvis + 21 body) that the editor exposes as
# draggable handles. Names are resolved/validated against the real model.
NUM_BODY_JOINTS = 22

# Kinematic parent of each of the 22 body joints (verified against
# smplx model.parents[:22] this session). Used for limb drawing + the editor.
SMPLX_BODY_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]


def body_limbs():
    """(child, parent) bone pairs for the 22 body joints (skips the root)."""
    return [(i, p) for i, p in enumerate(SMPLX_BODY_PARENTS) if p >= 0]


# Joints the editor exposes as draggable handles: the 22 body joints PLUS the
# 2x15 finger joints (SMPL-X joints 25-54). Jaw (22) + eyes (23,24) are skipped
# (not draggable). Indices are SMPL-X joint indices, used verbatim as IK targets.
EDITABLE_JOINTS = list(range(NUM_BODY_JOINTS)) + list(range(25, 55))


def editable_limbs(parents):
    """
    (child, parent) bones among EDITABLE_JOINTS, given the model's parents array
    (model.parents). A finger base's parent is its wrist (a body joint), so the
    hands stay visually connected to the arms. Non-editable ancestors (jaw/eyes)
    are skipped by climbing the tree until an editable joint is reached.
    """
    ed = set(EDITABLE_JOINTS)
    limbs = []
    for j in EDITABLE_JOINTS:
        p = int(parents[j])
        while p >= 0 and p not in ed:
            p = int(parents[p])
        if p >= 0:
            limbs.append((j, p))
    return limbs


def _smplx_joint_names():
    """Return the canonical SMPL-X joint-name list (first 55 kinematic joints)."""
    from smplx.joint_names import JOINT_NAMES
    return list(JOINT_NAMES)


def build_coco_to_smplx():
    """
    Resolve the COCO-17 -> SMPL-X correspondence to index arrays.

    Returns (coco_idx, smplx_idx): parallel lists where keypoint coco_idx[k]
    constrains SMPL-X joint smplx_idx[k]. Raises if any name is unknown or if a
    left/right side mismatch is detected (the swap guard).
    """
    names = _smplx_joint_names()
    coco_idx, smplx_idx = [], []
    for coco_name, smplx_name in COCO_TO_SMPLX_NAME:
        if coco_name not in COCO_JOINT_NAMES:
            raise KeyError(f"unknown COCO joint name: {coco_name!r}")
        if smplx_name not in names:
            raise KeyError(f"unknown SMPL-X joint name: {smplx_name!r}")
        # Swap guard: a left_* keypoint must map to a left_* SMPL-X joint (and right).
        for side in ("left", "right"):
            if coco_name.startswith(side + "_") and not smplx_name.startswith(side + "_"):
                raise ValueError(
                    f"L/R swap in map: COCO {coco_name!r} -> SMPL-X {smplx_name!r}"
                )
        coco_idx.append(COCO_JOINT_NAMES.index(coco_name))
        smplx_idx.append(names.index(smplx_name))
    return coco_idx, smplx_idx


def build_h36m_to_smplx():
    """
    Resolve MotionAGFormer H36M -> SMPL-X to index arrays (h36m_idx, smplx_idx),
    BY NAME, with the same left/right swap guard as build_coco_to_smplx.
    """
    from ..motionagformer.inference import H36M_JOINT_NAMES
    names = _smplx_joint_names()
    h_idx, s_idx = [], []
    for h_name, s_name in H36M_TO_SMPLX_NAME:
        if h_name not in H36M_JOINT_NAMES:
            raise KeyError(f"unknown H36M joint name: {h_name!r}")
        if s_name not in names:
            raise KeyError(f"unknown SMPL-X joint name: {s_name!r}")
        for side in ("left", "right"):
            if h_name.startswith(side + "_") and not s_name.startswith(side + "_"):
                raise ValueError(f"L/R swap in H36M map: {h_name!r} -> {s_name!r}")
        h_idx.append(H36M_JOINT_NAMES.index(h_name))
        s_idx.append(names.index(s_name))
    return h_idx, s_idx


def body_joint_names():
    """Names of the 22 SMPL-X body joints exposed in the editor."""
    return _smplx_joint_names()[:NUM_BODY_JOINTS]
