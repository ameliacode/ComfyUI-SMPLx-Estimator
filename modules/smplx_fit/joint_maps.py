"""
SMPL-X joint structure for the editor: the body/finger joints exposed as
draggable handles, their kinematic parents, and the limb (bone) tables. Joint
names come from ``smplx.joint_names.JOINT_NAMES`` (the canonical 55).
"""

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


def body_joint_names():
    """Names of the 22 SMPL-X body joints exposed in the editor."""
    return _smplx_joint_names()[:NUM_BODY_JOINTS]
