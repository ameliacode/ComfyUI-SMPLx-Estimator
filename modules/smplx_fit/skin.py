"""
Reduce SMPL-X LBS weights (55 joints) to the 22 editable body joints.

The editor exposes 22 body-joint handles (NUM_BODY_JOINTS). SMPL-X skins each
vertex over 55 joints (body + jaw + 2 eyes + 2x15 hand). For the browser's live
soft-skinning preview we fold every non-body joint's weight onto its nearest
body ancestor by walking the kinematic tree (hand joints -> wrist, jaw/eyes ->
head), then keep the top-K influences per vertex.

This is a *preview* skinning (weight-blended joint translation), not the
authoritative deformation — the server's IK re-solve (resolve_edit) remains the
source of truth for the emitted SMPL-X parameters.
"""

import numpy as np
import torch

from .joint_maps import NUM_BODY_JOINTS

_skin_cache: dict = {}


def body_skin_weights(model, topk: int = 4):
    """
    Return per-vertex skin weights over the 22 body joints.

    {"indices": (V,topk) int32, "weights": (V,topk) float32 (rows sum to 1)}.
    Cached per model instance (topology is fixed).
    """
    key = (id(model), topk)
    if key in _skin_cache:
        return _skin_cache[key]

    # lbs_weights / parents are model buffers; read them outside inference mode
    # so we never hold on to inference tensors.
    with torch.inference_mode(False):
        W = model.lbs_weights.detach().float().cpu().numpy()        # (V, J)
        parents = model.parents.detach().cpu().numpy().astype(int)  # (J,)

    V, J = W.shape
    Wb = W[:, :NUM_BODY_JOINTS].copy()
    for j in range(NUM_BODY_JOINTS, J):
        a = j
        while a >= NUM_BODY_JOINTS:                # walk up to a body ancestor
            a = int(parents[a])
            if a < 0:
                break
        if 0 <= a < NUM_BODY_JOINTS:
            Wb[:, a] += W[:, j]

    Wb /= np.clip(Wb.sum(1, keepdims=True), 1e-8, None)

    idx = np.argsort(-Wb, axis=1)[:, :topk]                         # (V, topk)
    w = np.take_along_axis(Wb, idx, axis=1)                         # (V, topk)
    w /= np.clip(w.sum(1, keepdims=True), 1e-8, None)               # renorm kept-k

    out = {"indices": idx.astype(np.int32), "weights": w.astype(np.float32)}
    _skin_cache[key] = out
    return out


_edit_skin_cache: dict = {}


def editable_skin_weights(model, topk: int = 6):
    """
    Per-vertex skin weights over the EDITABLE joints (body 0-21 + fingers 25-54),
    indices in SMPL-X joint-index space (0-54). Non-editable joints (jaw 22, eyes
    23/24) are folded onto their nearest editable ancestor (the head). Used by the
    viewer's live soft-skinning so BOTH body and finger drags deform the surface.

    {"indices": (V,topk) int32 in [0,54], "weights": (V,topk) float32 (rows sum 1)}.
    """
    from .joint_maps import EDITABLE_JOINTS

    key = (id(model), topk)
    if key in _edit_skin_cache:
        return _edit_skin_cache[key]

    with torch.inference_mode(False):
        W = model.lbs_weights.detach().float().cpu().numpy()        # (V, J)
        parents = model.parents.detach().cpu().numpy().astype(int)  # (J,)

    V, J = W.shape
    ed = set(EDITABLE_JOINTS)
    Wb = np.zeros((V, J), np.float32)
    Wb[:, EDITABLE_JOINTS] = W[:, EDITABLE_JOINTS]
    for j in range(J):                          # fold non-editable -> editable ancestor
        if j in ed:
            continue
        a = j
        while a >= 0 and a not in ed:
            a = int(parents[a])
        if a >= 0:
            Wb[:, a] += W[:, j]

    Wb /= np.clip(Wb.sum(1, keepdims=True), 1e-8, None)
    idx = np.argsort(-Wb, axis=1)[:, :topk]                         # (V, topk) in 0..54
    w = np.take_along_axis(Wb, idx, axis=1)
    w /= np.clip(w.sum(1, keepdims=True), 1e-8, None)

    out = {"indices": idx.astype(np.int32), "weights": w.astype(np.float32)}
    _edit_skin_cache[key] = out
    return out
