"""
SMPLXEditor IK re-solve. Dragging a body joint re-solves body_pose
(``resolve_edit``); dragging a finger joint re-solves that hand's hand_pose
(``resolve_hand_edit``). Everything else is frozen, so edits stay localized.
Runs outside ComfyUI's inference_mode so autograd works; deterministic given a
fixed seed + L-BFGS iteration count.
"""

import numpy as np
import torch


def resolve_edit(smplx_dict, corrections, model, device, *,
                 iters=60, seed=0, target_w=2000.0, stay_w=1.0):
    """IK re-solve wrapper. Runs outside ComfyUI's inference_mode so autograd works."""
    if not corrections:
        return smplx_dict
    with torch.inference_mode(False), torch.enable_grad():
        return _resolve_edit_impl(smplx_dict, corrections, model, device,
                                  iters=iters, seed=seed, target_w=target_w, stay_w=stay_w)


def _resolve_edit_impl(smplx_dict, corrections, model, device, *,
                       iters=60, seed=0, target_w=2000.0, stay_w=1.0):
    """
    IK re-solve for the editor: drag body joints to 3D targets by changing ONLY
    body_pose (raw). global_orient, transl and betas stay FROZEN, so a dragged joint
    maps to the body articulation alone and edits stay localized (target is
    approached, not guaranteed).

    corrections: {str(joint_idx 0..21): [x, y, z]} in SMPL-X world-metric coords.
    Returns an updated SMPLX dict (body_pose + joints_3d re-emitted).
    """
    torch.manual_seed(seed)

    def t(a, n):
        return torch.as_tensor(np.asarray(smplx_dict[a]), dtype=torch.float32, device=device).view(1, n)

    go, tr, be = t("global_orient", 3), t("transl", 3), t("betas", len(smplx_dict["betas"]))
    bp0 = t("body_pose", 63)

    idxs = [int(k) for k in corrections]
    targ = torch.as_tensor([corrections[k] for k in corrections], dtype=torch.float32, device=device)

    bp = bp0.clone().requires_grad_(True)
    body = lambda: bp                                             # noqa: E731
    stay = lambda: stay_w * ((bp - bp0) ** 2).sum()              # noqa: E731

    opt = torch.optim.LBFGS([bp], lr=1.0, max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        j = model(global_orient=go, body_pose=body(), betas=be, transl=tr).joints[0]
        loss = target_w * ((j[idxs] - targ) ** 2).sum() + stay()
        loss.backward()
        return loss

    opt.step(closure)

    with torch.no_grad():
        body_final = body()
        joints_3d = model(global_orient=go, body_pose=body_final, betas=be,
                          transl=tr).joints[0, :55].cpu().numpy().astype(np.float32)

    out = dict(smplx_dict)
    out["body_pose"] = body_final.detach().cpu().numpy()[0].astype(np.float32)
    out["joints_3d"] = joints_3d
    return out


def resolve_hand_edit(smplx_dict, targets, model, device, *,
                      iters=60, seed=0, target_w=2000.0, stay_w=1.0):
    """IK re-solve for dragged FINGER joints. Outside inference_mode (autograd)."""
    if not targets:
        return smplx_dict
    with torch.inference_mode(False), torch.enable_grad():
        return _resolve_hand_edit_impl(smplx_dict, targets, model, device,
                                       iters=iters, seed=seed,
                                       target_w=target_w, stay_w=stay_w)


def _resolve_hand_edit_impl(smplx_dict, targets, model, device, *,
                            iters=60, seed=0, target_w=2000.0, stay_w=1.0):
    """
    Drag finger joints to 3D targets by changing ONLY the relevant hand_pose(s).
    Everything else (body_pose, global_orient, transl, betas) stays FROZEN, so a
    dragged finger maps to hand articulation alone and stays localized.

    targets: {str(joint_idx 25..54): [x,y,z]} in SMPL-X world-metric coords
             (25-39 left hand, 40-54 right hand).
    """
    torch.manual_seed(seed)

    def t(a, n):
        return torch.as_tensor(np.asarray(smplx_dict[a]), dtype=torch.float32,
                               device=device).view(1, n)

    go, tr, be = t("global_orient", 3), t("transl", 3), t("betas", len(smplx_dict["betas"]))
    bp = t("body_pose", 63)
    jw, le, re_, ex = (t("jaw_pose", 3), t("leye_pose", 3),
                       t("reye_pose", 3), t("expression", 10))
    lh0, rh0 = t("left_hand_pose", 45), t("right_hand_pose", 45)

    idxs = [int(k) for k in targets]
    targ = torch.as_tensor([targets[k] for k in targets], dtype=torch.float32, device=device)
    left_active = any(25 <= i < 40 for i in idxs)
    right_active = any(40 <= i < 55 for i in idxs)

    lh = lh0.clone()
    rh = rh0.clone()
    params = []
    if left_active:
        lh.requires_grad_(True)
        params.append(lh)
    if right_active:
        rh.requires_grad_(True)
        params.append(rh)
    if not params:
        return smplx_dict

    opt = torch.optim.LBFGS(params, lr=1.0, max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        j = model(global_orient=go, body_pose=bp, betas=be, transl=tr,
                  left_hand_pose=lh, right_hand_pose=rh, jaw_pose=jw,
                  leye_pose=le, reye_pose=re_, expression=ex).joints[0]
        loss = target_w * ((j[idxs] - targ) ** 2).sum()
        if left_active:
            loss = loss + stay_w * ((lh - lh0) ** 2).sum()
        if right_active:
            loss = loss + stay_w * ((rh - rh0) ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)

    out = dict(smplx_dict)
    if left_active:
        out["left_hand_pose"] = lh.detach().cpu().numpy()[0].astype(np.float32)
    if right_active:
        out["right_hand_pose"] = rh.detach().cpu().numpy()[0].astype(np.float32)
    return out
