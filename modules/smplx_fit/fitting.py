"""
Fit SMPL-X to 3D joints from a 3D pose estimator (MotionAGFormer), and the editor
IK re-solve. The fitter maps the lifted H36M joints to SMPL-X by name, Umeyama-
aligns them into SMPL-X metric space, then optimizes global_orient/body_pose/transl
to match the 3D joints (betas frozen at 0). No camera reprojection, no VPoser.
Deterministic given a fixed seed + fixed L-BFGS iteration counts.
"""

import numpy as np
import torch


def _np1(x):
    return x.detach().cpu().numpy()[0].astype(np.float32)


def fit_smplx_3d(target_joints, target_names, model, device, *,
                 iters=80, seed=0, num_betas=10, gender="neutral", model_path=""):
    """
    Fit SMPL-X to 3D joints from a 3D pose estimator (MotionAGFormer).

    target_joints: (J,3) H36M-17 3D joints (normalized/root-relative space).
    Pipeline: map H36M->SMPL-X by name -> Umeyama-align targets onto the SMPL-X
    rest pose (-> metric) -> optimize global_orient/body_pose/transl to match the
    3D joints (betas frozen at 0). No camera, no reprojection, no VPoser.
    """
    with torch.inference_mode(False), torch.enable_grad():
        return _fit_3d_impl(target_joints, target_names, model, device,
                            iters=iters, seed=seed, num_betas=num_betas,
                            gender=gender, model_path=model_path)


def _fit_3d_impl(target_joints, target_names, model, device, *,
                 iters=80, seed=0, num_betas=10, gender="neutral", model_path=""):
    from .joint_maps import build_h36m_to_smplx, body_joint_names

    torch.manual_seed(seed)
    h_idx, s_idx = build_h36m_to_smplx()
    tgt = np.asarray(target_joints, np.float32)[h_idx]          # (N,3) lifted joints

    betas = torch.zeros(1, num_betas, device=device)
    with torch.no_grad():
        rest = model(global_orient=torch.zeros(1, 3, device=device),
                     body_pose=torch.zeros(1, 63, device=device),
                     betas=betas, transl=torch.zeros(1, 3, device=device))
        rest_joints = rest.joints[0, s_idx].cpu().numpy()       # (N,3) metric rest pose

    # Align lifted joints into SMPL-X metric space, root-centered:
    #   - scale from the MEDIAN radial-distance ratio (robust to per-bone proportion
    #     differences between H36M and SMPL-X; the overall-extent Umeyama scale was
    #     biased by the rest T-pose width and distorted the fit).
    #   - rotation-only Procrustes onto the rest frame (no extent scaling).
    tc, rc = tgt - tgt[0], rest_joints - rest_joints[0]
    rr = np.linalg.norm(rc[1:], axis=1)
    tr = np.linalg.norm(tc[1:], axis=1)
    scale = float(np.median(rr / np.maximum(tr, 1e-6)))
    ts = tc * scale
    M = rc.T @ ts
    U, _, Vt = np.linalg.svd(M)
    Dz = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        Dz[2, 2] = -1.0
    R = U @ Dz @ Vt
    tgt_metric = torch.as_tensor((R @ ts.T).T + rest_joints[0], dtype=torch.float32, device=device)

    global_orient = torch.zeros(1, 3, device=device, requires_grad=True)
    body_pose = torch.zeros(1, 63, device=device, requires_grad=True)
    transl = torch.zeros(1, 3, device=device, requires_grad=True)
    s_idx_t = torch.tensor(s_idx, device=device)

    opt = torch.optim.LBFGS([global_orient, body_pose, transl], lr=1.0,
                            max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        j = model(global_orient=global_orient, body_pose=body_pose,
                  betas=betas, transl=transl).joints[0]
        loss = ((j[s_idx_t] - tgt_metric) ** 2).sum() + 1e-4 * (body_pose ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)

    with torch.no_grad():
        j = model(global_orient=global_orient, body_pose=body_pose, betas=betas, transl=transl).joints[0]
        fit_loss = float(((j[s_idx_t] - tgt_metric) ** 2).sum())
        joints_3d = j[:55].cpu().numpy().astype(np.float32)

    z = lambda n: np.zeros(n, np.float32)  # noqa: E731
    return {
        "global_orient": _np1(global_orient),
        "body_pose": _np1(body_pose),
        "betas": _np1(betas),
        "transl": _np1(transl),
        "left_hand_pose": z(45), "right_hand_pose": z(45),
        "jaw_pose": z(3), "leye_pose": z(3), "reye_pose": z(3), "expression": z(10),
        "gender": gender, "model_path": model_path,
        "joints_3d": joints_3d, "joint_names": body_joint_names(),
        "fit_loss": fit_loss,
    }


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

    lh = lh0.clone(); rh = rh0.clone()
    params = []
    if left_active:
        lh.requires_grad_(True); params.append(lh)
    if right_active:
        rh.requires_grad_(True); params.append(rh)
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
