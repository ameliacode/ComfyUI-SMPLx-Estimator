"""
ComfyUI nodes for SMPL-X fitting + editing (replaces MotionAGFormer / Pose3DEditor
for the SMPL-X pipeline).

    ClickPose POSE_KEYPOINTS (COCO-17) -> SMPLXFit -> SMPLX -> SMPLXEditor -> SMPLX

SMPLXFit fits an SMPL-X body to the 2D keypoints (cold VPoser-regularized
SMPLify-X, all assets on disk). The SMPLX type is a plain dict (see
modules/smplx_fit/fitting.py) so it caches/serialises like the other types.
"""

import hashlib
import json
import os
import re
from typing import Optional

import numpy as np
import torch

from ..modules.smplx_fit.model import load_smplx, resolve_device, DEFAULT_SMPLX_PARENT
from ..modules.smplx_fit.fitting import resolve_edit, resolve_hand_edit
from ..modules.smplx_fit.joint_maps import (
    body_limbs, editable_limbs, EDITABLE_JOINTS, NUM_BODY_JOINTS,
)
from ..modules.smplx_fit.render import render_maps
from ..modules.smplx_fit.skin import editable_skin_weights


def _img(arr: np.ndarray) -> torch.Tensor:
    """uint8 RGB (H,W,3) -> ComfyUI IMAGE tensor (1,H,W,3) float[0,1]."""
    return torch.from_numpy(arr.astype(np.float32) / 255.0).unsqueeze(0)


def _ground(smplx_dict: dict, verts: np.ndarray):
    """Shift the body so its lowest mesh vertex sits on the floor (Y=0)."""
    v = np.asarray(verts, np.float32)
    min_y = float(v[:, 1].min())
    out = dict(smplx_dict)
    out["transl"] = np.asarray(smplx_dict["transl"], np.float32).copy()
    out["transl"][1] -= min_y
    out["joints_3d"] = np.asarray(smplx_dict["joints_3d"], np.float32).copy()
    out["joints_3d"][:, 1] -= min_y
    vg = v.copy()
    vg[:, 1] -= min_y
    return out, vg


# Hand grasp: the SMPL-X MANO hand PCA component 0 is the dominant open<->close
# axis. With flat_hand_mean=True a zero hand_pose is flat/open; adding a positive
# multiple of component 0 curls the fingers. GRASP_SCALE maps grasp=1.0 to a full
# fist (empirically the fingertip->wrist distance bottoms out near 4.0; beyond
# that the fingers over-curl through the palm).
GRASP_SCALE = 4.0


def _apply_grasp(smplx_dict: dict, model, left_grasp: float, right_grasp: float) -> dict:
    """
    Manual grasp OVERRIDE in [0,1] (0 = keep incoming pose, >0 = curl to that fist).

    A hand with grasp 0 is left untouched so an estimated hand pose (from
    WholeBodyHandDetector -> SMPLXFit) survives into the editor; a positive slider
    overrides that hand to a uniform fist.
    """
    cL = getattr(model, "np_left_hand_components", None)
    cR = getattr(model, "np_right_hand_components", None)
    if cL is None or cR is None:        # model built without hand PCA — leave as-is
        return smplx_dict
    out = dict(smplx_dict)
    if left_grasp and left_grasp > 0:
        out["left_hand_pose"] = (float(left_grasp) * GRASP_SCALE
                                 * np.asarray(cL[0], np.float32)).astype(np.float32)
    if right_grasp and right_grasp > 0:
        out["right_hand_pose"] = (float(right_grasp) * GRASP_SCALE
                                  * np.asarray(cR[0], np.float32)).astype(np.float32)
    return out


def _parse_csv(s, n):
    """Comma/space-separated floats -> length-n float32 (pad/truncate). '' -> None."""
    if not s or not str(s).strip():
        return None
    toks = [x for x in re.split(r"[,\s]+", str(s).strip()) if x]
    try:
        vals = [float(x) for x in toks]
    except ValueError:
        return None
    out = np.zeros(n, np.float32)
    out[:min(n, len(vals))] = vals[:n]
    return out


def _apply_params(smplx_dict, *, betas="", expression="", jaw_open=0.0):
    """
    Direct SMPL-X parameter edits:
      betas/expression : absolute (CSV; empty = keep current),
      jaw_open         : 0..1 mouth open (jaw pitch).
    """
    out = dict(smplx_dict)
    b = _parse_csv(betas, len(out["betas"]))
    if b is not None:
        out["betas"] = b
    e = _parse_csv(expression, 10)
    if e is not None:
        out["expression"] = e
    if jaw_open:
        out["jaw_pose"] = np.array([float(jaw_open) * 0.5, 0.0, 0.0], np.float32)
    return out


def _forward_mesh(smplx_dict, model, device):
    """Forward SMPL-X -> (vertices, static faces, joints[:55]).

    Returns the 55 joints too so the editor's joint handles always match the
    current mesh (after hand pose / betas / expression / global edits)."""
    with torch.inference_mode(False):
        def t(k, n):
            return torch.as_tensor(np.asarray(smplx_dict[k]), dtype=torch.float32,
                                   device=device).view(1, n)
        out = model(global_orient=t("global_orient", 3), body_pose=t("body_pose", 63),
                    betas=t("betas", len(smplx_dict["betas"])), transl=t("transl", 3),
                    left_hand_pose=t("left_hand_pose", 45), right_hand_pose=t("right_hand_pose", 45),
                    jaw_pose=t("jaw_pose", 3), leye_pose=t("leye_pose", 3),
                    reye_pose=t("reye_pose", 3), expression=t("expression", 10))
    verts = out.vertices[0].detach().cpu().numpy().astype(np.float32)
    joints = out.joints[0, :55].detach().cpu().numpy().astype(np.float32)
    faces = np.asarray(model.faces, dtype=np.int32)
    return verts, faces, joints


def _smplx_payload(smplx_dict, limbs, vertices=None, faces=None, skin=None,
                   editable=None):
    """JSON pushed to the JS viewer: 55 joints + editable handle list + limb table
    + (optional) mesh + skin.

    Joints are sent in SMPL-X joint-index space (0-54); ``editable`` lists which
    indices get draggable handles (22 body + 30 finger joints). ``skin`` indices
    are in the same 0-54 space, so finger drags deform the surface too. The viewer
    skins by blending each vertex's displacement toward the joints it is weighted
    to, so the mesh follows the handles (body AND fingers) as they are dragged.
    """
    j3d = np.asarray(smplx_dict["joints_3d"], np.float32)
    n = min(55, j3d.shape[0])
    data = {
        "joints_3d": j3d[:n].tolist(),
        "joint_names": smplx_dict["joint_names"],
        "limbs": limbs,
        "editable": list(editable) if editable is not None else list(range(NUM_BODY_JOINTS)),
        "editorMode": True,
    }
    if vertices is not None and faces is not None:
        # round verts to 4 dp (mm precision) to keep the payload light
        data["vertices"] = np.round(vertices, 4).tolist()
        data["faces"] = faces.tolist()
    if skin is not None:
        # per-vertex top-k (jointIdx in 0..54, weight)
        data["skin"] = {
            "indices": skin["indices"].tolist(),
            "weights": np.round(skin["weights"], 4).tolist(),
        }
    return json.dumps(data)


class SMPLXEditor:
    """
    Edit SMPL-X in 3D: drag joints (body + fingers) and tweak parameters.

    Dragging a BODY joint (idx 0-21) re-solves body_pose; dragging a FINGER joint
    (idx 25-54) re-solves that hand's hand_pose — everything else frozen, so edits
    stay local (the dragged joint is approached, not guaranteed). The viewer sends
    POSE3D_CORRECTIONS = {jointIdx: [x,y,z]} in SMPL-X world-metric coords.

    Direct parameter widgets edit shape (betas), facial expression + jaw, and the
    global orientation/translation, on top of the dragging.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "smplx": ("SMPLX",),
                "size": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 64}),
                "reik_iters": ("INT", {"default": 80, "min": 2, "max": 400}),
                "device": (["auto", "cuda", "cpu"],),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
            },
            "optional": {
                "corrections": ("STRING", {"default": "", "multiline": False}),
                "camera": ("STRING", {"default": "", "multiline": False,
                                      "tooltip": "Editor camera (set automatically as you "
                                                 "orbit the 3D view). Renders the output maps "
                                                 "from that viewpoint; empty = front view."}),
                # ── direct SMPL-X parameter edits (strings tolerate empty/stale values) ──
                "betas": ("STRING", {"default": "", "multiline": False,
                                     "tooltip": "Body shape: up to 10 comma-separated betas "
                                                "(e.g. '1.5,-0.5'). Empty = keep fitted shape. "
                                                "Range ~[-5,5]; beta0≈overall size."}),
                "expression": ("STRING", {"default": "", "multiline": False,
                                          "tooltip": "Facial expression: up to 10 comma-separated "
                                                     "coefficients. Empty = neutral."}),
            },
        }

    RETURN_TYPES = ("SMPLX", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("smplx", "pose", "depth", "normal", "canny")
    OUTPUT_NODE = True
    FUNCTION = "edit"
    CATEGORY = "editpose"

    @classmethod
    def IS_CHANGED(cls, smplx, size, reik_iters, device, seed, corrections=None, camera=None,
                   betas="", expression=""):
        h = hashlib.sha256()
        for k in ("global_orient", "body_pose", "betas", "transl",
                  "left_hand_pose", "right_hand_pose"):
            h.update(np.asarray(smplx[k], np.float32).tobytes())
        h.update(repr((size, reik_iters, device, seed, corrections or "", camera or "",
                       betas, expression)).encode())
        return h.hexdigest()

    def edit(self, smplx, size, reik_iters, device, seed, corrections=None, camera=None,
             betas="", expression=""):
        dev = resolve_device(device)
        model = load_smplx(smplx["model_path"], smplx.get("gender", "neutral"), dev)

        # 1) direct parameter edits first (shape/expression) so IK uses them
        out = _apply_params(smplx, betas=betas, expression=expression)

        # 2) joint drags -> body_pose (idx 0-21) and/or hand_pose (idx 25-54) IK
        if corrections and corrections.strip():
            try:
                targets = json.loads(corrections)
            except json.JSONDecodeError:
                targets = {}
            body_t = {k: v for k, v in targets.items() if int(k) < NUM_BODY_JOINTS}
            hand_t = {k: v for k, v in targets.items() if 25 <= int(k) < 55}
            if body_t:
                out = resolve_edit(out, body_t, model, dev, iters=reik_iters, seed=seed)
            if hand_t:
                out = resolve_hand_edit(out, hand_t, model, dev, iters=reik_iters, seed=seed)
            if body_t or hand_t:
                print(f"[smplx_fit] edit: re-solved {len(body_t)} body + "
                      f"{len(hand_t)} finger joint(s)")

        cam = None
        if camera and camera.strip():
            try:
                c = json.loads(camera)
                if isinstance(c, dict) and "eye" in c:
                    cam = c
            except json.JSONDecodeError:
                pass

        verts, faces, joints = _forward_mesh(out, model, dev)
        out["joints_3d"] = joints                             # handles match the mesh
        out, verts = _ground(out, verts)                      # feet on the floor
        with torch.inference_mode(False):
            parents = model.parents.detach().cpu().numpy().astype(int)
        skin = editable_skin_weights(model)                   # live skinning (body+fingers)
        pose, depth, normal, canny = render_maps(verts, faces, dev, size=size,
                                                 ground=False, camera=cam)
        payload = _smplx_payload(out, editable_limbs(parents), verts, faces, skin,
                                 editable=EDITABLE_JOINTS)
        return {
            "ui": {"smplx_json": [payload]},
            "result": (out, _img(pose), _img(depth), _img(normal), _img(canny)),
        }
