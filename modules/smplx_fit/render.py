"""
Headless SMPL-X mesh rendering -> ControlNet-style maps: pose (shaded), depth,
normal, canny. Used by the SMPL-X editor's outputs and the estimator previews.

Pure NumPy/OpenCV rasterizer (painter's algorithm) — no PyTorch3D / CUDA build /
OpenGL, so it works on any torch/CUDA. cv2 is already a dependency.
"""

import numpy as np
import cv2


def _to_img(a):
    return (np.clip(a, 0, 1) * 255).astype(np.uint8)


def _look_at(eye, at, up):
    """Return world->view rotation R (3x3) for a camera at `eye` looking at `at`
    (camera looks down -Z, OpenGL convention)."""
    eye = np.asarray(eye, np.float64)
    f = np.asarray(at, np.float64) - eye
    f /= np.linalg.norm(f) + 1e-9
    r = np.cross(f, np.asarray(up, np.float64))
    r /= np.linalg.norm(r) + 1e-9
    u = np.cross(r, f)
    R = np.stack([r, u, -f], axis=0)                 # rows = view axes
    return R, eye


def _rasterize(V, faces, R, eye, fov_deg, size):
    """Painter's-algorithm rasterizer.
    Returns pix_to_face (H,W int32, -1=bg), zbuf (H,W float32 front-distance,
    inf=bg), and fn_view (F,3) view-space face normals."""
    Vv = (V - eye) @ R.T                             # view space; camera looks -Z
    zc = -Vv[:, 2]                                    # front distance (>0 visible)
    focal = 1.0 / np.tan(np.radians(fov_deg) * 0.5)
    zsafe = np.maximum(zc, 1e-6)
    px = ((focal * Vv[:, 0] / zsafe) * 0.5 + 0.5) * size
    py = (1.0 - ((focal * Vv[:, 1] / zsafe) * 0.5 + 0.5)) * size   # y flipped
    P = np.stack([px, py], axis=1)                    # (N,2) pixel coords

    fv = P[faces]                                     # (F,3,2)
    fz = zc[faces].mean(axis=1)                        # (F,) face depth
    infront = (zc[faces] > 1e-6).all(axis=1)
    finite = np.isfinite(fv).all(axis=(1, 2))
    order = np.argsort(-fz)                            # far -> near
    order = order[infront[order] & finite[order]]

    pix_to_face = np.full((size, size), -1, np.int32)
    for fi in order:                                  # near overwrites far
        cv2.fillConvexPoly(pix_to_face, np.round(fv[fi]).astype(np.int32),
                           int(fi), lineType=cv2.LINE_8)

    # per-pixel depth via barycentric interpolation of the covering face
    zbuf = np.full((size, size), np.inf, np.float32)
    yy, xx = np.where(pix_to_face >= 0)
    if len(xx):
        pf = pix_to_face[yy, xx]
        tri = fv[pf]                                  # (M,3,2)
        a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
        p = np.stack([xx, yy], axis=1).astype(np.float64)
        v0, v1, v2 = b - a, c - a, p - a
        den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
        den = np.where(np.abs(den) < 1e-9, 1e-9, den)
        w1 = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / den
        w2 = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / den
        w0 = 1.0 - w1 - w2
        fzc = zc[faces[pf]]                           # (M,3)
        zbuf[yy, xx] = (w0 * fzc[:, 0] + w1 * fzc[:, 1] + w2 * fzc[:, 2]).astype(np.float32)

    tv = V[faces]                                     # (F,3,3)
    fn = np.cross(tv[:, 1] - tv[:, 0], tv[:, 2] - tv[:, 0])
    fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-9
    fn_view = fn @ R.T                                # world -> view
    return pix_to_face, zbuf, fn_view


def render_maps(verts_np, faces_np, device=None, size=512, azim=0.0, ground=True,
                camera=None):
    """
    Render the SMPL-X mesh to (pose, depth, normal, canny) uint8 RGB images.

    verts_np: (V,3) world-metric (Y-up). faces_np: (F,3).
    Default: front perspective (fov 40) framed to the body.

    camera (optional): the editor's three.js camera as
    ``{"eye": [x,y,z], "at": [x,y,z], "up": [x,y,z], "fov": deg}`` in the SAME
    world space as ``verts_np`` — renders from that exact viewpoint so the output
    matches the editor viewport. ``device`` is accepted for API compatibility.
    """
    V = np.asarray(verts_np, np.float64).copy()
    F = np.asarray(faces_np).astype(np.int64)
    if ground:
        V[:, 1] -= V[:, 1].min()                      # feet on Y=0

    if camera is not None:
        eye = camera["eye"]
        at = camera.get("at", [0.0, 0.8, 0.0])
        up = camera.get("up", [0.0, 1.0, 0.0])
        fov = float(camera.get("fov", 40.0))
        R, eye = _look_at(eye, at, up)
    else:
        center = V.mean(0)
        h = float(V[:, 1].max() - V[:, 1].min())
        dist = max(h * 1.6, 0.6)
        rad = np.radians(azim)
        eye = center + np.array([np.sin(rad) * dist, 0.0, np.cos(rad) * dist])
        R, eye = _look_at(eye, center, [0.0, 1.0, 0.0])
        fov = 40.0

    pix_to_face, zbuf, fn_view = _rasterize(V, F, R, eye, fov, size)
    mask = pix_to_face >= 0

    # ── normal map (camera space, [-1,1] -> [0,1]) ───────────────────────────
    n = np.zeros((size, size, 3), np.float64)
    n[mask] = fn_view[pix_to_face[mask]]
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9
    normal_rgb = (n * 0.5 + 0.5)
    normal_rgb[~mask] = 0.0

    # ── depth map (near = white) ─────────────────────────────────────────────
    depth = np.zeros((size, size), np.float64)
    if mask.any():
        zv = zbuf[mask]
        zmin, zmax = float(zv.min()), float(zv.max())
        depth[mask] = np.clip(1.0 - (zbuf[mask] - zmin) / (zmax - zmin + 1e-6), 0, 1)
    depth_rgb = np.repeat(depth[..., None], 3, axis=2)

    # ── pose (frontal-lit shaded body) ───────────────────────────────────────
    shade = np.clip(np.abs(n[..., 2]) * 0.7 + 0.3, 0.0, 1.0)     # front-facing -> bright
    pose = np.zeros((size, size, 3), np.float64)
    pose[mask] = shade[mask, None] * np.array([0.85, 0.72, 0.62])

    pose_img = _to_img(pose)
    depth_img = _to_img(depth_rgb)
    normal_img = _to_img(normal_rgb)

    # ── canny (edges of the shaded body) ─────────────────────────────────────
    edges = cv2.Canny(cv2.cvtColor(pose_img, cv2.COLOR_RGB2GRAY), 40, 120)
    canny_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

    return pose_img, depth_img, normal_img, canny_img
