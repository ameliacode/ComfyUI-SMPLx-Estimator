"""
Headless SMPL-X mesh rendering (PyTorch3D, GPU) -> ControlNet-style maps:
pose (shaded), depth, normal, canny. Used by the SMPL-X editor's outputs.
"""

import numpy as np
import torch
import cv2


def _to_img(t):
    return (t.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)


_WARNED_NO_P3D = False


def render_maps(verts_np, faces_np, device, size=512, azim=0.0, ground=True,
                camera=None):
    """
    Render the SMPL-X mesh to (pose, depth, normal, canny) uint8 RGB images.

    verts_np: (V,3) world-metric (Y-up). faces_np: (F,3).
    Default: front orthographic-ish view (perspective, fov 40) framed to the body.

    camera (optional): the editor's three.js camera as
    ``{"eye": [x,y,z], "at": [x,y,z], "up": [x,y,z], "fov": deg}`` in the SAME
    world space as ``verts_np``. When given, the render is taken from that exact
    viewpoint (eye/at/up fed to look_at_view_transform) so the output matches the
    editor viewport. Auto framing is skipped.

    Rasterization uses PyTorch3D. If it isn't installed (no prebuilt wheel for your
    torch/CUDA), the maps come back blank with a one-time warning — estimation and
    the interactive 3D editor still work, only the rendered maps are unavailable.
    """
    try:
        from pytorch3d.structures import Meshes
        from pytorch3d.renderer import (
            look_at_view_transform, FoVPerspectiveCameras,
            RasterizationSettings, MeshRasterizer,
        )
    except Exception:
        global _WARNED_NO_P3D
        if not _WARNED_NO_P3D:
            print("[render] PyTorch3D not available — pose/depth/normal/canny maps will be "
                  "blank. Install PyTorch3D to enable rendering: "
                  "https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md")
            _WARNED_NO_P3D = True
        blank = np.zeros((size, size, 3), np.uint8)
        return blank, blank.copy(), blank.copy(), blank.copy()

    V = torch.as_tensor(np.asarray(verts_np), dtype=torch.float32, device=device)
    Fc = torch.as_tensor(np.asarray(faces_np).astype(np.int64), device=device)
    if ground:
        V = V.clone()
        V[:, 1] -= V[:, 1].min()                      # feet on Y=0
    mesh = Meshes(verts=[V], faces=[Fc])

    if camera is not None:
        eye = torch.tensor([camera["eye"]], dtype=torch.float32, device=device)
        at = torch.tensor([camera.get("at", [0.0, 0.8, 0.0])], dtype=torch.float32, device=device)
        up = torch.tensor([camera.get("up", [0.0, 1.0, 0.0])], dtype=torch.float32, device=device)
        R, T = look_at_view_transform(eye=eye, at=at, up=up, device=device)
        fov = float(camera.get("fov", 40.0))
    else:
        center = V.mean(0)
        h = float(V[:, 1].max() - V[:, 1].min())
        dist = max(h * 1.6, 0.6)
        R, T = look_at_view_transform(dist=dist, elev=0.0, azim=azim,
                                      at=center[None], up=((0, 1, 0),), device=device)
        fov = 40.0
    cam = FoVPerspectiveCameras(device=device, R=R, T=T, fov=fov)
    # bin_size=None -> coarse-to-fine binned rasterization (memory-efficient for a
    # 20k-face mesh; bin_size=0 forces naive rasterization and OOMs on a shared GPU).
    rs = RasterizationSettings(image_size=size, blur_radius=0.0, faces_per_pixel=1)
    frag = MeshRasterizer(cameras=cam, raster_settings=rs)(mesh)

    p2f = frag.pix_to_face[0, ..., 0]                 # (H,W) face idx, -1 = bg
    zbuf = frag.zbuf[0, ..., 0]                       # (H,W) depth, -1 = bg
    mask = p2f >= 0

    # ── normal map (camera space) ────────────────────────────────────────────
    fn = mesh.faces_normals_packed()                  # (F,3) world
    fn_cam = fn @ R[0]                                 # rotate into camera frame
    normal = torch.zeros(size, size, 3, device=device)
    pf = p2f.clamp(min=0)
    n = fn_cam[pf]                                     # (H,W,3)
    n = torch.nn.functional.normalize(n, dim=-1)
    normal_rgb = (n * 0.5 + 0.5)                       # [-1,1] -> [0,1]
    normal_rgb[~mask] = 0.0
    normal = normal_rgb

    # ── depth map (near=white) ───────────────────────────────────────────────
    depth = torch.zeros(size, size, device=device)
    if mask.any():
        zv = zbuf[mask]
        zmin, zmax = zv.min(), zv.max()
        dn = 1.0 - (zbuf - zmin) / (zmax - zmin + 1e-6)   # near -> 1
        depth[mask] = dn[mask].clamp(0, 1)
    depth_rgb = depth[..., None].repeat(1, 1, 3)

    # ── pose (frontal-lit shaded body) ───────────────────────────────────────
    # |n_z| = how front-facing the surface is -> bright front, dim grazing edges.
    shade = (n[..., 2].abs() * 0.7 + 0.3).clamp(0.0, 1.0)    # +0.3 ambient
    pose = torch.zeros(size, size, 3, device=device)
    body_col = torch.tensor([0.85, 0.72, 0.62], device=device)
    pose[mask] = (shade[..., None] * body_col)[mask]
    pose_rgb = pose

    pose_img = _to_img(pose_rgb)
    depth_img = _to_img(depth_rgb)
    normal_img = _to_img(normal_rgb)

    # ── canny (edges of the shaded body) ─────────────────────────────────────
    gray = cv2.cvtColor(pose_img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    canny_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

    return pose_img, depth_img, normal_img, canny_img
