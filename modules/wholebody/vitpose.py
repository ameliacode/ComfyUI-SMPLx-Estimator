"""
ViTPose whole-body (COCO-WholeBody 133) 2D keypoint inference via ONNX Runtime.

Top-down: a person bbox -> 256x192 crop -> ViTPose -> 133 heatmaps -> keypoints
in image coords. We only consume it for the 2x21 HAND keypoints (the body path
stays on ClickPose), but the full 133 are decoded and returned.

COCO-WholeBody 133 layout:
    0-16   body (COCO-17)
    17-22  feet (6)
    23-90  face (68)
    91-111 left hand (21)
    112-132 right hand (21)

The crop/affine + DARK heatmap decode below are vendored from mmpose
(Apache-2.0; via Alibaba Wan's ComfyUI-WanAnimatePreprocess) and kept verbatim
so decoding matches the trained model exactly. Only thin wrappers are ours.
"""

import os
import warnings

import cv2
import numpy as np
import onnxruntime as ort

# COCO-WholeBody hand slices (inclusive start, exclusive stop)
LEFT_HAND = slice(91, 112)
RIGHT_HAND = slice(112, 133)

# Default on-disk model (single-file ONNX; no external data blob).
DEFAULT_VITPOSE_ONNX = "/home/wswg3/github/ComfyUI/ComfyUI/models/onnx/vitpose-l-wholebody.onnx"

_IMG_NORM_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_IMG_NORM_STD = np.array([0.229, 0.224, 0.225], np.float32)
_INPUT_HW = (256, 192)          # ViTPose input (H, W)

_sess_cache: dict = {}


# ── vendored mmpose affine + DARK decode (Apache-2.0) ───────────────────────────
def _get_transform(center, scale, res, rot=0):
    crop_aspect_ratio = res[0] / float(res[1])
    h = 200 * scale
    w = h / crop_aspect_ratio
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / w
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / w + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    return t


def _transform(pt, center, scale, res, invert=0):
    t = _get_transform(center, scale, res)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0] - 1, pt[1] - 1, 1.]).T
    new_pt = np.dot(t, new_pt)
    return np.array([round(new_pt[0]), round(new_pt[1])], dtype=int) + 1


def _crop(img, center, scale, res):
    ul = np.array(_transform([1, 1], center, max(scale), res, invert=1)) - 1
    br = np.array(_transform([res[1] + 1, res[0] + 1], center, max(scale), res, invert=1)) - 1
    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(new_shape, dtype=np.float32)
    new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
    old_x = max(0, ul[0]), min(len(img[0]), br[0])
    old_y = max(0, ul[1]), min(len(img), br[1])
    new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1], old_x[0]:old_x[1]]
    return cv2.resize(new_img, (res[1], res[0]))


def _bbox_to_center_scale(bbox_xyxy, input_resolution, rescale=1.25):
    """bbox [x0,y0,x1,y1] -> (center, scale) with the model's H/W aspect."""
    crop_h, crop_w = input_resolution
    aspect = crop_h / float(crop_w)
    cx = (bbox_xyxy[0] + bbox_xyxy[2]) / 2.0
    cy = (bbox_xyxy[1] + bbox_xyxy[3]) / 2.0
    bw = bbox_xyxy[2] - bbox_xyxy[0]
    bh = bbox_xyxy[3] - bbox_xyxy[1]
    size = max(bw * aspect, bh)
    scale = np.array([size / aspect, size], np.float32) / 200.0 * rescale
    return np.array([cx, cy], np.float32), scale


def _get_max_preds(heatmaps):
    N, K, _, W = heatmaps.shape
    hr = heatmaps.reshape((N, K, -1))
    idx = np.argmax(hr, 2).reshape((N, K, 1))
    maxvals = np.amax(hr, 2).reshape((N, K, 1))
    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
    preds[:, :, 0] = preds[:, :, 0] % W
    preds[:, :, 1] = preds[:, :, 1] // W
    preds = np.where(np.tile(maxvals, (1, 1, 2)) > 0.0, preds, -1)
    return preds, maxvals


def _gaussian_blur(heatmaps, kernel=11):
    border = (kernel - 1) // 2
    N, K, H, W = heatmaps.shape
    for i in range(N):
        for j in range(K):
            origin_max = np.max(heatmaps[i, j])
            dr = np.zeros((H + 2 * border, W + 2 * border), np.float32)
            dr[border:-border, border:-border] = heatmaps[i, j].copy()
            dr = cv2.GaussianBlur(dr, (kernel, kernel), 0)
            heatmaps[i, j] = dr[border:-border, border:-border].copy()
            heatmaps[i, j] *= origin_max / max(np.max(heatmaps[i, j]), 1e-12)
    return heatmaps


def _taylor(heatmap, coord):
    H, W = heatmap.shape[:2]
    px, py = int(coord[0]), int(coord[1])
    if 1 < px < W - 2 and 1 < py < H - 2:
        dx = 0.5 * (heatmap[py][px + 1] - heatmap[py][px - 1])
        dy = 0.5 * (heatmap[py + 1][px] - heatmap[py - 1][px])
        dxx = 0.25 * (heatmap[py][px + 2] - 2 * heatmap[py][px] + heatmap[py][px - 2])
        dxy = 0.25 * (heatmap[py + 1][px + 1] - heatmap[py - 1][px + 1]
                      - heatmap[py + 1][px - 1] + heatmap[py - 1][px - 1])
        dyy = 0.25 * (heatmap[py + 2][px] - 2 * heatmap[py][px] + heatmap[py - 2][px])
        derivative = np.array([[dx], [dy]])
        hessian = np.array([[dxx, dxy], [dxy, dyy]])
        if dxx * dyy - dxy ** 2 != 0:
            offset = -np.linalg.inv(hessian) @ derivative
            coord += np.squeeze(np.array(offset.T), axis=0)
    return coord


def _transform_preds(coords, center, scale, output_size):
    scale_x = scale[0] / output_size[0]
    scale_y = scale[1] / output_size[1]
    out = np.ones_like(coords)
    out[:, 0] = coords[:, 0] * scale_x + center[0] - scale[0] * 0.5
    out[:, 1] = coords[:, 1] * scale_y + center[1] - scale[1] * 0.5
    return out


def _decode(heatmaps, center, scale, kernel=11):
    """DARK (unbiased) decode -> image-space keypoints (K,2) + scores (K,1)."""
    heatmaps = heatmaps.copy()
    N, K, H, W = heatmaps.shape
    preds, maxvals = _get_max_preds(heatmaps)
    hm = np.log(np.maximum(_gaussian_blur(heatmaps, kernel), 1e-10))
    for n in range(N):
        for k in range(K):
            preds[n][k] = _taylor(hm[n][k], preds[n][k])
    preds[0] = _transform_preds(preds[0], center, scale * 200.0, [W, H])
    return preds[0], maxvals[0]


# ── public API ──────────────────────────────────────────────────────────────
def _session(onnx_path, device):
    key = (onnx_path, device)
    if key not in _sess_cache:
        if not os.path.isfile(onnx_path):
            raise FileNotFoundError(
                f"ViTPose whole-body ONNX not found: {onnx_path!r}. Point "
                f"'vitpose_onnx_path' at the on-disk model (default {DEFAULT_VITPOSE_ONNX!r})."
            )
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if device == "cuda" else ["CPUExecutionProvider"])
        try:
            sess = ort.InferenceSession(onnx_path, providers=providers)
        except Exception as e:                       # CUDA EP missing -> CPU
            warnings.warn(f"[wholebody] ONNX CUDA provider failed ({e}); using CPU.")
            sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        _sess_cache[key] = sess
    return _sess_cache[key]


def detect_wholebody(image_rgb01, bbox_xyxy=None, onnx_path=DEFAULT_VITPOSE_ONNX,
                     device="cuda"):
    """
    Run ViTPose whole-body on one image.

    image_rgb01: (H,W,3) float RGB in [0,1] (a single ComfyUI IMAGE frame).
    bbox_xyxy:   [x0,y0,x1,y1] person box; None -> whole image.

    Returns (keypoints (133,2) image px, scores (133,)).
    """
    img = np.asarray(image_rgb01, np.float32)
    H, W = img.shape[:2]
    if bbox_xyxy is None:
        bbox_xyxy = [0, 0, W, H]
    center, scale = _bbox_to_center_scale(bbox_xyxy, _INPUT_HW)
    crop = _crop(img, center, scale, _INPUT_HW)             # (256,192,3) RGB [0,1]
    inp = ((crop - _IMG_NORM_MEAN) / _IMG_NORM_STD).transpose(2, 0, 1)[None].astype(np.float32)

    sess = _session(onnx_path, device)
    heatmaps = sess.run(None, {sess.get_inputs()[0].name: inp})[0]
    kps, scores = _decode(heatmaps, center, scale)
    return kps.astype(np.float32), scores[:, 0].astype(np.float32)
