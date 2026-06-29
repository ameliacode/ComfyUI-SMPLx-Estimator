"""
ComfyUI nodes for single-image motion capture (Phase 1).

Pipeline:
    IMAGE -> ClickPose -> POSE_KEYPOINTS -> MotionAGFormer -> POSE_3D
         -> Pose3DEditor -> POSE_3D (corrected)
"""

import hashlib
import json
import os
import uuid
from typing import Optional

import cv2
import folder_paths
import numpy as np
import torch

from ..modules.clickpose.inference import apply_corrections_with_state, run_detection
from ..modules.clickpose.model import ClickPoseModel
from ..modules.motionagformer.inference import run_lifting
from ..modules.motionagformer.model import MotionAGFormerModel

_REGISTRY_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints", "registry.json")


def _checkpoint_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_checkpoint(path: str) -> None:
    """Register or verify a checkpoint SHA256 without blocking model load."""
    try:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        current = _checkpoint_sha256(path)
        registry: dict = {}
        if os.path.isfile(_REGISTRY_PATH):
            with open(_REGISTRY_PATH) as handle:
                registry = json.load(handle)

        key = os.path.basename(path)
        if key not in registry:
            registry[key] = {"sha256": current, "path": path}
            with open(_REGISTRY_PATH, "w") as handle:
                json.dump(registry, handle, indent=2)
            print(f"[editpose] registered {key} sha256={current[:16]}...")
        elif registry[key]["sha256"] != current:
            print(
                f"[editpose] WARNING: {key} hash mismatch "
                f"(expected {registry[key]['sha256'][:16]}..., got {current[:16]}...) "
                "file may have been replaced or corrupted."
            )
    except FileNotFoundError:
        raise
    except Exception as exc:
        print(f"[editpose] checkpoint verify error (non-fatal): {exc}")


_clickpose_cache: dict = {}
_maf_cache: dict = {}

_COCO_LIMBS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 6),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]

_H36M_LIMBS = [
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10),
    (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16),
]


def _to_bgr(image_tensor: torch.Tensor) -> np.ndarray:
    """Convert a ComfyUI IMAGE tensor to a BGR uint8 array."""
    img = image_tensor[0].cpu().numpy()
    img = (img * 255).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def _to_tensor(bgr: np.ndarray) -> torch.Tensor:
    """Convert a BGR uint8 array to a ComfyUI IMAGE tensor."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(rgb).unsqueeze(0)


def _draw_pose(image_tensor: torch.Tensor, pose_keypoints: dict) -> torch.Tensor:
    """Overlay COCO-17 keypoints and skeleton on the source image."""
    bgr = _to_bgr(image_tensor)
    kps = pose_keypoints["keypoints"]
    scores = pose_keypoints["scores"]

    for i, j in _COCO_LIMBS:
        if scores[i] > 0.3 and scores[j] > 0.3:
            pt1 = (int(kps[i, 0]), int(kps[i, 1]))
            pt2 = (int(kps[j, 0]), int(kps[j, 1]))
            cv2.line(bgr, pt1, pt2, (0, 200, 100), 2, cv2.LINE_AA)

    for idx in range(len(kps)):
        x, y = int(kps[idx, 0]), int(kps[idx, 1])
        score = float(scores[idx])
        g = int(255 * score)
        r = int(255 * (1.0 - score))
        cv2.circle(bgr, (x, y), 5, (0, g, r), -1, cv2.LINE_AA)
        cv2.putText(
            bgr,
            str(idx),
            (x + 6, y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return _to_tensor(bgr)


def _project_pose_3d_to_canvas(joints: np.ndarray, size: int, margin: float = 0.12) -> tuple:
    """Project 3D joints to a centered 2D canvas using X/Y only."""
    xy = joints[:, [0, 1]].astype(np.float32).copy()
    xy[:, 1] = -xy[:, 1]
    mins = xy.min(axis=0)
    maxs = xy.max(axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    usable = size * (1.0 - 2.0 * margin)
    scale = usable / float(max(span))
    centered = xy - (mins + maxs) * 0.5
    proj = centered * scale + size * 0.5
    return proj, joints[:, 2].astype(np.float32).copy()


def _render_pose_3d_pack(pose_3d: dict, size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render pose, depth, and normal previews from H36M joints."""
    joints = pose_3d["joints_3d"]
    proj, depth = _project_pose_3d_to_canvas(joints, size)

    pose = np.zeros((size, size, 3), dtype=np.uint8)
    depth_img = np.zeros_like(pose)
    normal = np.zeros_like(pose)

    zmin = float(depth.min())
    zspan = float(depth.max() - zmin) or 1.0

    for i, j in _H36M_LIMBS:
        p1 = tuple(np.round(proj[i]).astype(int))
        p2 = tuple(np.round(proj[j]).astype(int))
        zavg = ((depth[i] + depth[j]) * 0.5 - zmin) / zspan
        intensity = int(np.clip(64 + zavg * 191, 0, 255))
        bone = joints[j] - joints[i]
        bone_norm = np.linalg.norm(bone) or 1.0
        normal_color = np.clip(((bone / bone_norm + 1.0) * 0.5) * 255.0, 0, 255).astype(np.uint8)
        cv2.line(pose, p1, p2, (0, 170, 255), 4, cv2.LINE_AA)
        cv2.line(depth_img, p1, p2, (intensity, intensity, intensity), 4, cv2.LINE_AA)
        cv2.line(normal, p1, p2, tuple(int(c) for c in normal_color), 4, cv2.LINE_AA)

    for idx, (x, y) in enumerate(proj):
        center = (int(round(x)), int(round(y)))
        znorm = (depth[idx] - zmin) / zspan
        intensity = int(np.clip(80 + znorm * 175, 0, 255))
        cv2.circle(pose, center, 6, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(depth_img, center, 6, (intensity, intensity, intensity), -1, cv2.LINE_AA)
        cv2.circle(normal, center, 6, (128, 128, 255), -1, cv2.LINE_AA)

    return pose, depth_img, normal


class ClickPose:
    """
    Combined ClickPose detector and interactive editor.

    - First queue for an image: full encoder + decoder detection.
    - Same image with corrections: decoder-only refinement.
    - New image: stale corrections are ignored and detection reruns.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "checkpoint_path": (
                    "STRING",
                    {
                        "default": "custom_nodes/comfyui-mocap/checkpoints/ClickPose_model_only_R50.pth",
                        "tooltip": "Path to ClickPose checkpoint (.pth), relative to the ComfyUI working directory.",
                    },
                ),
                "device": (["auto", "cuda", "cpu"],),
            },
            "optional": {
                "corrections": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("POSE_KEYPOINTS", "IMAGE")
    RETURN_NAMES = ("pose_keypoints", "visualization")
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = "editpose"

    def __init__(self):
        self._last_image_id: Optional[str] = None
        self._last_pose_keypoints: Optional[dict] = None
        self._last_refine_state: Optional[dict] = None
        self._last_model_key: Optional[tuple] = None

    @classmethod
    def IS_CHANGED(cls, image: torch.Tensor, checkpoint_path: str, device: str, corrections: Optional[str] = None):
        if image is None:
            return float("nan")
        image_hash = hashlib.md5(image.cpu().numpy().tobytes()).hexdigest()
        corrections_hash = hashlib.md5((corrections or "").encode()).hexdigest()
        return f"{image_hash}:{checkpoint_path}:{device}:{corrections_hash}"

    def run(
        self,
        image: torch.Tensor,
        checkpoint_path: str,
        device: str,
        corrections: Optional[str] = None,
    ) -> tuple:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        cache_key = (checkpoint_path, device)
        if cache_key not in _clickpose_cache:
            _verify_checkpoint(checkpoint_path)
            model = ClickPoseModel()
            model.load(checkpoint_path, device)
            _clickpose_cache[cache_key] = model
        model = _clickpose_cache[cache_key]

        img_np = (image[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        incoming_image_id = hashlib.md5(img_np.tobytes()).hexdigest()[:12]
        same_image = incoming_image_id == self._last_image_id and cache_key == self._last_model_key

        want_corrections = bool(corrections and corrections.strip())

        # Reuse the cached detection + its captured refine state only when this is
        # the same image AND we still hold valid state. Otherwise run a full detect
        # now and capture the refine state IMMEDIATELY, so a later detect() on this
        # shared, cached model can never clobber it.
        have_state = (
            same_image
            and self._last_pose_keypoints is not None
            and self._last_refine_state is not None
            and self._last_refine_state.get("out") is not None
        )

        if have_state:
            base_pose = self._last_pose_keypoints
            refine_state = self._last_refine_state
        else:
            base_pose = run_detection(model, img_np)
            refine_state = model.get_refine_state()

        # Apply corrections in the SAME run that established the state, so a
        # correction is never silently dropped. Previously the first corrected run
        # after a restart / workflow reload re-detected and ignored the correction,
        # then IS_CHANGED cached that uncorrected result away.
        if want_corrections:
            click_pose_state = {
                "model": model,
                "image_size": base_pose["image_size"],
                "image_id": base_pose.get("image_id", incoming_image_id),
                "refine_state": refine_state,
            }
            pose_keypoints = apply_corrections_with_state(base_pose, click_pose_state, corrections)
        else:
            pose_keypoints = base_pose

        # Persist the BASE (uncorrected) detection + its refine state. Corrections
        # are always re-applied from the detection, so this keeps refinement stable
        # across queue runs.
        image_id = base_pose.get("image_id", incoming_image_id)
        self._last_image_id = image_id
        self._last_pose_keypoints = base_pose
        self._last_refine_state = refine_state
        self._last_model_key = cache_key

        vis = _draw_pose(image, pose_keypoints)

        temp_dir = folder_paths.get_temp_directory()
        os.makedirs(temp_dir, exist_ok=True)
        filename = f"mocap_pose_{uuid.uuid4().hex[:8]}.png"
        vis_np = (vis[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(temp_dir, filename), cv2.cvtColor(vis_np, cv2.COLOR_RGB2BGR))

        kps_payload = json.dumps(
            {
                "keypoints": pose_keypoints["keypoints"].tolist(),
                "scores": pose_keypoints["scores"].tolist(),
                "image_size": list(pose_keypoints["image_size"]),
                "image_id": image_id,
                "refine_method": pose_keypoints.get("refine_method", "detect"),
            }
        )

        return {
            "ui": {
                "images": [{"filename": filename, "subfolder": "", "type": "temp"}],
                "kps_json": [kps_payload],
            },
            "result": (pose_keypoints, vis),
        }


class MotionAGFormer:
    """
    Lift COCO-17 2D keypoints to H36M-17 3D joints.

    Set enabled=False while refining the 2D pose to reuse the cached 3D result.
    """

    def __init__(self):
        self._cached_pose_3d = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_keypoints": ("POSE_KEYPOINTS",),
                "checkpoint_path": (
                    "STRING",
                    {
                        "default": "custom_nodes/comfyui-mocap/checkpoints/motionagformer-l-h36m.pth.tr",
                        "tooltip": "Path to MotionAGFormer checkpoint, relative to the ComfyUI working directory.",
                    },
                ),
                "device": (["auto", "cuda", "cpu"],),
                "seq_len": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 243,
                        "tooltip": "Temporal window the checkpoint was trained with. Use 1 for single-frame inference.",
                    },
                ),
                "enabled": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Disable while refining 2D pose to skip relifting. Returns the last cached result.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("POSE_3D",)
    RETURN_NAMES = ("pose_3d",)
    FUNCTION = "lift"
    CATEGORY = "editpose"

    def lift(
        self,
        pose_keypoints: dict,
        checkpoint_path: str,
        device: str,
        seq_len: int,
        enabled: bool,
    ) -> tuple:
        if not enabled and self._cached_pose_3d is not None:
            return (self._cached_pose_3d,)

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        cache_key = (checkpoint_path, device, seq_len)
        if cache_key not in _maf_cache:
            _verify_checkpoint(checkpoint_path)
            model = MotionAGFormerModel()
            model.load(checkpoint_path, device, seq_len)
            _maf_cache[cache_key] = model

        pose_3d = run_lifting(_maf_cache[cache_key], pose_keypoints)
        self._cached_pose_3d = pose_3d
        return (pose_3d,)


class Pose3DEditor:
    """Edit POSE_3D joint positions interactively and return the corrected pose."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_3d": ("POSE_3D",),
                "size": (
                    "INT",
                    {
                        "default": 768,
                        "min": 256,
                        "max": 2048,
                        "step": 64,
                        "tooltip": "Preview image width and height in pixels.",
                    },
                ),
            },
            "optional": {
                "corrections": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("POSE_3D", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("pose_3d", "pose", "depth", "normal")
    OUTPUT_NODE = True
    FUNCTION = "edit"
    CATEGORY = "editpose"

    @staticmethod
    def _apply_corrections(pose_3d: dict, corrections: Optional[str]) -> dict:
        joints = np.array(pose_3d["joints_3d"], dtype=np.float32, copy=True)
        if corrections and corrections.strip():
            parsed = json.loads(corrections)
            if not isinstance(parsed, dict):
                raise ValueError("3D pose corrections must be a JSON object.")
            for key, value in parsed.items():
                idx = int(key)
                if not (0 <= idx < len(joints)):
                    continue
                if not isinstance(value, (list, tuple)) or len(value) != 3:
                    raise ValueError(f"Correction for joint {key} must be [x, y, z].")
                joints[idx] = np.array(value, dtype=np.float32)
        return {"joints_3d": joints, "joint_names": list(pose_3d.get("joint_names", []))}

    def edit(
        self,
        pose_3d: dict,
        size: int,
        corrections: Optional[str] = None,
    ):
        corrected = self._apply_corrections(pose_3d, corrections)
        pose_img, depth_img, normal_img = _render_pose_3d_pack(corrected, size)

        pose_payload = json.dumps(
            {
                "joints_3d": corrected["joints_3d"].tolist(),
                "joint_names": corrected.get("joint_names", []),
            }
        )

        return {
            "ui": {
                "pose3d_json": [pose_payload],
            },
            "result": (
                corrected,
                _to_tensor(pose_img),
                _to_tensor(depth_img),
                _to_tensor(normal_img),
            ),
        }

    @classmethod
    def IS_CHANGED(cls, pose_3d: dict, size: int, corrections: Optional[str] = None):
        if pose_3d is None:
            return float("nan")
        digest = hashlib.sha256()
        digest.update(np.asarray(pose_3d["joints_3d"], dtype=np.float32).tobytes())
        digest.update(f"{size}".encode())
        digest.update((corrections or "").encode())
        return digest.hexdigest()
