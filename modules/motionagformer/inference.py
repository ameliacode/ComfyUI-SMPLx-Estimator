"""
Stateless inference helpers for MotionAGFormer.

Handles:
  - COCO-17 → H36M-17 keypoint remapping
  - 2D normalization matching H36M training convention
  - Adding confidence channel (3rd dim expected by the model)

H36M joint order used by MotionAGFormer (from data/const.py):
    0  sacrum (hip center)
    1  left_hip       2  left_knee    3  left_foot
    4  right_hip      5  right_knee   6  right_foot
    7  center_torso   8  upper_torso  9  neck_base   10 center_head
    11 right_shoulder 12 right_elbow  13 right_hand
    14 left_shoulder  15 left_elbow   16 left_hand

Mapping verified against:
    https://github.com/TaatiTeam/2DEstimatorEval/blob/master/data/prepare_2d_estimation.py
    (function coco2h36m)

NOTE on left/right convention:
    Both COCO and H36M use anatomical left/right (subject's perspective).
    No mirroring needed — COCO left_hip(11) → H36M left_hip(1), etc.

Normalization (h36m.py DataReaderH36M.read_2d):
    x_norm = px / W * 2 - 1
    y_norm = py / W * 2 - H / W        ← both axes normalized by image WIDTH
"""
import numpy as np


H36M_JOINT_NAMES = [
    "sacrum",                                        # 0
    "left_hip",    "left_knee",   "left_foot",       # 1-3
    "right_hip",   "right_knee",  "right_foot",      # 4-6
    "center_torso", "upper_torso", "neck_base", "center_head",  # 7-10
    "right_shoulder", "right_elbow", "right_hand",   # 11-13
    "left_shoulder",  "left_elbow",  "left_hand",    # 14-16
]


def coco_to_h36m(coco_kps: np.ndarray) -> np.ndarray:
    """
    Remap COCO 17-joint keypoints to H36M 17-joint order for MotionAGFormer.

    Exact port of coco2h36m() from TaatiTeam/2DEstimatorEval:
        https://github.com/TaatiTeam/2DEstimatorEval/blob/master/data/prepare_2d_estimation.py

    Args:
        coco_kps: (..., 17, D)  — last two axes are joint-index and feature dim

    Returns:
        h36m_kps: (..., 17, D)  in H36M joint order
    """
    h = np.zeros_like(coco_kps)

    # Averaged joints — must be computed before the joints that depend on them
    h[..., 0, :]  = (coco_kps[..., 11, :] + coco_kps[..., 12, :]) * 0.5   # sacrum
    h[..., 8, :]  = (coco_kps[..., 5, :]  + coco_kps[..., 6, :])  * 0.5   # upper_torso
    h[..., 7, :]  = (h[..., 0, :]         + h[..., 8, :])          * 0.5   # center_torso
    h[..., 9, :]  = (coco_kps[..., 0, :]  + h[..., 8, :])          * 0.5   # neck_base (nose + thorax)
    h[..., 10, :] = (coco_kps[..., 1, :]  + coco_kps[..., 2, :])   * 0.5   # center_head (avg eyes)

    # Direct mappings — both COCO and H36M use anatomical left/right, no swap needed
    h[..., 1, :]  = coco_kps[..., 11, :]  # left_hip       ← COCO left_hip
    h[..., 2, :]  = coco_kps[..., 13, :]  # left_knee      ← COCO left_knee
    h[..., 3, :]  = coco_kps[..., 15, :]  # left_foot      ← COCO left_ankle
    h[..., 4, :]  = coco_kps[..., 12, :]  # right_hip      ← COCO right_hip
    h[..., 5, :]  = coco_kps[..., 14, :]  # right_knee     ← COCO right_knee
    h[..., 6, :]  = coco_kps[..., 16, :]  # right_foot     ← COCO right_ankle
    h[..., 11, :] = coco_kps[..., 6, :]   # right_shoulder ← COCO right_shoulder
    h[..., 12, :] = coco_kps[..., 8, :]   # right_elbow    ← COCO right_elbow
    h[..., 13, :] = coco_kps[..., 10, :]  # right_hand     ← COCO right_wrist
    h[..., 14, :] = coco_kps[..., 5, :]   # left_shoulder  ← COCO left_shoulder
    h[..., 15, :] = coco_kps[..., 7, :]   # left_elbow     ← COCO left_elbow
    h[..., 16, :] = coco_kps[..., 9, :]   # left_hand      ← COCO left_wrist

    return h


def normalize_2d(kps_2d: np.ndarray, image_size: tuple) -> np.ndarray:
    """
    Normalize pixel coords to match MotionAGFormer training convention.

    Formula (from DataReaderH36M.read_2d):
        x_norm = px / W * 2 - 1
        y_norm = py / W * 2 - H / W    (note: both divided by WIDTH)

    Args:
        kps_2d:     (17, 2) pixel coords
        image_size: (H, W)

    Returns:
        (17, 2) normalized coords
    """
    H, W = image_size
    norm = kps_2d.copy().astype(np.float32)
    norm[:, 0] = norm[:, 0] / W * 2 - 1
    norm[:, 1] = norm[:, 1] / W * 2 - H / W
    return norm


def run_lifting(model, pose_keypoints: dict) -> dict:
    """
    Run MotionAGFormer 2D → 3D lifting on a single-frame POSE_KEYPOINTS dict.

    Steps:
      1. COCO 17-joint → H36M 17-joint remap  (via coco_to_h36m)
      2. Normalize pixel coords (both axes by image width)
      3. Append confidence channel (1.0 where detected, else 0.0)
      4. Forward through model  (T=1, padded to n_frames internally)

    Returns a POSE_3D dict:
        joints_3d:   (17, 3) float32  — H36M 3D positions (normalized space)
        joint_names: list[str]        — H36M joint names (17 entries)
    """
    coco_kps    = pose_keypoints["keypoints"]    # (17, 2) pixel coords
    coco_scores = pose_keypoints["scores"]       # (17,)
    image_size  = pose_keypoints["image_size"]   # (H, W)

    h36m_kps    = coco_to_h36m(coco_kps[np.newaxis])[0]          # (17, 2)
    norm_kps    = normalize_2d(h36m_kps, image_size)              # (17, 2)

    # Remap confidence scores with the same COCO→H36M mapping.
    # Averaged joints get the average of their source scores.
    scores_2d   = np.stack([coco_scores, coco_scores], axis=-1)   # (17, 2) dummy xy
    h36m_scores = coco_to_h36m(scores_2d[np.newaxis])[0, :, 0]   # (17,)
    conf        = (h36m_scores > 0).astype(np.float32)

    poses_2d = np.concatenate([norm_kps, conf[:, None]], axis=-1)  # (17, 3)
    poses_2d = poses_2d[np.newaxis]                                 # (1, 17, 3) — T=1

    poses_3d = model.lift(poses_2d)                                 # (1, 17, 3)

    joints_3d = poses_3d[0].copy()  # (17, 3)
    # MotionAGFormer outputs image-space coords: Y increases downward.
    # Negate Y so the skeleton is upright in standard 3D space (Y-up).
    joints_3d[:, 1] = -joints_3d[:, 1]

    return {
        "joints_3d":   joints_3d,            # (17, 3)
        "joint_names": H36M_JOINT_NAMES,
    }
