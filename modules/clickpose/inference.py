"""
Stateless inference helpers for Click-Pose.

The caller owns the ClickPoseModel instance and passes it in.
"""

import json
import logging
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


def _log_fallback(reason: str) -> None:
    """Make refinement degradation observable.

    Direct override = snap the clicked joint to the cursor with NO model
    refinement of the other joints. This used to happen silently, so refinement
    looked 'weak'. Now every fallback is logged loudly (and surfaced via the
    'refine_method' key on the result).
    """
    msg = (
        "[editpose] WARN: DIRECT OVERRIDE (joint snap, no ClickPose refinement) — "
        + reason
    )
    log.warning(msg)
    print(msg)

# COCO 17-joint names (order matches Click-Pose / COCO keypoint output)
COCO_JOINT_NAMES = [
    "nose",  # 0
    "left_eye",
    "right_eye",  # 1, 2
    "left_ear",
    "right_ear",  # 3, 4
    "left_shoulder",
    "right_shoulder",  # 5, 6
    "left_elbow",
    "right_elbow",  # 7, 8
    "left_wrist",
    "right_wrist",  # 9, 10
    "left_hip",
    "right_hip",  # 11, 12
    "left_knee",
    "right_knee",  # 13, 14
    "left_ankle",
    "right_ankle",  # 15, 16
]


def apply_corrections_with_state(
    pose_keypoints: dict,
    click_pose_state: Optional[dict],
    corrections: Optional[str],
) -> dict:
    """
    Apply user-supplied joint corrections to a POSE_KEYPOINTS dict.

    Correction string format (JSON, joint index → [x, y] in absolute pixels):
        '{"0": [320, 95], "5": [210, 180]}'
        '{"nose": [320, 95], "left_shoulder": [210, 180]}'

    Behaviour:
    - No / empty / blank corrections → returns pose_keypoints unchanged.
    - Invalid JSON → returns pose_keypoints unchanged (logs a warning).
    - With click_pose_state → calls model.refine() for decoder-only refinement;
      falls back to direct override if refine() raises.
    - Without click_pose_state → directly patches the corrected joints.
    """
    if not corrections or not corrections.strip():
        return pose_keypoints

    try:
        overrides_dict = json.loads(corrections)
    except json.JSONDecodeError as exc:
        log.warning("ClickPoseEditor: corrections JSON is invalid (%s), skipping", exc)
        return pose_keypoints

    if not overrides_dict:
        return pose_keypoints

    # Normalise keys to str(int)
    numeric_overrides: dict = {}
    for key, xy in overrides_dict.items():
        if isinstance(key, str) and key.isdigit():
            numeric_overrides[str(int(key))] = xy
        elif key in COCO_JOINT_NAMES:
            numeric_overrides[str(COCO_JOINT_NAMES.index(key))] = xy
        else:
            log.warning("ClickPoseEditor: unknown joint key %r, skipping", key)

    if not numeric_overrides:
        return pose_keypoints

    # ── Decoder-only refinement (preferred) ───────────────────────────────────
    # refine_state (encoder memory + raw detection proposals) is captured per-image
    # in click_pose_state, so it is immune to the shared cached model being re-run
    # on a different image in a later queue.
    if click_pose_state is not None:
        model = click_pose_state["model"]
        img_size = click_pose_state["image_size"]
        refine_state = click_pose_state.get("refine_state")
        pose_image_id = pose_keypoints.get("image_id")
        state_image_id = click_pose_state.get("image_id")

        if pose_image_id and state_image_id and pose_image_id != state_image_id:
            _log_fallback(
                f"refine state belongs to a different image "
                f"(pose={pose_image_id} state={state_image_id})"
            )
        elif not refine_state or refine_state.get("out") is None:
            _log_fallback("no encoder/decoder state captured from detect()")
        else:
            try:
                refined = model.refine(refine_state, numeric_overrides, img_size)
                refined["refine_method"] = "model_refine"
                print(
                    f"[editpose] refine: ClickPose decoder pass "
                    f"({len(numeric_overrides)} corrected joint(s))"
                )
                return refined
            except Exception as exc:  # noqa: BLE001
                _log_fallback(f"model.refine() raised: {exc}")

    # ── Direct override fallback (LOUD — this is NOT model refinement) ─────────
    kps = pose_keypoints["keypoints"].copy()
    scores = pose_keypoints["scores"].copy()
    for idx_str, xy in numeric_overrides.items():
        idx = int(idx_str)
        if 0 <= idx < 17:
            kps[idx] = [float(xy[0]), float(xy[1])]
            scores[idx] = 1.0
    return {
        **pose_keypoints,
        "keypoints": kps,
        "scores": scores,
        "refine_method": "direct_override",
    }


def run_detection(model, image_np: np.ndarray) -> dict:
    """Run Click-Pose detection and return a POSE_KEYPOINTS dict."""
    return model.detect(image_np)


def apply_overrides(pose_keypoints: dict, overrides_json: Optional[str]) -> dict:
    """
    Apply per-joint overrides to a POSE_KEYPOINTS dict.

    Override format — any subset of joints, by name or index:
        {"nose": [x, y], "left_shoulder": [x, y]}
        {"0": [x, y], "5": [x, y]}

    Overridden joints get their score set to 1.0.
    Returns a new dict; the original is not mutated.
    """
    if not overrides_json or not overrides_json.strip():
        return pose_keypoints

    try:
        overrides = json.loads(overrides_json)
    except json.JSONDecodeError as e:
        raise ValueError(f"keypoint_overrides is not valid JSON: {e}")

    kps = pose_keypoints["keypoints"].copy()  # (17, 2)
    scores = pose_keypoints["scores"].copy()  # (17,)

    for key, xy in overrides.items():
        if isinstance(key, str) and key.isdigit():
            idx = int(key)
        elif key in COCO_JOINT_NAMES:
            idx = COCO_JOINT_NAMES.index(key)
        else:
            raise ValueError(
                f"Unknown joint key in overrides: '{key}'. "
                f"Use a joint name from COCO_JOINT_NAMES or an integer index 0-16."
            )

        if not (0 <= idx < 17):
            raise ValueError(f"Joint index {idx} out of range [0, 16].")
        if len(xy) != 2:
            raise ValueError(f"Joint override for '{key}' must be [x, y], got {xy}.")

        kps[idx] = [float(xy[0]), float(xy[1])]
        scores[idx] = 1.0  # manually corrected → treat as fully confident

    return {**pose_keypoints, "keypoints": kps, "scores": scores}
