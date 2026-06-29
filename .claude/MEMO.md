# Memo

## 2026-04-14

### ClickPose detector/editor state
- `CLICK_POSE_STATE` is now image-scoped, not just model-scoped.
- `ClickPoseModel.detect()` stamps an `image_id` onto the detection result and keeps the same `image_id` in the model cache used by `refine()`.
- `ClickPoseDetector.detect()` forwards that `image_id` into `CLICK_POSE_STATE`.
- `apply_corrections_with_state()` now refuses decoder-only refine when:
  - `pose_keypoints.image_id`
  - `click_pose_state.image_id`
  - `model._last_image_id`
  do not all match.
- Fallback in that stale-state case is direct joint override, which prevents a new image from accidentally reusing encoder memory from a previous image.
- Preferred Stage 1 workflow is now the combined `ClickPose` node:
  - first queue for an image: detect
  - later queues for the same image with corrections: refine
  - new image: stale corrections are dropped and detect runs again
- Legacy split nodes `ClickPose Detector` and `ClickPose Editor` were removed from node registration so only `ClickPose` shows in the UI.

### 3D editor bridge
- `ComfyUI_3dPoseEditor` (`Hina.PoseEditor3D`) is not wireable from `POSE_3D`.
- That node has no pose input socket; it only reads filenames from `temp/3dposeeditor` widget dropdowns.
- Added `Pose 3D Editor Export` to bridge our `POSE_3D` output into that ecosystem:
  - returns `pose/depth/normal/canny` images directly
  - writes `*_pose.png`, `*_depth.png`, `*_normal.png`, `*_canny.png` into `temp/3dposeeditor`
- This is interoperability, not true direct graph connection. A real direct connection would require changes inside the Hina node itself.
