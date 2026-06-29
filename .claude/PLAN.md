# ComfyUI Motion Capture Plan

## Development Phases

| Phase | Input | Output | Key Models |
|---|---|---|---|
| **Phase 1 (current)** | Single image | 3D pose BVH | ClickPose + MotionAGFormer |
| **Phase 2 (future)** | Video | Global 3D pose BVH | ClickPose (per-frame fix) + GVHMR (or SOTA replacement) |

---

## Phase 1 — Current Goal
Single image → 3D pose → BVH file

**Pipeline:**
```
Image → ClickPose Detector → ClickPose Editor (HITL) → MotionAGFormer Lifter → BVH Exporter
```

---

## Stage 1: 2D Pose + HITL Editor ← **CURRENT FOCUS**

### Overview
- ClickPose runs encoder + decoder → 17-joint COCO keypoints
- User can drag mis-detected joints in the canvas editor
- Only the human-to-keypoint decoder re-runs with corrected joints frozen (`delta_mask`)
- Encoder memory is cached in `CLICK_POSE_STATE` — encoder never runs again on re-queue

### Node A: `ClickPoseDetector`
- Input: `IMAGE`, checkpoint path, device
- Runs full encoder + decoder
- Output: `POSE_KEYPOINTS`, `CLICK_POSE_STATE` (model ref + cached encoder memory), `IMAGE` (visualization)
- Sends `kps_json` + image URL to JS canvas via `ui` dict on `onExecuted`

### Node B: `ClickPoseEditor`
- Input: `IMAGE`, `POSE_KEYPOINTS`, optional `CLICK_POSE_STATE`, hidden `corrections` STRING widget
- JavaScript canvas widget (`web/js/pose_editor.js`):
  - Modal popup: image background + COCO-17 joints as draggable colored dots + skeleton lines
  - Green = confident, Red = low-confidence, Blue = user-dragged
  - "Apply" sends `{joint_idx: [x, y]}` JSON into the hidden `corrections` widget
  - Thumbnail rendered inside node for quick preview
- Python `edit()` method:
  - No corrections → pass through unchanged
  - Corrections present → call `model.refine(corrections, image_size)` → returns refined keypoints
- Output: `POSE_KEYPOINTS`, `IMAGE` (updated visualization)

### 2D Pose Editor UI decision
Researched all known ComfyUI pose editors (2026-04):
- `ComfyUI-OpenPose-Studio` — COCO-18 only, cannot edit COCO-17
- `ComfyUI-ultimate-openpose-editor` — COCO-18 only
- `ComfyUI-OpenPoser` — COCO-18 only
- `comfyui-2dpose-editor` — 3D mannequin rig, not image overlay
- **Decision: build our own** (already done — `web/js/pose_editor.js`)

### Step-by-step checklist for Stage 1
- [ ] 1. Verify `ClickPoseModel.load()` and `run_detection()` work end-to-end
- [ ] 2. Verify `ClickPoseDetector.detect()` returns correct `CLICK_POSE_STATE` and sends `ui` dict
- [ ] 3. Verify JS canvas widget loads image and renders COCO-17 pose after detector runs
- [ ] 4. Verify drag → Apply writes corrected JSON to `corrections` widget
- [ ] 5. Verify `ClickPoseEditor.edit()` calls `model.refine()` when corrections present
- [ ] 6. End-to-end test: detect → drag joint → re-queue → refined keypoints change

---

## Stage 2: 3D Lifting with MotionAGFormer

- Input: `POSE_KEYPOINTS` from Stage 1
- Model: MotionAGFormer (2D→3D pose lifting)
- Single-frame mode (T=1)
- Output: `POSE_3D` (17 joints, 3D coords)

---

## Stage 3: BVH Export

- Input: `POSE_3D`
- Map COCO-17 → BVH skeleton hierarchy
- Compute bone rotations from joint positions
- Output: `.bvh` file to `output/mocap/`

---

## Active Nodes (Phase 1)

| Node | Class | Status |
|---|---|---|
| ClickPose Detector | `ClickPoseDetector` | exists, needs end-to-end test |
| ClickPose Editor | `ClickPoseEditor` | exists, needs end-to-end test |
| MotionAGFormer Lifter | `MotionAGFormerLifter` | exists, Stage 2 |
| Pose 3D Viewer | `Pose3DViewer` | exists, Stage 2 debug aid |
| BVH Exporter | `BVHExporter` | exists, Stage 3 |

### Removed nodes (OpenPose Studio approach abandoned)
- ~~`PoseKeypointsToOpenPoseJson`~~ — deleted
- ~~`PoseKeypointFromOpenPose`~~ — deleted

---

## Custom Types

- `POSE_KEYPOINTS` — `{keypoints: (17,2), scores: (17,), image_size: (H,W), bbox_score: float}`
- `CLICK_POSE_STATE` — `{model: ClickPoseModel, image_size: (H,W)}`
- `POSE_3D` — `{joints_3d: (17,3), joint_names: [str]}`

---

## File Structure

```
comfyui-mocap/
├── nodes.py                        # Node registrations
├── web/js/pose_editor.js           # COCO-17 canvas widget (custom, HITL)
├── modules/
│   ├── clickpose/
│   │   ├── model.py               # ClickPose wrapper (load + refine)
│   │   └── inference.py           # Detection logic, COCO_JOINT_NAMES
│   ├── motionagformer/
│   │   ├── model.py               # MotionAGFormer wrapper
│   │   └── inference.py           # 2D→3D lifting
│   └── bvh/
│       └── export.py              # 3D joints → BVH
└── vendor/Click-Pose/             # ClickPose source
```

---

## Phase 2 — Future: Video + Global Pose

### Goal
Video → per-frame 2D pose (with HITL correction) → global 3D pose → BVH

**Pipeline:**
```
Video → frame extraction → ClickPose (per-frame 2D) → [HITL editor per frame] → GVHMR → BVH export
```

### New Nodes (Phase 2)

| Node | Class | Inputs | Outputs |
|---|---|---|---|
| Video Loader | `VideoLoader` | `video_path`, `fps` | `IMAGE` batch |
| Batch Pose Detector | `ClickPoseBatchDetector` | `IMAGE` batch | `POSE_SEQUENCE` |
| Global 3D Estimator | `GVHMREstimator` | `IMAGE` batch, `POSE_SEQUENCE` | `GLOBAL_POSE_3D` |
| BVH Exporter (video) | `BVHSequenceExporter` | `GLOBAL_POSE_3D`, `filename`, `fps` | `BVH_PATH` |

### Custom Types (Phase 2)
- `POSE_SEQUENCE` — list of `POSE_KEYPOINTS` dicts, one per frame
- `GLOBAL_POSE_3D` — `{smpl_poses: (T,72), global_trans: (T,3), betas: (10,)}`
