# Session 2026-06-29 — Camera render + live mesh skinning

## User requests
1. Add a camera to render pose / depth / normal / canny.
2. When the user fixes (drags) a joint, the skinned SMPL-X mesh should move too.

## Findings — both already implemented in WIP code
- **Feature #1 (camera render): COMPLETE.**
  - `SMPLXEditor` (nodes/smplx_nodes.py) has optional `camera` input + outputs `pose/depth/normal/canny` (4 IMAGE).
  - `viewer_pose3d.html` `sendCamera()` posts `POSE3D_CAMERA` (eye/at/up/fov) on orbit-end + after pose load.
  - `js/extension.js` writes it into the node's `camera` widget.
  - `modules/smplx_fit/render.py render_maps(..., camera=cam)` feeds eye/at/up to pytorch3d `look_at_view_transform`.
  - Verified standalone: render produces 4 maps; explicit camera dict yields a DIFFERENT view (camera applied). ✓

- **Feature #2 (live mesh skinning): had a one-line relay bug, NOW FIXED.**
  - Server sends `skin` (top-k body-joint weights) in `_smplx_payload`; viewer has full soft-skinning (`applySkinning`).
  - BUG: `js/extension.js` `setPoseData()` rebuilt `currentPose` field-by-field and **dropped `skin`** → viewer's `applySkinning()` always early-returned → mesh stayed rigid.
  - FIX: added `skin: pose3dData.skin || undefined` to the relayed object (js/extension.js ~line 561).
  - Verified `body_skin_weights` computes correctly (rows sum to 1). ✓

## Changes made
- `js/extension.js`: relay `skin` through `setPoseData` (fix for feature #2).
- `nodes/smplx_nodes.py`: added hand grasp.

## Feature #3 — Hand grasp (added this session)
- SMPL-X loaded with `use_pca=False, flat_hand_mean=True` → zero hand_pose = flat/open.
- The model still exposes `np_left_hand_components`/`np_right_hand_components` (6×45 MANO PCA).
- Component 0 (positive) is the open→close axis. Empirically the fist bottoms out at
  ~4.0× (fingertip→wrist distance min); beyond that fingers over-curl. → `GRASP_SCALE=4.0`.
- Added `left_grasp`/`right_grasp` FLOAT inputs (0..1) to `SMPLXEditor` (optional).
- `_apply_grasp()` sets `left/right_hand_pose = grasp*4.0*component[0]` before `_forward_mesh`,
  so the grasp shows in the rendered maps (pose/depth/normal/canny) AND the viewer mesh,
  and is carried in the output SMPLX dict for downstream export.
- Verified: grasp=1 → ~12.5cm fingertip curl; grasp=0.5 → ~8cm. Node registers the inputs.
- NOTE: grasp is a node slider (re-queue to apply), not a live 3D drag — hands aren't
  draggable joints. Live hand posing in the viewer would need MANO components in JS (future).

## Feature #4 — Image-based hand estimation (added this session)
User asked "is hand estimation working?" → it WASN'T: COCO-17 (ClickPose) has no
finger keypoints, so SMPL-X hands were always flat. Added a whole-body detector path.
- **Assets (on disk, local-only):** `vitpose-l-wholebody.onnx` at
  `/home/wswg3/github/ComfyUI/ComfyUI/models/onnx/`; `onnxruntime` installed.
  NOTE: onnxruntime-gpu CUDA libs are NOT installed (libcublasLt.so.12 missing) →
  detector runs on CPU (default device=cpu). One image is fine.
- **modules/wholebody/vitpose.py:** ViTPose top-down inference. bbox→256x192 crop,
  ImageNet norm (RGB [0,1]), DARK heatmap decode (vendored from mmpose/Alibaba Wan,
  Apache-2.0). Output: COCO-WholeBody 133 kps. Hands = idx 91-111 (L), 112-132 (R).
- **modules/wholebody/hands.py:** per-finger CURL from 2D straightness ratio
  (chord/path; open≈0.97→curl0, fist≈0.45→curl1). `curls_to_hand_pose` maps curls
  to 45-dim hand_pose via per-finger slice of MANO PCA comp[0] (×GRASP_SCALE=4.0).
  Finger→block: index(0:9) middle(9:18) pinky(18:27) ring(27:36) thumb(36:45)
  (verified vs smplx JOINT_NAMES[25:40]).
- **nodes/wholebody_nodes.py — `WholeBodyHandDetector`:** IMAGE (+optional ClickPose
  POSE_KEYPOINTS for person bbox) → HAND_KEYPOINTS + preview. device combo (cpu/cuda).
- **SMPLXFit:** new optional `hand_keypoints` input → `_apply_estimated_hands` sets
  hand_pose from curls (flows to mesh/render/output SMPLX dict).
- **SMPLXEditor:** `_apply_grasp` now an OVERRIDE — grasp 0 keeps the incoming
  (estimated) hand, >0 forces a uniform fist. So estimation survives into the editor.
- **Wiring:** WholeBodyHandDetector → SMPLXFit.hand_keypoints. Pass ClickPose's
  POSE_KEYPOINTS into the detector too for a tight crop.
- **Verified:** kps land on hands (clasped-hands test img); real-image curls →4.7cm
  mesh deform; forced full curl →12.5cm; per-finger isolation correct. New gate
  `tests/test_wholebody_hands.py` passes; all 4 existing tests pass; 6 nodes register.
- **Limitation (camera-free):** captures finger FLEXION only — not abduction/spread,
  thumb opposition, or out-of-plane pointing. Upgrade path = weak-perspective
  reprojection IK on the 2D hand kps (would add a camera to the fit stage).
- Updated `scripts/dev_refresh.sh` EXPECTED_NODES to all 6 nodes.

## Feature #5 — Finger-joint editing (IK) + direct SMPL-X parameter editing
User: editor had NO hand joints (only 22 body joints sent); also wanted to edit
SMPL-X params (betas, expression+jaw, global orient+transl). Built both.
- **Editable joints now 52:** body 0-21 + fingers 25-54 (jaw/eyes 22-24 skipped).
  `joint_maps.EDITABLE_JOINTS` + `editable_limbs(parents)` (finger base -> wrist).
- **Skin over fingers:** `skin.editable_skin_weights` (top-k LBS in 0-54 idx space,
  jaw/eyes folded to head) so finger drags deform the mesh live.
- **Hand IK:** `fitting.resolve_hand_edit` optimizes left/right hand_pose to finger
  targets (idx 25-54), body/global/transl/betas frozen. Verified: drag joint27
  4cm->1.5cm.
- **edit() split:** corrections idx<22 -> resolve_edit (body_pose); 25<=idx<55 ->
  resolve_hand_edit (hand_pose). `_forward_mesh` now returns joints too; joints_3d
  recomputed post-edit so handles match the posed mesh.
- **Param widgets (optional) on SMPLXEditor:** betas (CSV, absolute), expression
  (CSV), jaw_open (0..1 -> jaw pitch *0.5), rot_x/y/z (deg OFFSET, composed via
  cv2.Rodrigues), tx/ty/tz (m OFFSET). `_apply_params` applied BEFORE IK so IK uses
  edited shape. grasp sliders stay as hand override.
- **Payload:** `_smplx_payload` now sends 55 joints + `editable` list + full
  editable limbs + editable skin (0-54).
- **Viewer (js/viewer_pose3d.html):** sparse jointSpheres keyed by 0-54; finger
  handles smaller+cyan (fingerGeo/matFinger); childrenMap from limbs; beginDrag
  snapshots descendants, dragPropagate rigidly carries the sub-chain (wrist drag
  moves fingers); applySkinning reads ALL handle positions vs bind; raycast uses
  pickableSpheres. extension.js setPoseData passes `editable` through.
- **Verified:** integration test of SMPLXEditor.edit (finger drag + betas + 30deg Y
  + jaw) returns 5 outputs, applies all, payload has editable=52/joints=55/skin/
  limbs=51. JS syntax clean (node --check). All 5 tests pass; 6 nodes register.
- **To activate:** browser hard-refresh (JS changed).

## NEXT (agreed direction) — SOTA SMPL-X backbone, path B
Architecture user committed to: 2D pose (ClickPose) -> 2D HITL edit -> 3D estimate
(consuming corrected 2D) -> 3D HITL (SMPLXEditor). Keeps ClickPose meaningful.
Plan: replace MotionAGFormer lift with a SOTA SMPL-X regressor INIT (real betas+
hands+face) + a 2D reprojection refine to the corrected ClickPose keypoints (so 2D
edits bite) -> needs a weak-persp camera in the fit. Output quality ranking
(excl. SAM 3D Body=MHR): **SMPLest-X** best benchmark fidelity; **NLF** best
in-the-wild + global, and **NLF weights already on disk at ComfyUI/models/nlf/**.
First step next session: inventory models/nlf/ + check for SMPLest-X weights
(local-only) before wiring. SAM 3D Body is the only keypoint-promptable SOTA but
outputs MHR + SAM license.

## Feature #6 — NLF SOTA one-pass SMPL-X estimator (DONE)
User chose "Download NLF" for a one-pass SOTA backbone. KEY FINDING: NLF's released
torchscript outputs **SMPL, not SMPL-X** (detect_smpl_batched -> pose72/betas10/
trans3/6890 verts/24 joints; no detect_smplx_batched).
- **Model downloaded:** `ComfyUI/models/nlf/nlf_l_multi_0.3.2.torchscript` (493MB) from
  github.com/isarandi/nlf/releases v0.3.2. Loader reference: ComfyUI-WanVideoWrapper/
  MTV/nodes.py. MUST `import torchvision` before jit.load (registers torchvision::nms).
  Input = uint8 (0-255) BCHW; offload model to CPU after inference (GPU is tight:
  NLF+SMPL-X+pytorch3d co-resident OOMs).
- **SMPL->SMPL-X mapping (modules/nlf/estimate.py):** global_orient=pose[:3],
  body_pose=pose[3:66] (SMPL body joints 1-21 == SMPL-X body joints 1-21), drop
  pose[66:72] (SMPL hand-root). **Frame fix: rotate 180° about X** (_RFIX) — NLF is
  OpenCV cam (Y-down); our renderer/model is Y-up (verified: was upside-down, fixed).
  CAVEATS: betas DON'T transfer (SMPL!=SMPL-X shape basis) -> betas=0 neutral (edit
  via SMPLXEditor betas widget, or add official SMPL->SMPL-X transfer later); no
  hands/face from NLF -> use WholeBodyHandDetector or flat.
- **Node (nodes/nlf_nodes.py) `NLFSMPLXEstimator`:** IMAGE (+optional hand_keypoints)
  -> SMPLX + preview. Reuses smplx_nodes._forward_mesh/_ground/_img/_apply_estimated_hands.
- **Verified:** render matches the photo pose (clasped-hands portrait, upright);
  full chain NLF->SMPLXEditor returns 5 outputs, editable=52, joints=55. All 5 tests
  pass; 7 nodes register.
- **Also fixed:** prestartup_script.py had a stray `does` token (NameError, pre-existing,
  unrelated) -> removed; PRESTARTUP no longer fails.

## Hand-detector person crop (bbox) — both pipelines
WholeBodyHandDetector crops a tight person box for ViTPose. Sources (priority):
  1. `bbox` input (BBOX [x0,y0,x1,y1]) — e.g. from NLFSMPLXEstimator's new `bbox` output.
  2. `pose_keypoints` (ClickPose) — derived box.
  3. else whole image.
NLFSMPLXEstimator now returns (smplx, preview, **bbox**); estimate_smplx_params
returns the chosen person's box (image px, no frame fix). Verified NLF bbox -> hand
detector works (tight crop, hands detected). So the ClickPose-less NLF pipeline can
still feed a tight crop to the hand detector.

## Two pipelines now coexist (both -> SMPLXEditor)
- **SOTA one-pass:** image -> NLFSMPLXEstimator (+WholeBodyHandDetector) -> SMPLXEditor.
  Best pose, neutral shape, 2D edit not used.
- **HITL 2D-first (legacy):** ClickPose -> (2D edit) -> SMPLXFit -> SMPLXEditor.
NOTE the user's stated ideal was "2D edit then 3D" (keeps ClickPose); the one-pass NLF
path does NOT use ClickPose's 2D edits. If they want both, NEXT would be a 2D
reprojection-refine on the NLF init (needs a camera). Possible future: SMPL->SMPL-X
betas transfer for real shape.

## Suggested graph wiring
ClickPose ─┬─ POSE_KEYPOINTS ─────────────► SMPLXFit ─► SMPLXEditor
           └─ (image) ─► WholeBodyHandDetector ─ HAND_KEYPOINTS ─► SMPLXFit.hand_keypoints
(feed ClickPose POSE_KEYPOINTS into the detector too for a tight person crop)

## To activate
- JS change → browser hard-refresh (Ctrl+Shift+R). No server restart needed.

## Notes / loose ends
- `WEB_DIRECTORY = "./js"` is the served frontend. `web/` holds STALE duplicates (pose_editor.js, viewer_pose3d.html) from the old OpenPose approach — not served; candidate for deletion.
- `scripts/dev_refresh.sh` node list is stale (checks ClickPose/MotionAGFormer/3D Pose Editor only; not SMPLXFit/SMPLXEditor). Import still passes.
- Possible polish (untested in-browser): pytorch3d NDC has +X to the left, so the rendered maps may be horizontally mirrored vs the three.js viewport. Verify visually; flip in render_maps if needed.
</content>
</invoke>
