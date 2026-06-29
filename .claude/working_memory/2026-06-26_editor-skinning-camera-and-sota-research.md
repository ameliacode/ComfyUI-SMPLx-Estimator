# Editor skinning + camera, and SMPL-X SOTA research (2026-06-26)

## Context
Resumed from 2026-06-25 SMPL-X work. Pipeline currently:
ClickPose(COCO-17, HITL) -> MotionAGFormer lift(H36M-17 3D) -> fit_smplx_3d (Umeyama+IK, betas=0) -> SMPLXEditor.
All headless tests pass (test_joint_map/fit_3d/edit_p4). Server verified up on :8188, SMPLXFit + SMPLXEditor registered.
NOTE: render.py/fitting.py/SMPLXEditor outputs (pose/depth/normal/canny) are newer than the 06-25 memory file.

## SOTA research (saved: sources/research_single_image_smplx_sota_2026-06.md)
- Current SMPLXFit = pragmatic, NOT SOTA (betas=0 ceiling; MAF is a video lifter used off-label; 2-stage error).
- 2025-26 SOTA: SAM 3D Body (Meta, Nov 2025) promptable by 2D keypoints+masks, predicts shape, but outputs MHR (not SMPL-X), "SAM License". SMPLest-X (TPAMI 2025, SMPL-X). Multi-HMR (ECCV24). NLF (2024).
- Existing ComfyUI nodes already wrap SAM3DBody (PozzettiAndrea/ComfyUI-SAM3DBody: mesh+keypoints+rigged FBX; tori29umai0123 utils).

## DECISIONS (user, 2026-06-26)
- DO NOT integrate / recreate SAM3DBody. Stay on SMPL-X (SMPLXFit).
- Build the editor RIG-AGNOSTIC: operate on generic (verts + bone hierarchy + per-vertex skin weights),
  fed by SMPL-X for now. Future-proof, not hard-wired to SMPL-X internals.
- Project moat = the skinned, camera-aware HITL editor (NOT recovery accuracy).

## TODO #1 — Skinned editing (mesh deforms with skeleton)
Now: js/viewer_pose3d.html bodyMesh is a static THREE.Mesh; bones are separate cylinders -> independent.
Plan: THREE.SkinnedMesh + THREE.Skeleton.
- Server emits: per-vertex skinIndex/skinWeight (SMPL-X lbs_weights 10475x55, top-4 per vertex),
  bone hierarchy (parents/kintree), bind-pose joint positions.
- Browser: build bones, attach weights -> drag joint rotates bone -> GPU LBS deforms mesh LIVE (no round-trip).
- Keep server resolve_edit IK as the AUTHORITATIVE final SMPLX re-solve; browser skinning = live preview.

## TODO #2 — Camera in editor -> render in camera view
Now: render.py:38-40 hardcoded front look_at, decoupled from editor OrbitControls camera.
Plan: capture editor camera (world matrix + target + fov) into POSE3D_CORRECTIONS; render_maps() uses it.
- Toggle: "input camera" (fit's predicted perspective cam, aligns with source photo for ControlNet) vs
  "free editor camera". Predicted camera = better default for ControlNet pipeline.

## Files in scope
- modules/smplx_fit/{model,fitting,render}.py ; nodes/smplx_nodes.py
- js/viewer_pose3d.html (SkinnedMesh + camera capture) ; js/extension.js (data plumbing)

## Status: IMPLEMENTED (headless-verified) — awaiting USER UI verification of drag+camera.

### What was built (2026-06-26)
- NEW modules/smplx_fit/skin.py: body_skin_weights(model, topk=4). Folds SMPL-X lbs_weights
  (10475x55) onto 22 body joints by walking model.parents (hands->wrist, jaw/eyes->head),
  top-4 per vertex, rows renormalised. Cached per model id.
- render.py render_maps(..., camera=None): when camera={eye,at,up,fov} given, uses
  look_at_view_transform(eye,at,up)+fov (editor viewpoint) instead of hardcoded front view.
  Using eye/at/up sidesteps three.js<->PyTorch3D convention math.
- nodes/smplx_nodes.py: _smplx_payload adds data["skin"]={indices,weights}; SMPLXEditor has new
  optional "camera" STRING input (in IS_CHANGED too), parses it, passes cam to render_maps and
  skin to payload. body_skin_weights computed each edit (cheap, cached).
- js/viewer_pose3d.html: stores meshBaseVerts + skinIndices/Weights(flattened) + bindJoints;
  applySkinning() does v=base+Σw*(joint_now-joint_bind), called on TransformControls objectChange
  (live mesh follows dragged joints). sendCamera() posts POSE3D_CAMERA{eye,at,up,fov} on
  controls 'end' + wheel + after setPoseEditor.
- js/extension.js: POSE3D_CAMERA handler writes node "camera" widget (mirrors corrections).
- NEW tests/test_skin.py (gate): shape/range/normalised + hand-mass-folds-to-wrists geometric check.

### Verification done
- py_compile OK; viewer inline JS + extension.js node --check OK.
- All 4 headless tests pass (test_joint_map/fit_3d/edit_p4/skin).
- Camera render path: front vs side differ, both non-empty (headless GPU render).
- dev_refresh.sh PASS; object_info shows SMPLXEditor optional inputs ['corrections','camera'],
  SMPLXFit present. Server left RUNNING on :8188.

### OPEN / user to verify in UI (browser hard-refresh Ctrl+Shift+R first)
- [ ] Drag a joint -> mesh should now deform live (soft-skinning), not stay frozen.
- [ ] Orbit the 3D view, re-queue -> output pose/depth/normal/canny render from that viewpoint.
- [ ] CAVEAT: PyTorch3D NDC has +X to the left; the camera render MAY be horizontally mirrored
      vs the editor view. If so, fix = flip output images horizontally (np.fliplr) or negate eye/at X.
      Verify orientation against the editor before trusting it.
- [ ] Soft-skin is a LINEAR preview (translation-blended); the authoritative deform is still the
      server IK on release. Large drags may look slightly stretchy until re-solve — expected.
