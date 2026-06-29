# SMPL-X pipeline implementation (2026-06-25)

Replaced MotionAGFormer with cold VPoser-regularized SMPLify-X + IK joint editing.
BVH dropped per user. Decision: Option A (cold solve, zero downloads); init model
(GATOR) rejected — emits vertices, needs missing SMPL .pkl.

## Done (P0-P4, all tested)
- **P0**: loguru installed; human_body_prior loaded via sys.path (NOT pip — torch pin
  conflict; keeps ClickPose venv safe). VPoser load_model returns tuple (unpacked).
  smplx.create needs PARENT dir (appends smplx/SMPLX_NEUTRAL.npz). joint_maps.py BY-NAME
  (12 COCO->SMPLX, nose dropped). tests/test_joint_map.py HARD GATE — name + geometric
  (pose left arm -> only left_wrist moves) = no L/R swap.
- **P1**: camera.py (perspective f=max(H,W)) + fitting.py reprojection + Stage-0 warmup.
  test_fit_p1: synthetic pose recovered at 1.34px rmse.
- **P2**: VPoser latent solve (encode .mean warm-start, annealed prior), betas frozen=0.
  test_fit_p2: 9px rmse, 5.9s (<8s), bit-identical determinism.
- **P3**: nodes/smplx_nodes.py SMPLXFit (POSE_KEYPOINTS->SMPLX+IMAGE), IS_CHANGED, preview
  render. Registered. ClickPose still clean.
- **P4**: resolve_edit() RAW body_pose IK (vposer=None) => LOCALIZED edits (wrist to
  target 0.1cm, ankle 0.0cm, root frozen). SMPLXEditor node. js/viewer_pose3d.html editor
  mode: draggable spheres + TransformControls + POSE3D_CORRECTIONS, normalizePose BYPASSED
  (true metric). extension.js: SMPLXEditor binding + setPoseData passes editorMode+limbs.
  Both JS files node --check clean.

## Key files
- modules/smplx_fit/{__init__,joint_maps,model,camera,fitting}.py
- nodes/smplx_nodes.py ; nodes/__init__.py + __init__.py (registered SMPLXFit, SMPLXEditor)
- js/viewer_pose3d.html (editor mode) ; js/extension.js (SMPLXEditor ext)
- tests/test_{joint_map,fit_p1,fit_p2,edit_p4}.py — run with venv python (no pytest)

## Graph: ClickPose -> SMPLXFit -> SMPLXEditor (drag) ; SMPLXFit.preview -> PreviewImage

## Runtime fixes after first ComfyUI run
- **inference_mode**: ComfyUI runs nodes under torch.inference_mode() (execution.py:720) ->
  autograd dies. Fix: load_smplx/load_vposer build under `inference_mode(False)`; fit_smplx/
  resolve_edit wrap body in `inference_mode(False)+enable_grad()`. See lesson_comfyui_inference_mode_autograd.
- **upside-down body** (user: "NOT SMPL RESULT"): SMPL-X is Y-UP, image is Y-DOWN. camera.py
  had no Y-flip -> optimizer rotated global_orient ~180deg about X -> body inverted in 3D.
  Fix: perspective_project v = cy - f*Y/Z (+ init_translation ty sign). Now upright,
  global_orient ~0. GUARD: tests/test_fit_upright.py (realistic image-space kps; asserts
  head_y>pelvis_y>ankle_y + |orient|<90). The synthetic self-consistent tests could NOT
  catch this — always validate with realistic image-space keypoints, not model-projected GT.

## ARCHITECTURE PIVOT (user): SMPLXFit is now MotionAGFormer-based, VPoser DELETED
- New flow: ClickPose COCO-17 -> MotionAGFormer lift (H36M-17 3D) -> fit_smplx_3d
  (Umeyama-align lifted joints onto SMPL-X rest pose -> 3D-joint IK, betas=0). NO camera
  reprojection, NO VPoser. 3D joints constrain the pose -> no depth/Y-flip/front-back issues.
- New: modules/smplx_fit/align.py (umeyama); joint_maps.build_h36m_to_smplx (BY NAME, swap-
  guarded); fitting.fit_smplx_3d. SMPLXFit inputs: pose_keypoints, smplx_model_path,
  maf_checkpoint_path, gender, device, iters, seed.
- DELETED: VPoser (model.load_vposer + human_body_prior dep), the 2D reproj path
  (fit_smplx/_fit_smplx_impl/gmof/_reproj_loss), camera.py, tests test_fit_p1/p2/upright.
  resolve_edit no longer has a vposer option (raw body_pose only).
- Tests now: test_joint_map, test_fit_3d, test_edit_p4 (all pass).
- VERIFIED end-to-end: real MAF ckpt lift -> fit_smplx_3d on realistic kps -> upright body,
  per-joint rmse ~6.4cm (residual = H36M vs SMPL-X bone proportions w/ betas frozen; OK).
  global_orient small. Editor (mesh + drag IK) unchanged and still works.
- Note: human_body_prior/loguru installs from earlier are now unused (harmless; left in venv).

## NOT done / open
- Browser drag interaction UNTESTED (needs UI — picking, TransformControls gizmo). User to verify.
- web/ dead duplicate (viewer_pose3d.html, pose_editor.js) NOT deleted — surfaced to user, untracked, WEB_DIRECTORY=js/.
- P6 cleanup (remove MotionAGFormer/Pose3DEditor from default graph) — left registered, harmless.
- Hardcoded asset paths (DEFAULT_SMPLX_PARENT, DEFAULT_VPOSER_EXPR, HUMAN_BODY_PRIOR_PATH) in model.py — node inputs override.
- Caveat: betas=0 (no per-subject shape); single-image depth/scale weakly constrained.
