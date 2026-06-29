# Session progress — refinement diagnosis + dev test loop (2026-06-25)

## Done this session
1. **Diagnosed ClickPose refinement "not working" on ComfyUI.** Root cause = stateful
   decoder-only refinement collides with ComfyUI's stateless-node + global model cache:
   - `model._last_out` / `_last_encoder_memory` live on the **shared cached model**
     (`editpose_nodes.py:241-247`); any other `detect()` clobbers them → stale-state guard
     (`inference.py:92-104`) silently falls back to **direct override** (joint-snap, no model
     refinement).
   - First correction after state loss is **dropped** (`editpose_nodes.py:256` requires
     `_last_pose_keypoints is not None`); IS_CHANGED then caches the uncorrected result.
   - All error paths swallowed → user can't tell if real refinement ran.
   - Fidelity: refine is **non-iterative** (always from raw detection, box not updated,
     best-person index can differ from detect) vs reference `human_feedback_loop_*`
     (`vendor/Click-Pose/models/clickpose/clickpose.py:412-586`).
   - NOTE: the `transformer()` call signature in `model.py:264-275` is an **exact match**
     to the reference — the port is algorithmically correct for ONE click. Problem is state, not math.
   - Proposed fix (NOT yet implemented): move encoder memory into `CLICK_POSE_STATE` keyed by
     `image_id`; make fallback loud; never drop the first correction; optionally iterative refine.

2. **Built develop→refresh test loop:** `scripts/dev_refresh.sh`
   - restart tester ComfyUI (`~/github/ComfyUI`, venv py3.10) → wait for `/system_stats` 200 →
     check no IMPORT FAILED / no traceback frame inside `custom_nodes/comfyui-mocap/*.py` →
     confirm `ClickPose`, `MotionAGFormer`, `3D Pose Editor` via `/object_info` (jq).
   - Flags: `--tests` (pytest, graceful skip), `--no-start` (verify running server only).
   - Validated: full restart PASS + `--no-start` PASS, all 3 nodes registered. Tightened the
     traceback heuristic so other nodes' failures (xatlas, comfy_dynamic_widgets) don't false-trip.
   - Wired into `.claude/CLAUDE.md` Key commands as the canonical post-change step.

## Workflow rule going forward
After every Python change to the node: run `./scripts/dev_refresh.sh` (refresh + verify).
JS-only changes: browser hard-refresh (Ctrl+Shift+R), no restart.

## Refinement fix — IMPLEMENTED (2026-06-25)
- `model.py`: added `get_refine_state()` (snapshots out/encoder_memory/image_id); `refine()` now
  takes `refine_state` explicitly instead of reading `self._last_*` → immune to the shared cached
  model being re-run on another image (#1 fixed).
- `editpose_nodes.py`: `run()` captures refine_state immediately after detect; re-detects when
  state is missing and applies corrections in the SAME run → first correction never dropped (#2);
  persists BASE detection (not refined) for stable from-detection refinement.
- `inference.py`: `_log_fallback()` makes every degradation LOUD (log + stdout); results carry
  `refine_method` = model_refine | direct_override | detect (#3). JS already clears stale
  corrections on image change (pose_editor.js:609-615), so always-apply is safe.
- Validated: py_compile OK; `dev_refresh.sh` PASS (all 3 nodes register); 5-case smoke test of
  apply_corrections_with_state all correct (model_refine / direct_override / loud fallbacks / passthrough).
- ComfyUI left RUNNING for user testing (port 8188).
- NOT done: #4 iterative multi-click fidelity (follow-up); real decoder math untested headless
  (needs checkpoint+GPU) — user testing in UI.

## Next / open
- [ ] (#4) iterative multi-click refine: feed prior refined kps + updated box for closer reference fidelity.
- [ ] Add real pytest tests so `--tests` becomes meaningful (none exist yet; pytest not in venv).
- [ ] Pre-existing non-fatal log noise: "Torch already imported" warning + comfy-env "env not
      built yet" — node still loads fine; revisit only if it bites.
