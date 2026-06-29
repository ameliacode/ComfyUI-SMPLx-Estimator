# ComfyUI Mocap

ComfyUI custom node package: single image → 2D pose (ClickPose + HITL) → 3D lift (MotionAGFormer) → BVH export.

See `.claude/PLAN.md` for current stage and step-by-step checklist.

## Key commands

```bash
# Develop → refresh test loop — RUN THIS AFTER EVERY PYTHON CHANGE.
# Restarts the tester ComfyUI, waits for it to come up, and verifies the node
# imported cleanly + all 3 nodes are registered (via /object_info). Exit 0 = good.
cd /home/wswg3/project/comfyui-mocap && ./scripts/dev_refresh.sh
./scripts/dev_refresh.sh --tests     # also run pytest first (skips gracefully if none)
./scripts/dev_refresh.sh --no-start  # verify an already-running server only

# Run tests (must all pass before committing) — none exist yet
cd /home/wswg3/project/comfyui-mocap && pytest -v

# Manual restart (what dev_refresh.sh automates)
kill $(ps aux | grep 'venv/bin/python3.10 main.py' | grep wswg3 | awk '{print $2}')
cd /home/wswg3/github/ComfyUI && nohup venv/bin/python3.10 main.py --listen 0.0.0.0 --port 8188 > /tmp/comfyui.log 2>&1 &

# Check ComfyUI logs
strings /tmp/comfyui.log | grep -i "mocap\|error" | tail -20

# JS changes: browser hard refresh only (Ctrl+Shift+R) — no restart needed
```

## Key paths

| What | Path |
|---|---|
| ComfyUI root | `/home/wswg3/github/ComfyUI/` |
| Custom node (symlinked) | `custom_nodes/comfyui-mocap/` → `/home/wswg3/project/comfyui-mocap/` |
| Checkpoints | `checkpoints/ClickPose_model_only_R50.pth`, `checkpoints/motionagformer-l-h36m.pth.tr` |
| Checkpoint path in UI | `custom_nodes/comfyui-mocap/checkpoints/<filename>` |
| ClickPose vendor source | `vendor/Click-Pose/` (read-only) |
| ComfyUI log | `/tmp/comfyui.log` |

## Architecture rules (violations cause bugs)

- **COCO-17 only** — no COCO-18 anywhere in the pipeline
- **COCO→H36M mapping is direct** — no left/right swap. Both use anatomical left/right. Do not revert this.
- **ClickPose HITL**: encoder runs once in `ClickPoseDetector`, cached in `CLICK_POSE_STATE`. Only decoder re-runs in `ClickPoseEditor` when corrections present.
- **Checkpoint paths** in node UI are relative to ComfyUI working directory (`/home/wswg3/github/ComfyUI/`)

## Do not

- Modify anything under `vendor/` — read-only third-party source
- Add features beyond the current phase in `.claude/PLAN.md`
- Use `git add -A` — stage files explicitly
- Implement Phase 2 (video/GVHMR) until Phase 1 passes end-to-end

## Working memory & lessons (session continuity)

Everything below stays **local to this repo** under `./.claude/` — never write to global `~/.claude/` for this project.

- **Save progress:** Persist detailed progress of the current session to `working_memory/{current_session_name}.md` in the repo root (use the session id and/or a short topic slug, e.g. `2026-06-25_skills-agents-memory-setup.md`). Update it **as work progresses**, not only at the end.
- **KPT template:** A reusable retrospective template lives at `working_memory/_templates/kpt-retrospective-template.md`. Copy it for each new retro.
- **(1) Resume:** At the start of a session related to saved memory, look in `working_memory/` for a matching file, load it, and continue from where the previous session left off.
- **(2) Extract lessons:** When the user or the agent surfaces a lesson worth keeping, extract it from the session and persist it as one or more of:
  - a new/updated rule in this `.claude/CLAUDE.md`, and/or
  - a new/updated skill under `.claude/skills/<name>/SKILL.md`, and/or
  - an agent-memory file under `.claude/memory/` (one fact per file, indexed in `.claude/memory/MEMORY.md`).
- **Memory locations:** Repo-local lessons → `.claude/memory/` (absolute: `/home/wswg3/project/comfyui-mocap/.claude/memory/`). Working-session progress → `working_memory/`. Do not install skills/agents/memory globally for this repo.
