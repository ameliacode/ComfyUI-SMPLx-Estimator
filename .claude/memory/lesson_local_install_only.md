---
name: lesson-local-install-only
description: For comfyui-mocap, install skills/agents/memory locally in ./.claude, never global
metadata:
  type: feedback
---

For the comfyui-mocap repo, all skills, subagents, and lesson/memory files must be installed **locally** under `./.claude/` (i.e. `/home/wswg3/project/comfyui-mocap/.claude/`), NOT in the global `~/.claude/`.

**Why:** The user wants this repo's tooling self-contained so it travels with the code and does not pollute or depend on global config. Stated explicitly and repeatedly ("make sure to install in current repo ./.claude locally. not in global").

**How to apply:** Copy/clone into `.claude/skills/<name>/`, `.claude/agents/<name>.md`, and `.claude/memory/`. Do not write to `~/.claude/` for this project. See [[lesson-inventory-before-fetch]] for sourcing.
