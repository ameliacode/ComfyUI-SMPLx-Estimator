# Memory Index (repo-local — comfyui-mocap)

> Repo-local lesson store. Lives in the repo so lessons travel with the code.
> One fact per file. For durable cross-project facts the global store at
> `/home/wswg3/.claude/projects/-home-wswg3-project-comfyui-mocap/memory/` still applies.

- [Install skills/agents/memory locally in ./.claude](lesson_local_install_only.md) — never global for this repo
- [Inventory before fetching](lesson_inventory_before_fetch.md) — check ~/.claude + marketplaces first, copy known-good local copies
- [Verify a source repo exists before installing](lesson_verify_source_exists.md) — git ls-remote; safishamsi/gstack does not exist
- [Surface missing items, don't silently substitute](lesson_surface_missing_not_substitute.md) — VoltAgent had no critic → asked
- [graphify engine already pip-installed](lesson_graphify_engine_installed.md) — ~/.local/bin/graphify; the skill wraps it
- [ComfyUI inference_mode breaks autograd](lesson_comfyui_inference_mode_autograd.md) — nodes that optimize must opt out of torch.inference_mode + build models outside it
