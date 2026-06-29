---
name: lesson-graphify-engine-installed
description: graphify memory engine is already pip-installed at ~/.local/bin/graphify; the skill wraps it
metadata:
  type: project
---

The graphify memory/knowledge-graph engine is already installed on this machine at `/home/wswg3/.local/bin/graphify` (the safishamsi/graphify Python package). The `graphify` skill (now at `.claude/skills/graphify/`) is the thin wrapper that invokes this engine via the `/graphify` trigger.

**Why:** Avoids a redundant reinstall and clarifies that "install graphify for memory" was already satisfied at the engine layer — only the local skill copy was needed.

**How to apply:** To use graphify memory, invoke the `/graphify` skill; it shells out to the installed engine. No `pip install` needed. Verify with `which graphify` if behavior looks broken.
