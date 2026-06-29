---
name: lesson-inventory-before-fetch
description: Check ~/.claude and on-machine marketplaces before downloading skills/agents
metadata:
  type: feedback
---

Before fetching skills/agents from the internet, inventory what already exists on the machine: `~/.claude/skills/`, `~/.claude/agents/`, and `~/.claude/plugins/marketplaces/`.

**Why:** In the 2026-06-25 setup session, most requested skills (scientific-writing, the Notion set, code-review, graphify) already existed globally or in a marketplace. Copying known-good local copies was faster and more reliable than re-downloading.

**How to apply:** `ls ~/.claude/skills ~/.claude/agents` and `find ~/.claude/plugins/marketplaces` first; only `git clone` the authoritative source when the item is missing or you need the canonical upstream version. Then place copies locally per [[lesson-local-install-only]].
