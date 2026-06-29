---
name: lesson-surface-missing-not-substitute
description: When a requested item is absent from its suggested source, surface it and offer alternatives
metadata:
  type: feedback
---

When a user names a source and a requested item is not in it, surface the gap and offer concrete alternatives instead of silently substituting something else.

**Why:** The user suggested VoltAgent/awesome-claude-code-subagents for a "critic" agent, but that repo has no critic. Asking one focused question (architecture vs. scientific/writing vs. code-quality critic) let the user pick all three deliberately, rather than getting a guessed substitute.

**How to apply:** Confirm presence in the named source; if absent, present 2–4 real, sourced options via a single question, note overlaps with existing tooling, then install the chosen ones. Pairs with [[lesson-verify-source-exists]].
