---
name: lesson-verify-source-exists
description: Verify a source repo exists (git ls-remote) before planning an install
metadata:
  type: feedback
---

Verify that a third-party repo actually exists before building an install plan around it. `git ls-remote <url> HEAD` is a cheap existence check.

**Why:** In the 2026-06-25 session the user's `gstack` link was a copy-paste duplicate of the graphify URL, and `https://github.com/safishamsi/gstack` returns "Repository not found." Confirming this avoided fabricating a non-existent install.

**How to apply:** When given a URL/owner/name, run `git ls-remote` (or `gh repo view`) first. If it doesn't resolve, surface it to the user and ask for the correct reference rather than guessing. See [[lesson-surface-missing-not-substitute]].
