---
type: kpt-retrospective
session: "2026-06-25_skills-agents-memory-setup"
date: "2026-06-25"
duration: "~1 working session (single-day, multi-turn)"
scope: "Current session (local skills/agents/memory setup) + reconstructed view of prior comfyui-mocap sessions inferred from agent-memory artifacts (project_*.md) and the sibling Click-Pose/working_memory logs — NOT from full prior transcripts."
---

# KPT Retrospective — 2026-06-25_skills-agents-memory-setup

> KPT = **Keep** (what worked, continue) · **Problem** (what hurt, what blocked) · **Try** (concrete next experiments).
> Keep entries short, factual, and outcome-oriented. Each Problem should map to at least one Try.

## 0. Context (1–3 lines)
- Goal of the session(s): Install a curated set of skills + subagents + a memory system **locally** into `./.claude` for the comfyui-mocap repo (scientific writing, Notion, code review, PM/researcher/critic agents, graphify memory). Then stand up working-memory + lesson-extraction conventions.
- Outcome: **Done** for installs; memory/working-memory conventions established this session. `gstack` skipped (repo does not exist under the given owner).
- Artifacts produced: 9 new skills + 3 new agents under `./.claude`; `working_memory/` + KPT template; local `.claude/memory/` lesson store; new CLAUDE.md memory rule.

## ✅ Keep — what went well and should continue
- [x] **Inventory before fetching** — checked `~/.claude/` and on-machine `plugins/marketplaces/` first; most requested skills already existed and were copied locally instead of re-downloaded. Saved network round-trips and guaranteed known-good copies.
- [x] **Verify source existence before acting** — `git ls-remote` confirmed the three real repos and proved `safishamsi/gstack` returns "Repository not found"; the gstack link was a copy-paste duplicate of graphify. Avoided fabricating an install.
- [x] **Surface a gap instead of silently substituting** — VoltAgent had no `critic` agent; raised it via one focused question rather than guessing, then installed all three chosen critics.
- [x] **Validate every install** — confirmed each skill has `SKILL.md`, each agent has `name:` frontmatter, and no stray `.git` dirs were copied in.
- [x] **Stable, documented architecture decisions (prior sessions)** — `COCO-17 only`, `COCO→H36M direct (no swap)`, and `encoder-once / decoder-only HITL` were captured as durable memory + CLAUDE.md rules, preventing re-litigation.

## ⚠️ Problem — what went wrong, was slow, or got in the way
- [x] **Global vs. local ambiguity** — many skills already lived in `~/.claude` (global); the user had to repeat "install locally, not global." Root cause: no project-level convention previously documenting that this repo wants self-contained `.claude/`.
- [x] **Memory location split-brain** — global CLAUDE.md points agent memory at `/home/wswg3/.claude/projects/-home-wswg3/memory/`, but the user wants lessons inside the repo at `.claude/memory/`. Two competing "memory" locations is confusing.
- [x] **"Previous sessions" not directly observable** — no transcript of prior sessions is available to this session; the retro had to be reconstructed from memory artifacts, so prior-session Keep/Problem items are inferred, not measured.
- [x] **Underspecified third-party links** — `gstack`/graphify shared one URL; ambiguity cost a verification step.

## 🚀 Try — concrete experiments for next time
> Make each one a testable action, ideally linked to a Problem above.
- [x] **Add an explicit "install locally in ./.claude" rule to project CLAUDE.md** — addresses: global/local ambiguity · success looks like: future sessions never ask "global or local?".
- [x] **Standardize the repo lesson store at `./.claude/memory/` with its own MEMORY.md index** — addresses: memory split-brain · success: one obvious place per repo for lessons.
- [ ] **Write `working_memory/{session}.md` progressively, not just at the end** — addresses: previous sessions not observable · success: the next session can `git log`/read working_memory and resume without guessing.
- [ ] **When given a third-party link, run `git ls-remote` before committing to an install plan** — addresses: underspecified links · success: zero time spent on non-existent repos.

## 📌 Lessons to persist (graduate to rules / skills / memory)
> Anything here should be copied into agent memory or CLAUDE.md / a skill.
- **Lesson:** This repo wants skills/agents/memory installed **locally in `./.claude`**, never global. → **Where it lives:** `.claude/CLAUDE.md` rule + `.claude/memory/lesson_local_install_only.md`
- **Lesson:** Inventory `~/.claude` + on-machine marketplaces before fetching; copy known-good local. → **Where it lives:** `.claude/memory/lesson_inventory_before_fetch.md`
- **Lesson:** Verify a source repo exists (`git ls-remote`) before planning an install; `safishamsi/gstack` does not exist. → **Where it lives:** `.claude/memory/lesson_verify_source_exists.md`
- **Lesson:** When a requested item is missing from its suggested source, surface it and offer alternatives. → **Where it lives:** `.claude/memory/lesson_surface_missing_not_substitute.md`
- **Lesson:** graphify memory engine is already pip-installed at `~/.local/bin/graphify`; the `graphify` skill wraps it. → **Where it lives:** `.claude/memory/lesson_graphify_engine_installed.md`

## 🔗 Follow-ups / open threads
- [ ] Confirm whether the local `.claude/memory/` should fully replace the global auto-memory for this repo, or mirror it.
- [ ] Commit the new `.claude/` skills+agents+memory and `working_memory/` (currently untracked) when the user is ready.
- [ ] If `gstack` is genuinely wanted, get the correct repo URL/owner.
- [ ] Resume actual project work: Phase 1 pipeline (ClickPose → MotionAGFormer → BVH) per `.claude/PLAN.md`.
