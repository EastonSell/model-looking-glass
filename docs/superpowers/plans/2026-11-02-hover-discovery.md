# Hover Discovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make sparse captured-site bands discoverable without selecting or reading data.
**Architecture:** Reuse bounded Three raycasting and the existing on-demand render scheduler; track transient hover separately from persistent selection.
**Tech Stack:** Existing vanilla JS, Three.js and Node/pytest contracts.
**Spec:** `docs/superpowers/specs/2026-11-02-hover-discovery-design.md`

## Global Constraints
Existing keys/labels only. No API/model/device/schema/dependency changes. No permanent extra labels or idle rendering. No direct-main or browser/release retries; publish and merge via authorized PR route with checks.

## Review Focus
- Pointer bursts coalesce and do not move camera or trigger selection.
- Hover on another band does not replace or recolor the selected site.
- Touch, secondary pointers, dragging and drag-back do not preview/select accidentally.
- Pending hover never crosses layer replacement, hide, context loss, or disposal.
- Native List and existing click behavior remain unchanged.

### Task 1: Transient captured-site preview
**Files:** `looking_glass/app_static/app_scene.js`, `tests/app_scene_contract.cjs`, `docs/LOCAL_APP.md`.
**Interfaces:** Existing scene API/callbacks unchanged; new private pointermove/pointerleave handlers reuse real hit testing.
- [x] Add failing real-math hover/coalescing/lifecycle regressions.
- [x] Implement bounded transient preview and deliberate-click drag guard.
- [x] Run scene/DOM/app tests and record inherited historical collection gaps.
- [x] Independently review and fix lifecycle recovery with regression coverage.
- [ ] Publish a scoped PR and wait for checks.
- [ ] Merge using expected-head protection and verify main and post-merge checks.
