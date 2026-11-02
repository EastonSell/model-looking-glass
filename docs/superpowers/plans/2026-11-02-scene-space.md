# Scene Space Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reclaim scene space when either inspector collapses, and reveal intentionally selected table observations.

**Architecture:** CSS grid tracks follow workspace collapse attributes. app.js measures visible dock overlap and supplies `getViewInsets()` to app_scene.js; the renderer fits real captured layers into that gap.

**Tech Stack:** Existing vanilla JavaScript/CSS, Three.js, Node contract tests, Python structural tests.

**Spec:** `docs/superpowers/specs/2026-11-02-scene-space-design.md`

## Global Constraints
No inference, device, dependency, data storage, or API changes. No fabricated neurons or connections. No browser-access or external-staging retries.

## Review Focus
- Narrow screens: below-scene docks must not reserve scene width.
- Either collapsed side: remaining expanded dock must remain unobscured.
- Both collapsed: content grows and remains framed, with selected highlight intact.
- Explicit table selection: reopen the inspector, without additional data requests.
- Scene unavailable: existing exact list remains usable.

### Task 1: Reclaim scene space and reveal intentional selection
**Files:** `looking_glass/app_static/{app.css,app.js,app_scene.js}`, `tests/{app_scene_contract.cjs,app_ui_contract.cjs,test_app_layout.py}`, `docs/LOCAL_APP.md`.
**Interfaces:** `getViewInsets(): {left: number, right: number}` supplies CSS-pixel insets for the scene canvas; absent/invalid insets reserve no width.
- [x] Add failing projection, DOM, and structural regression coverage.
- [x] Run `node tests/app_scene_contract.cjs`, `node tests/app_ui_contract.cjs`, and `python -m pytest tests/test_app_layout.py -q`; confirm expected failures.
- [x] Implement measured framing, shrinking tracks, and explicit table-selection reveal; update the existing user guide.
- [x] Run the focused tests, app tests and bare `pytest`; report inherited collection gaps separately.
- [x] Request independent review, fix material findings, and commit only this scope.
- [ ] Publish through the permitted GitHub connector with an expected-head lease, verify exact content and CI; hand release notes to the parent rather than bypass a denied release route.
