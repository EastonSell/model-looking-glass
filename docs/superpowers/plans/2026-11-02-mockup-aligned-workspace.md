# Mockup-aligned Workspace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Apply the approved concept's scene-first layout while preserving actual capture semantics.
**Architecture:** Existing DOM controls move into their visual context; CSS grid owns scene/canvas/toolbar geometry; Three renders selected-only module labels and a warm selected-site token.
**Tech Stack:** Vanilla HTML/CSS/JS, bundled Three.js, Node contracts and pytest.
**Spec:** `docs/superpowers/specs/2026-11-02-mockup-aligned-workspace-design.md`

## Global Constraints
No fabricated neurons/connections, chat features, inference/device/API/schema changes, dependencies, browser retries, or direct main writes. Publish only to the existing feature branch; the separately authorized PR merge requires passing checks and an expected-head lease, without a PR approval.

## Review Focus
- Long inspector content still scrolls instead of covering the toolbar.
- Mobile Run remains beside the actual prompt/image input.
- Empty capture and WebGL failure keep the toolbar state and native list coherent.
- Selection color and label visibility reset correctly across selection/replacement.
- Canvas and toolbar stay in the scene row when warnings are visible.

### Task 1: Scene-led chrome and selection hierarchy
**Files:** `looking_glass/app_static/{index.html,app.css,app.js,app_scene.js}`, UI/scene/layout contracts, local-app guide.
**Interfaces:** Existing control IDs and API handlers retained. New `scene-toolbar` visibility follows retained capture state; `--selected` is presentation only.
- [x] Add/run failing label-color, control-placement, and toolbar-state regression tests.
- [x] Implement compact docks, grid-aligned floating toolbar, contextual Run and quiet labels.
- [x] Run app suite and contracts; record historical sparse-collection limitation separately.
- [x] Obtain independent review and fix the mobile toolbar/list overlap with red/green coverage.
- [ ] Publish to the existing feature branch with a head lease, and verify CI and source bytes.
- [ ] Merge through the authorized PR route after checks; verify main contains the tested source tree.
