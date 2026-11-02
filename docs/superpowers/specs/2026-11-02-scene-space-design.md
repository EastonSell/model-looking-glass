# Reclaim the captured-layer scene

## Intent and constraints
Keep the existing sparse, near-black wireframe explorer, while making its panel-collapse controls actually release space to the captured layers. This is a small UI correction under the user's autonomous UI-improvement authorization, not a new model capability. No inference, device, dependency, data storage, or API changes. Preserve exact-site navigation, accessibility, reduced-motion behavior, and the list fallback. Do not fabricate neurons or connections.

## Approach
Retaining the fixed camera reservations leaves collapsed panels visually empty but unusable. Duplicating more CSS measurements in the renderer risks future divergence. Instead, let collapsed grid tracks shrink to their visible toggle rails and have the UI provide measured left/right dock insets to the existing Three.js framing math. Docks stacked below the scene on narrow screens reserve no horizontal space. Keep the full-bleed canvas.

An explicit observation-table click should reopen the selection inspector, matching scene/list selection. Background first-event loading must not force it open.

## Verification and limits
Use actual Three.js projection/raycast math with the existing renderer test double; test each collapsed state, responsive/no-overlap, selection preservation, and no model/API work for layout changes. Extend DOM contracts and structural CSS checks. Real browser/WebGL capture remains unavailable under the existing access restrictions; do not claim visual or GPU validation.
