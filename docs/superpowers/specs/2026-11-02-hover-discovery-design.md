# Discover captured sites before selecting

The scene intentionally suppresses unselected module labels, but this makes its recorded bands hard to identify before clicking. Reveal the existing module label and a subtle neutral highlight on mouse/pen hover. Keep the warm selected-site highlight distinct and leave the inspector and camera untouched until a deliberate click.

Use only existing captured-site keys/labels and real Three raycasts. No tensor/API request, inference, model/device change, storage, invented neurons, or connections. Preserve touch selection, drag-to-orbit, the native keyboard List alternative, reduced motion, and context-loss fallback.

Coalesce pointer movement into the existing requested animation frame, at most one hover raycast per rendered frame against the bounded 72-site page. No idle loop. Clear hover/queued coordinates on leaving, dragging, cancelling, replacing layers, resizing, hiding the scene/document, or disposal. A drag that returns to its starting point must not become a click.

Alternative: permanently show all labels again. Rejected because it restores dense-scene clutter. Alternative: fetch details on hover. Rejected because metadata is already available and hover should not read model data.

Verification uses real Three math and existing DOM contracts, with a fake renderer explicitly not browser/GPU proof. Independent review and CI precede the authorized PR merge. No denied direct-main or browser/release route retries.
