# Same-site ViT CPU study runner

`tools/run_vision_coco_same_site.py` executes the eight frozen cases in
`experiments/vision_coco_same_site_v3.json` under the previously published
same-site protocol. The annotation-only selection was committed before any
model forward; its public source is recorded in the manifest. The earlier
failed spatial selector remains in the repository as a separate record.

The controller verifies the pinned manifest, source files, selection ancestry,
and the reaped direct child. The child reconstructs the annotation selection,
checks JPEGs and processed pixels against their hashes, loads the pinned model
offline, and records returned full logits and same-forward intervention
witnesses. It preserves every completed case, including baseline-ineligible and
negative outcomes. The controller checks the complete forward schedule, byte
layouts, hashes, and pair records before marking its receipt complete. A
separate model-free replay must independently check the saved scientific
calculations and controls before results are characterized as verified.

The trusted checkout, interpreter, dependencies, and cooperative CPU custody
are assumptions; hashes and direct-child reaping do not establish execution,
process-tree custody, public chronology, official COCO origin, or pretraining
independence. The caller must independently observe the controller exit and
retain failures. A checkpoint preflight forbids forward calls, and the
synthetic rehearsal uses fabricated tensors and an untrained fake model.

Usage requires an existing absolute `--custody` directory, absolute `--assets`,
`--output`, and `--pillow` paths, and the full committed `--revision`. Use a
fresh output directory for each attempt. `--preflight-only` and
`--synthetic-rehearsal` are separate no-study-forward checks. A real run has
neither flag. The caller must not retry exposed cases after an incomplete run.
