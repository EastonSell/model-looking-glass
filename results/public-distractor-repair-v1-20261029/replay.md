# Replaying the saved evidence

The independent reader uses Python's standard library and does not load a model. From the repository root, with this result directory and the fixed public case file present, run:

```text
python -B results/public-distractor-repair-v1-20261029/audit_public_distractor_repair_v1.py results/public-distractor-repair-v1-20261029 experiments/public_distractor_repair_cases_v1.json --phase-prefix run- --pre-run results/public-distractor-repair-v1-20261029/pre-run/pre-run.json --freeze results/public-distractor-repair-v1-20261029/discovery-freeze.json --result results/public-distractor-repair-v1-20261029/result.json --output fresh-independent-report.json
```

Use a fresh output path: the auditor creates it exclusively. It checks all 448 actions, the two full-vocabulary positions, full landing and untouched bits, the separately stored binary64 displacements, controls, frozen predictions, per-case and aggregate values, receipt file hashes and logs. The [retained independent report](public-distractor-heldout-independent-v2.json) has SHA-256 `2ec0175b3886ad5239de39a42971d17f1d2fea7594b978f3b87a9a091e60689f`. The auditor's SHA-256 is `071c666ffe81251a9d3b0051df29cb636fab4ca8e159b28f6f3931a0e06b932f`.

For discovery alone, the adjacent `audit_public_distractor_discovery_v1.py` (SHA-256 `f6c661234b51be200fc092cad5591847d97fda16ac3103e22474f010b2de3f48`) can be run with the discovery directory, fixed case file and freeze as positional arguments, and `--pre-run`, `--receipt` and a fresh `--output`. The retained discovery report has SHA-256 `b65fc1a1c1d2ed5073dc951cad0b058412a3f28ad6557d060acfbcfdfcb5c86d`. Both scripts use nearest-even checks and require the saved bytes and compatible binary64 arithmetic.

The public [pre-run card](pre-run/pre-run.json) pins the reviewed source and historical v38 reservation; the [discovery launch observation](pre-run/launch-observation.json) and [held-out start observation](heldout-start-observation.json) record point-in-time public currentness and freeze-readback context. The discovery freeze was independently read from public PR #671 before the direct child started. A saved-byte audit and cooperative receipt do not authenticate execution, descendant processes, complete exposure history or unknown pretraining exposure.
