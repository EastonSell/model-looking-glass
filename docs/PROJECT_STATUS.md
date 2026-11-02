# Project scope

## Reusable starting point

The central workflow is **observe → specify an intervention → compare against controls → report a bounded conclusion**. Keep direct capture, explicit coordinates, bounded trace storage, exclusive model ownership, intervention plans, paired metrics, and experiment records together. The eager GPT-2/DistilBERT/T5/ViT adapters are useful entry points within their documented model-specific limits.

These components require their own runtime/dependency checks. Inclusion here is a maintenance direction, not a claim that this cleanup validated every platform, dtype, model family, or public API. Never simplify away failure cleanup, model leases, alias checks, or thread ownership merely to reduce line count.

## Experimental, not a supported service

The broker, IPC, GPU budget, resident-worker, device-lease and quarantine components are experiments in safe execution coordination. Fixed serial smoke checks do not establish a concurrent GPU service, admission fairness under arbitrary load, physical memory enforcement, or general throughput improvement. Keep them out of the default onboarding path until an integrated service has reproducible acceptance coverage. The [consolidated infrastructure reference](EXPERIMENTAL_INFRASTRUCTURE.md) keeps their ownership, accounting and failure limits in one place.

The many per-study owners, launchers, receipts, source snapshots and replay scripts are **historical reproduction code**, not separate endorsed APIs. Their hashes and cross-references are part of the evidence. They have not been indiscriminately deleted or moved: preserving a runnable historical source boundary is safer than breaking provenance to make the package look smaller. Further source extraction requires dependency-aware replay validation.

## Research evidence

This is the short findings map, not a new experiment or a substitute for each study’s frozen decision rules. The [methods synthesis](CAUSAL_INTERVENTION_METHODS_NOTE_20261029.md) explains why matched dose, donor identity, eligibility and collateral effects must be evaluated separately.

The original observation experiment and its [denormal counterexample](GRADIENT_DENORMAL_V1.md) establish what the recording mechanism does and does not preserve. Dense traces can locate the first *observed* numerical difference; this is not a unique causal explanation.

The preserved [pre-cleanup research index](https://github.com/EastonSell/model-looking-glass/blob/3cbdec78134855532811a58a64e8dd5d4d457185/README.md#exploratory-intervention-study) contains the full study-by-study account. Important distinctions survive cleanup:

| Study family | Bounded conclusion / limitation | Representative evidence |
|---|---|---|
| DistilGPT2 agreement heads | Some fixed public comparisons show score effects; controls and nonpositive cells limit mechanism/generalization claims. | [Fixed donor](../results/distilgpt2-agreement-head-v3-fixed-donor/README.md), [fresh-context successor](../results/distilgpt2-agreement-head-v4-1-fresh/README.md) |
| Copying-head studies | Initial candidates weakened under competing-head and matched-dose controls; later primary analyses were sometimes ineligible. | [Factorial](../results/distilgpt2-copy-head-factorial-v1-20261028/README.md), [output-dose follow-up](../results/distilgpt2-copy-head-output-dose-v1-20261029/README.md) |
| BERT context/cue studies | Preserve held-out, factorial and specificity failures alongside apparent positive effects. | [Cue/frame factorial](../results/bert-cue-factorial-cpu-v1-20261029/README.md) |
| FLAN-T5 interventions | Teacher-forced score shifts do not establish generated translation or unique computation; baseline failures and inconclusive held-out results remain evidence. | [Held-out response](../results/flan-t5-heldout-response-cpu-v1-20261029/README.md) |
| ViT interventions | Negative/inconclusive composites and failed eligibility screens do not prove absence of visual information. | [CLS/spatial comparison](../results/vision-coco-cls-v5-20261029/README.md), [failed screen](../results/vision-coco-small-dose-v2-selection-failure-20261029/README.md) |
| Targeted parameter edits | Target effects coexist with locality failures or ineligible baselines; broad capability preservation is not established. | [Original edit](../results/targeted-edit-cpu-v1-20261028/README.md), [screened matched edit](../results/targeted-edit-screened-matched-v1-20261029/README.md) |
| GPU execution experiments | A later successful run does not erase a failed run or prove earlier cleanup occurred. | [Grouped development pilot](../results/grouped-gpu-resource-v3-20261029/README.md), [one-shot record](GPU_ONESHOT_RESULTS_V2.md) |

Archival selection is based on payload format, **never on whether the outcome is positive**. Uncompressed result reports, JSON/CSV decisions, failure records and source-bound protocols stay in their original paths. Historical explanatory manuals are separately indexed in the [documentation archive](../archives/README.md#historical-documentation); that consolidation does not alter study records. Compressed decision/report files are also archived: they are not all directly readable in the current tree. Their bytes and classifications remain preserved at the pinned source commit. Archived bytes can be restored from a pinned ancestor commit using the exact manifest. Existing historical auditors may require that restoration; their source hashes must not be silently updated to disguise absent inputs.

## Validation boundary

Latest complete default CPU rerun: **8,145 passed, 285 skipped, zero failures**. The skips comprise 57 explicit experimental opcode exclusions and 228 other prerequisites; they are not passing coverage. The seven required screened-evaluation matrices are restored byte-for-byte. See the [final validation record](../results/maintenance-cpu-20261030/final-summary.json) and [cleanup report](MAINTENANCE_2026-10-29.md#successful-complete-default-cpu-rerun--2026-10-30). The earlier failing run remains preserved as history.

Use focused, named tests for the component being changed. A complete historical test run requires PyTorch, pytest, the pinned optional Transformers dependency, and selected original artifacts; some tests also have explicit environment prerequisites. No dependency skip is evidence that the skipped behavior works. [Current cleanup report](MAINTENANCE_2026-10-29.md) records actual executed and blocked checks.

Historical exceptions injected from opcode trace callbacks are [explicitly experimental](EXPERIMENTAL_OPCODE_TESTS.md), because Python documents undefined interpreter behavior for that mechanism. They are not evidence of general safety at arbitrary bytecode boundaries. Supported ordinary callback-failure, cancellation, cleanup and custody controls remain in the default test suite; exact experimental cases require explicit isolated opt-in. Native experimental failures remain failures, not successes or ordinary prerequisite skips.
