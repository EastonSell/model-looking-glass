# DistilBERT multisuffix execution v1

This note binds the CPU fallback for the prospective
[protocol](BERT_MULTISUFFIX_AGREEMENT_V1.md). The dedicated owner loads one
pinned checkpoint, holds it for the complete schedule and streams retained
float32 selected-position logits and subject witnesses to disk. It does not
spawn another model for each arm. The direct CLI is CPU-only: CUDA requires a
separately reviewed physical-lease owner with device-specific admission,
calibration, synchronization and cleanup. Prior GPT-2 profiles do not provide
DistilBERT resource bounds.

The owner verifies the source commit and local source bytes before importing
model dependencies, checks the tokenizer and five checkpoint asset hashes,
uses local-only loading with empty missing/unexpected/mismatched/error
loading diagnostics, and rechecks source and assets before completion. It
records the one-thread, deterministic, subnormal-preserving CPU policy, seed,
model cleanup and dependency versions. The selected mask-position logits and
subject before/after vectors are the saved evidence; source-based reports of
full-model output finiteness or unchanged unsaved positions are not independently
attested by those retained slices.

The observer records fresh host and output-volume availability, starts one
direct CPU child, and preserves its original controller directory with a
pending sentinel until the child has exited and a completion receipt has been
written. Failure, nonfinite evidence, uncertain child creation/exit or an
unmatched start is incomplete; do not silently retry an uncertain attempt.
The observer does not establish process-tree or operating-system execution
attestation. Capture any outer observer's exit separately. Its admission
measurements are point-in-time readings, not reservations or a claim that C:
receives no incidental writes.

The predeclared maximum is 1,088 actions with 139,516,416 raw float32 tensor
bytes. A full-vocabulary fabricated owner-main run (no pretrained weights)
completed all actions and retained **140,228,406 bytes** including its journals,
plan, metadata and completion file; this is below the 192 MiB attempt cap.
The cap applies to disk evidence, including a bounded failure message; traceback
formatting can still allocate memory and is not bounded by that disk limit.
After all 160 clean captures, finite zero-dose or finite rounded-landing
ineligibility retains those clean records and skips the affected base’s 29
post-clean arms while other bases continue. A failure during clean capture can
leave fewer than 160 records and remains incomplete. Saved replay must preserve all clean outcomes, distinguish
those ineligible cases and mark the full-set donor criterion not evaluable.

Before any pretrained execution, freeze all source and reader bytes, refresh
the named public/pending input comparison, check the attempt and controller
output volume and host headroom, and have independent model-free reviews of
the owner, intervention arithmetic and saved replay. The user-visible study
remains adaptive to prior results, limited to the fixed five contexts and
repeated frames/lemmas, and is not an assertion of pretraining independence.
