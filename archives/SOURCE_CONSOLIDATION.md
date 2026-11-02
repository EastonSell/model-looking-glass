# Consolidated historical auditors

Ten standalone auditor copies were removed from `scripts/` and `tools/`; their byte-identical result-local copies below remain in the current tree. Git blob equality was checked for every pair. No inbound path or module-stem references were found in the active package, scripts, tools, tests, experiments, docs, examples or studies. The original paths remain in immutable commit `3cbdec78134855532811a58a64e8dd5d4d457185`.

This consolidates duplicate historical entry points, not scientific outcomes. No result-local auditor, unique implementation, test, or source-hash contract was rewritten. Run the retained copy only with its original study prerequisites; preservation is not a new claim that every historical auditor was executed in this cleanup.

| Removed duplicate | Retained exact copy | Git blob |
|---|---|---|
| `tools/audit_bert_cue_class_full_saved.py` | [results/bert-cue-class-cpu-v1-20261029/audit_full.py](../results/bert-cue-class-cpu-v1-20261029/audit_full.py) | `8971997fe1aa6d9a423bfe31ba92603c1a12e1ef` |
| `tools/audit_bert_cue_heldout_frames_full_saved.py` | [results/bert-cue-heldout-frames-cpu-v1-20261029/audit_full.py](../results/bert-cue-heldout-frames-cpu-v1-20261029/audit_full.py) | `2268330e26218d01f7dc6732c088dcb275d878de` |
| `tools/audit_bert_negation_full_saved.py` | [results/bert-negation-cue-cpu-v1-20261029/audit_full.py](../results/bert-negation-cue-cpu-v1-20261029/audit_full.py) | `80f0d1f614080d53642da5e137a500d9735bae3b` |
| `scripts/audit_bert_cue_context_signed_full_saved.py` | [results/bert-cue-context-signed-cpu-v1-20261029/audit_bert_cue_context_signed_full_saved.py](../results/bert-cue-context-signed-cpu-v1-20261029/audit_bert_cue_context_signed_full_saved.py) | `9163980a304ccb037e556742a4aa42d7c1c110e1` |
| `scripts/audit_bert_cue_dose_continuation_full_saved.py` | [results/bert-cue-dose-continuation-cpu-v1-20261029/audit_cont_full.py](../results/bert-cue-dose-continuation-cpu-v1-20261029/audit_cont_full.py) | `d79eff865348ff9611d7b69425a823a8f3cc3565` |
| `scripts/audit_bert_cue_dose_response_full_saved.py` | [results/bert-cue-dose-response-partial-cpu-v1-20261029/audit_full.py](../results/bert-cue-dose-response-partial-cpu-v1-20261029/audit_full.py) | `c640e95350e6da75eaa639cd1c771c0df965a4d7` |
| `scripts/audit_bert_cue_factorial_full_saved.py` | [results/bert-cue-factorial-cpu-v1-20261029/audit_full.py](../results/bert-cue-factorial-cpu-v1-20261029/audit_full.py) | `81aed1c38b0f069bc8c1b3049361b8fe90097508` |
| `scripts/flan_t5_dose_response_independent_saved_audit.py` | [results/flan-t5-dose-response-cpu-v1-20261029/audit/full-saved-replay.py](../results/flan-t5-dose-response-cpu-v1-20261029/audit/full-saved-replay.py) | `37a03689c91b709c9ad85e113f845a992c429ab7` |
| `scripts/flan_t5_equal_dose_independent_saved_audit.py` | [results/flan-t5-equal-dose-cpu-v1-20261029/audit/full-saved-replay.py](../results/flan-t5-equal-dose-cpu-v1-20261029/audit/full-saved-replay.py) | `ae2f5b1925577e6577f16ed122ae496417046688` |
| `scripts/flan_t5_equal_length_independent_saved_audit.py` | [results/flan-t5-equal-length-response-cpu-v1-20261029/audit/full-saved-replay.py](../results/flan-t5-equal-length-response-cpu-v1-20261029/audit/full-saved-replay.py) | `92779253f0df5833082dbd2fc723bf80003493ad` |

Removed 228,814 duplicate source bytes. Broader study families remain intact because their manifests, dynamic source-hash checks and replay tools form coupled reproducibility contracts. Deleting a leaf from those families would require retiring the entire workflow and validating that boundary first.


## Retired superseded v4 preparation workflow

The original agreement-head v4 proposal was explicitly superseded **before model forwarding**, because three subject lemmas already appeared in exposed development cases. Its three preparation/comparison scripts are retired together. The [original protocol](../docs/DISTILGPT2_AGREEMENT_HEADS_FRESH_CONTEXT_V4.md), [original case records](../results/distilgpt2-agreement-head-v4-cases/README.md), and corrected v4.1 workflow remain. Full tracked-reference review found no incoming active runtime/test dependency on the retired scripts; only their historical case README named them, and those mentions now link to the immutable source. No scientific result is reclassified.

| Retired immutable source | Original Git blob | Retained successor |
|---|---|---|
| [scripts/prepare_agreement_head_v4_fresh_cases.py](https://github.com/EastonSell/model-looking-glass/blob/3cbdec78134855532811a58a64e8dd5d4d457185/scripts/prepare_agreement_head_v4_fresh_cases.py) | `26bc2fc1becea16afb3e716d993d768705b3067c` | [scripts/prepare_agreement_head_v4_1_fresh_cases.py](../scripts/prepare_agreement_head_v4_1_fresh_cases.py) |
| [scripts/check_agreement_head_v4_public_overlap.py](https://github.com/EastonSell/model-looking-glass/blob/3cbdec78134855532811a58a64e8dd5d4d457185/scripts/check_agreement_head_v4_public_overlap.py) | `ec566fdce5f0068ed621bcbb24e5aa63bb1726ae` | [scripts/check_agreement_head_v4_1_public_overlap.py](../scripts/check_agreement_head_v4_1_public_overlap.py) |
| [scripts/natural_public_cases_crosscheck_v2.py](https://github.com/EastonSell/model-looking-glass/blob/3cbdec78134855532811a58a64e8dd5d4d457185/scripts/natural_public_cases_crosscheck_v2.py) | `02d42e3e4266c9a9d270da35f9bff7a91d40b5f1` | [scripts/natural_public_cases_crosscheck_v3.py](../scripts/natural_public_cases_crosscheck_v3.py) |

These three files total 18,337 bytes. Historical source can be inspected with `git show 3cbdec78134855532811a58a64e8dd5d4d457185:scripts/prepare_agreement_head_v4_fresh_cases.py` after obtaining the pinned commit. The result-payload restore tool is not a source-code restore command; use the immutable links or the [historical checkout instructions](README.md#replaying-a-historical-study).
