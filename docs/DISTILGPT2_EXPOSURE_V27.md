# DistilGPT2 exposure inventory v27

V27 preserves the reviewed v26 inventory and appends the 80 fixed public
[subject-dependence inputs](../experiments/distilgpt2_agreement_subject_dependence_cases_v1.json)
as `planned-*` records. Each record carries its case role and eight candidate
continuation IDs as metadata. This reservation does not assert a forward on
these new inputs and adds no private rows or nonce. The canonical inventory
SHA-256 is
`ef42683a664dda450e82c69afedda34a8a34f2daf9f50f2db54f6c9d846f07e5`;
it contains 723 records and 4,623 distinct input prefixes. Exact and both-way
full-row-prefix relations are rejected for the inputs and candidate extensions
against prior known inputs/extensions and between distinct new cases. The
unknown pretraining and historical exposure of the model is not certified.

The 48 full-top records retain their v26 planned labels unchanged. Their owner
subsequently reported and saved the 36 target inputs in the merged
[full-top result](../results/targeted-edit-full-top-cpu-v1-20261029/screen/README.md); a separate
status-only reconciliation is pending. The 12 locality rows remain planned.
These historical labels in v27 do not describe the latest observation status.

The [study protocol](AGREEMENT_SUBJECT_DEPENDENCE_V1.md) and
[case note](../results/distilgpt2-agreement-subject-dependence-cases-v1/README.md)
define the matched controls, public-support check and remaining launch-time
custodian/current-inventory requirements. A reservation does not establish a
study result or by itself authorize a forward.
