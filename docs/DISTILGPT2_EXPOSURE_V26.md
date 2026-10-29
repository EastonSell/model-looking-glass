# DistilGPT2 exposure inventory v26

V26 preserves the complete v25 inventory and appends the 48 fixed public
[full-top study prompts](../experiments/targeted_edit_full_top_cases_v1.json)
as `planned-*` inputs. Their continuation token IDs are metadata; this update
does not assert a forward on any new prompt. It adds no private rows or nonce.
The canonical inventory SHA-256 is
`8fd80cb0b24225100628489dcd0fe2b7efd11e87dfcbb624cf039e54a8a3a62d`,
with 643 records and 4,475 distinct input prefixes. Exact and both-direction
full-row-prefix collisions are rejected against the previous inventory and
among the 48 inputs. The model's unknown pretraining and exposure history is
not certified.

The 48 v25 year/label inputs remain in the inventory and retain their previous
planned labels. Their owner subsequently reported a partially completed run
that stopped under its planning rule; an independently audited, public failure
package and evidence-bound status update are separate future work. This
operator report does not establish an individual row's saved outcome here.

The [full-top protocol](TARGETED_EDIT_FULL_TOP_CPU_V1.md),
[scope clarification](TARGETED_EDIT_FULL_TOP_SCOPE_V1.md) and
[case note](TARGETED_EDIT_FULL_TOP_CASES.md) describe the baseline screen,
the public natural-support inference and remaining custody requirements. A
reservation by itself is not permission or evidence that the study ran.
