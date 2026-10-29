# Full-top study: fixed public case scope

The [case file](../experiments/targeted_edit_full_top_cases_v1.json) freezes 36
target prompts and 12 locality prompts in the order required by the
[protocol](TARGETED_EDIT_FULL_TOP_CPU_V1.md) and its
[scope clarification](TARGETED_EDIT_FULL_TOP_SCOPE_V1.md). Its SHA-256 is
`4057696ddef9e3b0d09db02f46e6454561a61ac1a07f8b8a13b92718944f4261`.
The proposals were constructed with the pinned tokenizer only; the operator
reports no model forward for case assembly. Their factual/alternative pairs,
roles, city-answer meanings, locality relations, accepted variants and all 88
single-token append boundaries received independent review. The case file
records exact prompt IDs and accepted continuation IDs. Its 48 prompt rows,
not the continuation metadata, are the planned model inputs.

For the octagon case the accepted set includes the single-token numeral ` 8`
as well as ` eight` and ` Eight`, as explicitly permitted by the pre-forward
scope clarification. It is a small prespecified semantic-variant set, not a
claim that all valid answers are enumerated. The discovery, calibration and
evaluation prompts are distinct strings, but some share the Spain–Madrid
in-context example; they are not independent facts or independent samples.
Evaluation phrases will be selected using baseline top predictions and must
not be described as blind, representative or held-out facts.

The public-only [natural support report](../experiments/targeted_edit_full_top_public_frame_v1.json)
reconstructed all 128 possible original natural inputs from the published
sampler, checked the same pinned tokenizer and model-input preprocessing, and
found no bidirectional complete-row-prefix relation with these 48 rows. It also
found none against the 256 published natural successor inputs. The original
split custodian reports that the existing split is unchanged and that its
recorded source-bound preparation sampled only from this frame. Thus the
public-superset inference is conditional on that preparation and custody
account; it is not a private split query, proof of authenticated execution,
or a guarantee about unknown exposures. The report's source, binding,
tokenizer and [public support](../experiments/targeted_edit_full_top_public_support_v1.json)
hashes identify the reconstruction. The public support contains all possible
inputs, not the sealed selection.

Independent public-inventory comparison also found no input-prefix relation
against all 547 v24 rows and the 595-row v25 candidate. The latter was not yet
merged at the time of that check. These are historical checks, not an assertion
that they remain current: refresh the latest reviewed inventory and relevant
custodians, and reserve all 48 rows before their first model forward. Preserve
all failed screens and do not replace cases after observing model results.
