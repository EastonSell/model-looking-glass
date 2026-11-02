# Public agreement-direction evaluation inputs

This file set fixes 32 target/donor prompts and eight no-explicit-subject
context fragments for the prospective
[agreement-direction protocol](../../docs/AGREEMENT_DIRECTION_STUDY_V1.md).
The 32 target prompts are four base frames crossed with subject and attractor
number, with one regular and one irregular subject lemma per frame. The eight
singular–plural subject pairs each recur in two attractor contexts; they are
not independent samples, and morphology remains entangled with lemma identity.
The context fragments are imperfect token-bias diagnostics. The two `Beside`
fragments tokenize to five IDs, while the other six tokenize to three; their
comparisons can reflect tokenization and positional differences as well as bias.

The protocol merged at `a39121c2f3cf2602281252a8856d48efadcb2a98` and the
fixed discovery-derived directions merged at
`15c0ef7f8141322aeb1fa31dc02cab5a68fc16b8` before these cases were assembled.
The previously available development-vector hashes are disclosed in the
[direction report](../distilgpt2-agreement-direction-training-v1/README.md).
The [case file](../../experiments/distilgpt2_agreement_direction_cases_v1.json)
has SHA-256 `a68aa63ff06aa97c5c68d941fe27160c4c4698cdf070325b3cefa2b29ec58b13`.
The compact JSON serialization of its 40 ordered `token_ids` lists has SHA-256
`4b4faf815ae0d3d0943efec5fbc318fd251dc6fea80b6e32c770b470f01dae1d`.

The preparer used the pinned tokenizer only, checked every input and each
of four appended answer-token boundaries, and compared equality and both
complete-row-prefix directions against the independently verified v18 public
inventory (`3fa3eda9508ca314505aa658b1c02d04dbc98681bfc776337cd9ca1c9c384f5a`,
415 records). It separately compared the eight operator-reported rows already
included in v18, 44 additional public screened proposals and 48 public
year-direction proposals. The operator retrospective records a reported
execution, not retained logits or exact executed source. No new model was
loaded or forwarded to construct these cases.

After case assembly, the natural custodian reported that these exact 40 rows
have no equality or complete-row-prefix relation in either direction to the
sealed natural split. The [comparison receipt](natural-comparison.json) binds
the [adapted public input rows](natural-comparison-inputs.json); their order
was independently matched to the case file. The case JSON's `pending` field
is its historical status at assembly. This is a scoped trusted-custodian
report, not an authenticated proof of private disjointness or evidence about
unknown exposures. Injection custody separately reported no held-out rows and
no direct conflict for the fixed attention sites. Refresh the latest reviewed
exposure inventory and both custodians' applicable status before a model
forward; the planned reservation and owner reviews are separate requirements.
The previous natural held-out owner stopped incomplete without a scientific
result after a possible-forward gate; its split is not reused here.

The model-free preparer source SHA-256 is
`68e440bfeb6a31abc83102d502197c9fa870081c45da11539f6a87ac2251246b`.
It checks canonical inventory bytes and bounds file reads. Five focused tests
passed for pool eligibility, prefix relations, fixed crossing, continuation
boundaries and a corrupted inventory. Independent tokenizer/row review is
recorded in the associated pull request. Cases are public, exposed and
nonblind; a positive or negative result will apply to these fixed examples and
interventions only.
