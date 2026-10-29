# Public DistilGPT2 input reservations: v31

This additive public inventory reserves 34 inputs before their proposed
forwards: 32 copying-head prompts in the fixed, tokenizer-only case bundle,
and two separate synthetic service inputs with all-one attention masks. It
asserts no forward for these new records. The 915 v30 predecessor records
are preserved byte-for-byte as data, including their historical status labels;
some may have been forwarded since those labels were assigned. A later
evidence-bound update can reconcile those statuses.

The copying cases were selected using the then-reviewed v29 public input
inventory; v30 changed statuses while preserving those input rows and prefix
sets. Their published case bundle is
[`distilgpt2_copy_head_prompt_dose_cases_v1.json`](../experiments/distilgpt2_copy_head_prompt_dose_cases_v1.json)
at commit `115d9bb42aadc481482de302d2df11c9c2d30789`, SHA-256
`b9a6fdca9998d26c1880c56160438a5b368621cf13bc1c4b212971512f1baf69`.
Their appended target tokens are labels and comparison metadata; the 11-token
rows alone are the planned forward inputs.

The two service rows are `[50256] + range(1001,1016)` and
`[50256] + range(2001,2032)`. Their independently supplied public-scope
comparison receipt has SHA-256
`9ff6f36cb71960398a1113e5731e224d83e5483748cdd307ae719a6dc9dbbe39`.
The fixed synthetic inputs and receipt are public at commit
`63886558b85212ec59d73f04c78e5e722f1e36f6` in
`experiments/short_cpu_service_reference_inputs_v1.json` and
`docs/SHORT_SERVICE_SYNTHETIC_PUBLIC_SUPPORT_V1.json`; the input manifest
SHA-256 is `47077a930963da6785d2208a21c782b045191068a835579cf295c1e990fb7f2c`.
The receipt is an observation about public inputs, not model execution.

The v31 builder rejects a full-row-prefix relation in either direction between
the 34 proposed inputs, or between any proposed input and a v30 input or its
recorded candidate continuation. The 34 inputs were also checked against the
separate public 128-row natural support and 256-row natural successor sets;
no relations were found. These are checks of the named public scopes only. No
other study's protected inputs were read, and no claim is made about unknown
exposures, pretraining, or related prompts. Selection was not blind to previous
published study results.

The v30 canonical SHA-256 is
`921025a7f9c1246c5f10e97f2b987ca4971462241b2d05ebbc525d4aa0736c79`.
The v31 canonical SHA-256 is
`2d710a40a49a84b7d664cd851fccd2527ea318c5a8c911111fe7275b7567e565`;
it contains 949 records and 5,789 distinct recorded input prefixes.

The [independent case](../experiments/copy_head_prompt_cases_independent_audit_v1.json)
and [inventory](../experiments/pretrained_exposure_v31_independent_audit.json)
reports are the reviewer's captured JSON with CRLF normalized to LF for this
repository. The original capture SHA-256 values were respectively
`9f3fea32b0a033d789e5089aca847d3e65467c297ce595aa613217cd03d7f0a1`
and `1bac0b234d0737bc34d1efc25f402b24aff854dc0f8f4d01f8c83af631f2eafa`.
The accompanying independent scripts preserve their original workspace-specific
paths; their results check saved public bytes, not a forward or protected data.
