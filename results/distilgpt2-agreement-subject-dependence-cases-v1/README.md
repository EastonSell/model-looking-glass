# Prospective agreement subject-dependence cases

This fixed public, nonblind bundle contains 64 target sentences and 16
fragments, arranged in four crossed frames and four attractor conditions as
specified in the [published protocol](../../docs/AGREEMENT_SUBJECT_DEPENDENCE_V1.md).
It was constructed with the pinned tokenizer after the protocol and
development output-slope reference were published. No model forward has
been performed for these cases. Publication, scientific review and a planned
reservation do not themselves establish a result or authorize a launch.

The selected regular subjects are horse/horses and bird/birds; the selected
irregular subjects are mouse/mice and wolf/wolves. Goose/geese was skipped
because the plural is two tokens. The four attractor pairs are bridge,
road, ship and hill with their plurals. The case file records the complete
ordered token rows, all pool eligibility decisions, subject/final positions
and labels. There are 48 eight-token and 16 six-token target rows, and 12
six-token and four four-token fragments. No-attractor `nearby` and mass-noun
`water` conditions are not equivalent contexts: token positions, lexical
semantics and morphology remain limitations.

The preparer checked all 80 input rows and their eight candidate continuations
against canonical v25, the separately published 48 full-top inputs, the 128
original natural-sampler public-support rows and the 256 public successor
inputs in both complete-row-prefix directions. It found no collision. The
construction-time v25 revision is an externally observed input, not a fact
authenticated by the preparer itself. A subsequent separate check against
merged v26 compared all 720 input/continuation sequences with 2,183 known
sequences and found no collision; its [receipt](v26-comparison.json) records
the scope and hashes. Independent tokenizer-only review reconstructed the
selection, all 80 texts and all 640 answer boundaries.

The natural custodian independently compared the fixed 80 input rows against
the full 128-row reconstructed original support and 256-row public successor
pool; its [digest-only receipt](natural-support-comparison.json) reports no
complete-row-prefix relation. Coverage of the old sealed split is conditional
on the recorded source-bound sampler and tokenizer preparation; this is not
an inspection of private membership. At that observation no successor split
or nonce existed. The injection custodian reported no protected rows and a
scoped no-direct-target-conflict assessment for the fixed L3-final attention
site. Refresh the applicable custody and latest reviewed inventory before any
forward, and publish a separately reviewed planned reservation first.

The custodian's public-only comparison is reproducible with explicit paths:

```text
python -B tools/compare_agreement_subject_public_support_v1.py \
  --case experiments/distilgpt2_agreement_subject_dependence_cases_v1.json \
  --original-support experiments/natural_agreement_original_public_support_v1.json \
  --successor-inputs experiments/natural_agreement_successor_inputs_v1.json \
  --output NEW_RECEIPT_PATH
```

Key SHA-256s:

* [Case file](../../experiments/distilgpt2_agreement_subject_dependence_cases_v1.json):
  `5f3a2c850454e74da2bdc3819d492357f829994c126e1ac95d90ca3813b0643e`.
* Ordered compact input rows: `d31073b25d816258ff1bcbfcb7237eac5346b160a5ed2de65fea220bdb806ff1`.
* Preparer `tools/prepare_agreement_subject_cases_v1.py`:
  `3ef4b8081888ab085019f2ceae8c02acbdfce94903ae3a201f11970672cc1ba4`.
* Natural public-support receipt:
  `61d1f8555ad901924364fb89c731b0870ff057d6caeebcf4b5d293e819be1753`.
* v26 public comparison receipt:
  `b9feffe754d24d89a85656f48dd15c1d75191589a41751a89d4913540dfacfb1`.

The preparer requires `tokenizers==0.22.2` and checks the pinned tokenizer
JSON SHA `8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6`.
To reconstruct the original inputs, independently verify the public protocol
and reference revisions, then run the preparer with absolute or relative paths
for the checked protocol, slope vector, tokenizer JSON, original
`binding-v2.json`, prior direction case JSON, canonical v25 inventory bytes,
original public-support JSON, successor-input JSON, full-top JSON and a **new**
output path. The exact argument names are visible via `--help`; supply v25
version `distilgpt2-development-exposures-v25`, revision
`2c1863ec25af072e54fcc5f42506f486f46553f8` and canonical SHA
`32cc4bcbe4b482ded122715c5f96c6c5c765a76a1ef3532360cdf9890827f144`.
The canonical v25 bytes can be generated from the reviewed v25 module's
`canonical_inventory_bytes()` function. The preparer checks bytes/version
against supplied values; publication and currentness of the revision are
separate observations. No private input is required to reproduce these public
checks.
