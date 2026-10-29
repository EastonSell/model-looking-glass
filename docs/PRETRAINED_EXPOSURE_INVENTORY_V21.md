# DistilGPT2 known exposure inventory v21

V21 extends the reviewed v20 known-input inventory with the 48 fixed public
explicit-year-direction prompts. These are **planned** inputs: this snapshot
does not assert that they have been forwarded. Their four groups of possible
continuation token IDs are retained as case metadata; they are not counted as
additional forwarded prompts or as input prefixes. The case file records v17
as its assembly-time snapshot; this successor checks the later v20 records.

The predecessor's 499 records and 3,620 input prefixes are preserved. Adding
the 48 distinct 13-token prompts yields 547 records and 3,888 input prefixes.
The reservation checks both complete-row prefix directions against prior
inputs, recorded target continuations, and v20's candidate continuations. It
also checks the listed year-direction candidate extensions against those known
rows. These checks do not prove that pretraining or outside uses are known.

- Predecessor v20 payload SHA-256:
  `3df696303a06b2ec6ddf4113df1969897dbae89566375b7d3a9cecb85b3d59cd`.
- This canonical payload SHA-256:
  `a391c5f3a8ca33ad32e9378417b424338b12ea55257979977d2afd5a0c7f90ae`.
- Protocol: `docs/DISTILGPT2_EXPLICIT_YEAR_DIRECTION_V1.md`, public head
  `e5f71eb66e233a54f1171d1e6f489c0fca587156`, SHA-256
  `68f028e401fd4c43d1cac89772da10c92cf5e5de7af30c4043c18c21678b902e`.
- Cases: `experiments/distilgpt2_explicit_year_direction_v1.json`, merged
  `3425253a9479ee4d09241baaf0da6c499667f2c8`, SHA-256
  `419d74c23333407f9a9dc0b6f7d7b70764ab5d1078e2b0a4c84d37ba194df8a9`.
  The ordered compact JSON list of input rows has SHA-256
  `2454f4176696b53a8a7a6b395d5b594fd9ed3c081a5909b69af11e61c330cb66`.
- The public-only adapted inputs and digest-only natural comparison receipt
  accompany this snapshot under
  `results/distilgpt2-explicit-year-direction-cases-v1/`. Their SHA-256s are
  `0ebd8c2de1126a21b31ee06a33a8c19cb29fe2938a09b45bf5fb1acdb48b1478`
  and `ecdb68744d67c596365021ed7920922e69209a8abb9e02cb2ca45835762e9695`.
  The receipt reports no bidirectional complete-row-prefix relation with the
  16 already sealed natural rows. It is a trusted-operator observation, not
  independent proof of the private inputs or of model execution.

Each new record ID is `planned-` followed by the exact case ID, and
`EXPLICIT_YEAR_DIRECTION_ROWS` exports the prompts in case-file order. Known
history is incomplete, so the snapshot does not authorize a study seal or
claim that these public examples are blind.
