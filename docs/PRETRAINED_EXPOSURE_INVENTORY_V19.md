# DistilGPT2 development exposure inventory v19

This successor reserves the 44 public screened-edit prompts before a
pretrained forward on those cases. It preserves all 415 v18 records, including
the eight operator-reported grouped inputs, and adds 44 `planned` records.
The known input-prefix union grows from 3,275 to 3,522. The 76 continuation
encodings in the public case file are comparison metadata and are not added as
forwarded prompts.

The canonical module is `looking_glass.pretrained_exposures_v19`. Its v18
predecessor payload SHA-256 is
`3fa3eda9508ca314505aa658b1c02d04dbc98681bfc776337cd9ca1c9c384f5a`;
the v19 canonical payload SHA-256 is
`e2eb185b32e74a4442ec68c43e53bd851573f60b7d7423b0d83195950f21b175`.
The resulting 459 records include the 44 new planned prompts. The module
exports `SCREENED_CASES` and `SCREENED_ROWS` in the published case order, and
each new record ID is `planned-` followed by its exact public case ID.

The protocol was fixed at `3d9a5a3c5cb6a6a905e90ef029fe627a147de4e6`,
SHA-256 `6a2bb800923e4d694819bb6fc8164e6b58698d0d12bd33d88397387b81783cde`.
The public case file merged at `e27798622cd56eff1db0866e112bbde1f3926971`,
SHA-256 `fc3f00a0b4f7884a3c00011186420d16322009edf75b6b09e4296953dad198e3`.
It records v17 as its construction snapshot; this reservation separately
checks all 44 prompts against the newer v18 input rows and recorded
target-appended rows in both full-row prefix directions, as well as against
one another. Shared short prefixes remain in the exclusion set.

The public natural-split comparison receipt is
`experiments/targeted_edit_screened_natural_receipt_v1.json` at the same case
merge, SHA-256
`bbd78ce2cb63db2a67cfc85cb030ac8cefea2f24c88c4105bff30d7392ef2c2b`.
It reports no bidirectional full-row-prefix collision against 16 sealed rows.
Its comparison-input SHA-256
`48574a625548ee16fc6d58ccb6249ab9455b8fc464411700700c39c199d0256b`
binds a canonical `{"cases":[{"input_ids":...}]}` projection of the 44
case-file token rows in order; the focused test reconstructs those exact bytes.
The receipt is a trusted-operator overlap observation, not an execution
attestation or proof about other private or unrecorded inputs.

No model forward is asserted for the 44 new prompts. Known-prefix checks cover
recorded and reported history only; they do not establish all prior exposure
or pretraining and do not authorize a study seal.
