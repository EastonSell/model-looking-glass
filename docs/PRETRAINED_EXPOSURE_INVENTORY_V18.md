# DistilGPT2 development exposure inventory v18

This successor adds eight operator-reported inputs from a CPU grouped-chain
test of the pinned DistilGPT2 checkpoint. The worker reports forwarding them
without preceding exposure coordination or a protected-split comparison, and
stopped further pretrained forwards after reporting the event retrospectively.
Each addition is classified `observed-reported`, not `observed-saved`.

The v17 predecessor's 407 input rows and texts are preserved. Its 48
year-transfer records are updated from `planned` to `observed-saved` using
the completed public evidence, and the eight reported uses are appended as
separate records. The first input, `[1, 2, 3, 4]`, already
appears as `synthetic-gpt2-control-1` with a `synthetic-only` label; the new
entry records the additional reported pretrained use without rewriting that
earlier record. The other seven full inputs are new to the inventory. There
are 415 records and known input prefixes grow from 3,247 to 3,275.

The canonical module is `looking_glass.pretrained_exposures_v18`. Its v17
predecessor payload SHA-256 is
`e5734470ccd3b4e2655c91608caf8ffa158fa8650ff0bdd7f5aeaa1028046f3a`;
the v18 canonical payload SHA-256 is
`3fa3eda9508ca314505aa658b1c02d04dbc98681bfc776337cd9ca1c9c384f5a`.
The immutable `GROUPED_REPORTED_ROWS` export contains the eight rows in the
reported order. A blank `reference_commit` on these additions means the
retrospective is retained with this inventory snapshot; it does not identify
the unrecorded executed source.

The retained operator retrospective is
`results/distilgpt2-grouped-unreserved-cpu-v1/retrospective.json`, SHA-256
`e087d03ae0cc2c3d0c0c71a241b4dd583fedea4f609364c96ad1ec7c8412e5e4`.
It reports a completed 12-call CPU test, the eight inputs, and a passing test
outcome. The test used an evolving uncommitted tree: exact executed source
bytes, execution time, per-forward logs and logits were not retained. The
retrospective is an operator report, not independently replayable numerical
evidence or an execution attestation.

The public year-transfer evidence was fixed at
`01e632d7b535cf006e00fb1a659c6e136d338c69` under
`results/distilgpt2-year-transfer-v1-20261029/`. Its manifest matches the 48
previously reserved inputs; the saved result and journal report 336 returned
actions, seven per case, and the direct-process receipt reports exit 0 without
interruption or a failure file. The manifest, result, journal and receipt
SHA-256 values are, respectively,
`b412d7a29f573e43e62b810b2bd63a9d92f1f2c698a2beb52bb8347e980cbb1d`,
`fad27de70b301e7171c6fc55c8b600e79f9d7a59c6d4f8feb1ee66362d5e926f`,
`9e94fc4c30f563c1679e0fafd577133127ac413b72aa318064c2dd8987292c2b` and
`1bd457a772a4985bf223624ea5248e8343771bde97495b240d9717748d997a42`.
These are saved-evidence consistency checks, not independent attestation of
the execution.

The natural-split custodian reports a private comparison of all eight
reported rows against its 16 sealed rows, with no equality or full-row-prefix
collision in either direction. The order-preserving comparison input file is
`results/distilgpt2-grouped-unreserved-cpu-v1/comparison-inputs.json`,
SHA-256 `b436a0487cb630c5f96c7f83a9098b7654e7b1c06917662b13e0988d6cd26aa9`.
Its rows match the retrospective. The digest-only comparison receipt is
`results/distilgpt2-grouped-unreserved-cpu-v1/comparison-receipt.json`,
SHA-256 `ac8205584df6e045025a3f0edc72b5303cffb3c385bbfbb445044f25e68f46dd`;
it binds the comparison-input file hash, not the retrospective hash. This is
an overlap check for the reported inputs; it does not independently
establish what the unlogged execution processed. It does not certify unknown
prior exposure, pretraining or other private splits, and known-prefix checks
alone do not authorize a study seal.
