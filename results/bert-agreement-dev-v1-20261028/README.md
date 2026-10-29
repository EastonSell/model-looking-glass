# DistilBERT agreement development attempts

These are public-fixture development runs, not held-out evaluation. The
candidate and controls are described in `docs/BERT_AGREEMENT_V1.md`; the
evaluation path remained disabled. The model and tokenizer were pinned to
`distilbert/distilbert-base-uncased` revision
`12040accade4e8a0f71eabdb258fecc2e7e948be`. The journal retains selected
float32 logits and intervention vectors, while full mask logits are represented
only by hashes. Those hashes do not permit independent reconstruction of the
full logits or prove that a recorded execution occurred.

The immutable protocol and v1 input inventory retain the earlier pilot
snapshot: four original prompts were exposed and the remaining development
rows were reserved. This description was already stale in source revision
`186cbb8` when attempt 02 began: attempt 01 had exposed all eight
development-family-1 A/B rows, and attempt 02 repeated them. Those rows are
development data, not held-out;
the separate public evaluation rows had no model forwards in these attempts.

## Attempt 01: incomplete controller outcome

The original attempt used source commit `cc66516375c181eb6d40f04eed0ed96800e0dc84`
and run ID `411eb82f68859d3f15a6701dd2d44344`. Both saved controller terminals
say **incomplete**. The owner exited with recorded code 0, and its journal
contains 140 started and 140 completed forward records and 48 passing control
records. The controller process exit code itself was not independently
retained. The owner recorded an empty OS machine string, which definitely
violates the controller's nonempty runtime contract; the original failure
record did not identify a stage, so the exact rejecting condition is not
independently isolated. The later source revision records an unavailable
machine string as `unknown` and adds bounded failure stages.

An independent read-only audit checked the immutable source hashes, case
identities, input/mask records, all event pairings, patch landings, control
records and each saved scalar contrast. In the owner candidate, the four-case
mean signed donor-number effects were 5.76077 at the chosen block-2 subject
site, -0.22613 at the block-2 distractor, and 8.51654 at the fixed block-0
subject comparison. The other fixed random comparisons were -0.10013 at the
block-3 distractor and -0.01488 at the block-4 distractor. The block-0
comparison exceeded the chosen site. For the
plural-subject/plural-distractor case, both opposite-number donor patches left
the `is` minus `are` margin negative, so they did not flip that prediction.
These are exploratory observations from an incomplete attempt, not a held-out
or population result; lexical replacements and the one-family sample limit
interpretation. A future evaluation would also change the template and nouns
together, preventing separation of those two changes.

`attempt-01/inventory.json` lists the copied files and their hashes. The
compressed journal expands byte-for-byte to the original JSONL with SHA-256
`ff14574bf1b35058b3cff51e4752180ec589982cf3a547baba318a9591af8b67`.
The preserved files do not retroactively change the incomplete status.

## Attempt 02: complete development retry

The corrected source revision `186cbb8510caa28610348ee000b5504dc2cf6179`
reran the same eight already-exposed development input rows under run ID
`dd100dc14f21292d5017a05828bb9288`. The terminal reports **complete**, the
owner was reaped with recorded exit code 0, and the local wrapper and execution
tool separately reported controller exit code 0. The wrapper exit file is a
local observation, not independent process attestation; only its exit-capture
code excerpt is preserved, with the local-path launch preamble omitted. The
terminal inventories the owner evidence and records the unavailable OS machine
string as `unknown`. The recorded local action order was to publish this
commit on `research/bert-mlm-study-v1`, fetch that exact remote branch and
merge it into the local checkout, and then start the retry. This is a recorded
local sequence, not an independently witnessed public timestamp.

The retry produced the same 1,187,199 journal bytes as attempt 01, so both
inventories refer to the one compressed copy in `attempt-01/`. It reproduced
all four case effects, all five site means, and all 48 required control
outcomes. The numerical observations and limitations above apply to this
complete development retry, but not to a held-out evaluation. No evaluation
case was forwarded. The four unpatched target cases all preferred the
grammatically matching `is`/`are` token in this family; this is a four-case
descriptive count, not an accuracy estimate for a broader population.
