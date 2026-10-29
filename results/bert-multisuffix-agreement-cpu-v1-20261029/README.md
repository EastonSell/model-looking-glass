# DistilBERT agreement across five mask contexts

This fixed-case study asks whether three continuations change the probability
of `is` or `are` at a masked verb, compared with a period alone, and whether
subject-position donor interventions move the agreement margin beyond matched
controls. It is an adaptive follow-up to the earlier [`ready.` study](../bert-agreement-suffix-cpu-v1-20261029/README.md),
not a sample of all contexts or a test of pretraining independence. The
[protocol](../../docs/BERT_MULTISUFFIX_AGREEMENT_V1.md) and
[160 input variants](../../experiments/bert_multisuffix_agreement_cases_v1.json)
were fixed before these model outputs.

All three continuations met the prespecified descriptive mass criterion on
these 32 paired bases, but their effects were very different. Against a period
alone, mean `P(is)+P(are)` increased by **82.27 percentage points** for
`present.`, **0.286 points** for `outside.`, and **7.971 points** for
`awake.`. The `outside.` difference was negative in six cases. Higher pair
mass did not imply better subject agreement: the strict binary sign was
correct in **28/32** `present.` cases versus **31/32** for period alone.

## Clean predictions

The same 32 bases use two surface frames, four subject lemmas, both subject
numbers, and both attractor numbers. Each has the same prefix through `[MASK]`
and one of five endings. The entries are complete-vocabulary softmax mass, not
the probability normalized only over the two verbs. Ties are absent here.

| Ending | Mean `P(is)+P(are)` | Binary agreement correct | `is` or `are` is full-vocabulary top |
| --- | ---: | ---: | ---: |
| Terminal `[SEP]` | 1.278% | 25/32 | 0/32 |
| Period only | 0.417% | 31/32 | 0/32 |
| `present.` | 82.683% | 28/32 | 32/32 |
| `outside.` | 0.702% | 31/32 | 0/32 |
| `awake.` | 8.387% | 30/32 | 8/32 |

| Continuation versus period | Mean paired mass change | Cases with positive change | Frame-by-lemma groups with positive mean |
| --- | ---: | ---: | ---: |
| `present.` | +82.2666 percentage points | 32/32 | 8/8 |
| `outside.` | +0.2855 percentage points | 26/32 | 8/8 |
| `awake.` | +7.9706 percentage points | 32/32 | 8/8 |

The fixed rule required a positive mean, at least 24/32 positive case changes,
and at least 6/8 positive group means; it was met for each suffix. These are
descriptive thresholds, not independent significance tests. Relative to period
alone, `present.` changed three correct binary signs to incorrect and repaired
none; `awake.` changed one correct sign to incorrect, and `outside.` changed
none. As a secondary comparison, punctuation alone changed six terminal errors
to correct signs and changed no correct signs to errors, while reducing the
two-verb mass in 25/32 cases.

## Subject-state interventions

On the pinned pretrained DistilBERT checkpoint, we captured the output of
block 3 at the subject token. Same- and opposite-number donors used the same
other subject lemma within the same frame, attractor, and ending. Each eligible
base used the same capped input displacement `q` across all five endings, with
a separately seeded random direction shared across its contexts. Self and
clean-after controls were also run. There were 160 clean forwards, 800 ordinary
intervention/control forwards, and 128 fixed-terminal-direction forwards;
all 32 bases were eligible and all 1,088 actions were retained.

The signed donorward margin change from the opposite-number donor exceeded
**both** active controls in 31/32 terminal and period cases and in 32/32 cases
for each of the three continuations. The predeclared donor criterion was met
in each context. Mean contrasts are differences in `is` minus `are` logit
margin, signed toward the opposite-number donor:

| Ending | Opposite minus same-number | Opposite minus random | Positive against both |
| --- | ---: | ---: | ---: |
| Terminal | +0.194 | +0.196 | 31/32 |
| Period | +0.231 | +0.224 | 31/32 |
| `present.` | +0.238 | +0.228 | 32/32 |
| `outside.` | +0.199 | +0.191 | 32/32 |
| `awake.` | +0.273 | +0.252 | 32/32 |

Those margin shifts rarely reversed the binary preference. The opposite donor
flipped one terminal case from correct to incorrect and flipped no binary
signs in any of the other four endings. The same-number control flipped a
different terminal case from incorrect to correct; random flipped none. The
full-vocabulary unique top changed under the opposite donor in 0, 4, 0, 3, and
3 cases in the table order above; the same-number control changed it in 0, 5,
0, 4, and 4 cases, and random in 1, 1, 0, 2, and 2. Median full-vocabulary
total-variation distances for opposite donors were 0.0212, 0.0332, 0.0199,
0.0423, and 0.0238 respectively. All arm-level TV/KL values, signs, and tops
are retained in the [per-case replay](saved-replay.json.gz).

All recorded `q` values were 2.0, while the ten natural same/opposite donor
distances per base ranged from 8.75 to 14.83. Equal input L2 does not imply
equal behavioral potency. Reapplying the terminal-derived intended opposite
direction in the other contexts is a descriptive additional check. Its mean
context delta minus the shared-random delta was +0.0262, +0.0369, **−0.0064**,
and +0.0423 for period, `present.`, `outside.`, and `awake.`. Different base
states, rounded landings, and nonlinear responses prevent interpreting this
as an isolated downstream gain.

## Evidence, review, and limits

The one-thread CPU attempt used the pinned DistilBERT revision and a 192 MiB
**disk-evidence** cap, not a GPU memory estimate. The original child and outer
observer recorded zero exits and no pending/failure marker. The separately
retained outer tool-exit record is an operator transcription, not independent
OS or process-tree attestation. Independent saved-byte reviews checked the
1,088-action schedule, all donor geometry and landings, selected logits,
bitwise clean/self/clean-after controls, and the original closed inventories.
Selected witnesses cannot establish behavior at unsaved positions. The
complete, independently verified local evidence archive is 122,677,157 bytes,
SHA-256 `4d1f462222b3420094094bf96979a257ea98e1acd41a5652c8de4460adc0bf0a`;
it is retained by the operator rather than in this repository.

The compact [public raw subset](selected-evidence-v1.zip), SHA-256
`47b8106c4d71ce91b618f27f03a441af70f42911dcea75dddded70dd78194bb9`, contains
the original starts, events, plan, metadata, receipt, and 1,088 × 4 original
float32 logits in the order `is`, `are`, `was`, `were`. Its manifest identifies
the source hashes and row-major layout; an independent
[byte check](review-multisuffix-selected-archive-v1.json) matched all 4,352
values to the full saved logits. It omits full-vocabulary logits and subject
witnesses, so it cannot by itself reproduce the probability, TV/KL, or
geometry checks. The full local archive includes those bytes.

The [saved replay](saved-replay.json.gz) has uncompressed SHA-256
`a503c1d165a8d12244965737842537a292f01ade93d182baf283bcd1bf4faa45` and
contains every case, ending, arm, group, and retained contrast. The executed
reader can obscure an extremely small paired probability difference by
subtracting separately rounded values. A separate
[corrected paired audit](bert-multisuffix-actual-stable-primary-v1.json)
resolved all 96 actual comparisons away from zero; its bounds assume
binary64 round-to-nearest and the documented `exp`/`expm1` error limits.
An independently formulated [70-digit Decimal recheck](bert-multisuffix-actual-decimal-primary-v1.json)
agreed on all 96 comparisons and fixed decisions. These are numerical checks,
not execution attestation. The corrected audit and its companion helper are
included here; the Decimal script records its original machine-specific paths.

The operator's source-ordered launch records identify local commit
`42519c13a5b33ce3601b89ba65d67b07ebd2f78d` as frozen before model execution;
no separately trusted pre-forward timestamp for that exact commit was retained.
Its eight relevant blobs were independently matched to the subsequently public source.
The [incremental source bundle and instructions](../../audits/bert_multisuffix_agreement_v1/EXECUTED_SOURCE.md)
require the public prerequisite commit `2a492669b2ebb16d2612366fbefdf80c4265faca`;
the bundle cannot restore an empty repository by itself. Its publication was
after the forward, so it is not evidence of earlier public availability.
[Public-input checks](../../audits/bert_multisuffix_agreement_v1/README.md)
cover named sources and fixed pending bundles, not unknown exposure or model
pretraining. These repeated frame/lemma combinations are dependent, the three
suffixes were chosen after prior results, and context changes the masked task
and available right context. The results establish sensitivity for these
specific interventions and inputs; they do not identify a complete reasoning
route or show that a general behavior was repaired.

To repeat the corrected primary calculation, extract the full local archive,
decompress `saved-replay.json.gz` to a fresh path, and run the retained audit
with the companion `bert_multisuffix_stable_pair.py` in the same directory:

```sh
python results/bert-multisuffix-agreement-cpu-v1-20261029/audit_bert_multisuffix_stable_primary.py \
  --attempt /absolute/extracted/bert-multisuffix-cpu-v1-20261029-01 \
  --cases experiments/bert_multisuffix_agreement_cases_v1.json \
  --source-report /absolute/decompressed-saved-replay.json \
  --output /absolute/new-primary-audit.json
```

The output must be a new file outside the closed attempt and controller. The
audit needs the full local logits, validates their recorded hash, and does not
execute a model. Restoring the exact recorded source for the older saved reader
requires the incremental bundle prerequisite and the recorded Git executable
identity; an arbitrary replacement executable is not an identical runtime.
