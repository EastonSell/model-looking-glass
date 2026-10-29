# Post-hoc full-vocabulary analysis of cue transfers

This is a model-free descriptive analysis of the already completed
[DistilBERT negation-cue study](../results/bert-negation-cue-cpu-v1-20261029/README.md).
It was designed after that study's outputs were known. It is not a new
preregistered prediction, blind test, or population inference.

The original fixed public sentence family has **one** template, two development
contexts and six evaluation contexts. For each evaluation context the two
negative cues `not` and `never` give 12 context/cue comparisons; they are not
12 independent contexts. This analysis retains every development and evaluation
context, both transfer directions, all four recorded sites, all eight directed
transfers at each site and the full-logit self/repeat/final-cue controls.

For each negative cue `q` and context, pair the following at the **same target**
and the original common requested dose:

* `q -> just` versus `also -> just`.
* `just -> q` versus `also -> q`.

The `also -> just` action is the same saved control for both negative cues in
the same context; counts of 12 reuse six control actions. In the reverse
comparison `just` and `also` are both affirmative donors into a negative
recipient. This probes whether the effect is unique to `just`; it is not a
polarity-preserving null and does not by itself refute a polarity-class effect.
The remaining `just -> also` transfer is retained as an unpaired descriptive
action. The contextual object and mask sites are active comparators, not
assumed nulls; the final cue site is a structural full-logit control.

Let `p` be softmax of the clean target mask logits and `q` be softmax of the
patched target logits. Report:

* Full-vocabulary total variation `0.5 * sum(abs(q-p))`.
* The fraction of total absolute probability movement contributed by the two
  `any` and `some` entries. It is undefined for zero movement.
* Conditional total variation after excluding those two tokens and
  independently renormalizing all the remaining tokens. This measures a
  change in their relative distribution; it does not show that those tokens'
  raw logits caused the change.
* Full and conditional displacement cosines for the primary transfer and its
  matched active comparator. Cosine is undefined when either displacement is
  zero and does not imply a common mechanism.
* Full-vocabulary top-five token IDs, strings and probabilities; deterministic
  `any`/`some` ranks (ties favor lower token ID); `any`/`some` margin and
  probability changes; and the largest probability movers outside those two
  tokens.

Descriptive medians, ranges and paired counts are reported separately for
development and evaluation and for each site and transfer direction. No
case is discarded based on its outcome. Top-token movement and full-vocabulary
TV are output measurements, not semantic or grammaticality judgments.

The cues differ in more than polarity: `never` changes temporal meaning,
`just` changes focus/time, `also` carries an additive presupposition and
`any` can be free-choice. These public examples were held out only from this
experiment's development selection, not known to be absent from pretraining
or other exposure. Equal input L2 does not imply equal functional dose.
These observations are limited to the fixed model, positions, contexts and
controls and do not establish a unique circuit or general negation capability.

## Reproduction and evidence scope

The `journal` mode streams the locally retained original gzip JSONL, checking
its pinned 85,368,356-byte SHA-256, the exact public manifest and original
result, and the pinned tokenizer vocabulary. It checks the 456-record action
order and input rows, saved `any`/`some` effects and original decision,
reported scalar dose consistency, all 128 self, 32 final-repeat and 64
final-cue full-logit equalities. It does **not** re-audit intervention landing
or all activation witnesses; the original full saved-data audit does that.
Saved records and hashes do not independently attest model execution.

The original complete journal is retained locally and was not published with
the original compact result. `journal` also creates a lossless, case-local
deduplicated gzip stream of the selected full float32 mask-logit vectors,
with exact source event indices and per-vector hashes in the report. The
288 logical selected event references share 224 physically stored vectors
after case-local deduplication; each original event remains separately indexed.
The public subset makes the full-vocabulary calculations replayable; its linkage
to the unshared complete journal is a reported, independently checked
relationship, not an execution attestation.

The following examples use POSIX shell line continuations.

```text
python -m looking_glass.bert_negation_distribution_posthoc journal \
  --events <original-events.jsonl.gz> \
  --manifest experiments/bert_negation_public_cases_v1.json \
  --result results/bert-negation-cue-cpu-v1-20261029/result.json \
  --vocab <pinned-vocab.txt> \
  --subset-output <new-subset.f32.gz> --output <new-report.json>
```

The published subset can instead be verified and replayed without the full
journal. First reconstruct it from the public parts as documented with the
result, then run:

```text
python -m looking_glass.bert_negation_distribution_posthoc subset \
  --subset <reconstructed-subset.f32.gz> --index <published-report.json> \
  --manifest experiments/bert_negation_public_cases_v1.json \
  --result results/bert-negation-cue-cpu-v1-20261029/result.json \
  --vocab <pinned-vocab.txt>
```

Both input modes require the original fixed manifest/result and the
`distilbert-base-uncased` vocabulary with SHA-256
`07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3`.
They do not load a tokenizer, model, image or GPU. The reader retains at most
36 selected 30,522-token vectors per case, limits each journal line to 2 MiB
and limits the number of records and subset bytes. Those are logical bounds,
not a measured process RSS ceiling. `--trace-memory` optionally measures the
Python-traced heap through report construction, excluding serialization,
native allocations and interpreter baseline, and makes the replay slower.
The reader checks for binary64 subnormal flushing before and after analysis;
ordinary softmax underflow below the binary64 range is still permitted.
Subset replay compares floating-point metrics with a relative tolerance of
`1e-12` and no absolute tolerance; exact zero and nonzero cannot match.
