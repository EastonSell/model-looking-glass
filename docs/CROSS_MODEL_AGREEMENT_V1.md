# Cross-model subject-state intervention, v1

This fixed public prospective protocol compares DistilGPT-2 and DistilBERT on
the same syntactic prefixes. It is not a blind evaluation or a claim that the
models share a mechanism. The held-out qualifier means held out from the
studies' explicit development cases and candidate-site selection, not from
either checkpoint's pretraining data. Known-public exclusions are scoped; they
cannot establish an exhaustive exposure history.

## Question and intervention

For each model, does replacing the layer-3 subject-token hidden state in a
target sentence with a dose-matched displacement toward the same lemma in the
opposite number change the binary agreement margin toward the donor's number
more than (a) a different-lemma same-number donor and (b) a fixed pseudorandom
direction of the same input L2? This is a test of one concrete intervention,
not proof of an isolated number representation. Subject inflection, lexical
identity, semantically meaningful donor differences, input-dependent gain and
the fact that equal input L2 does not imply equal output effect remain
alternative explanations. A null result cannot establish feature absence.

The candidate block is zero-indexed block 3 of each six-block model; hook its
whole hidden-state output and replace exactly the subject token's 768-vector.
For GPT-2 this is `transformer.h[3]` output element 0; for BERT this is
`distilbert.transformer.layer[3]` output element 0. Invoke exactly once per
unbatched input. The pre-intervention donor and target states are captured from
separate clean forwards; each arm starts from the original target state. Select
`q=min(2.0, norm(opposite-target), norm(same-target))`, calculated separately
for every target and model. Scale each of the two donor displacement vectors
to q. Draw one random Gaussian 768-vector using NumPy PCG64 seed
`20261029 + case_index` and scale it to q. A nonfinite or zero norm makes that
case incomplete; retain it and do not replace it. Record actual rounded
float32 before/after vectors, dose, model site, hook count and complete final
vocabulary logits. Include clean, self-replacement, opposite donor,
same-number donor, random control and clean-after. Random-vector similarity
or dissimilarity to number is not assumed; it is one fixed active control, not
a null distribution. Compare clean, self and clean-after logits exactly.
Each active replacement must have strictly positive realized L2 and relative
dose error at most `1e-4`. The L2 error between the rounded displacement and
its intended binary64 direction, divided by q, must also be at most `1e-4`;
a rounded-to-zero or otherwise out-of-tolerance arm leaves the attempt
incomplete. Both tolerances are fixed before model use.

The GPT-2 score is the final-prefix next-token `logit(' is')-logit(' are')`.
The BERT score is `logit('is')-logit('are')` at the `[MASK]` following the
same text. BERT also sees its terminal `[SEP]` and can use bidirectional
context; GPT-2 is causal and has no mask. Their tokenizers, heads and
logit scales differ. Compare signs and controls *within each model*, never raw
logit magnitudes across models. Surface similarity does not establish a common
internal computation. Also retain `was/were` margins, top-1 changes and full
vocabulary total variation and KL from the clean output.

For a singular target let `s=-1`, and for a plural target `s=+1`. For each
arm let `effect=s*(margin_arm-margin_clean)`; positive means a change toward
the opposite-number donor. The per-case contrasts are `effect_opp-effect_same`
and `effect_opp-effect_random`. Report all 32 cases and each model separately.
The fixed descriptive prediction is that, for each model, both mean contrasts
are strictly positive and at least 24/32 cases are positive against both
controls. Treat a model as baseline-limited when fewer than 24/32 clean
is/are margins have the correct sign, and report the full denominator without
filtering. The 24/32 rule is a descriptive threshold, not a statistical test;
two frames and four lemmas are reused and outcomes are dependent. Report
per-frame and per-lemma outcomes, dose distribution, and any failed control.
No results are used to change the location, cases, controls, or threshold.

## Inputs and preparation

Use `distilbert/distilgpt2` revision `2290a62682d06624634c1f46a6ad5be0f47f38aa`
and `distilbert/distilbert-base-uncased` revision
`12040accade4e8a0f71eabdb258fecc2e7e948be`. The tokenizer-only case generator
uses four subject lemmas `basket/baskets`, `barrel/barrels`, `cup/cups`, and
`crate/crates`, with fixed same-number donors paired basket↔barrel and
cup↔crate. For each of two frames, cross subject lemma and number with both
numbers of the frame's attractor:

* `Inside the warehouse, the {subject} near the {chair/chairs}`;
* `At the marketplace, the {subject} beside the {table/tables}`.

Each exact prefix is the GPT-2 input; append ` [MASK]` for BERT. Every subject
and attractor form must be a single distinct token in both pinned tokenizers,
and, within each frame/attractor-number cell, all inputs must have identical
lengths and identical tokens outside the subject position. Derive and save the
actual positions and IDs. The exact tokenizer-only cases must be published and
checked against the latest applicable public inventories and pending public
bundles before any model forward. Compare both prefix directions for GPT-2
inputs and the declared ` is`, ` are`, ` was`, ` were` extensions, including
both original and successor public natural-agreement supports. Compare BERT
complete masked rows and SEP-removed rows against its known public inventory.
There is no private membership query. Refresh inventory currentness before
launch and document the relevant custodians' scoped reports.

## Runtime and evidence

Use the pinned cached assets, float32, `eval`, no-grad, fixed seed, eager
attention, one unbatched row, bounded retained subject traces and exclusive
single-model ownership. Invoke the source file directly using a base Python
interpreter with `-I -S -B`, an explicitly supplied existing dependency site
directory and a fresh, empty `-X pycache_prefix` directory. Local sources and
all snapshot assets are byte-checked against their pins; installed dependency
contents and the absence of concurrent source mutation remain trusted.
Save the ordered manifest, runtime/software/device,
checkpoint and source hashes, all per-action selected vocabulary logits,
before/after vectors, arm order and any failure. Persist each forward-start
record before model entry and retain available returned payloads on a best-effort
basis even during an interruption; an unmatched start remains uncertain. Retain the original attempt;
an interrupted or partial attempt is not a complete result and is not silently
retried. CPU results and any GPU execution have distinct device identities.
An observed direct-child exit records that child's status only; it does not
prove operating-system-wide descendant, allocator, or driver cleanup.
Scientific replay requires a matching successful external direct-child exit
observation and rejects any attempt containing a failure sentinel.
For GPU, coordinate with the shared device owner, admit a workload-specific
small profile, synchronize memory observations, retain no whole-model GPU
trace, and report allocated/reserved peaks and free-device samples separately.
Keep full raw outputs for an independently reviewed saved-byte replay.
