# DistilGPT-2 public recipient screen and conditional patch plan

This is a small public, non-blind development experiment on the pinned
`distilbert/distilgpt2` revision `2290a62682d06624634c1f46a6ad5be0f47f38aa`.
It asks whether the model prefers the non-giver name in a constrained
two-name completion. It is not a reproduction of the GPT-2-small circuit in
[Wang et al.](https://arxiv.org/abs/2211.00593), nor an assessment of all
recipient reasoning. [Zhang and Nanda](https://arxiv.org/abs/2309.16042)
show that patching conclusions depend on corruption and metric; retain both
directions and the individual logits. A recent unreviewed
[DistilGPT-2 preprint](https://arxiv.org/abs/2610.37157) reports a weak IOI
baseline on its own distribution; its exact checkpoint and examples are not
established to match ours.

## Fixed inputs

Eight name pairs, in this order, are `(John, Mary)`, `(Michael, Sarah)`,
`(Daniel, Emma)`, `(Peter, Lisa)`, `(Alice, Claire)`, `(Anna, Susan)`,
`(Paul, Mark)` and `(David, Robert)`. The two templates are:

* `When {first} and {second} went to the store, {giver} gave a book to`
* `After {first} and {second} went to the park, {giver} handed a gift to`

For every pair and template, cross initial-name order with giver identity,
giving four cases and 64 total. The intended recipient is the other member of
the pair; this is a forced choice between two single-token names, not a claim
that either name must be the top next token. Do not discard cases based on
model outcomes. Preserve the exact text, input IDs and both answer IDs.

Before any forward, the pinned tokenizer must reproduce each whole prompt,
each space-prefixed name as one distinct token, and the complete prompt plus
either answer as the prompt IDs followed by that answer token. Within each
template all 32 prompts must have the same 14-token alignment:
initial names at positions 1 and 3, repeated giver at 9, final ` to` at 13.
For fixed initial order, opposite-giver prompts must differ only at position 9.
Fail before execution if any condition is false; do not replace inconvenient
names or cases after observing model outputs.

## Behavior screen

Run one unhooked, CPU float32, evaluation-mode forward per prompt using the
locally cached verified weights, eager attention, no KV cache and one intra-op
thread. Record the complete 50,257-element final-position logits and both
names' logits, ranks, target log probabilities and top token. For each case,
`margin = logit(recipient) - logit(giver)`; a strict positive margin is a
forced-choice success and a tie is not. A matched pair consists of the two
opposite-giver cases at fixed names, template and initial order.

The *predeclared exploratory gate* passes only if all of these hold:

1. At least 24 of the 32 matched pairs have positive margins in **both**
   directions.
2. Each template has at least 12 of its 16 matched pairs positive in both
   directions.
3. At least six of the eight name pairs have at least three of their four
   matched pairs positive in both directions.

All 64 margins, template/order/name summaries and the exact failure state are
reported regardless of the gate. These cases are correlated and public; the
threshold is an engineering decision rule, not a confidence interval or a
claim about the model's general accuracy. If the gate fails, stop: report the
low or unstable baseline and do not run the conditional intervention.

## Conditional activation patch, fixed before the screen

Only after the gate passes, use every case of template 1 (32 cases; no
outcome-based filtering). First obtain a passive layer-3 block-output recording
for each case and require its final logits to match that case's unhooked
baseline exactly. The planned hypothesis is that replacing the **entire**
768-component residual at the final position after zero-based transformer
block 3 with the corresponding opposite-giver donor residual can shift the
recipient-versus-giver logit margin toward the donor's intended recipient
(the target's giver). The donor's observed preference is reported separately. This
does not isolate a head, a semantic variable or a unique circuit.

For each target, retain full final logits and run: hooked no-op; same-site
self patch; same-site opposite-giver donor at identical name order; same-site
donor with the same giver but swapped initial name order; opposite-giver donor
patched at position 0; and genuinely unhooked clean-after. The opposite-giver
comparison is done in both directions. The position-0 donor precedes the
changed giver in a causal model and should be bit-identical to the target at
that position; it checks delivery/control mechanics, not a strong independent
test of site specificity. The same-giver/different-order donor need not be
neutral and is reported rather than assumed harmless.

Capture pre/post block-output tensors and assert the intended destination
slice equals the donor bits and every other coordinate is unchanged. Verify
self/no-op and clean-after logits against the unhooked baseline, hook removal,
source/target token alignment, and CPU runtime/model state. Record every arm,
including null or reversed shifts. The primary patch statistic is the signed
change in the *target* case's recipient-minus-giver logit margin; also retain
both logits, both log probabilities, ranks, full logits and donor baselines.
Define the signed effect for each target and arm as
`delta = margin_patched(target) - margin_clean(target)`; negative values reduce
the target's recipient margin. For the opposite-giver donor only, this is a
shift toward the donor-intended name. Report each direction and arm and the mean
delta for every arm over all 32 predeclared targets, including ties and
reversals; compute each mean as `math.fsum` of the 32 ordered deltas divided
by 32. Preserve both endpoint logits and per-name-pair outcomes. A
donorward shift can reflect lexical or positional changes or erasure, and
does not establish uniquely localized recipient computation. No normalized
recovery ratio is used as the sole outcome.

The conditional plan requires 32 passive-recording forwards and 192 arm
forwards after the 64 screen forwards, for at most 288 total. Model outputs,
intervention witnesses, journal and direct-process exit/failure information
must be preserved for independent replay; any incomplete attempt stays
incomplete. Runtime and source identities, exact public inputs and the result
are published. Publish and bind the protocol, all fixed case rows and runner
source before the first pretrained forward. Store each full 14×768 passive
block-output capture, including positions 0 and 13, for donor landing replay.
Rank exact ties by lower token ID; compute each log probability in binary64 as
`(target_logit - maximum) - log(fsum(exp(logit - maximum)))` and retain the
original float32 logits. The case set and order are deterministic, not
randomized.
