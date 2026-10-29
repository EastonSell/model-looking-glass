# DistilGPT-2 public year-comparison screen

This fixed public, non-blind development study measures next-token behavior on
short year-span prompts using the pinned `distilbert/distilgpt2` revision
`2290a62682d06624634c1f46a6ad5be0f47f38aa`. It draws on the year-span task
studied in a different GPT-2 checkpoint by
[Hanna, Liu and Variengien](https://arxiv.org/abs/2305.00586). A recent
[DistilGPT-2 preprint](https://arxiv.org/abs/2610.37157) reports stronger
behavior on its own distribution; its exact model and cases are not known to
match this study. This experiment tests behavior, not the preprint's proposed
mechanism, an arithmetic algorithm, or a unique circuit.

## Fixed prompts and controls

Use suffixes `12, 21, 32, 43, 54, 65, 76, 87`, event nouns `war` and `reign`,
and century `17`. For each suffix `YY` and event, use all three templates:

1. Forward: `The {event} lasted from the year 17{YY} to the year 17`
2. Reverse: `The {event} lasted until the year 17{YY}, after starting in the year 17`
3. Neutral: `The {event} was recorded in the year 17{YY} beside the year 17`

There are 16 matched triples and 48 prompts. No cases are filtered based on
model outputs. The forward and reverse contexts predict opposite chronological
directions; the neutral wording provides a contextual baseline, but it is not
assumed to erase historical or language priors. Fixed templates and years
are public examples, not a random sample or held-out evaluation.

Before any model forward, the pinned tokenizer must reproduce each full prompt,
the initial century and start suffix as separate expected tokens, and all 99
`01`–`99` continuations as the prompt tokens followed by distinct single
tokens for their two decimal digits. `00` is a single next-token choice, but
retokenizing the full text ending in `1700` merges the century and suffix;
its mass is reported separately and is not claimed as a tokenization-aligned
full-text continuation. The final ` 17` token must be identical across all
templates. For fixed template and event, different start suffixes must change
only the first start-suffix token. Reject the entire attempt on structural
failure; do not substitute or select cases using model outcomes.

## Predeclared comparison

For start suffix `y`, define the lower set as the eight values `y-8, ..., y-1`
and the upper set as `y+1, ..., y+8`. Both sets contain eight single-token
two-digit suffixes; `y` itself is excluded and reported separately. Let `m_H`
and `m_L` be the maxima of the upper and lower float32 logits promoted to
binary64. Define the signed upper-minus-lower **log-mass ratio** by

`M = (m_H - m_L) + log(fsum(exp(h - m_H))) - log(fsum(exp(l - m_L)))`.

The primary contrast for each matched triple is `M_forward - M_reverse`;
positive means that the forward wording shifts relative mass toward later
suffixes compared with the reverse wording. Also report `M_forward - M_neutral`,
`M_reverse - M_neutral`, every template's `M`, each individual suffix logit,
the exact-start suffix logit, and the full-vocabulary probability mass in
the upper, lower, equal, all 99 context-aligned two-digit tokens, the separate
`00` next-token choice, and the remaining vocabulary.
Report each template's sign as well as the contrast: a positive forward-minus-
reverse difference does not itself mean the two directions have opposite
signs. The forward and reverse sentences differ in words and token positions,
so the contrast cannot uniquely isolate a numerical comparison. Compute each 16-triple mean as `math.fsum`
of the fixed ordered values divided by 16. Retain all signs, including null
and reversed comparisons. Equal-size nearby candidate sets avoid a simple
unequal-count advantage, but their numeric frequencies can differ and the
neutral sentence cannot perfectly control the change in wording.

Run 48 unhooked, CPU float32, evaluation-mode forwards with the locally cached
verified checkpoint, eager attention, no KV cache and one intra-op thread.
Save each complete 50,257-element final-position float32 logit vector, exact
input IDs, case metrics, source/runtime identities, journal and direct-process
exit/failure observation. Retain bounded returned data before post-return
numeric validation; incomplete attempts stay incomplete. Publish the protocol,
fixed cases and runner source before the first pretrained forward. Independent
saved-byte reconstruction is required before reporting. No intervention or
mechanism attribution is part of this screen.
