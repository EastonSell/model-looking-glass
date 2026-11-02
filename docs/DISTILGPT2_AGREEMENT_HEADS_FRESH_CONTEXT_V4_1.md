# Fresh-context fixed-donor agreement check, corrected v4.1

## Question and scope

This version supersedes the unforwarded v4 case proposal (manifest SHA-256
`e90584985df00a7749687d49139a83aa7488bff47f65af16f79b24572bba33ee`).
The original proposal reused three subject lemmas from the earlier exposed
head-development rows, so its unqualified novelty wording was incorrect.
No model forward or exposure reservation was made on that proposal. This
correction changes only the subject lemmas and their prescribed crossing; it
was selected using tokenizer structure and the public prior case lists, not
model outcomes on these frames.

Does the previously fixed layer-3 head 4 retain the same signed donor-number
contrast, compared with the adaptive head-8 development runner-up, on new
public sentence frames and subject lemmas absent from the earlier head-study
development and evaluation cases? This protocol fixes the inputs,
mapping and comparison rule before any model forward on these cases. The
heads and the fixed-other-lemma design were chosen after earlier exposed
development and exploratory results. The new cases are public and not blind;
four hand-written frames cannot establish population-level generalization or
pretraining independence.

## Fixed cases

Construct exactly these four frames with `subject` replaced by each member
of its assigned two-lemma pair, in the listed order. For each lemma, use its
singular form followed by its regular plural (suffix `s`). Order each quartet
as A singular, A plural, B singular, B plural. The frames and pairs are:

1. `Before the ceremony, the {subject} near the lanterns`: singer, dancer.
2. `After the performance, the {subject} beside the museum`: lawyer, actor.
3. `During the lesson, the {subject} behind the curtains`: singer, lawyer.
4. `Throughout the morning, the {subject} by the orchard`: dancer, actor.

Each lemma occurs in two frames; each frame has two lemmas and two numbers.
The lemma novelty claim is limited to the earlier head-study development and
evaluation cases; it does not claim the words were absent from all other
model inputs or pretraining.
The last intervening noun is plural in frames 1 and 3, singular in frames 2
and 4. Before any forward, use the pinned DistilGPT-2 tokenizer to require
four distinct single-token subject variants at one otherwise identical local
position in each quartet, no padding and at most 12 tokens per row. Verify
that appending ` is`, ` are`, ` was` and ` were` yields exactly one additional
token with IDs 318, 389, 373 and 547 respectively. If any structural test
fails, stop; do not silently select replacement cases from model behavior.

Publish the tokenized manifest and compare all 16 bases and their 64 declared
answer extensions, in both complete-row-prefix directions, with the latest
reviewed public inventory, known pending bundles and both published natural
support pools. Obtain the applicable current custodian scope, publish a
planned-only exposure reservation, and recheck the actual resource and input
state before a one-shot run. A public overlap check is not a private split
query and does not establish unknown exposure or pretraining exclusion.

## Interventions and outcome

At the final-token input to layer 3 `attn.c_proj`, use fixed heads 4 and 8.
For a receiver's local quartet index, the same-number donor is index XOR 2
and the opposite-number donor is index XOR 3. Both donors have the same other
lemma but different inflected token forms. Independently for each head and
receiver, interpolate both donor displacements to q = 5% of the receiver head
slice's L2 norm. Require both distances at least q, actual dose relative error
at most 1e-5 and within-pair dose ratio at most 1.00001. Precompute the full
plan before any patch; if either fixed head fails on any receiver, report
`dose_ineligible` and stop after the 16 clean observations. Do not replace a
head or case based on the result.

Let margin be the final-token `is` logit minus the `are` logit. For each
receiver, D is +1 for a singular or -1 for a plural receiver times
`margin_same - margin_opposite`. Treat ties as incorrect for the clean
baseline and as nonpositive for D. The baseline is adequate if at least
12/16 clean signs and at least 3/4 in every frame are correct. With an
adequate baseline, the fixed descriptive rule passes only if head 4's mean D
is at least 0.05 logits and exceeds head 8's mean absolute D, at least 10/16
head-4 D values are positive, and at least three frame means are positive and
exceed the corresponding head-8 mean absolute D. Otherwise report
`does_not_pass_fixed_rule`; an inadequate baseline is `baseline_limited`.
No case-wise outcome may change these criteria or the fixed heads.

Save clean full logits, projection-input and replacement witnesses, exact
landings and untouched bits, 32 self-patch controls and 16 clean repeats;
the complete eligible run has 128 forwards. Save and report all per-case
donor-to-receiver distances and interpolation fractions q/d, absolute and
receiver-relative delivered doses, margins and contrasts,
frame and lemma dependence, plus exploratory full-vocabulary or top-token
effects. Require observed child completion and independent saved-byte replay
before publishing either a positive, negative, baseline-limited or
dose-ineligible outcome.

The other-lemma control removes the prior between-lemma donor mismatch, but
singular and plural donor forms are different lexical tokens. Head 8 was
chosen adaptively and is not a random-head or downstream-dose-matched null;
equal relative input doses can yield unequal absolute or output changes.
Any comparison by intervening-noun number is descriptive and confounded by
frame and subject pairing in this small fixed design.
This comparison cannot by itself identify a unique grammatical circuit,
measure generated text, or generalize beyond these few cases.
