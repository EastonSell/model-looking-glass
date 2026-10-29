# Prospective invariant-noun agreement response, v1

**Status: fixed prospective protocol.** Before assembling full cases, publish
the reviewed protocol and its model-free schedule. The fixed direction and the
earlier studies are exposed development evidence. Candidate word fragments
were examined with the tokenizer before this protocol; no full case or model
forward for this study has been made. Discovery and evaluation cases and all
outcomes are public and nonblind. Evaluation is held out only from this
follow-up's discovery inputs and outcomes, not from pretraining or the wider
research record. This is a fixed, small descriptive test, not an independent
sample or a significance test.

## Question and alternatives

The [previous noun-substitution study](https://github.com/EastonSell/model-looking-glass/tree/257a7f2518c231c5d5ccb9c11fdf456999ecf516/results/distilgpt2-agreement-noun-substitution-study-v1)
found a positive fixed difference between opposite-number and same-number
different-noun intervention responses. It also found substantial response on
subjectless fragments. Here, ask whether a signed response changes when
`this` is replaced by `these` while the noun's surface form and token are
held fixed, and compare that with changing the noun at the same determiner.

This design reduces the noun-inflection confound but changes a determiner and
its syntax. A positive difference may reflect determiner identity, a
determiner-sensitive context, lexical or nonlinear gain, or grammatical
number; it does not identify a unique abstract number representation. A
linear readout of an authentic number coordinate can have the same odd
response for both forms. Therefore a zero or negative result does not show
that number is absent. A prompt-independent additive output shift predicts
equal odd responses and zero differences. The active directions and
subjectless and demonstrative comparators are not a population null.

Use DistilGPT-2 `distilbert/distilgpt2`, revision
`2290a62682d06624634c1f46a6ad5be0f47f38aa`, weights SHA-256
`e1ff18884359fe8beb795a5f414feb85a6ce3d929ad019c0d958c039d2b94a1b`,
and tokenizer JSON SHA-256
`8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6`.
Use the already published `transformer.h.3.attn` last-input-token directions
from `results/distilgpt2-agreement-direction-training-v1/`: u SHA-256
`d1ddf7d4ee3103400ca6c23f25536e615135beebb24f2d326ad6fe5d46728be0`,
v1 `4a08130c4fd2c3557f9e40b92865c23411d395123c53251a286b63c796b476bb`,
v2 `4ca1f10e411e0476bcee76b98bc8515705e61035cfa6e8d71608e15af80f4b02`.
The site and u were selected on exposed data; v1 is a fixed number-by-lemma
modulation direction and v2 a fixed pseudorandom orthogonal direction. These
are matched input-dose active controls, not guaranteed non-number directions
or a randomization distribution. Do not retune the site, vectors, dose,
ordered pools, frames, criteria or execution order using new model outputs.

Tokenizer-only feasibility report SHA-256
`6e48358b332a6f3fad3e90bf6f55d72d510daee5f6d9ceeec0bfb0c0833979f0`
examined `this/these` and noun fragments with the pinned tokenizer; it did not
establish model competence on these cases. The prior saved run shows the
checkpoint and intervention worked on its own 48 target baselines and 324
delivered vectors; it does not test invariant nouns. New baseline adequacy is
measured only after the full cases and their rules have been frozen.

## Input construction and evaluation partition

After publication of this protocol, select the first two eligible noun forms
from each **disjoint ordered** pool, without using model output:

* Discovery: `sheep`, `deer`, `fish`, `moose`, `elk`, `cod`.
* Evaluation: `bison`, `squid`, `shrimp`, `carp`, `aircraft`, `spacecraft`,
  `swine`, `offspring`.

These forms are proposed for use with both determiners; corpus and dialect
usage, mass readings, semantics and frequency differ. The two forms selected
within a partition must be distinct, as must all four selected across the
partitions. A form is eligible only if its leading-space noun encodes as one
token, ` this <noun>` and ` these <noun>` differ only at the one determiner
token (`428` vs `777`), and all eight answer boundaries below append their
expected single token. Exclude forms or leading-space token IDs that appeared
as subjects in the earlier public natural discovery, direction, subject-form
and noun-substitution case bundles; record the exact public bundle hashes and
all eligibility decisions. If either pool yields fewer than two, stop and
publish an incomplete preparation; do not add or replace candidates after
observing a score. A selected evaluation noun cannot equal any discovery noun
or its token. Call the selected forms A and B in pool order per partition.

Use discovery frames `Before breakfast,` and `After dinner,`, and evaluation
frames `At midnight,` and `During autumn,`, in that order. Select four distinct
attractor pairs in order from `(gate,gates)`, `(lamp,lamps)`, `(rock,rocks)`,
`(tree,trees)`, `(arch,arches)`, `(box,boxes)`, assigning one to each frame
in discovery-then-evaluation order. Both leading-space forms must encode as
one distinct token; neither word nor token may overlap any selected noun or
earlier public discovery/direction/subject-form/noun-substitution subject or
attractor. No attractor form/token may repeat among selections. Record all
decisions and stop if exhausted. These tokenizer and public-history rules do
not use language-model accuracy or intervention effects. Each frame has its
own attractor pair, so frame and attractor-lemma effects are not separately
identified; the noun sets also differ between discovery and evaluation.

The four prior public bundles are the natural discovery binding SHA-256
`23575704f3e55d741123d3c1024accfb15aa0a048a84ce415bc765573d913ba0`,
direction cases `a68aa63ff06aa97c5c68d941fe27160c4c4698cdf070325b3cefa2b29ec58b13`,
subject-dependence cases `5f3a2c850454e74da2bdc3819d492357f829994c126e1ac95d90ca3813b0643e`,
and noun-substitution cases `94910e59415936a09c18f81478e0eaffb790c9b69d0962ab3f6b70947fbe1dfe`.
From the binding, collect every `input_ids[subject_index]` as a subject and
every `input_ids[-1]` as a prior attractor. From direction cases, collect
target `token_ids[subject_index]` as subjects and every row's last token as
an attractor/context token. From each later case bundle, collect target
`token_ids[subject_index]` as subjects and each row's last token when
`condition != 'none'` as an attractor; this includes mass and subjectless
locatives. Check the stated form and token exclusions separately.

For each partition, frame and one of three contexts, form all four targets:
`<frame> this <A> nearby`, `<frame> these <A> nearby`, and the corresponding
two for B; replace `nearby` with `near the <singular attractor>` or
`near the <plural attractor>` in the other two contexts. Add one shared
subjectless fragment per frame/context: `<frame> nearby` or `<frame> near the
<attractor>`. Add `<frame> this nearby` and `<frame> these nearby` for each
frame as noun-omitted demonstrative comparators. There are 24 targets, six
subjectless fragments and four demonstrative comparators per partition:
68 inputs total. The latter can be read as demonstrative pronouns, so do not
describe them as subjectless, neutral or a controlled removal of the noun.
The fragments change syntax, position and context, and are descriptive,
not controlled subject deletion or a feature-free null. `nearby` also
changes the final token and position relative to locatives.

Within each partition/frame/context, require identical target token count
and IDs outside the determiner and noun token positions; the this/these pair
for each noun must differ at exactly the determiner position, and the A/B
pair for a determiner at exactly the noun position. The two demonstrative
comparators in each frame must likewise differ only at the determiner token.
Locative singular and plural rows must differ only at the final attractor token. Record text,
token IDs, exact subject/determiner/final positions and case identifiers;
the same A and B are reused across a partition's frames and contexts. Inputs
must have at most 12 tokens. For answers ` is`, ` are`, ` was`, ` were`,
` the`, ` a`, ` in`, ` on`, require IDs `318,389,373,547,262,257,287,319`
and `encode(text + answer) == input_ids + [answer_id]`. A failure stops
preparation; do not change a frame or case after seeing model output.

Compare each input and each of its eight answer extensions against every
other case group in both complete-row-prefix directions, and require eight
distinct extensions within each group. A case's base row is naturally a
prefix of its own answer extensions; exempt only that intended relation.
Compare all 612 case/answer sequences against the latest reviewed public
inventory including grouped continuations, relevant pending public bundles,
and the 128-row natural original public support SHA-256
`5638b5ccc1d6585184fcd359e825ac3c2f6a1e377e24afde78bc5a9a2fc845d6`
and 256-row successor pool SHA-256
`6303eb0713d64b020eea3fe491a686c9020ea424754443372aef73749bc87bc4`.
Ask the natural custodian about public-support scope and later protected
pools. Old-split coverage is conditional on the published source-bound
preparation; do not query private membership or rows. Ask the injection
custodian about the fixed site and any protected-row status. Publish cases
and a reviewed planned reservation, and refresh public inventory and both
custody observations before launch. These comparisons cannot establish
absence of unknown exposures.

## Intervention, fixed ordering and evidence

At batch 0, the last input position of output element 0 of
`transformer.h.3.attn`, deliver `+u, -u, +v1, -v1, +v2, -v2` with nominal
input-space L2 q=2.0. This was chosen from the prior study, not from these
new outputs. Starting from each original f32 source, evaluate componentwise
in binary64 `source + signed_q * direction`, then directly round to IEEE
f32 nearest-even. Record full before/after witnesses, source norm, realized
L2, signed projected displacement and changed elements. Require finite values,
the correct projection sign, and both norm and projection magnitude within
1e-5 relative of 2; otherwise retain the failed witness and stop incomplete.
Equal input L2 does not imply equal functional strength.

The exact schedule is `agreement_invariant_schedule_v1.py` and is tested by
the companion model-free test. The fixed seed string is
`agreement-invariant-noun-v1:20261029`, UTF-8 encoded. For sorting, compute
SHA-256 of `seed|split|domain|identity`, order by digest bytes, and break an
exact digest tie by the identity string. This is deterministic ordering, not
random sampling or a randomization test. Case identity is
`<split>_target_f<f>_c<c>_n<n>_<this|these>`,
`<split>_fragment_f<f>_c<c>`, or
`<split>_demonstrative_f<f>_<this|these>` with zero-based frame and
selected-noun indices; c=0 means none, c=1 singular attractor and c=2 plural
attractor. Pair identities
omit the last determiner component. The paired members share an arm order.

The six patterns start with `u+,u-,v1+,v1-,v2+,v2-`; reverse every sign
for odd pattern j and rotate the six entries left by `2*(j//2)` for j=0..5.
For a target pair, sign reversal parity is `(frame+context+noun)%2`; for a
demonstrative pair it is `frame`. Independently sort the pair identities by
the `pair-pattern` domain within each parity and assign ascending pattern IDs
with counts `[2,2,3,2,2,3]` in discovery and `[2,2,2,3,3,2]` in evaluation.
Sort fragments by `fragment-pattern`, assigning ascending pattern IDs with
counts `[2,1,0,1,1,1]` and `[1,2,2,0,0,1]` respectively. Then sort pairs by
`pair-order` and fragments by `fragment-order`. Execute one fragment after
pair numbers 2, 4, 6, 8, 10 and 12. Within a target pair run `these` first
iff `(frame+noun+split_index)%2==1`; within a demonstrative pair use
`(frame+split_index)%2==1`. Discovery (split index 0) runs wholly before
evaluation (1). Continue to the evaluation cases regardless of discovery's
baseline or scientific outcome; only the stated technical failure stops the
one-shot attempt. For every case use clean, no-op, the six assigned arms, and
clean-after. In each partition 17 cases have each sign first; pair member
order is seven this-first and seven these-first. Each arm occupies each slot
11 or 12 times across the 68 cases. Record and verify the actual 612-action
schedule and case identity before model work.

Save f32 full last-position vocabulary logits per action and whole/selected
hook before/after witnesses. Require clean, no-op and clean-after last-logit
equality and no-op capture equality; check all full-sequence logits finite
before marking a forward complete, while disclosing that unsaved positions
cannot be independently replayed. The 612 forwards have at most
`612*50257*4 + 476*(2*12*768*4 + 2*768*4) = 161,048,208` logical tensor
bytes on successful completion before deduplication. Set the per-forward
payload cap to 4 MiB, successful retained tensors to 256 MiB, event journal to
32 MiB, apparent attempt files to 512 MiB and external receipts to 16 MiB.
Reserve 2,492,208 bytes for a failed 12-position full-sequence return and
6,144 bytes for an invalid delivered vector. These are apparent/logical byte
limits, not a measurement of allocated filesystem blocks. Use a source-bound gated CPU
supervisor, retain partial evidence on failure and never silently retry an
uncertain forward. Direct-child observations do not authenticate execution
or prove descendant drain. Allow at most 1,800 seconds after the supervisor's
grant; its limit does not cover time before grant. Stage output on E:, require
at least 2 GiB available there and 512 MiB on the actual cache-staging volume
(the pinned loader stages 355,679,200 bytes), with both amounts accounted for
if they share a volume. Recheck the volumes immediately before launch; this
is a point observation, not a reservation. Avoid duplicate full archives.

## Measurements, eligibility and fixed decisions

For margin `m = z(is) - z(are)`, direction a, and each case, define
`O_a = (m(+2a)-m(-2a))/2` and
`E_a = (m(+2a)+m(-2a))/2 - m(clean)`. For each of the six frame/context cells
in each partition, write `A_this, A_these, B_this, B_these` for its four O
values and calculate, separately for u, v1 and v2:

* `K_A = A_this-A_these` and `K_B = B_this-B_these`;
* `S = (K_A+K_B)/2`, the signed determiner/number contrast;
* `D_num = (abs(K_A)+abs(K_B))/2`;
* `D_lex = (abs(A_this-B_this)+abs(A_these-B_these))/2`;
* `Delta = D_num-D_lex`;
* `L = ((A_this-B_this)+(A_these-B_these))/2`;
* `I = K_A-K_B`.

Retain all individual signed responses, all four factorial values, the
same-determiner noun differences, K_A/K_B, S/L/I, distances and Delta, and
their zero/positive/negative counts. Use every fixed cell, `math.fsum` in
canonical cell order divided by its fixed count, and no normalization by a
potentially zero distance. Report frame, context and noun breakdowns and
actual interventions. Because forms and frames recur, the six cells are not
six independent draws.

For each partition, a target baseline is correct only if its clean `is-are`
margin is strictly positive for `this` or strictly negative for `these`;
ties are incorrect. Adequacy requires at least 18/24 overall, 9/12 in each
frame, 9/12 for each noun, 6/8 in each context, and at least 4/6 cells with
all four forms correct. Report every case and failed/tied baseline; never
filter by response or replace a case. If a partition fails any adequacy rule,
label its results baseline-limited, not evidence that number is absent.

Conditional on adequate baselines, report two fixed descriptive criteria
separately, including pass/fail/mixed outcomes:

1. **Signed sensitivity:** mean S_u is at least 0.05 logit and strictly
   exceeds each mean of the six individual `abs(S_v1)` and `abs(S_v2)`;
   at least 4/6 S_u values and both frame means are positive, and the mean
   K_u is positive for each noun across its six cells.
2. **Number versus noun change:** mean Delta_u is at least 0.02 logit and
   strictly exceeds each mean of the six individual `abs(Delta_v1)` and
   `abs(Delta_v2)`; at least 4/6 Delta_u values and both frame means are
   positive.

These floors are fixed descriptive choices informed by earlier exposed
results, not calibrated effect sizes, confidence intervals or significance
tests. Evaluation's decisions are fixed now and are not selected from
discovery outcomes. A failed criterion with adequate baselines is a negative
result under that rule; a positive result supports only this specified
response pattern, not an isolated grammatical feature. Both, neither or
only one criterion can pass.

For example, `m(n,t)=3n+2t` with n=+1/-1 and t=+2/-2 gives equal odd
responses and zero S/Delta even if n is a genuine number coordinate. In
contrast, `m(n,t)=3n+2t+0.5*n*t` gives positive S, which is also possible
when n denotes determiner identity rather than grammatical number. For
responses `(A_this,A_these,B_this,B_these)=(0,0,2,2)`, Delta is -2;
zero and negative values must not be silently discarded.

Also report the same O/E and factorial contrasts for `was-were`, `the-a`
and `in-on`, all six subjectless and all four demonstrative comparator
responses per partition, per-direction full-vocabulary top-20, eight target
ranks, KL(clean||arm), top-one changes and actual dose. Use stable softmax
and deterministic token-ID tie breaking. Publish positive, negative,
baseline-limited or incomplete findings with the fixed cases, source and
bounded saved evidence; after the observed owner exit perform an independent
saved-byte replay. Source and process observations are cooperative, and the
absence of unsaved or unknown exposure cannot be proved by the replay.
