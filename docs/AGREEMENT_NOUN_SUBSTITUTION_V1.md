# Prospective agreement noun-substitution comparison, v1

**Status: fixed prospective protocol.** The earlier direction, site and results are exposed
development evidence. No cases for this follow-up have been assembled or
forwarded. During protocol review, the listed word forms were inspected with
the tokenizer only; no evaluation sentence or model output was constructed.
Publish this reviewed protocol before assembling the fixed public
cases. Separate case, exposure, custody, executor and prelaunch checks are
required before a forward. All cases and all outcomes are public and
nonblind; they are not an independent-sample statistical test.

## Question and fixed alternatives

The [previous subject-form study](https://github.com/EastonSell/model-looking-glass/blob/bc4caaace1746358ae6cfcfeca62346174b16c17/results/distilgpt2-agreement-subject-dependence-v1/RESULTS.md)
found a larger `is − are` response to a fixed L3-final direction for singular
than plural forms in 31/32 matched cells, and substantial response in
subjectless fragments. Here we compare *different-noun input substitutions*:
does changing both noun and number produce a larger difference in the signed
directional response than changing the noun while keeping number fixed? This
is a comparison of responses to the same activation intervention, not a donor
activation patch. The two noun pairs, their two frames and three contexts are
fixed and reused; morphology and lexeme remain entangled.

Use the published DistilGPT-2 checkpoint revision
`2290a62682d06624634c1f46a6ad5be0f47f38aa` and L3-final attention-output
directions from `results/distilgpt2-agreement-direction-training-v1/`:
`u` SHA-256 `d1ddf7d4ee3103400ca6c23f25536e615135beebb24f2d326ad6fe5d46728be0`,
`v1` `4a08130c4fd2c3557f9e40b92865c23411d395123c53251a286b63c796b476bb`,
and `v2` `4ca1f10e411e0476bcee76b98bc8515705e61035cfa6e8d71608e15af80f4b02`.
The site and u were selected on exposed discovery; v1 is a number-by-lemma
modulation direction and v2 a fixed pseudorandom orthogonal direction. They
are active controls, not a randomization distribution or guaranteed
non-number directions. Do not refit site, vectors, dose, case pool or
thresholds from these new outputs.

The fixed prompt-independent additive-output alternative gives the same
odd response on every input, so both same- and opposite-number differences
are zero. Input-dependent gain or surface-sensitive nonlinear readout can
produce a positive number contrast and a positive distance difference
without an abstract number feature. Conversely, a true number coordinate
with a linear readout may give no response difference. A null outcome here
does not establish that number is absent. A subjectless response is an
additional descriptive control: the fragments change syntax, token position
and context, so they are not clean subject deletion or feature-free nulls.

## Fixed public inputs

After this protocol is published, use only the pinned tokenizer JSON SHA-256
`8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6` to
select the first two eligible pairs from each ordered list:

* Regular: `(hand,hands)`, `(arm,arms)`, `(leg,legs)`, `(eye,eyes)`,
  `(ear,ears)`, `(finger,fingers)`, `(toe,toes)`.
* Irregular: `(foot,feet)`, `(tooth,teeth)`.

These are body-part nouns, chosen to reduce a broad semantic-category
difference, not to eliminate lexical, semantic, spelling or morphology
confounds. A pair is eligible if its two distinct leading-space forms each
encode to one distinct token and neither form/token occurs as a subject in
the exposed natural discovery, earlier direction-study cases, or earlier
subject-study cases. Reject reused forms/tokens among selections and record
every eligibility decision. If either list does not yield two pairs, stop
as incomplete; do not substitute an unlisted pair. Call them R1, R2, I1, I2
in selection order. Regular and irregular each form their own same-morphology
different-noun pair. No model score is used for selection.

Use frames `Before sunrise,` and `After sunset,`, in that order. Select the
first two eligible distinct attractor pairs from `(stone,stones)`,
`(wall,walls)`, `(gate,gates)`, `(lamp,lamps)`, `(rock,rocks)`,
`(box,boxes)`. Each leading-space form must be one distinct token and must
not overlap a selected subject or any prior discovery/direction/subject
subject or attractor form/token; record every decision and stop if exhausted.
Assign one pair to each frame in selection order.

For every frame, morphology group, lemma and number, use the three target
forms `<frame> the <subject> nearby`, `<frame> the <subject> near the
<singular attractor>`, and the corresponding plural attractor. Add one
shared subjectless fragment for each frame/context: `<frame> nearby` or
`<frame> near the <attractor>`. The first condition changes final token and
position compared with the locatives; attractor form changes lexical token.
Do not interpret the subjectless/target comparison as controlled deletion.
This produces 48 target inputs plus six shared fragments = 54 inputs. The
same four lemmas are used across both frames: this is a prospective fixed
test, not a separate lexical holdout or secret frame holdout.

Before any model forward require all target inputs in each frame/context to
have the same token count and identical IDs outside one subject token.
Locative singular/plural pairs must be identical outside the last token for
each subject and fragment. Record exact token IDs, subject/final positions,
lengths and the 54 ordered input digest. Inputs must have at most 12 tokens.
For ` is`, ` are`, ` was`, ` were`, ` the`, ` a`, ` in`, ` on`, require the
published IDs `318,389,373,547,262,257,287,319` respectively and
`encode(input + answer) == input_ids + [answer_id]`. A failure leaves an
incomplete preparation; do not choose replacements based on model output.

Independently compare all 54 fixed input rows with one another and compare
each input and input-plus-answer extension with every *other case's* input
and extensions, in both complete-row-prefix directions. An input is naturally
a prefix of its own answer extensions and that intended relation is exempt;
the eight extensions of one case must be distinct. Compare all inputs and
extensions in both directions against the latest reviewed public inventory
including grouped continuations, and all relevant pending public bundles.
Also compare against the 128-row original natural public
support (SHA-256 `5638b5ccc1d6585184fcd359e825ac3c2f6a1e377e24afde78bc5a9a2fc845d6`)
and 256-row natural successor public pool (SHA-256
`6303eb0713d64b020eea3fe491a686c9020ea424754443372aef73749bc87bc4`).
Ask the natural custodian for the scope of the comparison and any later
protected-pool change; coverage of the old split is conditional on its
recorded preparation, and no private membership or rows should be inspected.
Ask the injection custodian for the fixed L3-final site and any protected-row
status. Publish the cases and an independently reviewed planned reservation;
refresh latest inventory and both custody observations at launch. A failed,
unavailable or uncertain check stops the attempt before forwarding. Do not
adapt cases to a private comparison or retry an uncertain forward. These
checks cannot prove the absence of unknown historical exposures.

## Intervention and ordering

At batch 0, last input position of output element 0 of
`transformer.h.3.attn`, deliver `+u, −u, +v1, −v1, +v2, −v2` with common
nominal input-space L2 q=2.0, the same development-derived dose as the prior
study. q=2 is below that study's earlier donor-derived dose range, but does
not establish local linearity or equal functional strength. Starting from
the original f32 source each time, evaluate each component as binary64
`source + signed_q * direction`, then round directly to IEEE f32
nearest-even using bit-defined conversion. Record before/after bits, source
norm, actual L2, changed elements and signed projected displacement. Require
both actual L2 and the signed projection magnitude within 1e-5 relative of
2 and the correct projection sign; any nonfinite value, zero displacement or
failed check makes the one-shot attempt incomplete.

Canonical case order is frame, condition (`none`, `attractor_singular`,
`attractor_plural`), morphology (regular then irregular), lemma (A then B),
number (singular then plural), then the one fragment for that frame/condition.
For execution, let `n=3*frame_index+condition_index`. Alternate the first
morphology using `(frame_index+condition_index)%2`, regular first on zero.
Within each morphology execute the four target forms in the following order
indexed by n:

| n | Order |
|---:|---|
| 0 | A_s, A_p, B_s, B_p |
| 1 | A_p, A_s, B_p, B_s |
| 2 | B_s, B_p, A_s, A_p |
| 3 | B_p, B_s, A_p, A_s |
| 4 | A_s, B_s, A_p, B_p |
| 5 | B_p, A_p, B_s, A_s |

Execute the fragment last within each frame/condition. For every case use
clean, no-op at the site, six direction arms, clean-after. Target cells use
the plus-first base `u+,u−,v1+,v1−,v2+,v2−` when the indexed sign table
`[+,−,+,−,−,+]` at n is plus for regular and minus for irregular; otherwise
use the sign-reversed base. The fragment uses plus-first iff
`(frame_index+condition_index)%2==0`. Rotate the chosen six-arm base left by
`2*condition_index`. All four targets within one morphology cell use the
same arm order. Each arm appears in each slot nine times across 54 cases;
each morphology has three plus-first and three minus-first cells. The
six-order table gives each morphology three singular-first and three
plural-first cells; each target form occurs in every within-cell position
once or twice. Record actual order and verify these counts with a no-model
schedule test. Ordering is deterministic, not randomized.

Record the full f32 last-position vocabulary logits for every action and
whole/selected hook before/after witnesses. Require clean, no-op and
clean-after last-logit equality and no-op capture equality; check all
full-sequence logits finite before marking each forward complete, but note
that unsaved positions cannot be independently replayed. The plan is exactly
486 forwards. At a 12-token input bound, the successful logical tensor
payload is at most `486*50257*4 + 378*(2*12*768*4 + 2*768*4)` =
127,891,224 bytes before deduplication. Bound a failed full-sequence return,
journal, receipts and physical output separately. Retain one original attempt
and avoid creating repeated archive copies; recheck output/cache-volume free
space immediately before launch. Require at least 2 GiB free for output and
512 MiB for temporary model staging, jointly accounted if on one volume.
Use the reviewed gated CPU supervisor or equivalent; the direct-child receipt
does not authenticate execution or prove descendant drain. A failure or
interruption retains incomplete evidence and is never silently retried.

## Measurements, eligibility and predictions

For `m=z(is)−z(are)` define `O_a(x)=(m(x,+2a)−m(x,−2a))/2` and
`E_a(x)=(m(x,+2a)+m(x,−2a))/2−m(x,clean)` for each direction a. For each
of 12 frame/context/morphology cells, write the four O values `A_s,A_p,B_s,B_p`.
Compute and retain all four signed cross-lemma differences, plus

* `D_opp = (abs(A_s−B_p)+abs(B_s−A_p))/2`;
* `D_same = (abs(A_s−B_s)+abs(A_p−B_p))/2`;
* `Delta = D_opp − D_same`;
* `N = ((A_s−A_p)+(B_s−B_p))/2` (number-form contrast);
* `L = ((A_s−B_s)+(A_p−B_p))/2` (lemma contrast);
* `I = (A_s−A_p)−(B_s−B_p)` (number-by-lemma interaction).

Perform these computations for u, v1 and v2 with the same signed dose.
Use the complete fixed denominator of 12 cells; all averages use ordered
`math.fsum/n`. Do not normalize by `D_same` or any near-zero value. Report
all 12 individual cells, signs and zeros, frame/context/morphology summaries,
the four signed distances, N/L/I, O/E and realized doses. The primary
specificity comparison uses the mean of the 12 individual absolute control
`Delta` values, not the absolute value of a control mean.

Before interpreting the response criteria, count all 48 target clean
`is−are` margins: singular must be strictly positive, plural strictly
negative; a tie is incorrect. Baseline adequacy requires at least 36/48
correct overall, 18/24 within each morphology and each frame, 12/16 within
each condition, 9/12 within each lemma, and at least 8/12 cells with all
four forms correct. Retain and report all cases regardless. If any baseline
criterion fails, label the comparison baseline-limited without filtering or
replacing a case.

Conditional on baseline adequacy, the preregistered exploratory
opposite-versus-same response criterion passes only if (i) mean `Delta_u`
is at least 0.05 logit and strictly exceeds each mean individual
`abs(Delta_v1)` and `abs(Delta_v2)`, (ii) each morphology's six-cell mean
`Delta_u` is at least 0.02 and exceeds both corresponding mean absolute
control values, (iii) at least 8/12 `Delta_u` cells and at least 4/6 in each
morphology are strictly positive, and (iv) both frame means are strictly
positive. The floors are descriptive choices motivated by earlier exposed
data, not validated practical thresholds or statistical tests. If the fixed
criterion fails with adequate baselines, report no support under this rule,
including negative and zero cells; this is not evidence of no number feature.

The counterexample is explicit. If O has a common term plus additive lemma
offset ±d/2 and number term ±b, then `D_same=abs(d)` and
`D_opp=max(abs(d),2*abs(b))`. Thus `Delta=0` whenever lexical difference
dominates, even for nonzero b. Conversely, surface inflection or lexical
gain can produce positive Delta and N without abstract grammatical number.
The signed N/L/I and per-case outcomes are essential, not an inferential
separation of these alternatives. A common additive u-induced output shift
gives both distances zero, yet is compatible with real encoding under a
linear downstream readout.

Also report `was−were`, `the−a` and `in−on` with the same factorial
contrasts, all six subjectless O/E values, and full-vocabulary top-20, target
ranks, stable `KL(clean||arm)` and top-one changes, with ties by ascending
token ID. Equal input L2 does not imply equal functional output; report each
control's output changes. The fragments are descriptive controls and have no
role in baseline selection or in the primary denominator. Publish a result
whether positive, negative, baseline-limited or incomplete, together with
bounded raw records and an independent saved-byte replay after the observed
owner exit.
