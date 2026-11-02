# Prospective masked-antonym preference study, v1

This is an outcome-informed, separate proposal prompted by the stopped
[color-preference screen](../results/distilbert-color-preference-v2-20261029/README.md).
The color screen remains baseline-ineligible: none of its three candidates met
the fixed three-row rule, so no color edit or locality test was run. Its saved
outputs motivate a more explicitly specified relation; they are not confirmation
data for this proposal. Publish and independently review this protocol before
constructing its tokenizer-only case record. Publish and independently review
the exact case record and owner before any new pretrained forward.

After the first tokenizer-only draft (SHA-256
`3938d487e98068b0d515edbb16ee34d6d8c590f248ad28c81e8cad60dbda5e52`), a
named-public comparison at main `88666f1f0d4ad9c2f543983dea5efc6b22921b72`
(report SHA-256 `81ae408e504afcbd2e8c5b472b2148c1cfd5aced0cf330473261dad94d39ab19`)
found that `A violin is a musical [MASK].` was already a planned row in the
earlier color study, referenced in two public snapshots. This pre-forward
revision replaces only that unrelated locality with the trumpet sentence
below and retains `{instrument}`. No antonym model forward occurred before
the correction; publish and review this revision, rebuild the cases, and repeat
the full named-public comparison before the screen.

The question is whether a final-block parameter change can reverse one fixed
masked-token antonym preference on public rephrasings, and how its locality
compares with one deliberately global, efficacy-matched decoder-bias control.
It is not a generated answer, proof of novelty in pretraining, a capability
claim, or a comparison against a representative set of editors. The fixed
held-outs are same-pair paraphrases, not independently sampled facts. The
global bias is deliberately broad; lower disturbance relative to it would not
establish general editing superiority or causal knowledge localization.

## Model, inputs and chronology

Use `distilbert/distilbert-base-uncased` revision
`12040accade4e8a0f71eabdb258fecc2e7e948be`, verified offline, in eager
evaluation-mode CPU float32 with a deterministic one-thread runtime. Save all
30,522 mask-vocabulary logits for every completed forward. Build and freeze all
token IDs, attention masks, mask indices and full-sentence fill checks before
the screen; each focal, alternative, wrong-target and accepted answer must be a
whole, non-special single token in its stated context. If a fixed text, sense,
accepted set or tokenization is invalid, revise and independently review the
public protocol before any forward rather than substituting after a result.

The fixed candidate order and operational answers are:

| Order | Dimension | Cue / counterfactual target | Factual opposite | Other accepted factual fills | Object | Same-object accepted fill |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | size | large | small | little | box | container |
| 2 | length | long | short | none | rope | cord |
| 3 | temperature | hot | cold | none | drink | beverage |
| 4 | brightness | bright | dark | dim | room | space |

The preferred factual answer, not any alternative, is used by the edit and
screen rules. Alternatives are reported for semantic context, not used to
rescue a failed preferred-token gate. These are fixed operational answer sets,
not exhaustive inventories of every sensible completion. The fixed wrong
target is `red` for all candidates; it is an adjective in an unrelated
dimension rather than an antonym. The desired counterfactual token is the cue
word itself, so an effect may reflect copying or lexical preference rather than
reversing an abstract relation.

Substitute the candidate table values in these exact lower-case templates. All
four fixed object words take `a` where an indefinite article is used; the
held-out `this {object}` phrase has no indefinite article. Positions are
determined by the tokenizer rather than assumed equal across rows.

| Role | Exact template |
| --- | --- |
| discovery | `For a {object}, the opposite of {cue} in {dimension} is [MASK].` |
| calibration 1 | `In {dimension}, the opposite of a {cue} {object} is a [MASK] {object}.` |
| calibration 2 | `A {cue} {object} is opposite in {dimension} to a [MASK] {object}.` |
| held-out 1 | `For this {object}, the opposite of {cue} in {dimension} is [MASK].` |
| held-out 2 | `In terms of {dimension}, a {cue} {object} is opposite to a [MASK] {object}.` |
| held-out 3 | `A {cue} {object} and a [MASK] {object} are opposites in {dimension}.` |
| held-out 4 | `The {dimension} of a {cue} {object} is opposite to that of a [MASK] {object}.` |
| same-object locality 1 | `A {object} is a kind of [MASK].` |
| same-object locality 2 | `A {object} is a type of [MASK].` |
| reverse locality 1 | `In {dimension}, the opposite of a {factual} {object} is a [MASK] {object}.` |
| reverse locality 2 | `A {factual} {object} is opposite in {dimension} to a [MASK] {object}.` |

The two same-object localities use that candidate's accepted fill above and are
two phrasings of one relation. Both reverse localities accept the cue token:
the target word is correct there, so a global output bias can help rather than
only harm. They are not independent facts. The fixed alternate-use locality for
each selected candidate is:

| Cue | Exact alternate-use input | Accepted set |
| --- | --- | --- |
| large | `A [MASK] number of visitors arrived.` | `{large}` |
| long | `The concert lasted a [MASK] time.` | `{long}` |
| hot | `The article became a [MASK] topic of discussion.` | `{hot}` |
| bright | `The [MASK] student solved the difficult puzzle.` | `{bright}` |

These are different contexts or referents, not a claim that each word is in an
independent lexical sense: quantity and spatial size for `large`, and time and
spatial length for `long`, can share a magnitude meaning. `hot` and `bright`
also have figurative uses here. The cue is accepted on every alternate-use row,
so the global cue bias can improve its score, as on the reverse localities.

The other seven localities, common to every candidate, are:

| Stratum | Exact input | Accepted set |
| --- | --- | --- |
| wrong-target correct | `The background color of a standard stop sign is [MASK].` | `{red}` |
| unrelated antonym | `The opposite of day is [MASK].` | `{night}` |
| unrelated antonym | `The opposite of the direction left is [MASK].` | `{right}` |
| unrelated antonym | `The opposite of open is [MASK].` | `{closed, shut}` |
| unrelated relation | `A dolphin is a kind of [MASK].` | `{mammal}` |
| unrelated relation | `A trumpet is a musical [MASK].` | `{instrument}` |
| unrelated relation | `A week has [MASK] days.` | `{seven}` |

There are 12 candidate-specific inputs per candidate and seven common
localities: 55 planned inputs in all. The selected candidate contributes four
held-out target rows and five candidate-specific localities; with seven common
localities, evaluation has 12 locality rows. Compare every fixed input in both
prefix directions against the named current public DistilBERT and cross-model
case/exposure snapshots, using complete token rows and rows with one final SEP
removed; coordinate current relevant custodians and publish the exact scoped
comparison before forwarding. This is no certification of unknown exposure,
private data or pretraining history.

## Screen and change

Run all 12 original discovery/calibration inputs, in candidate order and then
discovery, calibration 1, calibration 2 order. A candidate is eligible only if
its preferred factual token is the strict unique full-vocabulary top on the
discovery row and the exact saved-f32 factual-minus-cue margin is strictly
greater than decimal 0.10 on all three rows. Compare to exact rational 1/10;
rounded display values do not decide. Ties fail the unique-top test and an
exact threshold tie fails the margin test. Report the full-vocabulary winner,
all factual/cue/wrong-target probabilities and ranks, accepted alternatives,
and combined factual-plus-cue mass, even where a candidate fails. Evaluate all
12 rows and choose the first eligible candidate. Publish all screen logits and
selection before constructing an edit. If none qualifies, publish
`baseline-ineligible`, retain every row and stop; do not draw replacement
cases, change thresholds or use later evaluation outcomes to select a pair.

For a selected candidate use the same fixed construction and fail-closed
evidence contracts in the [color-preference v2 protocol](DISTILBERT_COLOR_PREFERENCE_V2.md)
at commit `d94594f69426aa5e41a7cf90acd1e98be42f9087`, document SHA-256
`41ffb6f85d0017a3644b4ee5cc7328ffa1c90c3fec36faeb438fe96625dfc34d`, with
`purple` replaced by the fixed candidate cue and `blue` by `red`. The numerical
and action modules referenced there are pinned at `707e5806ef9a77137a8b03b80a5b7ce716579ca0`.
The earlier color owner is not itself an implementation of this protocol.

On discovery, differentiate cue-minus-factual with respect to the final block
feed-forward `lin2.weight` at the mask. Check the full gradient and captured
site factors, finite nonzero values and rank-one residual at most `1e-5`.
From the restored model, measure all six fixed normalized gradient updates
with intended Frobenius norm fractions `(0.0005, 0.001, 0.002, 0.004, 0.008,
0.016)` of the original weight norm, casting and adding in float32 once. Select
the first whose discovery cue-minus-factual margin exceeds `0.10`; if none
reverses, retain the final fixed dose for descriptive evaluation and mark the
reversal gate failed. No hold-out or locality result may tune this choice.

At that realized norm use the fixed opposite-gradient, one fixed seeded
Gaussian rank-one direction and the `red`-minus-factual wrong-target gradient;
use seed `20261121`, output factor then input factor, float64 draws with no
intervening draws. Require the earlier `0.999–1.001` realized-norm tolerance,
no redrawing, and retained construction/rounding/state evidence. A safely
established unavailable auxiliary direction is recorded and skipped only
after the original state is verified; any comparison requiring it is not
evaluable. A failed/nonfinite model action, invalid installed weight, failed
primary construction or uncertain restoration stops incomplete.

Construct the two-coordinate global decoder-bias control using only the
discovery clean and primary logits and the earlier shifted-partition solve.
Preserve the remaining vocabulary partition, aim to match primary
cue-minus-factual margin and cue full-vocabulary log-probability, and replay
the actual float32 update within `0.001` nats for both. Keep the earlier
primary-gain floors `0.10` margin and `0.025` target log-probability. If the
solve or replay is unavailable, record the cause; do not invent a match. If
the discovery solve works but calibration later fails, keep that frozen bias
for descriptive held-out/locality evaluation without retuning. Publish all
discovery/calibration logits, all grid points, selected updates, control
construction, bias feasibility and calibration decisions before any held-out
action. Every arm starts from the same verified restored baseline model.

The fixed development order is clean discovery, primary gradient, six grid
points, wrong-target gradient, then available opposite/random/wrong-target
arms on discovery and the two calibrations, primary on the two calibrations,
and an available bias on discovery and both calibrations. Development starts
at most 23 forwards. Evaluation visits the four held-outs, two same-object,
two reverse, one alternate-use and seven common localities, in their table
order. For each row run original, primary, opposite, random, wrong-target and
bias, omitting only previously documented unavailable controls. It starts at
most 96 forwards. Screen plus development plus evaluation is at most 131
starts, under the shared 256-start ceiling. The 128 MiB retained original
serialized-evidence ceiling includes every logit, gradient/factor, update,
metadata, failure, supervisor and observation record. An uncertain or partial
stage is incomplete and must not be silently retried or promoted.

## Matching, locality and reporting

The strict unique factual top is required only on discovery. Calibration and
held-out original eligibility is pairwise; a matched pair of edited arms can
have a third token as their shared top. Report the actual clean and edited tops
and accepted status, and do not call a pairwise reversal a predicted-answer
switch. An original target-row top tie is retained and does not itself prevent
matching; the two edited tops must each be unique and identical. Locality top
ties follow the stricter locality rule below.

On each calibration and held-out row, primary and bias must both gain at least
`0.10` in cue-minus-factual margin and `0.025` in cue log-probability versus
the same saved unedited row. For each gain separately their absolute difference
must be at most `0.10 * max(abs(primary), abs(bias))`; each full-vocabulary top
must be unique and the two top IDs must agree. A tie in either edited top or a
missing/mismatched arm makes that row's match unavailable; do not replace it or
retune. Report each
factual/cue/wrong-target logit, rank, probability and log-probability, top,
pair mass, baseline errors and both arms' full-vocabulary TV. Even a strong
two-token preference can coexist with little full-vocabulary pair mass, and
matched endpoint gains do not match the rest of the distribution.

The primary-versus-bias locality comparison is evaluable only if the discovery
reversal and both calibration matches pass, each unedited held-out has factual
minus cue margin strictly greater than `0.10`, every held-out match passes, and
both arms reverse to cue-minus-factual greater than `0.10` on at least two of
the same four held-outs. Held-out observations determine evaluability but never
tune the edit. Preserve every failed/baseline-wrong/tied row and explicitly
report an unavailable comparison rather than a selective success.

For each of the 12 locality rows and every available arm, save full logits and
report TV, `KL(clean || arm)`, unique top/ties, factual/cue/red probabilities
and ranks, accepted-set NLL and whether a baseline-accepted unique top remains
accepted. Use stable log-sum-exp for each nonempty, deduplicated, in-range,
non-special accepted set. A relative pass requires primary mean locality KL
strictly below `0.8` times bias mean, at least `1e-6` lower, and no more changed
unique tops. Also require primary mean KL `<= 0.02`, maximum KL `<= 0.10`,
mean accepted-set NLL deterioration `<= 0.05`, maximum deterioration `<= 0.20`,
and retention of every initially accepted unique top. Only original, primary
and bias are required for this comparison; a missing one, a top tie in any of
those arms, or zero initially accepted localities makes it not evaluable. An
unavailable auxiliary arm does not by itself block this specific comparison.

Follow the referenced protocol's binary64 KL roundoff policy. Independently
recompute every decision-driving quantity from the saved full-f32 rows: screen
top and 0.10 margins with exact saved values and rational arithmetic, and
decision-driving log-probability, bias-match, TV, KL and accepted-set NLL
quantities using a separately reviewed Decimal reader with at least 100 digits,
stable tiny-tail handling and adaptive precision. The reader must decode saved
binary32 exactly and use a bounded single-row or row-pair working set rather
than decoding the entire journal at once. Before any forward, test that the
exact-f32 row `(0, -300)` retains the roughly `5.1482e-131` log-normalizer
correction rather than silently rounding it to zero. The reader must report
its precision and any bounded tail omission; if the side of a relevant
threshold cannot be established, classify the decision as indeterminate.
Finite precision is a cross-check rather than a general error certificate.
Retain its per-row values and every gate decision. A failed/unavailable reader,
nonfinite calculation or disagreement between its gate decision and the owner
is **numerically indeterminate**, not a pass or failure; retain all observations
and do not silently rerun or exclude a row. The Decimal replay is a numerical
cross-check, not a proof of model execution. KL measures any
distribution change, including improvements; accepted-answer changes and
baseline correctness are separate observations. Report all per-row outcomes
and the same-object, reverse, alternate-use, wrong-target and unrelated
strata. Locality masked fills do not establish preservation of other token
states or downstream tasks. Separate cooperative process reports, saved-byte
consistency and independent replay from execution attestation.
