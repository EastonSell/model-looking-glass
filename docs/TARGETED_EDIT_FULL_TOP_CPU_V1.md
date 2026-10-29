# Full-vocabulary targeted edit: prospective CPU protocol

**Protocol only.** This is a new, small exploratory study of the pinned
DistilGPT2 checkpoint. Publish and independently review this protocol before
assembling the new case pool or forwarding it through a model. The earlier
screened study remains negative and is not reclassified. Its projected edit
reversed a London–Paris margin on one of eight targets, but neither city was
the full-vocabulary top token on any of those targets in any condition.
Consequently that result did not show a changed top prediction or generated
answer. Its lower projected locality KL also came with weaker and heterogeneous
target effects. A later logit-space interpolation is a post-hoc illustration,
not an executed, efficacy-matched model edit.

The behavioral endpoint here is deliberately narrower: at a prompt that asks
for a single city name, determine whether the specified counterfactual city is
the **unique full-vocabulary next-token top**. The factual and counterfactual
names must each be one token at the frozen continuation boundary. This is a
first-token prediction, not a complete greedy or sampled answer, an assertion
about stored knowledge, or evidence that the model will emit the city in a
longer generation. Paris, Tokyo, Rome and London are the true national capitals
of France, Japan, Italy and the United Kingdom respectively; the alternative
names below deliberately specify false counterfactuals.

## Fixed pool and baseline-only selection

After this protocol is published, construct exactly 36 target prompts and 12
locality prompts. The four fact/alternative pairs, in selection order, are
France: Paris to London; Japan: Tokyo to Paris; Italy: Rome to London; and
United Kingdom: London to Paris. For **each** fact, freeze in this order one
discovery prompt, two efficacy-calibration prompts and six evaluation
paraphrases. Each must explicitly request that country's national capital as a
single city name, in an answer-slot context whose immediate continuation is the
name, not a prefix such as `the`. Use distinct natural answer-slot wordings;
do not substitute seat-of-government or city-location relations. Discovery,
efficacy-calibration and evaluation are separate prompt roles. The evaluation
phrases concern the **same selected fact**, so they are held-out phrasings,
not held-out facts. They are public and selected using their original model
outputs, not blind or representative samples. The prior screened target and
locality cases are not candidates and must not be relabeled held out.

Freeze the 12 locality prompts in their own order: four facts involving named
cities or their non-capital relations, four other geographic or national facts,
and four unrelated facts. None may ask for the national capital of any of the
four candidate countries, because the selected fact is not yet known. At least
one prompt must have London and one Paris as a correct answer
to a different relation. For each locality row freeze an expected one-token
continuation and an explicit nonempty set of accepted single-token spelling or
capitalization variants, including the expected token. Check each relation and
variant independently for meaning before any forward; do not alter accepted
variants after observing logits. If an accepted spelling is not a single-token
append at that prompt boundary, it cannot be included as a single-token
variant. These small locality probes are not a representative capability test.

Publish exact text, ordered case and role IDs, country, answer strings, token
IDs, each expected/accepted variant and the protocol identity. Verify against
the pinned tokenizer that every factual and counterfactual target continuation
and each locality variant appends exactly one specified token without changing
the prompt's encoding. The prompt token rows, rather than the continuation
encodings, are the model inputs to reserve. Check the 48 inputs against each
other and the then-current reviewed public exposure inventory in both
complete-row-prefix directions. Independently check case semantics. Before
each forward stage, refresh the relevant public inventory and protected-split
custody for the pinned model/revision and applicable input-token namespace:
where protected rows exist, require a private comparison of
these exact input rows without exposing those rows; a scoped report that none
exist is distinct from a private pass. A collision, unknown/unavailable custody
or unexplained relevant new public exposure stops the stage. Reserve all 48 exact prompt
rows before their first forward. Later checks may exempt only the identity-bound
own reservation or this attempt's authorized exposures, and must still reject
an independently named duplicate or prefix relation. These checks do not
establish disjointness from unknown exposures or pretraining.

Before the first baseline screen forward, publish and independently review the
executable source revision and the manifest binding all 48 cases, the screening
rule, fixed grids and thresholds, random seeds, checkpoint and runtime. Later
selection and development publications follow the order below. Git ancestry
and recorded observations are cooperative evidence of the reported ordering,
not authenticated clocks or proof against undisclosed earlier work.

With the unedited model, forward **all 36 target prompts**, once in the frozen
case order, and save every finite full-vocabulary float32 logit row. A target
row is baseline-eligible exactly when its factual answer token has a strictly
greater logit than every other token; ties, including a tie for first, are
ineligible. Report the factual and alternative token logits and log
probabilities, unique top ID and decoded token, top gap, all 36 eligibility
decisions, per-role counts and the complete input/row hashes. Select the first
fact in the fixed fact order with an eligible discovery row, **both** eligible
efficacy-calibration rows and at least four eligible evaluation rows. For that
fact choose the first four eligible evaluation rows in frozen order. Selection
uses only the original-model top condition and fixed order: it does not rank by
margin or examine edited outputs. Save and publish the selection and all
screen rows before constructing an edit. If no fact qualifies, the study is
**ineligible**; retain all 36 baseline results, select nothing, do not redraw,
relax the rule or perform edited evaluation under this protocol.

For every tied maximum, record a null unique-top ID, the number of tied IDs and
the smallest tied ID as a deterministic representative. A tied maximum is
never a unique factual or counterfactual top. Apply the same convention to
screening, development and evaluation; the representative is not a prediction
of a unique winner.

## Model, construction and fixed doses

Use the pinned DistilGPT2 checkpoint, final-block MLP `c_proj` weight, CPU
eager float32 execution, deterministic settings, one intra-op and inter-op
thread, the previous study's denormal-preservation and baseline-state checks,
and offline pinned tokenizer/checkpoint assets. The frozen manifest must bind
the exact executable source, tokenizer, checkpoint, cases and runtime. Use a
supervised isolated child and a trusted quiescent checkout/cache; report the
source, process and clock records as cooperative attestations, not as proof
against hostile mutation or as authenticated timing. Preserve all partial
records and give uncertain cleanup or provenance precedence over numerical
results. The existing public four projection-calibration prompts and their 18
site-input positions from the earlier projected study are reused solely to
define the projection; they are not the new efficacy-calibration prompts.

On the selected discovery prompt, form the gradient of counterfactual-minus-
factual final-position logits with respect to the site weight. Save the full
gradient, input and output factors, and the grad-enabled discovery site inputs,
outputs and full logits. Require byte-identical no-grad site inputs, outputs
and logits and agreement with the selected saved baseline row before
construction. Capture the original site inputs at every position of the four
projection-calibration prompts. With CPU float64 SVD and the previous study's
relative singular-value cutoff 1e-10, project the rank-one input factor off
their span. Require finite nonzero factors, retained fraction at least 1e-6,
relative projection-null residual at most 1e-8 and full-gradient rank-one
relative residual at most 1e-5. Record singular values, rank, both residuals,
factor and gradient hashes, and the gradient dots for intended and realized
updates. Matching forward values does not establish equality of downstream
Jacobians; the saved gradient is a derivative of the grad-enabled execution.

Normalize the projected and unprojected rank-one directions in float64 to each
planned fraction of the original weight Frobenius norm, cast the intended
update once to float32, and check the actual post-addition float32 weight.
Require each nonzero realized update norm within [0.999, 1.001] of its planned
norm. Record planned direction, intended update, realized weight and realized
delta hashes and norms; do not publish raw checkpoint weights. The pinned
checkpoint and factors permit independent reconstruction. Retain all
projection-calibration input/output rows, 18 per-token observed shifts and
algebraic `X @ realized_delta` effects. Equality on those construction inputs
is an implementation check, not an independent locality result. Under every
edit, verify the discovery and projection-calibration site input bytes remain
equal to their original captures.

Starting with the original model for each condition, measure the projected
discovery output at every fixed ascending planned norm fraction:
`0.0005, 0.001, 0.002, 0.004, 0.008, 0.01`. Select the first fraction at
which the counterfactual city is the unique full-vocabulary top. Save the
entire grid; do not stop recording after a success. If no fraction achieves
that top, select the last fraction `0.01` as a predeclared descriptive fallback,
record **development top not achieved**, and retain its evaluation without
calling the behavioral gate successful. This dose grid is informed by the
earlier public edit studies, not a blind or optimized search.

At the selected projected dose, form the equal-norm unprojected edit, a
projected random rank-one control, the exact opposite projected update, and a
rank-matched nuisance-span control. Use independent fixed CPU generators with
seeds 20261210 and 20261211 for the random and nuisance controls respectively,
using the same draw order, QR/rank checks and no-redraw rules as the preceding
screened study. Require their realized norms within [0.999, 1.001] of the
projected realized norm. The random and nuisance controls are single
descriptive draws, not samples from a control distribution.

## Development-only efficacy match

Before any edited evaluation or locality forward, measure the projected edit
and every unprojected dose in the fixed grid
`0.00025, 0.0005, 0.00075, 0.001, 0.0015, 0.002, 0.003, 0.004, 0.006, 0.008, 0.01, 0.012, 0.015`
on **both** separate efficacy-calibration prompts. Save every full-vocabulary
logit row, actual update identities and restoration record. Reuse the exact
screening rows as their original baselines. For each calibration case and each
edit compute (a) the change in full-vocabulary log probability of the
counterfactual token and (b) the change in counterfactual-minus-factual
logit margin. Compute log probabilities with shifted float64 log-softmax from
the saved float32 logits. Form margins by promoting each saved float32 logit
to float64 before subtraction. Also classify its unique top as factual,
counterfactual, other, or tied.

An unprojected dose is an eligible match only if, on **each** of the two
calibration rows, both projected and unprojected gains for **both** measures
are strictly greater than 0.05, their top classifications agree, and each
absolute gain discrepancy is at most `max(0.005, 0.10 * abs(projected_gain))`.
For every grid point record the four discrepancies divided by those four
tolerances. Of eligible points choose the one with the smallest maximum ratio;
break an exact tie by the smaller planned dose. If none qualifies, record
**match unavailable**, selected matched dose `null`, all grid results and the
nearest point separately. The nearest is the smallest maximum normalized
discrepancy over the complete valid grid, breaking an exact tie by lower dose;
report its gain-floor and top-category failures separately. Do not interpolate,
add a dose or use evaluation
outputs to retune. A failed forward, nonfinite measurement, uncertain update
or restoration is **invalid/incomplete**, not an unavailable match. If the
selected matched dose equals the projected dose, require the matching and
equal-norm unprojected intended and realized hashes to agree, reuse its saved
condition and disclose that both comparisons share it. Publish the complete
development grid, calibration rows, controls, selected dose and independently
reviewed hash-bound candidate before any edited evaluation/locality forward.

## Evaluation and fixed outcomes

Use the four selected evaluation prompts and all 12 fixed locality prompts.
Reuse their exact screening rows for the four original evaluation baselines;
obtain the 12 original locality rows from the pinned original model. Evaluate
projected, equal-norm unprojected, projected-random, opposite and nuisance
controls on every row; add the matched unprojected condition if available.
Verify each intended and realized update before its condition, restore the
full model/weight/buffer state between conditions and save all finite raw
full-vocabulary rows, ordered row hashes, actual top IDs and decoded tokens.
Save and report for every row and condition the factual/counterfactual logits,
log probabilities, margin and changes when applicable, locality expected-token
NLL, accepted-set NLL, accepted-token status, and `KL(original || condition)`.
Accepted-set NLL is minus the logarithm of the sum of full-vocabulary
probabilities for the unique preregistered accepted token IDs; compute it
stably from the same float64 log-softmax. The canonical expected-token NLL is
also reported separately. Report every
locality row, including those whose baseline top is not an accepted answer;
report baseline-correct and baseline-other cases separately without dropping
either from aggregate KL and NLL. Exact top-ID changes and accepted-variant
changes are separate measures; an agreed spelling-only variant is not by
itself evidence of semantic loss.

The projected behavioral gate requires at least two of the four evaluation
rows to have the counterfactual token as their **unique full-vocabulary top**,
and a strictly larger count than the single projected-random control. Report
all four outcomes and each condition's count. Define the locality safeguard
as all locality rows whose baseline unique top is in their preregistered
accepted set remaining uniquely topped by a token in that set, mean locality
KL at most 0.02, maximum locality KL at most 0.10, and maximum individual
accepted-set NLL deterioration at most 0.20 nats. If no locality baseline
top is accepted, the accepted-answer safeguard is **not evaluable**, not a
vacuous pass. Individual harms, baseline errors, and all groups remain visible.

The **equal-norm** comparison is `positive` only when the projected
development top was achieved, the behavioral and locality safeguards pass, and
the projected mean locality KL is at most 0.9 times the equal-norm
unprojected mean and is at least 1e-6 smaller in absolute KL. Otherwise a
complete eligible comparison is `negative`; list each failed gate. If no
locality baseline top is accepted, both comparisons are `ineligible: no
baseline-accepted locality`, because that safeguard is not evaluable. The KL
comparison is a fixed-dose observation even if target effects differ.

For the **effect-matched** comparison, first assess the four evaluation rows
without excluding any. On each row, require both projected and matched
unprojected gains for both development matching measures to exceed 0.05, and
their absolute difference for each measure to be at most
`0.05 * max(abs(projected_gain), abs(unprojected_gain))`; their top
classifications must also agree. If any row fails, the matched-locality
comparison is `ineligible: evaluated strength imbalance`; report all effects
and locality values descriptively and do not select favorable cases or tune a
dose. If the development match is unavailable, this comparison is
`ineligible: no development match`. Otherwise it is `positive` only if the
same projected development, behavioral and locality gates pass and its mean
locality KL is at most 0.9 times, and at least 1e-6 below, the matched
unprojected mean; otherwise it is `negative`. These tolerances are exploratory
and cannot establish exact efficacy equivalence or rule out smaller effects as
an explanation for lower KL.

In every case, an incomplete or uncertain model forward, nonfinite value,
failed restoration, unverified source/custody or uncertain child cleanup
overrides all numerical labels as **invalid/incomplete**. A valid but
underfilled baseline screen is **ineligible**. If no locality baseline is
accepted, that ineligibility takes precedence for both comparisons; otherwise
an unavailable development match takes precedence over any evaluation balance
check for the matched comparison. Retain every applicable ineligibility reason,
including overlapping reasons, even when one is designated primary. A completed no-top development
fallback remains a negative behavioral observation even if descriptive
evaluation follows. Publish all failed gates, the entire selection and screen,
all grid measurements, raw full-vocabulary evidence and a separately reviewed
saved-output audit. Do not infer statistical significance, broad capability
preservation, full generated answers, held-out-fact generalization, or unique
causal localization from this single selected fact and the single controls.
