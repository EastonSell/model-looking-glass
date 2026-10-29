# Prospective masked-color preference study, v2

This is a new, outcome-informed proposal. It must be published and independently
reviewed before its tokenizer-only case record is constructed, and the exact
cases and executable must be published and reviewed before any new pretrained
forward. The earlier [full-vocabulary-top screen](../results/distilbert-color-edit-top-v1-20261029/screen/README.md)
is closed as baseline-ineligible and is not revised or discarded.

In that exposed screen, the usual color beat the fixed alternative `purple`
in 8 of 9 saved rows while it was the full-vocabulary top in 0 of 9. The two
calibration frames also allowed non-color completions. Those observed results
motivated the present explicit-color phrasing and **pairwise** endpoint; they
are not confirmation data for this protocol. This experiment asks whether a
targeted change can reverse a specified masked-token preference while limiting
measured collateral change on the fixed localities. It does not claim a
generated answer, full-vocabulary top-token edit, broad capability preservation
or a general fact about color knowledge. The global-bias comparator is one
deliberately broad active control, not a representative distribution of editors.

The earlier protocol is fixed at source `707e5806ef9a77137a8b03b80a5b7ce716579ca0`,
with `docs/DISTILBERT_COLOR_EDIT_TOP_V1.md` SHA-256
`d414ab04137b842cd28d232b31e2ca5b62833ade06ef0ede4a6ce17da081f937`.
At that source, the module SHA-256 pins are:

| Module | SHA-256 |
| --- | --- |
| `looking_glass/distilbert_color_edit_metrics_v1.py` | `f5bf19141593152a78ce91c1c53ed04aaf725b7686a9f64337f48b2794ac324b` |
| `looking_glass/distilbert_color_edit_stages_v1.py` | `4b244ff0bf8457f7c93704d1b43e6f6cd07108f8febaf650f919320f27a47475` |
| `looking_glass/distilbert_color_edit_action_v1.py` | `20853b72737d4498e8516be804b49fb770f14dd84bd73289bab1ffd2180b7064` |

These are references for the stated numerical and evidence contracts; their
v1 screen and selection code is not an executable implementation of this v2.

## Fixed cases and chronology

Use the same pinned `distilbert/distilbert-base-uncased` revision
`12040accade4e8a0f71eabdb258fecc2e7e948be` and the verified offline, eager CPU
float32 runtime in the earlier protocol. Freeze the new tokenizer-only rows,
full attention masks, positions and accepted wordpiece IDs before a forward.
All fills below must be verified as whole, non-special single vocabulary
tokens at the exact mask boundary. The usual-color labels describe
conventional ripe examples, not every variety or every lighting condition.

The fixed candidate order is `(banana, yellow)`, `(strawberry, red)`, then
`(lemon, yellow)`. For each object substitute the lower-case singular name in
the following exact strings. `purple` is the desired alternative and `blue`
is the wrong-target control; they are fixed before the screen.

| Role | Exact template |
| --- | --- |
| discovery | `A ripe {object} has a [MASK] color.` |
| calibration 1 | `The color of a ripe {object} is usually [MASK].` |
| calibration 2 | `A ripe {object} appears [MASK] in color.` |
| held-out 1 | `The usual color of a ripe {object} is [MASK].` |
| held-out 2 | `The ripe {object} is colored [MASK].` |
| held-out 3 | `A typical ripe {object} has [MASK] coloring.` |
| held-out 4 | `If the {object} is ripe, its color is [MASK].` |
| same-object locality 1 | `A {object} is classified as a [MASK].` |
| same-object locality 2 | `A ripe {object} is a type of [MASK].` |

Both same-object localities have the fixed singleton accepted set `{fruit}`;
they are two phrasings of one relation, not two independent properties. The
held-out rows are public near-paraphrases for the same object, not independent
facts or evidence of generalization to new objects. Related-object locality is
covered separately below. Exact wording may be rejected by independent
semantic or tokenizer review; any correction requires a new public protocol
revision **before** the screen, not a post-screen replacement.

The other ten fixed localities, scored for every available arm, are:

| Stratum | Exact input | Fixed accepted set |
| --- | --- | --- |
| other color | `The usual color of a ripe orange is [MASK].` | `{orange}` |
| other color | `The color of fresh snow is [MASK].` | `{white}` |
| other color | `The color of a ripe lime is [MASK].` | `{green}` |
| other color | `A ripe eggplant is [MASK] in color.` | `{purple}` |
| other color | `In daylight, a clear sky is [MASK] in color.` | `{blue}` |
| unrelated | `A calf is a young [MASK].` | `{cow}` |
| unrelated | `A foal is a young [MASK].` | `{horse}` |
| unrelated | `A clarinet is a musical [MASK].` | `{instrument}` |
| unrelated | `A hawk is a kind of [MASK].` | `{bird}` |
| unrelated | `A tulip is a kind of [MASK].` | `{flower}` |

The `purple`-correct and `blue`-correct localities deliberately test changes
to the desired and wrong-target words in other contexts. These fixed answer
sets are operational labels, not exhaustive semantic answer lists. The 37
public inputs comprise 27 candidate-specific rows and 10 common localities.
Only the selected candidate's two same-object localities and the 10 common
localities enter an evaluation. Check all 37 exact and one-terminal-SEP-removed
input rows in both prefix directions against named same-model public exposure
snapshots and the applicable custodians before forwarding. Such checks do not
certify unknown histories, private splits or pretraining novelty.

## Baseline screen and edit construction

For each of the nine discovery/calibration rows, save the complete unedited
30,522-token mask logits. Define factual preference as
`logit(factual) - logit(purple)` on that same row. A candidate is eligible
only when all three of its fixed screen rows have finite factual preference
strictly greater than `0.10` nats. This is a predeclared pairwise screen; a
full-vocabulary top other than the factual color does not itself disqualify a
row. Ties at the threshold fail. Evaluate all nine rows, choose the first
eligible candidate in the fixed order, and publish all nine logits, summaries
and the selected identity before constructing an edit. If none is eligible,
publish **baseline-ineligible** and stop; do not add or replace cases.

On the selected discovery row, obtain the gradient of
`logit(purple) - logit(factual)` with respect to the final block's feed-forward
`lin2.weight` at the mask. Use the same saved-site and full-gradient witnesses,
rank-one reconstruction, full-state restoration, fixed CPU runtime and
one-shot evidence rules as the earlier owner. Construct the six ordered
positive-gradient weight updates with intended Frobenius norms
`(0.0005, 0.001, 0.002, 0.004, 0.008, 0.016) * ||W||_F`, cast and add in
float32, and save the realized norms and full logits for every grid point.
Select the first whose discovery `logit(purple) - logit(factual)` is strictly
greater than `0.10`; if none achieves it, retain the last point for descriptive
evaluation and label the reversal gate failed. No grid value is selected using
held-out or locality outcomes.

Construct the fixed opposite-gradient, one rank-one random direction and
blue-minus-factual wrong-target updates at the selected realized norm, using
the earlier owner's `0.999–1.001` norm tolerance. For the random direction,
initialize a CPU generator with seed `20261121`, draw the output factor and
then the input factor from the float64 standard normal distribution, with no
intervening draws, and take their outer product. Do not redraw or choose a
direction using its observed effect. If an auxiliary direction is nonfinite,
zero, or demonstrably cannot meet the tolerance after float32 rounding, record
that control as **unavailable** before installing it and proceed only after
verifying the original model state is restored. Valid primary and other
available arms can then continue; any comparison requiring the absent control
is not evaluable. Save enough construction evidence and the reason to audit
this decision. An actual failed or nonfinite model action, failure of the
primary construction, invalid installed weights or uncertain restoration
remains **invalid/incomplete** and stops the study. Retain available controls'
full logits, realized updates and restoration witnesses.

Construct the two-coordinate global decoder-bias comparator from discovery
only, using the earlier protocol's shifted-partition solve to match the
primary discovery purple/factual margin and purple full-vocabulary
log-probability. Save original bias bits, intended and realized updates, and
replay; discovery must agree within `0.001` nats on both quantities. If the
primary discovery gains do not reach `0.10` margin and `0.025` purple
log-probability, or the solve/replay is unavailable, retain that reason and
do not claim a matched comparison. If a valid discovery solve fails the
calibration match below, keep that fixed bias arm for descriptive evaluation;
do not retune it.

Before any held-out action, publish all discovery/calibration logits, the
entire grid, selected update, active controls, bias feasibility and calibration
decision, with the exact source and direct-process observations. A partial or
uncertain stage is incomplete and is never silently retried.

The ordered forward schedule is fixed. Screen all nine candidate rows in the
table's candidate order, with discovery then calibration 1 then calibration 2.
Development then uses the selected discovery row for clean-site, primary
gradient, all six grid points and wrong-target gradient. Next run each
available opposite, random and wrong-target arm, in that order, on discovery
and the two calibration rows; run the primary arm on both calibration rows.
Attempt the bias on discovery only when its fixed gain floors and solve allow;
if its discovery replay passes, run that same bias on both calibration rows.
Thus development starts at most 23 forwards, fewer only for predeclared
unavailable auxiliary or bias arms. Do not replace omitted arms.

Evaluation order is the four held-out rows, the selected object's two
same-object localities, the five other-color localities and the five unrelated
localities, each in the order shown above. For each row use the restored
original model, then primary, opposite, random, wrong-target and bias, omitting
only arms recorded unavailable under the rules above. A valid discovery bias
is evaluated even if its calibration match failed. This is at most 96
evaluation forwards and 128 total starts across all stages; the larger
256-start resource ceiling still applies. Every original and edited action
uses the fixed restored baseline model, and both edited arms are compared to
the same saved original row for that input.

## Held-out match, locality and reporting

For each calibration row and each of the four held-out rows, compare primary
and bias on the same input. Both must have a purple-minus-factual margin gain
of at least `0.10` and a purple full-vocabulary log-probability gain of at
least `0.025` versus their own unedited row. For each gain independently,
`abs(primary - bias) <= 0.10 * max(abs(primary), abs(bias))`. Their
full-vocabulary tops must both be unique and have the same token ID; ties or
failure of any requirement make the match unavailable. Match failure does
not remove the row or authorize retuning. Report factual and purple logits,
full-vocabulary probabilities and ranks, margin, top/ties, and their actual
changes for every clean and available edited row. A margin gain alone can be
caused by lowering the factual token, so it is never the sole efficacy measure.
A large conditional preference between the two colors can coexist with
negligible total full-vocabulary probability on both; report their combined
probability mass for every clean and edited target row. Matching the two gains
and top ID does not match the rest of the vocabulary distribution; report
primary-versus-bias target-row TV descriptively without changing the gate.

The efficacy-matched locality comparison is evaluable only if the discovery
reversal gate and both calibration matches pass, all four held-out unedited
factual preferences exceed `0.10`, every held-out primary/bias match passes,
and both arms reverse the pair (`purple - factual > 0.10`) on at least two
of the **same** four held-out rows. Preserve baseline-wrong rows and ties;
they make the comparison unavailable rather than disappearing. Held-out
reversal without these gates is descriptive and not an efficacy-matched result.
The held-out rows are not used to tune the edit, but their outcomes determine
whether the matching gate is evaluable; the comparison is conditional on that
observed matchability.

On all 12 localities retain unedited and available-arm full logits, full
distribution TV and `KL(clean || arm)`, unique top/ties, factual and purple
ranks/probabilities, accepted-set NLL and whether a baseline-accepted unique
top remains accepted. Use stable log-sum-exp for each nonempty, distinct,
in-vocabulary, non-special accepted set. The predeclared relative prediction
passes only if primary mean locality KL is strictly below `0.8` times bias
mean and at least `1e-6` lower, and primary has no more changed unique tops.
Absolute safeguards additionally require primary mean KL `<= 0.02`, maximum
KL `<= 0.10`, mean accepted-set NLL deterioration `<= 0.05`, maximum
deterioration `<= 0.20`, and retention of every initially accepted unique top.
For this primary-versus-bias decision the required arms are original, primary
and bias: a missing one, a top tie in any one, or zero initially accepted
localities makes the decision **not evaluable**. An unavailable opposite,
random or wrong-target arm does not itself block this particular comparison;
any separate comparison that needs that arm remains not evaluable. Report
every individual row, including baseline errors, and same-object, other-color
and unrelated strata separately; never infer preservation beyond them.
Locality KL measures distribution change, including possible improvements;
accepted-answer changes and baseline correctness are separate observations.
The shared weight can also change other token positions. These masked-fill
localities do not establish preservation of those states or downstream tasks.

Use the earlier protocol's raw binary64 KL and small-negative clamp policy,
and independently audit saved actual rows at high precision before relying on
near-zero or threshold decisions. The record must distinguish saved bytes,
cooperative process observations and independent replay from execution
attestation. The same shared `256` started-forward and `128 MiB` original
serialized-evidence limits apply across all stages, including failure and
supervisor/observer records. No model/GPU work or hidden-pool access is needed
to construct or review this proposal.
