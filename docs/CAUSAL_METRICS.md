# Held-out intervention metrics, contract v1

This document specifies utilities, not an experimental result. The prediction
metrics measure whether **preregistered scalar predictions** match held-out
intervention effects. A separate descriptive helper summarizes measured paired
interventions. Neither establishes a mechanism or shows that an activation
trace explains a model's reasoning.

Before looking at held-out outcomes, save the hypothesis, intervention,
measurement and sign convention, independent case IDs, predicted effects,
optional prediction intervals and their nominal coverage, baselines, zero
tolerance, and evaluation and resampling settings. Save raw cases and all
failure records along with the summary. The utility checks numerical input but
cannot enforce preregistration or establish case independence.

## Input and estimand

`EffectCase(case_id, predicted, observed, interval_low=None, interval_high=None)`
contains one scalar prediction and one observed effect per case. The prediction
and observation must use the same task metric, direction, units, and contrast;
for example, intervention score minus its matched no-intervention score. Pass a
frozen prediction, not one selected after seeing that case's outcome. Run
the evaluator separately for each declared candidate or control, using the same
held-out cases. A patch's residual error relative to a desired target can be
assessed by supplying its measured effect as `predicted` and the desired target
as `observed`; label this as effectiveness against a target, not prospective
prediction accuracy.

Cases receive equal weight. Each ID must be unique, but unique IDs do **not**
prove that cases are independent or exchangeable. This v1 interface accepts only
one scalar row per case; it does not implement clustered or hierarchical
resampling of several token positions, intervention sites, prompts, or model
seeds per independent unit. Do not label correlated rows as separate cases.
Aggregating rows changes the estimand; prespecify and justify any aggregation,
or use a suitable cluster-aware analysis. A case bootstrap under one checkpoint
does not measure uncertainty across checkpoints or model seeds.

## Descriptive candidate-versus-control summary

`summarize_interventions` accepts one `InterventionCase` per case. The three
readouts must be matched scalar measurements in the same units: a common
reference and the candidate and control interventions. Supply the same metric,
sign convention, case alignment and fixed eligibility rule used by the study.
For example:

```python
from looking_glass.causal_metrics import InterventionCase, summarize_interventions

summary = summarize_interventions([
    InterventionCase('case-a', 'supported', 'eligible', 2.0, 5.0, 3.0),
    InterventionCase('case-b', 'unsupported', 'eligible', 1.0, 4.0, 2.0),
    InterventionCase('case-c', 'supported', 'eligible', 0.0, None, 1.0),
])
print(summary['paired_cases'])  # 1 included, 2 excluded, with reasons
print(summary['cases'][0]['candidate_effect'])  # +3.0, relative to reference
print(summary['cases'][0]['candidate_minus_control'])  # +2.0
```

`baseline_status` must be `supported`, `unsupported` or `unverified`;
`eligibility` must be `eligible`, `ineligible` or `unverified`. These are
caller-supplied assessments, not conclusions the helper derives from the
readouts; `eligibility` concerns the complete candidate/control comparison.
Use `None` for an unavailable readout. Each input case is retained in
the original order, with its raw values, statuses, available signed effects
(`candidate - reference`, `control - reference`) and direct
`candidate - control` contrast. A contrast can be available even when the
reference is missing. A positive contrast means a larger scalar readout for the
candidate; it does not by itself mean a better or more causal intervention.

All three finite readouts, a supported baseline and an eligible status are
required for a case to enter the paired summary. Each excluded case receives
one reason, in this order: unsupported or unverified baseline; ineligible or
unverified intervention; missing readout. `paired_cases` reports included and
excluded denominators and these mutually exclusive reason counts. The separate
baseline, eligibility and missing-readout counts retain the overlapping facts.
All three means and the above/equal/below counts use the **same** included
cases; when none qualify, means are `None` and all three counts are zero. The
means sum the rounded finite binary64 per-case values exactly before the final
rounding. The directly computed paired contrast can differ slightly from
subtracting the displayed mean effects due to floating-point rounding.

An empty case list, duplicate or empty IDs, unsupported status strings, booleans, nonfinite
readouts and differences that overflow raise `ValueError`, including on excluded
cases. The helper performs no model call, resampling, normalization by a
baseline gap or significance test. It cannot establish that a case was held
out, that its values are aligned, that observations are independent, or that a
control is matched; retain the study's raw evidence and exclusions separately.

## Point metrics

For predictions `p_i` and observations `y_i`, `point` contains:

- `mae`: mean of `abs(p_i - y_i)`; `case_errors` also contains each absolute
  error and the corresponding zero-null absolute error.
- `rmse`: square root of the mean of `(p_i - y_i)^2`.
- `sign_accuracy`: fraction of **directional cases** whose predicted sign
  matches the observed sign. A value with `abs(value) <= zero_tolerance` is
  treated as zero. Cases with `abs(y_i) <= zero_tolerance` have no defined
  direction and are excluded from this metric only. A zero prediction on a
  directional case counts as incorrect. `directional_cases` reports both the
  included and excluded counts; the score is `null` when none are included.
- `prediction_interval_coverage`: fraction of all cases whose observed effect
  lies between its prespecified lower and upper prediction bounds, endpoints
  included. `prediction_interval_coverage_minus_nominal` subtracts the declared
  `prediction_interval_level`. Bounds and nominal level must be supplied for
  all cases, or neither for any case. Coverage and the difference are `null`
  when bounds are absent. These are observed frequencies, not proof of
  calibration, especially for a small or nonrepresentative case set.

The zero-effect baseline predicts `0` for each case, and its `null_mae`,
`null_rmse`, and `null_sign_accuracy` use the same observations and direction
rule. The baseline's directional accuracy is always zero when at least one case
is directional, and is `null` otherwise. A zero predictor is a deliberately weak
directional baseline; report stronger, task-appropriate controls separately.
`*_improvement_over_null` is `null error - candidate error` for MAE and RMSE,
and `candidate accuracy - null accuracy` for sign accuracy. Positive values
therefore favor the candidate. Zero tolerance affects the direction metric, not
continuous errors or interval coverage. The error improvements are computed in
paired, numerically stabilized forms; they may differ slightly from subtracting
the already-rounded displayed aggregate scores. Calculations still use finite
floating-point values and cannot resolve effects below their representable
precision.

## Matched comparison with another predictor

`compare_effects(candidate_cases, baseline_cases, ...)` reports paired
candidate and baseline MAE, RMSE, and directional accuracy against the same
observed scalar target, as well as their paired improvements and per-case
absolute errors. Every ID and observation must match exactly between inputs;
missing, duplicated, or mismatched cases raise `ValueError`. Observations are
compared as supplied, before conversion to Python floats, so distinct integers
that round to the same binary64 value do not falsely match. Inputs are sorted
by ID and bootstrap resampling, when enabled, keeps the two predictions and
their common observation together. Positive `mae_improvement` and
`rmse_improvement` mean lower error for the candidate, and positive
`sign_accuracy_improvement` means higher directional accuracy. The same
directional exclusion and bootstrap-suppression rules described above apply.
Prediction bounds are not evaluated by this comparator and are rejected;
evaluate each predictor's coverage separately using `evaluate_effects`.

The common-target requirement defines the estimand. For example, random-site
responses can be compared with candidate-site responses as predictors of a
shared desired correction, or as effectiveness against that target, provided
the role is labeled. This is **not** a direct contrast between different
interventions' observed effects: that would require separate per-intervention
outcomes and its own declared estimand and analysis. This utility does not
divide by a clean–corrupted gap, substitute an epsilon, or omit zero-denominator
cases.

The comparator defaults to `bootstrap_samples=0`; a nonzero setting asserts the
independent, exchangeable case-sampling assumptions below. The older
`evaluate_effects` API retains its default of 2,000 resamples, so set it to
zero explicitly for a fixed suite. A seed by itself does not justify inference.
Paired loss differences are formed from exact rational representations of the
supplied finite floating-point values before averaging, avoiding cancellation
from subtracting individually rounded losses. The RMSE denominator and the
reported point errors still use finite floating-point arithmetic; numerical
precision remains limited by the supplied inputs and that denominator.

## Bootstrap and its limits

Use `bootstrap_samples=0` for descriptive analysis of a fixed suite when no
target population and case-sampling model are justified. In that mode a single
case is allowed, `bootstrap.method` is `none`, and every bootstrap interval is
`null`.

With an enabled bootstrap, at least two independently sampled, exchangeable
cases and at least two resamples are required; one resample carries no useful
bootstrap uncertainty. Two is only a computational minimum: prespecify a
sufficiently large number to stabilize the tail quantiles. The utility sorts cases by ID, then uses a
local seeded Python random generator to resample the **whole case** with
replacement, `n` times per replicate. Prediction, observation, bounds, and the
zero baseline stay paired. Every metric, including the nonlinear RMSE and
improvement over null, is recomputed for every resample. For each metric, the
two-sided percentile interval uses linear interpolation at quantile positions
`p * (B - 1)`. The bootstrap `confidence` level refers to uncertainty in a
summary metric; it is distinct from the nominal prediction-interval level.

`bootstrap.defined_samples` and `bootstrap.undefined_samples` count defined
and undefined replicate metrics. If any resample has no directional case, or
the original data contain fewer than two directional cases, the three
directional bootstrap intervals are suppressed (`null`); no undefined draws
are silently discarded. Metrics whose prediction bounds were absent likewise
have `null` intervals. A disabled bootstrap has zero defined and undefined
draws. Seeds, sample count, confidence level, and all interval outputs are
recorded in the result. Given the same case values, IDs, settings, Python RNG
semantics, and implementation, input order does not change the draws.

Percentile intervals depend on the prespecified case-sampling assumptions. They
can be unstable for small samples and can collapse to a point for all-success or
all-failure outcomes, even when substantial uncertainty remains. A non-null
interval is not a guarantee of frequentist coverage or of generalization to
new models, tasks, or populations. `bootstrap_samples=0` avoids suggesting an
inferential interpretation for a fixed convenience suite.

## Missing values and reproducible use

No silent case deletion or imputation is performed. Missing effects, partial or
mixed bound coverage, nonfinite values, invalid levels, empty or duplicate IDs,
and unrepresentable prediction differences raise `ValueError`. Values must be
finite real scalars; booleans and numeric strings are rejected. Values are
converted to Python floats before metric calculations; values not exactly
representable in binary64 are rejected rather than silently rounded. Supplying fewer than two cases to an enabled
bootstrap also raises `ValueError`. Define any
failure and missing-data analysis in the protocol, and retain the original
failure records; do not drop hard cases after observing results.

```python
from looking_glass.causal_metrics import EffectCase, evaluate_effects

cases = [
    EffectCase("held-out-001", predicted=0.3, observed=0.2,
               interval_low=0.1, interval_high=0.5),
    EffectCase("held-out-002", predicted=-0.2, observed=-0.1,
               interval_low=-0.4, interval_high=0.0),
]

# Descriptive metrics: bounds and nominal level were frozen before evaluation.
result = evaluate_effects(cases, zero_tolerance=0.01,
                          prediction_interval_level=0.90,
                          bootstrap_samples=0)

# Compare two predictors against exactly the same observed targets. Bounds are
# intentionally omitted; coverage is evaluated separately with evaluate_effects.
from looking_glass.causal_metrics import compare_effects
candidate = [EffectCase("held-out-001", 0.3, 0.2), EffectCase("held-out-002", -0.2, -0.1)]
baseline = [EffectCase("held-out-001", 0.0, 0.2), EffectCase("held-out-002", 0.0, -0.1)]
paired = compare_effects(candidate, baseline, zero_tolerance=0.01)

# Only when cases are independent exchangeable draws from a specified population:
# result = evaluate_effects(cases, zero_tolerance=0.01,
#                           prediction_interval_level=0.90,
#                           bootstrap_samples=2000, confidence=0.95, seed=17)
```

Persist the case-level inputs, protocol and result together. `case_errors` is a
useful diagnostic, but it is not a replacement for saving the preregistered
predictions, observed values, model/checkpoint identity, and experiment manifest.
