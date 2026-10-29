# Held-out intervention metrics, contract v1

This document specifies an evaluation utility, not an experimental result. Its
metrics measure whether **preregistered scalar predictions** match held-out
intervention effects. A good score does not, by itself, establish a mechanism or
show that an activation trace explains a model's reasoning.

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
finite real scalars; booleans and numeric strings are rejected. Supplying fewer
than two cases to an enabled bootstrap also raises `ValueError`. Define any
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

# Only when cases are independent exchangeable draws from a specified population:
# result = evaluate_effects(cases, zero_tolerance=0.01,
#                           prediction_interval_level=0.90,
#                           bootstrap_samples=2000, confidence=0.95, seed=17)
```

Persist the case-level inputs, protocol and result together. `case_errors` is a
useful diagnostic, but it is not a replacement for saving the preregistered
predictions, observed values, model/checkpoint identity, and experiment manifest.
