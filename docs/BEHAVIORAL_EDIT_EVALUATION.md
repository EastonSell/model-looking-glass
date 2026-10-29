# Behavioral edit observations

`looking_glass.behavioral_edit_evaluation` is a model-free reporting utility.
It compares supplied observations before and after an edit for target and
locality cases, optionally measures change in a supplied categorical
distribution, and reports separately supplied rollback evidence. It does not
execute a model, select inputs, prescribe thresholds, or decide whether an edit
is successful or causes semantic damage. For scalar *predictions of effects*,
use the separate [causal metrics](CAUSAL_METRICS.md) interface.

```python
from looking_glass.behavioral_edit_evaluation import (
    BehavioralCase, BehavioralObservation, CategoricalDistribution,
    RollbackEvidence, ScalarComparison, StateDigest, evaluate_behavioral_edit,
)

case = BehavioralCase(
    case_id="target-1", input_id="prompt-1", role="target",
    before=BehavioralObservation(value="Paris", token_ids=(7, 1),
                                 decoded_text="Paris", termination="eos"),
    after=BehavioralObservation(value="London", token_ids=(8, 1),
                                decoded_text="London", termination="eos"),
    references=("Paris",), targets=("London",),
    before_distribution=CategoricalDistribution(
        support=(0, 1, 2), probabilities=(.7, .2, .1), context="first-step"),
    after_distribution=CategoricalDistribution(
        support=(0, 1, 2), probabilities=(.2, .7, .1), context="first-step"),
    measures=(ScalarComparison("target_log_probability", -2.0, -1.0),),
)
report = evaluate_behavioral_edit(
    [case], rollback=RollbackEvidence(
        attempted=True,
        before=StateDigest("parameters-and-buffers-v1", "a" * 64),
        after=StateDigest("parameters-and-buffers-v1", "a" * 64),
    ),
)
```

All examples and digests above are artificial. Each case has a unique `case_id`,
an optional input identity and a `target` or `locality` role. Inputs and labels
are caller-defined; the utility does not establish that different IDs represent
independent cases or that a locality set measures a general capability. Keep
all rows and the original input/evidence records; summaries are descriptive and
only cover the supplied cases.

## Outcomes and scores

An observation's `value` is an exact string, integer, boolean, or tuple of nonnegative
token IDs. `references` and `targets` are optional nonempty tuples of accepted
values in that same representation. They have different meanings: references
are the caller's declared correctness labels, while targets describe desired
edited behavior and may be counterfactual. Both before and after matches are
reported. Matching uses exact values and types (`True` is not integer `1`); no normalization, semantic
equivalence, tie-breaking or truth checking is performed. A lost reference
match is a change against the supplied label, not a determination of semantic
damage. A changed answer can be a gain, loss, another wrong answer or unscored.

Optional `token_ids`, `decoded_text` and `termination` are retained verbatim.
Each is compared separately when both observed records supply it; a matching
`value` does not imply matching token IDs, text or termination.
When a sequence is the comparison value, its `token_ids`, if also supplied,
must agree exactly. An owner that studies generation should supply the complete
generated sequence, including EOS when emitted, and record how it terminated;
the utility does not verify tokenization, EOS, decoding or sequence completeness.
For a masked-token task, the value can instead be a token or class ID. These
representations describe different endpoints and should not be mixed.

`status="observed"` requires a comparison value. `missing`, `tied` and
`invalid` carry no comparison value and remain distinct, with an optional reason;
`None` is shorthand for a missing observation. Unavailable values produce
`null` matches and changes, not zero. Each summary includes paired denominators
for reference and target matches and paired **value** changes. Reference rates and
their difference use only rows with both observations and a reference; they are
`null` when no such rows exist. Missing rows and ties must still be inspected.

`ScalarComparison` holds a caller-named before and after number, such as a
complete-sequence margin or a target-label log probability. A finite paired
delta is `after - before`; missing, nonfinite or unrepresentable values are
flagged and never silently counted as zero. The utility cannot establish the
units, direction or comparability of caller-named scores. It does not aggregate
them or apply a cutoff.

## Categorical distributions

A `CategoricalDistribution` contains an ordered, nonempty tuple of unique
integer or string support labels, a matching tuple of probabilities, and a
nonempty caller-declared context. Both contexts and the complete ordered
supports must match to compare a pair. The context can distinguish, for example,
a fixed masked-token position from a first-step generation distribution. The
utility cannot establish that either support really is complete, that the
contexts refer to the same actual forward, or that a first-step distribution is
a distribution over complete generated answers. Preserve the original vectors
and their provenance for independent audit; the report contains a SHA-256 of
the ASCII-escaped compact JSON support, not the probability vectors themselves.

Values must be finite built-in Python integers or floats in `[0, 1]`, and the
sum must be within `1e-6` of one. This is a numeric input-validity tolerance,
not a behavioral threshold. For a valid pair the utility divides each vector by
its recorded sum and reports total variation, `0.5 * sum(abs(p - q))`, over all
supplied labels; it clips a possible positive roundoff beyond one. Missing,
nonfinite, out-of-range, unnormalized, or incomparable pairs have no TV value
and retain their status and available sums/reasons. Malformed shapes or duplicate
support labels raise `ValueError`. Arithmetic uses ordinary binary64; it is not
a bit-exact audit of saved logits, and may lose effects below runtime precision
or under denormal-flushing settings. No KL or full-logit reconstruction is
claimed. Summary means and maxima cover only valid compared pairs, and report
their denominator so missing comparisons are not hidden.

## Rollback evidence

`RollbackEvidence` records whether rollback was reported attempted and optional
before/after `StateDigest` values: a nonempty scope string and a lowercase
SHA-256. A match is labeled `reported_match` only when both scopes and digests
match; different digests, different scopes, incomplete evidence, an unknown
attempt and an unattempted rollback have distinct statuses. The function does
not hash a live model or verify that the scope includes all parameters, buffers,
hooks, optimizer state, caches or external resources. Digest equality is a
comparison of supplied records, not proof of rollback execution or cleanup.
Each case may also supply a restored behavioral observation. Value, token-ID,
decoded-text and termination matches with the baseline are reported separately
when both sides provide them, and separately from the state-digest comparison.

The caller must supply and retain quiescent input records during evaluation,
preserve failures and provenance, and independently verify any execution,
source identity, state measurement and cleanup claims. An invalid or incomplete
record does not authorize model reuse or imply a successful rollback.
