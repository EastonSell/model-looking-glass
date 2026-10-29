# Comparing observed activation traces

`looking_glass.localization.compare_traces` compares the **original**
`looking_glass.probes.TraceEvent` streams. It matches a named tap and its
per-tap occurrence ordinal explicitly; it never assumes that the *n*th value in
one stream corresponds to the *n*th value in the other. The current function
does not accept the newer transformer event schema: use an explicit,
coordinate-preserving adapter before comparing those events. In particular,
node-and-occurrence labels alone do not encode layer, token, output selection,
forward index, or generation/cache context.

```python
from looking_glass.localization import compare_traces

report = compare_traces(
    reference_session.events, changed_session.events,
    atol=0.0, rtol=0.0,
    reference_complete=True, candidate_complete=True,
)
print(report["status"], report["first_observed_interval"])
```

Only set each completion flag if that stream finished successfully and includes
every emitted event under its declared capture policy. The default is `False`.
This attestation covers the declared taps, not the whole model. An omitted
module, hidden branch, unrecorded token, truncated execution, or discarded
artifact is not negative evidence. The caller is responsible for checking model
and code revisions, selected inputs and tokens, instrumentation policy, runtime,
and other experimental conditions; matching event labels do not establish any
of these.

The JSON-compatible report contains the chosen tolerances, individual matched
comparisons, both streams' counts and completion flags, run-identity consistency,
sequence and occurrence continuity, missing and duplicate coordinates, full
coordinate-order agreement, unavailable values, and incompatible tensor metadata.
Missing values, nonfinite or empty tensors, unsupported layouts or devices,
mismatched shape/dtype/device, and inconsistent saved-value metadata cannot
establish equality. Empty streams cannot establish equality either. Only dense
strided CPU or CUDA tensors in float16, bfloat16, float32, or float64 are
supported; sparse, meta, float8, named tensors, lazy negative/conjugate views,
and other values are reported as unavailable. Values are compared elementwise
against the reference magnitude:
`abs(candidate - reference) <= atol + rtol * abs(reference)`. This inequality
uses exact real arithmetic over the recorded finite IEEE tensor values and the
binary64 tolerance values. It can therefore differ at a boundary from the
previously implemented rounded binary64 computation. Signed zeros compare equal.
Both tolerances must be finite, nonnegative built-in Python `int` or `float`
values; integers are converted to binary64 first. Other numeric types must be
explicitly converted by the caller before use. Restricting the conversion avoids
silently flushing a subnormal in an implicit conversion. Zero tolerance is the
default. Subnormal bits are preserved during comparison even if the calling
CPU thread flushes denormals; this does not establish the numerical mode used
when the model produced the retained values.

The `tolerance` report includes `rtol_binary64_hex` and `atol_binary64_hex`,
each exactly 16 hexadecimal digits encoding the authoritative IEEE-754
binary64 bits. These are bit patterns, not Python's `float.hex()` strings.
The corresponding numeric `rtol` or `atol` field is `null` for subnormal
tolerances in either CPU mode: Python JSON formatting can otherwise print a
retained subnormal as zero while denormal flushing is enabled. Other finite
tolerances keep their numeric fields, and the hex fields preserve signed zero.

- `difference_observed`: a difference is present at a matched observation and
  all earlier supplied observations form an aligned, available, equal prefix.
  Both completion flags must be explicitly true. `first_observed_interval`
  reports the immediately preceding equal tap, if any, and the unequal tap.
- `no_difference_observed`: both streams were explicitly declared complete,
  contain at least one comparable value, and every event aligned and compared
  equal within tolerance. This is a result about those captured values only.
- `inconclusive`: neither conclusion is supported. A later matched difference may
  still appear in `first_compared_difference` (in reference-stream order), but an unmatched, ambiguous,
  reordered or unavailable earlier event makes the first-observed interval
  unknown.

An occurrence is an ordinal in the observed stream. If a repeated module call
was skipped or changed semantically, equal ordinals need not refer to the same
computation. Even for matching events, the final equal observation and next
unequal observation are sample boundaries, not a proof that the intervention's
cause lies between them: effects can appear and disappear between taps or travel
through unobserved paths. `causal_origin` is always reported as `"unknown"`.
Establishing or rejecting a causal explanation requires separately controlled
interventions and held-out evaluation; activation differences alone do not
explain model reasoning.

This comparator does not save tensor values, impose a trace-memory budget, or
schedule GPU work. Finite-value checks and element comparisons can synchronize a
GPU. Exact comparisons use temporary integer views and masks, and nonzero
tolerances stage complete saved snapshots on CPU before processing bounded
chunks of Python integers. A noncontiguous value can require a full-size
contiguous copy, including a device-side staging copy for CUDA. The caller must
budget source-sized device and host copies and use supervised GPU ownership;
the integer chunks do not bound this snapshot staging. Nonzero-tolerance
comparison may be substantially slower than vectorized float comparison.
Measure or budget these costs separately, and prefer bounded CPU-staged traces
when appropriate. No new empirical localization result is claimed here.
