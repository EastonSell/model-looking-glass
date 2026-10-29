# V2 sparse trace pair validation

`looking_glass.causal_trace_v2.compare_sparse_pair` validates one pair against the
four sparse observation points prescribed in [CAUSAL_CONFIRMATION_V2.md](CAUSAL_CONFIRMATION_V2.md).
It is an input to a future v2 runner, not a completed v2 study. It does not
generate cases, choose sites, receive injected-operation labels, run a model,
or enforce a discovery/held-out freeze.

The caller must provide two successfully completed, externally paired traces
with distinct opaque run IDs and set the corresponding completion and pairing
flags. Those attestations must be backed by the experiment's model, data,
intervention, runtime, and capture records; matching labels cannot establish
that provenance. Use `TraceSession` with sparse placement, `max_events=4`,
`max_tensor_bytes=128`, and the protocol's CPU float32 `[1,8]` settings.
The helper checks the retained traces contain the four expected block outputs in
order with zero invocation indices, valid per-stream IDs, exact metadata,
finite CPU float32 values, and 128 total logical payload bytes. It cannot
verify that a cap was applied before capture or account for temporary memory.
Payloads must be ordinary unnamed strided tensors without unresolved negative
or conjugate view bits; unsupported formats fail rather than being coerced.

The result contains the first unequal boundary, the previous equal boundary,
and four L2 norms in block order. Finite float32 payloads are decoded from
their integer bit patterns into exact Python binary64 values, so CPU subnormal
flush mode cannot silently turn a saved difference into zero. Signed zeros
compare equal. Norms use the v2-prescribed Python binary64
`math.sqrt(math.fsum(...))` calculation, not a PyTorch reduction. Results have
three meanings:

- `difference_observed`: an eligible block output is the first unequal value.
- `upstream_difference`: block 0 differs; an active v2 cell is invalid, while a
  no-injection comparison must report this as a false detection.
- `no_difference_observed`: all four snapshots match exactly. For an active
  injection this is an observation miss; for a clean-repeat control it is the
  expected result. It is not a decision that an entire cell must abstain.

Missing, malformed, nonfinite, incompletely captured, or unverified pairs raise
`SparseTraceInvalid`, rather than becoming observation misses. The custodian
must apply the protocol's sanitized failure reporting and preserve private
diagnostics. The runner remains responsible for all 32 discovery pairs, the
per-cell aggregation, the no-injection controls, the private target integrity
checks, and any separate held-out phase. Include this helper in the source hash
manifest. A first observed boundary does not prove
the exact location or causal origin of a change.

Comparison is insensitive to the calling CPU thread's denormal flush mode;
model forwards are not necessarily insensitive. The protocol and runner must
separately fix and verify the forward-execution denormal policy before a v2 run.
