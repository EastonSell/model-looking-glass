# Sampled mixed-batch validation protocol v1

This protocol compares each request's serial logits with its slice of one
mixed-owner forward. A complete pass means **agreement on the specified sample
within the frozen tolerances**. It does not establish universal separability,
privacy, model identity, correct checkpoint loading, safe device memory use,
or a performance improvement. This validator is not wired into the broker.

## Freeze before looking at results

Create a trusted `BatchProfile` and derive a JSON binding with
`binding_from_profile(profile, model_fingerprint=..., checkpoint_sha256=...,
adapter_sha256=..., backend=...)`. The digests must be lowercase SHA-256
strings; the trusted service, not this helper, must verify them against the
actual loaded model, checkpoint and adapter. The binding contains every
microbatch profile and execution-contract field, including row sizes, payload
limits, seed, and exact binary64 tolerances. It also records the backend and
runtime identity. The runtime identity must cover the actual physical device,
driver, software versions, precision, numerical and determinism settings, and
other execution properties relevant to the model. Matching metadata alone is
not verification of those facts.

Use a fixed, prespecified selection procedure and seeds. Prepare one or more
ordered groups of at least two requests, from at least two owners and with at
least two distinct complete request inputs. A request is
`{'case_id': ..., 'owner': ..., 'inputs': {'input_ids': tensor, ...}}`; a
group is `{'group_id': ..., 'cases': [...]}`. Tensors must match the profile's
CPU input schema and request shape. Groups must fit its owner, request, token,
logical payload and approved-row limits. At most 64 groups are supported.

Compute `input_row_digests(binding, discovery_inputs)` for each complete
discovery request and combine those hashes into a nonempty list. Then call
`freeze_protocol(binding, groups, discovery_row_hashes=...)` and retain the
entire returned protocol **and its `protocol_sha256` separately before running
held-out comparisons**. The protocol records ordered case identities, complete
request hashes and per-row hashes. Hashes sort input fields by name, so a schema
order change does not evade the split check. Even for a multirow request, no
complete held-out input row may exactly match a discovery row.
Repeating the same held-out request or row in different groups is allowed to
test different co-batches. Group and case counts therefore do not imply that
all observations are independent; analyze distinct row hashes separately.

These checks rely on the caller supplying the complete discovery set and
holding inputs, mappings, binding and protocol stable during copying. A hash
does not prove when a protocol was frozen or that discovery is complete. It
also does not detect semantic overlap, near duplicates, or equal token IDs
paired with different masks, positions, other fields, dtypes or shapes. Hashes
of small or guessable inputs are not confidential commitments.

## Execute and retain evidence

Call `run_validation(protocol, actual_binding, groups, execute,
expected_protocol_sha256=frozen_hash, cancel_event=...)`. The callback has the
signature `execute(inputs, *, request_id, seed)` and must return a completed,
detached CPU logits tensor. It must enforce the bound model/runtime/policy,
authenticate the result's request ID, keep model state fixed, respect a
measured whole-batch memory lease, synchronize all relevant CUDA work and CPU
offload before returning, and propagate errors and cancellation. The callback
must not infer batch separability from eval mode or a metadata flag. In the
current resident worker, a wrapper can submit a `ForwardRequest`, check the
returned request ID, and return its `.logits`.

For each group the validator runs every request serially, then submits the
concatenated mixed group once. It snapshots input data, verifies it against the
frozen manifest, and snapshots serial outputs before the mixed call. Shape,
dtype, CPU placement, autograd and finiteness are checked. The callback and
model remain trusted: a matching request ID or nonce does not prove that the
claimed model produced the returned data. Do not run this protocol alongside
an isolated latency benchmark.

Each element must satisfy `abs(mixed - serial) <= atol + rtol * abs(serial)`.
The threshold uses exact binary64 values, with exact rational arithmetic at
ambiguous or overflowing boundaries. Float16, bfloat16 and float32 outputs are
exactly representable in binary64. Maximum absolute and relative errors are
reported in binary64; an overflowing absolute error is `null` with its flag,
and a nonfinite relative maximum is `null` with `zero_reference` or `overflow`
as the reason. A zero serial value with a nonzero difference has infinite
relative error. Comparisons use bounded chunks of Python scalar arithmetic on
the calling thread. This can be slower than tensor-wide arithmetic; it is a
validation protocol, not a throughput measurement.

The calling CPU thread must preserve float32 and float64 denormals. The
validator checks both denormal inputs and underflowed outputs without changing
the caller's floating-point mode. If the mode is unsupported at admission or
changes during execution, `NumericModeError` is raised and no evidence is
published; the caller must record the attempted run as incomplete. The check
does not attest the numeric mode of the model's other threads, a GPU, or an
external executor. The trusted runtime identity and executor must account for
those modes. Caller-side comparisons and protocol serialization must run under
the supported mode, including when saving or loading evidence.

`status` is `passed` only for a complete set of matching cases; any observed
mismatch produces `failed`, even if another case later fails to execute.
Otherwise a stopped or invalid run is `inconclusive`. All planned cases remain
in the report with their individual status. The report records serial and mixed
call counts and `tested_rows`, the batch row counts with completed comparisons.
An error stops further execution; the callback and broker own cancellation,
device synchronization, OOM recovery and quarantine. A cancellation observed
through the last pre-publication event check prevents a passing result; a
later cancellation does not revoke a completed report.
`phase_error` covers exceptions from the callback or from local copying,
output checking and comparison during the named serial or mixed phase; its
exception type is recorded without claiming which component caused the error.

`save_evidence(path, report, protocol=protocol,
expected_protocol_sha256=frozen_hash)` creates a new JSON file and returns its
evidence digest; it never overwrites an existing file. Retain that digest in a
separate trusted location. `load_evidence(path, protocol=protocol,
expected_protocol_sha256=frozen_hash, evidence_sha256=retained_digest)` checks
the frozen manifest, report digest, ordered cases, coverage, call counts,
statuses and basic error-metric consistency. If writing fails, the caller must
discard any incomplete file and record the run failure.

Before using a passing report as one input to a service admission decision,
call `require_evidence(report, protocol=protocol, actual_binding=actual_binding,
expected_protocol_sha256=frozen_hash, evidence_sha256=retained_digest,
rows=combined_row_count)`. It rejects failed or incomplete runs, changed
bindings and row counts that were not actually tested. For example, a passing
two-row group does not cover four rows even if the profile lists both sizes.
The service must also check independently established coverage and safety
requirements, including multiple representative held-out cases and memory
profiles. One matching sample cannot prove that arbitrary unseen combinations
are safe. Checksums and structural checks cannot authenticate execution or
detect a plausible fabricated report; trusted storage and execution are
required.

## Current evidence

CPU tests exercise a deterministic separable model and a small randomly
initialized GPT-2 through the real resident worker, and detect a deterministic
batch-mean model as a negative control. They also cover frozen-profile mutation,
split overlap, cancellation, malformed outputs, saved evidence, numerical edge
cases and untested batch sizes. No GPU or pretrained model has been certified
by these tests, and no shared GPU throughput has been measured by this
component.
