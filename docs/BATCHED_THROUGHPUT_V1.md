# Batched shared-model throughput: proposed protocol v1

Status: **development protocol and offline analyzer, not a GPU result or an
admission policy**. The current broker executes one forward at a time; the
grouped broker is still under review. This protocol must not be reported as
executed until the real broker emits the measurements below and the complete
owner/device chain has been independently checked. The earlier four-request
shared-load result used no batching and cannot serve as an exact baseline for a
different workload.

## Question and fixed workload

Compare an unbatched and a grouped service using the **same** two or more
closed-loop team clients and the same frozen jobs, one outstanding request per
client. Each job has a unique ID, team, complete input digest, common seed,
shape, input-position count, capture policy and model/runtime binding. Use
distinct inputs across teams; if grouped execution requires equal RNG seeds,
give all eligible jobs the same prespecified seed in both modes. Keep the
earlier per-prompt-seed quick run as historical evidence, not an identical
baseline. The client arrival times can differ between the windows because they
are closed-loop: this is a comparison under two offered-load policies, not a
replay of identical arrivals or a claim of isolated latency.

Freeze the source, checkpoint and tokenizer hashes, software and physical
device identity, float dtype, eager/eval/cache and deterministic settings,
input and output schemas, queue limits, exact batching policy and timeout,
CPU thread settings, repetitions, warmup, randomization seed, tolerances and
complete workload manifest before collection. Record the seed/order used to
alternate modes. Warm each actually used shape and mode outside its timed
window. Use one resident model and the same model state across both modes;
stop on mutation, missing drain or uncertain device state.

GPU work requires supervised physical-device ownership, a matching live lease,
and independently reviewed exact resource profiles for **each actual row
count**, including serial and whole grouped calls, input/output host storage
and any finite-check scratch space. Bind each launch to its own approved
profile; an unbatched profile or previous two-token smoke cannot authorize a
two-row group. A conservative capped bootstrap, if separately authorized and
reviewed, remains development-only and is not proof of a memory bound. Do not
run service load during isolated latency or tap-versus-gradient measurements.

## Event and result evidence

The instrumented worker, not the planner, must record the actual input shape
at each model entry, its bound profile, one forward-call ID and a unique group
ID. Record the ordered broker-scoped job IDs and row offsets in each group,
including members later cancelled. Report the actual batch-size distribution,
member/team composition and number of forward calls. A `batched` claim requires
an observed call with at least two rows from at least two teams with different
frozen input digests. A selected but never launched group does not count.

Use one documented high-resolution clock for a window. For each job retain
caller attempt and return, ready-to-dispatch and selection timestamps; for
each launched group retain actual launch and synchronized completion times.
Keep ready-to-selection and selection-to-launch/admission delay separate from
the synchronized forward interval. Record result-splitting/offload and output
validation timing where the runner does them. A cancelled member can still
consume GPU work; retain its status and the complete original membership.
Record all attempts, rejection, timeout, OOM, cancellation, unresolved work,
errors, cleanup and whether the owner drained. Never start a subsequent window
while work might still be active. On failed synchronization or uncertain
ownership, retain the lease and unsafe marker until the supervisor's recovery
protocol establishes safe handoff.

Record cancellation request and decision times and acceptance separately from
the final member outcome. These timestamps establish their observed order;
they do not themselves attest the producer's linearization point. An accepted
cancellation cannot deliver a successful result;
a failed launched call has failed member outcomes even when cancellation was
accepted, and the cancellation fields preserve both facts. Record broker
terminal state separately from client-observed delivery: a failed member can
have an unresolved client after disconnection. Terminal delivery for a launched
member follows the recorded call finish. A client can remain unresolved even
after separately verified owner drain; this invalidates a pair without implying
that device work is still active.

Start the denominator at common barrier release, after all clients are ready,
and end only after every planned attempt has terminal handling and output
offload. Output transfer and the same per-request validation/hash policy must
be inside both modes' timed boundaries. Numerical comparison may be performed
after the timed window using bounded, retained CPU outputs from the actual
measured requests. Record successful jobs and input positions per total wall
window, failure counts and rates, per-team counts and wait/response
distributions, and allocated/reserved allocator peaks plus device-wide
free/total endpoint snapshots. Do not call a device-wide endpoint difference a
device-wide peak; record whether external GPU activity was measured or unknown.
If memory telemetry is absent, say so and do not make a memory claim. For tiny
samples, report observation counts and maxima; nearest-rank p95 is the maximum
for fewer than 20 observations.

## Correctness controls

Before performance collection, use the independent
[`BATCH_VALIDATION.md`](BATCH_VALIDATION.md) protocol with a frozen binding,
disjoint discovery and held-out complete input rows, multiple independent
mixed-owner groups and a negative control. Keep evidence and its digest, and
only use row counts actually tested. This provides sampled agreement, not
universal batch separability, model authentication, or safe memory admission.

For every successful measured job, compare its retained batch-split output to
the same job's serial output. Check exact expected CPU dtype, shape and finite
values first, then the prespecified elementwise condition
`abs(grouped - serial) <= atol + rtol * abs(serial)` using serial as reference.
Record per-job output fingerprints, maximum absolute/relative error, the
elementwise pass/fail and tolerance. Exact hashes are fingerprints and need not
match across legal floating-point batch outputs. A missing, nonfinite,
detected misrouting, incomplete or out-of-tolerance result invalidates the paired claim;
save failure evidence instead of dropping that job. Output buffers and hashes
must be bound to their broker-scoped job and actual group split. Tensor data or
an independently replayable comparison record must be retained if later
readers are expected to recompute the numerical check. Row identities are
trusted producer evidence: numerical equality cannot prove routing when two
outputs are equal or indistinguishable within tolerance. A row-swap negative
control tests sensitivity on the selected cases, not all future traffic.

After collection, independently reconcile frozen manifests, raw attempts,
group membership, real call counts and row shapes, timestamps, profiles,
output evidence, summaries, memory and cleanup. A hash establishes byte
consistency, not that a model executed or that a self-reported event happened.
Report single-run descriptive results and variation across prespecified paired
repetitions without claiming fairness or GPU kernel overlap from concurrent
client submission.

## Offline event analyzer

`looking_glass.batched_throughput.freeze_protocol` snapshots a bounded JSON
manifest and returns its SHA-256; retain that digest separately before looking
at the outputs. `analyze_pair` takes that manifest/digest, a serial window, a
grouped window and maps of the retained detached CPU tensors from the actual
measured jobs. It checks their recorded tensor fingerprints, shape, finite
values and frozen tolerance, and returns per-job maximum errors and a
`measured_pair_consistent` flag. It validates supplied event identities,
single-owner call nonoverlap, per-team closed-loop ordering, row-count-specific
profile IDs, cancellation precedence and state-conditioned timing summaries.
It requires at most one internally consistent broker identity in each window;
the two modes may use different broker instances under the same independently
verified resident-model owner. The recorded owner or broker ID does not itself
prove that model state was continuous. It compares every job with delivered
success in both windows, even when another job failed; an incomplete pair
still cannot pass.
It counts failed calls as well as member outcomes and never treats a selected
but unlaunched cohort as a model call. `measured_pair_consistent` is only a
consistency statement about supplied records and tensors; it is not profile
approval, a held-out certificate, proof of execution or a GPU safety claim.

The manifest binds one-batch input and output shapes, float32 output dtype, a
per-output logical cap, common seed and distinct full-input hashes, model and
runtime fingerprints, checkpoint and source-manifest digests, separate serial
and grouped scheduling-policy digests, a no-capture policy, a held-out evidence
digest and distinct exact profile digests by row count. It bounds the planned
total serial-plus-grouped logical
output allowance to 512 MiB. That arithmetic does not cap process RSS, larger
backing storages or aliases, snapshots, finite checks or comparison scratch;
the trusted controller must separately budget those. Inputs, records and tensors
must remain exclusively owned until the analyzer has snapshotted them.

Each window names its distinct window and owner session, protocol/model/runtime
digests and the appropriate scheduling-policy digest, `perf_counter_ns` clock
and resolution, release and drain boundaries,
an independently established owner-drain flag, attempt rows and actual call
rows. Attempt rows retain the frozen job/input/request identities, broker and
broker-job IDs, state/error, attempt/ready/selection/return timestamps, call
ID, output fingerprint, broker terminal state, and cancellation request and
decision times/acceptance. Call rows
retain separate unique group and model-call IDs, ordered broker-scoped member
IDs and row offsets, observed model-entry shape, exact row-count profile,
launch/finish timestamps, state/error and synchronized-completion flag. Job
output fingerprints use `canonical_digest` on the split tensor itself. Missing
client outcomes remain invalid even if the owner later drains; if prior drain
is unconfirmed, a subsequent window is rejected. This analyzer does not
collect or validate warmup records, GPU/host memory telemetry, checkpoint
provenance, saved held-out evidence or clean-supervisor attestation; the
controller must verify and retain those separately before any experiment claim.
