# Supervised direct two-client composition

`SupervisedConcurrentRun` composes the existing prepared fixed service and its
one-shot `ConcurrentClients` plan. Construct and retain the prepared service,
client plan, and run object before calling `run()`. The composition starts the
existing supervised owner, starts the two clients, waits for their bounded
metadata, and then requests shutdown through the same prepared controller. It
publishes a result only after a clean, reaped, zero-exit supervisor result, an
absent unsafe marker, and a repeated source/asset preflight. The result includes
the source-base and asset-manifest SHA-256 values and the three approved worker
profile IDs. The client report keeps its own cancellation, failure, queue and
no-trace semantics.

This entry defaults **closed**: no qualifying host observation is included,
and a run without independently approved, matching evidence is rejected before
the owner starts. The fixed worker's GPU profiles do not measure the two-client
IPC codec, Python objects, serializer, or aggregate host memory. The per-frame
caps are not a host-memory guarantee. A bounded development-only observation
has its own explicit entry described below; it does not silently open the
ordinary evidence-gated entry.
This gate protects this entry only. The direct client and prepared service APIs
still require their own admission; this module is not a global host guard.

The optional `ApprovedHostEvidence` takes the exact private canonical JSON
bytes and a separately supplied SHA-256 approval pin. The operator must select
the pin only after independent review of an actual clean, reaped GPU-owner
observation of this exact two-client path. Hash equality selects bytes and
does not authenticate their origin or establish that the recorded events took
place. Synthetic CPU observations and the grouped GPU pilot are ineligible.
The record binds the source and offline assets, runtime, three worker profile
IDs, base Python executable, 1×2 fixed inputs, two team/seed slots, no requested
cancellation, two connections, queue and frame limits, and zero trace. It
records three model calls, both successful clients, supervisor and marker
outcomes, and process IDs, birth times and executable hashes checked against
retained process handles. These are trusted, independently reviewable claims,
not facts inferred from the digest. Two configured client slots and two
successful results do not by themselves establish that request admissions or
response buffers overlapped. A simultaneous-response bound needs separately
instrumented evidence.

For one initial observation, `run_development_bootstrap()` is a distinct
explicit opt-in on the same one-shot run object. It uses the same fixed
no-requested-cancellation 1×2 client plan, supervisor and worker, and still
requires all three independently approved profiles for the exact final source
and live owner path. It does not accept an approved host record or approve the
observation it produces. Before startup it charges a fixed 8 GiB logical host
allowance and requires a separate 2 GiB reserve, so all three fresh physical,
system-commit and calling-process-commit values must be at least 10 GiB. The
claim and original run graph stay rooted until controller process exit even
if startup, collection or shutdown is uncertain. The returned result and the
retained claim identify `development-bootstrap`; a returned result must still
be checked for actual client outcomes and clean supervision. This allowance is
a provisional policy choice, not an OS reservation or an RSS cap.

The bootstrap itself does not collect or approve a host profile. Use a
separately reviewed bounded observer on the exact final source to record
process identity and lifetime peaks, stage samples and outcome, then reconcile
the owner, controller, supervisor and durable marker. A failed, cancelled or
incomplete observation is not an approval. Device quarantine recovery,
source-specific worker profiles, current under-lease GPU admission and a
coordinated device window remain independent prerequisites.

The private record contains ordered samples from before start through owner
reap, with gaps no larger than two seconds, plus each controller and owner
process's lifetime peak working set and peak private commit. It excludes a
separate observer and helper processes. Sampled caller-available commit must
be collected by the controller, as identified by the record; a separate
observer cannot substitute its own caller-specific value. The logical
commitment covers the larger of sampled aggregate working set, sampled
aggregate private commit, or
the sum of each process's lifetime peaks, plus at least 1 GiB; it is capped at
16 GiB. A separate reserve of at least 2 GiB is required. The sums can double
count shared pages or combine peaks from different times, while sampling can
miss short aggregate transients. Neither the margin nor the sampled maximum is
a measured worst-case bound. `PeakPagefileUsage` is the Windows lifetime peak
commit charge; it is distinct from peak working set and the current
`PrivateUsage` value.

Immediately before service startup, the controller checks a fresh physical
memory point and both system-wide and caller-available commit against the
larger of 8 GiB or commitment plus reserve, and checks physical total against
the approved observation. System commit availability uses
`(CommitLimit - CommitTotal) * PageSize` from `GetPerformanceInfo`;
`GlobalMemoryStatusEx.ullAvailPageFile` applies to the calling process and is
not a child guarantee. The reads are sequential, may change immediately, and
occur before the owner takes its device lease. Charging the whole observed
process footprint against currently available memory deliberately double
counts memory already in use. These screens are policy heuristics, not an OS
reservation, hard RSS ceiling or under-lease host measurement. The
process-local claim and original run/service/client graphs stay rooted until
controller process exit, including after errors or ambiguous handoff; a second
claim in the same process is refused. It does not account for independent
processes. The checked plan is a cooperative integrity boundary: external
cancellation or other mutation is outside the measured success workload and
can make the final binding check fail, while existing supervised recovery and
the retained claim remain required. The measurement does not cover the host
peaks of cancellation, OOM, or other failure paths; it does not authorize a
general cancellation-enabled service.

The existing fixed owner independently reconstructs its physical identity and
the custody-bound model-load, cold and warm profiles under its device lease.
The grouped 1×4/2×4 pilot and unrelated one-shot GPU results do not authorize
this serial 1×2 path. After the owner, concurrent-host admission and all
compatibility markers are finalized, freeze the final source and runtime,
collect and independently approve matching custody profiles and host evidence,
and perform the device owner's fresh under-lease admission in a coordinated
slot. The host screen above is a separate pre-start point. Source changes after
measurement require new matching profiles and host evidence. This composition
enables no activation recording; a trace-enabled successor requires its own
versioned path, budgets and measurement.

The prepared service and clients must have exclusive cooperative ownership
while `run()` is active. A client timeout, startup
interruption, failed shutdown, lost response, rejected request, OOM, or missing
supervisor result can leave work uncertain. The run object retains exceptions
from its run, shutdown, supervisor-wait and source-check boundaries, but the
retained process-local claim roots all three objects, but the operator must
still reconcile the original controller, client threads and unsafe marker.
`wait_supervisor()` rechecks the pinned
controller even if the prepared binding changed. Do not replay an uncertain request. The
existing supervisor's process-level watchdog remains necessary for stalled or
unobservable thread/native failures.

The checked source inventory and profile IDs are trusted reproducibility
bindings, not a sandbox or proof of every loaded Python/native byte. A clean
run also checks the canonical prepared-service and client methods before and
after startup; this is a point-in-time cooperative integrity check, not an
atomic defense against arbitrary in-process mutation. A clean
supervisor result and an absent marker are scoped to the supervisor's documented
protocol and the instant of observation. A client report with no detected fault
is never itself evidence of teardown. The returned metadata contains no raw
logits, credentials, endpoint, local paths or physical device identifier, and
does not establish sustained-backlog fairness or simultaneous GPU execution.
