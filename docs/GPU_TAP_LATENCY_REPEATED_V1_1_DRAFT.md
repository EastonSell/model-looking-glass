# Repeated tap benchmark v1.1 quiet-window clarification, prospective draft

This prospective clarification changes the coordination rule in the proposed
[v1 protocol](GPU_TAP_LATENCY_REPEATED_V1.md). It does not alter the earlier
contaminated pilot or describe an observed benchmark. No v1.1 GPU run is
authorized by this document. Before execution, the owner, controller and replay
must bind this clarification and the exact frozen implementation under a new
`gpu-tap-latency-repeated-v1.1` experiment identity.

The five sessions, six-permutation schedule, fixed workload, admission,
measurements, screening, failure rules and analysis otherwise remain as in v1.
Use the same saved schedule bytes and preserve their original schedule hash;
do not regenerate them or pool any prior pilot observations.

For each short timing window, coordinate exclusion of other cooperating GPU
work and ask project owners whose CPU-intensive or model work could materially
affect host timing to pause that work. Unrelated work that is not expected to
contend may continue. Announce the actual start and end, and retain a record of
known participating pauses, known concurrent work and the limits of that
record. Do not terminate or close user applications. The coordinator cannot
verify system-wide CPU quiescence or guarantee that work classified as
noncontending had no effect; host scheduling remains an uncontrolled source of
timing variation. The device activity screen remains required for the entire
workload, with sampled observations and explicit query gaps preserved. Define
eligible pre-workload idle samples as queries starting within two seconds
before preflight and ending before it. Define eligible post-workload idle
samples as queries starting at least one second after the workload and ending
within five seconds of its end. Require at least three in each interval; the
last three establish the selected coverage endpoints, but any eligible sample
over 5% utilization marks contamination. Preserve every valid sampled timestamp,
utilization and memory value in the PID/UUID-free public projection. A failed
or malformed query makes availability unverified; the current collector keeps
only a bounded sanitized error code/count, not the invalid raw bytes, stderr,
or necessarily the failed query's timestamps. A post-workload
utilization window may overlap the benchmark itself. The screen does not
establish continuous GPU inactivity between samples or physical exclusivity.
For NVIDIA command-line memory counters reported in whole MiB, treat
`used + free > total + 2 MiB` as unavailable telemetry. The two-MiB allowance
is a conservative rounding tolerance; it does not require the counters to sum
to total, since other reserved memory may exist.

Bind and retain the exact CUDA runtime total separately from the stable
NVIDIA-SMI total reported in whole MiB for the same supervised physical UUID.
The CUDA total controls the allocator cap and admission. Permit an absolute
cross-provider difference of at most 2 MiB as a prospective conservative
reconciliation allowance for provider accounting and rounding, not as proof
of device identity or physical reservation. Each provider's total must stay
constant throughout its own observations; a larger difference, drift, or an
unverified UUID makes the screen unavailable. For example, the previously
observed 8,589,606,912-byte CUDA total and 8,589,934,592-byte NVIDIA-SMI total
differ by 327,680 bytes and satisfy this bound.

The parent must obtain at least three GPU points whose query starts are within
two seconds of the actual first preflight start, as well as a before-process
query started within that interval. The child must recheck their original
timestamps and the one-use grant's expiry immediately before the first
preflight. A shared, reviewed cross-process clock binding is required; similar
clock field names alone are insufficient. The journal's preflight start and
the activity screen's workload start must be the same source observation; the
screen's workload end must equal the source-stamped completion of the last
sample's postcheck. That completion must follow the final allocator/availability
query and validation, fall within two seconds of the measured wall end, and
precede a no-more-device-work wait for the parent's finish grant. Cleanup occurs
after the screen. This keeps the final device-facing postcheck inside the
screen while preventing a freely shifted idle window; the two-second bound is
prospective and not a physical timing guarantee. Shifting either bound could
hide stale or contaminated points. NVIDIA-SMI total, used, and free values must retain the whole-MiB
representation of the query provider.

This postcheck endpoint prospectively changes v1's measured-sample endpoint:
the final eligible idle query may end up to seven seconds after the measured
wall end (two seconds of postcheck plus five seconds after its completion).
The timed wall interval itself is unchanged.
Each preflight/sample postcheck also records its own source-stamped completion;
it must be no earlier than that observation's end, within two seconds of it,
and no later than the next observation's start. A late postcheck is a failed
attempt, not an exclusion of a recorded timing. The replay cannot
independently authenticate the stamp.

For this prospective instrumented implementation, owner/lease guard counts
and timing boundaries must match across modes. Direct and gradient modes also
perform their necessary trace-specific selection and container validation;
the none mode checks absence and substitutes checked no-ops. Those extra
mode-specific checks and container costs are part of the measured path, so
differences do not isolate pure hook or kernel cost. This clarifies v1's
identical-additional-check wording and must be disclosed without subtracting a
post hoc overhead estimate. The many owner and lease checks inside wall time
can dominate or dilute a two-token difference despite matched counts; their
latency and host scheduling variance remain. CUDA-event spans are secondary
and cover a different interval, so neither measure is a hook-only cost.

The parent must retain the child's process handle and verify it is alive
immediately before and after each before/after process query. The child waits
for the finish grant through the latter query. If that identity is not held
through either query, the record is incomplete and the numeric PID cannot be
excluded from the foreign-process screen; PID reuse could hide another
process. Parent and child identities and the activity UUID must agree before
the public screen is projected. These checks are trusted-record consistency
checks, not authentication by the offline replay.

The CPU replay distinguishes the v1.1 experiment/run identity from the
unchanged v1 schedule file and requires v1.1 postcheck ordering. It is not an
executable v1.1 GPU study: the eventual owner and controller still need a
separately reviewed binding, preregistered eligibility rules and complete raw
evidence before any measured forward.
