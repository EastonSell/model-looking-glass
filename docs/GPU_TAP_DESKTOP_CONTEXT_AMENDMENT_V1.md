# GPU tap timing: prospective desktop-context amendment

This amendment defines two separate, single-session observations on the
Windows desktop. It does not revise or replace the completed contaminated
v1.3 session or the earlier incomplete attempts. Publish this amendment and
review each exact observer before either new arm. The recorded timing source
and internal protocol identity remain visible: use the frozen v1.3 executable
`0c14778447ed6dcbc4890984641370659a09ffef`, not a relabeled source.

## Workload and safety

For each arm retain the v1.3 fixed checkpoint, two-token input, site, model
seed 17, preflight, six warmups and 12 balanced three-mode rounds. Both arms
use the executable's only enabled schedule slot 0, whose frozen schedule seed
is 53101. Forward calls are serial, each with completion synchronization and
output/capture validation. The primary wall interval runs from immediately
before start-event recording through completion synchronization, before the
later postcheck and disposal. The CUDA-event span covers the recorded default
stream and can include host-induced gaps. The guarded interval also includes
pre-sample synchronization and memory preparation, owner postcheck, disposal,
second synchronization and final memory postcheck. Keep all three distinct.
Direct captures an activation; the auxiliary-gradient mode differentiates a
local dot product with respect to a fresh zero probe. It is not a model-loss
backward pass. Per-mode differences include each complete instrumented path;
none of these intervals is a pure hook or kernel time.

Keep the existing source, checkpoint, supervisor, owner, exclusive
cooperative device lease, memory admission, activity-screen availability,
failure evidence, synchronization, disposal and reaped-exit requirements.
No query or run is permitted while an unsafe marker or coordinator hold is
unresolved. A fresh coordinator-assigned slot is required for each arm. Do
not terminate desktop applications, automatically retry failed or incomplete
attempts, or select a run based on its measurements or contamination status.

## Two scheduling contexts

* **Desktop-observed:** run at the next safe assigned opportunity with normal
  desktop and unrelated CPU work allowed to continue. The owner still holds
  the exclusive cooperative GPU lease for its lifetime. Do not defer solely
  because WDDM lists foreign processes or sampled utilization exceeds 5%.
* **Cooperative agent-GPU window:** separately reserve an interval in which
  the coordinator prevents other cooperating agent GPU jobs from overlapping
  the screened workload and its before/after activity observations. The
  owner holds the same lease. Unrelated CPU work and uncontrollable desktop
  processes may continue. This is a coordinated project window, not proof
  that the physical device, Windows desktop or external workloads are idle.

The same device lease excludes other cooperating owners during the measured
owner in both arms; the second arm adds an explicitly recorded coordinated
window around its activity observations. Record actual arm order and timings.
These are separate descriptive observations, not a randomized isolation
experiment, and differing sessions cannot establish the effect of isolation.
Use new, distinct result/observer paths and record which arm actually ran.
The prepared second observer is for one attempt only; any additional
observer must receive independent source/path review and a no-model rehearsal.
Because both independent arms use slot 0, their internal v1.3 trace run IDs
repeat; preserve their distinct supervisor/process identities and never treat
the two records as one multi-session result.

## Context and analysis

Retain the original v1.3 GPU activity and process queries, query intervals,
and complete-coverage rules. Preserve both screen `status` and `availability`
and their reason lists. Foreign listed GPU PIDs or idle samples above 5%
mean status `contaminated`. Missing or stale coverage makes availability
`unverified`; status remains `contaminated` if an independently observed
contamination reason exists, and otherwise is `unverified`. Unverified
availability leaves the formal session incomplete even if raw samples exist.
The frozen initial gate stops before the first model forward when its required
GPU or process telemetry is unavailable, invalid or stale. An after-workload
utilization sample may include the benchmark's own preceding work.
The new scheduling policy does not turn either result into a clean or
physically isolated benchmark. WDDM process presence alone is not evidence that
a process performed GPU work.

Preserve Windows `GetSystemTimes` counters bracketing the controller, and
record a read-only system-process inventory at the start and end, with
timestamps and known overlapping project-work intervals. For processes that
are accessible at both endpoints, retain PID plus creation time and cumulative
CPU times where available; explicitly mark inaccessible or missing data.
These records include the measurement/observer and may cover only the caller's
processor group on systems with more than 64 logical processors. Neither a
process listing nor a whole-controller CPU bracket identifies per-sample
interference. Retain query-start/end timestamps so any telemetry overlap with
timed windows can be reported. An optional periodic CPU-only sampler must be
separately reviewed and rehearsed without CUDA; it must not synchronously add
calls to the measured owner. The prepared observer records `cpu_context` as
`unknown`; additional context does not silently change that field. Keep
process and device identities private.

Save every completed and partial attempt, raw ordered samples, warmups,
postchecks, activity, context and independent owner/exit evidence. For each
complete arm, report each round's raw mode times, paired gradient/direct and
direct/no-tap ratios and differences, their medians and ranges, mode wall and
event medians, screen status and availability, and timing-query overlap. No sample is
discarded or reweighted because of observed load. A single session is
descriptive: do not claim a general slowdown, statistical confidence interval
or causal isolation effect. Preserve the existing v1.3 contaminated result
and its original label alongside any later result.
