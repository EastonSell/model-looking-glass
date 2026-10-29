# Agreement CUDA adapter and calibration boundary

This is a dedicated supervised-owner prototype, not a shared GPU service or an
authorization to launch GPU work. The ordinary `NaturalAgreementRunner`
remains CPU-only. `CudaAgreementRunner` is a separate backend for the fixed
five-token discovery workload; it rejects held-out mode until the committed
case-binding adapter supports and reviews this backend. A trusted launcher must
bind the exact reviewed public case manifest, verified checkpoint bytes, source
and runtime, and independently audit the resulting journal.

The controller uses the common canonical physical-device identity, owns the
device lease through child exit, verifies the runtime CUDA UUID and holds a
live memory lease. The process is dedicated to one workload and one owner
thread; the adapter's process-local gate does not control other processes.
The exact device-bound `GuardedAgreementMemoryProbe` checks the session,
device lease, bound job lease and, for calibration, resident lease between individual telemetry
queries. The native allocator must have the fixed 25% fraction; both are
rechecked before each headroom sample. The adapter checks live budget inequalities and
at least 3 GiB of device-wide free memory before each forward, state hash and
device RNG operation. These checks include current allocations and held
commitments; they cannot prevent later pressure or guarantee an allocation.

The real study requires two exact measured profiles: a matching cold warmup
profile and the ordered 13-action warm profile. Their IDs and the capture and
warmup policy hashes are recorded. Profile estimates are per-sample incremental
peaks, not whole-stage envelopes. The larger incremental estimate alone does
not cover persistent carryover: the owner must account for it using held claims
and fresh telemetry, including model-load and native/context allocations. The
2 MiB GPU and host trace claims cover specified trace payloads only. Host RAM
for state-hash scratch, Python objects and other allocations needs separate
accounting. A trusted loader must actually load the verified checkpoint; a
checkpoint hash in a profile does not prove the live weights. The adapter
checks model values, per-name identity/storage witnesses and execution
configuration against the warmup state on later checks, including between
calibration actions and study entry. Its CPU denormal probe observes result
underflow on the calling thread, not independently changed denormal-input
mode or floating-point modes in other threads.

`CudaAgreementCalibrator` exists solely to obtain the missing workload
measurements. It uses the same backend and the previously exposed public
five-token cat/cats pair. It refuses study execution. Its separate, provisional
bootstrap requires a 1.5 GiB job commitment and a distinct 2 GiB resident/load
commitment on the same budget, both 2 MiB trace claims, the same
device/allocator checks and live reserve inequalities. Both commitments are
provisional policy choices rather than observed bounds.
One sample per case is also not a worst-case resource guarantee.

Under a reviewed dedicated owner, the collector measures the entire
`warmup()` callback as the cold sample, then calls `calibrate_action(kind)` in
the exact `CALIBRATION_KINDS` order for the 13 warm samples. These callbacks
exercise the actual state checks, inputs, capture, source transfer, patch,
offload, synchronization and disposal path. Their returned records contain
plan coordinates and checks, not logits or activation values. Donor bytes are
held temporarily in process memory and dropped after the final action; they
are not saved by this component. A separate collector must independently bind
the callback, recorded execution metadata, case digest, seed, and raw telemetry
to the resulting profile. Model loading and other startup work need their own
resource evidence. Here **cold** means the verified model is already loaded
and CUDA is initialized under the owner, but that owner has not run a model
forward. The controller must establish this starting state; it is not inferred
from a profile label. The callbacks do not measure storage of the study's
journal tensor payloads, or establish a total host-memory bound.

The warmup returns only its sample ID, policy hash, seed, model-state digest and
plan metadata. The collector uses `synchronize_for_measurement()`, `reset_measurement_peak()`,
`measurement_snapshot()` and `measurement_peaks()` around each whole callback.
It must not use a collector that automatically retries a failed CUDA drain.
Treat every callback exception or uncertain return as terminal: call
`quarantine_after_uncertain_callback()` if possible, keep the device and memory
claims, and terminate the supervised owner without a clean attestation or
retry. An interruption can occur at a callback's return boundary after its own
handler exits. On any failed drain no further CUDA call is permitted. Known
returned device buffers, including plan-owned source and available capture
clones, are rooted until process exit when possible; Python interruption
before assignment, allocation failure during custody and model-internal
temporaries limit what can be retained.

`quarantined` is a quiescent-boundary health check: it is also true while a
healthy synchronization is in progress. The supervisor must impose a bounded
completion deadline and treat a hung or unknown drain as unsafe. Failure during
construction after the owner gate is installed also requires process
replacement. Following a successful run, `dispose_after_success()` only drops
the adapter's known roots; the owner must still dispose other references,
synchronize, verify cleanup and claims, and obtain supervised clean
attestation before the marker can be cleared.
