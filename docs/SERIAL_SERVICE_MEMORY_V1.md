# Optional serial service memory observation

`prepare_two_token_service(..., observe_serving_memory=True)` enables a
bounded diagnostic for the fixed, supervised, zero-trace 1 × 2 service.
The default is disabled. The prepared handle exposes the private
`serving_memory_path`; it is derived from the private launch directory and
is not a public network endpoint. The caller must retain the prepared handle
and reconcile its normal supervisor exit and coordination marker. A memory
file alone does not establish clean teardown.

This is one window for the entire serving lifetime, not a control interface
for multiple benchmark windows. After model construction and the cold
warmup, the owner checks its device lease, synchronizes, resets PyTorch peak
statistics and records a baseline before accepting clients. Its reported
device total must match the resolved profile before further allocator reads.
While serving,
existing worker admission checks provide bounded point samples. After the
service handlers and broker have joined, the remaining signalers have
stopped, and the final guarded drain succeeds, the owner synchronizes and
records an endpoint and the process-local PyTorch allocated/reserved peaks.
Each added CUDA operation is guarded by the owner lease and the exact retained
resident/workspace claims before and after the call. Sampling
adds work to the execution path and may affect timing.

The private JSON contains the source/asset digests, approved profile pins,
owner PID and device identity used by the child. It stores at most 128
intermediate samples and counts dropped samples; baseline and endpoint are
separate. Timestamps use the process's monotonic clock. A sample's device
free/total and process allocator counters are read sequentially and are not
an atomic snapshot. Device free is a device-wide point observation, and may
miss transient external pressure. PyTorch peaks cover that process's
allocator since the reset, not CUDA context, other processes, all native
allocations, or a device-wide peak. Neither points nor peaks are a future
memory bound or an operating-system memory reservation.

The owner writes the file before clean attestation; a failed write, device
query or guard fails the owner closed. A partial file or a file without a
clean independently reconciled supervisor result is not a completed
measurement. A failure may leave no memory file at all. The existing request
outcomes, cancellation/OOM handling, retained resource claims, and strict
exit rules continue to apply.

Enabling the observer changes the package source fingerprint and adds CUDA
calls. The previous `dd819b87` worker profiles and host observation cannot
authorize this changed source. Before any GPU use, freeze the exact source and
launch path, obtain matching worker profiles and a reviewed workload/host
admission, and coordinate a fresh device slot. This optional diagnostic does
not enable ordinary service admission, batching, sustained-fairness claims,
or activation traces.
